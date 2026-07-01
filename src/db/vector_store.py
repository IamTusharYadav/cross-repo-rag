import os
import logging
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

load_dotenv()

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Owns the full lifecycle of the Qdrant collection and the embedding model.

    Public surface used by RepositoryIngestionPipeline:
        init_collection()                        -- idempotent collection + index setup
        embed_chunks(texts)                      -- encode text -> dense + sparse vectors
        upsert_chunks(texts, embeddings, metas)  -- build PointStructs and write to collection
        upsert_points(points)                    -- write pre-built PointStructs (testing)
        delete_file_chunks(repo, path)           -- wipe every vector belonging to a file
        prune_file_chunks(repo, path, active_ids)
                                                 -- remove stale vectors after a re-index

    Callers never touch self.client directly, and never construct Qdrant model objects;
    all Qdrant internals are contained within this class.
    """

    # Persistent local fallback path used when QDRANT_URL is not set.
    # Keeps data alive between runs
    LOCAL_QDRANT_PATH = "./data/qdrant"

    def __init__(self, collection_name: str = "cross_repo_rag"):
        self.collection_name = collection_name

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            logger.info(f"Connecting to Qdrant instance at: {qdrant_url}")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            logger.info(
                f"QDRANT_URL not set. Using local persistent Qdrant at: {self.LOCAL_QDRANT_PATH}"
            )
            self.client = QdrantClient(path=self.LOCAL_QDRANT_PATH)

        # Dense Encoder Setup
        self.dense_embedding_model_name = "BAAI/bge-base-en-v1.5"
        logger.info(
            f"Loading dense embedding model: {self.dense_embedding_model_name}..."
        )
        self.encoder = SentenceTransformer(self.dense_embedding_model_name)
        self.vector_size = self.encoder.get_embedding_dimension()

        # Sparse Encoder Setup
        self.sparse_embedding_model_name = "Qdrant/bm25"
        logger.info(
            f"Loading sparse embedding model: {self.sparse_embedding_model_name}..."
        )
        self.sparse_encoder = SparseTextEmbedding(
            model_name=self.sparse_embedding_model_name
        )

    # Collection management

    def init_collection(self):
        """
        Creates the Qdrant collection if it does not already exist, then ensures
        payload indexes exist on the fields used in every filter query.
        """
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating collection '{self.collection_name}'")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                    # Inject server-side lexical index engine
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            modifier=models.Modifier.IDF,  # Dynamically handles BM25 IDF math
                            index=models.SparseIndexParams(on_disk=True),
                        )
                    },
                )
            else:
                logger.info(f"Collection '{self.collection_name}' already exists.")

            # Re-apply index creation on every startup so collections created
            # outside this manager also get indexes, and so new indexes added
            # here are applied without manual migration steps.
            self._ensure_payload_indexes()
        except Exception as e:
            logger.error(f"Failed to initialise Qdrant collection: {e}")
            raise

    def _ensure_payload_indexes(self):
        """
        Creates keyword indexes on repo_name and file_path if not already present.
        """
        for field in ("repo_name", "file_path"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info(f"Payload index ensured for field: '{field}'")
            except Exception as e:
                logger.debug(f"Payload index for '{field}' may already exist: {e}")

    # Embedding

    def embed_chunks(self, texts: list[str]) -> Tuple[List[List[float]], List[Any]]:
        """
        Encodes a list of text strings into L2-normalised dense vectors and sparse embeddings.
        """
        if not texts:
            return [], []

        # Compute Dense Vector via SentenceTransformer
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        dense_embeddings = embeddings.tolist()

        # Compute Sparse Matrix via FastEmbed
        sparse_embeddings = list(self.sparse_encoder.embed(texts))

        return dense_embeddings, sparse_embeddings

    # Write operations

    def upsert_chunks(
        self,
        texts: list[str],
        embeddings: Tuple[List[List[float]], List[Any]],
        metas: List[Dict[str, Any]],
    ) -> None:
        """
        Constructs PointStructs from pre-computed hybrid embeddings and metadata dicts,
        then writes them to the collection in a single upsert call.
        """
        if not texts:
            return

        dense_embeddings, sparse_embeddings = embeddings

        if not (
            len(texts) == len(dense_embeddings) == len(sparse_embeddings) == len(metas)
        ):
            raise ValueError(
                "texts, dense embeddings, sparse embeddings, and metadata must have equal lengths."
            )

        points = []
        for i in range(len(texts)):
            payload = {"text_content": texts[i], **metas[i]}

            points.append(
                models.PointStruct(
                    id=metas[i]["point_id"],
                    vector={
                        "dense": dense_embeddings[i],
                        "sparse": models.SparseVector(
                            indices=sparse_embeddings[i].indices.tolist(),
                            values=sparse_embeddings[i].values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def upsert_points(self, points: list[models.PointStruct]) -> None:
        """
        Writes a list of pre-built PointStructs into the collection.
        """
        if not points:
            return
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_file_chunks(self, repo_name: str, file_path: str) -> None:
        """
        Deletes every vector associated with a specific file in a repository.
        Used when a file is deleted from the filesystem or yields zero chunks.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="repo_name", match=models.MatchValue(value=repo_name)
                    ),
                    models.FieldCondition(
                        key="file_path", match=models.MatchValue(value=file_path)
                    ),
                ]
            ),
        )

    def prune_file_chunks(
        self, repo_name: str, file_path: str, active_ids: list[str]
    ) -> None:
        """
        Removes stale vectors for a file after a re-index.

        Deletes every vector whose point ID is NOT in active_ids, leaving only
        the chunks that belong to the current structural state of the file.
        Called exactly once per file, after all of its chunks have been upserted.
        """
        if not active_ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="repo_name", match=models.MatchValue(value=repo_name)
                    ),
                    models.FieldCondition(
                        key="file_path", match=models.MatchValue(value=file_path)
                    ),
                ],
                must_not=[models.HasIdCondition(has_id=active_ids)],
            ),
        )
