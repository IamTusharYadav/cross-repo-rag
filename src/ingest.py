import os
import uuid
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.db.state_manager import IngestionStateManager
from src.db.vector_store import VectorStoreManager
from src.chunkers.chunk_markdown import MarkdownHierarchyChunker
from src.chunkers.chunk_python import PythonASTChunker
from src.utils.redactor import redact_python_text, redact_markdown_text

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".py", ".md"}


class RepositoryIngestionPipeline:
    """
    Orchestrates incremental ingestion of source repositories into a Qdrant vector store.

    Lifecycle per run:
      1. Walk each repo under DATA_DIR.
      2. Skip files unchanged since last run (needs_processing check).
      3. For each changed file: chunk -> redact -> embed -> upsert -> prune stale chunks -> mark processed.
      4. Reconcile deleted files against the state ledger.
    """

    DATA_DIR = "data"
    BATCH_SIZE = 128

    def __init__(self):
        self.repo_root = os.path.abspath(self.DATA_DIR)
        self.state_manager = IngestionStateManager("sqlite:///data/ingestion_state.db")
        self.vdb_manager = VectorStoreManager()
        self.md_chunker = MarkdownHierarchyChunker()
        self.py_chunker = PythonASTChunker()

    # Public entry point

    def run(self):
        self.vdb_manager.init_collection()

        if not os.path.exists(self.DATA_DIR):
            logger.error(f"Target data workspace root missing: {self.DATA_DIR}")
            return

        repos = [
            d
            for d in os.listdir(self.DATA_DIR)
            if os.path.isdir(os.path.join(self.DATA_DIR, d)) and not d.startswith(".")
        ]

        processed_files_tracker = set()
        total_skipped_files = 0

        for repo in repos:
            repo_path = os.path.join(self.DATA_DIR, repo)

            for root, _, files in os.walk(repo_path):
                for file in files:
                    # Skip unsupported file types early; avoids confusing "0 chunks" log noise
                    if Path(file).suffix not in SUPPORTED_EXTENSIONS:
                        continue

                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.repo_root).replace(
                        "\\", "/"
                    )

                    if not self.state_manager.needs_processing(
                        file_path, self.repo_root, repo
                    ):
                        total_skipped_files += 1
                        continue

                    logger.info(f"Processing file: {rel_path}")

                    if self._process_file(file_path, rel_path, repo):
                        processed_files_tracker.add(rel_path)

        logger.info(
            f"Ingestion run complete. "
            f"Processed: {len(processed_files_tracker)} | Cache Hits (Skipped): {total_skipped_files}"
        )

        self._reconcile_deleted_files()

    # File-level processing

    def _process_file(self, file_path: str, rel_path: str, repo: str) -> bool:
        """
        Processes a single file end-to-end:
          chunk -> redact -> embed -> upsert -> prune stale chunks -> mark processed.
        Returns True if the file was fully committed; False on any failure so the
        next run retries it.
        """
        suffix = Path(file_path).suffix

        raw_chunks = []
        try:
            if suffix == ".md":
                raw_chunks = self.md_chunker.chunk_file(file_path, state_manager=None)
            elif suffix == ".py":
                raw_chunks = self.py_chunker.chunk_file(file_path, state_manager=None)
        except Exception as parse_error:
            logger.error(
                f"Critical structural parsing failure on: {rel_path}. Error: {parse_error}"
            )
            return False

        if not raw_chunks:
            logger.warning(
                f"File {rel_path} yielded 0 chunks. Clearing indices and updating state ledger."
            )
            self.vdb_manager.delete_file_chunks(repo, rel_path)
            self.state_manager.mark_as_processed(file_path, self.repo_root, repo)
            return True

        logger.info(f"Generated {len(raw_chunks)} chunks from {rel_path}")

        # Build the full text/metadata lists for the entire file before any upsert.
        # Drift pruning must only run after every chunk is in the index not after
        # an arbitrary batch boundary to avoid deleting not-yet-uploaded chunks.
        all_texts, all_metas = self._prepare_chunks(raw_chunks, rel_path, repo, suffix)

        # Embed and upsert in batches, collecting all point IDs for this file
        all_active_ids = []
        for batch_start in range(0, len(all_texts), self.BATCH_SIZE):
            text_batch = all_texts[batch_start : batch_start + self.BATCH_SIZE]
            meta_batch = all_metas[batch_start : batch_start + self.BATCH_SIZE]
            logger.info(
                f"Flushing batch of {len(text_batch)} vectors for {rel_path}..."
            )
            batch_ids = self._embed_and_upsert(text_batch, meta_batch)
            all_active_ids.extend(batch_ids)

        # Prune stale chunks once, after all of this file's chunks are safely in the index.
        # If pruning fails the file is NOT marked processed so the next run retries cleanly.
        try:
            self.vdb_manager.prune_file_chunks(repo, rel_path, all_active_ids)
        except Exception as prune_err:
            logger.error(
                f"Post-upsert drift pruning failed for {rel_path}: {prune_err}"
            )
            return False

        self.state_manager.mark_as_processed(file_path, self.repo_root, repo)
        return True

    def _prepare_chunks(
        self,
        raw_chunks: list,
        rel_path: str,
        repo: str,
        suffix: str,
    ) -> tuple[list, list]:
        """Redacts raw chunk text and assembles payload metadata for every chunk in a file."""
        all_texts = []
        all_metas = []

        for chunk in raw_chunks:
            meta = chunk.get("metadata", {}).copy()

            raw_text = chunk["text"]
            cleaned_text = (
                redact_python_text(raw_text)
                if suffix == ".py"
                else redact_markdown_text(raw_text)
            )

            meta.update(
                {
                    "repo_name": repo,
                    "file_path": rel_path,
                    "file_type": suffix,
                    "redacted": (cleaned_text != raw_text),
                    # Retain the chunker's original token count (pre-redaction)
                    "source_token_count": meta.get("token_count"),
                }
            )

            all_texts.append(cleaned_text)
            all_metas.append(meta)

        return all_texts, all_metas

    # Vector store operations

    def _embed_and_upsert(self, text_batch: list, meta_batch: list) -> list[str]:
        """
        Resolves semantic chunk identities, stamps each metadata dict with
        semantic_chunk_id and point_id, then delegates embedding and upsert
        entirely to VectorStoreManager.

        Returns the list of point IDs written, for use in post-file drift pruning.
        No qdrant_client types are imported or constructed here.
        """
        if not text_batch:
            return []

        embeddings = self.vdb_manager.embed_chunks(text_batch)
        upserted_ids = []

        for i, text in enumerate(text_batch):
            meta = meta_batch[i]

            repo_name = meta.get("repo_name", "")
            symbol_path = meta.get("symbol_path")
            chunk_index = meta.get("chunk_index")

            if symbol_path is not None and chunk_index is not None:
                semantic_chunk_id = f"{repo_name}::{symbol_path}::chunk{chunk_index}"
            else:
                semantic_chunk_id = meta.get("chunk_id") or (
                    f"{meta['file_path']}_"
                    f"{meta.get('chunk_start_line', i)}_"
                    f"{meta.get('chunk_end_line', i)}_"
                    f"{text}"
                )

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, semantic_chunk_id))

            # Preserve the chunker's original chunk_id (SHA-256) for debugging and
            # retrieval evaluation.  semantic_chunk_id is the ingestion-layer identity
            # used to derive point_id
            meta["semantic_chunk_id"] = semantic_chunk_id
            meta["point_id"] = point_id
            # meta["page_content"] = text

            upserted_ids.append(point_id)

        self.vdb_manager.upsert_chunks(text_batch, embeddings, meta_batch)
        return upserted_ids

    def _reconcile_deleted_files(self):
        """
        Compares the SQLite ledger against the physical filesystem.
        Removes orphaned vectors from Qdrant and drops ledger records for deleted files.
        """
        logger.info("Starting reconciliation phase for deleted files...")
        tracked_files = self.state_manager.get_all_tracked_files()

        deleted_count = 0
        for record in tracked_files:
            if not os.path.exists(record["absolute_path"]):
                logger.info(
                    f"Detected deleted file: {record['file_path']}. Removing orphaned vectors..."
                )
                self.vdb_manager.delete_file_chunks(
                    record["repo_name"], record["file_path"]
                )
                self.state_manager.remove_file_record(
                    record["repo_name"], record["file_path"]
                )
                deleted_count += 1

        if deleted_count > 0:
            logger.info(
                f"Reconciliation complete. Purged {deleted_count} deleted files from index."
            )
        else:
            logger.info("Reconciliation complete. No orphaned files detected.")


if __name__ == "__main__":
    RepositoryIngestionPipeline().run()
