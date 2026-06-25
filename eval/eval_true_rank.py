import os
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()


async def diagnostic_scan():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
    )
    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    query = "How do I safely initialize a background vector upsert inside a FastAPI lifespan without blocking the event loop?"
    query_vector = embed_model.encode(query).tolist()

    target_repos = ["fastapi", "qdrant-client"]

    print("Executing Deep Retrieval Diagnostic Scan (Limit=150 per repo)...")

    for repo in target_repos:
        print(f" REPOSITORY: {repo}")

        response = await asyncio.to_thread(
            lambda: client.query_points(
                collection_name="cross_repo_rag",
                query=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo_name", match=models.MatchValue(value=repo)
                        )
                    ]
                ),
                limit=150,
                with_payload=True,
            )
        )
        hits = response.points if hasattr(response, "points") else response

        found_any = False
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload
            file_path = payload.get("file_path", "")
            score = hit.score

            # Highlight target files or core production structures
            is_target = (
                "applications.py" in file_path or "qdrant_client.py" in file_path
            )
            is_src = "tests/" not in file_path and "docs/" not in file_path

            if is_target:
                print(f"[RANK {rank}] SCORE: {score:.4f} -> TARGET FOUND: {file_path}")
                found_any = True
            elif is_src and rank <= 20:
                # Print non-test files in top 20 to see what production code is visible
                print(
                    f"   [RANK {rank}] SCORE: {score:.4f} -> Source File: {file_path}"
                )
                found_any = True

        if not found_any:
            print("Target production files did not appear within the Top 150 hits.")


if __name__ == "__main__":
    asyncio.run(diagnostic_scan())
