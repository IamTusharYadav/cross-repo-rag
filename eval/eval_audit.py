import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

print("--- Samples of actual paths indexed in Qdrant ---")
scroll_results = client.scroll(
    collection_name="cross_repo_rag", limit=10, with_payload=["file_path", "repo_name"]
)

for point in scroll_results[0]:
    payload = point.payload
    print(
        f"Repo: {payload.get('repo_name')} -> Indexed Path: {payload.get('file_path')}"
    )
