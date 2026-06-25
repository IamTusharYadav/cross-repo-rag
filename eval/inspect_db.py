import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


def inspect_payloads():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
    )

    print("Fetching 3 sample points from Qdrant...\n")
    points, _ = client.scroll(
        collection_name="cross_repo_rag",
        limit=3,
        with_payload=True,
    )

    for idx, p in enumerate(points):
        print(f"--- Point {idx + 1} ---")
        print(json.dumps(p.payload, indent=2))
        print("\n")


if __name__ == "__main__":
    inspect_payloads()
