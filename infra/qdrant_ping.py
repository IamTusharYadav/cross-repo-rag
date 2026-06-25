import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# Initialize the Qdrant Cloud client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Retrieve and print the current collections list
collections_response = client.get_collections()
print(f"Successfully connected! Collections: {collections_response.collections}")
