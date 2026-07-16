import os
import chromadb
from chromadb.config import Settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_PATH,
        is_persistent=True,
        anonymized_telemetry=False
    )
)

COLLECTION_NAME = "legal_clauses"

collection = client.get_or_create_collection(COLLECTION_NAME)

print("Total vectors:", collection.count())
