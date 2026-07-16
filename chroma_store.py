import os
import uuid
import chromadb
from chromadb.config import Settings

from embeddings.embedder import embed_text


class ChromaClauseStore:
    COLLECTION_NAME = "legal_clauses"

    def __init__(self, persist_dir: str):
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_dir,
                is_persistent=True,
                anonymized_telemetry=False
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME
        )

    def add_clauses(self, clauses: list[dict]):
        for clause in clauses:
            embedding_input = (
                f"{clause['contract_type']} | "
                f"{clause['clause_title']} | "
                f"{clause['clause_text']}"
            )

            vector = embed_text(embedding_input)

            clause_id = f"{clause['contract_type']}_{clause['clause_title']}"
            clause_id = clause_id.replace(" ", "_").lower()
            
            self.collection.add(
                ids=[clause_id],
                embeddings=[vector],
                documents=[clause["clause_text"]],
                metadatas=[{
                    "contract_type": clause["contract_type"],
                    "clause_title": clause["clause_title"]
                }]
            )
        self.client.persist()

    def similarity_search(self, query_vector, top_k=5, contract_type: str | None = None):
        where_filter = None

        if contract_type:
            where_filter = {
                "contract_type": contract_type
            }

        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
