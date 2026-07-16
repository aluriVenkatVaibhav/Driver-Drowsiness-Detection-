import json
import os

from vectorstore.chroma_store import ChromaClauseStore


# ----------------------------
# CONFIG
# ----------------------------

CLAUSE_JSON_PATH = "data/clause.json"   # adjust if needed
CHROMA_DB_PATH = "chroma_db"


# ----------------------------
# LOAD CLAUSES FROM JSON
# ----------------------------

def load_clauses(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clauses = []

    for contract in data:
        contract_type = contract["contract_type"]

        for clause in contract["clauses"]:
            clauses.append({
                "contract_type": contract_type,
                "clause_title": clause["clause_title"],
                "clause_text": clause["clause_text"]
            })

    return clauses


# ----------------------------
# BUILD VECTOR INDEX
# ----------------------------

def build_index():
    print("[IndexBuilder] Loading clauses...")
    clauses = load_clauses(CLAUSE_JSON_PATH)

    print(f"[IndexBuilder] Total clauses found: {len(clauses)}")

    store = ChromaClauseStore(CHROMA_DB_PATH)

    print("[IndexBuilder] Adding clauses to Chroma...")
    store.add_clauses(clauses)

    print("[IndexBuilder] Clause indexing completed successfully.")


# ----------------------------
# ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    build_index()
