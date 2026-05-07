import uuid

import chromadb
from config import CHROMA_DB_PATH, CHROMA_RESULTS_RETURNED

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = chroma_client.get_or_create_collection(name="user_files")


def add_chunks(chunks, embeddings, filename):
    try:
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(uuid.uuid4()) for _ in chunks],
            metadatas=[
                {"filename": filename, "chunk_index": i} for i in range(len(chunks))
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Error while adding documents to ChromaDB: {e}")


def search(query_vector):
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=CHROMA_RESULTS_RETURNED,
            include=["documents", "metadatas", "distances"],
        )
        return results
    except Exception as e:
        raise RuntimeError(f"Error while shearching from ChromaDB: {e}")
