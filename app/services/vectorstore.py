import uuid

import chromadb

from app.config import CHROMA_DB_PATH, CHROMA_RESULTS_RETURNED

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = chroma_client.get_or_create_collection(name="user_files")


def add_chunks(chunks, embeddings, filename):
    """
    inserts text chunks and their embeddings into ChromaDB

    Args:
        chunks: The text chunks extracted from documents
        embeddings: embeddings of the chunks
        filename: from which file those chunks are from

    Returns:
            number of chunks inserted

    Raises:
        RuntimeError: if the chunks were failed to insert in ChromaDB
    """
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

    return len(chunks)


def search(query_vector):
    """
    Searches for the most similar chunks to the query vector in ChromaDB

    Args:
        query_vector: the question user asked as a vector

    Returns:
            top 5 closest result related to the users question

    Raises:
        RuntimeError: if retreving results from chromaDB
    """
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=CHROMA_RESULTS_RETURNED,
            include=["documents", "metadatas", "distances"],
        )
        return results
    except Exception as e:
        raise RuntimeError(f"Error while searching from ChromaDB: {e}")


def delete_by_file_name(filename):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        results = collection.get(where={"filename": filename})
        ids = results["ids"]
        if ids:
            collection.delete(ids=ids)
            return len(ids)
        return 0
    except Exception as e:
        raise RuntimeError(f"Error deleting the file: {e}")


def get_stats():
    try:
        total = collection.count()

        if total == 0:
            return {"total_chunks": 0, "documents": []}

        all_metadata = collection.get(include=["metadatas"])["metadatas"]
        documents = sorted(set(m["filename"] for m in all_metadata))

        return {"total_chunks": total, "documents": documents}

    except Exception as e:
        raise RuntimeError(f"Error while retreving stats from ChromaDB: {e}")
