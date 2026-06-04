from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text):
    """
    Embed the text chunks extracted from documents
    before storing it into ChromaDB

    Args:
    text: text chunks to embed.

    Returns:
    list of vector embeddings

    Raises:
    Runtime error: if the embedding model fails to encode the texts
    """
    if not text:
        raise ValueError("text cannot be empty")

    try:
        embeddings = get_model().encode(text, normalize_embeddings=True)
        return embeddings.tolist()
    except Exception as e:
        raise RuntimeError(f"Error encoding text: {e}")


def embed_query(query):
    """
    Embed the query text into vector for similarity search
    in ChromaDB

    Args:
    query: The users question

    Returns:
    embedding vector as a list

    Raises:
    Runtime error: if the embedding model fails to encode the query
    """
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    try:
        embedding = get_model().encode([query], normalize_embeddings=True)
        return embedding[0].tolist()
    except Exception as e:
        raise RuntimeError(f"Error encoding query: {e}")
