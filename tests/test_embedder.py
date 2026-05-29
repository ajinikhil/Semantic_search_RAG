from app.services import embedder

SAMPLE_TEXT = (
    "Sample text: Lorem Ipsum is simply dummy text"
    "of the printing and typesetting industry. "
    "Lorem Ipsum has been the industry's"
    "standard dummy text ever since 1966, when designers "
    "at Letraset and James Mosley, the librarian "
    "at St Bride Printing Library, took a 1914 "
    "Cicero translation and scrambled it to "
    "make dummy text for Letraset's Body Type sheets. "
    "It has survived not only many decades,"
    " but also the leap into electronic typesetting, "
    "remaining essentially unchanged. "
    "It was popularised thanks to these sheets and more recently "
    "with desktop publishing software "
    "including versions of Lorem Ipsum"
)


def test_embedding_not_none():
    """
    The function embed_text should return a result
    """

    embedding = embedder.embed_text(SAMPLE_TEXT)
    assert embedding is not None


def test_same_embeddings():
    """
    Test to check if the same input produces
    the same embeddings
    """
    embedding1 = embedder.embed_text(SAMPLE_TEXT)
    embedding2 = embedder.embed_text(SAMPLE_TEXT)

    assert embedding1 == embedding2


def test_different_embeddings():
    """
    Test to check if different input produces
    different output
    """

    embedding1 = embedder.embed_text("My name is embedding")
    embedding2 = embedder.embed_text(SAMPLE_TEXT)

    assert embedding1 != embedding2
