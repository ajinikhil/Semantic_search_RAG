from config import CHUNK_OVERLAP, CHUNK_SIZE
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyxtxt import xtxt


def extract_text(document):
    """
    Extract the text from the documents (PDF, docx, txt)
    provided by the user

    Args:
    document: the documents, texts needed to be extracted from

    Returns:
    Text extracted from the documents

    Raises:
    Runtime error: if pyxtxt fails to extract text.
    """
    try:
        return xtxt(document)
    except Exception as e:
        raise RuntimeError(f"error extracting text: {e}")


def chunk_text(text):
    """
    Function to chunk the extracted texts from the documents

    Args:
    text: text extracted from the documents

    Returns:
    A list of text chunks of size CHUNK_SIZE with CHUNK_OVERLAP

    Raises:
    Runtime Error: if RecursiveCharacterTextSplitter fails whilw text splitting
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        return text_splitter.split_text(text)
    except Exception as e:
        raise RuntimeError(f"error chunking text: {e}")
