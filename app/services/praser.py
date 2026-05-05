from config import CHUNK_OVERLAP, CHUNK_SIZE
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyxtxt import xtxt

"Function to extract text from documents (PDF, docx, txt)"


def extract_text(document):
    return xtxt(document)


"Function to chunk the extracted text"


def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_text(text)
