import random
import time

import ollama
from pydantic import BaseModel

from app.config import (
    BASE_DELAY,
    MAX_ATTEMPTS,
    MODEL_NAME,
    OLLAMA_HOST,
    RETRYABLE_STATUS,
)


class _AnswerWithSources(BaseModel):
    """
    Structured answer returned by the LLM for a RAG query.

    Args:
        answer: the answer text generated from the sources.
        used_sources: 1-based numbers of the sources the answer actually
            relies on (matching the [n] labels in the prompt). Empty when
            the answer is not grounded in any source (e.g. "I don't know").
    """

    answer: str
    used_sources: list[int]


_client = None


def get_client():
    """
    Return a lazily-created, module-level Ollama client.

    The client is built once and reused across requests instead of being
    re-instantiated on every call, mirroring the singleton pattern in
    embedder.get_model. Honors OLLAMA_HOST when set, otherwise talks to
    the local Ollama daemon.
    """
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    return _client


def _generate(prompt, response_format=None):
    """
    Call the local LLM with the given prompt, retrying transient server
    errors (429/500/503) with exponential backoff and jitter.

    Args:
        prompt: the fully built prompt string to send to the model.
        response_format: optional JSON schema (dict) constraining the
            output to structured JSON. Passed to Ollama's ``format``.

    Returns:
        The generated text response.

    Raises:
        ollama.ResponseError: if a non-retryable error occurs, or the
            request still fails after the final retry.
    """
    client = get_client()
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                format=response_format,
                options={"temperature": 0},
            )
            return response["message"]["content"]
        except ollama.ResponseError as e:
            is_last = attempt == MAX_ATTEMPTS - 1
            if e.status_code not in RETRYABLE_STATUS or is_last:
                raise
            delay = BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)
            time.sleep(delay)


def build_prompt(question, chunks):
    """
    This function builds a prompt for the LLM
    using the users question and retrieved chunks.

    Args:
    question: question or query asked by the user

    chunks: Dictionary containing retrived document chunks

    Returns:
    prompt: A prompt containing the context and question for
            the language model.

    Raise:
    Runtime error: if the function fails to build a prompt
    """
    try:

        context = "\n\n".join(
            f"[{i}] Source: {chunk.file_name}, chunk {chunk.chunk_index}\n{chunk.text}"
            for i, chunk in enumerate(chunks, start=1)
        )

        prompt = f"""you are a helpful assistant. Answer
                    the question only from the numbered sources below.
                    If the answer is not in the sources say
                    "I don't know".

                    In "used_sources", list the numbers of only the
                    sources you actually relied on for the answer. Leave
                    it empty if the answer is not based on any source.

                    SOURCES:
                    {context}

                    QUESTION:
                    {question}
                    """
        return prompt
    except Exception as e:
        raise RuntimeError(f"Error building prompt {e}")


def generate_answer(question, chunks):
    """
    Generates answer for the user's question using
    the retrived chunks and a local LLM.

    Args:
        question: the question asked by the user
        chunks: Dictionary containing retrived document chunks
                used as context for genereating answer.

    Returns:
        A tuple of (answer, used_sources) where answer is the generated
        text and used_sources is the list of 1-based source numbers the
        answer relied on (matching the order of ``chunks``).

    Raise:
     Runtime error: if the model fails to generate an answer
    """
    try:
        prompt = build_prompt(question, chunks)
        schema = _AnswerWithSources.model_json_schema()
        raw = _generate(prompt, response_format=schema)
        result = _AnswerWithSources.model_validate_json(raw)
        return result.answer, result.used_sources
    except Exception as e:
        raise RuntimeError(f"Error generating answer {e}")


def build_summarize_prompt(text):
    """
    This function builds a prompt for the LLM
    to summarize the document for the user

    Args:
    text: the extracted text from the document

    Returns:
    prompt: prompt for the LLM to summarize
    """
    try:
        prompt = f"""You are a summarization expert.
                    Write a clear, concise summary of the text below,
                    capturing its main points and key details.

                    Important:
                    - Do not add facts, assumptions, or outside knowledge.
                    - Focus on the main ideas, important details, and conclusions.

                    CONTEXT:
                    {text}

                    SUMMARY:
                    """
        return prompt
    except Exception as e:
        raise RuntimeError(f"Error building prompt for summary: {e}")


def generate_summary(text):
    """
    Generates a summary of the given document text using the local LLM.

    Args:
        text: the extracted text from the document to summarize

    Returns:
        summary: the generated summary from Gemini

    Raise:
        RuntimeError: if the model fails to generate a summary
    """
    try:
        prompt = build_summarize_prompt(text)
        return _generate(prompt)
    except Exception as e:
        raise RuntimeError(f"Error generating summary {e}")
