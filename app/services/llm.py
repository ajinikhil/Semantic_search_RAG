import json
import random
import time

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from app.config import BASE_DELAY, MAX_ATTEMPTS, MODEL_NAME, RETRYABLE_STATUS


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
    Return a lazily-created, module-level Gemini client.

    The client (config/credential setup) is built once and reused across
    requests instead of being re-instantiated on every call, mirroring the
    singleton pattern in embedder.get_model.
    """
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _generate(prompt, config=None):
    """
    Call the LLM with the given prompt, retrying transient server errors
    (429/500/503) with exponential backoff and jitter.

    Args:
        prompt: the fully built prompt string to send to the model.
        config: optional GenerateContentConfig (e.g. to request structured
            JSON output). Passed straight through to the Gemini client.

    Returns:
        The generated text response.

    Raises:
        errors.APIError: if a non-retryable error occurs, or the request
            still fails after the final retry.
    """
    client = get_client()
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=(prompt),
                config=config,
            )
            return response.text
        except errors.APIError as e:
            is_last = attempt == MAX_ATTEMPTS - 1
            if e.code not in RETRYABLE_STATUS or is_last:
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
    the retrived chunks and Gemini LM.

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
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_AnswerWithSources,
        )
        raw = _generate(prompt, config=config)
        result = _AnswerWithSources(**json.loads(raw))
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
    Generates a summary of the given document text using Gemini.

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
