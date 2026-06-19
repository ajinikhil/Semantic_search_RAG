import random
import time

from google import genai
from google.genai import errors

from app.config import MODEL_NAME

# Transient errors that are safe to retry: rate limit (429),
# internal error (500) and "model overloaded" (503).
_RETRYABLE_STATUS = {429, 500, 503}
_MAX_ATTEMPTS = 4
_BASE_DELAY = 1.0  # seconds; doubled each retry with jitter


def get_client():
    return genai.Client()


def _generate(prompt):
    """
    Call the LLM with the given prompt, retrying transient server errors
    (429/500/503) with exponential backoff and jitter.

    Args:
        prompt: the fully built prompt string to send to the model.

    Returns:
        The generated text response.

    Raises:
        errors.APIError: if a non-retryable error occurs, or the request
            still fails after the final retry.
    """
    client = get_client()
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=(prompt),
            )
            return response.text
        except errors.APIError as e:
            is_last = attempt == _MAX_ATTEMPTS - 1
            if e.code not in _RETRYABLE_STATUS or is_last:
                raise
            delay = _BASE_DELAY * (2**attempt) + random.uniform(0, 0.5)
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
            f"[Source: {chunk.file_name}, chunk {chunk.chunk_index}]\n{chunk.text}"
            for chunk in chunks
        )

        prompt = f"""you are a helpful assistant. Answer
                    the question only from context below.
                    If the answer is not in the context say
                    "I don't know".

                    CONTEXT:
                    {context}

                    QUESTION:
                    {question}

                    ANSWER:
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
        answer: The genarated answer from Gemini LM

    Raise:
     Runtime error: if the model fails to generate an answer
    """
    try:
        prompt = build_prompt(question, chunks)
        return _generate(prompt)
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
