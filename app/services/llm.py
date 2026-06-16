from google import genai

from app.config import MODEL_NAME


def get_client():
    return genai.Client()


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

        client = get_client()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=(prompt),
        )
        answer = response.text
        return answer
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
