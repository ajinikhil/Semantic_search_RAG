def build_prompt(question, chunks):
    texts = chunks["documents"][0]

    context = "\n\n".join(texts)

    prompt = f"""you are a helpful assistant. Answer
                the question only from context below.
                If the answer is not in the context say
                "I don't know".

                CONTEXT:
                {context}

                QUESTION:
                {question}

                ANSWER: ""
                """
    return prompt
