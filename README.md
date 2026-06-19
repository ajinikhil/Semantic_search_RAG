# Semantic Search Engine

> [!IMPORTANT]
> currently in active development : )

A search engine which provides answers only from the the documents you provide.

> *  For simplicity and faster development the project is currenly using Gemini.
> * The main goal of this system is to keep your files private and ask questions about them without any data leaving your machine.
> * No hallucinating or guessing. if the answer is not in the context provided, the LLM will indicate the answer is not in the documents.
---

## How it works
* User uploads documents.
* The system will parse the texts form the documents.
* the extracted texts will be chuked and embed and add it to ChromaDB.
* User askes a question and the question will be embeded and compare with the embedded chunks in ChromaDB.
* Top 5 chunks will be retrieved from ChromaDB and will be send to LLM (Currenly using Gemini).
* The LLM checks the chunks and generate an answer if it is in the retrived chunks.
* The user will be presented the answer as well as the sources (Document name and the chunks).

# Gemini API Key

get your API key at: `https://aistudio.google.com/api-keys`

* Create an env file and add: `GEMINI_API_KEY="your_APIA_key"`

# Hugging Face Token

* Create an account in Hugging Face
* Go to: `https://huggingface.co/settings/tokens`
* Create token
* Paste `export HUGGINGFACEHUB_API_TOKEN="paste_your_token_here"` in your terminal

## Run the server

`uvicorn app.main:app --reload`

## Run Streamlit UI

`streamlit run ui/ui.py`

## CI/CD

The pipeline runs on every push to main and on all pull requests to main

## Credits

* Testing (pytests) and streamlit UI: @claude
* Everything Else: Nikhil Aji (nklajiofficial@gmail.com)
