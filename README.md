# Semantic Search Engine

Work in progress — currently in active development

A local, fully private semantic search engine that lets you upload your own documents and ask questions about them in plain English. No cloud, no data leaving your machine.

How it worksÉ

You upload a document (PDF, DOCX, or TXT)
The system breaks it into chunks and converts each chunk into a vector (a mathematical representation of meaning)
Those vectors get stored in a local database (ChromaDB)
When you ask a question, it gets converted into a vector too
ChromaDB finds the chunks most similar in meaning to your question
Those chunks get sent to a local LLM (Ollama) along with your question
The LLM generates a grounded answer and sends it back to you

No guessing. No hallucinating facts that aren't in your documents.

@claude is writing the pytests
