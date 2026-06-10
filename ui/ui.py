import requests
import streamlit as st

BASE_URL = "http://localhost:8000"

st.title("Semantic Search RAG")

# --- State init ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me anything about the documents you uploaded.",
        }
    ]
if "thinking" not in st.session_state:
    st.session_state.thinking = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- Sidebar: document upload + document list ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded_file is not None:
        if st.button("upload"):
            with st.spinner("Ingesting..."):
                try:
                    resp = requests.post(
                        f"{BASE_URL}/ingest/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file,
                                uploaded_file.type,
                            )
                        },
                        timeout=60,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.success(data["message"])
                        st.caption(f"{data['chunks_created']} chunks created")
                        st.session_state.uploader_key += 1
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Upload failed"))
                except requests.ConnectionError:
                    st.error("Cannot reach the backend. Is the server running?")

    st.divider()
    st.header("Documents")
    try:
        db_resp = requests.get(f"{BASE_URL}/ingest/documents", timeout=10)
        if db_resp.ok:
            db_info = db_resp.json()
            docs = db_info.get("documents", [])
            if docs:
                st.caption(f"{db_info['total_chunks']} total chunks")
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"📄 {doc}")
                    if col2.button("Delete", key=f"del_{doc}"):
                        del_resp = requests.delete(
                            f"{BASE_URL}/ingest/documents/{doc}", timeout=10
                        )
                        if del_resp.ok:
                            st.success(f"Deleted {doc}")
                            st.rerun()
                        else:
                            st.error(del_resp.json().get("detail", "Delete failed"))
            else:
                st.caption("No documents ingested yet.")
        else:
            st.warning("Could not load documents.")
    except requests.ConnectionError:
        st.caption("Backend unavailable.")

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.markdown(
                        f"**{src['file_name']}** — chunk {src['chunk_index']} "
                        f"(score: {src['similarity_score']})\n\n> {src['text']}"
                    )

# --- Phase 2: process pending query while input is disabled ---
if st.session_state.thinking:
    pending = next(
        m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"
    )
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BASE_URL}/query",
                    json={"query": pending},
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()
                    answer = data["response"]
                    sources = data.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                name = src["file_name"]
                                idx = src["chunk_index"]
                                score = src["similarity_score"]
                                st.markdown(
                                    f"**{name}** — chunk {idx} "
                                    f"(score: {score})\n\n> {src['text']}"
                                )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    error = resp.json().get("detail", "Query failed")
                    st.error(error)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Error: {error}"}
                    )
            except requests.ConnectionError:
                msg = "Cannot reach the backend. Is the server running on port 8000?"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
    st.session_state.thinking = False
    st.rerun()

# --- Phase 1: capture new prompt, disable input until answer arrives ---
prompt = st.chat_input(
    "Ask a question about your documents",
    disabled=st.session_state.thinking,
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.thinking = True
    st.rerun()
