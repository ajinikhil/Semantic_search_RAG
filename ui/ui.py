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
if "summaries" not in st.session_state:
    st.session_state.summaries = {}

# --- Sidebar: document upload + document list ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_files = st.file_uploader(
        "Choose files (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded_files:
        if st.button("upload"):
            total_chunks = 0
            errors = []
            with st.spinner(f"Ingesting {len(uploaded_files)} file(s)..."):
                for f in uploaded_files:
                    try:
                        resp = requests.post(
                            f"{BASE_URL}/ingest/upload",
                            files={"file": (f.name, f, f.type)},
                            timeout=60,
                        )
                        if resp.ok:
                            total_chunks += resp.json()["chunks_created"]
                        else:
                            errors.append(
                                f"{f.name}: " + resp.json().get("detail", "failed")
                            )
                    except requests.ConnectionError:
                        errors.append(f"{f.name}: backend unreachable")
            if total_chunks:
                st.success(
                    f"""{len(uploaded_files)} file(s) ingested
                      — {total_chunks} chunks created"""
                )
            for err in errors:
                st.error(err)
            st.session_state.uploader_key += 1
            st.rerun()

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
                    col1, col2, col3 = st.columns([3, 1, 1])
                    col1.markdown(f"📄 {doc}")
                    if col2.button("Summary", key=f"sum_{doc}"):
                        with st.spinner(f"Summarizing {doc}..."):
                            try:
                                sum_resp = requests.get(
                                    f"{BASE_URL}/ingest/documents/{doc}/summary",
                                    timeout=120,
                                )
                                if sum_resp.ok:
                                    st.session_state.summaries[doc] = sum_resp.json()[
                                        "summary"
                                    ]
                                else:
                                    st.error(
                                        sum_resp.json().get("detail", "Summary failed")
                                    )
                            except requests.ConnectionError:
                                st.error("Backend unreachable.")
                    if col3.button("Delete", key=f"del_{doc}"):
                        del_resp = requests.delete(
                            f"{BASE_URL}/ingest/documents/{doc}", timeout=10
                        )
                        if del_resp.ok:
                            st.session_state.summaries.pop(doc, None)
                            st.success(f"Deleted {doc}")
                            st.rerun()
                        else:
                            st.error(del_resp.json().get("detail", "Delete failed"))
                    if st.session_state.summaries.get(doc):
                        with st.expander(f"Summary of {doc}"):
                            st.markdown(st.session_state.summaries[doc])
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
