import os
import streamlit as st
from rag_engine import extract_pdf_pages, make_chunks, PDFRAG

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
    if "GROQ_MODEL" in st.secrets:
        os.environ.setdefault("GROQ_MODEL", st.secrets["GROQ_MODEL"])
except Exception:
    pass

st.set_page_config(page_title="GPT-OSS-20B PDF RAG Lab", page_icon="📘", layout="wide")
st.title("📘 GPT-OSS-20B PDF RAG Lab")
st.write("Upload a text-based PDF, build a searchable index, and ask questions. "
         "Retrieval uses local TF-IDF; answer generation uses Groq's API with openai/gpt-oss-20b when configured.")

with st.sidebar:
    st.header("Retrieval settings")
    chunk_size = st.slider("Chunk size (characters)", 400, 2000, 1200, 100)
    overlap = st.slider("Chunk overlap (characters)", 0, 400, 180, 30)
    top_k = st.slider("Retrieved chunks", 1, 6, 4)
    st.caption("The last chunk of each page is preserved. Overlap helps keep context.")

uploaded = st.file_uploader("1. Upload a text-based PDF", type=["pdf"])
if uploaded and st.button("2. Read PDF and build RAG index", type="primary"):
    try:
        with st.spinner("Extracting PDF pages and building the index..."):
            pages = extract_pdf_pages(uploaded.getvalue())
            chunks = make_chunks(pages, chunk_size, overlap)
            st.session_state["rag"] = PDFRAG(chunks)
            st.session_state["pdf_name"] = uploaded.name
            st.session_state["page_count"] = len(pages)
            st.session_state["chunk_count"] = len(chunks)
            st.session_state["text_chars"] = sum(len(p["text"]) for p in pages)
        st.success("PDF indexed. Ask a question below.")
    except Exception as e:
        st.error(f"Could not index PDF: {e}")

if "rag" in st.session_state:
    a, b, c, d = st.columns(4)
    a.metric("PDF", st.session_state["pdf_name"])
    b.metric("Readable pages", st.session_state["page_count"])
    c.metric("Text characters", f"{st.session_state['text_chars']:,}")
    d.metric("Chunks", st.session_state["chunk_count"])
    with st.form("ask_form"):
        question = st.text_area("3. Ask a question about your PDF",
                                placeholder="What are the main ideas in this document?", height=90)
        submitted = st.form_submit_button("Find answer", type="primary")
    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving passages and generating answer..."):
                result = st.session_state["rag"].answer(question, top_k)
            st.subheader("Answer")
            st.caption("Mode: " + result["mode"])
            st.markdown(result["answer"])  # No fixed-height container: full answer can expand.
            with st.expander("Inspect retrieved source passages"):
                for src in result["sources"]:
                    st.markdown(f"**PDF page {src['page']} · chunk {src['chunk_id']} · similarity {src['score']:.3f}**")
                    st.write(src["text"])
                    st.divider()
else:
    st.info("Upload a PDF and click **Read PDF and build RAG index** to begin.")

with st.expander("How this RAG lab works"):
    st.markdown("""
    1. Extract text from every readable PDF page.
    2. Split text into overlapping chunks; the tail of each page is preserved.
    3. Build a local TF-IDF retrieval index.
    4. Retrieve passages relevant to the question.
    5. If a Groq key is configured, ask `openai/gpt-oss-20b` to answer using those passages.

    In this lab, “train the RAG” means indexing the PDF, not fine-tuning the LLM. Scanned PDFs need OCR.
    """)
