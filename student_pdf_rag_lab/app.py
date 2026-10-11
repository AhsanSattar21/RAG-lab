import os
import streamlit as st

# Streamlit Community Cloud stores configured values in st.secrets.
# Copy the optional key into the environment so the engine can read it.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
    if "GROQ_MODEL" in st.secrets:
        os.environ.setdefault("GROQ_MODEL", st.secrets["GROQ_MODEL"])
except Exception:
    pass

from rag_engine import extract_pdf_text, split_into_chunks, SimpleRAG

st.set_page_config(page_title="Student PDF RAG Lab", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Lab")
st.write(
    "Upload a PDF, build a searchable index, and ask questions. "
    "This classroom demo retrieves relevant passages using TF-IDF. "
    "An optional free-tier Groq API key can generate a natural-language answer."
)

with st.sidebar:
    st.header("RAG settings")
    chunk_size = st.slider("Chunk size (characters)", min_value=300, max_value=1800, value=900, step=100)
    overlap = st.slider("Chunk overlap (characters)", min_value=0, max_value=400, value=150, step=50)
    top_k = st.slider("Retrieved passages", min_value=1, max_value=6, value=3)
    st.caption("Tip: overlap helps preserve context across chunk boundaries.")

uploaded_pdf = st.file_uploader("1. Upload a text-based PDF", type=["pdf"])

if uploaded_pdf is not None:
    if st.button("2. Read PDF and build RAG index", type="primary"):
        try:
            with st.spinner("Reading PDF and creating the searchable index..."):
                pdf_bytes = uploaded_pdf.getvalue()
                text = extract_pdf_text(pdf_bytes)
                chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                rag = SimpleRAG(chunks)
                st.session_state["rag_object"] = rag
                st.session_state["pdf_name"] = uploaded_pdf.name
                st.session_state["pdf_chars"] = len(text)
                st.session_state["chunk_count"] = len(chunks)
            st.success("PDF indexed successfully. You can now ask questions.")
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")

if "rag_object" in st.session_state:
    st.divider()
    st.subheader("Index status")
    col1, col2, col3 = st.columns(3)
    col1.metric("PDF", st.session_state.get("pdf_name", "Uploaded"))
    col2.metric("Text characters", f"{st.session_state.get('pdf_chars', 0):,}")
    col3.metric("Chunks", st.session_state.get("chunk_count", 0))

    question = st.text_input("3. Ask a question about your PDF", placeholder="What are the main ideas in this document?")
    if st.button("Find answer"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Searching the PDF..."):
                result = st.session_state["rag_object"].answer(question, top_k=top_k)
            st.markdown("### Answer / retrieved passages")
            st.write(result["answer"])
            st.caption(f"Mode: {result['mode']}")
            with st.expander("View retrieved source passages"):
                for item in result["sources"]:
                    st.markdown(
                        f"**Chunk {item['chunk_number']} · similarity {item['score']:.3f}**"
                    )
                    st.write(item["chunk"])
                    st.divider()
else:
    st.info("Upload a PDF and click **Read PDF and build RAG index** to begin.")

with st.expander("What is happening in this RAG lab?"):
    st.markdown(
        """
        1. **Load:** extract text from the uploaded PDF.
        2. **Chunk:** divide the text into overlapping passages.
        3. **Index:** convert passages into TF-IDF vectors.
        4. **Retrieve:** find passages most similar to the student's question.
        5. **Generate (optional):** send the question and retrieved passages to a free-tier LLM API.

        The index is created in memory for the current app session; it is not permanently saved.
        Scanned PDFs need OCR, which is intentionally left as an extension exercise.
        """
    )
