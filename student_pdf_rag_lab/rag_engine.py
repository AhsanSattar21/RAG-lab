"""Small, classroom-friendly RAG engine for PDF files.

Retrieval uses TF-IDF (free, local, no embedding API). Optional answer generation
uses Groq's free-tier API if GROQ_API_KEY is configured. Without an API key,
the app returns relevant passages instead of pretending to have generated an LLM answer.
"""
from io import BytesIO
import os
import re
from typing import List, Dict

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a text-based PDF."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_number}]\n{text}")
    full_text = "\n\n".join(pages).strip()
    if not full_text:
        raise ValueError(
            "No readable text was found. This may be a scanned/image-only PDF. "
            "Try a text-based PDF or add OCR as a separate lab exercise."
        )
    return full_text


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split text into overlapping character chunks for a simple demo."""
    if chunk_size < 100:
        raise ValueError("chunk_size should be at least 100 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


class SimpleRAG:
    """A tiny TF-IDF retriever with optional hosted LLM answer generation."""

    def __init__(self, chunks: List[str]):
        if not chunks:
            raise ValueError("No text chunks were created from this PDF.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        indices = scores.argsort()[::-1][: min(top_k, len(self.chunks))]
        results = []
        for index in indices:
            results.append({
                "chunk": self.chunks[int(index)],
                "score": float(scores[int(index)]),
                "chunk_number": int(index) + 1,
            })
        return results

    def answer(self, question: str, top_k: int = 3) -> Dict:
        results = self.retrieve(question, top_k=top_k)
        context = "\n\n---\n\n".join(
            f"Passage {i + 1}:\n{item['chunk']}" for i, item in enumerate(results)
        )

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if api_key:
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful assistant for a student RAG lab. "
                                "Answer using only the supplied PDF passages. "
                                "If the passages do not contain the answer, say you cannot "
                                "find it in the uploaded PDF. Do not invent facts."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"PDF passages:\n{context}\n\nQuestion: {question}",
                        },
                    ],
                    temperature=0.1,
                )
                return {
                    "answer": response.choices[0].message.content or "",
                    "mode": "Groq LLM + TF-IDF retrieval",
                    "sources": results,
                }
            except Exception as exc:
                return {
                    "answer": (
                        "The LLM request did not work (often because the free API quota, "
                        "API key, or model access is unavailable). Here are the most relevant "
                        "PDF passages instead.\n\n" + context
                    ),
                    "mode": f"Passage fallback (LLM error: {type(exc).__name__})",
                    "sources": results,
                }

        return {
            "answer": (
                "No GROQ_API_KEY is configured, so this is retrieval-only mode rather "
                "than an LLM-generated answer. Review the passages below to answer the question:\n\n"
                + context
            ),
            "mode": "TF-IDF retrieval only (no API key required)",
            "sources": results,
        }
