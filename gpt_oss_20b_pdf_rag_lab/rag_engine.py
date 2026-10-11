import os
from io import BytesIO
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_pdf_pages(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_no, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_no, "text": text})
    if not pages:
        raise ValueError("No readable text found. This may be a scanned PDF; OCR is needed.")
    return pages


def make_chunks(pages, chunk_size=1200, overlap=180):
    if chunk_size < 200 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size >= 200 and overlap between 0 and chunk_size-1.")
    chunks = []
    for page in pages:
        text = re.sub(r"[ \\t]+", " ", page["text"]).strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": page["page"], "text": piece})
            if end >= len(text):
                break
            start = end - overlap
    if not chunks:
        raise ValueError("No text chunks could be created.")
    return chunks


class PDFRAG:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def retrieve(self, question, top_k=4):
        if not question.strip():
            return []
        qv = self.vectorizer.transform([question])
        scores = cosine_similarity(qv, self.matrix).ravel()
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [
            {"page": self.chunks[i]["page"], "text": self.chunks[i]["text"],
             "score": float(scores[i]), "chunk_id": i + 1}
            for i in order[:min(top_k, len(order))]
        ]

    def answer(self, question, top_k=4):
        sources = self.retrieve(question, top_k)
        if not sources or all(s["score"] <= 0 for s in sources):
            return {"answer": "I couldn't find a relevant passage in this PDF. Try different wording.",
                    "mode": "No matching passage", "sources": sources}
        context = "\n\n".join(
            f"[PDF page {s['page']}, chunk {s['chunk_id']}]\n{s['text']}" for s in sources
        )
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            return {"answer": "LLM generation is not configured. These are retrieved PDF passages, not a generated answer:\n\n" + context,
                    "mode": "Retrieval only (GROQ_API_KEY missing)", "sources": sources}
        try:
            from groq import Groq
            client = Groq(api_key=api_key, timeout=90.0, max_retries=2)
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                messages=[
                    {"role": "system", "content":
                     "Answer in simple English using only the supplied PDF passages. Give a complete answer, do not stop mid-sentence, and do not invent facts. If the answer is absent, say so. End with the PDF page numbers used."},
                    {"role": "user", "content": f"PDF passages:\n{context}\n\nQuestion: {question}\n\nGive a complete, clear answer."}
                ],
                temperature=0.2,
                max_completion_tokens=1800,
            )
            choice = response.choices[0]
            answer = choice.message.content or "The model returned no answer text. Try a narrower question."
            if getattr(choice, "finish_reason", None) == "length":
                answer += "\n\nThe model reached its output limit. Ask a narrower question for a complete answer."
            return {"answer": answer, "mode": "Groq · " + os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                    "sources": sources}
        except Exception as exc:
            return {"answer": "LLM request failed. Check the Groq key, model availability, and free-tier limits. Retrieved passages are shown below:\n\n" + context,
                    "mode": f"LLM request failed ({type(exc).__name__})", "sources": sources}
