# GPT-OSS-20B PDF RAG Lab

A simple student lab: upload a PDF, index its text, retrieve relevant passages, and ask questions.

## Free-use note
This project does not call the paid OpenAI API. It calls `openai/gpt-oss-20b` through Groq when `GROQ_API_KEY` is set. Free access and model availability depend on Groq's current plan and quotas; unlimited hosted tokens cannot be promised. Retrieval works without a key, using local TF-IDF.

## Run locally (Python 3.12, Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:GROQ_API_KEY="your-groq-key"
streamlit run app.py
```
Get a key from https://console.groq.com/. Do not commit API keys to GitHub. Without a key, the app clearly labels output as retrieved passages.

## GitHub files
Upload `app.py`, `rag_engine.py`, `requirements.txt`, and `README.md` to the repository root.

## Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Create app, choose your repository and `main` branch.
3. Main file path: `app.py`; choose Python 3.12 if offered.
4. In Advanced settings / Secrets, enter:
```toml
GROQ_API_KEY = "your-groq-key"
GROQ_MODEL = "openai/gpt-oss-20b"
```
5. Save and deploy. For an existing app use Manage app → Settings → Secrets.

Never store API keys in GitHub source files.

## Avoiding cut-off answers
- The answer renders in a normal expanding page, not a fixed-height box.
- Page text is split into overlapping chunks and the final text of each page is retained.
- Generation requests up to 1800 completion tokens. An API cannot guarantee arbitrarily long answers; if the model reaches its output limit, the app says so.
- Ask focused questions and inspect source passages for verification.
- This demo uses lexical TF-IDF search, not semantic embeddings. Scanned PDFs need OCR.
- The PDF index exists in memory for the current app session and is not permanently saved.
