# Student PDF RAG Lab

A simple RAG (Retrieval-Augmented Generation) classroom project.

## Important about free models and tokens

There is no reliable promise of unlimited tokens on a free hosted LLM API. This project does not require any API key to run: it retrieves relevant PDF passages using local TF-IDF search. To add generated answers, you can optionally configure a Groq API key using its free tier, subject to the provider's current rate/token limits. Never commit API keys to GitHub.

## Files

- `app.py` — Streamlit front end.
- `rag_engine.py` — PDF text extraction, chunking, retrieval, and optional LLM call.
- `requirements.txt` — Python dependencies.

## Run locally (Windows)

1. Install Python 3.11 or 3.12.
2. Download or clone this repository.
3. Open PowerShell in the project folder.
4. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the app:

   ```powershell
   streamlit run app.py
   ```

7. A browser tab should open. Upload a text-based PDF, click **Read PDF and build RAG index**, then ask a question.

### Optional: enable generated answers with Groq

1. Create an account and API key at the Groq developer console.
2. Set the key only in your local terminal (do not put it in source code):

   ```powershell
   $env:GROQ_API_KEY="paste-your-key-here"
   streamlit run app.py
   ```

The app uses `llama-3.3-70b-versatile` by default. Provider model availability and free-tier limits can change. If the model name is unavailable, set `GROQ_MODEL` to a currently available model.

Without the key, the app still works in retrieval-only mode and shows the best matching passages.

## Put the project on GitHub

1. Sign in at https://github.com and create a new repository, e.g. `student-pdf-rag-lab`.
2. Keep it public if you want the easiest Streamlit Community Cloud deployment.
3. Upload `app.py`, `rag_engine.py`, `requirements.txt`, and `README.md` to the repository root.
4. Do not upload private PDFs, API keys, `.env` files, or virtual-environment folders.

## Deploy for free on Streamlit Community Cloud

1. Push the files to GitHub.
2. Open https://share.streamlit.io and sign in using GitHub.
3. Click **Create app**.
4. Select the repository, branch (`main`), and main file path `app.py`.
5. In Advanced settings, choose Python 3.12 if available, then deploy.
6. Wait for installation and launch. Streamlit will install the packages listed in `requirements.txt`.
7. Open the app URL and test it with a small, text-based PDF.

### Optional Groq key on Streamlit Cloud

1. Open the deployed app's settings and find **Secrets**.
2. Add the following TOML entry (replace the placeholder with your real key):

   ```toml
   GROQ_API_KEY = "your-key-here"
   ```

3. Save and reboot/redeploy the app.

The app reads `GROQ_API_KEY` from environment variables. Streamlit Cloud exposes configured secrets as environment variables for simple string keys. If you prefer, adapt the code to read `st.secrets["GROQ_API_KEY"]`.

## Teaching notes / limitations

- This is a small educational prototype, not production-grade RAG.
- “Training” here means processing the PDF, splitting it into chunks, and building a retrieval index. The LLM itself is not fine-tuned or trained on the PDF.
- TF-IDF is lightweight and free but is lexical search, not a semantic embedding model.
- Scanned/image-only PDFs are not supported unless OCR is added.
- The uploaded PDF/index lives in the current app session; it is not saved permanently.
- Free API plans have quotas and can change. No unlimited hosted token allowance is promised.
