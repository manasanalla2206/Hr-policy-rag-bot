# HR Policy Assistant — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about an uploaded
HR policy document (`.txt`, `.pdf`, or `.docx`), using LangChain, FAISS, and Groq's free
Llama 3 API. Built as a fresher AI Engineer portfolio project.

---

## How It Works (Architecture)

1. **Upload** — user uploads a document (HR policy, in this case)
2. **Chunking** — document is split into small overlapping text chunks
3. **Embedding** — each chunk is converted into a vector using a free local embedding model
   (`sentence-transformers/all-MiniLM-L6-v2`)
4. **Vector storage** — chunks + vectors are stored in a FAISS vector database
5. **Retrieval** — when a question is asked, the top-matching chunks are retrieved by
   semantic similarity
6. **Generation** — retrieved chunks + the question are sent to Llama 3 (via Groq's free
   API) to generate a grounded answer
7. **Eval display** — retrieval latency, generation latency, and a basic "context overlap"
   grounding check are shown for transparency

---

## Files in this project

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web app (deployable) |
| `requirements.txt` | Python dependencies needed to run the app |
| `sample_hr_policy.txt` / `.docx` / `.pdf` | Sample knowledge base to test with |

---

## Part 1: Push to GitHub

1. Create a free account at [github.com](https://github.com) if you don't have one
2. Click **New repository** → name it (e.g. `hr-policy-rag-bot`) → set **Public** → Create
3. Click **Add file → Upload files**
4. Upload `app.py` and `requirements.txt`
5. Scroll down, click **Commit changes**

**Important:** never upload a file containing your real API key. `app.py` as given does
not contain one — it reads the key at runtime, so it's safe to make this repo public.

---

## Part 2: Deploy on Streamlit Community Cloud (free hosting)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **New app**
4. Select:
   - Repository: `your-username/hr-policy-rag-bot`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**
6. Wait 2–4 minutes while it installs dependencies (`requirements.txt`)
7. You'll get a live public URL, e.g. `[[https://your-app-name.streamlit.app](https://hr-policy-rag-bot-g3puo3taacmctka2ptgkow.streamlit.app/)]

---

## Part 3: Add your real API key (safely)

**Do NOT put your key in the code.** Use Streamlit's built-in Secrets manager instead:

1. Get a free Groq API key at [console.groq.com](https://console.groq.com) → API Keys → Create key
2. On your deployed app's page on Streamlit Cloud, click the **⋮** menu (top right) → **Settings** → **Secrets**
3. Paste this into the box (replace with your real key):
   ```
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```
4. Click **Save** — the app will automatically restart and pick up the key
5. When you open the app now, the sidebar should show **"Groq API key loaded from Secrets ✓"**
   instead of asking you to type it in

If you ever run the app somewhere without Secrets configured (e.g. locally), it will
fall back to asking for the key in the sidebar manually — both paths work.

---

## Part 4: Test the app end-to-end

### Step-by-step test
1. Open your live app URL
2. Confirm the sidebar shows the key is loaded (or paste it manually if testing locally)
3. In the sidebar, upload `sample_hr_policy.txt` (or `.pdf` / `.docx`)
4. Click **Process Document(s)** — wait for the success message showing chunk count
5. In the chat box at the bottom, ask a test question (see table below)
6. Confirm the answer is accurate, then click **"Retrieval + eval details"** under the
   answer to see the retrieved chunks, similarity scores, and latency

### Expected test questions and answers

| Question | Expected Answer |
|---|---|
| How many annual leave days do I get? | 18 days per year, accrued at 1.5 days/month |
| What is the notice period for resignation? | Minimum 30 days written notice |
| Am I eligible for health insurance from day one? | Yes, no waiting period |
| Can I work remotely full-time? | Only with approval from manager and HR beyond the standard hybrid model |
| What happens if I violate the code of conduct? | Escalation: verbal warning → written warning → termination |

### Testing file format support
Repeat the upload + test-question process separately for `.txt`, `.docx`, and `.pdf` —
confirm all three give consistent, accurate answers.

### Testing edge cases (important for interview credibility)
- Upload an unsupported file type (e.g. `.jpg`) → should show a clear warning, not crash
- Ask something not in the document (e.g. "Is there a signing bonus?") → should say it's
  not covered in the context, not make something up
- Ask a vague/summarization question ("what is this file about?") → may return weak or
  no relevant chunks — this is an expected limitation of similarity-based retrieval,
  not a bug (see Limitations below)

---

## Known Limitations & Trade-offs

Being able to explain these clearly is a strong signal in an interview — it shows you
understand the system, not just that it "works."

| Limitation | Why | Production fix |
|---|---|---|
| Free local embeddings | Lower semantic accuracy than paid models | OpenAI/Cohere embeddings |
| FAISS is in-memory | Index rebuilds every restart, no persistence | Pinecone/Weaviate/Qdrant |
| Fixed chunk size | Manually tuned (300 chars), not adaptive | Dynamic chunking per document type |
| No re-ranking | Top-k by similarity only, not always most relevant | Add a re-ranker model |
| No conversation memory | Each question is independent | Add chat history to prompt |
| Basic grounding check | Simple word-overlap, not a real faithfulness score | RAGAS or similar eval framework |
| No OCR | Scanned PDFs return empty text | Add Tesseract OCR step |
| No access control | Anyone with the app can query any uploaded doc | Role-based access per document |
| Weak on summarization | RAG retrieval is built for factual Q&A, not "summarize this" | Route summarization queries to full-document context instead of retrieval |

---

## How to explain this project in an interview (summary)

> "I built a RAG-based HR Policy Assistant — a chatbot that answers employee questions
> using a company's actual policy document instead of relying on the LLM's general
> knowledge. The document gets chunked, embedded using a local Hugging Face model, and
> stored in a FAISS vector database. When a question comes in, the most relevant chunks
> are retrieved by similarity search and passed to Llama 3 via Groq's free API to generate
> a grounded answer. I deployed it as a live Streamlit app, added visibility into the
> retrieval process and basic eval metrics, and handled the API key securely using
> Streamlit's Secrets manager rather than hardcoding it. I also know its limitations —
> for example, it's optimized for factual Q&A, not document summarization, and the
> in-memory FAISS index wouldn't scale to production without a persistent vector database."

---

## Quick Reference Links
- Groq free API keys: https://console.groq.com
- Streamlit Cloud deployment: https://share.streamlit.io
- Your live app: *(fill in after deployment)*
- Your GitHub repo: *(fill in after upload)*
