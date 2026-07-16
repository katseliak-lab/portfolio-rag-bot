# 💬 Portfolio RAG Support Bot

A retrieval-augmented (RAG) support bot that answers questions about a candidate,
grounded in a real knowledge base and powered by the **Anthropic Claude API**.
Ask it about experience, projects, or skills — it answers only from the documents
it was given, and says so when it doesn't know.

**🔗 Live demo:** [add your Streamlit Cloud URL here]

## How it works

```
User question
     │
     ▼
sentence-transformers  ──►  embed the question
     │
     ▼
NumPy cosine search    ──►  retrieve the most relevant chunks
     │                       from the knowledge base (Markdown files)
     ▼
Anthropic Claude       ──►  answer, constrained to the retrieved context
     │
     ▼
Streamlit chat UI (streamed response)
```

1. **Ingest** — every Markdown file in `knowledge/` is split into chunks tagged
   with their heading (`rag.py`).
2. **Embed** — chunks are embedded once at startup with
   `all-MiniLM-L6-v2` (a small, free, CPU-friendly model).
3. **Retrieve** — each question is embedded and matched against the chunks by
   cosine similarity; the top matches become the context.
4. **Generate** — Claude answers using *only* that context, in the user's
   language (Ukrainian or English), and refuses to invent facts.

## Tech stack

| Layer | Choice |
|---|---|
| LLM | Anthropic Claude (`claude-haiku-4-5`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | In-memory NumPy cosine similarity |
| UI | Streamlit chat (streaming) |
| Hosting | Streamlit Community Cloud (free) |

The knowledge base is small, so an in-memory vector search is simpler and faster
to boot than a full vector database. Swapping in FAISS or Chroma is a one-function
change in `rag.py`.

## Run locally

```bash
git clone https://github.com/katseliak-lab/portfolio-rag-bot
cd portfolio-rag-bot
python -m venv .venv && . .venv/Scripts/activate    # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Paste your key into a local secrets file:
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it
streamlit run streamlit_app.py                                # opens http://localhost:8501
```

## Make it yours

Replace the placeholder content in `knowledge/` with your own résumé:

- `about.md` — summary, location, contact
- `experience.md` — roles and education
- `projects.md` — projects with tech stacks
- `skills.md` — languages, tools, ways of working

No code changes needed — just edit the Markdown and restart.

## Deploy the live demo (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app** → pick your `portfolio-rag-bot` repo, branch `main`,
   main file `streamlit_app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. **Deploy.** After the build you get a public URL — put it in this README and
   your CV.

## Project structure

```
portfolio-rag-bot/
├── streamlit_app.py   # Streamlit UI + Claude generation
├── rag.py             # chunking, embeddings, retrieval
├── knowledge/         # the knowledge base (edit these)
│   ├── about.md
│   ├── experience.md
│   ├── projects.md
│   └── skills.md
├── requirements.txt
├── .streamlit/secrets.toml.example
└── README.md
```
