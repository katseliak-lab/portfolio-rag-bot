---
title: Portfolio RAG Support Bot
emoji: 💬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 💬 Portfolio RAG Support Bot

A retrieval-augmented (RAG) support bot that answers questions about a candidate,
grounded in a real knowledge base and powered by the **Anthropic Claude API**.
Ask it about experience, projects, or skills — it answers only from the documents
it was given, and says so when it doesn't know.

**🔗 Live demo:** [add your Hugging Face Space URL here]

![screenshot](docs/screenshot.png) <!-- optional: add a screenshot -->

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
Gradio chat UI (streamed response)
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
| UI | Gradio `ChatInterface` (streaming) |
| Hosting | Hugging Face Spaces (free tier) |

The knowledge base is small, so an in-memory vector search is simpler and faster
to boot than a full vector database. Swapping in FAISS or Chroma is a one-function
change in `rag.py`.

## Run locally

```bash
git clone https://github.com/<your-handle>/portfolio-rag-bot
cd portfolio-rag-bot
python -m venv .venv && . .venv/Scripts/activate   # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

cp .env.example .env        # then paste your ANTHROPIC_API_KEY into .env
python app.py               # opens http://127.0.0.1:7860
```

## Make it yours

Replace the placeholder content in `knowledge/` with your own résumé:

- `about.md` — summary, location, contact
- `experience.md` — roles and education
- `projects.md` — projects with tech stacks
- `skills.md` — languages, tools, ways of working

No code changes needed — just edit the Markdown and restart.

## Deploy the live demo (Hugging Face Spaces)

1. Push this repo to GitHub.
2. Create a new **Space** on [huggingface.co/spaces](https://huggingface.co/spaces)
   → SDK: **Gradio**.
3. Link it to your GitHub repo (or push the code directly to the Space's git).
4. In the Space → **Settings → Secrets**, add `ANTHROPIC_API_KEY`.
5. The Space builds and serves your live demo URL. Put that URL in this README
   and in your CV.

## Project structure

```
portfolio-rag-bot/
├── app.py            # Gradio UI + Claude generation
├── rag.py            # chunking, embeddings, retrieval
├── knowledge/        # the knowledge base (edit these)
│   ├── about.md
│   ├── experience.md
│   ├── projects.md
│   └── skills.md
├── requirements.txt
├── .env.example
└── README.md
```
