"""RAG support bot: answers questions about the candidate strictly from the
knowledge base, with an Anthropic Claude model doing the generation.

Run locally:  python app.py       (opens http://127.0.0.1:7860)
On HF Spaces: this file is the entry point; set ANTHROPIC_API_KEY as a Secret.
"""

from __future__ import annotations

import os

import anthropic
import gradio as gr

from rag import KnowledgeBase

# Load .env for local development (no-op on Spaces, where the Secret is injected).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL = "claude-haiku-4-5"  # fast + cheap; swap to "claude-opus-4-8" for max quality
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a friendly assistant on a candidate's portfolio site. \
Recruiters and hiring managers ask you questions about the candidate; you answer \
on the candidate's behalf.

Rules:
- Answer ONLY using the information in the CONTEXT block below.
- If the answer is not in the context, say you don't have that detail and suggest \
what you can talk about (experience, projects, skills). Never invent facts, \
employers, dates, or numbers.
- Be concise, warm, and professional. A few sentences is usually enough.
- Reply in the same language the user writes in (Ukrainian or English).
- When useful, mention which project or role the information comes from.
"""

# Build the knowledge base once at startup.
kb = KnowledgeBase(knowledge_dir="knowledge")
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def respond(message: str, history: list[dict]):
    """Gradio ChatInterface handler. Streams Claude's answer token by token."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        yield (
            "⚠️ ANTHROPIC_API_KEY is not set. Add it to a local .env file, or in "
            "Hugging Face Space → Settings → Secrets."
        )
        return

    context = kb.build_context(message)
    system = SYSTEM_PROMPT + "\n\nCONTEXT:\n" + (context or "(no relevant context found)")

    # Convert Gradio's message history into Anthropic's format.
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    messages.append({"role": "user", "content": message})

    partial = ""
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                partial += text
                yield partial
    except anthropic.APIStatusError as e:
        yield f"⚠️ API error ({e.status_code}): {e.message}"
    except anthropic.APIConnectionError:
        yield "⚠️ Could not reach the Anthropic API. Check your connection and try again."


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="💬 Ask me about [Your Name]",
    description=(
        "A retrieval-augmented (RAG) assistant that answers questions about the "
        "candidate's experience, projects, and skills — grounded in a real "
        "knowledge base, powered by Anthropic Claude."
    ),
    examples=[
        "What has this person worked on recently?",
        "Which programming languages and tools do they know?",
        "Розкажи про проєкти з автоматизації",
        "Do they have experience with LLMs or RAG?",
    ],
)

if __name__ == "__main__":
    demo.launch()
