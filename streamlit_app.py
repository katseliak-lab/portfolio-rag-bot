"""RAG support bot (Streamlit UI): answers questions about the candidate strictly
from the knowledge base, with an Anthropic Claude model doing the generation.

Run locally:  streamlit run streamlit_app.py
On Streamlit Community Cloud: this file is the entry point; set ANTHROPIC_API_KEY
in the app's Secrets.
"""

from __future__ import annotations

import os

import anthropic
import streamlit as st

from rag import KnowledgeBase

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

EXAMPLES = [
    "What has Oleh worked on recently?",
    "Which tools and languages does he know?",
    "Розкажи про проєкти з автоматизації",
    "Does he have experience with generative video or voice AI?",
]

st.set_page_config(page_title="Ask me about Oleh Katseliak", page_icon="💬")


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_kb() -> KnowledgeBase:
    """Embed the knowledge base once and reuse it across reruns."""
    return KnowledgeBase(knowledge_dir="knowledge")


def get_api_key() -> str | None:
    """Read the key from Streamlit Secrets (cloud) or the environment (local)."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass  # no secrets.toml present locally — fall back to the environment
    return os.environ.get("ANTHROPIC_API_KEY")


kb = load_kb()
api_key = get_api_key()

st.title("💬 Ask me about Oleh Katseliak")
st.caption(
    "A retrieval-augmented (RAG) assistant grounded in a real knowledge base, "
    "powered by Anthropic Claude. Answers in your language (UK / EN)."
)

if not api_key:
    st.error(
        "ANTHROPIC_API_KEY is not set. Add it in Streamlit → Settings → Secrets, "
        "or in a local .streamlit/secrets.toml file."
    )
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show example questions only before the conversation starts.
pending: str | None = None
if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            pending = example

# Render the conversation so far.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


def stream_answer(user_msg: str, history: list[dict]):
    """Retrieve context and stream Claude's answer token by token."""
    context = kb.build_context(user_msg)
    system = SYSTEM_PROMPT + "\n\nCONTEXT:\n" + (context or "(no relevant context found)")
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": user_msg})
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


prompt = st.chat_input("Ask about experience, projects, skills…") or pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(prompt, st.session_state.messages[:-1]))
        except anthropic.APIStatusError as e:
            answer = f"⚠️ API error ({e.status_code}): {e.message}"
            st.markdown(answer)
        except anthropic.APIConnectionError:
            answer = "⚠️ Could not reach the Anthropic API. Please try again."
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
