"""Lightweight semantic retrieval over the knowledge base.

Design choice: the knowledge base for a personal support bot is small
(a few dozen chunks), so an in-memory NumPy cosine search is faster to load,
easier to read, and has zero external service dependencies than a full vector
database. Swapping in FAISS or Chroma later is a one-function change in `search`.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

# Small (~80 MB), fast, CPU-friendly. Good default for free hosting tiers.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Target characters per chunk. Small enough to keep retrieved context tight,
# large enough that a chunk carries a complete thought.
CHUNK_TARGET_CHARS = 600


@dataclass
class Chunk:
    text: str
    source: str  # e.g. "projects.md > Djinni Job Search"


def _split_markdown(md: str, filename: str) -> list[Chunk]:
    """Split one markdown file into chunks, tagging each with its heading."""
    # Drop HTML comments (editing instructions) so they aren't embedded/retrieved.
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)

    chunks: list[Chunk] = []
    current_heading = filename
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(Chunk(text=text, source=f"{filename} > {current_heading}"))
        buffer.clear()

    for line in md.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading:
            flush()
            current_heading = heading.group(1).strip()
            continue

        buffer.append(line)
        # Flush on blank-line boundaries once the buffer is big enough,
        # so chunks break on paragraph edges rather than mid-sentence.
        if not line.strip() and sum(len(x) for x in buffer) >= CHUNK_TARGET_CHARS:
            flush()

    flush()
    return chunks


def load_chunks(knowledge_dir: str) -> list[Chunk]:
    """Read every .md file in the knowledge directory into chunks."""
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(knowledge_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            md = f.read()
        chunks.extend(_split_markdown(md, os.path.basename(path)))
    if not chunks:
        raise RuntimeError(f"No markdown chunks found in {knowledge_dir!r}")
    return chunks


class KnowledgeBase:
    """Embeds the knowledge base once and answers similarity queries."""

    def __init__(self, knowledge_dir: str = "knowledge"):
        self.chunks = load_chunks(knowledge_dir)
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        # normalize_embeddings=True => cosine similarity is a plain dot product.
        self.embeddings = self.model.encode(
            [c.text for c in self.chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.embeddings @ q  # cosine similarity, shape (n_chunks,)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def build_context(self, query: str, top_k: int = 4, min_score: float = 0.15) -> str:
        """Return the retrieved chunks formatted for the LLM prompt."""
        hits = [(c, s) for c, s in self.search(query, top_k) if s >= min_score]
        if not hits:
            return ""
        blocks = [f"[Source: {c.source}]\n{c.text}" for c, _ in hits]
        return "\n\n---\n\n".join(blocks)
