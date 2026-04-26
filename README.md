# Capsule — Transfer Learning for LLM Chats

When an LLM chat dies (context overflow, hallucination, or you just want a fresh
session), you lose all the work-in-progress state. Re-explaining wastes tokens
and time.

**Capsule** turns a chat transcript into a compact, structured handoff document
(a "Context Capsule") that you can paste into a new chat to bring it fully up
to speed.

It captures the things naive summaries miss:

- **Decision log** — *why* choices were made, not just what
- **Dead ends** — what was tried and rejected (so the new chat doesn't redo it)
- **Open threads** — what's in progress, what's next
- **User preferences** — tone, stack, "don't use X"

**No LLM. No API keys. No external services.** The default mode is a pure
classical-IR pipeline that picks the right sentences out of your chat and
slots them into a structured handoff.

## Install

```bash
# stdlib only — no dependencies needed for the default algorithmic mode
python3 -m capsule.cli build chat.md -o capsule.md
# (or `pip install -e .` to expose the `capsule` command on $PATH)
```

## Usage

### 1. Export your chat

Save the chat as plain text or markdown (most chat UIs have a copy/export
option). Call it `chat.md`. Turn headers like `**User:**` / `Assistant:` are
recognized but not required.

### 2. Build a capsule

```bash
capsule build chat.md -o capsule.md
```

This produces:

- `capsule.md` — the structured handoff
- `resume.txt` — a one-shot prompt to paste into the new chat

### 3. Start the new chat

Paste the contents of `resume.txt` (which references / inlines `capsule.md`)
as your first message in the new chat. Done.

### 4. Update across sessions

When *that* chat also fills up:

```bash
capsule update capsule.md new_chat.md -o capsule.md
```

The capsule evolves — that's the actual "transfer learning" loop.

## Modes

| Mode          | Flag          | What it does                                                  |
| ------------- | ------------- | ------------------------------------------------------------- |
| Algorithmic   | (default)     | TF-IDF + TextRank + MMR over the chat. No API calls.          |
| LLM           | `--llm`       | Calls OpenAI/Anthropic. Requires `OPENAI_API_KEY` or similar. |
| Skeleton      | `--skeleton`  | Empty schema + regex hints (a fallback for tiny chats).       |

## How the algorithmic mode works (the math)

The pipeline is fully deterministic and runs on Python stdlib only.

```
chat transcript
   │
   ▼
[1] Segment into sentences, tag each with turn role (user/assistant) and
    normalized position in [0, 1].   →  capsule/segment.py
   │
   ▼
[2] Tokenize → build TF-IDF vectors per sentence.
        tf(t,d)   = count(t,d) / total(d)
        idf(t)    = log((1+N) / (1+df(t))) + 1
        tfidf(t,d) = tf(t,d) * idf(t)
    →  capsule/features.py
   │
   ▼
[3] Build a sentence-similarity graph (cosine similarity, edge threshold).
    Run weighted PageRank (power iteration) to get a centrality score per
    sentence — this is the TextRank algorithm (Mihalcea & Tarau, 2004).
        PR(v) = (1−d)/N + d · Σ PR(u)·w(u,v)/W(u)
    →  capsule/textrank.py
   │
   ▼
[4] Apply positional/role boosts:
        + recency boost for sentences in the last 25% of the chat
        + small boost for user-authored sentences (their goals/preferences)
        − length penalty for fragments < 15 or > 400 chars
        + tag boost when a discourse rule fires
   │
   ▼
[5] Discourse classifier (rule-based, weighted regex bank) tags each
    sentence with roles like {goal, preference, decision, dead_end,
    todo, error, constraint, stack, snippet}.   →  capsule/router.py
   │
   ▼
[6] Route tagged sentences into capsule sections, then run **MMR**
    (Maximal Marginal Relevance, Carbonell & Goldstein 1998) to pick
    top-K sentences per section that are simultaneously high-scoring AND
    diverse:
        argmax_s [ λ · rel(s) − (1−λ) · max_{s' ∈ S} cos(s, s') ]
    →  capsule/mmr.py
   │
   ▼
[7] Render the 7-section markdown capsule.   →  capsule/render.py
```

**Why this is better than naive truncation/compression**

- TF-IDF surfaces sentences with information density (rare-but-meaningful tokens).
- TextRank captures *structural* importance — a sentence echoed by many others ranks high.
- MMR removes redundancy — the capsule won't waste space on near-duplicate bullets.
- The discourse router gives each sentence a *role*, so the schema slots are filled by relevant content (not just "important" content).

**Trustworthy by construction**

Every bullet in the capsule is a verbatim sentence from the source chat. The
pipeline cannot invent facts — if a section is sparse, it's because the chat
lacked that signal, not because the model hallucinated.

**Limits**

- No paraphrasing or compression *within* sentences.
- Discourse classifier is English-only and rule-based; nuanced phrasing may be missed.
- Files / stack mentions rely on conventional formatting (fenced paths, common framework names).

## The Capsule Schema

Every capsule has these sections — the design is the product:

1. **Project Meta** — goal, stack, constraints, user preferences
2. **Current State** — file tree, key symbols, what runs
3. **Decision Log** — choices + reasoning + alternatives rejected
4. **Dead Ends** — what failed and why (negative knowledge)
5. **Open Threads** — in-progress task, next step, TODOs
6. **Critical Snippets** — code the new LLM cannot infer
7. **Resume Prompt** — ready-to-paste opener for the new chat

## Why not just compress the chat?

- Lossless compression (gzip) doesn't help — the tokenizer doesn't decompress.
- Embeddings are lossy and need RAG infra.
- Token tricks (abbreviations) save ~15% but the LLM burns reasoning to decode.

Structured selectivity beats compression. A 500-token clean capsule outperforms
a 5000-token raw transcript.

## Project layout

```
capsule/
  segment.py    — turn-aware sentence segmentation
  features.py   — TF-IDF + cosine similarity (stdlib)
  textrank.py   — weighted PageRank via power iteration
  router.py     — discourse classifier + section routing
  mmr.py        — Maximal Marginal Relevance selection
  render.py     — wires the pipeline; renders the 7-section schema
  build.py      — mode dispatch (algorithmic / llm / skeleton)
  schema.py     — the markdown schema + LLM prompts
  extract.py    — simple regex extractor (used by skeleton mode)
  llm.py        — OpenAI/Anthropic client (used by --llm mode only)
  cli.py        — argparse entry point
examples/
  sample_chat.md
```
