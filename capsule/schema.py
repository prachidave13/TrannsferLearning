"""Capsule schema — the structure of a context handoff."""

# The empty skeleton. Order and headings are load-bearing: the LLM and the
# `update` mode both rely on them.
CAPSULE_SKELETON = """# Context Capsule

> A structured handoff of an LLM conversation. Paste into a new chat to resume.

## 1. Project Meta
- **Goal**: <one sentence>
- **Tech stack**: <languages, frameworks, key libs>
- **Constraints**: <perf, deploy target, deadlines, etc.>
- **User preferences**: <tone, verbosity, "don't use X">

## 2. Current State
### File tree (relevant files only)
```
<path>  — <one-line purpose>
```
### Key symbols / contracts
- `name(args) -> ret` — what it does, invariants
### What runs / what doesn't
- Working: <...>
- Broken: <...>

## 3. Decision Log
*Why* we chose what we chose. Format: decision — reasoning — rejected alternatives.
- **<decision>** — <why> — *rejected:* <alternatives + why>

## 4. Dead Ends (do not retry)
Negative knowledge — things tried that failed.
- **<approach>** — <how it failed> — <why it can't be fixed by retrying>

## 5. Open Threads
- **In progress**: <task> — last action: <what was just done>
- **Next step**: <planned next action>
- **TODOs / known bugs**:
  - [ ] <item>

## 6. Critical Snippets
Only code the new LLM cannot reasonably infer from the file tree.
```<lang>
<snippet>
```

## 7. Resume Prompt
A ready-to-paste opener for the new chat:

> I'm continuing work from a previous session. Read the Context Capsule above
> carefully — especially sections 3 (Decision Log) and 4 (Dead Ends) — before
> proposing anything. My next ask is: <fill in>.
"""


# System prompt for the LLM pass. Strict, schema-locked, anti-hallucination.
SYSTEM_PROMPT = """You are a "Context Capsule" generator. Your job: read an LLM
chat transcript and produce a structured handoff document so a fresh LLM
session can resume the work without re-asking questions.

CRITICAL RULES:
1. Output ONLY the capsule in the exact markdown structure given below. No
   preamble, no explanation, no closing remarks.
2. Use ONLY information present in the transcript. Do NOT invent files,
   functions, decisions, or TODOs. If a section has no content, write
   "_(none recorded)_" under it.
3. Prioritize signal that a new LLM CANNOT recover from the codebase alone:
   - WHY decisions were made (not just what)
   - Dead ends (what was tried and failed)
   - User preferences and constraints stated in passing
   - Open threads / what was just being worked on when the chat ended
4. Be terse. Bullets over prose. No filler.
5. For the Decision Log and Dead Ends, quote or paraphrase the actual
   reasoning from the transcript — do not generalize.
6. In "Critical Snippets", include code ONLY if it is non-obvious config,
   tricky logic, or an API contract. Do NOT dump large code blocks that
   exist in the user's files.
7. In section 7, write a concrete one-paragraph resume prompt that names the
   immediate next step.

OUTPUT FORMAT — follow this skeleton exactly (keep all 7 numbered sections,
in order, with the same headings):

""" + CAPSULE_SKELETON


UPDATE_SYSTEM_PROMPT = """You are updating an existing Context Capsule with
new information from a follow-up chat transcript.

CRITICAL RULES:
1. Output the FULL updated capsule in the same 7-section markdown format.
2. PRESERVE entries from the old capsule unless the new transcript explicitly
   supersedes or invalidates them. Especially preserve the Decision Log and
   Dead Ends — these accumulate across sessions.
3. When a decision is reversed, MOVE the old decision to Dead Ends with a
   note on why it was reversed, and add the new decision to the Decision Log.
4. Append new TODOs; mark completed ones as done (strike through with ~~).
5. Update "Open Threads" to reflect the END state of the new transcript.
6. Do not invent. Do not pad. Terse bullets.
7. Output ONLY the updated capsule. No preamble.
"""


# Self-summarize prompt — used in offline mode, to paste into the dying chat.
SELF_SUMMARIZE_PROMPT = """Before this chat ends, produce a "Context Capsule"
so I can resume in a fresh chat without re-explaining. Follow this exact
markdown structure. Be terse. Use only what we actually discussed. If a
section is empty, write "_(none recorded)_".

""" + CAPSULE_SKELETON
