# HW4 — Wikipedia Navigation Agent

You are building an AI agent that navigates Wikipedia by following hyperlinks — starting from one article and finding its way to a target article. Your agent uses an LLM to decide which link to follow at each step. The challenge is doing this intelligently within a strict time budget.

---

## The game

The scorer gives your agent a list of pairs and a 2-minute deadline for all of them:

```python
results = solve_all(pairs, deadline)
```

You do not know in advance how many pairs you will be evaluated on. Your agent must manage its own time.

**Scoring:** each pair scores `max(0, 15 − steps)` if the target is reached, else 0. A path completed in 1 step scores 14, in 14 steps scores 1, in 15 or more steps scores 0. Pairs where the target is not reached score 0. You will be evaluated on **3 test sets of different sizes** (roughly 10, 15, and 20 pairs) — your agent must work well regardless of how many pairs it receives. Your final agent score is the total across all three sets.

**Grading:** your agent score counts for **70%** of the homework grade, evaluated on a curve relative to the class — you are not penalized for the problem being hard, only for how you do relative to your peers. The remaining **30%** is a written reflection (see below).

---

## What to submit

1. **`alice_bob.py`** — your agent file, renamed with both team members' first names separated by an underscore (e.g. `alice_bob.py`)
2. **`alice_bob_writeup.md`** — a one-page markdown file, same naming convention, responding to the two questions below

---

## Written reflection (30%)

Answer both questions in about one page total. There are no right answers — we are looking for honest, specific reflection grounded in what you actually built and observed.

**Q1. Where did your agent break down, and what did that reveal about how LLMs reason?**

**Q2. What is one principle you take away from this that you would apply the next time you build anything with an LLM?**

---

## Setup

The scorer and all provided tools use Python stdlib only — no packages required to run `score.py`.

Your agent must use **Gemini 2.5 Flash** (`gemini-2.5-flash`). This is the model it will be evaluated with. Get an API key at [aistudio.google.com](https://aistudio.google.com), save it to `~/gemini_api_key.txt` (one line, no quotes), and read it from there in your code. Never hardcode the key.

**The most important thing:** your agent must be evaluable by `score.py`. Run this before submitting and confirm it produces a score:

```bash
python score.py --agent alice_bob.py --pairs practice_pairs.csv
```

`score.py` accepts any `.py` file via `--agent`. It loads your file, calls `solve_all(pairs, deadline)`, and scores the results. The only requirement is that the function exists, uses Gemini 2.5 Flash, and returns the right format in time.

---

## Rules

- Only use `get_links()` from `wiki_tool.py` to fetch Wikipedia links — no scraping, no direct API calls
- Do not modify `wiki_tool.py` or `score.py`
- Every link in your path is independently verified — fabricated paths score 0

---

## Return format

Each result dict in your list must have:

```python
{
    "pair_id"     : str,    # same as input
    "path"        : list,   # Wikipedia URLs, start → target inclusive
    "steps"       : int,    # len(path) - 1
    "llm_calls"   : int,
    "success"     : bool,
    "link_counts" : list,   # get_links() sizes at each page visited
}
```

Read `agent.py` for a working but naive starting point. It times out and scores 0 — fixing that is the assignment.
