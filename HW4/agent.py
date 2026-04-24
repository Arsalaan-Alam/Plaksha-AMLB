"""
agent.py — YOUR FILE TO SUBMIT.

Implement solve_all() below. The scorer calls it once with all pairs and a
shared deadline. You have 2 minutes total across ALL pairs.

    from agent import solve_all
    results = solve_all(pairs, deadline)

Arguments:
    pairs    : list of dicts, each with keys:
                 "pair_id"    — unique identifier (string)
                 "start"      — start Wikipedia URL
                 "target"     — target Wikipedia URL
                 "difficulty" — "easy" / "medium" / "hard"
    deadline : float — time.time() + 120. Check this and stop early
               if you are running out of time.

Return:
    list of result dicts, one per pair, in the same order as pairs.
    Each result dict must have:
        "pair_id"     : str   — must match the input pair_id
        "path"        : list  — Wikipedia URLs from start to target (inclusive)
        "steps"       : int   — number of hops (len(path) - 1)
        "llm_calls"   : int   — number of LLM API calls made
        "success"     : bool  — True if target was reached
        "link_counts" : list  — number of links get_links() returned at each
                                page visited (in order)

Scoring:
    Each pair scores max(0, 15 - steps) if the target is reached, else 0.
    Pairs not attempted score 0. Total = sum across all pairs.

API key:
    Put your Gemini key in ~/gemini_api_key.txt (one line, no quotes).
"""

import json, os, random, time, urllib.request
from wiki_tool import get_links

# ── API key — read from file, never hardcode ──────────────────────────────────
with open(os.path.expanduser("~/gemini_api_key.txt")) as f:
    _API_KEY = f.read().strip()

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key=" + _API_KEY
)


def _ask_gemini(prompt: str) -> str:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}]
    }).encode("utf-8")
    req = urllib.request.Request(
        _GEMINI_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _solve_one(pair: dict, deadline: float) -> dict:
    start_url  = pair["start"]
    target_url = pair["target"]
    target     = target_url.split("/wiki/")[-1].replace("_", " ")

    current     = start_url
    path        = [current]
    link_counts = []
    llm_calls   = 0

    for _ in range(15):
        if time.time() > deadline:
            break

        page_title = current.split("/wiki/")[-1].replace("_", " ")

        if page_title.lower() == target.lower():
            return {"pair_id": pair["pair_id"], "path": path,
                    "steps": len(path) - 1, "llm_calls": llm_calls,
                    "success": True, "link_counts": link_counts}

        links = get_links(current)
        link_counts.append(len(links))

        # ask Gemini to pick the next link
        numbered = "\n".join(f"{i+1}. {l['text']}" for i, l in enumerate(links))
        prompt   = (f"Navigate Wikipedia to reach: {target}\n"
                    f"Current page: {page_title}\n\n"
                    f"{numbered}\n\n"
                    f"Reply with ONLY the number of the best link.")
        try:
            choice = int(_ask_gemini(prompt).strip().split()[0])
            if not (1 <= choice <= len(links)):
                raise ValueError
        except Exception:
            choice = random.randint(1, len(links))

        llm_calls += 1
        current = links[choice - 1]["url"]
        path.append(current)

    return {"pair_id": pair["pair_id"], "path": path,
            "steps": len(path) - 1, "llm_calls": llm_calls,
            "success": False, "link_counts": link_counts}


# ── entry point ───────────────────────────────────────────────────────────────

def solve_all(pairs: list, deadline: float) -> list:
    """
    Solve all pairs within the shared 2-minute deadline.

    This barebones implementation attempts each pair in order with no
    time management or strategy. Improve it to score higher:
      - Check the deadline and skip pairs you won't finish
      - Add a visited set to avoid revisiting pages
      - Improve your prompting strategy
      - Decide which pairs to attempt first
      - Reuse cached links across pairs
    """
    results = []
    for pair in pairs:
        try:
            result = _solve_one(pair, deadline)
        except Exception:
            result = {"pair_id": pair["pair_id"], "path": [pair["start"]],
                      "steps": 0, "llm_calls": 0, "success": False,
                      "link_counts": []}
        results.append(result)
    return results
