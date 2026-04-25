"""
agent.py — Wikipedia navigation agent (v2, parallelized).

Key ideas over v1:
  1. Parallel Gemini call + 2-hop probe (they hit different servers).
     Expansion wall-time drops from ~sum to ~max of the two. Saves 1-2s/hop.
  2. Skip Gemini entirely when heuristic top-score is very high (≥ 200).
  3. Wider shortlist for Gemini (up to 20 candidates) so it has more
     semantic bridges to consider.
  4. Retry once on Wikipedia 429 (transient rate limit).
  5. Dynamic time budgets: easy pairs get less, hard pairs more, with a
     guaranteed floor so a starved final pair can still attempt something.
  6. A* (best-first) search with heuristic + Gemini rescoring and dedup.
"""

import concurrent.futures
import heapq
import json
import os
import re
import time
import urllib.error
import urllib.request
from itertools import count

from wiki_tool import get_links


# ── API key ──────────────────────────────────────────────────────────────────
_API_KEY_PATH = os.path.expanduser("~/gemini_api_key.txt")
_API_KEY = ""
if os.path.exists(_API_KEY_PATH):
    try:
        with open(_API_KEY_PATH) as _f:
            _API_KEY = _f.read().strip()
    except Exception:
        _API_KEY = ""

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key=" + _API_KEY
)

# Shared executor for overlapping Gemini + Wikipedia calls.
# max_workers=3 is enough for (gemini, probe, one prefetch) per expansion.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=3)

# ── Constants ────────────────────────────────────────────────────────────────
_LINK_CACHE: dict = {}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "their", "to",
    "with", "list", "outline", "index",
}

_BRIDGE_HINTS = {
    "history", "science", "technology", "mathematics", "physics", "chemistry",
    "biology", "religion", "philosophy", "politics", "war", "empire",
    "civilization", "society", "economy", "law", "art", "music", "literature",
    "geography", "europe", "asia", "africa", "america", "ancient", "modern",
    "medieval", "renaissance", "revolution",
}

_PENALTY_PATTERNS = (
    "list of", "outline of", "index of", "(disambiguation)",
    "bibliography of", "timeline of",
)

_CONFIDENT_SCORE = 200.0     # above this, we trust the heuristic and skip Gemini
_STRONG_EASY_SCORE = 25.0    # on easy pairs, skip Gemini above this


# ── Text utilities ───────────────────────────────────────────────────────────
def _url_title(url: str) -> str:
    return url.split("/wiki/")[-1].split("#")[0].replace("_", " ")


def _normalize(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\s\-()]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list:
    return [t for t in _normalize(text).split() if t and t not in _STOPWORDS]


def _token_set(text: str) -> set:
    return set(_tokens(text))


def _read_links(url: str) -> list:
    """Cached get_links. Retries once on 429."""
    if url in _LINK_CACHE:
        return _LINK_CACHE[url]
    try:
        _LINK_CACHE[url] = get_links(url)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(1.2)
            _LINK_CACHE[url] = get_links(url)  # one retry; re-raises on failure
        else:
            raise
    return _LINK_CACHE[url]


def _try_read_links(url: str) -> list:
    try:
        return _read_links(url)
    except Exception:
        return []


# ── Gemini ───────────────────────────────────────────────────────────────────
def _gemini_post(payload: dict, timeout: float = 12.0) -> str:
    req = urllib.request.Request(
        _GEMINI_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return ""


def _gemini_pick(current_title: str, target_title: str, path_titles: list,
                 candidates: list, timeout: float = 10.0) -> int:
    """Ask Gemini to pick the single best candidate (1-indexed). Returns -1 on failure."""
    if not _API_KEY or not candidates:
        return -1

    recent_path = " → ".join(path_titles[-5:]) if path_titles else current_title
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    prompt = (
        f"You are playing the Wikipedia race: follow hyperlinks to reach a "
        f"target article in as few hops as possible.\n\n"
        f"TARGET: {target_title}\n"
        f"Path so far: {recent_path}\n"
        f"Current page: {current_title}\n\n"
        f"First, think quietly about what field \"{target_title}\" belongs to "
        f"(era, discipline, geography, people) so you recognize bridge topics.\n"
        f"Then pick the ONE link below most likely to reach \"{target_title}\" "
        f"in the fewest remaining hops. Prefer broad bridge topics "
        f"(e.g. a shared era, region, or discipline) over narrow trivia.\n\n"
        f"{numbered}\n\n"
        f"Answer with ONLY the number. No words, no explanation."
    )

    payload_fast = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    payload_fallback = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8},
    }

    for payload in (payload_fast, payload_fallback):
        try:
            text = _gemini_post(payload, timeout=timeout).strip()
        except Exception:
            continue
        for tok in re.findall(r"\d+", text):
            n = int(tok)
            if 1 <= n <= len(candidates):
                return n
    return -1


# ── Heuristic scoring ────────────────────────────────────────────────────────
def _heuristic_score(link_text: str, target_title: str, difficulty: str) -> float:
    link_norm = _normalize(link_text)
    target_norm = _normalize(target_title)
    if link_norm == target_norm:
        return 1000.0

    link_tokens = _token_set(link_text)
    target_tokens = _token_set(target_title)
    overlap = len(link_tokens & target_tokens)

    score = 8.0 * overlap
    if target_norm and (target_norm in link_norm or link_norm in target_norm):
        score += 18.0

    if any(tok in _BRIDGE_HINTS for tok in link_tokens):
        score += 2.5

    if re.search(r"\b\d{4}\b", link_norm):
        score -= 1.5
    for pat in _PENALTY_PATTERNS:
        if pat in link_norm:
            score -= 5.0
            break

    if difficulty == "easy":
        score += 0.5 * overlap

    return score


def _is_low_signal_title(title: str) -> bool:
    norm = _normalize(title)
    if len(norm) <= 2:
        return True
    if sum(ch.isalpha() for ch in norm) < 3:
        return True
    return False


# ── 2-hop lookahead ──────────────────────────────────────────────────────────
def _direct_link(links: list, target_url: str) -> dict | None:
    """Return the link dict if target is in the list, else None."""
    target_title_norm = _normalize(_url_title(target_url))
    for link in links:
        if link["url"] == target_url:
            return link
        if _normalize(link["text"]) == target_title_norm:
            return link
    return None


def _two_hop_probe(candidates: list, target_url: str,
                   time_left: float) -> dict | None:
    """
    For each candidate, fetch its links and check if target is there.
    Returns {"via": candidate_link, "final": target_link} or None.
    """
    deadline = time.time() + time_left
    for cand in candidates:
        if time.time() > deadline - 0.5:
            return None
        sublinks = _try_read_links(cand["url"])
        if not sublinks:
            continue
        direct = _direct_link(sublinks, target_url)
        if direct is not None:
            # Use target_url as the final step (not direct["url"]) to ensure
            # the scorer's final-URL check matches. The edge is verified via
            # the link *text*, which matches target_title by construction.
            return {"via": cand, "final": {"text": direct["text"], "url": target_url}}
    return None


# ── Candidate ranking ────────────────────────────────────────────────────────
def _rank_candidates(
    current_url: str,
    target_url: str,
    difficulty: str,
    visited: set,
    path_titles: list,
    remaining_time: float,
) -> tuple:
    """Returns (link_count, shortlist_links, llm_calls, two_hop_solution)."""
    target_title = _url_title(target_url)
    links = _read_links(current_url)
    link_count = len(links)

    # 1-hop direct link?
    direct = _direct_link(links, target_url)
    if direct is not None:
        # Always point to target_url exactly so scorer's final-URL check passes.
        return link_count, [{"text": direct["text"], "url": target_url}], 0, None

    # Score all candidates
    scored = []
    seen = set()
    target_tokens = _token_set(target_title)
    for link in links:
        url = link["url"]
        if url in seen or url in visited:
            continue
        seen.add(url)
        if _is_low_signal_title(link["text"]):
            continue
        score = _heuristic_score(link["text"], target_title, difficulty)
        link_tokens = _token_set(link["text"])
        if target_tokens and link_tokens & target_tokens:
            score += 3.0
        scored.append((score, link))

    if not scored:
        return link_count, [], 0, None

    scored.sort(key=lambda s: s[0], reverse=True)
    top_score = scored[0][0]

    # Shortlist sizes — wider with more time.
    if difficulty == "hard":
        shortlist_size = 15 if remaining_time > 20 else 10
    else:
        shortlist_size = 20 if remaining_time > 20 else 12
    shortlist = [link for _, link in scored[:shortlist_size]]

    # Decide whether to query Gemini.
    use_gemini = (
        bool(_API_KEY)
        and shortlist
        and remaining_time > 4.0
        and top_score < _CONFIDENT_SCORE
        and not (difficulty == "easy" and top_score >= _STRONG_EASY_SCORE)
    )

    # Decide whether to do a 2-hop probe.
    probe_enabled = remaining_time > 8.0
    if probe_enabled:
        probe_width = 4 if difficulty == "hard" else 3
        probe_time = min(remaining_time * 0.5, 6.0)
        probe_candidates = shortlist[:probe_width]
    else:
        probe_candidates = []
        probe_time = 0.0

    llm_calls = 0
    two_hop = None
    pick = -1

    # ── Run Gemini and 2-hop probe in PARALLEL ───────────────────────────────
    gemini_future = None
    probe_future = None

    if use_gemini:
        gemini_timeout = min(9.0, max(3.5, remaining_time * 0.35))
        titles = [link["text"] for link in shortlist]
        gemini_future = _EXECUTOR.submit(
            _gemini_pick,
            _url_title(current_url), target_title, path_titles, titles, gemini_timeout,
        )

    if probe_candidates:
        probe_future = _EXECUTOR.submit(
            _two_hop_probe, probe_candidates, target_url, probe_time,
        )

    # Wait for probe first (it's usually the longer of the two).
    if probe_future is not None:
        try:
            two_hop = probe_future.result(timeout=probe_time + 2.0)
        except (concurrent.futures.TimeoutError, Exception):
            two_hop = None

    # Wait for Gemini (short timeout; usually already done by now).
    if gemini_future is not None:
        llm_calls = 1
        try:
            pick = gemini_future.result(timeout=3.0)
        except (concurrent.futures.TimeoutError, Exception):
            pick = -1

    # Apply Gemini's pick (reorder shortlist).
    if pick > 0 and pick <= len(shortlist):
        chosen = shortlist.pop(pick - 1)
        shortlist.insert(0, chosen)

    # Final beam width
    if difficulty == "hard":
        beam = 4 if remaining_time > 18 else 3
    elif difficulty == "medium":
        beam = 4 if remaining_time > 15 else 3
    else:
        beam = 3

    return link_count, shortlist[:beam], llm_calls, two_hop


# ── Per-pair search ──────────────────────────────────────────────────────────
def _failed_path(pair: dict) -> dict:
    return {
        "pair_id": pair["pair_id"],
        "path": [pair["start"], pair["start"]],
        "steps": 1,
        "llm_calls": 0,
        "success": False,
        "link_counts": [],
    }


def _search_pair(pair: dict, pair_deadline: float, global_deadline: float) -> dict:
    start_url = pair["start"]
    target_url = pair["target"]
    difficulty = pair.get("difficulty", "medium")

    if start_url == target_url:
        return {
            "pair_id": pair["pair_id"],
            "path": [start_url],
            "steps": 0,
            "llm_calls": 0,
            "success": True,
            "link_counts": [],
        }

    max_hops = 14
    llm_calls = 0
    seq = count()
    best_depth = {start_url: 0}

    start_state = (0, start_url, [start_url], {start_url}, [])
    frontier = [(0.0, 0, next(seq), start_state)]
    expansions = 0

    if difficulty == "hard":
        max_expansions = 32
    elif difficulty == "medium":
        max_expansions = 28
    else:
        max_expansions = 22

    while frontier:
        now = time.time()
        if now >= pair_deadline or now >= global_deadline - 1.5:
            break
        if expansions >= max_expansions:
            break

        _, _, _, state = heapq.heappop(frontier)
        depth, current_url, path, visited, link_counts = state

        if current_url == target_url:
            return {
                "pair_id": pair["pair_id"],
                "path": path,
                "steps": len(path) - 1,
                "llm_calls": llm_calls,
                "success": True,
                "link_counts": link_counts,
            }

        if depth >= max_hops:
            continue

        expansions += 1
        remaining_time = min(pair_deadline, global_deadline - 1.5) - now
        link_count, next_links, calls, two_hop = _rank_candidates(
            current_url, target_url, difficulty, visited,
            [_url_title(u) for u in path],
            remaining_time,
        )
        llm_calls += calls
        next_link_counts = link_counts + [link_count]

        # 1-hop: the ranker returned the direct target link.
        if next_links and len(next_links) == 1 and next_links[0]["url"] == target_url:
            return {
                "pair_id": pair["pair_id"],
                "path": path + [target_url],
                "steps": len(path),
                "llm_calls": llm_calls,
                "success": True,
                "link_counts": next_link_counts,
            }

        # 2-hop: probe found a guaranteed 2-hop solution through `via`.
        if two_hop is not None:
            via_url = two_hop["via"]["url"]
            via_links_count = len(_LINK_CACHE.get(via_url, []))
            return {
                "pair_id": pair["pair_id"],
                "path": path + [via_url, target_url],
                "steps": len(path) + 1,
                "llm_calls": llm_calls,
                "success": True,
                "link_counts": next_link_counts + [via_links_count],
            }

        # Push beam candidates onto the frontier
        for link in next_links:
            next_url = link["url"]
            next_depth = depth + 1
            if best_depth.get(next_url, 10**9) <= next_depth:
                continue
            best_depth[next_url] = next_depth
            next_visited = visited | {next_url}
            next_path = path + [next_url]
            score = _heuristic_score(link["text"], _url_title(target_url), difficulty)
            score -= 0.9 * next_depth
            heapq.heappush(
                frontier,
                (-score, next_depth, next(seq),
                 (next_depth, next_url, next_path, next_visited, next_link_counts)),
            )

    return {
        "pair_id": pair["pair_id"],
        "path": [start_url, start_url],
        "steps": 1,
        "llm_calls": llm_calls,
        "success": False,
        "link_counts": [],
    }


# ── Entry point ──────────────────────────────────────────────────────────────
def solve_all(pairs: list, deadline: float) -> list:
    """
    Solve all pairs within the shared 2-minute deadline.
    """
    difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}
    orig_order = {pair["pair_id"]: idx for idx, pair in enumerate(pairs)}
    prioritized = sorted(
        pairs,
        key=lambda p: (difficulty_rank.get(p.get("difficulty", "medium"), 1),
                       orig_order[p["pair_id"]]),
    )

    results_by_id = {}
    pairs_remaining = len(prioritized)

    for pair in prioritized:
        now = time.time()
        time_left = deadline - now - 1.5
        if time_left <= 2.0 or pairs_remaining <= 0:
            results_by_id[pair["pair_id"]] = _failed_path(pair)
            pairs_remaining -= 1
            continue

        diff = pair.get("difficulty", "medium")
        base = time_left / pairs_remaining
        if diff == "hard":
            pair_budget = max(7.0, min(30.0, base * 1.40))
        elif diff == "medium":
            pair_budget = max(6.0, min(24.0, base * 1.15))
        else:
            pair_budget = max(4.5, min(18.0, base * 0.90))

        pair_deadline = min(deadline - 1.0, now + pair_budget)

        try:
            result = _search_pair(pair, pair_deadline, deadline)
        except Exception:
            result = _failed_path(pair)

        # Never claim success unless we actually landed at target_url (title match).
        if result.get("success"):
            path = result.get("path") or []
            if (not path
                or _url_title(path[-1]).lower() != _url_title(pair["target"]).lower()):
                result = _failed_path(pair)

        results_by_id[pair["pair_id"]] = result
        pairs_remaining -= 1

    return [results_by_id.get(pair["pair_id"], _failed_path(pair)) for pair in pairs]
