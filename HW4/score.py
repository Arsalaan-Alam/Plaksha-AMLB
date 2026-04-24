"""
score.py — run this to validate your agent before submitting.

    python score.py --agent agent.py --pairs practice_pairs.csv

Your agent has 2 minutes total across all pairs.
Each pair scores max(0, 15 - steps) if the target is reached, else 0.

The instructor runs this same script against the secret test pairs.
"""

import argparse, csv, importlib.util, json, multiprocessing, os, sys, time

_OFFICIAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _OFFICIAL_DIR)

from wiki_tool import verify_path, get_links, _url_to_title

_SCORE_BASE   = 15    # score per pair = max(0, _SCORE_BASE - steps)
_TIME_LIMIT   = 120   # 2 minutes total for all pairs


# ── subprocess worker ─────────────────────────────────────────────────────────

def _worker(module_path, pairs, official_dir, deadline, q):
    import importlib.util, sys, os
    sys.path.insert(0, official_dir)
    agent_dir = os.path.dirname(os.path.abspath(module_path))
    if agent_dir != official_dir:
        sys.path.append(agent_dir)
    spec = importlib.util.spec_from_file_location("agent", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    q.put(mod.solve_all(pairs, deadline))


def run_agent(module_path, pairs, official_dir):
    """Run solve_all() in a subprocess with a hard 2-minute kill."""
    deadline = time.time() + _TIME_LIMIT
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_worker,
        args=(module_path, pairs, official_dir, deadline, q)
    )
    p.start()
    p.join(timeout=_TIME_LIMIT + 5)   # +5s grace for cleanup
    if p.is_alive():
        p.terminate()
        p.join()
        return None, "timeout"
    try:
        return q.get_nowait(), "ok"
    except Exception:
        return None, "error"


# ── validation ────────────────────────────────────────────────────────────────

def _failed(pair_id, reason):
    return {"pair_id": pair_id, "score": 0, "reason": reason,
            "steps": 0, "llm_calls": 0, "path": []}


def validate_one(result, pair):
    """Returns a scored record dict."""
    pair_id    = pair["pair_id"]
    target_url = pair["target"]

    if result is None:
        return _failed(pair_id, "no result returned")

    for key in ("pair_id", "path", "steps", "llm_calls", "success", "link_counts"):
        if key not in result:
            return _failed(pair_id, f"missing key: '{key}'")

    if result["pair_id"] != pair_id:
        return _failed(pair_id, f"pair_id mismatch: got '{result['pair_id']}'")

    path = result["path"]
    if not isinstance(path, list) or len(path) < 2:
        return _failed(pair_id, "path must be a list with ≥ 2 URLs")

    if _url_to_title(path[-1]).lower() != _url_to_title(target_url).lower():
        return _failed(pair_id,
            f"target not reached (ended at: {_url_to_title(path[-1])})")

    report = verify_path(path)
    if not report["valid"]:
        bad = next((e for e in report["edges"] if not e["valid"]), None)
        reason = (f"invalid edge: {bad['from']} → {bad['to']}"
                  if bad else "invalid path")
        return _failed(pair_id, reason)

    # spot-check: link_counts[0] must be within 20% of actual
    lc = result.get("link_counts", [])
    if lc:
        try:
            actual = len(get_links(path[0]))
            if actual and abs(actual - lc[0]) / actual > 0.20:
                return _failed(pair_id,
                    f"link_counts[0] wrong: got {lc[0]}, actual {actual}")
        except Exception:
            pass

    steps = result.get("steps", len(path) - 1)
    pts   = max(0, _SCORE_BASE - steps)
    return {"pair_id": pair_id,
            "start": _url_to_title(pair["start"]),
            "target": _url_to_title(target_url),
            "difficulty": pair.get("difficulty", ""),
            "score": pts, "max_score": _SCORE_BASE,
            "reason": f"valid  {steps} steps → {pts} pts",
            "steps": steps,
            "llm_calls": result.get("llm_calls", 0),
            "path": path}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent",  required=True)
    ap.add_argument("--pairs",  required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    with open(args.pairs, newline="") as f:
        pairs = list(csv.DictReader(f))

    module_path = os.path.abspath(args.agent)
    if not os.path.exists(module_path):
        sys.exit(f"ERROR: {module_path} not found")

    print(f"\n{'='*65}")
    print(f"  Agent  : {os.path.basename(module_path)}")
    print(f"  Pairs  : {args.pairs}  ({len(pairs)} pairs)")
    print(f"  Budget : {_TIME_LIMIT}s total  |  {_SCORE_BASE} pts max per pair")
    print(f"{'='*65}\n")

    t0 = time.time()
    raw_results, status = run_agent(module_path, pairs, _OFFICIAL_DIR)
    elapsed_agent = time.time() - t0

    if status == "timeout":
        print("  ✗ Agent process exceeded time limit — all pairs scored 0.\n")
        raw_results = []
    elif status == "error":
        print("  ✗ Agent process crashed — all pairs scored 0.\n")
        raw_results = []

    # index returned results by pair_id
    result_map = {r["pair_id"]: r for r in raw_results} if raw_results else {}

    records = []
    for pair in pairs:
        raw = result_map.get(pair["pair_id"])
        rec = validate_one(raw, pair)
        records.append(rec)
        tag = f"[{pair.get('difficulty','')}]"
        print(f"  {'✓' if rec['score'] > 0 else '✗'} {tag:<8} "
              f"{_url_to_title(pair['start']):<22} → "
              f"{_url_to_title(pair['target']):<24}  {rec['reason']}")

    total_score = sum(r["score"] for r in records)
    max_score   = _SCORE_BASE * len(pairs)
    reached     = sum(1 for r in records if r["score"] > 0)
    valid_steps = [r["steps"] for r in records if r["score"] > 0 and r["steps"]]
    avg_steps   = round(sum(valid_steps) / len(valid_steps), 1) if valid_steps else 0

    print(f"\n{'='*65}")
    print(f"  Score   : {total_score}/{max_score}  ({100*total_score/max_score:.1f}%)")
    print(f"  Reached : {reached}/{len(pairs)} pairs")
    print(f"  Avg steps (reached): {avg_steps}")
    print(f"  Agent time: {elapsed_agent:.1f}s")
    print(f"{'='*65}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"score": total_score, "max_score": max_score,
                       "pct": round(100 * total_score / max_score, 1),
                       "reached": reached, "total_pairs": len(pairs),
                       "avg_steps": avg_steps,
                       "agent_time_s": round(elapsed_agent, 1),
                       "pairs": records}, f, indent=2)
        print(f"  Written: {args.output}\n")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
