"""
Build the LightGBM learning-to-rank dataset.

One *query* = one (session, turn). For every query we

  1. cut the conversation to what is visible at that turn (training/turns.py),
  2. plan it (LLM planner, cached on disk so re-runs are identical),
  3. run the selected retrievers over the full catalog (candidate_k each),
  4. build features for the union of candidates (recommender_v2/features.py),
  5. label each candidate 1 if it is that turn's ground-truth track, else 0.

Nothing is ever injected into the pool: if retrieval misses the gold track the
query simply has no positive row (and is reported in the recall diagnostics).

Outputs (for --output training_data_train.csv):

    training_data_train.csv           rows: META_COLUMNS + feature columns
    training_data_train.csv.queries.csv   one line per query (recall diagnostics)
    training_data_train.csv.planner.jsonl planner cache (query_id -> QueryState)
    training_data_train.csv.done          finished query ids (for --resume)
    training_data_train.csv.meta.json     feature schema, config, summary stats

Example:

    python -m recommender_v2.training.build_dataset \
        --split train --num-sessions 200 --output training_data_train.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from recommender_v2.features import FeatureBuilder, KNOWN_RETRIEVERS
from recommender_v2.planner import QueryPlanner, QueryState
from recommender_v2.ranker import RRFRanker
from recommender_v2.training.turns import get_turn_context


DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"

META_COLUMNS = [
    "query_id",
    "session_id",
    "user_id",
    "turn_number",
    "track_id",
    "label",
    "query_has_positive",
]

DEFAULT_MODELS = [
    "bm25",
    "bge",
    "cf",
    "item_cf",
    "item_clap",
    "item_metadata",
    "artist",
    "album",
]

ALL_MODELS = DEFAULT_MODELS + ["clap"]


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------

def build_retrievers(model_names, device="cpu", text_fields=None):

    from recommender_v2 import config as C

    factories = {
        "bm25": lambda: C.build_bm25(text_fields=text_fields),
        "bge": lambda: C.build_bge(device=device, text_fields=text_fields),
        "cf": lambda: C.build_cf_bpr(),
        "clap": lambda: C.build_clap(device=device),
        "item_cf": lambda: C.build_item_cf(),
        "item_clap": lambda: C.build_item_clap(),
        "item_metadata": lambda: C.build_item_metadata(),
        "artist": lambda: C.build_artist_lookup(),
        "album": lambda: C.build_album_lookup(),
    }

    retrievers = []

    for name in model_names:       # keep the order the user gave
        print(f"Loading {name}...")
        r = factories[name]()
        if r.name not in KNOWN_RETRIEVERS:
            raise ValueError(
                f"Retriever '{r.name}' is not in features.KNOWN_RETRIEVERS."
            )
        retrievers.append(r)

    return retrievers


# ---------------------------------------------------------------------------
# Planner cache
# ---------------------------------------------------------------------------

def load_planner_cache(path: Path, tag: str) -> Dict[str, dict]:
    """
    Only plans produced with the same planner settings (`tag`) are reused, so
    switching model or prompt layout can never silently mix plans in one dataset.
    """
    cache = {}
    skipped = 0
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("planner") == tag:
                    cache[rec["query_id"]] = rec
                else:
                    skipped += 1
    if skipped:
        print(f"Planner cache: ignoring {skipped} plans made with other settings.")
    return cache


def fallback_state(query: str) -> QueryState:
    """Used only when the LLM planner fails for a query (flagged in output)."""
    return QueryState(
        raw_query=query,
        bm25_query=query,
        semantic_query=query,
        planner_confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):

    p = argparse.ArgumentParser()

    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--allow-test", action="store_true",
                   help="Required to build a dataset from the held-out test split.")
    p.add_argument("--start-session", type=int, default=0)
    p.add_argument("--num-sessions", type=int, default=100)
    p.add_argument("--start-turn", type=int, default=1)
    p.add_argument("--end-turn", type=int, default=8)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    p.add_argument("--candidate-k", type=int, default=100,
                   help="Candidates requested from EACH retriever.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--planner-model", default="qwen3:8b")
    p.add_argument("--output", default="training_data_train.csv")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--planner-workers", type=int, default=1,
                   help="Planner requests in flight at once (threads). Use with "
                        "OLLAMA_NUM_PARALLEL=<same number> on the Ollama server. "
                        "Plans do not depend on each other, so the dataset is identical.")
    p.add_argument("--text-fields", nargs="+", default=None,
                   help="Catalog fields indexed by BM25/BGE (default: track_name "
                        "artist_name album_name release_date). Example: add tag_list.")
    p.add_argument("--no-metadata", action="store_true",
                   help="Skip catalog metadata (artist/album/genre/year features become NaN).")
    p.add_argument("--max-fallback-rate", type=float, default=0.05,
                   help="Abort if more than this fraction of queries needed the planner fallback.")

    args = p.parse_args(argv)

    if args.split == "test" and not args.allow_test:
        sys.exit(
            "Refusing to build training data from the test split. "
            "Pass --allow-test only for final evaluation datasets."
        )

    out = Path(args.output)
    done_path = Path(str(out) + ".done")
    queries_path = Path(str(out) + ".queries.csv")
    cache_path = Path(str(out) + ".planner.jsonl")
    meta_path = Path(str(out) + ".meta.json")

    if out.exists() and not (args.resume or args.overwrite):
        sys.exit(f"{out} exists. Use --resume to continue or --overwrite to start over.")

    if args.overwrite:
        for f in (out, done_path, queries_path, meta_path):
            if f.exists():
                f.unlink()
        # The planner cache is kept on purpose: it makes re-runs deterministic
        # and saves LLM time. Delete it by hand to re-plan everything.

    # ---------------- data ----------------
    from datasets import load_dataset

    print("Loading challenge dataset...")
    sessions_all = load_dataset(DATASET_NAME)[args.split]

    end = min(args.start_session + args.num_sessions, len(sessions_all))
    sessions = sessions_all.select(range(args.start_session, end))

    print(f"Split={args.split}  sessions {args.start_session}..{end - 1} "
          f"({len(sessions)})  turns {args.start_turn}..{args.end_turn}")

    # ---------------- components ----------------
    track_metadata = None
    if not args.no_metadata:
        from recommender_v2.training.metadata import load_track_metadata
        track_metadata = load_track_metadata()

    feature_builder = FeatureBuilder(track_metadata=track_metadata)
    feature_names = feature_builder.feature_names()

    from recommender_v2.planner.ollama import OllamaQwenPlanner
    from recommender_v2 import RecommenderV2

    planner = QueryPlanner(llm=OllamaQwenPlanner(args.planner_model), mode="llm")

    recommender = RecommenderV2(
        retrievers=build_retrievers(args.models, device=args.device, text_fields=args.text_fields),
        planner=planner,
        candidate_k=args.candidate_k,
        final_k=20,
        ranker=RRFRanker(),            # unused here; avoids loading a stale model
        feature_builder=feature_builder,
    )

    # Identifies the planner configuration that produced a cached plan.
    planner_tag = f"{args.planner_model}|query=latest_user|v2"
    planner_cache = load_planner_cache(cache_path, planner_tag)

    done = set()
    if args.resume and done_path.exists():
        done = {l.strip() for l in done_path.read_text().splitlines() if l.strip()}
        print(f"Resuming: {len(done)} queries already finished.")

    fieldnames = META_COLUMNS + feature_names
    retriever_names = [r.name for r in recommender.retrievers]
    query_fields = (
        ["query_id", "session_id", "turn_number", "pool_size", "gold_in_pool",
         "gold_is_anchor", "planner_fallback", "no_candidates",
         "planner_sec", "retrieve_sec", "features_sec"]
        + [f"gold_rank_{n}" for n in retriever_names]
        + [f"sec_{n}" for n in retriever_names]
    )

    new_file = not out.exists() or out.stat().st_size == 0
    new_queries_file = not queries_path.exists() or queries_path.stat().st_size == 0

    errors: List[str] = []
    n_queries = n_fallback = 0
    t0 = time.perf_counter()

    with open(out, "a", newline="", encoding="utf-8") as f_out, \
         open(queries_path, "a", newline="", encoding="utf-8") as f_q, \
         open(done_path, "a", encoding="utf-8") as f_done, \
         open(cache_path, "a", encoding="utf-8") as f_cache:

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        qwriter = csv.DictWriter(f_q, fieldnames=query_fields)

        if new_file:
            writer.writeheader()
        if new_queries_file:
            qwriter.writeheader()

        for s_idx, session in enumerate(sessions, start=args.start_session):

            session_id = session["session_id"]
            user_id = session["user_id"]

            # Plan all uncached turns of this session concurrently (the LLM is
            # the bottleneck; retrieval/features below stay sequential).
            prefetched = {}
            if args.planner_workers > 1:
                todo = []
                for turn in range(args.start_turn, args.end_turn + 1):
                    qid = f"{session_id}:{turn}"
                    if qid not in done and qid not in planner_cache:
                        todo.append((qid, turn))

                def _plan_one(item):
                    qid, turn = item
                    t_plan = time.perf_counter()
                    try:
                        c = get_turn_context(session["conversations"], turn)
                        st = recommender.plan(
                            query=c.planner_query,
                            conversation_history=c.history,
                            anchor_track_ids=c.anchor_track_ids,
                        )
                        return qid, st, None, time.perf_counter() - t_plan
                    except Exception as e:
                        return qid, None, e, time.perf_counter() - t_plan

                if todo:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=args.planner_workers) as pool:
                        for qid, st, err, sec in pool.map(_plan_one, todo):
                            prefetched[qid] = (st, err, sec)

            for turn in range(args.start_turn, args.end_turn + 1):

                query_id = f"{session_id}:{turn}"

                if query_id in done:
                    continue

                try:
                    ctx = get_turn_context(session["conversations"], turn)

                    # ---- plan (cached) ----
                    used_fallback = False
                    planner_sec = 0.0           # 0 when served from the cache
                    cached = planner_cache.get(query_id)

                    if cached is not None:
                        state = QueryState(**cached["state"])
                        used_fallback = bool(cached.get("fallback", False))
                        state.anchor_track_ids = list(ctx.anchor_track_ids)
                    else:
                        t_plan = time.perf_counter()
                        try:
                            if query_id in prefetched:
                                state, err, planner_sec = prefetched.pop(query_id)
                                if err is not None:
                                    raise err
                            else:
                                state = recommender.plan(
                                    query=ctx.planner_query,
                                    conversation_history=ctx.history,
                                    anchor_track_ids=ctx.anchor_track_ids,
                                )
                                planner_sec = time.perf_counter() - t_plan
                        except Exception as e:      # planner failed -> flagged fallback
                            errors.append(f"{query_id} planner: {e!r}")
                            state = fallback_state(ctx.query)
                            state.anchor_track_ids = list(ctx.anchor_track_ids)
                            used_fallback = True
                            planner_sec = time.perf_counter() - t_plan

                        # Failed plans are not cached, so a re-run retries them.
                        if not used_fallback:
                            rec = {"query_id": query_id, "planner": planner_tag,
                                   "state": state.to_dict(), "fallback": False}
                            f_cache.write(json.dumps(rec) + "\n")
                            f_cache.flush()
                            planner_cache[query_id] = rec

                    # ---- retrieve ----
                    timings = {}
                    t_ret = time.perf_counter()
                    state, retrieval_results = recommender.retrieve_for_ranking(
                        query=ctx.planner_query,
                        user_id=user_id,
                        conversation_history=ctx.history,
                        anchor_track_ids=ctx.anchor_track_ids,
                        state=state,
                        timings=timings,
                    )
                    retrieve_sec = time.perf_counter() - t_ret

                    # ---- features (same call the inference path makes) ----
                    t_feat = time.perf_counter()
                    features = recommender.build_features(
                        retrieval_results=retrieval_results,
                        user_id=user_id,
                        anchor_track_ids=ctx.anchor_track_ids,
                        query_state=state,
                        turn_index=turn,
                    )

                    features_sec = time.perf_counter() - t_feat

                    gold = ctx.ground_truth
                    pool_ids = {r["track_id"] for r in features}
                    has_positive = int(gold in pool_ids)

                    for r in features:
                        row = dict(r)
                        row["query_id"] = query_id
                        row["session_id"] = session_id
                        row["user_id"] = user_id
                        row["turn_number"] = turn
                        row["label"] = int(r["track_id"] == gold)
                        row["query_has_positive"] = has_positive
                        writer.writerow(row)

                    gold_ranks = {}
                    for name, results in retrieval_results.items():
                        gold_ranks[f"gold_rank_{name}"] = next(
                            (res.rank for res in results if res.track_id == gold), ""
                        )

                    qwriter.writerow({
                        "query_id": query_id,
                        "session_id": session_id,
                        "turn_number": turn,
                        "pool_size": len(features),
                        "gold_in_pool": has_positive,
                        "gold_is_anchor": int(gold in set(ctx.anchor_track_ids)),
                        "planner_fallback": int(used_fallback),
                        "no_candidates": int(len(features) == 0),
                        "planner_sec": round(planner_sec, 2),
                        "retrieve_sec": round(retrieve_sec, 2),
                        "features_sec": round(features_sec, 2),
                        **gold_ranks,
                        **{f"sec_{n}": round(v, 3) for n, v in timings.items()},
                    })

                    f_out.flush()
                    f_q.flush()
                    f_done.write(query_id + "\n")
                    f_done.flush()

                    n_queries += 1
                    n_fallback += int(used_fallback)

                except Exception as e:
                    errors.append(f"{query_id}: {e!r}")
                    print(f"  ! skipped {query_id}: {e!r}")

            elapsed = time.perf_counter() - t0
            print(f"[session {s_idx}] queries this run: {n_queries}  "
                  f"errors: {len(errors)}  elapsed: {elapsed / 60:.1f} min")

            if n_queries >= 50 and n_fallback / n_queries > args.max_fallback_rate:
                sys.exit(
                    f"Planner fallback rate {n_fallback / n_queries:.1%} exceeds "
                    f"{args.max_fallback_rate:.0%}. Check Ollama / the planner prompt."
                )

    if errors:
        err_path = Path(str(out) + ".errors.txt")
        err_path.write_text("\n".join(errors))
        print(f"{len(errors)} issues logged to {err_path}")

    # ---------------- summary ----------------
    import pandas as pd

    q = pd.read_csv(queries_path)
    if q.empty:
        sys.exit("No queries were produced.")

    summary = {
        "queries": int(len(q)),
        "sessions": int(q["session_id"].nunique()),
        "union_recall": float(q["gold_in_pool"].mean()),
        "mean_pool_size": float(q["pool_size"].mean()),
        "queries_without_candidates": int(q["no_candidates"].sum()),
        "gold_is_anchor_rate": float(q["gold_is_anchor"].mean()),
        "planner_fallback_rate": float(q["planner_fallback"].mean()),
        "per_retriever_recall": {
            n: float(q[f"gold_rank_{n}"].notna().mean()) for n in retriever_names
        },
        "recall_by_turn": {
            int(t): float(v)
            for t, v in q.groupby("turn_number")["gold_in_pool"].mean().items()
        },
    }

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "start_session": args.start_session,
        "num_sessions_requested": args.num_sessions,
        "turns": [args.start_turn, args.end_turn],
        "models": args.models,
        "retriever_names": retriever_names,
        "candidate_k": args.candidate_k,
        "planner_model": args.planner_model,
        "planner_tag": planner_tag,
        "metadata_features": not args.no_metadata,
        "meta_columns": META_COLUMNS,
        "feature_names": feature_names,
        "summary": summary,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 60)
    print("Dataset summary")
    print("=" * 60)
    print(f"  Output file          : {out}")
    print(f"  Queries (groups)     : {summary['queries']}")
    print(f"  Sessions             : {summary['sessions']}")
    print(f"  Mean pool size       : {summary['mean_pool_size']:.1f}")
    print(f"  Union recall         : {summary['union_recall']:.4f}"
          f"   <- upper bound for any reranker's hit rate")
    for n, v in summary["per_retriever_recall"].items():
        print(f"    {n:<16} recall {v:.4f}")
    print(f"  Gold is an anchor    : {summary['gold_is_anchor_rate']:.4f}")
    print(f"  Planner fallback     : {summary['planner_fallback_rate']:.4f}")
    print(f"  Features             : {len(feature_names)}")


if __name__ == "__main__":
    main()