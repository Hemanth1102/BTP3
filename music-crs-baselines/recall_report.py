"""Measure candidate recall on the dev diagnostic slice without calling an LLM."""

import argparse
import csv
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

import torch

from recommender_v2 import RecommenderV2
from recommender_v2.features import FEATURE_GROUPS
from recommender_v2.planner import QueryPlanner, QueryState
from recommender_v2.ranker import RRFRanker
from recommender_v2.splits import DEV_SPLIT, DIAGNOSTIC, FINAL_HOLDOUT, TURNS
from recommender_v2.training.build_dataset import (
    ALL_MODELS,
    DATASET_NAME,
    DEFAULT_MODELS,
    build_retrievers,
    load_planner_cache,
)
from recommender_v2.training.turns import get_turn_context


def measure_recall(sessions, recommender, planner_cache, profile=None):
    names = [retriever.name for retriever in recommender.retrievers]
    hits, lengths, only_hits = Counter(), Counter(), Counter()
    turn_hits, turn_totals = Counter(), Counter()
    union_hits = pool_size = missing_plans = 0

    for session in sessions:
        session_id = session["session_id"]
        for retriever in recommender.retrievers:
            if hasattr(retriever, "begin_session"):
                retriever.begin_session(session_id, held_out=False)

        for turn in TURNS:
            ctx = get_turn_context(session["conversations"], turn)
            cached = planner_cache.get(f"{session_id}:{turn}")
            if cached is None or cached.get("fallback"):
                missing_plans += 1
                state = QueryState()
            else:
                state = QueryState.from_dict(cached["state"])
            state.raw_query = ctx.latest_user_message
            state.anchor_track_ids = list(ctx.anchor_track_ids)

            _, results = recommender.retrieve_for_ranking(
                ctx.latest_user_message,
                user_id=session.get("user_id"),
                conversation_history=ctx.history,
                anchor_track_ids=ctx.anchor_track_ids,
                state=state,
            )
            if profile is not None:
                profile.collect(recommender, results, state, session.get("user_id"), turn)
            union, found = set(), []
            for name, candidates in results.items():
                ids = {candidate.track_id for candidate in candidates}
                lengths[name] += len(candidates)
                union.update(ids)
                if ctx.ground_truth in ids:
                    found.append(name)
            hits.update(found)
            if len(found) == 1:
                only_hits[found[0]] += 1
            union_hits += bool(found)
            pool_size += len(union)
            turn_totals[turn] += 1
            turn_hits[turn] += bool(found)

    total = sum(turn_totals.values())
    if not total:
        raise ValueError("No queries in the selected session range.")
    rows = [
        {"source": name, "queries": total, "recall": hits[name] / total,
         "mean_list_length": lengths[name] / total, "marginal_recall": only_hits[name] / total}
        for name in names
    ]
    rows.append({"source": "union", "queries": total, "recall": union_hits / total,
                 "mean_list_length": pool_size / total, "marginal_recall": None})
    by_turn = {turn: turn_hits[turn] / turn_totals[turn] for turn in TURNS}
    return rows, by_turn, missing_plans


class FeatureProfile:
    """Measure feature time and coverage on the first queries of a report."""

    def __init__(self, limit):
        self.limit = limit
        self.times = []
        self.pool_sizes = []
        self.groups = FEATURE_GROUPS
        self.missing = Counter()

    def collect(self, recommender, results, state, user_id, turn):
        if len(self.times) >= self.limit:
            return
        started = time.perf_counter()
        rows = recommender.build_features(results, user_id=user_id, query_state=state,
                                          anchor_track_ids=state.anchor_track_ids, turn_index=turn)
        self.times.append(time.perf_counter() - started)
        self.pool_sizes.append(len(rows))
        for features in self.groups.values():
            for name in features:
                self.missing[name] += sum(row[name] != row[name] for row in rows)

    def write(self, directory, tag):
        total = sum(self.pool_sizes)
        path = directory / f"recall_{tag}_features"
        header = (f"{len(self.times)} queries, {total} candidate rows, mean pool {mean(self.pool_sizes):.1f}. "
                  f"Feature time {median(self.times):.3f} s median and {max(self.times):.3f} s maximum on CPU.")
        lines = ["# Feature coverage", "", header, "", "| Group | Feature | NaN rate |", "| --- | --- | ---: |"]
        with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["group", "feature", "candidate_rows", "nan_rows", "nan_rate"])
            for group, features in self.groups.items():
                for name in features:
                    missing = self.missing[name]
                    rate = missing / total if total else float("nan")
                    writer.writerow([group, name, total, missing, rate])
                    lines.append(f"| {group} | {name} | {rate:.2%} |")
        path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(header)


def write_report(directory, tag, topk, session_range, rows, by_turn, missing_plans):
    directory.mkdir(parents=True, exist_ok=True)
    total = rows[-1]["queries"]
    path = directory / f"recall_{tag}"
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["scope", "source", "turn", "queries", "missing_plans", "topk",
                         "recall", "mean_list_length", "marginal_recall"])
        for row in rows:
            writer.writerow(["all_turns", row["source"], "", row["queries"], missing_plans, topk,
                             row["recall"], row["mean_list_length"], row["marginal_recall"]])
        for turn, recall in by_turn.items():
            writer.writerow(["turn", "union", turn, len(session_range), "", topk, recall, "", ""])

    lines = [
        f"# Recall@{topk} on dev sessions {session_range.start}–{session_range.stop - 1}",
        "",
        f"{total} queries. Missing cached plans: {missing_plans}.",
        "",
        f"| Source | Recall@{topk} | Mean list length / pool size | Marginal recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        value = row["marginal_recall"]
        marginal = "" if value is None else f"{value:.4f}"
        lines.append(f"| {row['source']} | {row['recall']:.4f} | "
                     f"{row['mean_list_length']:.2f} | {marginal} |")
    lines += ["", f"| Turn | Union recall@{topk} |", "| ---: | ---: |"]
    lines += [f"| {turn} | {recall:.4f} |" for turn, recall in by_turn.items()]
    report = "\n".join(lines) + "\n"
    path.with_suffix(".md").write_text(report, encoding="utf-8")
    print(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-session", type=int, default=DIAGNOSTIC.start)
    parser.add_argument("--num-sessions", type=int, default=len(DIAGNOSTIC))
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch retrieval.")
    parser.add_argument("--planner-cache-from", nargs="+", type=Path, default=[])
    parser.add_argument("--planner-model", default="qwen3:8b")
    parser.add_argument("--session-bm25-weights", type=int, nargs="+",
                        help="Repeat weights in recency order, one per included anchor.")
    parser.add_argument("--session-bm25-no-tags", action="store_true")
    parser.add_argument("--feature-stats-queries", type=int, default=0,
                        help="Time feature building and report NaN rates on this many queries.")
    parser.add_argument("--tag", default="diagnostic")
    args = parser.parse_args(argv)

    if args.start_session < 0 or min(args.num_sessions, args.topk, args.threads) <= 0:
        parser.error("use a nonnegative start session and positive session count, topk and thread count")
    if len(args.models) != len(set(args.models)):
        parser.error("each model may be selected only once")
    if args.feature_stats_queries < 0:
        parser.error("the feature statistics query count must be nonnegative")
    if args.session_bm25_weights and min(args.session_bm25_weights) < 0:
        parser.error("session BM25 weights must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("the tag may contain only letters, digits, underscores and hyphens")
    if any(not path.is_file() for path in args.planner_cache_from):
        parser.error("every planner cache path must refer to an existing file")
    session_range = range(args.start_session, args.start_session + args.num_sessions)
    if max(session_range.start, FINAL_HOLDOUT.start) < min(session_range.stop, FINAL_HOLDOUT.stop):
        parser.error("recall reports cannot use final holdout sessions")

    from datasets import load_dataset

    torch.set_num_threads(args.threads)
    dev = load_dataset(DATASET_NAME, split=DEV_SPLIT)
    if session_range.stop > len(dev):
        parser.error(f"requested session range exceeds the {len(dev)} dev sessions")
    sessions = dev.select(session_range)
    planner_tags = tuple(f"{args.planner_model}|query=latest_user|v{v}" for v in (2, 3))
    cache = {}
    if args.planner_cache_from:
        *extra, last = args.planner_cache_from
        cache = load_planner_cache(last, planner_tags, extra_paths=extra)
    session_bm25_options = {}
    if args.session_bm25_weights:
        session_bm25_options.update(num_anchors=len(args.session_bm25_weights),
                                   repeat_weights=args.session_bm25_weights)
    if args.session_bm25_no_tags:
        session_bm25_options["include_tags"] = False
    profile = FeatureProfile(args.feature_stats_queries) if args.feature_stats_queries else None
    track_metadata = None
    if profile is not None:
        from recommender_v2.training.metadata import load_track_metadata

        track_metadata = load_track_metadata()
    recommender = RecommenderV2(
        build_retrievers(args.models, device="cpu", session_bm25_options=session_bm25_options),
        planner=QueryPlanner(mode="mock"), ranker=RRFRanker(), candidate_k=args.topk,
        track_metadata=track_metadata,
    )
    rows, by_turn, missing_plans = measure_recall(sessions, recommender, cache, profile=profile)
    directory = Path("results")
    write_report(directory, args.tag, args.topk, session_range, rows, by_turn, missing_plans)
    if profile is not None:
        profile.write(directory, args.tag)


if __name__ == "__main__":
    main()
