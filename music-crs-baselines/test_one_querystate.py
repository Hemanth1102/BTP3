"""
One-case end-to-end smoke test for Music CRS with the Qwen3 QueryState planner.

Run from the repository root:
    python test_one_querystate.py

Prerequisites:
  - Ollama is running and has qwen3:8b available (`ollama list`)
  - Project dependencies and retrieval caches/models are installed.
  - GPU is optional for the retrieval set below; BGE is intentionally omitted.

This script uses one test session/turn, builds the supported retrievers, runs the
Qwen planner, prints its QueryState, then runs retrieval and recommendation.
It does not generate or overwrite the training dataset.
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from typing import Any

from datasets import load_dataset

DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"
NUM_CANDIDATES = 100
FINAL_K = 20
SESSION_INDEX = 0
TARGET_TURN = 1


def print_json(title: str, value: Any) -> None:
    print(f"\n--- {title} ---")
    try:
        print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    except Exception:
        print(value)


def make_history(session: dict, target_turn: int):
    """Only include user/assistant messages before the target response."""
    history = []
    anchors = []
    ground_truth = None

    for message in session["conversations"]:
        turn = int(message["turn_number"])
        if turn > target_turn:
            break

        role = message["role"]
        if role == "music":
            if turn < target_turn:
                anchors.append(message["content"])
            elif turn == target_turn:
                ground_truth = message["content"]
        elif role == "user":
            history.append(message)
        elif role == "assistant" and turn < target_turn:
            history.append(message)

    current_user_messages = [
        m for m in session["conversations"]
        if int(m["turn_number"]) == target_turn and m["role"] == "user"
    ]
    if not current_user_messages:
        raise RuntimeError(f"No user message found for target turn {target_turn}.")

    query = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    return query, anchors, ground_truth


def call_planner_compatibly(planner, query, history, anchors):
    """
    Invoke the installed planner using only arguments its signature accepts.
    This intentionally reports the signature to make API mismatches visible.
    """
    method = getattr(planner, "plan", None)
    if method is None:
        raise AttributeError(
            f"{type(planner).__name__} has no .plan() method. "
            "Inspect recommender_v2/planner/planner.py and adapt this runner."
        )

    signature = inspect.signature(method)
    params = signature.parameters
    kwargs = {}

    possible_values = {
        "query": query,
        "conversation_history": history,
        "history": history,
        "anchor_track_ids": anchors,
        "session_track_ids": anchors,
    }

    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    for name, value in possible_values.items():
        if name in params or accepts_kwargs:
            kwargs[name] = value

    # If the planner's first required parameter has a different name, pass query
    # positionally and avoid passing it a second time by keyword.
    first_required = next(
        (
            p for p in params.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.default is inspect.Parameter.empty
        ),
        None,
    )
    if first_required and first_required.name in kwargs:
        kwargs.pop(first_required.name)

    print(f"Planner method signature: {signature}")
    return method(query, **kwargs)


def main():
    print("Loading one test session...")
    dataset = load_dataset(DATASET_NAME)
    sessions = dataset["test"]
    if SESSION_INDEX >= len(sessions):
        raise IndexError(f"SESSION_INDEX={SESSION_INDEX} exceeds test split size.")

    session = sessions[SESSION_INDEX]
    query, anchors, ground_truth = make_history(session, TARGET_TURN)

    print(f"Session index: {SESSION_INDEX}")
    print(f"Session ID: {session.get('session_id')}")
    print(f"User ID: {session.get('user_id')}")
    print(f"Target turn: {TARGET_TURN}")
    print(f"Previous music anchors: {len(anchors)}")
    print_json("Query sent to planner", query)
    print_json("Anchor track IDs", anchors)
    print(f"Ground-truth track ID: {ground_truth!r}")
    # print("\n--- DEBUG: Original query ---")
    # print(repr(query))

    # print("\n--- DEBUG: Anchors ---")
    # print(repr(anchors))

    # print("\n--- DEBUG: Conversation history sent to planner ---")
    # print(f"Number of conversation entries: {len(session['conversations'])}")
    # for i, turn in enumerate(session["conversations"]):
    #     print(f"\nTurn {i}:")
    #     print(repr(turn))

    print("\nInitializing Qwen3 QueryState planner...")
    # exit(0)
    from recommender_v2.planner.ollama import OllamaQwenPlanner
    from recommender_v2.planner.planner import QueryPlanner

    llm = OllamaQwenPlanner("qwen3:8b")
    planner = QueryPlanner(llm=llm, mode="llm")

    start = time.perf_counter()
    query_state = call_planner_compatibly(
        planner=planner,
        query=query,
        history=session["conversations"],
        anchors=anchors,
    )
    print(f"Planner elapsed: {time.perf_counter() - start:.2f}s")
    print_json("Qwen QueryState", query_state)

    # Use the retrievers actually implemented in config.py. These are selected
    # to give a reasonable smoke test without forcing BGE/GPU initialization.
    print("\nBuilding retrievers...")
    from recommender_v2.config import (
        build_bm25,
        build_cf_bpr,
        build_item_cf,
        build_item_clap,
        build_item_metadata,
        build_artist_lookup,
        build_album_lookup,
    )
    from recommender_v2 import RecommenderV2

    retrievers = [
        build_bm25(),
        build_cf_bpr(),
        build_item_cf(),
        build_item_clap(),
        build_item_metadata(),
        build_artist_lookup(),
        build_album_lookup(),
    ]
    print("Loaded retrievers:", [type(r).__name__ for r in retrievers])

    # The current RecommenderV2 constructor in this project creates its own
    # QueryPlanner. We already ran the planner above to expose and inspect the
    # QueryState. This next section runs the actual recommendation pipeline.
    recommender = RecommenderV2(
        retrievers=retrievers,
        candidate_k=NUM_CANDIDATES,
        final_k=FINAL_K,
    )

    print("\nRunning retrieval...")
    start = time.perf_counter()
    retrieval_results = recommender.retrieve_all(
        query=query,
        user_id=session.get("user_id"),
        conversation_history=session["conversations"],
        anchor_track_ids=anchors,
        session_track_ids=anchors,
    )
    retrieval_elapsed = time.perf_counter() - start

    print("\nCandidate recall by retriever:")
    union_ids = set()
    for source, results in retrieval_results.items():
        ids = [r.track_id for r in results]
        union_ids.update(ids)
        hit_rank = next(
            (getattr(r, "rank", i + 1) for i, r in enumerate(results)
             if r.track_id == ground_truth),
            None,
        )
        print(
            f"  {source:<24} candidates={len(results):>4} "
            f"ground_truth_rank={hit_rank}"
        )

    print(f"Union candidate count: {len(union_ids)}")
    print(f"Ground truth in union: {ground_truth in union_ids}")

    print("\nRunning final recommendation...")
    start = time.perf_counter()
    recommendations = recommender.recommend(
        query=query,
        user_id=session.get("user_id"),
        conversation_history=session["conversations"],
        anchor_track_ids=anchors,
        session_track_ids=anchors,
    )
    recommend_elapsed = time.perf_counter() - start

    print(f"\nTop {min(FINAL_K, len(recommendations))} recommended track IDs:")
    for rank, track_id in enumerate(recommendations[:FINAL_K], 1):
        marker = "  <-- GROUND TRUTH" if track_id == ground_truth else ""
        print(f"{rank:>2}. {track_id}{marker}")

    if ground_truth is not None:
        try:
            rank = recommendations.index(ground_truth) + 1
        except ValueError:
            rank = None
        print(f"\nGround-truth rank: {rank}")
        print(f"Hit@20: {int(rank is not None and rank <= 20)}")
        print(
            "nDCG@20: "
            f"{(1.0 / __import__('math').log2(rank + 1)) if rank and rank <= 20 else 0.0:.6f}"
        )

    print(f"\nRetrieval elapsed: {retrieval_elapsed:.2f}s")
    print(f"Recommendation elapsed: {recommend_elapsed:.2f}s")
    print("\nSmoke test finished. No training dataset or model file was modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "If this fails at the planner call, check the actual .plan() signature "
            "in recommender_v2/planner/planner.py. The runner prints that signature "
            "before invoking it.",
            file=sys.stderr,
        )
        raise
