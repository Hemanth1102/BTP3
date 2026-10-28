"""
Build a candidate-level dataset for training a LightGBM ranker.

Run from the music-crs-baselines repository root, for example:
python build_lgbm_dataset.py --num-sessions 20 --start-session 0 --start-turn 1 --end-turn 8 --output data/processed/lgbm_candidates.csv

Each row is one (query, candidate-track) pair. The ground-truth track receives
label=1 if it appears in the retrieved candidate union; all other candidates
receive label=0. Queries whose ground-truth track is not retrieved are retained
for recall diagnostics, but should be excluded from initial LambdaRank training
because they have no positive candidate.
"""

import argparse
import math
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"

# Use only retrievers that build_recommender() in evaluate_recommender_only.py
# actually supports. Keep this list aligned with that function.
DEFAULT_MODELS = [
    "bm25", "bge", "cf", "item_cf", "item_clap",
    "item_metadata", "artist", "album",
]


def load_evaluator():
    """Import the existing project evaluator instead of duplicating its setup."""
    import evaluate_recommender_only as evaluator
    return evaluator


def history_available_at_turn(session, target_turn):
    """
    Return only user/assistant messages available when the target user request
    arrives. Excludes music responses and all future turns to prevent leakage.
    """
    history = []
    for message in session["conversations"]:
        turn = message["turn_number"]
        if turn > target_turn:
            break
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        # Current-turn assistant response is future information.
        if role == "assistant" and turn >= target_turn:
            continue
        history.append(message)
    return history


def get_turn_data_without_leakage(session, target_turn):
    """
    Return query, target track ID, prior anchor track IDs, and truncated history.
    Follows the message structure used by evaluate_recommender_only.py.
    """
    messages = session["conversations"]
    history = history_available_at_turn(session, target_turn)
    ground_truth = None
    anchors = []

    for message in messages:
        turn = message["turn_number"]
        if turn > target_turn:
            break
        if message.get("role") != "music":
            continue
        if turn < target_turn:
            anchors.append(message["content"])
        elif turn == target_turn:
            ground_truth = message["content"]

    if ground_truth is None:
        raise ValueError(f"No ground-truth music response for turn {target_turn}")

    query = "\n".join(
        f"{message['role']}: {message['content']}" for message in history
    )
    return query, ground_truth, anchors, history


def get_candidate_score(result):
    """Return a score if the retriever result exposes one; otherwise NaN."""
    for name in ("score", "similarity", "retrieval_score", "relevance_score"):
        value = getattr(result, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Retriever keys supported by evaluate_recommender_only.py")
    parser.add_argument("--num-sessions", type=int, default=20)
    parser.add_argument("--start-session", type=int, default=0)
    parser.add_argument("--start-turn", type=int, default=1)
    parser.add_argument("--end-turn", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--output", default="data/processed/lgbm_candidates.csv")
    parser.add_argument("--cpu-bge", action="store_true",
                        help="Build BGE on CPU instead of CUDA")
    args = parser.parse_args()

    evaluator = load_evaluator()

    # If CPU BGE is requested, patch the imported builder at the narrowest point.
    if args.cpu_bge:
        original_build_recommender = evaluator.build_recommender

        def build_recommender_cpu(model_names, candidate_k):
            from recommender_v2.config import (
                build_bm25, build_cf_bpr, build_item_cf, build_item_clap,
                build_item_metadata, build_artist_lookup, build_album_lookup,
                build_bge,
            )
            retrievers = []
            if "bm25" in model_names:
                print("Loading BM25...")
                retrievers.append(build_bm25())
            if "bge" in model_names:
                print("Loading BGE on CPU...")
                retrievers.append(build_bge(device="cpu"))
            if "cf" in model_names:
                print("Loading CF-BPR...")
                retrievers.append(build_cf_bpr())
            if "item_cf" in model_names:
                print("Loading Item CF-BPR...")
                retrievers.append(build_item_cf())
            if "item_clap" in model_names:
                print("Loading Item CLAP...")
                retrievers.append(build_item_clap())
            if "item_metadata" in model_names:
                print("Loading Item Metadata...")
                retrievers.append(build_item_metadata())
            if "artist" in model_names:
                print("Loading Artist Lookup...")
                retrievers.append(build_artist_lookup())
            if "album" in model_names:
                print("Loading Album Lookup...")
                retrievers.append(build_album_lookup())
            if not retrievers:
                raise ValueError("No models selected.")
            from recommender_v2 import RecommenderV2
            return RecommenderV2(retrievers=retrievers,
                                 candidate_k=candidate_k, final_k=20)

        evaluator.build_recommender = build_recommender_cpu

    print("Loading challenge dataset...")
    dataset = load_dataset(DATASET_NAME)
    sessions = dataset["test"]
    end_session = args.start_session + args.num_sessions
    if args.start_session < 0 or end_session > len(sessions):
        raise ValueError(f"Requested sessions [{args.start_session}, {end_session}) "
                         f"but test split has {len(sessions)} sessions.")

    recommender = evaluator.build_recommender(
        model_names=set(args.models),
        candidate_k=args.candidate_k,
    )

    rows = []
    diagnostics = []
    total_start = time.perf_counter()

    for session_index in range(args.start_session, end_session):
        session = sessions[session_index]
        print(f"\nSession {session_index}: {session['session_id']}")

        for target_turn in range(args.start_turn, args.end_turn + 1):
            try:
                query, ground_truth, anchors, history = get_turn_data_without_leakage(
                    session, target_turn
                )
            except ValueError as exc:
                print(f"  Skipping turn {target_turn}: {exc}")
                continue

            started = time.perf_counter()
            result_by_source = recommender.retrieve_all(
                query=query,
                user_id=session["user_id"],
                conversation_history=history,
                anchor_track_ids=anchors,
                session_track_ids=anchors,
            )
            elapsed = time.perf_counter() - started

            # Merge candidates and retain per-retriever score/rank/presence.
            per_track = {}
            for source, results in result_by_source.items():
                for position, result in enumerate(results, start=1):
                    track_id = getattr(result, "track_id", None)
                    if track_id is None:
                        continue
                    entry = per_track.setdefault(track_id, {})
                    entry[f"{source}__retrieved"] = 1
                    rank = getattr(result, "rank", None)
                    entry[f"{source}__rank"] = (
                        float(rank) if rank is not None else float(position)
                    )
                    entry[f"{source}__score"] = get_candidate_score(result)

            query_id = f"{session['session_id']}__turn_{target_turn}"
            union_ids = list(per_track)
            union_hit = int(ground_truth in per_track)
            diagnostics.append({
                "query_id": query_id,
                "session_id": session["session_id"],
                "turn_number": target_turn,
                "ground_truth_track_id": ground_truth,
                "candidate_count": len(union_ids),
                "ground_truth_in_candidates": union_hit,
                "retrieval_latency_sec": elapsed,
            })

            for track_id, features in per_track.items():
                row = {
                    "query_id": query_id,
                    "session_id": session["session_id"],
                    "turn_number": target_turn,
                    "track_id": track_id,
                    "label": int(track_id == ground_truth),
                    "ground_truth_track_id": ground_truth,
                    "candidate_count": len(union_ids),
                    "num_retrievers": sum(
                        1 for key, value in features.items()
                        if key.endswith("__retrieved") and value == 1
                    ),
                }
                row.update(features)
                rows.append(row)

            print(
                f"  Turn {target_turn}: candidates={len(union_ids)}, "
                f"ground_truth_hit={union_hit}, retrieval={elapsed:.1f}s"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_df = pd.DataFrame(rows)
    diagnostic_df = pd.DataFrame(diagnostics)

    # Stable numeric defaults for missing retriever outputs:
    # missing score -> NaN (LightGBM supports missing values);
    # missing rank -> 0 and missing retrieved flag -> 0.
    for column in candidate_df.columns:
        if column.endswith("__rank"):
            candidate_df[column] = candidate_df[column].fillna(0)
        elif column.endswith("__retrieved"):
            candidate_df[column] = candidate_df[column].fillna(0).astype("int8")

    candidate_df.to_csv(output_path, index=False)
    diagnostics_path = output_path.with_name(output_path.stem + "_queries.csv")
    diagnostic_df.to_csv(diagnostics_path, index=False)

    print("\nDataset build complete.")
    print(f"Candidate rows: {len(candidate_df):,}")
    print(f"Queries: {candidate_df['query_id'].nunique() if not candidate_df.empty else 0:,}")
    print(f"Positive rows: {int(candidate_df['label'].sum()) if not candidate_df.empty else 0:,}")
    print(f"Queries with ground truth retrieved: "
          f"{int(diagnostic_df['ground_truth_in_candidates'].sum()) if not diagnostic_df.empty else 0:,}"
          f"/{len(diagnostic_df):,}")
    print(f"Candidate dataset: {output_path.resolve()}")
    print(f"Query diagnostics: {diagnostics_path.resolve()}")
    print(f"Total elapsed: {time.perf_counter() - total_start:.1f}s")

    if not candidate_df.empty and candidate_df["label"].sum() == 0:
        print("\nWARNING: no positive candidate rows were retrieved. Do not train a ranker "
              "on this dataset yet; first improve candidate recall or expand the candidate pool.")


if __name__ == "__main__":
    main()
