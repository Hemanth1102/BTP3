import argparse
import math
import time
from collections import defaultdict

from datasets import load_dataset


DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"


# ============================================================
# Conversation parsing
# ============================================================

def get_turn_data(session, target_turn):
    """
    Build the information available to the recommender at target_turn.

    Includes:
        - previous user messages
        - previous assistant messages
        - current user message

    Excludes:
        - current-turn assistant response
        - current-turn ground-truth music
        - future turns
    """

    messages = session["conversations"]

    history = []
    anchor_track_ids = []
    ground_truth = None

    for message in messages:
        turn_number = message["turn_number"]

        if turn_number > target_turn:
            break

        if message["role"] == "music":

            if turn_number < target_turn:
                anchor_track_ids.append(
                    message["content"]
                )

            elif turn_number == target_turn:
                ground_truth = message["content"]

            continue

        if message["role"] == "user":
            history.append(message)

        elif message["role"] == "assistant":
            if turn_number < target_turn:
                history.append(message)

    if ground_truth is None:
        raise ValueError(
            f"No ground truth found for turn {target_turn}"
        )

    query = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    return (
        query,
        ground_truth,
        anchor_track_ids,
    )


# ============================================================
# Metrics
# ============================================================

def ndcg_at_k(ground_truth, predictions, k):
    predictions = predictions[:k]

    for rank, prediction in enumerate(
        predictions,
        start=1,
    ):
        if prediction == ground_truth:
            return 1.0 / math.log2(rank + 1)

    return 0.0


def hit_at_k(ground_truth, predictions, k):
    return int(
        ground_truth in predictions[:k]
    )


def contains_ground_truth(
    ground_truth,
    results,
):
    """
    `results` is a list of RetrievalResult.
    """
    return any(
        result.track_id == ground_truth
        for result in results
    )


# ============================================================
# Recommender construction
# ============================================================

def build_recommender(
    model_names,
    candidate_k,
):
    from recommender_v2.config import (
        build_bm25,
        build_cf_bpr,
        build_item_cf,
        build_item_clap,
        build_item_metadata,
        build_artist_lookup,
        build_album_lookup,
        build_bge
    )

    retrievers = []

    if "bm25" in model_names:
        print("Loading BM25...")
        retrievers.append(build_bm25())

    if "bge" in model_names:
        print("Loading BGE...")
        retrievers.append(build_bge(device="cuda"))

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
        retrievers.append(
            build_artist_lookup()
        )

    if "album" in model_names:
        print("Loading Album Lookup...")
        retrievers.append(
            build_album_lookup()
        )

    if not retrievers:
        raise ValueError("No models selected.")

    from recommender_v2 import RecommenderV2

    return RecommenderV2(
        retrievers=retrievers,
        candidate_k=candidate_k,
        final_k=20,
    )


# ============================================================
# Evaluate one session
# ============================================================

def evaluate_session(
    recommender,
    session,
    start_turn,
    end_turn,
    verbose=False,
):
    session_id = session["session_id"]
    user_id = session["user_id"]

    turn_metrics = []

    for target_turn in range(
        start_turn,
        end_turn + 1,
    ):

        (
            query,
            ground_truth,
            anchor_track_ids,
        ) = get_turn_data(
            session,
            target_turn,
        )

        start_time = time.perf_counter()

        retrieval_results = recommender.retrieve_all(
            query=query,
            user_id=user_id,
            conversation_history=session["conversations"],
            anchor_track_ids=anchor_track_ids,
            session_track_ids=anchor_track_ids,
        )

        final_ids = recommender.recommend(
            query=query,
            user_id=user_id,
            conversation_history=session["conversations"],
            anchor_track_ids=anchor_track_ids,
            session_track_ids=anchor_track_ids,
        )

        elapsed = time.perf_counter() - start_time

        # ----------------------------------------------------
        # Per-retriever recall
        # ----------------------------------------------------

        retriever_recall = {}

        for source, results in retrieval_results.items():

            retriever_recall[source] = {
                "hit": int(
                    contains_ground_truth(
                        ground_truth,
                        results,
                    )
                )
            }

        # ----------------------------------------------------
        # Union recall
        # ----------------------------------------------------

        union_ids = set()

        for results in retrieval_results.values():
            for result in results:
                union_ids.add(result.track_id)

        union_hit = int(
            ground_truth in union_ids
        )

        # ----------------------------------------------------
        # Final ranking
        # ----------------------------------------------------

        metrics = {
            "ndcg@1": ndcg_at_k(
                ground_truth,
                final_ids,
                1,
            ),
            "ndcg@10": ndcg_at_k(
                ground_truth,
                final_ids,
                10,
            ),
            "ndcg@20": ndcg_at_k(
                ground_truth,
                final_ids,
                20,
            ),
            "hit@1": hit_at_k(
                ground_truth,
                final_ids,
                1,
            ),
            "hit@10": hit_at_k(
                ground_truth,
                final_ids,
                10,
            ),
            "hit@20": hit_at_k(
                ground_truth,
                final_ids,
                20,
            ),
        }

        # ----------------------------------------------------
        # Rank of ground truth within each branch
        # ----------------------------------------------------

        retriever_ranks = {}

        for source, results in retrieval_results.items():

            rank = None

            for result in results:
                if result.track_id == ground_truth:
                    rank = result.rank
                    break

            retriever_ranks[source] = rank

        final_rank = None

        for rank, track_id in enumerate(
            final_ids,
            start=1,
        ):
            if track_id == ground_truth:
                final_rank = rank
                break

        turn_metrics.append(
            {
                "session_id": session_id,
                "turn_number": target_turn,
                **metrics,
                "retriever_recall": retriever_recall,
                "retriever_ranks": retriever_ranks,
                "union_hit": union_hit,
                "union_size": len(union_ids),
                "final_rank": final_rank,
                "latency_sec": elapsed,
            }
        )

        if verbose:

            print(
                f"\nTurn {target_turn}"
            )

            print(
                f"Ground truth: {ground_truth}"
            )

            print(
                f"Union size: {len(union_ids)}"
            )

            print(
                "\nRetriever recall:"
            )

            for source in retrieval_results:

                hit = retriever_recall[
                    source
                ]["hit"]

                rank = retriever_ranks[
                    source
                ]

                if rank is None:
                    print(
                        f"  {source:<20} MISS"
                    )
                else:
                    print(
                        f"  {source:<20} "
                        f"HIT @ {rank}"
                    )

            print(
                f"  {'UNION':<20} "
                f"{'HIT' if union_hit else 'MISS'}"
            )

            print(
                "\nFinal:"
            )

            if final_rank is None:
                print(
                    "  FINAL: MISS"
                )
            else:
                print(
                    f"  FINAL: HIT @ {final_rank}"
                )

            print(
                f"  nDCG@20: "
                f"{metrics['ndcg@20']:.6f}"
            )

            print(
                f"  latency: "
                f"{elapsed:.4f}s"
            )

    return turn_metrics


# ============================================================
# Aggregation
# ============================================================

def aggregate_metrics(all_turn_metrics):
    by_turn = defaultdict(list)

    for row in all_turn_metrics:
        by_turn[row["turn_number"]].append(row)

    turn_means = {}

    # Collect all retriever names.
    sources = set()

    for row in all_turn_metrics:
        sources.update(
            row["retriever_recall"].keys()
        )

    for turn_number, rows in by_turn.items():

        source_recall = {}

        for source in sources:
            hits = sum(
                row["retriever_recall"]
                .get(source, {})
                .get("hit", 0)
                for row in rows
            )

            source_recall[source] = (
                hits / len(rows)
            )

        union_recall = (
            sum(
                row["union_hit"]
                for row in rows
            )
            / len(rows)
        )

        avg_union_size = (
            sum(
                row["union_size"]
                for row in rows
            )
            / len(rows)
        )

        turn_means[turn_number] = {
            "ndcg@1": sum(
                row["ndcg@1"]
                for row in rows
            ) / len(rows),

            "ndcg@10": sum(
                row["ndcg@10"]
                for row in rows
            ) / len(rows),

            "ndcg@20": sum(
                row["ndcg@20"]
                for row in rows
            ) / len(rows),

            "hit@1": sum(
                row["hit@1"]
                for row in rows
            ) / len(rows),

            "hit@10": sum(
                row["hit@10"]
                for row in rows
            ) / len(rows),

            "hit@20": sum(
                row["hit@20"]
                for row in rows
            ) / len(rows),

            "union_recall": union_recall,

            "avg_union_size": avg_union_size,

            "retriever_recall": source_recall,

            "latency_sec": sum(
                row["latency_sec"]
                for row in rows
            ) / len(rows),
        }

    num_turns = len(turn_means)

    sources = sorted(sources)

    macro = {
        "ndcg@1": sum(
            x["ndcg@1"]
            for x in turn_means.values()
        ) / num_turns,

        "ndcg@10": sum(
            x["ndcg@10"]
            for x in turn_means.values()
        ) / num_turns,

        "ndcg@20": sum(
            x["ndcg@20"]
            for x in turn_means.values()
        ) / num_turns,

        "hit@1": sum(
            x["hit@1"]
            for x in turn_means.values()
        ) / num_turns,

        "hit@10": sum(
            x["hit@10"]
            for x in turn_means.values()
        ) / num_turns,

        "hit@20": sum(
            x["hit@20"]
            for x in turn_means.values()
        ) / num_turns,

        "union_recall": sum(
            x["union_recall"]
            for x in turn_means.values()
        ) / num_turns,

        "avg_union_size": sum(
            x["avg_union_size"]
            for x in turn_means.values()
        ) / num_turns,

        "retriever_recall": {},
    }

    for source in sources:
        macro["retriever_recall"][source] = (
            sum(
                x["retriever_recall"]
                .get(source, 0.0)
                for x in turn_means.values()
            )
            / num_turns
        )

    macro["latency_sec"] = sum(
        x["latency_sec"]
        for x in turn_means.values()
    ) / num_turns

    return macro, turn_means


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Retrieval-only evaluator with "
            "candidate recall analysis."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=[
            "bm25",
            "cf",
            "bge",
            "metadata",
            "lyrics",
            "attributes",
            "clap",
            "item_cf",
            "item_clap",
            "item_metadata",
            "artist",
            "album",
        ]
    )

    parser.add_argument(
        "--num-sessions",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--start-session",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--start-turn",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--end-turn",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help=(
            "Number of candidates requested from "
            "each retriever. Default: 100."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    print("Loading challenge dataset...")

    dataset = load_dataset(
        DATASET_NAME
    )

    sessions = dataset["test"]

    end_session = (
        args.start_session
        + args.num_sessions
    )

    if end_session > len(sessions):
        raise ValueError(
            "Requested session range exceeds "
            "the test split."
        )

    selected_sessions = sessions.select(
        range(
            args.start_session,
            end_session,
        )
    )

    recommender = build_recommender(
        model_names=set(args.models),
        candidate_k=args.candidate_k,
    )

    all_turn_metrics = []

    overall_start = time.perf_counter()

    for session_index, session in enumerate(
        selected_sessions,
        start=args.start_session,
    ):

        print("\n" + "=" * 70)
        print(
            f"SESSION INDEX: {session_index}"
        )
        print(
            f"SESSION ID: {session['session_id']}"
        )
        print("=" * 70)

        metrics = evaluate_session(
            recommender=recommender,
            session=session,
            start_turn=args.start_turn,
            end_turn=args.end_turn,
            verbose=args.verbose,
        )

        all_turn_metrics.extend(metrics)

    total_time = (
        time.perf_counter()
        - overall_start
    )

    macro, turn_means = aggregate_metrics(
        all_turn_metrics
    )

    # ========================================================
    # Final results
    # ========================================================

    print("\n" + "=" * 70)
    print("RETRIEVAL-ONLY RESULTS")
    print("=" * 70)

    print(
        f"Sessions evaluated: {len(selected_sessions)}"
    )

    print(
        f"Turns evaluated: "
        f"{args.start_turn}-{args.end_turn}"
    )

    print(
        f"Candidates per retriever: "
        f"{args.candidate_k}"
    )

    print(
        f"Total time: {total_time:.2f}s"
    )

    print("\nFinal ranking metrics:")

    print(
        f"  nDCG@1  = {macro['ndcg@1']:.6f}"
    )

    print(
        f"  nDCG@10 = {macro['ndcg@10']:.6f}"
    )

    print(
        f"  nDCG@20 = {macro['ndcg@20']:.6f}"
    )

    print(
        f"  Hit@1   = {macro['hit@1']:.6f}"
    )

    print(
        f"  Hit@10  = {macro['hit@10']:.6f}"
    )

    print(
        f"  Hit@20  = {macro['hit@20']:.6f}"
    )

    print("\nCandidate recall:")

    for source in sorted(
        macro["retriever_recall"]
    ):
        print(
            f"  {source:<20} "
            f"recall = "
            f"{macro['retriever_recall'][source]:.6f}"
        )

    print(
        f"  {'UNION':<20} "
        f"recall = "
        f"{macro['union_recall']:.6f}"
    )

    print(
        f"\nAverage union size: "
        f"{macro['avg_union_size']:.1f}"
    )

    print(
        f"Average latency/turn: "
        f"{macro['latency_sec']:.4f}s"
    )

    print("\nBy turn:")

    for turn_number in sorted(
        turn_means
    ):

        values = turn_means[
            turn_number
        ]

        print(
            f"  Turn {turn_number}: "
            f"nDCG@20="
            f"{values['ndcg@20']:.6f}, "
            f"UnionRecall="
            f"{values['union_recall']:.4f}, "
            f"UnionSize="
            f"{values['avg_union_size']:.1f}"
        )


if __name__ == "__main__":
    main()