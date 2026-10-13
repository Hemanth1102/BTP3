import argparse
import json
from pathlib import Path
import time

from datasets import load_dataset

from recommender_v2 import RecommenderV2
from recommender_v2.planner.planner import QueryPlanner
from recommender_v2.planner.ollama import OllamaQwenPlanner
from recommender_v2.config import (
    build_bm25,
    build_cf_bpr,
    build_clap,
    build_item_cf,
    build_item_clap,
    build_item_metadata,
    build_artist_lookup,
    build_album_lookup,
    build_bge,
)


DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"


def build_selected_recommender(model_names):

    retrievers = []

    if "bm25" in model_names:
        retrievers.append(build_bm25())

    if "cf" in model_names:
        retrievers.append(build_cf_bpr())

    if "clap" in model_names:
        retrievers.append(build_clap())

    if "item_cf" in model_names:
        retrievers.append(build_item_cf())

    if "item_clap" in model_names:
        retrievers.append(build_item_clap())

    if "item_metadata" in model_names:
        retrievers.append(build_item_metadata())

    if "artist" in model_names:
        retrievers.append(build_artist_lookup())

    if "album" in model_names:
        retrievers.append(build_album_lookup())

    if "bge" in model_names:
        retrievers.append(build_bge())

    planner = QueryPlanner(
        llm=OllamaQwenPlanner("qwen3:8b"),
        mode="llm",
    )

    return RecommenderV2(
        retrievers=retrievers,
        planner=planner,
        candidate_k=100,
        final_k=20,
    )


def get_context_for_turn(session, target_turn):

    history = []
    anchor_track_ids = []

    for message in session["conversations"]:

        turn = message["turn_number"]

        if turn > target_turn:
            break

        if message["role"] == "music":

            # Do NOT include the current turn's
            # recommended/ground-truth track.
            if turn < target_turn:
                anchor_track_ids.append(message["content"])

            continue

        history.append(message)

    query = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    if not query:
        raise ValueError(
            f"No conversation found for turn {target_turn}"
        )

    return query, history, anchor_track_ids


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "bm25",
            "cf",
            "clap",
            "item_cf",
            "item_clap",
            "item_metadata",
            "artist",
            "album",
        ],
        choices=[
            "bm25",
            "cf",
            "clap",
            "item_cf",
            "item_clap",
            "item_metadata",
            "artist",
            "album",
            "bge",
        ],
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-sessions",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default="my_recommender_devset.json",
    )

    args = parser.parse_args()

    print("Loading development dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="test",
    )

    end = len(dataset)

    if args.num_sessions is not None:
        end = min(
            args.start + args.num_sessions,
            len(dataset),
        )

    sessions = dataset.select(
        range(args.start, end)
    )

    print(
        f"Evaluating {len(sessions)} sessions"
    )

    print(
        f"Models: {args.models}"
    )

    print()
    print("Building recommender...")

    recommender = build_selected_recommender(
        set(args.models)
    )

    predictions = []

    total = len(sessions) * 8
    completed = 0

    for session_index, session in enumerate(sessions):

        session_id = session["session_id"]
        user_id = session["user_id"]

        for target_turn in range(1, 9):

            try:

                query, history, anchor_track_ids = (
                    get_context_for_turn(
                        session,
                        target_turn,
                    )
                )

                # IMPORTANT:
                # Only information available BEFORE this
                # turn is passed to the recommender.
                t0 = time.time()

                final_ids = recommender.recommend(
                    query=query,
                    user_id=user_id,
                    conversation_history=history,
                    anchor_track_ids=anchor_track_ids,
                )

                print(
                    f"Turn {target_turn} recommendation time: "
                    f"{time.time() - t0:.2f}s",
                    flush=True,
                )

                # Safety: remove duplicates while preserving order.
                final_ids = list(
                    dict.fromkeys(final_ids)
                )

                # Official evaluator accepts up to 20.
                final_ids = final_ids[:20]

                predictions.append(
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "turn_number": target_turn,
                        "predicted_track_ids": final_ids,
                        "predicted_response": "",
                    }
                )

            except Exception as e:

                print(
                    f"ERROR session={session_id} "
                    f"turn={target_turn}: {e}"
                )

                raise

            completed += 1

            print(f"Progress: {completed}/{total}", flush=True)

    output_path = Path(args.output)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            predictions,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("PREDICTIONS CREATED")
    print("=" * 70)
    print(f"Sessions:    {len(sessions)}")
    print(f"Predictions: {len(predictions)}")
    print(f"Expected:    {len(sessions) * 8}")
    print(f"Output:      {output_path.resolve()}")


if __name__ == "__main__":
    main()