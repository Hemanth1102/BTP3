import argparse
import csv

from datasets import load_dataset

from recommender_v2 import RecommenderV2
from recommender_v2.config import (
    build_bm25,
    build_cf_bpr,
    build_bge,
    build_clap,
    build_item_cf,
    build_item_clap,
    build_item_metadata,
    build_artist_lookup,
    build_album_lookup,
)


DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"


def get_test_turn(session, target_turn=2):
    """
    Build the conversation context for a specific user turn.

    Returns:
        query: conversation available before/at target turn
        ground_truth: music track associated with target turn
        anchor_track_ids: tracks recommended in previous turns
    """

    history = []
    anchor_track_ids = []
    ground_truth = None

    for message in session["conversations"]:

        turn = message["turn_number"]

        if turn > target_turn:
            break

        if message["role"] == "music":

            if turn < target_turn:
                anchor_track_ids.append(
                    message["content"]
                )

            elif turn == target_turn:
                ground_truth = message["content"]

            continue

        history.append(message)

    query = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    if not query:
        raise ValueError(
            f"No user conversation found for turn {target_turn}"
        )

    if ground_truth is None:
        raise ValueError(
            f"No ground-truth track found for turn {target_turn}"
        )

    return query, ground_truth, anchor_track_ids


def build_selected_recommender(model_names):

    retrievers = []

    if "bm25" in model_names:
        retrievers.append(build_bm25())

    if "cf" in model_names:
        retrievers.append(build_cf_bpr())

    if "bge" in model_names:
        retrievers.append(build_bge())

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

    return RecommenderV2(
        retrievers=retrievers,
        candidate_k=100,
        final_k=20,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["bm25", "cf"],
    )

    parser.add_argument(
        "--num-sessions",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--target-turn",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output",
        default="training_data.csv",
    )

    args = parser.parse_args()

    print("Loading challenge dataset...")

    dataset = load_dataset(DATASET_NAME)

    train = dataset["train"]

    end = min(
        args.start + args.num_sessions,
        len(train),
    )

    sessions = train.select(
        range(args.start, end)
    )

    print(
        f"Building training data from "
        f"{len(sessions)} sessions..."
    )

    print(
        f"Models: {args.models}"
    )

    recommender = build_selected_recommender(
        set(args.models)
    )

    rows = []

    for session_index, session in enumerate(sessions):

        try:

            query, ground_truth, anchor_track_ids = (
                get_test_turn(
                    session,
                    target_turn=args.target_turn,
                )
            )

            user_id = session.get(
                "user_id",
                None,
            )

            state, retrieval_results = (
                recommender.retrieve_for_ranking(
                    query=query,
                    user_id=user_id,
                    conversation_history=session[
                        "conversations"
                    ],
                    anchor_track_ids=anchor_track_ids,
                )
            )
            
            print("QUERY:")
            print(query)

            print("ANCHORS:")
            print(anchor_track_ids)

            print("QUERY STATE:")
            print(state.to_dict())

            features = recommender.build_features(
                retrieval_results=retrieval_results,
                user_id=user_id,
                anchor_track_ids=anchor_track_ids,
                query_state=state,
            )

            for feature_row in features:

                track_id = feature_row[
                    "track_id"
                ]

                label = (
                    1
                    if track_id == ground_truth
                    else 0
                )

                row = dict(feature_row)

                row["session_id"] = (
                    session.get(
                        "session_id",
                        session_index,
                    )
                )

                row["label"] = label

                rows.append(row)

        except Exception as e:

            print(
                f"Skipping session "
                f"{session_index}: {e}"
            )

    if not rows:

        raise RuntimeError(
            "No training rows were generated."
        )

    # Keep track/session IDs for inspection,
    # but LightGBM will ignore them later.
    fieldnames = list(
        rows[0].keys()
    )

    with open(
        args.output,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "Training dataset created:"
    )
    print(
        f"  Rows: {len(rows)}"
    )
    print(
        f"  Features: "
        f"{len(fieldnames) - 3}"
    )
    print(
        f"  Output: {args.output}"
    )


if __name__ == "__main__":
    main()