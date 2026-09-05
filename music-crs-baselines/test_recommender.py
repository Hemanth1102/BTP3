import argparse

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
    build_qwen_retrievers,
)


DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"


def get_user_query(session):
    """Get the first user message from a challenge session."""
    for message in session["conversations"]:
        if message["role"] == "user":
            return message["content"]

    raise ValueError("No user message found in session.")


def get_ground_truth(session):
    """Get the first ground-truth music track ID."""
    for message in session["conversations"]:
        if message["role"] == "music":
            return message["content"]

    raise ValueError("No ground-truth music track found.")

def get_anchor_tracks(session):
    """
    Get music tracks that occurred before the first user turn.

    For the first turn there are normally no anchors.
    """

    anchors = []

    for message in session["conversations"]:

        if message["role"] == "music":
            anchors.append(message["content"])

    return anchors

def get_test_turn(session, target_turn=2):
    """
    Build the conversation context for a specific user turn.

    Returns:
        query: conversation text available before/at the target user turn
        ground_truth: music track associated with the target turn
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
                anchor_track_ids.append(message["content"])
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
    """
    Build only the retrieval models requested by --models.

    Supported:
        bm25
        cf
        metadata
        lyrics
        attributes
        clap
        all
    """

    retrievers = []

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    if "bm25" in model_names or "all" in model_names:
        print("Loading BM25...")
        retrievers.append(build_bm25())

    # BERT

    if "bert" in model_names or "all" in model_names:
        print("Loading BERT metadata retriever...")
        retrievers.append(build_bert())

    # BGE

    if "bge" in model_names or "all" in model_names:
        print("Loading BGE...")
        retrievers.append(
            build_bge(device="cpu")
        )

    # ITEM_CLAP

    if "item_clap" in model_names or "all" in model_names:
        print("Loading Item CLAP...")
        retrievers.append(
            build_item_clap()
        )

    # ---------------------------------------------------------
    # CF-BPR
    # ---------------------------------------------------------

    if "cf" in model_names or "all" in model_names:
        print("Loading CF-BPR...")
        retrievers.append(build_cf_bpr())

    # ---------------------------------------------------------
    # Qwen metadata / lyrics / attributes
    # ---------------------------------------------------------

    qwen_names = {
        "metadata",
        "lyrics",
        "attributes",
    }

    requested_qwen = (
        qwen_names.intersection(model_names)
        or "all" in model_names
    )

    if requested_qwen:
        print("Loading Qwen3 retrievers...")

        qwen_retrievers = build_qwen_retrievers(
            device="cpu"
        )

        for retriever in qwen_retrievers:

            if (
                "all" in model_names
                or retriever.name.replace("qwen_", "") in model_names
            ):
                retrievers.append(retriever)

    # ---------------------------------------------------------
    # CLAP
    # ---------------------------------------------------------

    if "clap" in model_names or "all" in model_names:
        print("Loading CLAP...")
        retrievers.append(
            build_clap(device="cpu")
        )

    if "item_cf" in model_names or "all" in model_names:
        print("Loading Item CF-BPR...")
        retrievers.append(
            build_item_cf()
        )

    if "item_metadata" in model_names or "all" in model_names:
        print("Loading Item Metadata...")
        retrievers.append(
            build_item_metadata()
        )

    if "artist" in model_names or "all" in model_names:
        print("Loading Artist Lookup...")
        retrievers.append(
            build_artist_lookup()
        )

    if "album" in model_names or "all" in model_names:
        print("Loading Album Lookup...")
        retrievers.append(
            build_album_lookup()
        )

    if not retrievers:
        raise ValueError(
            "No valid models selected."
        )

    print("\nActive retrievers:")

    for retriever in retrievers:
        print(f"  - {retriever.name}")

    return RecommenderV2(
        retrievers=retrievers,
        candidate_k=100,
        final_k=20,
    )


def run_test(recommender, session):
    """Run one challenge session through the recommender."""

    session_id = session["session_id"]
    user_id = session["user_id"]

    target_turn = 2

    query, ground_truth, anchor_track_ids = get_test_turn(
        session,
        target_turn=target_turn,
    )

    print(f"\nTarget turn: {target_turn}")
    print(f"Anchor tracks: {anchor_track_ids}")
    print("\n" + "=" * 70)
    print("TEST CASE")
    print("=" * 70)

    print(f"Session:       {session_id}")
    print(f"User:          {user_id}")
    print(f"\nQuery:\n{query}")
    print(f"\nGround truth:\n{ground_truth}")

    # ---------------------------------------------------------
    # Run every selected retriever
    # ---------------------------------------------------------

    retrieval_results = recommender.retrieve_all(
        query=query,
        user_id=user_id,
        anchor_track_ids=anchor_track_ids,
    )

    features = recommender.build_features(
        retrieval_results=retrieval_results,
        user_id=user_id,
        anchor_track_ids=anchor_track_ids,
    )

    # ---------------------------------------------------------
    # LightGBM ranking diagnostics
    # ---------------------------------------------------------

    ranker_scores = recommender.ranker.predict(features)

    ranked_features = sorted(
        zip(features, ranker_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    ground_truth_rank = None
    ground_truth_score = None

    for rank, (row, score) in enumerate(
        ranked_features,
        start=1,
    ):
        if row["track_id"] == ground_truth:
            ground_truth_rank = rank
            ground_truth_score = score
            break

    print("\n" + "=" * 70)
    print("LIGHTGBM DIAGNOSTICS")
    print("=" * 70)

    print(f"Ground truth score: {ground_truth_score}")
    print(f"Ground truth rank: {ground_truth_rank}")

    print("\nTOP 10 LIGHTGBM SCORES")

    for rank, (row, score) in enumerate(
        ranked_features[:10],
        start=1,
    ):
        print(
            f"{rank:3d}. "
            f"{row['track_id']} "
            f"score={score:.6f}"
        )

    print(
        "Candidates:",
        len(features)
    )

    print(
        "Features:",
        len(features[0])
    )

    print(
        features[0]
    )

    # ---------------------------------------------------------
    # Show top 10 for every retriever
    # ---------------------------------------------------------

    for name, results in retrieval_results.items():

        print("\n" + "-" * 70)
        print(f"{name.upper()} TOP 10")
        print("-" * 70)

        for result in results[:10]:

            marker = (
                " <-- GROUND TRUTH"
                if result.track_id == ground_truth
                else ""
            )

            print(
                f"{result.rank:3d}. "
                f"{result.track_id} "
                f"score={result.score:.5f}"
                f"{marker}"
            )

    # ---------------------------------------------------------
    # Candidate union
    # ---------------------------------------------------------

    candidate_sets = {}

    for name, results in retrieval_results.items():

        candidate_sets[name] = {
            result.track_id
            for result in results
        }

    union = set()

    for candidates in candidate_sets.values():
        union.update(candidates)

    print("\n" + "=" * 70)
    print("CANDIDATE ANALYSIS")
    print("=" * 70)

    for name, candidates in candidate_sets.items():

        print(
            f"{name:20s}: "
            f"{len(candidates)} candidates"
        )

    print(
        f"{'UNION':20s}: "
        f"{len(union)} candidates"
    )

    # ---------------------------------------------------------
    # Final recommendation
    # ---------------------------------------------------------

    final_ids = recommender.recommend(
        query=query,
        user_id=user_id,
        anchor_track_ids=anchor_track_ids,
    )

    print("\n" + "=" * 70)
    print("FINAL TOP 20")
    print("=" * 70)

    for rank, track_id in enumerate(
        final_ids,
        start=1,
    ):

        sources = [
            name
            for name, candidates in candidate_sets.items()
            if track_id in candidates
        ]

        marker = (
            " <-- GROUND TRUTH"
            if track_id == ground_truth
            else ""
        )

        print(
            f"{rank:3d}. "
            f"{track_id} "
            f"[{', '.join(sources)}]"
            f"{marker}"
        )

    # ---------------------------------------------------------
    # Hit information
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for name, results in retrieval_results.items():

        ids = [
            result.track_id
            for result in results
        ]

        if ground_truth in ids:

            rank = ids.index(ground_truth) + 1

            print(
                f"{name:20s}: "
                f"HIT @ {rank}"
            )

        else:

            print(
                f"{name:20s}: MISS"
            )

    if ground_truth in final_ids:

        rank = final_ids.index(ground_truth) + 1

        print(
            f"{'FINAL':20s}: "
            f"HIT @ {rank}"
        )

    else:

        print(
            f"{'FINAL':20s}: MISS"
        )


def main():

    parser = argparse.ArgumentParser(
        description="Test recommender_v2 retrieval models."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["bm25", "cf"],
        choices=[
            "bm25",
            "bge",
            "bert",
            "cf",
            "metadata",
            "lyrics",
            "attributes",
            "clap",
            "all",
            "item_cf",
            "item_clap",
            "item_metadata",
            "artist",
            "album",
        ],
        help=(
            "Retrieval models to use. "
            "Default: bm25 cf"
        ),
    )

    parser.add_argument(
        "--num-tests",
        type=int,
        default=1,
        help="Number of sessions to test. Default: 1",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting training-session index. Default: 0",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------

    if args.num_tests < 1:
        raise ValueError(
            "--num-tests must be >= 1"
        )

    if args.start < 0:
        raise ValueError(
            "--start must be >= 0"
        )

    # ---------------------------------------------------------
    # Load challenge dataset
    # ---------------------------------------------------------

    print("Loading challenge dataset...")

    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]

    end = args.start + args.num_tests

    if end > len(train):

        raise ValueError(
            f"Requested sessions "
            f"{args.start}..{end - 1}, "
            f"but only {len(train)} are available."
        )

    sessions = train.select(
        range(args.start, end)
    )

    # ---------------------------------------------------------
    # Build recommender once
    # ---------------------------------------------------------

    recommender = build_selected_recommender(
        set(args.models)
    )

    # ---------------------------------------------------------
    # Run tests
    # ---------------------------------------------------------

    for session in sessions:

        run_test(
            recommender,
            session,
        )


if __name__ == "__main__":
    main()