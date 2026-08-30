"""Interactive retrieval-only chat for music recommendations.

This mode avoids the LLM and Torch, so it is useful on Windows machines where
PyTorch DLL loading is not working yet.
"""

import argparse

from datasets import concatenate_datasets, load_dataset
from omegaconf import OmegaConf

from mcrs.retrieval_modules.bm25 import BM25_MODEL


def _format_value(value):
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item is not None)
    return str(value)


def _track_line(metadata):
    parts = []
    for key in ["track_name", "artist_name", "album_name", "release_date"]:
        if key in metadata and metadata[key] is not None:
            parts.append(f"{key}: {_format_value(metadata[key])}")
    if "track_id" in metadata:
        parts.append(f"track_id: {metadata['track_id']}")
    return " | ".join(parts)


def _track_title(metadata):
    track_name = _format_value(metadata.get("track_name", "Unknown track"))
    artist_name = _format_value(metadata.get("artist_name", "Unknown artist"))
    return f"{track_name} by {artist_name}"


def _follow_up_question(user_query):
    query = user_query.lower()
    if any(word in query for word in ["relax", "calm", "sleep", "study", "focus", "late night"]):
        return "Do you want this to stay soft and focused, or should I make it a little more upbeat?"
    if any(word in query for word in ["party", "dance", "gym", "workout", "energy", "hype"]):
        return "Should I push this more toward dance-pop, hip-hop, electronic, or rock energy?"
    if any(word in query for word in ["sad", "heartbreak", "melancholy", "emotional", "cry"]):
        return "Do you want something deeply emotional, or more comforting and hopeful?"
    if any(word in query for word in ["artist", "like", "similar", "sounds like"]):
        return "Should I stay close to that artist's sound, or branch into nearby styles?"
    return "Do you want me to narrow the next recommendations by mood, genre, artist, decade, or activity?"


def _response_text(user_query, recommended_metadata):
    top_tracks = [_track_title(metadata) for metadata in recommended_metadata[:3]]
    if not top_tracks:
        return "I could not find a strong match yet. Tell me an artist, genre, mood, or activity and I will try again."

    if len(top_tracks) == 1:
        recommendation_text = top_tracks[0]
    elif len(top_tracks) == 2:
        recommendation_text = f"{top_tracks[0]} and {top_tracks[1]}"
    else:
        recommendation_text = f"{top_tracks[0]}, {top_tracks[1]}, and {top_tracks[2]}"

    return (
        f"I found a few tracks that fit: {recommendation_text}. "
        f"{_follow_up_question(user_query)}"
    )


def _load_metadata_dict(dataset_name, split_types):
    dataset = load_dataset(dataset_name)
    merged = concatenate_datasets([dataset[split_type] for split_type in split_types])
    return {item["track_id"]: item for item in merged}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with BM25 music retrieval only.")
    parser.add_argument(
        "--tid",
        default="llama1b_bm25_devset",
        help="Config name from config/<tid>.yaml.",
    )
    parser.add_argument(
        "--show_topk",
        type=int,
        default=10,
        help="How many recommended tracks to print after each message.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(f"config/{args.tid}.yaml")
    if config.retrieval_type != "bm25":
        raise ValueError("interactive_retrieval_chat.py currently supports BM25 configs only.")

    print(f"Loading BM25 recommender config: {args.tid}")
    retriever = BM25_MODEL(
        config.item_db_name,
        config.track_split_types,
        config.corpus_types,
        config.cache_dir,
    )
    metadata_by_id = _load_metadata_dict(config.item_db_name, config.track_split_types)
    history = []

    print("\nStart chatting. Type 'exit', 'quit', or 'reset'.")
    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break
        if user_query.lower() == "reset":
            history = []
            print("Conversation reset.")
            continue
        if not user_query:
            continue

        history.append(f"user: {user_query}")
        retrieval_query = "\n".join(history)
        track_ids = retriever.text_to_item_retrieval(retrieval_query, topk=args.show_topk)
        recommended_metadata = [metadata_by_id[track_id] for track_id in track_ids]

        print(f"\nAssistant: {_response_text(user_query, recommended_metadata)}")
        print("\nRecommended tracks:")
        for rank, metadata in enumerate(recommended_metadata, start=1):
            print(f"{rank}. {_track_line(metadata)}")


if __name__ == "__main__":
    main()
