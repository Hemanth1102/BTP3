"""Interactive console chat for the Music CRS baseline."""

import argparse
import torch
from omegaconf import OmegaConf

from mcrs import load_crs_baseline


def _resolve_device(config_device: str, override_device: str | None) -> str:
    if override_device:
        return override_device
    if config_device == "cuda" and not torch.cuda.is_available():
        print("Config asks for CUDA, but CUDA is not available. Falling back to CPU.")
        return "cpu"
    return config_device


def _load_system(tid: str, device: str | None):
    config = OmegaConf.load(f"config/{tid}.yaml")
    resolved_device = _resolve_device(config.device, device)
    dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32

    return load_crs_baseline(
        lm_type=config.lm_type,
        retrieval_type=config.retrieval_type,
        item_db_name=config.item_db_name,
        user_db_name=config.user_db_name,
        track_split_types=config.track_split_types,
        user_split_types=config.user_split_types,
        corpus_types=config.corpus_types,
        cache_dir=config.cache_dir,
        device=resolved_device,
        attn_implementation=config.attn_implementation,
        dtype=dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the Music CRS baseline.")
    parser.add_argument(
        "--tid",
        default="llama1b_bm25_devset",
        help="Config name from config/<tid>.yaml.",
    )
    parser.add_argument(
        "--user_id",
        default=None,
        help="Optional user_id for profile personalization.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Override the device from the config.",
    )
    parser.add_argument(
        "--show_topk",
        type=int,
        default=5,
        help="How many retrieved tracks to print after each response.",
    )
    args = parser.parse_args()

    print(f"Loading Music CRS config: {args.tid}")
    music_crs = _load_system(args.tid, args.device)

    print("\nStart chatting. Type 'exit', 'quit', or 'reset'.")
    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break
        if user_query.lower() == "reset":
            music_crs._reset_session_memory()
            print("Conversation reset.")
            continue
        if not user_query:
            continue

        result = music_crs.chat(user_query, user_id=args.user_id)
        print(f"\nAssistant: {result['response']}")
        print("\nRecommended tracks:")
        for rank, track_id in enumerate(result["retrieval_items"][: args.show_topk], start=1):
            print(f"{rank}. {music_crs.item_db.id_to_metadata(track_id)}")


if __name__ == "__main__":
    main()
