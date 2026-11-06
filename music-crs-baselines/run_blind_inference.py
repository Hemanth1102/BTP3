"""
Run the trained recommender on a blind set and write the submission JSON.

    python run_blind_inference.py --ranker models/final/ranker.txt \
        --output exp/inference/blindset_B/ours.json

Each blind row has `conversations` whose LAST entry is the user message at the
target turn; everything before it is history (user / assistant / music).
Targets are hidden, so this only produces predictions; scoring happens on the
challenge's submission site.

Output format (one object per row):
    {"session_id", "user_id", "turn_number", "predicted_track_ids", "predicted_response"}
`predicted_response` is empty until a response generator is added.
"""

import argparse
import json
import os


DEFAULT_DATASET = "talkpl-ai/TalkPlayData-Challenge-Blind-B"


def predict_rows(rows, recommender, limit=0, log=print):
    """rows: iterable of dicts with session_id, user_id, conversations."""
    out = []
    for n, item in enumerate(rows):
        if limit and n >= limit:
            break

        conv = list(item["conversations"])
        target = conv[-1]
        history = conv[:-1]

        anchors = [m["content"] for m in history if m["role"] == "music"]
        user_id = item.get("user_id") or None          # masked / missing -> None
        turn = int(target["turn_number"])

        ids = recommender.recommend(
            query=target["content"],                   # latest user message (same as training)
            user_id=user_id,
            # Same as training (turns.get_turn_context): the history shown to the
            # planner ends with the target turn's user message.
            conversation_history=conv,
            anchor_track_ids=anchors,
            turn_index=turn,
        )

        out.append({
            "session_id": item["session_id"],
            "user_id": item.get("user_id"),
            "turn_number": turn,
            "predicted_track_ids": list(ids),
            "predicted_response": "",
        })
        log(f"[{n + 1}] session {item['session_id']} turn {turn}: "
            f"{len(anchors)} anchors, user={'yes' if user_id else 'no'}, {len(ids)} tracks")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ranker", required=True, help="ranker.txt (ranker_features.json must sit next to it)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--planner-model", default="qwen3:8b")
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    args = ap.parse_args()

    from datasets import load_dataset
    from recommender_v2 import RecommenderV2
    from recommender_v2.planner import QueryPlanner
    from recommender_v2.planner.ollama import OllamaQwenPlanner
    from recommender_v2.training.build_dataset import DEFAULT_MODELS, build_retrievers
    from recommender_v2.training.metadata import load_track_metadata

    rows = load_dataset(args.dataset)[args.split]
    print(f"{args.dataset}[{args.split}]: {len(rows)} rows")

    recommender = RecommenderV2(
        retrievers=build_retrievers(args.models or DEFAULT_MODELS, device=args.device),
        planner=QueryPlanner(llm=OllamaQwenPlanner(args.planner_model), mode="llm"),
        candidate_k=args.candidate_k,
        final_k=20,
        ranker_path=args.ranker,
        track_metadata=load_track_metadata(),
    )

    # A missing model file would silently fall back to RRF; refuse instead.
    from recommender_v2.ranker import LightGBMRanker
    if not isinstance(recommender.ranker, LightGBMRanker):
        raise SystemExit(f"No LightGBM model loaded from {args.ranker}.")

    results = predict_rows(rows, recommender, limit=args.limit)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()