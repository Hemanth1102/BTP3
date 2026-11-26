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
`predicted_response` is empty by default. Use --generate-responses with a ranker
or --responses-only --predictions FILE to add replies to existing rankings.
"""

import argparse


DEFAULT_DATASET = "talkpl-ai/TalkPlayData-Challenge-Blind-B"


def predict_rows(rows, recommender, limit=0, log=print, response_generator=None, response_stats=None):
    """rows: iterable of dicts with session_id, user_id, conversations."""
    if response_generator is not None:
        from recommender_v2.response.predictions import generate_for_prediction

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

        prediction = {
            "session_id": item["session_id"],
            "user_id": item.get("user_id"),
            "turn_number": turn,
            "predicted_track_ids": list(ids),
            "predicted_response": "",
        }
        if response_generator is not None:
            reply = generate_for_prediction(response_generator, prediction, conv)
            prediction["predicted_response"] = reply.text
            if response_stats is not None:
                response_stats.append(reply)
        out.append(prediction)
        log(f"[{n + 1}] session {item['session_id']} turn {turn}: "
            f"{len(anchors)} anchors, user={'yes' if user_id else 'no'}, {len(ids)} tracks")
    return out


def fill_responses(predictions, rows, generator, response_stats):
    from recommender_v2.response.predictions import generate_for_prediction, prediction_id
    from recommender_v2.training.turns import session_query_id

    contexts = {}
    for row in rows:
        conversation = row["conversations"]
        target = conversation[-1]
        if target["role"] != "user":
            raise ValueError("Blind conversations must end at the current user message.")
        key = session_query_id(row["session_id"], int(target["turn_number"]))
        if key in contexts:
            raise ValueError(f"Duplicate blind context for {key}.")
        contexts[key] = conversation

    missing = [prediction_id(row) for row in predictions if prediction_id(row) not in contexts]
    if missing:
        raise ValueError(f"Missing blind contexts for {len(missing)} predictions (first: {missing[0]}).")
    results = []
    for prediction in predictions:
        reply = generate_for_prediction(generator, prediction, contexts[prediction_id(prediction)])
        response_stats.append(reply)
        results.append({**prediction, "predicted_response": reply.text})
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ranker", help="ranker.txt (ranker_features.json must sit next to it)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--planner-model", default="qwen3:8b")
    ap.add_argument("--candidate-k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    ap.add_argument("--generate-responses", action="store_true")
    ap.add_argument("--response-model", help="Defaults to --planner-model.")
    ap.add_argument("--num-demos", type=int, default=1, help="Number of training-split response demonstrations.")
    ap.add_argument("--responses-only", action="store_true", help="Preserve existing ranked track lists.")
    ap.add_argument("--predictions", help="Existing predictions as JSONL or a JSON array.")
    args = ap.parse_args(argv)
    if args.responses_only != bool(args.predictions):
        ap.error("--responses-only and --predictions are required together")
    if not args.responses_only and not args.ranker:
        ap.error("--ranker is required when computing recommendations")
    if args.limit < 0 or args.num_demos < 0:
        ap.error("--limit and --num-demos must be nonnegative")

    from datasets import load_dataset
    from recommender_v2.response.predictions import read_predictions, timing_summary, write_predictions
    from recommender_v2.training.metadata import load_track_metadata

    rows = load_dataset(args.dataset)[args.split]
    print(f"{args.dataset}[{args.split}]: {len(rows)} rows")
    metadata = load_track_metadata()
    generator = None
    stats = []
    if args.generate_responses or args.responses_only:
        from recommender_v2.response.generator import ResponseGenerator

        generator = ResponseGenerator(metadata, model=args.response_model or args.planner_model, num_demos=args.num_demos)

    if args.responses_only:
        predictions = read_predictions(args.predictions)
        results = fill_responses(predictions[:args.limit or None], rows, generator, stats)
    else:
        from recommender_v2 import RecommenderV2
        from recommender_v2.planner import QueryPlanner
        from recommender_v2.planner.ollama import OllamaQwenPlanner
        from recommender_v2.ranker import LightGBMRanker
        from recommender_v2.training.build_dataset import DEFAULT_MODELS, build_retrievers

        recommender = RecommenderV2(
            retrievers=build_retrievers(args.models or DEFAULT_MODELS, device=args.device),
            planner=QueryPlanner(llm=OllamaQwenPlanner(args.planner_model), mode="llm"),
            candidate_k=args.candidate_k,
            final_k=20,
            ranker_path=args.ranker,
            track_metadata=metadata,
        )
        if not isinstance(recommender.ranker, LightGBMRanker):
            raise SystemExit(f"No LightGBM model loaded from {args.ranker}.")
        results = predict_rows(rows, recommender, limit=args.limit, response_generator=generator, response_stats=stats)

    if generator is not None:
        median_sec, max_sec = timing_summary(stats)
        print(f"Response generation: {median_sec:.3f} s median, {max_sec:.3f} s maximum ({len(stats)} queries)")
    write_predictions(args.output, results, jsonl=args.responses_only and args.output.lower().endswith(".jsonl"))
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
