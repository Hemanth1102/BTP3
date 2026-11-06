"""
Show the whole process for ONE query, step by step (for a live demo).

    python demo_one_query.py --ranker models\\full\\ranker.txt --session 5 --turn 3 \
        --models bm25 bge cf item_cf item_clap item_metadata artist album --device cpu

--split test uses the 1,000-session dev set (unseen by the model); --split train
uses a training session (the model may have seen it, fine for a demo).
Prints: conversation so far -> planner output -> top candidates of each retriever
-> feature values of the best candidates -> final top 20 -> where the gold track is.
"""
import argparse


def label(meta, tid):
    m = meta.get(tid)
    return f"{m['track_name']} - {m['artist_name']}" if m else tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranker", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--session", type=int, default=0, help="index of the session in the split")
    ap.add_argument("--turn", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--planner-model", default="qwen3:8b")
    ap.add_argument("--show", type=int, default=5, help="rows to show per retriever / feature table")
    args = ap.parse_args()

    from datasets import load_dataset
    from recommender_v2 import RecommenderV2
    from recommender_v2.planner import QueryPlanner
    from recommender_v2.planner.ollama import OllamaQwenPlanner
    from recommender_v2.training.build_dataset import DATASET_NAME, DEFAULT_MODELS, build_retrievers
    from recommender_v2.training.metadata import ITEM_DATASET, load_track_metadata
    from recommender_v2.training.turns import get_turn_context

    names = {r["track_id"]: {"track_name": r["track_name"],
                             "artist_name": r["artist_name"]}
             for r in load_dataset(ITEM_DATASET, split="all_tracks")}

    session = load_dataset(DATASET_NAME)[args.split][args.session]
    ctx = get_turn_context(session["conversations"], args.turn)
    user_id = session["user_id"] or None

    rec = RecommenderV2(
        retrievers=build_retrievers(args.models or DEFAULT_MODELS, device=args.device),
        planner=QueryPlanner(llm=OllamaQwenPlanner(args.planner_model), mode="llm"),
        candidate_k=100, final_k=20, ranker_path=args.ranker,
        track_metadata=load_track_metadata(),
    )

    print("\n=== 1. CONVERSATION SO FAR (what the system may see) ===")
    for m in ctx.history:
        txt = label(names, m["content"]) if m["role"] == "music" else m["content"]
        print(f"  [turn {m['turn_number']}] {m['role']:<9}: {txt[:150]}")
    print(f"  user id: {user_id}   anchors (earlier tracks): {len(ctx.anchor_track_ids)}")

    state, results = rec.retrieve_for_ranking(
        query=ctx.latest_user_message, user_id=user_id,
        conversation_history=ctx.history, anchor_track_ids=ctx.anchor_track_ids)

    print("\n=== 2. PLANNER (LLM) OUTPUT ===")
    for k, v in vars(state).items():
        if v not in (None, "", [], {}):
            print(f"  {k}: {v}")

    print("\n=== 3. WHAT EACH RETRIEVER PROPOSES (top few) ===")
    for name, res in results.items():
        print(f"  {name}: {len(res)} candidates")
        for r in res[:args.show]:
            print(f"     #{int(r.rank):<3} {label(names, r.track_id)}")

    feats = rec.build_features(results, user_id=user_id,
                               anchor_track_ids=ctx.anchor_track_ids,
                               query_state=state, turn_index=args.turn)
    scores = rec.ranker.predict(feats)
    order = sorted(range(len(feats)), key=lambda i: -scores[i])
    pool = {f["track_id"] for f in feats}

    print(f"\n=== 4. CANDIDATE POOL: {len(feats)} unique tracks; gold in pool: {ctx.ground_truth in pool} ===")
    show = ["num_sources", "rrf_sum", "same_artist_as_anchor", "same_artist_as_last_anchor",
            "artist_match", "candidate_popularity"]
    print("  top candidates by LightGBM, with a few of the 106 features:")
    print("   score   " + "  ".join(f"{s[:14]:>14}" for s in show) + "   track")
    for i in order[:args.show]:
        f = feats[i]
        vals = "  ".join(f"{(f.get(s) if f.get(s) == f.get(s) else float('nan')):>14.3f}" for s in show)
        print(f"  {scores[i]:7.3f} {vals}   {label(names, f['track_id'])}")

    top20 = [feats[i]["track_id"] for i in order[:20]]
    print("\n=== 5. FINAL TOP 20 ===")
    for n, tid in enumerate(top20, 1):
        print(f"  {n:>2}. {label(names, tid)}{'   <-- GOLD' if tid == ctx.ground_truth else ''}")

    print("\n=== 6. RESULT ===")
    print(f"  gold track: {label(names, ctx.ground_truth)}")
    if ctx.ground_truth in top20:
        import math
        rank = top20.index(ctx.ground_truth) + 1
        print(f"  found at rank {rank}  ->  nDCG@20 = 1/log2({rank}+1) = {1/math.log2(rank+1):.3f}")
    else:
        print("  not in top 20  ->  nDCG@20 = 0")


if __name__ == "__main__":
    main()