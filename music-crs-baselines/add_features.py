"""
Add the newer embedding/pool features to an EXISTING dataset CSV, without
re-running the planner or any retriever.

The new features are functions of (candidate, the session's earlier tracks,
user_id, the pool), all of which are already in the CSV or the public dataset,
so a 600-session dataset can be upgraded in minutes.

    python add_features.py --input data\\v2\\v2_s0.csv --output data\\v5\\v5_s0.csv

Run it once per file, then train with --train on the new files. Note: it cannot
apply --user-dropout (that changes which candidates the retrievers return);
dropout needs a real rebuild with build_dataset.py.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default=None, help="default: read from <input>.meta.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset
    from recommender_v2.features import FeatureBuilder, EXTRA_FEATURES
    from recommender_v2.training.build_dataset import DATASET_NAME, build_retrievers
    from recommender_v2.training.metadata import load_track_metadata
    from recommender_v2.training.turns import get_turn_context

    inp, out = Path(args.input), Path(args.output)
    meta_in = Path(str(inp) + ".meta.json")
    meta = json.loads(meta_in.read_text()) if meta_in.exists() else {}
    split = args.split or meta.get("split") or "train"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    print(f"{inp}: {len(df)} rows, {df['query_id'].nunique()} queries")
    if all(c in df.columns for c in EXTRA_FEATURES):
        raise SystemExit("Input already has the extra features.")

    # Only the retrievers that hold embeddings are needed (no BGE, no planner).
    retrievers = build_retrievers(["cf", "item_cf", "item_clap", "item_metadata"],
                                  device=args.device)
    metadata = load_track_metadata()
    fb = FeatureBuilder(track_metadata=metadata)
    fb.attach_embeddings(retrievers)

    wanted = set(df["session_id"].unique())
    sessions = {s["session_id"]: s for s in load_dataset(DATASET_NAME)[split]
                if s["session_id"] in wanted}
    missing = wanted - set(sessions)
    if missing:
        raise SystemExit(f"{len(missing)} sessions not found in split '{split}' "
                         f"(wrong --split?).")

    new_cols = {c: [float("nan")] * len(df) for c in EXTRA_FEATURES}
    pos = {q: idx.tolist() for q, idx in df.groupby("query_id").indices.items()}
    pos = {q: list(v) for q, v in pos.items()}

    for n, (qid, rows_idx) in enumerate(pos.items(), 1):
        sub = df.iloc[rows_idx]
        sid, turn = sub["session_id"].iat[0], int(sub["turn_number"].iat[0])
        ctx = get_turn_context(sessions[sid]["conversations"], turn)
        uid = sub["user_id"].iat[0]
        uid = None if (pd.isna(uid) or uid == "") else uid
        rows = [{"track_id": t, "candidate_popularity": p}
                for t, p in zip(sub["track_id"], sub["candidate_popularity"])]
        fb.add_extra_features(rows, user_id=uid, anchor_track_ids=ctx.anchor_track_ids,
                              metadata_lookup=metadata)
        for c in EXTRA_FEATURES:
            vals = [r[c] for r in rows]
            for i, v in zip(rows_idx, vals):
                new_cols[c][i] = v
        if n % 100 == 0:
            print(f"  {n}/{len(pos)} queries")

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df.to_csv(out, index=False)

    # keep the side files so train_ranker can read the schema
    for suffix in (".queries.csv",):
        src = Path(str(inp) + suffix)
        if src.exists():
            Path(str(out) + suffix).write_bytes(src.read_bytes())
    if meta:
        meta["feature_names"] = list(meta.get("feature_names", [])) + list(EXTRA_FEATURES)
        meta["upgraded_from"] = str(inp)
        Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}  (+{len(EXTRA_FEATURES)} features)")


if __name__ == "__main__":
    main()
