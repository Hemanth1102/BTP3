"""
Train the LightGBM LambdaRank re-ranker on a dataset produced by
build_dataset.py.

    python -m recommender_v2.training.train_ranker --train training_data_train.csv

What this does differently from a plain `lgb.train(...)` call:

* Query groups are (session_id, turn) pairs ("query_id"), not sessions.
* The train/validation split is by *session*, so turns of one conversation never
  end up on both sides.
* Groups without any positive row (gold track not retrieved) are dropped from
  training - they carry no ranking signal - but are KEPT when scoring
  validation, because a miss is a 0 in the official nDCG@20.
  (LightGBM's built-in `ndcg` gives such queries a score of 1.0, which would
  inflate the number; it is only used for early stopping, on queries that have
  a positive.)
* The reported validation metric is the challenge-style nDCG@20 over ALL
  validation queries, next to simple baselines (RRF, each single retriever).
* Model + feature schema + config are saved together.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

META_COLUMNS = [
    "query_id",
    "session_id",
    "user_id",
    "turn_number",
    "track_id",
    "label",
    "query_has_positive",
]

TRAINING_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path):

    df = pd.read_csv(path)

    meta_path = Path(str(path) + ".meta.json")
    if meta_path.exists():
        feature_names = json.loads(meta_path.read_text())["feature_names"]
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            sys.exit(f"{path} is missing feature columns: {missing[:5]}")
    else:
        feature_names = [c for c in df.columns if c not in META_COLUMNS]
        print(f"No {meta_path.name}; inferring {len(feature_names)} features.")

    # Rows of one query must be contiguous for LightGBM's `group`.
    n_runs = int((df["query_id"] != df["query_id"].shift()).sum())
    if n_runs != df["query_id"].nunique():
        df = df.sort_values("query_id", kind="stable").reset_index(drop=True)

    return df, feature_names


def load_datasets(paths):
    """
    Load one or more dataset CSVs (e.g. chunks built on different machines) as
    one table. All files must share the feature schema and must not contain the
    same query twice.
    """
    paths = [Path(p) for p in paths]

    if len(paths) == 1:
        return load_dataset(paths[0])

    frames, feature_names = [], None

    for path in paths:
        df, feats = load_dataset(path)
        if feature_names is None:
            feature_names = feats
        elif feats != feature_names:
            sys.exit(f"{path} has a different feature schema than {paths[0]}.")
        print(f"  {path.name}: {df['query_id'].nunique()} queries, "
              f"{df['session_id'].nunique()} sessions")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    seen = {}
    for path, f in zip(paths, frames):
        for q in f["query_id"].unique():
            if q in seen:
                sys.exit(f"Query {q} appears in both {seen[q].name} and {path.name}; "
                         "chunks must cover disjoint sessions.")
            seen[q] = path

    # Each query's rows are contiguous within its file, and files are
    # concatenated whole, so groups stay contiguous.
    return df, feature_names


def split_by_session(df, val_fraction, seed):

    sessions = np.array(sorted(df["session_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(sessions)

    n_val = max(1, int(round(len(sessions) * val_fraction)))
    val_sessions = set(sessions[:n_val])

    is_val = df["session_id"].isin(val_sessions)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def group_sizes(df):
    return df.groupby("query_id", sort=False).size().to_numpy()


# ---------------------------------------------------------------------------
# Evaluation (matches music-crs-evaluator: single gold track per query,
# nDCG = 1/log2(rank+1) if the gold is in the top k, else 0)
# ---------------------------------------------------------------------------

def evaluate_scores(df, scores, k=20):
    """
    df must hold ALL candidate rows of every evaluated query (including queries
    whose gold track is not in the pool). Ties are broken by row order, which is
    candidate-union order - the same rule RecommenderV2.recommend() uses.
    """
    tmp = pd.DataFrame({
        "query_id": df["query_id"].to_numpy(),
        "label": df["label"].to_numpy(),
        "score": np.asarray(scores, dtype=np.float64),
    })

    tmp["pos"] = tmp.groupby("query_id", sort=False)["score"].rank(
        method="first", ascending=False
    )

    gold = tmp[tmp["label"] == 1]
    n_queries = tmp["query_id"].nunique()

    pos = gold["pos"].to_numpy()
    in_top = pos <= k

    ndcg = float(np.sum(1.0 / np.log2(pos[in_top] + 1.0)) / n_queries)

    return {
        f"ndcg@{k}": ndcg,
        "ndcg@10": float(
            np.sum(1.0 / np.log2(pos[pos <= 10] + 1.0)) / n_queries
        ),
        f"hit@{k}": float(in_top.sum() / n_queries),
        "hit@1": float((pos <= 1).sum() / n_queries),
        "pool_recall": float(len(gold) / n_queries),
        "queries": int(n_queries),
    }


def query_ndcg(df, scores, k=20):
    """nDCG@k per query (Series indexed by query_id, misses = 0)."""
    tmp = pd.DataFrame({
        "query_id": df["query_id"].to_numpy(),
        "label": df["label"].to_numpy(),
        "score": np.asarray(scores, dtype=np.float64),
    })
    tmp["pos"] = tmp.groupby("query_id", sort=False)["score"].rank(
        method="first", ascending=False
    )
    out = pd.Series(0.0, index=pd.Index(tmp["query_id"].unique(), name="query_id"))
    gold = tmp[(tmp["label"] == 1) & (tmp["pos"] <= k)]
    out.loc[gold["query_id"].to_numpy()] = 1.0 / np.log2(gold["pos"].to_numpy() + 1.0)
    return out


def session_bootstrap(per_query, sessions, n_boot=2000, seed=0):
    """
    Mean and 95% CI of a per-query metric, resampling whole sessions (turns of
    one conversation are correlated, so queries are not independent).
    """
    d = pd.DataFrame({"v": per_query.to_numpy(), "s": sessions.to_numpy()})
    g = d.groupby("s")["v"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(g), size=(n_boot, len(g)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(sums.sum() / counts.sum()), np.percentile(means, [2.5, 97.5])


def run_cv(df, feature_names, params, folds, rounds, seed):
    import lightgbm as lgb

    sessions = np.array(sorted(df["session_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(sessions)
    fold_sessions = np.array_split(sessions, folds)

    oof = np.full(len(df), np.nan)

    for i, fs in enumerate(fold_sessions):
        is_val = df["session_id"].isin(set(fs)).to_numpy()
        tr = df[(~is_val) & (df["query_has_positive"] == 1).to_numpy()]
        model = lgb.train(
            params,
            lgb.Dataset(tr[feature_names], label=tr["label"],
                        group=group_sizes(tr), feature_name=feature_names),
            num_boost_round=rounds,
        )
        oof[is_val] = model.predict(df.loc[is_val, feature_names])
        print(f"  fold {i + 1}/{folds}: {len(fs)} sessions held out")

    q_sessions = df.drop_duplicates("query_id").set_index("query_id")["session_id"]

    scorers = {"lightgbm": oof, "rrf_sum": df["rrf_sum"].to_numpy()}
    for name in sorted({c[:-5] for c in feature_names if c.endswith("_rank")}
                       - {"best", "worst", "mean", "median", "rrf"}):
        if df[f"{name}_present"].sum() > 0:
            scorers[f"only_{name}"] = retriever_baseline_scores(df, name)

    per_q = {n: query_ndcg(df, sc) for n, sc in scorers.items()}
    sess = q_sessions.loc[per_q["lightgbm"].index]

    print(f"\nCross-validated nDCG@20 over {len(sess)} queries "
          f"({sess.nunique()} sessions), 95% CI by session bootstrap:")
    print(f"  {'ranker':<22}{'nDCG@20':>9}   95% CI")
    for name, v in sorted(per_q.items(), key=lambda kv: -kv[1].mean()):
        m, (lo, hi) = session_bootstrap(v, sess)
        print(f"  {name:<22}{m:>9.4f}   [{lo:.4f}, {hi:.4f}]")

    for ref in ("rrf_sum", "only_item_metadata"):
        if ref in per_q:
            diff = per_q["lightgbm"] - per_q[ref]
            m, (lo, hi) = session_bootstrap(diff, sess)
            verdict = "significant" if lo > 0 else "NOT significant"
            print(f"  lightgbm - {ref}: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")


def retriever_baseline_scores(df, retriever):
    """Order by one retriever's rank; candidates it did not return go last."""
    rank = df[f"{retriever}_rank"].to_numpy(dtype=np.float64)
    return np.where(np.isnan(rank), -1e9, -rank)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):

    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", default=["training_data_train.csv"],
                   help="One or more dataset CSVs (disjoint sessions).")
    p.add_argument("--val", default=None,
                   help="Optional separate validation CSV. Default: split --train by session.")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(TRAINING_DIR))
    p.add_argument("--num-rounds", type=int, default=2000)
    p.add_argument("--early-stopping", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--min-data-in-leaf", type=int, default=50)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--cv-folds", type=int, default=0,
                   help="Session-grouped cross-validation over ALL queries "
                        "(no model is saved). Use this to judge whether the "
                        "reranker really beats the baselines.")
    p.add_argument("--cv-rounds", type=int, default=100,
                   help="Boosting rounds per CV fold (fixed, no early stopping).")
    p.add_argument("--final-rounds", type=int, default=0,
                   help="Train on ALL data for exactly this many rounds and save "
                        "the model (no validation split).")
    args = p.parse_args(argv)

    import lightgbm as lgb

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    df, feature_names = load_datasets(args.train)

    if args.val:
        val_df, val_features = load_dataset(Path(args.val))
        if val_features != feature_names:
            sys.exit("Train and validation files have different feature schemas.")
        train_df = df
        overlap = set(train_df["session_id"]) & set(val_df["session_id"])
        if overlap:
            sys.exit(f"{len(overlap)} sessions appear in both train and validation.")
    else:
        train_df, val_df = split_by_session(df, args.val_fraction, args.seed)

    def describe(name, d):
        q = d.groupby("query_id").label.max()
        print(f"  {name:<11} rows={len(d):>9}  queries={q.size:>6}  "
              f"sessions={d['session_id'].nunique():>5}  "
              f"queries with positive={int(q.sum()):>6} ({q.mean():.1%})")

    describe("train", train_df)
    describe("validation", val_df)
    print(f"  features: {len(feature_names)}")

    # ---- training set: only queries that contain the gold track ----
    train_fit = train_df[train_df["query_has_positive"] == 1].reset_index(drop=True)
    val_fit = val_df[val_df["query_has_positive"] == 1].reset_index(drop=True)

    if train_fit.empty or val_fit.empty:
        sys.exit("No query with a positive candidate in train or validation.")

    dtrain = lgb.Dataset(
        train_fit[feature_names],
        label=train_fit["label"],
        group=group_sizes(train_fit),
        feature_name=feature_names,
    )
    dval = lgb.Dataset(
        val_fit[feature_names],
        label=val_fit["label"],
        group=group_sizes(val_fit),
        reference=dtrain,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        # LightGBM sorts eval positions ascending, so listing several would make
        # early stopping follow nDCG@1. Track only nDCG@20 here; the other
        # metrics are computed afterwards by evaluate_scores().
        "ndcg_eval_at": [20],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "lambdarank_truncation_level": 40,
        "seed": args.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }

    if args.cv_folds:
        print(f"\n{args.cv_folds}-fold session-grouped CV, {args.cv_rounds} rounds/fold")
        cv_params = {k: v for k, v in params.items() if k not in ("metric", "ndcg_eval_at")}
        run_cv(df, feature_names, cv_params, args.cv_folds, args.cv_rounds, args.seed)
        return

    if args.final_rounds:
        print(f"\nFinal fit on all data: {args.final_rounds} rounds")
        full = df[df["query_has_positive"] == 1].reset_index(drop=True)
        fit_params = {k: v for k, v in params.items() if k not in ("metric", "ndcg_eval_at")}
        model = lgb.train(
            fit_params,
            lgb.Dataset(full[feature_names], label=full["label"],
                        group=group_sizes(full), feature_name=feature_names),
            num_boost_round=args.final_rounds,
        )
        model.save_model(str(out_dir / "ranker.txt"))
        (out_dir / "ranker_features.json").write_text(json.dumps(feature_names, indent=2))
        (out_dir / "ranker_config.json").write_text(json.dumps({
            "train_file": [str(t) for t in args.train], "seed": args.seed, "params": fit_params,
            "rounds": args.final_rounds, "mode": "final fit on all data",
            "train_queries": int(full["query_id"].nunique()),
            "lightgbm_version": lgb.__version__,
        }, indent=2))
        print(f"Saved model: {out_dir / 'ranker.txt'}")
        return

    print("Training LightGBM LambdaRank...")

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.num_rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(args.early_stopping, first_metric_only=True),
            lgb.log_evaluation(25),
        ],
    )

    best_iter = model.best_iteration or model.current_iteration()
    print(f"Best iteration: {best_iter}")

    # ---- official-style validation over ALL validation queries ----
    X_val = val_df[feature_names]

    results = {
        "lightgbm": evaluate_scores(
            val_df, model.predict(X_val, num_iteration=best_iter)
        ),
        "rrf_sum": evaluate_scores(val_df, val_df["rrf_sum"].to_numpy()),
    }

    for name in sorted({c[:-5] for c in feature_names if c.endswith("_rank")}
                       - {"best", "worst", "mean", "median", "rrf"}):
        if f"{name}_present" in val_df and val_df[f"{name}_present"].sum() > 0:
            results[f"only_{name}"] = evaluate_scores(
                val_df, retriever_baseline_scores(val_df, name)
            )

    print("\nValidation (all queries, misses count as 0):")
    print(f"  {'ranker':<22}{'nDCG@20':>9}{'nDCG@10':>9}{'Hit@20':>9}{'Hit@1':>8}")
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["ndcg@20"]):
        print(f"  {name:<22}{m['ndcg@20']:>9.4f}{m['ndcg@10']:>9.4f}"
              f"{m['hit@20']:>9.4f}{m['hit@1']:>8.4f}")
    print(f"  pool recall (upper bound for Hit@20): "
          f"{results['lightgbm']['pool_recall']:.4f}")

    # ---- feature importance ----
    gain = model.feature_importance(importance_type="gain")
    top = sorted(zip(feature_names, gain), key=lambda x: -x[1])[:15]
    print("\nTop features by gain:")
    for n, g in top:
        print(f"  {n:<28}{g:>14.1f}")

    # ---- persist model + schema + config together ----
    model_path = out_dir / "ranker.txt"
    model.save_model(str(model_path), num_iteration=best_iter)

    (out_dir / "ranker_features.json").write_text(json.dumps(feature_names, indent=2))

    config = {
        "train_file": [str(t) for t in args.train],
        "val_file": str(args.val) if args.val else f"split of train, fraction={args.val_fraction}",
        "seed": args.seed,
        "params": params,
        "best_iteration": int(best_iter),
        "lightgbm_version": lgb.__version__,
        "train_queries": int(train_fit["query_id"].nunique()),
        "val_queries_all": int(val_df["query_id"].nunique()),
        "validation_metrics": results,
    }
    (out_dir / "ranker_config.json").write_text(json.dumps(config, indent=2))

    print(f"\nSaved model    : {model_path}")
    print(f"Saved schema   : {out_dir / 'ranker_features.json'}")
    print(f"Saved config   : {out_dir / 'ranker_config.json'}")


if __name__ == "__main__":
    main()
