import argparse
import json
import math
import os
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

import bm25s
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"


def norm_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value if x is not None)
    value = str(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def first_value(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def read_parquet(path, columns=None, limit=None):
    table = pq.read_table(path, columns=columns)
    if limit is not None:
        table = table.slice(0, limit)
    return table.to_pylist()


def load_tracks(metadata_path):
    rows = read_parquet(metadata_path)
    tracks = {}
    artist_to_tracks = defaultdict(list)
    album_to_tracks = defaultdict(list)
    alias_to_hits = defaultdict(list)
    for row in rows:
        tid = row["track_id"]
        row["_track_name"] = first_value(row.get("track_name"))
        row["_artist_name"] = first_value(row.get("artist_name"))
        row["_album_name"] = first_value(row.get("album_name"))
        row["_norm_track"] = norm_text(row.get("track_name"))
        row["_norm_artist"] = norm_text(row.get("artist_name"))
        row["_norm_album"] = norm_text(row.get("album_name"))
        row["_norm_tags"] = set(norm_text(row.get("tag_list")).split())
        tracks[tid] = row
        if row["_norm_artist"]:
            artist_to_tracks[row["_norm_artist"]].append(tid)
            alias_to_hits[row["_norm_artist"]].append((tid, 25))
        if row["_norm_album"]:
            album_to_tracks[row["_norm_album"]].append(tid)
            alias_to_hits[row["_norm_album"]].append((tid, 15))
        if row["_norm_track"]:
            alias_to_hits[row["_norm_track"]].append((tid, 100))
    return tracks, dict(artist_to_tracks), dict(album_to_tracks), dict(alias_to_hits)


def track_corpus(row):
    parts = [
        first_value(row.get("track_name")),
        first_value(row.get("artist_name")),
        first_value(row.get("album_name")),
        " ".join(row.get("tag_list") or []),
        row.get("release_date") or "",
    ]
    return "\n".join(parts).lower()


def build_bm25(tracks):
    track_ids = list(tracks)
    corpus = [track_corpus(tracks[tid]) for tid in track_ids]
    tokenized = bm25s.tokenize(corpus, stopwords="en")
    model = bm25s.BM25()
    model.index(tokenized)
    return model, track_ids


def bm25_retrieve(model, track_ids, query, topk):
    tokens = bm25s.tokenize([query.lower()], stopwords="en")
    result = model.retrieve(tokens, k=topk, return_as="tuple")
    docs = result.documents[0]
    scores = result.scores[0]
    out = []
    for rank, (doc, score) in enumerate(zip(docs, scores), start=1):
        doc_id = int(doc["id"]) if isinstance(doc, dict) else int(doc)
        out.append((track_ids[doc_id], rank, float(score)))
    return out


def exact_retrieve(query, tracks, artist_to_tracks, album_to_tracks, alias_to_hits, cap=200):
    tokens = norm_text(query).split()
    phrases = set()
    max_n = min(8, len(tokens))
    for n in range(1, max_n + 1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i + n])
            if len(phrase) >= 3:
                phrases.add(phrase)
    scored = []
    for phrase in phrases:
        scored.extend(alias_to_hits.get(phrase, []))
    expanded = {}
    for tid, score in scored:
        expanded[tid] = max(expanded.get(tid, 0), score)
        artist = tracks[tid]["_norm_artist"]
        album = tracks[tid]["_norm_album"]
        for other in artist_to_tracks.get(artist, [])[:50]:
            expanded[other] = max(expanded.get(other, 0), score - 10)
        for other in album_to_tracks.get(album, [])[:50]:
            expanded[other] = max(expanded.get(other, 0), score - 20)

    ranked = sorted(expanded.items(), key=lambda x: (-x[1], x[0]))[:cap]
    return [(tid, rank, float(score)) for rank, (tid, score) in enumerate(ranked, start=1)]


def parse_turns(row):
    conv = row["conversations"]
    by_turn = defaultdict(dict)
    for entry in conv:
        by_turn[entry["turn_number"]][entry["role"]] = entry["content"]
    return by_turn


def history_for_turn(row, turn_number, tracks):
    conv = row["conversations"]
    parts = []
    played = []
    for entry in conv:
        t = entry["turn_number"]
        role = entry["role"]
        if t > turn_number:
            continue
        if t == turn_number and role != "user":
            continue
        content = entry["content"]
        if role == "music" and content in tracks:
            played.append(content)
            tr = tracks[content]
            parts.append(f"played: {tr['_track_name']} by {tr['_artist_name']} from {tr['_album_name']}")
        elif role in {"user", "assistant"}:
            parts.append(f"{role}: {content}")
    return "\n".join(parts), played


def build_examples(rows, tracks, bm25_model, bm25_track_ids, artist_to_tracks, album_to_tracks, alias_to_hits, bm25_topk, exact_topk):
    examples = []
    for row in rows:
        turns = parse_turns(row)
        for turn in range(1, 9):
            query, played = history_for_turn(row, turn, tracks)
            gold = turns[turn]["music"]
            bm25_hits = bm25_retrieve(bm25_model, bm25_track_ids, query, bm25_topk)
            exact_hits = exact_retrieve(query, tracks, artist_to_tracks, album_to_tracks, alias_to_hits, exact_topk)
            candidates = {}
            union = []
            for source, hits in [("bm25", bm25_hits), ("exact", exact_hits)]:
                for tid, rank, score in hits:
                    if tid not in candidates:
                        candidates[tid] = {
                            "track_id": tid,
                            "bm25_rank": 0,
                            "bm25_score": 0.0,
                            "exact_rank": 0,
                            "exact_score": 0.0,
                        }
                        union.append(tid)
                    candidates[tid][f"{source}_rank"] = rank
                    candidates[tid][f"{source}_score"] = score
            examples.append({
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "turn_number": turn,
                "query": query,
                "gold": gold,
                "played": played,
                "candidates": [candidates[tid] for tid in union],
                "union_order": union,
            })
    return examples


def recall_at(examples, ks):
    vals = {}
    for k in ks:
        vals[f"recall@{k}"] = float(np.mean([ex["gold"] in ex["union_order"][:k] for ex in examples]))
    return vals


def source_recall(examples):
    counts = Counter()
    total = len(examples)
    for ex in examples:
        gold = ex["gold"]
        for cand in ex["candidates"]:
            if cand["track_id"] != gold:
                continue
            if cand["bm25_rank"]:
                counts["bm25"] += 1
            if cand["exact_rank"]:
                counts["exact"] += 1
            counts["union"] += 1
            break
    return {f"{k}_gold_in_pool": v / total for k, v in counts.items()}


def make_feature_rows(examples, tracks):
    rows = []
    groups = []
    labels = []
    meta = []
    for ex in examples:
        start = len(rows)
        played_artists = {tracks[tid]["_norm_artist"] for tid in ex["played"] if tid in tracks}
        played_albums = {tracks[tid]["_norm_album"] for tid in ex["played"] if tid in tracks}
        recent_artist = tracks[ex["played"][-1]]["_norm_artist"] if ex["played"] and ex["played"][-1] in tracks else ""
        q_terms = set(norm_text(ex["query"]).split())
        for idx, cand in enumerate(ex["candidates"], start=1):
            tr = tracks[cand["track_id"]]
            tags = tr["_norm_tags"]
            artist = tr["_norm_artist"]
            album = tr["_norm_album"]
            release_year = 0
            if tr.get("release_date") and len(tr["release_date"]) >= 4 and tr["release_date"][:4].isdigit():
                release_year = int(tr["release_date"][:4])
            rows.append([
                idx,
                cand["bm25_rank"] or 9999,
                cand["bm25_score"],
                1 if cand["bm25_rank"] else 0,
                cand["exact_rank"] or 9999,
                cand["exact_score"],
                1 if cand["exact_rank"] else 0,
                float(tr.get("popularity") or 0.0),
                math.log1p(float(tr.get("popularity") or 0.0)),
                release_year,
                len(tags),
                1 if cand["track_id"] in ex["played"] else 0,
                1 if artist and artist in played_artists else 0,
                1 if album and album in played_albums else 0,
                1 if artist and artist == recent_artist else 0,
                len(q_terms.intersection(tags)),
            ])
            labels.append(1 if cand["track_id"] == ex["gold"] else 0)
            meta.append((ex, cand["track_id"]))
        n = len(rows) - start
        if n:
            groups.append(n)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32), groups, meta


FEATURE_NAMES = [
    "union_rank",
    "bm25_rank",
    "bm25_score",
    "bm25_present",
    "exact_rank",
    "exact_score",
    "exact_present",
    "popularity",
    "log_popularity",
    "release_year",
    "tag_count",
    "in_history",
    "same_artist_as_history",
    "same_album_as_history",
    "same_artist_as_last",
    "query_tag_overlap",
]


def filter_gold_in_pool(examples):
    return [ex for ex in examples if ex["gold"] in {c["track_id"] for c in ex["candidates"]}]


def train_ranker(train_examples):
    train_examples = filter_gold_in_pool(train_examples)
    x, y, groups, _ = make_feature_rows(train_examples, TRACKS)
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=80,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=20260830,
        verbose=-1,
    )
    ranker.fit(x, y, group=groups, feature_name=FEATURE_NAMES)
    return ranker, len(train_examples), int(y.sum())


def ndcg_single(gold, preds, k):
    for i, pred in enumerate(preds[:k], start=1):
        if pred == gold:
            return 1.0 / math.log2(i + 1)
    return 0.0


def rerank_examples(model, examples):
    predictions = []
    ndcgs = []
    for ex in examples:
        x, _, _, meta = make_feature_rows([ex], TRACKS)
        scores = model.predict(x)
        ranked = [tid for (_, tid), _ in sorted(zip(meta, scores), key=lambda x: -x[1])]
        top20 = ranked[:20]
        ndcgs.append(ndcg_single(ex["gold"], top20, 20))
        predictions.append({
            "session_id": ex["session_id"],
            "user_id": ex["user_id"],
            "turn_number": ex["turn_number"],
            "predicted_track_ids": top20,
            "predicted_response": "",
        })
    return predictions, float(np.mean(ndcgs))


def baseline_union_ndcg(examples):
    return float(np.mean([ndcg_single(ex["gold"], ex["union_order"][:20], 20) for ex in examples]))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", default=str(DOWNLOADS / "train-00000-of-00001.parquet"))
    parser.add_argument("--dev-parquet", default=str(DOWNLOADS / "test-00000-of-00001.parquet"))
    parser.add_argument("--track-parquet", default=str(DOWNLOADS / "all_tracks-00000-of-00001.parquet"))
    parser.add_argument("--train-sessions", type=int, default=300)
    parser.add_argument("--dev-sessions", type=int, default=100)
    parser.add_argument("--bm25-topk", type=int, default=200)
    parser.add_argument("--exact-topk", type=int, default=200)
    parser.add_argument("--run-name", default="m1_bm25_exact_lambdarank_subset")
    args = parser.parse_args()

    global TRACKS
    TRACKS, artist_to_tracks, album_to_tracks, alias_to_hits = load_tracks(args.track_parquet)
    bm25_model, bm25_track_ids = build_bm25(TRACKS)

    train_rows = read_parquet(args.train_parquet, limit=args.train_sessions)
    dev_rows = read_parquet(args.dev_parquet, limit=args.dev_sessions)
    train_examples = build_examples(
        train_rows, TRACKS, bm25_model, bm25_track_ids, artist_to_tracks, album_to_tracks, alias_to_hits, args.bm25_topk, args.exact_topk
    )
    dev_examples = build_examples(
        dev_rows, TRACKS, bm25_model, bm25_track_ids, artist_to_tracks, album_to_tracks, alias_to_hits, args.bm25_topk, args.exact_topk
    )

    train_recall = recall_at(train_examples, [20, 50, 100, 200])
    dev_recall = recall_at(dev_examples, [20, 50, 100, 200])
    dev_source_recall = source_recall(dev_examples)
    model, train_groups_used, train_positives = train_ranker(train_examples)
    predictions, rerank_ndcg20 = rerank_examples(model, dev_examples)
    union_ndcg20 = baseline_union_ndcg(dev_examples)

    out_dir = ROOT / "experiments" / "runs" / args.run_name
    save_json(out_dir / "prediction_subset.json", predictions)
    metrics = {
        "run_name": args.run_name,
        "train_sessions": args.train_sessions,
        "train_turn_examples": len(train_examples),
        "train_groups_used_gold_in_pool": train_groups_used,
        "train_positives": train_positives,
        "dev_sessions": args.dev_sessions,
        "dev_turn_examples": len(dev_examples),
        "bm25_topk": args.bm25_topk,
        "exact_topk": args.exact_topk,
        "train_candidate_recall": train_recall,
        "dev_candidate_recall": dev_recall,
        "dev_source_recall": dev_source_recall,
        "union_order_ndcg@20": union_ndcg20,
        "lambdarank_ndcg@20": rerank_ndcg20,
        "feature_importance_gain": dict(sorted(
            zip(FEATURE_NAMES, model.booster_.feature_importance(importance_type="gain")),
            key=lambda x: -x[1],
        )),
    }
    save_json(out_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
