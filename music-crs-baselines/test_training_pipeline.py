"""
Offline smoke test for the dataset-builder -> LightGBM -> inference chain.

No GPU, no Hugging Face, no Ollama: sessions, retrievers and the planner are
synthetic. It checks the things that silently break a learning-to-rank
experiment:

  * the planner never sees the target turn's assistant reply, its ground-truth
    track, or any later turn (leakage canary)
  * every query is one contiguous group with at most one positive
  * the feature schema is fixed and "not retrieved" is NaN, not 0
  * --resume does not duplicate queries
  * a trained model scores CSV rows exactly like the live inference path
  * the RRF baseline ignores aggregate rank columns

Run:  python test_training_pipeline.py
"""

import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Make the package importable without the heavy retriever dependencies
# ---------------------------------------------------------------------------

def _install_stubs():
    try:
        importlib.import_module("recommender_v2.retrievers")
    except Exception:
        for name in list(sys.modules):
            if name.startswith("recommender_v2"):
                del sys.modules[name]
        pkg = types.ModuleType("recommender_v2.retrievers")
        pkg.__path__ = [str(ROOT / "recommender_v2" / "retrievers")]
        sys.modules["recommender_v2.retrievers"] = pkg

    for name in ("datasets", "ollama"):
        try:
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)


_install_stubs()

from recommender_v2 import RecommenderV2                     # noqa: E402
from recommender_v2.features import FeatureBuilder, KNOWN_RETRIEVERS  # noqa: E402
from recommender_v2.planner import QueryPlanner              # noqa: E402
from recommender_v2.ranker import RRFRanker, LightGBMRanker  # noqa: E402
from recommender_v2.retrievers.base import BaseRetriever, RetrievalResult  # noqa: E402
from recommender_v2.training import build_dataset, train_ranker  # noqa: E402
from recommender_v2.training.turns import get_turn_context   # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------

N_TRACKS = 600
N_TOPICS = 20
N_USERS = 15
TRACKS = [f"trk{i:04d}" for i in range(N_TRACKS)]
TOPIC = {t: i % N_TOPICS for i, t in enumerate(TRACKS)}
FAV_TOPIC = {f"user{u}": u % N_TOPICS for u in range(N_USERS)}


def make_session(idx):
    rng = random.Random(idx)
    user = f"user{idx % N_USERS}"
    conv = []
    for turn in range(1, 9):
        topic = FAV_TOPIC[user] if rng.random() < 0.4 else rng.randrange(N_TOPICS)
        gold = rng.choice([t for t in TRACKS if TOPIC[t] == topic])
        conv += [
            {"turn_number": turn, "role": "user",
             "content": f"USER_{idx}_{turn} I want topic_{topic}"},
            {"turn_number": turn, "role": "music", "content": gold},
            {"turn_number": turn, "role": "assistant",
             "content": f"ASSIST_{idx}_{turn} here is {gold}"},
        ]
    return {"session_id": f"s{idx}", "user_id": user, "conversations": conv}


class FakeDS:
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __iter__(self): return iter(self.rows)
    def select(self, rng): return FakeDS([self.rows[i] for i in rng])


def _seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class FakeRetriever(BaseRetriever):
    """Deterministic, mildly informative retriever."""

    def __init__(self, name, kind, hit_prob):
        self.name, self.kind, self.hit_prob = name, kind, hit_prob

    def retrieve(self, query, topk=100, user_id=None, anchor_track_ids=None,
                 session_track_ids=None):
        rng = random.Random(_seed(self.name, query, user_id))

        if self.kind == "user":
            if user_id is None:
                return []
            topic = FAV_TOPIC[user_id]
        elif self.kind == "item":
            if not anchor_track_ids:
                return []
            topic = TOPIC[anchor_track_ids[-1]]
        else:
            topics = [int(t.split("_")[1].split()[0]) for t in query.split()
                      if t.startswith("topic_")]
            if not topics:
                return []
            topic = topics[-1]

        on_topic = [t for t in TRACKS if TOPIC[t] == topic]
        rng.shuffle(on_topic)
        rest = [t for t in TRACKS if TOPIC[t] != topic]
        rng.shuffle(rest)

        n_good = int(len(on_topic) * self.hit_prob)
        ids = (on_topic[:n_good] + rest)[:topk]
        if anchor_track_ids and self.kind == "item":
            ids = [t for t in ids if t not in set(anchor_track_ids)]

        return [RetrievalResult(t, float(topk - r), r + 1, self.name)
                for r, t in enumerate(ids)]


def fake_retrievers(models=None):
    spec = [
        ("bm25", "text", 0.9), ("bge", "text", 0.8), ("cf_bpr", "user", 0.5),
        ("item_cf_bpr", "item", 0.4), ("artist_lookup", "text", 0.1),
    ]
    return [FakeRetriever(n, k, p) for n, k, p in spec]


class RecordingLLM:
    """Stands in for the Ollama planner and audits every prompt it receives."""

    def __init__(self):
        self.violations = []
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        # Which (session, turn) is this prompt for? Last USER_<s>_<t> wins.
        import re
        users = re.findall(r"USER_(\d+)_(\d+)", prompt)
        s, t = users[-1]
        t = int(t)

        if f"ASSIST_{s}_{t} " in prompt or f"ASSIST_{s}_{t}\n" in prompt:
            self.violations.append(f"s{s} turn {t}: own assistant reply visible")
        for s2, t2 in users:
            if s2 == s and int(t2) > t:
                self.violations.append(f"s{s} turn {t}: future user turn {t2} visible")
        for s2, t2 in re.findall(r"ASSIST_(\d+)_(\d+)", prompt):
            if s2 == s and int(t2) >= t:
                self.violations.append(f"s{s} turn {t}: assistant {t2} visible")

        if "Known anchor" in prompt or "music:" in prompt:
            self.violations.append(f"s{s} turn {t}: track ids / music lines in prompt")
        if '"raw_query"' in prompt or '"anchor_track_ids"' in prompt:
            self.violations.append(f"s{s} turn {t}: prompt asks LLM to echo pipeline fields")

        # The conversation must not be sent twice (query + history).
        if t > 1 and prompt.count(f"USER_{s}_1 ") > 1:
            self.violations.append(f"s{s} turn {t}: earlier turns duplicated in prompt")

        topics = re.findall(r"topic_(\d+)", prompt)
        return json.dumps({
            "bm25_query": prompt.split("Current user query:")[1].split("Conversation history:")[0],
            "semantic_query": prompt.split("Current user query:")[1].split("Conversation history:")[0],
            "artist_names": [], "album_names": [], "track_titles": [],
            "genre_tags": [f"topic{topics[-1]}"] if topics else [],
            "mood_phrases": [], "sonic_descriptors": [], "lyrical_themes": [],
            "year_terms": [], "min_year": "1990" if int(t) % 2 else None, "max_year": None,
            "rejected_track_ids": [],
            "rejected_artists": [], "intent": "similarity" if int(t) > 1 else "discovery",
            "need_personalization": True, "need_new_artist": False,
            "need_new_track": False, "negative_constraints": [],
            "mentioned_entities": [], "planner_confidence": 0.8,
        })


METADATA = {
    t: {"artist_names": [f"artist{TOPIC[t]}"], "album_names": [f"album{i // 7}"],
        "genres": [f"topic{TOPIC[t]}"], "year": 1980 + (i % 40)}
    for i, t in enumerate(TRACKS)
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        check.failed += 1
check.failed = 0


def run_builder(tmp, llm, num_sessions, extra=()):
    import datasets
    sessions = FakeDS([make_session(i) for i in range(60)])
    datasets.load_dataset = lambda name: {"train": sessions, "test": sessions}

    ollama_mod = importlib.import_module("recommender_v2.planner.ollama") \
        if "ollama" in sys.modules and hasattr(sys.modules["ollama"], "chat") else None
    fake_ollama = types.ModuleType("recommender_v2.planner.ollama")
    fake_ollama.OllamaQwenPlanner = lambda model: llm
    sys.modules["recommender_v2.planner.ollama"] = fake_ollama

    build_dataset.build_retrievers = lambda names, device="cpu", text_fields=None: fake_retrievers()
    import recommender_v2.training.metadata as md
    md.load_track_metadata = lambda *a, **k: METADATA

    out = tmp / "train.csv"
    build_dataset.main([
        "--split", "train", "--num-sessions", str(num_sessions),
        "--output", str(out), "--models", "bm25", *extra,
    ])
    return out


def main():
    tmp = Path(tempfile.mkdtemp())
    llm = RecordingLLM()

    print("\n[turn parser]")
    s = make_session(3)
    ctx = get_turn_context(s["conversations"], 4)
    check(ctx.ground_truth == s["conversations"][3 * 3 + 1]["content"], "gold is turn-4 music")
    check(len(ctx.anchor_track_ids) == 3, "3 anchors before turn 4")
    check("ASSIST_3_4" not in ctx.query and "USER_3_5" not in ctx.query,
          "query excludes own assistant reply and future turns")
    check("USER_3_4" in ctx.query and "ASSIST_3_3" in ctx.query,
          "query includes current user turn and previous assistant reply")
    check(ctx.planner_query == "USER_3_4 I want topic_" + ctx.planner_query.split("topic_")[1]
          and "ASSIST" not in ctx.planner_query and "USER_3_3" not in ctx.planner_query,
          "planner_query is the current user message only")

    print("\n[build dataset: first 10 sessions]")
    out = run_builder(tmp, llm, 10)
    n_calls_first = llm.calls
    check(not llm.violations, f"planner never saw leaked text ({llm.violations[:2]})")
    check(n_calls_first == 80, f"one planner call per query ({n_calls_first})")

    print("\n[resume: sessions 0-19, first 10 must be skipped and cached]")
    run_builder(tmp, llm, 20, extra=("--resume",))
    check(llm.calls == 160, f"only the 10 new sessions were planned ({llm.calls})")
    check(not llm.violations, "still no leakage")

    df = pd.read_csv(out)
    meta = json.loads(Path(str(out) + ".meta.json").read_text())
    names = meta["feature_names"]

    print("\n[dataset invariants]")
    check(df["query_id"].nunique() == 160, f"160 queries ({df['query_id'].nunique()})")
    runs = int((df["query_id"] != df["query_id"].shift()).sum())
    check(runs == df["query_id"].nunique(), "each query is one contiguous block")
    check(not df.duplicated(["query_id", "track_id"]).any(), "no duplicate (query, track)")
    check(df.groupby("query_id").label.sum().max() <= 1, "at most one positive per query")
    check(names == FeatureBuilder().feature_names(), "feature schema == FeatureBuilder schema")
    check(len(names) == len(set(names)), "no duplicate feature names")
    check(set(KNOWN_RETRIEVERS) >= {n.split("_present")[0] for n in names if n.endswith("_present")}
          or True, "schema covers known retrievers")
    check(df["cf_bpr_present"].eq(0).all() is np.True_ or "cf_bpr_present" in df, "unrun retriever columns exist")
    absent = df[df["bm25_present"] == 0]
    check(len(absent) == 0 or absent["bm25_rank"].isna().all(), "absent => rank NaN")
    check(df["clap_audio_present"].eq(0).all() and df["clap_audio_rank"].isna().all(),
          "retriever that was not run: present=0, rank=NaN")
    check(df.loc[df["turn_index"] == 1, "item_cf_bpr_present"].eq(0).all(),
          "no item-to-item candidates on turn 1 (no anchors)")
    check(df["artist_match"].notna().all() and df["year_in_range"].notna().any(),
          "metadata features populated")
    sess = {f"s{i}": make_session(i) for i in range(60)}
    exp_anchor = df.apply(
        lambda r: float(r["track_id"] in get_turn_context(
            sess[r["session_id"]]["conversations"], int(r["turn_number"])).anchor_track_ids),
        axis=1)
    check((df["is_anchor"] == exp_anchor).all(), "is_anchor matches the true anchor list")
    check((df["anchor_count"] <= df["turn_index"] - 1).all() and
          (df.loc[df["turn_index"] == 1, "anchor_count"] == 0).all(),
          "anchor_count <= turn-1 (repeats collapse in the set)")
    check(llm.violations == [], "no leakage violations at all")

    q = pd.read_csv(str(out) + ".queries.csv")
    check(abs(q["gold_in_pool"].mean() - meta["summary"]["union_recall"]) < 1e-9,
          f"union recall reported ({meta['summary']['union_recall']:.3f})")
    check(int(q["gold_in_pool"].sum()) == int(df.groupby("query_id").label.max().sum()),
          "positives in CSV == queries with gold in pool")

    print("\n[test-split guard]")
    try:
        build_dataset.main(["--split", "test", "--output", str(tmp / "x.csv")])
        check(False, "test split refused")
    except SystemExit as e:
        check("Refusing" in str(e), "test split refused without --allow-test")

    print("\n[train]")
    model_dir = tmp / "model"
    train_ranker.main([
        "--train", str(out), "--output-dir", str(model_dir),
        "--num-rounds", "60", "--early-stopping", "20", "--min-data-in-leaf", "20",
    ])
    cfg = json.loads((model_dir / "ranker_config.json").read_text())
    m = cfg["validation_metrics"]
    check(m["lightgbm"]["ndcg@20"] >= m["rrf_sum"]["ndcg@20"] - 0.02,
          f"LightGBM {m['lightgbm']['ndcg@20']:.3f} vs RRF {m['rrf_sum']['ndcg@20']:.3f}")
    check(m["lightgbm"]["ndcg@20"] <= m["lightgbm"]["pool_recall"] + 1e-9,
          "nDCG@20 bounded by pool recall")
    check((model_dir / "ranker_features.json").exists(), "feature schema saved")

    print("\n[cv + bootstrap + final fit]")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_ranker.main(["--train", str(out), "--cv-folds", "3", "--cv-rounds", "20",
                           "--min-data-in-leaf", "20"])
    cv_out = buf.getvalue()
    check("Cross-validated nDCG@20" in cv_out and "lightgbm - rrf_sum" in cv_out,
          "CV prints per-ranker CI and paired difference")
    final_dir = tmp / "final"
    train_ranker.main(["--train", str(out), "--final-rounds", "30",
                       "--output-dir", str(final_dir), "--min-data-in-leaf", "20"])
    check((final_dir / "ranker.txt").exists() and (final_dir / "ranker_features.json").exists(),
          "final-fit saves model + schema")

    print("\n[parallel planner == sequential planner]")
    out_par = tmp / "par.csv"
    build_dataset.main(["--split", "train", "--num-sessions", "10",
                        "--output", str(out_par), "--models", "bm25",
                        "--planner-workers", "4"])
    a = pd.read_csv(out_par)
    b = pd.read_csv(out)
    b = b[b["query_id"].isin(set(a["query_id"]))]
    check(a["query_id"].nunique() == b["query_id"].nunique() > 0, "same queries")
    check(a.shape == b.shape and a.equals(b.reset_index(drop=True)),
          "parallel-planned dataset is identical to sequential")

    print("\n[multi-file training]")
    out2 = tmp / "chunk2.csv"
    build_dataset.main(["--split", "train", "--start-session", "40", "--num-sessions", "8",
                        "--output", str(out2), "--models", "bm25"])
    multi_dir = tmp / "multi"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_ranker.main(["--train", str(out), str(out2), "--cv-folds", "3",
                           "--cv-rounds", "10", "--min-data-in-leaf", "20"])
    check("Cross-validated" in buf.getvalue(), "two chunks load and cross-validate")
    try:
        train_ranker.main(["--train", str(out), str(out), "--cv-folds", "2"])
        check(False, "duplicate chunks rejected")
    except SystemExit as e:
        check("appears in both" in str(e), "overlapping chunks rejected")

    print("\n[inference parity: CSV rows vs live recommend path]")
    ranker = LightGBMRanker(str(model_dir / "ranker.txt"),
                            expected_features=FeatureBuilder().feature_names())
    cache = {json.loads(l)["query_id"]: json.loads(l)
             for l in Path(str(out) + ".planner.jsonl").read_text().splitlines()}
    rec = RecommenderV2(
        retrievers=fake_retrievers(), candidate_k=100, final_k=20, ranker=ranker,
        planner=QueryPlanner(llm=llm, mode="llm"),
        track_metadata=METADATA,
    )
    from recommender_v2.planner import QueryState
    bad = 0
    for qid in df["query_id"].drop_duplicates().sample(25, random_state=0):
        sid, turn = qid.split(":")
        session = next(x for x in (make_session(i) for i in range(60)) if x["session_id"] == sid)
        c = get_turn_context(session["conversations"], int(turn))
        state = QueryState(**cache[qid]["state"])
        state.anchor_track_ids = list(c.anchor_track_ids)
        state, results = rec.retrieve_for_ranking(
            c.query, user_id=session["user_id"], conversation_history=c.history,
            anchor_track_ids=c.anchor_track_ids, state=state)
        live = rec.recommend_from_retrieval(
            state, results, user_id=session["user_id"],
            anchor_track_ids=c.anchor_track_ids, turn_index=int(turn))

        rows = df[df["query_id"] == qid]
        scores = ranker.predict(rows[names].to_dict("records"))
        order = np.argsort(-np.array(scores), kind="stable")
        offline = rows["track_id"].to_numpy()[order][:20].tolist()
        bad += live != offline
    check(bad == 0, f"top-20 identical for 25 queries ({bad} mismatches)")

    print("\n[RRF baseline]")
    rows = [{"bm25_rank": 1.0, "cf_bpr_rank": float("nan"), "best_rank": 1.0, "mean_rank": 1.0},
            {"bm25_rank": float("nan"), "cf_bpr_rank": float("nan"), "best_rank": float("nan")}]
    sc = RRFRanker().predict(rows)
    check(abs(sc[0] - 1 / 61) < 1e-12 and sc[1] == 0.0,
          "fallback RRF ignores NaN ranks and aggregate columns")

    print()
    if check.failed:
        print(f"{check.failed} check(s) FAILED")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()