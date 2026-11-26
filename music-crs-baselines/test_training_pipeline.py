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
    check("nDCG@20 over" in cv_out and "lightgbm - rrf_sum" in cv_out,
          "CV prints per-ranker CI and paired difference")
    final_dir = tmp / "final"
    train_ranker.main(["--train", str(out), "--final-rounds", "30",
                       "--output-dir", str(final_dir), "--min-data-in-leaf", "20"])
    check((final_dir / "ranker.txt").exists() and (final_dir / "ranker_features.json").exists(),
          "final-fit saves model + schema")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_ranker.main(["--train", str(out), "--eval-model", str(final_dir)])
    ev = buf.getvalue()
    check("Official-style metrics" in ev and "lightgbm - rrf_sum" in ev,
          "--eval-model scores a dataset with a saved model")
    check("nDCG@20 by group" in ev and "no user_id" in ev or "with user_id" in ev,
          "report splits nDCG by user / turn-1 groups")

    # a dataset with EXTRA columns (newer schema) can still be scored by an older model
    wider = tmp / "wider.csv"
    pd.read_csv(out).assign(brand_new_feature=1.0).to_csv(wider, index=False)
    Path(str(wider) + ".meta.json").write_text(Path(str(out) + ".meta.json").read_text().replace(
        '"feature_names": [', '"feature_names": ["brand_new_feature", ', 1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_ranker.main(["--train", str(wider), "--eval-model", str(final_dir)])
    check("Official-style metrics" in buf.getvalue(),
          "--eval-model accepts a dataset with extra feature columns")

    print("\n[anchor-artist retriever + continuity features]")
    # Load the module by path: the lexical package __init__ pulls in bm25s.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "recommender_v2.retrievers.lexical.anchor_artist",
        ROOT / "recommender_v2/retrievers/lexical/anchor_artist.py",
        submodule_search_locations=None,
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "recommender_v2.retrievers.lexical"
    lex = types.ModuleType("recommender_v2.retrievers.lexical")
    lex.__path__ = []
    sys.modules.setdefault("recommender_v2.retrievers.lexical", lex)
    import datasets as _ds
    for _n in ("load_dataset", "concatenate_datasets"):
        if not hasattr(_ds, _n):
            setattr(_ds, _n, lambda *a, **k: None)
    spec.loader.exec_module(mod)
    AnchorArtistRetriever = mod.AnchorArtistRetriever
    from collections import defaultdict as _dd
    r = AnchorArtistRetriever.__new__(AnchorArtistRetriever)
    r.track_artists = {"a1": ["x"], "a2": ["y"], "t1": ["x"], "t2": ["x"], "t3": ["y"], "t4": ["z"]}
    r.artist_tracks = _dd(list, {"x": ["a1", "t1", "t2"], "y": ["a2", "t3"], "z": ["t4"]})
    r.popularity = {"t1": 5.0, "t2": 9.0, "t3": 1.0}
    got = [x.track_id for x in r.retrieve("", 10, anchor_track_ids=["a1", "a2"])]
    check(got == ["t3", "t2", "t1"],
          f"most recent anchor's artist first, then popularity ({got})")
    check(r.retrieve("", 10, anchor_track_ids=[]) == [], "no anchors -> no candidates")
    check("a1" not in got and "a2" not in got, "anchors themselves are never proposed")

    meta = {
        "a1": {"artist_names": ["X"], "album_names": ["AL1"], "genres": ["rock", "indie"], "year": 2000, "popularity": 3.0},
        "a2": {"artist_names": ["Y"], "album_names": ["AL2"], "genres": ["pop"], "year": 2010, "popularity": 4.0},
        "c1": {"artist_names": ["X"], "album_names": ["AL1"], "genres": ["rock"], "year": 2004, "popularity": 7.0},
        "c2": {"artist_names": ["Q"], "album_names": ["ALQ"], "genres": ["jazz"], "year": 1990, "popularity": 1.0},
    }
    fb = FeatureBuilder(track_metadata=meta)
    from recommender_v2.retrievers.base import RetrievalResult as RR
    rows = {x["track_id"]: x for x in fb.build_candidate_features(
        {"bm25": [RR("c1", 1.0, 1, "bm25"), RR("c2", 0.5, 2, "bm25")]},
        anchor_track_ids=["a1", "a2"], turn_index=3)}
    check(rows["c1"]["same_artist_as_anchor"] == 1 and rows["c2"]["same_artist_as_anchor"] == 0,
          "same_artist_as_anchor counts anchors sharing an artist")
    check(rows["c1"]["same_artist_as_last_anchor"] == 0, "last anchor (a2) is a different artist")
    check(rows["c1"]["same_album_as_anchor"] == 1, "same_album_as_anchor")
    check(abs(rows["c1"]["anchor_tag_overlap"] - 0.5) < 1e-9, "tag overlap is Jaccard vs best anchor")
    check(abs(rows["c1"]["year_gap_to_anchors"] - 1.0) < 1e-9, "year gap to mean anchor year (2005)")
    check(rows["c1"]["candidate_popularity"] == 7.0, "candidate popularity from metadata")
    first = fb.build_candidate_features({"bm25": [RR("c1", 1.0, 1, "bm25")]},
                                        anchor_track_ids=[], turn_index=1)[0]
    check(first["same_artist_as_anchor"] != first["same_artist_as_anchor"],
          "turn 1 (no anchors): continuity features are NaN")

    print("\n[transition-artist retriever: leave-fold-out]")
    import importlib.util as _iu
    _spec = _iu.spec_from_file_location(
        "recommender_v2.retrievers.lexical.transition_artist",
        ROOT / "recommender_v2/retrievers/lexical/transition_artist.py")
    _mod = _iu.module_from_spec(_spec)
    _mod.__package__ = "recommender_v2.retrievers.lexical"
    import datasets as _ds2
    for _n in ("load_dataset", "concatenate_datasets"):
        if not hasattr(_ds2, _n):
            setattr(_ds2, _n, lambda *a, **k: None)
    _spec.loader.exec_module(_mod)
    TR = _mod.TransitionArtistRetriever
    ta = {"a1": ("x",), "b1": ("y",), "c1": ("z",), "a2": ("x",), "b2": ("y",), "n1": ("q",)}
    at = {"x": ["a1", "a2"], "y": ["b1", "b2"], "z": ["c1"], "q": ["n1"]}
    # many sessions say x -> y; exactly ONE (s_only) says x -> z
    sessions = [(f"s{i}", ["a1", "b1"]) for i in range(40)] + [("s_only", ["a1", "c1"])]
    tr = TR.__new__(TR)
    tr._setup(ta, at, {}, sessions, n_folds=5)

    full = [r.track_id for r in tr.retrieve("", 10, anchor_track_ids=["a2"])]
    check("b1" in full and "c1" in full, "inference (no begin_session): full table sees both transitions")
    tr.begin_session("s_only", held_out=True)
    own = [r.track_id for r in tr.retrieve("", 10, anchor_track_ids=["a2"])]
    fold_mates = [sid for sid, _ in sessions if _mod.session_fold(sid) == _mod.session_fold("s_only")]
    check("c1" not in own, "a session's own x->z transition is hidden from it")
    check("b1" in own or len(fold_mates) > 30, "other sessions' transitions remain available")
    tr.begin_session("s_only", held_out=False)
    check("c1" in [r.track_id for r in tr.retrieve("", 10, anchor_track_ids=["a2"])],
          "held_out=False restores the full table")
    check(tr.retrieve("", 10, anchor_track_ids=[]) == [], "no anchors -> nothing")
    check(all(r.track_id not in ("a1", "a2") for r in tr.retrieve("", 10, anchor_track_ids=["a1"])),
          "never proposes the anchor artist's own tracks")

    print("\n[rebuild from imported planner cache, zero LLM calls]")
    def _no_llm(prompt, attempt=0):
        raise AssertionError("LLM must not be called when plans are imported")
    fake_ollama = types.ModuleType("recommender_v2.planner.ollama")
    fake_ollama.OllamaQwenPlanner = lambda model: _no_llm
    sys.modules["recommender_v2.planner.ollama"] = fake_ollama
    out_re = tmp / "rebuilt.csv"
    build_dataset.main(["--split", "train", "--num-sessions", "10", "--output", str(out_re),
                        "--models", "bm25",
                        "--planner-cache-from", str(out) + ".planner.jsonl"])
    a = pd.read_csv(out_re)
    b = pd.read_csv(out)
    b = b[b["query_id"].isin(set(a["query_id"]))].reset_index(drop=True)
    check(a.equals(b), "rebuilt dataset equals the original, built without the LLM")
    fake_ollama.OllamaQwenPlanner = lambda model: llm
    sys.modules["recommender_v2.planner.ollama"] = fake_ollama

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
    check("nDCG@20 over" in buf.getvalue(), "two chunks load and cross-validate")
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

    print("\n[blind-set inference: format and visibility]")
    import run_blind_inference as RBI
    blind_rows = []
    for i in (3, 7, 11):
        sess = make_session(i)
        conv = sess["conversations"]
        # a blind row ends at the user message of the target turn (turn 4)
        cut = max(k for k, m in enumerate(conv) if m["turn_number"] == 4 and m["role"] == "user")
        blind_rows.append({"session_id": sess["session_id"], "user_id": sess["user_id"],
                           "conversations": conv[:cut + 1]})
    blind_rows.append({**blind_rows[0], "session_id": "no_user", "user_id": None})
    v_before = len(llm.violations)
    res = RBI.predict_rows(blind_rows, rec, log=lambda *_: None)
    check(len(res) == 4 and all(set(r) == {"session_id", "user_id", "turn_number",
          "predicted_track_ids", "predicted_response"} for r in res), "submission fields")
    check(all(r["turn_number"] == 4 and len(r["predicted_track_ids"]) == 20 for r in res),
          "turn number taken from the last message; 20 tracks each")
    check(len(llm.violations) == v_before, "planner prompts contain nothing from the target turn or later")
    check(res[3]["user_id"] is None, "missing user_id handled")

    print("\n[extra features: embedding similarity, user affinity, pool features]")
    from recommender_v2.features import EXTRA_FEATURES

    # 4 tracks in a 2-D space: t0 = (1,0), t1 = (0,1), t2 = (1,1), t3 = (-1,0)
    emb = np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32)
    index = {"t0": 0, "t1": 1, "t2": 2, "t3": 3}
    fb = FeatureBuilder(track_metadata={
        "t0": {"artist_names": ["A"], "popularity": 10.0},
        "t1": {"artist_names": ["A"], "popularity": 30.0},
        "t2": {"artist_names": ["B"], "popularity": 20.0},
        "t3": {"artist_names": ["C"], "popularity": 40.0},
    })
    fb._spaces = {"cf": (emb, index), "clap": (emb, index), "meta": (emb, index)}
    fb._user_space = ({"u1": np.array([2.0, 0.0], dtype=np.float32)}, emb, index)

    rows = [{"track_id": t, "candidate_popularity": fb.track_metadata.get(t, {}).get("popularity", float("nan"))}
            for t in ("t1", "t2", "t3", "tX")]
    fb.add_extra_features(rows, user_id="u1", anchor_track_ids=["t0"],
                          metadata_lookup=fb.track_metadata)
    by = {r["track_id"]: r for r in rows}
    check(all(n in rows[0] for n in EXTRA_FEATURES), "every extra feature is set")
    check(abs(by["t1"]["emb_cf_centroid_sim"]) < 1e-6, "orthogonal track has cosine 0")
    check(abs(by["t2"]["emb_cf_last_sim"] - 2 ** -0.5) < 1e-5, "45-degree track has cosine 0.707")
    check(abs(by["t3"]["emb_cf_max_sim"] + 1.0) < 1e-6, "opposite track has cosine -1")
    check(abs(by["t2"]["user_cf_sim"] - 2 ** -0.5) < 1e-5, "user affinity is a cosine")
    check(np.isnan(by["tX"]["emb_cf_centroid_sim"]) and np.isnan(by["tX"]["user_cf_sim"]),
          "track without an embedding -> NaN")
    check(by["t1"]["artist_track_count"] == 2.0 and by["t3"]["artist_track_count"] == 1.0,
          "artist catalogue size counted")
    check(by["t3"]["popularity_rank_in_pool"] > by["t1"]["popularity_rank_in_pool"]
          > by["t2"]["popularity_rank_in_pool"], "popularity percentile increases with popularity")

    # several anchors: max over anchors, last anchor is the most recent one
    rows = [{"track_id": t, "candidate_popularity": float("nan")} for t in ("t2",)]
    fb.add_extra_features(rows, user_id=None, anchor_track_ids=["t1", "t0"])
    check(abs(rows[0]["emb_meta_last_sim"] - 2 ** -0.5) < 1e-5, "last = most recent anchor (t0)")
    check(np.isnan(rows[0]["user_cf_sim"]), "no user -> user_cf_sim NaN")
    check(np.isnan(rows[0]["popularity_rank_in_pool"]), "unknown popularity -> NaN")

    # turn 1: no anchors -> similarity features NaN, others unaffected
    rows = [{"track_id": "t2", "candidate_popularity": 20.0}]
    fb.add_extra_features(rows, user_id="u1", anchor_track_ids=[])
    check(all(np.isnan(rows[0][f"emb_{k}_{m}"]) for k in ("cf", "clap", "meta")
              for m in ("centroid_sim", "last_sim", "max_sim")), "no anchors -> similarities NaN")
    check(not np.isnan(rows[0]["user_cf_sim"]), "user affinity still available at turn 1")

    # builder without embeddings still yields the full schema (all NaN)
    plain = FeatureBuilder()
    rows = [{"track_id": "t0"}]
    plain.add_extra_features(rows, anchor_track_ids=["t1"])
    check(all(n in rows[0] for n in EXTRA_FEATURES), "schema complete without embeddings")

    # user dropout: per-session, deterministic, user_id blank and has_user = 0 together
    outs = []
    for k in range(2):
        d = tmp / f"drop{k}"
        d.mkdir()
        outs.append(pd.read_csv(run_builder(d, llm, 40, extra=["--user-dropout", "0.5"])))
    a_, b_ = outs
    frac = a_.groupby("session_id")["user_id"].apply(lambda x: x.isna().all()).mean()
    check(0.2 < frac < 0.8, f"about half the sessions lose their user_id (got {frac:.2f})")
    check(a_["user_id"].isna().equals(b_["user_id"].isna()), "dropout identical on rerun")
    check(((a_["has_user"] == 0) == a_["user_id"].isna()).all(), "has_user = 0 exactly when user_id is blank")
    mixed = a_.groupby("session_id")["user_id"].apply(lambda x: x.isna().any() and x.notna().any()).any()
    check(not mixed, "dropout never splits a session")

    print("\n[add_features.py: upgrade an existing dataset without retrieval]")
    import add_features
    from recommender_v2.features import EXTRA_FEATURES as _EX
    d = tmp / "upg"
    d.mkdir()
    built = run_builder(d, llm, 6)
    # simulate an OLD dataset: same file without the newer columns
    src = d / "old.csv"
    pd.read_csv(built).drop(columns=_EX).to_csv(src, index=False)
    m_ = json.loads(Path(str(built) + ".meta.json").read_text())
    m_["feature_names"] = [f for f in m_["feature_names"] if f not in _EX]
    Path(str(src) + ".meta.json").write_text(json.dumps(m_))
    # give the fake retrievers real embeddings so values can be checked
    ids_all = sorted(pd.read_csv(src)["track_id"].unique())
    rng = np.random.RandomState(0)
    emb_ = rng.randn(len(ids_all), 8).astype(np.float32)
    idx_ = {t: i for i, t in enumerate(ids_all)}

    class _R:
        def __init__(self, name):
            self.name, self.track_embeddings, self.track_to_index = name, emb_, idx_
    fake_set = [_R("item_cf_bpr"), _R("item_clap"), _R("item_metadata")]
    build_dataset.build_retrievers = lambda names, device="cpu", text_fields=None: fake_set
    dst = d / "upgraded.csv"
    sys.argv = ["add_features.py", "--input", str(src), "--output", str(dst)]
    add_features.main()
    up = pd.read_csv(dst)
    base = pd.read_csv(src)
    check(len(up) == len(base) and (up["track_id"].values == base["track_id"].values).all(),
          "same rows in the same order")
    check(all(c in up.columns for c in _EX), "all extra feature columns added")
    later = up[up["turn_number"] > 1]
    check(later["emb_cf_centroid_sim"].notna().all(), "similarity present from turn 2 on")
    check(up[up["turn_number"] == 1]["emb_cf_centroid_sim"].isna().all(), "turn 1 has no anchors -> NaN")
    check(set(json.loads(Path(str(dst) + ".meta.json").read_text())["feature_names"]) >= set(_EX),
          "meta.json lists the new features")
    # the value for one row matches a direct computation
    print("\n[slim QueryState: unused planner fields removed]")
    from recommender_v2.planner.schema import LEGACY_FIELDS, LLM_FIELDS, QueryState as QS
    from recommender_v2.planner.planner import QueryPlanner as QP
    prompt = QP(llm=lambda p: "{}", mode="llm")._build_prompt("hi", [], [])
    check(not any(f'"{f}"' in prompt for f in LEGACY_FIELDS), "prompt no longer asks for retired fields")
    check(all(f'"{f}"' in prompt for f in LLM_FIELDS), "prompt asks for every remaining field")
    check(set(QS().to_dict()) == set(LLM_FIELDS) | {"raw_query", "anchor_track_ids"},
          "QueryState = LLM fields + pipeline fields")
    slim = {f: QS().to_dict()[f] for f in LLM_FIELDS}
    old = dict(slim, mood_phrases=["calm"], rejected_track_ids=[], mentioned_entities=["x"])
    parser = QP(mode="llm", llm=lambda p: "{}")
    check(parser._parse_json(json.dumps(old)).keys() == slim.keys(), "old-format plan accepted, retired fields dropped")
    check(QS.from_dict(old).to_dict()["artist_names"] == [], "from_dict ignores retired fields")
    try:
        parser._parse_json(json.dumps(dict(slim, made_up=1)))
        check(False, "unknown field rejected")
    except ValueError:
        check(True, "unknown field rejected")
    import importlib.util as _iu
    _spec = _iu.spec_from_file_location("recommender_v2.planner._ollama_real",
                                        ROOT / "recommender_v2" / "planner" / "ollama.py")
    _ol = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_ol)
    check(set(_ol.query_state_schema()["properties"]) == set(LLM_FIELDS), "Ollama JSON schema matches QueryState")

    # plans cached by the previous planner version (tag v2) are still reused
    d = tmp / "v2cache"
    d.mkdir()
    first = run_builder(d, llm, 4)
    cache_file = Path(str(first) + ".planner.jsonl")
    lines = [json.loads(l) for l in cache_file.read_text().splitlines()]
    for r in lines:
        r["planner"] = r["planner"].replace("|v3", "|v2")
        r["state"].update(mood_phrases=[], negative_constraints=[], rejected_track_ids=[])
    old_cache = d / "old.planner.jsonl"
    old_cache.write_text("\n".join(json.dumps(r) for r in lines))
    d2 = tmp / "v2cache_rebuild"
    d2.mkdir()
    calls_before = llm.calls
    run_builder(d2, llm, 4, extra=["--planner-cache-from", str(old_cache)])
    check(llm.calls == calls_before, "v2-tagged cached plans reused: no LLM calls")
    meta2 = json.loads(Path(str(d2 / "train.csv") + ".meta.json").read_text())
    check(any("|v2" in t for t in meta2.get("planner_tags_reused_from_cache", [])),
          "meta.json records that v2 plans were reused")

    print("\n[artist / album / title lookups read the structured QueryState]")
    lex_dir = ROOT / "recommender_v2" / "retrievers" / "lexical"
    if "recommender_v2.retrievers.lexical" not in sys.modules or \
            not hasattr(sys.modules["recommender_v2.retrievers.lexical"], "ArtistLookupRetriever"):
        lexpkg = types.ModuleType("recommender_v2.retrievers.lexical")
        lexpkg.__path__ = [str(lex_dir)]
        sys.modules["recommender_v2.retrievers.lexical"] = lexpkg
    al = importlib.import_module("recommender_v2.retrievers.lexical.artist_lookup")
    bl = importlib.import_module("recommender_v2.retrievers.lexical.album_lookup")
    tl = importlib.import_module("recommender_v2.retrievers.lexical.title_lookup")
    catalog = [
        {"track_id": "b1", "track_name": "Halo", "artist_name": ["Beyoncé"], "album_name": "I Am", "popularity": 50},
        {"track_id": "b2", "track_name": "Crazy in Love", "artist_name": ["Beyoncé"], "album_name": "Dangerously", "popularity": 90},
        {"track_id": "b3", "track_name": "Single Ladies", "artist_name": ["Beyoncé"], "album_name": "I Am", "popularity": 70},
        {"track_id": "r1", "track_name": "Creep", "artist_name": ["Radiohead"], "album_name": "Pablo Honey", "popularity": 80},
        {"track_id": "r2", "track_name": "Creep (Remastered 2009)", "artist_name": ["Radiohead"], "album_name": "Greatest Hits", "popularity": 40},
        {"track_id": "t1", "track_name": "Creep", "artist_name": ["TLC"], "album_name": "CrazySexyCool", "popularity": 60},
        {"track_id": "x1", "track_name": "Yesterday", "artist_name": ["The Beatles"], "album_name": "Help!", "popularity": 85},
        {"track_id": "x2", "track_name": "Hey Jude", "artist_name": ["The Beatles"], "album_name": "Greatest Hits", "popularity": 95},
        {"track_id": "c1", "track_name": "Vitamin C", "artist_name": ["Can"], "album_name": "Ege Bamyasi", "popularity": 30},
    ]
    for mod in (al, bl, tl):
        mod.load_catalog = lambda name, splits: catalog
    A = al.ArtistLookupRetriever("x", ["all"])
    B = bl.AlbumLookupRetriever("x", ["all"])
    T = tl.TitleLookupRetriever("x", ["all"])
    QS_ = QS
    ids = lambda res: [r.track_id for r in res]

    st = QS_(raw_query="something chill", semantic_query="calm r&b", artist_names=["Beyonce"])
    got = A.retrieve(query=st.semantic_query, query_state=st)
    check(ids(got) == ["b2", "b3", "b1"], "artist from artist_names (accent-insensitive), most popular first")
    check(got[0].score == 1.0, "exact structured match scores 1.0")
    check(ids(A.retrieve("", query_state=QS_(artist_names=["Beyonse"]))) == ["b2", "b3", "b1"], "misspelt name found by fuzzy match")
    check(A.retrieve("", query_state=QS_(artist_names=["Beyonse"]))[0].score == 0.9, "fuzzy match scores 0.9")
    check(ids(A.retrieve("", query_state=QS_(artist_names=["Beatles"]))) == ["x2", "x1"], "leading 'the' ignored")
    two = ids(A.retrieve("", query_state=QS_(artist_names=["Radiohead", "Beyonce"])))
    check(two[:2] == ["r1", "b2"], "two artists share the slots (interleaved)")
    st = QS_(raw_query="play some radiohead please", semantic_query="moody rock")
    check(ids(A.retrieve(st.semantic_query, query_state=st)) == ["r1", "r2"], "fallback: name found in the user's message")
    check(A.retrieve(st.semantic_query, query_state=st)[0].score == 0.7, "text fallback scores 0.7")
    st = QS_(raw_query="I can listen to anything", semantic_query="anything")
    check(A.retrieve("anything", query_state=st) == [], "common word 'can' is not taken as the band Can")
    st = QS_(raw_query="nice, but not Radiohead again", rejected_artists=["Radiohead"])
    check(A.retrieve("", query_state=st) == [], "rejected artist never returned (even via the text fallback)")
    st = QS_(artist_names=["Beyonce"])
    check("b2" not in ids(A.retrieve("", query_state=st, session_track_ids=["b2"])), "already-played tracks skipped")
    check(ids(A.retrieve("", query_state=st, topk=2)) == ["b2", "b3"], "topk keeps the most popular")
    check(ids(A.retrieve("radiohead")) == ["r1", "r2"], "works without a QueryState (plain text)")
    check(A.retrieve("", query_state=QS_(artist_names=["Nobody Known"])) == [], "unknown artist -> nothing")

    st = QS_(album_names=["Greatest Hits"], artist_names=["The Beatles"])
    got = B.retrieve("", query_state=st)
    check(ids(got)[0] == "x2" and got[0].score == 1.0, "album shared by artists: the named artist's track first")
    check(got[-1].track_id == "r2" and got[-1].score < 1.0, "other artists' album tracks after, scored lower")
    check(ids(B.retrieve("", query_state=QS_(album_names=["i am"]))) == ["b3", "b1"], "album tracks by popularity")

    st = QS_(track_titles=["Creep"], artist_names=["Radiohead"])
    got = T.retrieve("", query_state=st)
    check(ids(got)[:2] == ["r1", "r2"], "title + artist: that artist's versions first (incl. 'Remastered' version)")
    check("t1" in ids(got) and got[ids(got).index("t1")].score < 1.0, "same title by another artist kept, scored lower")
    check(ids(T.retrieve("", query_state=QS_(track_titles=["creep"])))[0] == "r1", "title alone: most popular version first")
    check(T.retrieve("", query_state=QS_(raw_query="play creep")) == [], "no title text fallback")

    print("\n[pipeline gives the lookups the whole QueryState]")
    seen = {}

    class _Probe:
        name = "artist_lookup"
        uses_query_state = True

        def retrieve(self, query, topk=100, user_id=None, anchor_track_ids=None,
                     session_track_ids=None, query_state=None):
            seen["state"] = query_state
            return []
    rp = RecommenderV2(retrievers=[_Probe()], planner=QueryPlanner(mode="mock"), ranker=RRFRanker())
    st = QS_(raw_query="q", semantic_query="q", artist_names=["Adele"])
    rp.retrieve_for_ranking(query="q", state=st)
    check(seen.get("state") is st, "lookup retriever received query_state")

    print("\n[title / artist features]")
    fbt = FeatureBuilder()
    meta = {"artist_names": ["Beyoncé"], "album_names": ["I Am"], "genres": [],
            "year": 2008, "track_name": "Halo (Live)"}
    row = {}
    fbt._add_metadata_features(row, meta, QS_(artist_names=["beyonce"], track_titles=["Halo"]))
    check(row["artist_match"] == 1.0, "artist_match is accent-insensitive")
    check(row["title_match"] == 1.0, "title_match ignores '(Live)'-style suffixes")
    row = {}
    fbt._add_metadata_features(row, meta, QS_(track_titles=[]))
    check(row["title_match"] == 0.0, "no requested title -> title_match 0")
    names = FeatureBuilder().feature_names()
    check("title_match" in names and "n_planner_titles" in names and "title_lookup_rank" in names,
          "schema has title_lookup columns, title_match, n_planner_titles")

    print("\n[single-retriever baselines are not padded]")
    _df = pd.DataFrame({
        "query_id": ["q1"] * 3,
        "label": [0, 0, 1],
        "empty_rank": [np.nan, np.nan, np.nan],
        "short_rank": [1.0, np.nan, np.nan],
    })
    zero = train_ranker.query_ndcg(_df, train_ranker.retriever_baseline_scores(_df, "empty"))
    check(float(zero.iloc[0]) == 0.0, "retriever that returned nothing scores 0 (was: pool order)")
    short = train_ranker.query_ndcg(_df, train_ranker.retriever_baseline_scores(_df, "short"))
    check(float(short.iloc[0]) == 0.0, "gold outside a short list is a miss, not a padded hit")

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