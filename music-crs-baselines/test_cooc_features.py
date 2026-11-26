"""Offline tests for cross-fitted track co-occurrence and Task 2 ranking features."""

import json
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import numpy as np

from recommender_v2.features import FEATURE_GROUPS, FEATURE_SCHEMA_VERSION, FeatureBuilder
from recommender_v2.planner import QueryState
from recommender_v2.retrievers.base import RetrievalResult
from recommender_v2.retrievers.dense.bge import BGERetriever
from recommender_v2.retrievers.item_to_item.cooc_track import CooccurrenceTrackRetriever
from recommender_v2.retrievers.lexical.session_bm25 import SessionBM25Retriever
from recommender_v2.retrievers.lexical.transition_artist import session_fold
from recommender_v2.splits import FINAL_HOLDOUT
from recommender_v2.training import build_dataset, metadata
from test_training_pipeline import FakeDS, METADATA, RecordingLLM, check, make_session


def track_session(session_id, tracks):
    return {"session_id": session_id,
            "conversations": [{"turn_number": i, "role": "music", "content": track}
                              for i, track in enumerate(tracks, 1)]}


def check_cooc(directory):
    sid = "scored-session"
    others = [f"session-{i}" for i in range(50)]
    same = next(other for other in others if session_fold(other) == session_fold(sid))
    other = next(other for other in others if session_fold(other) != session_fold(sid))
    reference = [track_session(same, ["a", "same-fold-only"]), track_session(other, ["a", "b", "c"])]
    complete = reference + [track_session(sid, ["c", "a", "own-gold"])]

    class Catalogue(list):
        _fingerprint = "test-train"

    with patch("recommender_v2.retrievers.item_to_item.cooc_track.load_dataset",
               return_value=Catalogue(complete)) as load:
        retriever = CooccurrenceTrackRetriever(cache_dir=directory)
        reopened = CooccurrenceTrackRetriever(cache_dir=directory, cooc_weight=0)
        rescaled = CooccurrenceTrackRetriever(cache_dir=directory, cooc_weight=1e-12, next_weight=0)
        check(all(call.kwargs == {"split": "train"} for call in load.call_args_list),
              "co-occurrence loads only the train split")
    with patch("recommender_v2.retrievers.item_to_item.cooc_track.load_dataset", return_value=reference):
        without_session = CooccurrenceTrackRetriever(cache_dir=None)

    for source in (retriever, reopened, rescaled, without_session):
        source.begin_session(sid, held_out=True)
    out = retriever.retrieve("", anchor_track_ids=["a"])
    check(out == without_session.retrieve("", anchor_track_ids=["a"]),
          "held-out scores do not depend on whether the scored session was in the table")
    check([(result.track_id, result.score) for result in out] == [("b", 2.0), ("c", 0.5)],
          "withheld fold is absent and the distance-weighted and directed terms match")
    result = retriever.retrieve("", anchor_track_ids=["a", "c"])
    check(len(result) == 1 and result[0].track_id == "b" and isclose(result[0].score, 2.4),
          "multiple anchors use recency decay and played anchors are skipped")
    check(not retriever.retrieve("", anchor_track_ids=["a", "c"], session_track_ids=["b"]),
          "session tracks are excluded too")
    check([(r.track_id, r.score) for r in reopened.retrieve("", anchor_track_ids=["a"])] == [("b", 1.0)],
          "cached counts can be reused with different term weights")
    check([r.track_id for r in rescaled.retrieve("", anchor_track_ids=["a"])] == ["b", "c"],
          "scaling positive weights preserves the co-occurrence ranking")
    retriever.begin_session(sid, held_out=False)
    check({r.track_id for r in retriever.retrieve("", anchor_track_ids=["a"])} ==
          {"b", "c", "own-gold", "same-fold-only"}, "dev and blind scoring use the full training counts")


def check_resume(directory):
    import datasets

    sessions = FakeDS([make_session(i) for i in range(3)])
    ollama = types.ModuleType("recommender_v2.planner.ollama")
    llm = RecordingLLM()
    ollama.OllamaQwenPlanner = lambda model: llm
    out = Path(directory) / "resumed.csv"

    def fake_load(name, split=None):
        return sessions if split == "train" else {"train": sessions}

    with patch.object(datasets, "load_dataset", side_effect=fake_load), \
         patch("recommender_v2.retrievers.item_to_item.cooc_track.load_dataset", side_effect=fake_load), \
         patch.object(metadata, "load_track_metadata", return_value=METADATA), \
         patch.dict(sys.modules, {ollama.__name__: ollama}):
        retriever = CooccurrenceTrackRetriever(cache_dir=None)
        args = ["--output", str(out), "--models", "cooc_track"]
        with patch.object(build_dataset, "build_retrievers", return_value=[retriever]), \
             patch.object(retriever, "begin_session", wraps=retriever.begin_session) as begin:
            build_dataset.main(args + ["--num-sessions", "2"])
            begin.reset_mock()
            build_dataset.main(args + ["--num-sessions", "3", "--resume"])
            check([(call.args[0], call.kwargs["held_out"]) for call in begin.call_args_list] ==
                  [(session["session_id"], True) for session in sessions],
                  "resume selects the held-out fold for every session, including already-built sessions")
    meta = json.loads(Path(str(out) + ".meta.json").read_text(encoding="utf-8"))
    check(meta["feature_schema_version"] == FEATURE_SCHEMA_VERSION == 2, "dataset records schema version 2")
    check(not llm.violations, "the existing visibility canary passes with co-occurrence enabled")


def check_demo():
    import datasets
    import demo_one_query

    session = make_session(0)
    ollama = types.ModuleType("recommender_v2.planner.ollama")
    ollama.OllamaQwenPlanner = lambda model: RecordingLLM()
    calls = []
    retrieval_calls = []
    source = types.SimpleNamespace(begin_session=lambda sid, held_out: calls.append((sid, held_out)))

    def retrieve(**kwargs):
        retrieval_calls.append(list(calls))
        return QueryState(), {}

    recommender = types.SimpleNamespace(retrievers=[source], retrieve_for_ranking=retrieve,
                                        build_features=lambda *args, **kwargs: [],
                                        ranker=types.SimpleNamespace(predict=lambda features: []))
    with patch.object(datasets, "load_dataset", side_effect=[[], {"train": [session]}]), \
         patch.object(metadata, "load_track_metadata", return_value=METADATA), \
         patch.object(build_dataset, "build_retrievers", return_value=[source]), \
         patch("recommender_v2.RecommenderV2", return_value=recommender), \
         patch.dict(sys.modules, {ollama.__name__: ollama}), \
         patch.object(sys, "argv", ["demo_one_query.py", "--ranker", "unused", "--split", "train"]), \
         redirect_stdout(StringIO()):
        demo_one_query.main()
    check(retrieval_calls == [[(session["session_id"], True)]],
          "train demo excludes the session's fold before retrieval")

    with patch.object(datasets, "load_dataset") as load, redirect_stderr(StringIO()), \
         patch.object(sys, "argv", ["demo_one_query.py", "--ranker", "unused", "--session", str(FINAL_HOLDOUT.start)]):
        try:
            demo_one_query.main()
        except SystemExit as error:
            check(error.code == 2 and not load.called, "demo refuses the final holdout before loading data")
        else:
            check(False, "demo refuses the final holdout before loading data")


def check_features(directory):
    def track(artist, album):
        return {"artist_names": [artist], "album_names": [album]}

    catalogue = {
        "old": track("South", "Echo"), "middle": track("North", "Écho"),
        "last": track("NORTH", "Echo"), "a": track("North", "Echo"),
        "b": track("South", "Echo"), "c": track("North", "Elsewhere"),
        "played": track("North", "Echo"), "off-pool": track("North", "Echo"),
    }
    bm25_catalogue = {name: {"track_name": "Instrumental" if name == "a" else name,
                             "artist_name": entry["artist_names"], "album_name": entry["album_names"]}
                      for name, entry in catalogue.items()}
    with patch.object(SessionBM25Retriever, "_load_corpus", return_value=bm25_catalogue):
        bm25 = SessionBM25Retriever("test", ["all"], directory)
    bge = BGERetriever.__new__(BGERetriever)
    bge.track_ids = ["a", "b", "c"]
    bge.track_to_index = {track_id: index for index, track_id in enumerate(bge.track_ids)}
    bge.track_embeddings = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    bge._last_query = None

    anchors = ["old", "middle", "last"]
    candidates = ["a", "b", "c", "unknown"]
    planner = QueryState(raw_query="The instrumental!", semantic_query="semantic specificity")
    builder = FeatureBuilder(track_metadata=catalogue)
    builder.attach_embeddings([bm25, bge])
    results = {
        "bm25": [RetrievalResult("a", 3.0, 1, "bm25"), RetrievalResult("b", 2.0, 2, "bm25"),
                 RetrievalResult("b", 1.0, 3, "bm25")],
        "bge": [RetrievalResult("a", 1.0, 1, "bge"), RetrievalResult("c", -1.0, 2, "bge")],
    }
    encode_query = lambda query: torch.tensor([1., 0.]) if query == planner.semantic_query else torch.tensor([0., 1.])
    with patch.object(bge, "_encode_query", side_effect=encode_query) as encode, \
         patch.object(bm25.bm25_model, "get_scores", wraps=bm25.bm25_model.get_scores) as score:
        bge.retrieve(planner.semantic_query, topk=1)
        bm25.retrieve("", topk=1, query_state=planner, anchor_track_ids=anchors)
        rows = builder.build_candidate_features(results, candidate_ids=candidates, anchor_track_ids=anchors,
                                                session_track_ids=["played"], query_state=planner)
        a, b, c, unknown = rows
        check(encode.call_count == 1 and score.call_count == 2,
              "pool scores reuse one BGE query encoding and one BM25 catalogue pass per token set")
        check(a["fp_bm25_msg_score"] > 0 and b["fp_bm25_msg_score"] == 0 and b["fp_bm25_session_score"] > 0,
              "both BM25 scores cover candidates outside the BM25 result list")
        check(a["fp_bm25_msg_score_rank_in_pool"] == 1 and b["fp_bm25_msg_score_pct_in_pool"] == .5,
              "text ranks use competition ties and divide by the full pool size")
        check([row["fp_bge_cos"] for row in rows[:3]] == [1.0, 0.0, -1.0]
              and unknown["fp_bge_cos"] != unknown["fp_bge_cos"],
              "BGE cosine scores available for non-retrieved candidates and NaN for unknown embeddings")
        fallback = builder.build_candidate_features(results, candidate_ids=candidates, anchor_track_ids=anchors,
                                                    query_state=planner, planner_fallback=True)
        check(encode.call_args.args == (planner.raw_query,) and fallback[1]["fp_bge_cos"] == 1.0,
              "BGE full-pool score uses raw current user text when planning fails")

    check(a["artist_streak_last"] == a["album_streak_last"] == 2 and a["same_album_as_last_anchor"] == 1
          and a["artist_anchor_share"] == 2 / 3 and a["turns_since_artist_played"] == 1,
          "streak and recency features match three anchors with two consecutive album matches")
    check(b["artist_streak_last"] == b["album_streak_last"] == b["same_album_as_last_anchor"] == 0
          and b["turns_since_artist_played"] == 3 and a["album_unplayed_count"] == 2,
          "same title by another artist is separate, and album counts include off-pool tracks")
    check(all(row["last_anchors_same_album"] == row["last_anchors_same_artist"] == 1 for row in rows)
          and np.isnan(unknown["album_unplayed_count"]), "last-anchor pair is query-level and missing albums are NaN")
    check(b["bm25_rank_pct"] == 2 / 3 and np.isnan(c["bm25_rank_pct"]),
          "per-source rank denominator includes the full original retrieval list")
    z = np.array([row["rrf_sum_z"] for row in rows])
    check(isclose(z.mean(), 0, abs_tol=1e-9) and isclose(z.std(), 1) and z[0] > z[1] == z[2] > z[3],
          "RRF uses a per-query population z-score")
    check([row["num_sources_pct_in_pool"] for row in rows] == [.25, .5, .5, 1.0],
          "source-count percentiles rank higher agreement first and share ties")
    plain = FeatureBuilder().build_candidate_features(results, candidate_ids=candidates)
    names = builder.feature_names()
    check(names.index("cooc_track_present") == 165 and all(set(row) == {"track_id", *names} for row in rows + plain)
          and all(set(features) <= set(names) for features in FEATURE_GROUPS.values()),
          "Task 2 groups are appended and feature schema is fixed across model selections")


if __name__ == "__main__":
    torch.set_num_threads(2)
    with TemporaryDirectory() as temp:
        check_cooc(temp)
        check_resume(temp)
        check_demo()
        check_features(temp)
    if check.failed:
        raise SystemExit(1)
    print("All Task 2 checks passed.")
