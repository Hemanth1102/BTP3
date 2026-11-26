"""Offline checks for the Task 1 retrievers and recall report."""

from copy import deepcopy
from math import isclose, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from recall_report import main as recall_main, measure_recall, write_report
from recommender_v2 import RecommenderV2
from recommender_v2.features import FeatureBuilder, KNOWN_RETRIEVERS
from recommender_v2.planner import QueryPlanner, QueryState
from recommender_v2.ranker import RRFRanker
from recommender_v2.retrievers.base import RetrievalResult
from recommender_v2.retrievers.item_to_item import embedding
from recommender_v2.retrievers.lexical import anchor_album
from recommender_v2.retrievers.lexical.session_bm25 import SessionBM25Retriever
from recommender_v2.splits import FINAL_HOLDOUT, TURNS
from recommender_v2.training.build_dataset import DEFAULT_MODELS, build_retrievers
from recommender_v2.training.turns import get_turn_context
from test_training_pipeline import RecordingLLM, check, make_session


ITEM_MODELS = [name for name in DEFAULT_MODELS if name.startswith("item_")]


def ids(results):
    return [result.track_id for result in results]


def make_catalogue():
    return [
        {"track_id": "old", "track_name": "First Track", "album_name": "Écho",
         "artist_name": ["North"], "tag_list": ["acoustic", "ballad"], "popularity": 100},
        {"track_id": "recent", "track_name": "Second Track", "album_name": "Echo",
         "artist_name": ["South"], "tag_list": ["jazz", "fusion"], "popularity": 90},
        {"track_id": "east", "track_name": "Eastern Strings", "album_name": "Echo",
         "artist_name": ["East"], "tag_list": ["instrumental"], "popularity": 10},
        {"track_id": "west", "track_name": "Western Strings", "album_name": "Echo",
         "artist_name": ["West"], "tag_list": ["instrumental"], "popularity": 20},
        {"track_id": "jazz", "track_name": "Shades", "album_name": "Special",
         "artist_name": ["Dusk"], "tag_list": ["jazz"], "release_date": "2012-05-06"},
        {"track_id": "generic", "track_name": "Piano", "album_name": "Best Of"},
        {"track_id": "other", "track_name": "Organ", "album_name": "best of"},
    ]


def check_lexical(directory):
    catalogue = make_catalogue()
    with patch.object(SessionBM25Retriever, "_load_corpus",
                      return_value={row["track_id"]: row for row in catalogue}):
        bm25 = SessionBM25Retriever("test", ["tracks"], directory, top_tags=1)
        untagged = SessionBM25Retriever("test", ["tracks"], directory, include_tags=False)
        reopened = SessionBM25Retriever("test", ["tracks"], directory, top_tags=1)
    check(bm25.message_tokens("And the JAZZ!") == ["jazz"], "message tokenization")
    check(bm25.session_tokens("strings", ["old", "recent"]) ==
          ["strings"] + ["south", "echo", "jazz"] * 3 + ["north", "écho", "acoustic"] * 2,
          "anchor order, top tags and repeat weights")
    state = QueryState(raw_query="jazz 2012", semantic_query="unrelated")
    check(ids(bm25.retrieve("unrelated", topk=1, query_state=state)) == ["jazz"],
          "turn 1 scores the current message against tags and release year")
    check(ids(reopened.retrieve("unrelated", topk=1, query_state=state)) == ["jazz"],
          "reopening the cached index preserves ranking")
    check(untagged.index_dir != bm25.index_dir and not untagged.retrieve("jazz"),
          "tag selection has its own cache and changes the indexed text")
    state.anchor_track_ids = ["old", "recent"]
    retrieved = bm25.retrieve("unrelated", query_state=state, session_track_ids=["jazz"])
    check(set(ids(retrieved)).isdisjoint({"old", "recent", "jazz"}) and retrieved,
          "session BM25 skips both anchors and other played tracks")

    album_catalogue = [
        dict(row, artist_name=["North"]) if row["track_id"] in {"east", "west"} else row
        for row in catalogue
    ] + [
        {"track_id": "north_recent", "album_name": "Echo", "artist_name": ["NORTH"]},
        {"track_id": "south_alternate", "album_name": "Echo", "artist_name": ["South"]},
    ]
    with patch.object(anchor_album, "load_catalog", return_value=album_catalogue):
        album = anchor_album.AnchorAlbumRetriever("test", ["tracks"])
        capped = anchor_album.AnchorAlbumRetriever("test", ["tracks"], max_album_tracks=3)
    results = album.retrieve("ignored", anchor_track_ids=["old", "north_recent"])
    check(ids(results) == ["west", "east"] and all(r.score == 1.5 for r in results),
          "album normalization, recency and popularity order")
    check(ids(album.retrieve("", anchor_track_ids=["old", "north_recent"],
                             session_track_ids=["west"])) == ["east"],
          "album retrieval skips both kinds of played tracks")
    check(ids(album.retrieve("", anchor_track_ids=["recent"])) == ["south_alternate"],
          "artists with the same album name retain separate track lists")
    check(not capped.retrieve("", anchor_track_ids=["old"]) and
          ids(capped.retrieve("", anchor_track_ids=["recent"])) == ["south_alternate"],
          "album size limit applies per artist and title")
    check(anchor_album.album_keys("Echo", ["North", "South"]) ==
          {("echo", "north"), ("echo", "south")}, "collaborations have one key per artist")

    numbered = [dict(row, track_number=number) for number, row in enumerate(album_catalogue, 1)]
    with patch.object(anchor_album, "load_catalog", return_value=numbered):
        by_track = anchor_album.AnchorAlbumRetriever("test", ["tracks"])
    check(ids(by_track.retrieve("", anchor_track_ids=["old", "north_recent"])) == ["east", "west"],
          "track number takes precedence over popularity when present")

    names = ["The Very Best of Queen", "Live at Wembley", "Greatest Hits Vol 2", "Alive", "Liveliness"]
    albums = [{"track_id": f"{i}-{track}", "album_name": name, "artist_name": ["Queen"]}
              for i, name in enumerate(names) for track in ("anchor", "candidate")]
    with patch.object(anchor_album, "load_catalog", return_value=albums):
        filtered = anchor_album.AnchorAlbumRetriever("test", ["tracks"])
        custom = anchor_album.AnchorAlbumRetriever("test", ["tracks"], generic_albums=("queen",))
    for i, name in enumerate(names):
        expected = [f"{i}-candidate"] if i >= 3 else []
        check(ids(filtered.retrieve("", anchor_track_ids=[f"{i}-anchor"])) == expected,
              f"generic phrase filter handles {name!r}")
    check(ids(custom.retrieve("", anchor_track_ids=["1-anchor"])) == ["1-candidate"],
          "generic phrase list can be configured")
    return bm25, album


def check_item_modes():
    spaces = (("cf-bpr", 128), ("audio-laion_clap", 512), ("metadata-qwen3_embedding_0.6b", 1024))
    rows = [{"track_id": track_id, **{column: vector + [0] * (dim - len(vector))
                                     for column, dim in spaces}}
            for track_id, vector in [("old", [1, 0]), ("recent", [0, 1]), ("east", [1, 0]),
                                     ("west", [0, 1]), ("jazz", [1, 1]), ("other", [-1, 0])]]
    with patch.object(embedding, "load_dataset", return_value={"all_tracks": rows}) as load:
        retrievers = build_retrievers(ITEM_MODELS)
    check(load.call_count == len(spaces), "mean, decay and max share each loaded embedding matrix")
    by_name = {r.name: r for r in retrievers}
    for name in ("item_cf_bpr", "item_clap", "item_metadata"):
        mean, decay, maximum = (by_name[name + suffix] for suffix in ("", "_decay", "_max"))
        check(mean.track_embeddings is decay.track_embeddings is maximum.track_embeddings,
              f"{name} variants point to the same tensor")
        args = {"anchor_track_ids": ["old", "recent"], "session_track_ids": ["east"]}
        check(ids(mean.retrieve("", topk=1, **args)) == ["jazz"], f"{name} legacy mean centroid")
        decay_results = decay.retrieve("", **args)
        check(ids(decay_results) == ["jazz", "west", "other"] and
              isclose(decay_results[1].score, 1 / sqrt(1 + 0.6 ** 2), abs_tol=1e-6),
              f"{name} decay centroid, score and played filtering")
        max_results = maximum.retrieve("", **args)
        check(ids(max_results) == ["west", "jazz", "other"] and max_results[0].score == 1,
              f"{name} max cosine and played filtering")
        check(not maximum.retrieve("", anchor_track_ids=["old", "a", "b", "c", "d"]),
              f"{name} maxsim only considers the last four anchors")
    return retrievers


def check_pipeline(retrievers, directory):
    llm = RecordingLLM()
    planner = QueryPlanner(llm=llm, mode="llm")
    rec = RecommenderV2(retrievers, planner=planner, ranker=RRFRanker())
    session = make_session(3)
    for turn, message in enumerate((m for m in session["conversations"] if m["role"] == "music")):
        message["content"] = ["old", "recent", "east", "jazz"][turn % 4]
    changed = deepcopy(session)
    for message in changed["conversations"]:
        if message["turn_number"] > 3 or (message["turn_number"] == 3 and message["role"] != "user"):
            message["content"] = "FORBIDDEN strings jazz western echo 2012"

    def retrieve(row):
        ctx = get_turn_context(row["conversations"], 3)
        return rec.retrieve_for_ranking(ctx.latest_user_message, conversation_history=ctx.history,
                                        anchor_track_ids=ctx.anchor_track_ids)

    state, results = retrieve(session)
    changed_state, changed_results = retrieve(changed)
    check(not llm.violations and state == changed_state and results == changed_results,
          "changing forbidden messages and tracks leaves the planner and real retrievers unchanged")
    full = rec.build_features(results, query_state=state, anchor_track_ids=["old", "recent"])
    reduced = RecommenderV2(retrievers[:1], planner=planner, ranker=RRFRanker())
    subset = reduced.build_features({"session_bm25": results["session_bm25"]}, query_state=state)
    names = FeatureBuilder().feature_names()
    check(rec.feature_builder.feature_names() == reduced.feature_builder.feature_names() == names
          and full and subset and all(set(row) == {"track_id", *names} for row in full + subset),
          "the full and reduced retriever sets produce identical feature schemas")
    new_sources = {r.name for r in retrievers if getattr(r, "mode", None) != "mean"}
    original = FeatureBuilder(retriever_names=[n for n in KNOWN_RETRIEVERS if n not in new_sources])
    original = original.feature_names()
    check(names[:len(original)] == original, "new columns are appended after the legacy schema")

    calls = llm.calls
    cache = {f"{session['session_id']}:1": {"state": QueryState(raw_query="ignored").to_dict()}}
    rows, by_turn, misses = measure_recall([session], rec, cache)
    check(llm.calls == calls and misses == len(TURNS) - 1, "recall report supplies states without the LLM")
    write_report(Path(directory), "tiny", 100, range(1), rows, by_turn, misses)
    check((Path(directory) / "recall_tiny.csv").is_file() and
          (Path(directory) / "recall_tiny.md").is_file(), "report outputs CSV and Markdown")
    with patch("sys.stderr"):
        try:
            recall_main(["--start-session", str(FINAL_HOLDOUT.start), "--num-sessions", "1"])
        except SystemExit as error:
            check(error.code == 2, "recall CLI refuses the final holdout before loading data")
        else:
            check(False, "recall CLI refuses the final holdout before loading data")

    class FixedRetriever:
        def __init__(self, name, tracks):
            self.name = name
            self.tracks = tracks

        def retrieve(self, **kwargs):
            return [RetrievalResult(track, 1.0, rank, self.name)
                    for rank, track in enumerate(self.tracks, 1)]

    fixed = RecommenderV2([FixedRetriever("first", ["old"]),
                           FixedRetriever("second", ["old", "recent"])],
                          planner=planner, ranker=RRFRanker())
    (first, second, union), by_turn, _ = measure_recall([session], fixed, {})
    check(first["recall"] == 0.25 and first["marginal_recall"] == 0
          and second["recall"] == 0.5 and second["marginal_recall"] == 0.25,
          "only exclusive hits count toward a retriever's marginal recall")
    check(union["recall"] == 0.5 and union["mean_list_length"] == 2
          and list(by_turn.values()) == [1, 1, 0, 0] * 2,
          "union reports distinct candidates and breaks down recall by turn")


if __name__ == "__main__":
    torch.set_num_threads(2)
    with TemporaryDirectory() as directory:
        lexical = check_lexical(directory)
        items = check_item_modes()
        check_pipeline([*lexical, *items], directory)
    if check.failed:
        raise SystemExit(1)
    print("All Task 1 checks passed.")
