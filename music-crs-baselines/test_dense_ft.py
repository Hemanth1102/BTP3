"""Offline checks for the dense text, training loss and fold-safe retrieval."""

from copy import deepcopy
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from recommender_v2 import RecommenderV2
from recommender_v2.dense_ft.artifacts import FULL_MODEL, N_FOLDS, Settings, create_manifest, model_record
from recommender_v2.dense_ft.encode import LiveIndex, _write_rankings, encode_inference, encode_oof, main as encode_main
from recommender_v2.dense_ft.retriever import DenseArtifactError, DenseFTRetriever
from recommender_v2.dense_ft.text import QueryText, TurnQuery, document_text, query_tokens, session_queries
from recommender_v2.dense_ft.train import batch_backward, contrastive_loss, train_fold
from recommender_v2.features import FEATURE_SCHEMA_VERSION, FeatureBuilder
from recommender_v2.planner import QueryPlanner, QueryState
from recommender_v2.ranker import RRFRanker
from recommender_v2.retrievers.lexical.transition_artist import session_fold
from test_training_pipeline import RecordingLLM, check, make_session


def check_text():
    original = make_session(3)
    changed = deepcopy(original)
    for msg in changed["conversations"]:
        if msg["turn_number"] > 4 or (msg["turn_number"] == 4 and msg["role"] != "user"):
            msg["content"] = "FORBIDDEN"
    catalogue = {m["content"]: {"track_name": str(m["turn_number"]), "artist_names": ["Artist"],
                                "album_names": ["Album"], "year": 2020, "genres": ["jazz"]}
                 for m in original["conversations"] if m["role"] == "music"}
    rows = list(session_queries([original], catalogue))
    changed_rows = list(session_queries([changed], catalogue))
    text = rows[3].text
    check(text == changed_rows[3].text and rows[3].gold != changed_rows[3].gold,
          "visible dense text is invariant to the current gold, current assistant and future messages")
    check(text.context.startswith("Recent listens: 3 \u2013 Artist (Album, 2020) | 2") and
          "\nSession so far: " in text.context and text.request.startswith("Looking for: USER_3_4"),
          "conversational fields have the required order and reverse anchor order")
    check(str(rows[0].text).startswith("Session so far: \nLooking for:") and
          document_text(next(iter(catalogue.values()))).endswith(" | year: 2020 | tags: jazz"),
          "turn 1 omits listens and document fields keep their order")
    blind = {**original, "conversations": [m for m in original["conversations"]
             if m["turn_number"] < 4 or (m["turn_number"] == 4 and m["role"] == "user")]}
    check(list(session_queries([blind], catalogue, blind=True))[0].text == text,
          "blind and labelled query text agree without needing target music")

    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False:
                                list(text.encode()) + ([0] if add_special_tokens else []))
    long = QueryText("Repeat this visible context. " * 50, "Looking for: a long request " * 10)
    expected_tail = list(("\n" + long.request).encode()) + [0]
    tokens = query_tokens(tokenizer, long, "Instruct:", 25)
    check(tokens[-len(expected_tail):] == expected_tail and len(tokens) > 25,
          "the request survives truncation even when it alone exceeds the nominal length")
    limited = query_tokens(tokenizer, QueryText("Earlier words " * 50, "Looking for: jazz"), "Instruct:", 45)
    check(len(limited) == 45 and bytes(limited).endswith(b"\nLooking for: jazz\x00"),
          "only the context is shortened when the request fits")

    with redirect_stderr(StringIO()):
        try:
            encode_main(["--split", "holdout"])
        except SystemExit as error:
            check(error.code == 2, "encoding final holdout is guarded before loading data")
        else:
            check(False, "encoding final holdout is guarded before loading data")


def check_loss():
    data = torch.eye(3)
    ids = torch.tensor([1, 1, 2])
    frequencies = torch.tensor([2 / 3, 2 / 3, 1 / 3]).log()
    logits = data / .05 - frequencies
    expected = (logits[0].logsumexp(0) - logits[0, :2].logsumexp(0)
                + logits[1].logsumexp(0) - logits[1, :2].logsumexp(0)
                + logits[2].logsumexp(0) - logits[2, 2]) / 3
    check(torch.allclose(contrastive_loss(data, data, ids, frequencies, .05), expected),
          "InfoNCE subtracts log-frequency and includes duplicate tracks as positives")

    class Tiny:
        device = torch.device("cpu")

        def __init__(self):
            self.model = torch.nn.Embedding(4, 3)

        def embed(self, token_ids):
            return F.normalize(self.model(torch.tensor(token_ids)), dim=-1)

    torch.manual_seed(7)
    direct = Tiny()
    replay = deepcopy(direct)
    settings = Settings(batch_size=3, micro_batch_size=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    a = batch_backward(direct, [0, 1, 2], [0, 0, 3], ids, frequencies, settings, scaler)
    b = batch_backward(replay, [0, 1, 2], [0, 0, 3], ids, frequencies,
                       replace(settings, micro_batch_size=1), scaler)
    check(abs(a - b) < 1e-6 and torch.allclose(direct.model.weight.grad, replay.model.weight.grad, atol=1e-6),
          "micro-batch replay preserves gradients from all logical batch negatives")


def check_retrieval(directory):
    root = Path(directory)
    sessions = [next(f"session-{i}" for i in range(100) if session_fold(f"session-{i}") == fold)
                for fold in range(N_FOLDS)]
    queries = [TurnQuery(sid, 2, QueryText("", f"Looking for: {sid}"), ("played",), "gold") for sid in sessions]
    catalogue = {"played": "played", "gold": "gold", "spare": "spare"}
    manifest = create_manifest(root, Settings(), queries, catalogue)
    trained_on = {}

    def save(folder):
        folder.mkdir(parents=True, exist_ok=True)

    mock_encoder = SimpleNamespace(model=SimpleNamespace(save_pretrained=save),
                                  tokenizer=SimpleNamespace(save_pretrained=save), device=torch.device("cpu"))
    selected = [*range(N_FOLDS), FULL_MODEL]
    with patch("recommender_v2.dense_ft.train.load_encoder", return_value=mock_encoder), \
         patch("recommender_v2.dense_ft.train.fit", side_effect=lambda encoder, qs, cat, config:
               trained_on.setdefault(selected[len(trained_on)], {q.session_id for q in qs}) and 1):
        for fold in selected:
            train_fold(root, manifest, queries, catalogue, fold, "cpu")

    class StubEncoder:
        def tokenize_query(self, text):
            return text.request[len("Looking for: "):]

        def embed(self, tokens):
            return torch.tensor([[1., 0.]] * len(tokens))

    def load_index(root, manifest, fold, device, **options):
        model_record(root, manifest, fold)

        class AssertingIndex(LiveIndex):
            def search(self, texts, played, topk=100):
                check(all(self.encoder.tokenize_query(text) not in trained_on[fold] for text in texts),
                      f"fold {fold} never scores a session it trained on")
                return super().search(texts, played, topk)

        return AssertingIndex(StubEncoder(), list(catalogue), torch.tensor([[1., 0.], [.8, .6], [0., 1.]]))

    unseen = TurnQuery("held-out", 2, QueryText("", "Looking for: held-out"), ("played",), "gold")
    with patch("recommender_v2.dense_ft.encode.load_index", side_effect=load_index) as load:
        encode_oof(root, manifest, queries, catalogue, range(N_FOLDS), "cpu", 64)
        encode_oof(root, manifest, queries, catalogue, range(N_FOLDS), "cpu", 64)
        check(load.call_count == N_FOLDS, "completed fold files are reused and the OOF parquet merges")
        encode_inference(root, manifest, "diagnostic", [unseen], catalogue, "full", "cpu", 64)
        live = DenseFTRetriever(root, mode="live", inference="rrf", metadata={})
        offline = DenseFTRetriever(root)
        q = queries[0]
        for source in (live, offline):
            result = source.retrieve(q.session_id, session_id=q.session_id, turn_number=2,
                                     anchor_track_ids=["played"], session_track_ids=["spare"])
            check([r.track_id for r in result] == ["gold"], "train scoring uses its fold and excludes both played sets")
        check([r.track_id for r in offline.retrieve("", session_id=unseen.session_id, turn_number=2)] ==
              ["gold", "spare"], "unseen precomputed scoring loads the full-model results")

        llm = RecordingLLM()
        rec = RecommenderV2([offline], planner=QueryPlanner(llm=llm, mode="llm"), ranker=RRFRanker())
        _, results = rec.retrieve_for_ranking(q.session_id, state=QueryState(),
                                             session_id=q.session_id, turn_number=2)
        rows = rec.build_features(results)
        check(not llm.calls and rec.feature_builder.feature_names() == FeatureBuilder().feature_names()
              and set(rows[0]) == {"track_id", *FeatureBuilder().feature_names()},
              "precomputed dense scoring uses the fixed schema and requires no planner call")

    invalid = root / "invalid"
    create_manifest(invalid, Settings(), queries, catalogue)
    _write_rankings(invalid / "oof_top100.parquet", manifest, "oof", [([q], [[("gold", .8)]], FULL_MODEL)])
    bad = DenseFTRetriever(invalid)
    try:
        bad.retrieve("", session_id=q.session_id, turn_number=2)
    except DenseArtifactError:
        check(True, "a training query row tagged with the full model is rejected")
    else:
        check(False, "a training query row tagged with the full model is rejected")
    try:
        offline.retrieve("", session_id=q.session_id, turn_number=99)
    except DenseArtifactError:
        check(True, "a missing OOF turn cannot fall back to full-model scores")
    else:
        check(False, "a missing OOF turn cannot fall back to full-model scores")
    names = FeatureBuilder().feature_names()
    check(FEATURE_SCHEMA_VERSION == 3 and names.index("dense_ft_present") == 210,
          "schema version 3 appends six fields after the Task 2 columns")
    for source in (live, offline, bad):
        source.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_text()
    check_loss()
    with TemporaryDirectory() as directory:
        check_retrieval(directory)
    if check.failed:
        raise SystemExit(1)
    print("All Task 3 checks passed.")
