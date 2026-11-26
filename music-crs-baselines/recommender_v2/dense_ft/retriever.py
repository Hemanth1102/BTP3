"""Dense retriever with mandatory session identity and model-fold verification."""

import math
from functools import lru_cache
from itertools import groupby
from pathlib import Path

from .artifacts import FOLD_ENSEMBLE, FULL_MODEL, INFERENCE_SPLITS, N_FOLDS, load_manifest, open_rankings
from .text import query_text
from ..retrievers.base import BaseRetriever, RetrievalResult
from ..retrievers.lexical.transition_artist import session_fold
from ..training.turns import session_query_id


class DenseArtifactError(ValueError):
    """Dense scores cannot be trusted for the requested query."""


class _RankingTable:
    def __init__(self, path, manifest, source):
        self.file = open_rankings(path, manifest, source)
        self.locations = {}
        self.groups = lru_cache(maxsize=8)(self._read_group)
        for group in range(self.file.num_row_groups):
            ids = self.file.read_row_group(group, columns=["query_id"])["query_id"].to_pylist()
            start = 0
            for query_id, rows in groupby(ids):
                stop = start + sum(1 for _ in rows)
                if query_id in self.locations:
                    self.close()
                    raise DenseArtifactError(f"Query {query_id} is split or duplicated in {path}.")
                self.locations[query_id] = (group, start, stop)
                start = stop

    def _read_group(self, group):
        return self.file.read_row_group(group)

    def close(self):
        self.groups.cache_clear()
        self.file.close()

    def lookup(self, session_id, turn, model_fold):
        query_id = session_query_id(session_id, turn)
        if query_id not in self.locations:
            raise DenseArtifactError(f"No dense ranking for query {query_id} from the required model.")
        group, start, stop = self.locations[query_id]
        rows = self.groups(group).slice(start, stop - start).to_pylist()
        for rank, row in enumerate(rows, 1):
            if (row["session_id"] != session_id or row["turn_number"] != turn or row["fold"] != model_fold or
                    row["rank"] != rank or not row["track_id"] or not math.isfinite(row["score"])):
                raise DenseArtifactError(f"Invalid dense-ranking provenance or ordering for query {query_id}.")
        return [(row["track_id"], row["score"]) for row in rows]


class DenseFTRetriever(BaseRetriever):
    name = "dense_ft"
    uses_turn_context = True

    def __init__(self, root="cache/dense_ft", *, mode="precomputed", split="diagnostic",
                 inference="full", device="cpu", metadata=None):
        if mode not in ("precomputed", "live") or inference not in ("full", "rrf") or split not in INFERENCE_SPLITS:
            raise ValueError("Invalid dense retriever mode, inference strategy or split.")
        self.root = Path(root)
        self.manifest = load_manifest(self.root)
        self.mode = mode
        self.split = split
        self.inference = inference
        self.device = device
        self.metadata = metadata
        self.tables = {}
        self.indexes = {}

    def close(self):
        for table in self.tables.values():
            table.close()
        self.tables.clear()
        self.indexes.clear()

    def _table(self, source):
        if source not in self.tables:
            name = "oof_top100.parquet" if source == "oof" else f"full_top100_{self.split}.parquet"
            try:
                self.tables[source] = _RankingTable(self.root / name, self.manifest, source)
            except (FileNotFoundError, ValueError) as error:
                raise DenseArtifactError(str(error)) from error
        return self.tables[source]

    def _index(self, fold):
        from .encode import load_index

        if fold not in self.indexes:
            try:
                self.indexes[fold] = load_index(self.root, self.manifest, fold, self.device)
            except (FileNotFoundError, ValueError) as error:
                raise DenseArtifactError(str(error)) from error
        return self.indexes[fold]

    def retrieve(self, query, topk=100, user_id=None, anchor_track_ids=None, session_track_ids=None,
                 *, session_id=None, turn_number=None, conversation_history=None):
        if session_id is None or not isinstance(turn_number, int) or turn_number < 1:
            raise DenseArtifactError("Dense retrieval requires the session ID and current turn to verify the fold.")
        session_id = str(session_id)
        training_fold = self.manifest["training_sessions"].get(session_id)
        if training_fold is not None and training_fold != session_fold(session_id):
            raise DenseArtifactError(f"Invalid held-out fold for training session {session_id}.")
        if topk <= 0:
            return []
        model_fold = training_fold if training_fold is not None else (
            FULL_MODEL if self.inference == "full" else FOLD_ENSEMBLE)
        played = set(anchor_track_ids or ()) | set(session_track_ids or ())
        if self.mode == "precomputed":
            source = "oof" if training_fold is not None else model_fold
            ranking = self._table(source).lookup(session_id, turn_number, model_fold)
        else:
            if self.metadata is None:
                from ..training.metadata import load_track_metadata

                self.metadata = load_track_metadata()
            text = query_text(query, conversation_history, anchor_track_ids, turn_number, self.metadata)
            if model_fold == FOLD_ENSEMBLE:
                from .encode import rrf

                ranking = rrf(self._index(fold).search([text], [played])[0] for fold in range(N_FOLDS))
            else:
                ranking = self._index(model_fold).search([text], [played], topk=topk)[0]
        available = [(track_id, score) for track_id, score in ranking if track_id not in played][:topk]
        return [RetrievalResult(track_id, score, rank, self.name)
                for rank, (track_id, score) in enumerate(available, 1)]
