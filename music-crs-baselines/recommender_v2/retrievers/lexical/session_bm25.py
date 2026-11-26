"""Planner-free BM25 over the latest user message and recent anchors."""

import hashlib
import json

import bm25s
import numpy as np

from ...training.metadata import _parse_year
from ..base import RetrievalResult
from .bm25 import BM25Retriever
from .entity_index import as_list


class SessionBM25Retriever(BM25Retriever):
    name = "session_bm25"
    uses_query_state = True

    def __init__(self, dataset_name, split_types, cache_dir="./cache", *,
                 num_anchors=3, repeat_weights=(3, 2, 1), include_tags=True, top_tags=5):
        if num_anchors < 0 or num_anchors > len(repeat_weights):
            raise ValueError("Provide a repeat weight for each recent anchor.")
        if any(weight < 0 for weight in repeat_weights) or top_tags < 0:
            raise ValueError("Repeat weights and tag counts must be nonnegative.")
        self.num_anchors = num_anchors
        self.repeat_weights = tuple(repeat_weights)
        self.include_tags = include_tags
        self.top_tags = top_tags
        fields = ["track_name", "artist_name", "album_name", "release_date"]
        if include_tags:
            fields.append("tag_list")
        super().__init__(dataset_name, split_types, fields, cache_dir)
        self._anchor_tokens = {}

    def _corpus_cache_key(self):
        # The remaining options affect the query only and can reuse the same index.
        config = [1, self.dataset_name, self.split_types, self.corpus_types]
        return hashlib.sha256(json.dumps(config).encode()).hexdigest()[:16]

    def _stringify_metadata(self, metadata):
        year = _parse_year(metadata.get("release_date"))
        return " ".join([
            *as_list(metadata.get("track_name")),
            *as_list(metadata.get("artist_name")),
            *as_list(metadata.get("album_name")),
            str(year) if year is not None else "",
            *(as_list(metadata.get("tag_list")) if self.include_tags else []),
        ])

    @staticmethod
    def message_tokens(message):
        return bm25s.tokenize([message or ""], return_ids=False, show_progress=False)[0]

    def _tokens_for_anchor(self, track_id):
        if track_id not in self._anchor_tokens:
            metadata = self.metadata_dict.get(track_id, {})
            parts = as_list(metadata.get("artist_name")) + as_list(metadata.get("album_name"))
            if self.include_tags:
                parts += as_list(metadata.get("tag_list"))[:self.top_tags]
            self._anchor_tokens[track_id] = self.message_tokens(" ".join(parts))
        return self._anchor_tokens[track_id]

    def session_tokens(self, message, anchors):
        tokens = self.message_tokens(message)
        for track_id, repeats in zip(reversed(anchors or []),
                                     self.repeat_weights[:self.num_anchors]):
            tokens.extend(self._tokens_for_anchor(track_id) * repeats)
        return tokens

    def score_tokens(self, tokens):
        """One BM25 score per track in index order, including zero matches."""
        if not tokens:
            return np.zeros(len(self.track_ids), dtype=np.float32)
        return self.bm25_model.get_scores(tokens)

    def retrieve(self, query, topk=100, user_id=None, anchor_track_ids=None,
                 session_track_ids=None, query_state=None):
        if topk <= 0:
            return []
        anchors = anchor_track_ids
        if anchors is None and query_state is not None:
            anchors = query_state.anchor_track_ids
        played = set(session_track_ids or ()) | set(anchors or ())
        message = query_state.raw_query if query_state is not None else query
        tokens = self.session_tokens(message, anchors)
        if not tokens:
            return []
        scores = self.score_tokens(tokens)
        matching = np.flatnonzero(scores > 0)
        ordered = matching[np.argsort(-scores[matching], kind="stable")]
        results = []
        for index in ordered:
            track_id = self.track_ids[index]
            if track_id not in played:
                results.append(RetrievalResult(track_id, float(scores[index]),
                                               len(results) + 1, self.name))
                if len(results) == topk:
                    break
        return results
