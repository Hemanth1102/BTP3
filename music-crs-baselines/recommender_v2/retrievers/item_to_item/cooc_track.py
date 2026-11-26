"""Cross-fitted catalogue retrieval from within-session track pairs."""

import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile

from datasets import load_dataset

from ...splits import TRAIN_SPLIT
from ..base import BaseRetriever, RetrievalResult
from ..lexical.transition_artist import CHALLENGE_DATASET, N_FOLDS, _gold_sequence, session_fold


def _new_counts():
    return Counter(), defaultdict(Counter), defaultdict(Counter)


def _count_folds(sessions):
    folds = [_new_counts() for _ in range(N_FOLDS)]
    for session in sessions:
        counts, cooc, transitions = folds[session_fold(session["session_id"])]
        tracks = _gold_sequence(session)
        counts.update(set(tracks))
        for right, target in enumerate(tracks):
            for left in range(right):
                anchor = tracks[left]
                if anchor == target:
                    continue
                weight = 1 / (right - left)
                cooc[anchor][target] += weight
                cooc[target][anchor] += weight
            if right and tracks[right - 1] != target:
                transitions[tracks[right - 1]][target] += 1
    return folds


def _load_folds(dataset_name, sessions, cache_dir):
    if cache_dir is None:
        return _count_folds(sessions)

    config = (1, dataset_name, TRAIN_SPLIT, N_FOLDS, sessions._fingerprint)
    key = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:16]
    path = Path(cache_dir) / "cooc_track" / f"{key}.json.gz"
    if path.is_file():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as data:
                return json.load(data)
        except (EOFError, OSError, ValueError):
            # A damaged cache can be rebuilt from the same training split.
            pass

    folds = _count_folds(sessions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as pending:
        temporary = Path(pending.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as data:
            json.dump(folds, data, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return folds


class CooccurrenceTrackRetriever(BaseRetriever):
    """Rank tracks using symmetric co-occurrences and directed next-track counts."""

    name = "cooc_track"

    def __init__(self, dataset_name=CHALLENGE_DATASET, *, cache_dir="./cache", decay=0.7,
                 cooc_weight=1.0, next_weight=1.0):
        if not 0 <= decay <= 1 or not (cooc_weight >= 0 and next_weight >= 0):
            raise ValueError("Use a decay in [0, 1] and nonnegative pair weights.")
        self.decay = decay
        self.cooc_weight = cooc_weight
        self.next_weight = next_weight

        train = load_dataset(dataset_name, split=TRAIN_SPLIT)
        self.folds = _load_folds(dataset_name, train, cache_dir)
        self.counts, self.cooc, self.transitions = _new_counts()
        for counts, cooc, transitions in self.folds:
            self.counts.update(counts)
            for anchor, neighbors in cooc.items():
                self.cooc[anchor].update(neighbors)
            for anchor, neighbors in transitions.items():
                self.transitions[anchor].update(neighbors)
        self.fold = None

    def begin_session(self, session_id, held_out=True):
        self.fold = session_fold(session_id) if held_out else None

    def retrieve(self, query, topk=100, user_id=None, anchor_track_ids=None,
                 session_track_ids=None):
        anchors = anchor_track_ids or ()
        if not anchors or topk <= 0:
            return []

        held_counts, held_cooc, held_next = self.folds[self.fold] if self.fold is not None else ({}, {}, {})
        played = set(anchors) | set(session_track_ids or ())
        scores = Counter()
        for age, anchor in enumerate(reversed(anchors)):
            n_anchor = self.counts.get(anchor, 0) - held_counts.get(anchor, 0)
            weight = self.decay ** age
            if n_anchor <= 0 or weight == 0:
                continue
            cooc = self.cooc.get(anchor, {})
            transitions = self.transitions.get(anchor, {})
            excluded_cooc = held_cooc.get(anchor, {})
            excluded_next = held_next.get(anchor, {})
            for target in (cooc.keys() | transitions.keys()) - played:
                n_target = self.counts[target] - held_counts.get(target, 0)
                if n_target <= 0:
                    continue
                within = cooc.get(target, 0) - excluded_cooc.get(target, 0)
                # Fold subtraction can leave a small residual for an absent pair.
                within = within if within > 1e-9 else 0.0
                following = transitions.get(target, 0) - excluded_next.get(target, 0)
                score = (self.cooc_weight * within / sqrt(n_anchor * n_target)
                         + self.next_weight * following / n_anchor)
                if score > 0:
                    scores[target] += weight * score

        ranked = sorted(scores, key=lambda tid: (-scores[tid], tid))[:topk]
        return [RetrievalResult(tid, scores[tid], rank, self.name)
                for rank, tid in enumerate(ranked, 1)]
