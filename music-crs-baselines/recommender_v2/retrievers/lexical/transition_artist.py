import hashlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

from datasets import load_dataset, concatenate_datasets

from ..base import BaseRetriever, RetrievalResult

CHALLENGE_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
N_FOLDS = 5


def session_fold(session_id, n_folds: int = N_FOLDS) -> int:
    """Deterministic fold of a session (stable across runs and machines)."""
    return int(hashlib.md5(str(session_id).encode()).hexdigest()[:8], 16) % n_folds


def _contrib(sequences: Sequence[Sequence[str]], track_artists) -> Dict[str, Counter]:
    """
    trans[a][b] += 1/(j-i) whenever an artist-b track is played j-i turns after
    an artist-a track in the same session (a != b: same-artist continuity is
    handled by the anchor_artist source).
    """
    trans: Dict[str, Counter] = defaultdict(Counter)
    for seq in sequences:
        arts = [track_artists.get(t, ()) for t in seq]
        for j, bj in enumerate(arts):
            for i in range(j):
                w = 1.0 / (j - i)
                for a in arts[i]:
                    for b in bj:
                        if a != b:
                            trans[a][b] += w
    return trans


def _gold_sequence(session) -> List[str]:
    by_turn = {m["turn_number"]: m["content"]
               for m in session["conversations"] if m["role"] == "music"}
    return [by_turn[t] for t in sorted(by_turn)]


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return [str(value)] if str(value).strip() else []


class TransitionArtistRetriever(BaseRetriever):
    """
    "Which artists follow which", learned from the training conversations.

    Given the artists already played in the session, ranks other artists by
    their recency-weighted transition probability and returns those artists'
    tracks (more popular first).

    Leakage control: when the dataset builder labels a TRAIN session it calls
    `begin_session(session_id, held_out=True)`. The session's fold is then
    subtracted from the global table, so a session never sees its own (or its
    fold's) transitions. Dev / blind sessions are not in the training data, and
    at inference time begin_session is not called, so the full table is used.
    """

    name = "transition_artist"

    def __init__(self, dataset_name: str, split_types: List[str],
                 challenge_dataset: str = CHALLENGE_DATASET, n_folds: int = N_FOLDS):
        print("Loading transition-artist index...")

        meta = load_dataset(dataset_name)
        meta = concatenate_datasets([meta[s] for s in split_types])

        track_artists, artist_tracks, popularity = {}, defaultdict(list), {}
        for row in meta:
            tid = row["track_id"]
            arts = [a.strip().lower() for a in _as_list(row.get("artist_name"))]
            arts = tuple(a for a in arts if a)
            track_artists[tid] = arts
            try:
                popularity[tid] = float(row.get("popularity") or 0.0)
            except (TypeError, ValueError):
                popularity[tid] = 0.0
            for a in arts:
                artist_tracks[a].append(tid)

        train = load_dataset(challenge_dataset)["train"]
        sessions = [(s["session_id"], _gold_sequence(s)) for s in train]

        self._setup(track_artists, artist_tracks, popularity, sessions, n_folds)

        print(f"Transition-artist index: {len(self.full)} source artists "
              f"from {len(sessions)} training sessions")

    def _setup(self, track_artists, artist_tracks, popularity, sessions, n_folds=N_FOLDS):
        self.track_artists = track_artists
        self.artist_tracks = artist_tracks
        self.popularity = popularity
        self.n_folds = n_folds

        by_fold = defaultdict(list)
        for sid, seq in sessions:
            by_fold[session_fold(sid, n_folds)].append(seq)

        self.delta = {f: _contrib(seqs, track_artists) for f, seqs in by_fold.items()}
        self.full: Dict[str, Counter] = defaultdict(Counter)
        for d in self.delta.values():
            for a, c in d.items():
                self.full[a].update(c)

        self.full_tot = {a: sum(c.values()) for a, c in self.full.items()}
        self.delta_tot = {f: {a: sum(c.values()) for a, c in d.items()}
                          for f, d in self.delta.items()}
        self.fold: Optional[int] = None

    def begin_session(self, session_id, held_out: bool = True) -> None:
        self.fold = session_fold(session_id, self.n_folds) if held_out else None

    def retrieve(
        self,
        query: str,
        topk: int = 100,
        user_id: Optional[str] = None,
        anchor_track_ids: Optional[List[str]] = None,
        session_track_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:

        anchors = list(anchor_track_ids or [])
        if not anchors:
            return []

        played = set(anchors) | set(session_track_ids or [])
        anchor_artists = [self.track_artists.get(t, ()) for t in anchors]
        played_artists = {a for arts in anchor_artists for a in arts}

        d = self.delta.get(self.fold, {}) if self.fold is not None else {}
        dt = self.delta_tot.get(self.fold, {}) if self.fold is not None else {}

        score: Counter = Counter()
        n = len(anchors)

        for idx, arts in enumerate(anchor_artists):
            age = n - idx                       # most recent anchor: 1
            for a in arts:
                tot = self.full_tot.get(a, 0.0) - dt.get(a, 0.0)
                if tot <= 1e-9:
                    continue
                dd = d.get(a)
                for b, c in self.full[a].items():
                    if dd is not None:
                        c = c - dd.get(b, 0.0)
                    if c > 1e-9 and b not in played_artists:
                        score[b] += c / tot / age

        results: List[RetrievalResult] = []
        for artist, s in score.most_common():
            tracks = sorted(
                (t for t in self.artist_tracks.get(artist, ()) if t not in played),
                key=lambda t: -self.popularity.get(t, 0.0),
            )
            for t in tracks:
                results.append(RetrievalResult(t, float(s), len(results) + 1, self.name))
                if len(results) >= topk:
                    return results
        return results
