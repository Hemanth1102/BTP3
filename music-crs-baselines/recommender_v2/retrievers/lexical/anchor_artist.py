from collections import defaultdict
from typing import Dict, List, Optional

from datasets import load_dataset, concatenate_datasets

from ..base import BaseRetriever, RetrievalResult


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return [str(value)] if str(value).strip() else []


def _popularity(row) -> float:
    try:
        return float(row.get("popularity") or 0.0)
    except (TypeError, ValueError):
        return 0.0


class AnchorArtistRetriever(BaseRetriever):
    """
    Session artist continuity: proposes every track by an artist that was
    already played earlier in the SAME conversation (the anchors).

    In the training data the gold track shares an artist with an earlier track
    of the session in ~40% of the turns, so this is a strong, query-independent
    candidate source. Anchors are weighted by recency (most recent first);
    within one anchor artist, more popular tracks come first.
    """

    name = "anchor_artist"

    def __init__(self, dataset_name: str, split_types: List[str]):
        print("Loading anchor-artist index...")

        dataset = load_dataset(dataset_name)
        combined = concatenate_datasets([dataset[s] for s in split_types])

        self.track_artists: Dict[str, List[str]] = {}
        self.artist_tracks: Dict[str, List[str]] = defaultdict(list)
        self.popularity: Dict[str, float] = {}

        for row in combined:
            tid = row["track_id"]
            artists = [a.strip().lower() for a in _as_list(row.get("artist_name"))]
            artists = [a for a in artists if a]
            self.track_artists[tid] = artists
            self.popularity[tid] = _popularity(row)
            for a in artists:
                self.artist_tracks[a].append(tid)

        for a, tracks in self.artist_tracks.items():
            self.artist_tracks[a] = list(dict.fromkeys(tracks))

        print(f"Anchor-artist index: {len(self.artist_tracks)} artists, "
              f"{len(self.track_artists)} tracks")

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

        played = set(session_track_ids or []) | set(anchors)

        scores: Dict[str, float] = {}

        # Most recent anchor first: weight 1, 1/2, 1/3, ...
        for age, anchor in enumerate(reversed(anchors), start=1):
            for artist in self.track_artists.get(anchor, []):
                for tid in self.artist_tracks.get(artist, []):
                    if tid in played:
                        continue
                    # Sum over anchors: an artist played several times counts more.
                    scores[tid] = scores.get(tid, 0.0) + 1.0 / age

        ranked = sorted(
            scores,
            key=lambda t: (-scores[t], -self.popularity.get(t, 0.0)),
        )[:topk]

        return [
            RetrievalResult(t, float(scores[t]), r + 1, self.name)
            for r, t in enumerate(ranked)
        ]
