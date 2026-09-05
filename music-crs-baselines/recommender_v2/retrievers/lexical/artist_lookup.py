import re
from collections import defaultdict
from typing import Dict, List, Optional

from datasets import load_dataset, concatenate_datasets

from ..base import BaseRetriever, RetrievalResult


class ArtistLookupRetriever(BaseRetriever):
    """
    Exact artist/entity retrieval.

    Builds lookup indexes once during initialization and reuses them
    for every retrieval call.
    """

    name = "artist_lookup"

    def __init__(
        self,
        dataset_name: str,
        split_types: List[str],
    ):
        self.dataset_name = dataset_name
        self.split_types = split_types

        print("Loading artist lookup...")

        dataset = load_dataset(dataset_name)

        combined = concatenate_datasets(
            [dataset[split] for split in split_types]
        )

        # normalized artist -> canonical artist name
        self.normalized_to_canonical: Dict[str, str] = {}

        # normalized artist -> tracks
        self.artist_to_tracks: Dict[str, List[str]] = defaultdict(list)

        for row in combined:
            track_id = row["track_id"]
            artists = row.get("artist_name", [])

            if isinstance(artists, str):
                artists = [artists]

            for artist in artists:
                if not artist:
                    continue

                normalized = self._normalize(artist)

                if not normalized:
                    continue

                self.normalized_to_canonical.setdefault(
                    normalized,
                    artist,
                )

                self.artist_to_tracks[
                    normalized
                ].append(track_id)

        # Remove duplicates while preserving order.
        for artist in self.artist_to_tracks:
            self.artist_to_tracks[artist] = list(
                dict.fromkeys(
                    self.artist_to_tracks[artist]
                )
            )

        self.normalized_artist_names = list(
            self.normalized_to_canonical.keys()
        )

        print(
            f"Artist lookup: "
            f"{len(self.artist_to_tracks)} artists"
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()

        text = re.sub(
            r"[^\w\s&'-]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        return text

    def _find_artists(
        self,
        query: str,
    ) -> List[str]:
        """
        Find artists mentioned in the query.

        Fast path:
            exact normalized query/name matching.

        Fallback:
            substring matching over catalog artist names.
        """

        normalized_query = self._normalize(query)

        if not normalized_query:
            return []

        # Fast path: query itself is exactly an artist name.
        if normalized_query in self.normalized_to_canonical:
            return [
                self.normalized_to_canonical[
                    normalized_query
                ]
            ]

        # For conversational queries, check whether a catalog
        # artist name occurs inside the query.
        matches = []

        for normalized_artist in self.normalized_artist_names:
            if normalized_artist in normalized_query:
                matches.append(
                    self.normalized_to_canonical[
                        normalized_artist
                    ]
                )

        return list(
            dict.fromkeys(matches)
        )

    def retrieve(
        self,
        query: str,
        topk: int = 100,
        user_id: Optional[str] = None,
        anchor_track_ids=None,
        session_track_ids=None,
    ) -> List[RetrievalResult]:

        matched_artists = self._find_artists(query)

        if not matched_artists:
            return []

        candidate_ids = []

        for artist in matched_artists:
            normalized = self._normalize(artist)

            for track_id in self.artist_to_tracks.get(
                normalized,
                [],
            ):
                if track_id not in candidate_ids:
                    candidate_ids.append(track_id)

                if len(candidate_ids) >= topk:
                    break

            if len(candidate_ids) >= topk:
                break

        return [
            RetrievalResult(
                track_id=track_id,
                score=1.0,
                rank=rank,
                source=self.name,
            )
            for rank, track_id in enumerate(
                candidate_ids,
                start=1,
            )
        ]