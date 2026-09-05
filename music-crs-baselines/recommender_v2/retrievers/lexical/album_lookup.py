import re
from collections import defaultdict
from typing import Dict, List, Optional

from datasets import load_dataset, concatenate_datasets

from ..base import BaseRetriever, RetrievalResult


class AlbumLookupRetriever(BaseRetriever):
    """
    Exact album/entity retrieval.

    Builds lookup indexes once during initialization and reuses them
    for every retrieval call.
    """

    name = "album_lookup"

    def __init__(
        self,
        dataset_name: str,
        split_types: List[str],
    ):
        self.dataset_name = dataset_name
        self.split_types = split_types

        print("Loading album lookup...")

        dataset = load_dataset(dataset_name)

        combined = concatenate_datasets(
            [dataset[split] for split in split_types]
        )

        # normalized album -> canonical album name
        self.normalized_to_canonical: Dict[str, str] = {}

        # normalized album -> tracks
        self.album_to_tracks: Dict[str, List[str]] = defaultdict(list)

        for row in combined:
            track_id = row["track_id"]
            albums = row.get("album_name", [])

            if isinstance(albums, str):
                albums = [albums]

            for album in albums:
                if not album:
                    continue

                normalized = self._normalize(album)

                if not normalized:
                    continue

                self.normalized_to_canonical.setdefault(
                    normalized,
                    album,
                )

                self.album_to_tracks[
                    normalized
                ].append(track_id)

        for album in self.album_to_tracks:
            self.album_to_tracks[album] = list(
                dict.fromkeys(
                    self.album_to_tracks[album]
                )
            )

        self.normalized_album_names = list(
            self.normalized_to_canonical.keys()
        )

        print(
            f"Album lookup: "
            f"{len(self.album_to_tracks)} albums"
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

    def _find_albums(
        self,
        query: str,
    ) -> List[str]:

        normalized_query = self._normalize(query)

        if not normalized_query:
            return []

        # Fast path.
        if normalized_query in self.normalized_to_canonical:
            return [
                self.normalized_to_canonical[
                    normalized_query
                ]
            ]

        matches = []

        for normalized_album in self.normalized_album_names:
            if normalized_album in normalized_query:
                matches.append(
                    self.normalized_to_canonical[
                        normalized_album
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

        matched_albums = self._find_albums(query)

        if not matched_albums:
            return []

        candidate_ids = []

        for album in matched_albums:
            normalized = self._normalize(album)

            for track_id in self.album_to_tracks.get(
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