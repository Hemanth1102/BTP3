"""Album continuation from previously played tracks."""

import re
from collections import defaultdict

from ...text_norm import album_keys, normalize_name
from ..base import BaseRetriever, RetrievalResult
from .lookup_base import load_catalog, popularity_of


GENERIC_ALBUMS = ("", "greatest hits", "best of", "live", "remastered", "deluxe")


def _track_number(row, column):
    try:
        return int(row.get(column))
    except (TypeError, ValueError):
        return float("inf")


class AnchorAlbumRetriever(BaseRetriever):
    name = "anchor_album"

    def __init__(self, dataset_name, split_types, *, generic_albums=GENERIC_ALBUMS,
                 max_album_tracks=60):
        catalogue = load_catalog(dataset_name, split_types)
        columns = getattr(catalogue, "column_names", None)
        if columns is None:
            columns = sorted({key for row in catalogue for key in row})
        number_column = next((column for column in ("track_number", "track_no")
                              if column in columns), None)
        if number_column is None:
            print(f"Anchor album uses popularity order. Track metadata columns: {columns}")

        phrases = sorted({normalize_name(name) for name in generic_albums} - {""})
        ignored = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, phrases)) + r")(?!\w)") if phrases else None
        album_rows = defaultdict(dict)
        self.track_albums = {}
        for row in catalogue:
            track_id = row["track_id"]
            albums = {key for key in album_keys(row.get("album_name"), row.get("artist_name"))
                      if ignored is None or not ignored.search(key[0])}
            self.track_albums[track_id] = sorted(albums)
            popularity = popularity_of(row)
            order = (_track_number(row, number_column),
                     -popularity if popularity is not None else float("inf"), track_id)
            for album in albums:
                album_rows[album][track_id] = order
        self.album_tracks = {
            album: sorted(tracks, key=tracks.__getitem__)
            for album, tracks in album_rows.items() if len(tracks) <= max_album_tracks
        }

    def retrieve(self, query, topk=100, user_id=None, anchor_track_ids=None,
                 session_track_ids=None):
        if topk <= 0:
            return []
        anchors = anchor_track_ids or ()
        played = set(anchors) | set(session_track_ids or ())
        weights = defaultdict(float)
        for age, anchor in enumerate(reversed(anchors), start=1):
            for album in self.track_albums.get(anchor, ()):
                if album in self.album_tracks:
                    weights[album] += 1 / age

        candidates = {}
        for album in sorted(weights, key=weights.__getitem__, reverse=True):
            for track_id in self.album_tracks[album]:
                if track_id not in played:
                    candidates.setdefault(track_id, weights[album])
                    if len(candidates) == topk:
                        break
            if len(candidates) == topk:
                break
        return [RetrievalResult(track_id, score, rank, self.name)
                for rank, (track_id, score) in enumerate(candidates.items(), start=1)]
