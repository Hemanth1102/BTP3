"""
Track-metadata lookup used by the metadata/entity features
(artist_match, album_match, genre_match, year_in_range, ...).

Reads the same catalog the BM25 / lookup retrievers use
(`talkpl-ai/TalkPlayData-Challenge-Track-Metadata`, split `all_tracks`) and
returns

    {track_id: {"artist_names": [...], "album_names": [...],
                "genres": [...], "year": int | None}}

Column names are the ones the repo already relies on
(track_name / artist_name / album_name / tag_list / release_date). The loader
prints what it found so a schema surprise is visible instead of silently
turning every metadata feature into NaN.
"""

import re
from typing import Any, Dict, Optional

ITEM_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"

_YEAR = re.compile(r"(1[89]\d\d|20\d\d)")


def _as_list(value: Any):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    value = str(value).strip()
    return [value] if value else []


def _parse_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    m = _YEAR.search(str(value))
    return int(m.group(1)) if m else None


def load_track_metadata(
    dataset_name: str = ITEM_DATASET,
    split: str = "all_tracks",
) -> Dict[str, Dict[str, Any]]:

    from datasets import load_dataset

    rows = load_dataset(dataset_name, split=split)

    print(f"Track metadata columns: {rows.column_names}")

    needed = {"track_id", "artist_name", "album_name", "tag_list", "release_date"}
    absent = needed - set(rows.column_names)
    if absent:
        raise RuntimeError(
            f"Track metadata is missing expected columns {sorted(absent)}; "
            f"update training/metadata.py for the real schema "
            f"(found: {rows.column_names})."
        )

    out: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        out[row["track_id"]] = {
            "artist_names": _as_list(row["artist_name"]),
            "album_names": _as_list(row["album_name"]),
            "genres": _as_list(row["tag_list"]),
            "year": _parse_year(row["release_date"]),
        }

    n = len(out)
    with_year = sum(1 for v in out.values() if v["year"])
    with_tags = sum(1 for v in out.values() if v["genres"])
    print(
        f"Track metadata: {n} tracks, "
        f"{with_year} with a release year, {with_tags} with tags."
    )

    return out
