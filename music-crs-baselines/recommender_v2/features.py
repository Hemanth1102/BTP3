import math
import re
from collections import defaultdict
from itertools import takewhile
from typing import Any, Dict, List, Optional

import numpy as np

from .text_norm import album_keys, base_title_norm, normalize_name, normalized_names


# Every retriever the project knows about. The feature schema is built from this
# list (NOT from whichever retrievers happen to be loaded), so a dataset built
# with `--models bm25 cf` has exactly the same columns as one built with all
# retrievers. Retrievers that were not run simply have present=0 / rank=NaN.
_LEGACY_RETRIEVERS = [
    "bm25",
    "bge",
    "cf_bpr",
    "clap_audio",
    "item_cf_bpr",
    "item_clap",
    "item_metadata",
    "artist_lookup",
    "album_lookup",
    "title_lookup",
    "anchor_artist",
    "transition_artist",
]

_NEW_ITEM_TO_ITEM_RETRIEVERS = [
    "item_cf_bpr_decay", "item_cf_bpr_max",
    "item_clap_decay", "item_clap_max",
    "item_metadata_decay", "item_metadata_max",
]
_SCHEMA_V2_RETRIEVERS = (
    _LEGACY_RETRIEVERS + ["session_bm25", "anchor_album"] + _NEW_ITEM_TO_ITEM_RETRIEVERS + ["cooc_track"]
)
KNOWN_RETRIEVERS = _SCHEMA_V2_RETRIEVERS + ["dense_ft"]
FEATURE_SCHEMA_VERSION = 3

QUERY_RETRIEVERS = ["bm25", "bge", "clap_audio", "session_bm25", "dense_ft"]
PERSONALIZED_RETRIEVERS = ["cf_bpr"]
ITEM_TO_ITEM_RETRIEVERS = (
    ["item_cf_bpr", "item_clap", "item_metadata"] + _NEW_ITEM_TO_ITEM_RETRIEVERS + ["cooc_track"]
)
ENTITY_RETRIEVERS = ["artist_lookup", "album_lookup", "title_lookup"]

INTENTS = [
    "discovery",
    "similarity",
    "artist_request",
    "album_request",
    "mood",
    "genre",
    "lyrical",
    "temporal",
    "conversational",
]

NAN = float("nan")

# Provided embedding spaces used for "similarity to the session's played tracks"
# features. key -> retriever that already holds the (normalised) track matrix.
EMBEDDING_SPACES = {
    "cf": "item_cf_bpr",
    "clap": "item_clap",
    "meta": "item_metadata",
}

EXTRA_FEATURES = (
    [f"emb_{k}_{kind}" for k in EMBEDDING_SPACES
     for kind in ("centroid_sim", "last_sim", "max_sim")]
    + ["user_cf_sim", "artist_track_count", "popularity_rank_in_pool"]
)

FULLPOOL_TEXT_SCORES = ("fp_bm25_msg_score", "fp_bm25_session_score", "fp_bge_cos")
FULLPOOL_TEXT_FEATURES = tuple(
    name + suffix for name in FULLPOOL_TEXT_SCORES for suffix in ("", "_rank_in_pool", "_pct_in_pool")
)
STREAK_FEATURES = (
    "artist_streak_last", "album_streak_last", "same_album_as_last_anchor", "artist_anchor_share",
    "turns_since_artist_played", "album_unplayed_count", "last_anchors_same_album", "last_anchors_same_artist",
)
POOL_AGGREGATE_FEATURES = ("rrf_sum_z", "num_sources_pct_in_pool")
POOL_RELATIVE_FEATURES = tuple(f"{name}_rank_pct" for name in _SCHEMA_V2_RETRIEVERS) + (
    *POOL_AGGREGATE_FEATURES, "dense_ft_rank_pct",
)
FEATURE_GROUPS = {
    "fullpool_text": FULLPOOL_TEXT_FEATURES,
    "streaks": STREAK_FEATURES,
    "pool_relative": POOL_RELATIVE_FEATURES,
}


def _descending_ranks(scores):
    """Competition ranks for finite scores with equal values sharing a rank."""
    scores = np.asarray(scores)
    valid = np.isfinite(scores)
    ranks = np.full(scores.shape, np.nan, dtype=float)
    _, positions, counts = np.unique(scores[valid], return_inverse=True, return_counts=True)
    ranks[valid] = (1 + np.sum(counts) - np.cumsum(counts))[positions]
    return ranks


def _take(matrix, idx):
    """Rows `idx` of a torch tensor / numpy array as float32 numpy (idx=None: whole array)."""
    x = matrix if idx is None else matrix[idx]
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _unit_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _norm_set(values: Any) -> set:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {n for n in (_norm(v) for v in values) if n}


class FeatureBuilder:
    """
    Build ranking features for the candidates produced by the retrievers.

    One candidate -> one flat dict of floats. The set of keys is fixed by
    `feature_names()` and is identical at training and inference time.

    Conventions
    -----------
    * "Not returned by retriever X"  ->  X_present = 0, X_rank = NaN,
      X_score = NaN, X_score_norm = NaN, X_rrf = 0.
      (Rank/score are NaN, never 0: 0 would look like the *best* rank.)
      LightGBM handles NaN natively.
    * Raw scores from different retrievers live on different scales, so the
      cross-retriever features only use ranks, RRF and min-max normalised
      scores (`*_score_norm`, computed within each retriever's candidate list).
    * Metadata-dependent features are NaN when no metadata is available for the
      candidate (unknown), and 0/1 otherwise.
    """

    RRF_K = 60.0

    def __init__(
        self,
        retriever_names: Optional[List[str]] = None,
        track_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.retriever_names = list(retriever_names or KNOWN_RETRIEVERS)
        self.track_metadata = track_metadata or {}
        self._spaces = {}          # key -> (matrix, {track_id: row})
        self._user_space = None    # (user_embeddings, track matrix, {track_id: row})
        self._artist_counts = None
        self._session_bm25 = self._bge = None
        self._streak_catalogue = self._index_streak_catalogue(self.track_metadata)

    # =========================================================
    # Schema
    # =========================================================

    def feature_names(self) -> List[str]:

        names: List[str] = []

        names += self._retriever_feature_names(
            source for source in self.retriever_names if source in _LEGACY_RETRIEVERS
        )

        names += [
            "num_sources",
            "best_rank",
            "worst_rank",
            "mean_rank",
            "median_rank",
            "max_score_norm",
            "mean_score_norm",
            "rrf_sum",
            "rrf_rank_in_pool",
            "pool_size",
            "has_query_retrieval",
            "has_personalized_retrieval",
            "has_item_to_item",
            "has_entity_lookup",
            "anchor_count",
            "session_track_count",
            "is_anchor",
            "seen_in_session",
            "has_user",
            "turn_index",
        ]

        names += [f"intent_{i}" for i in INTENTS]

        names += [
            "need_new_artist",
            "need_new_track",
            "need_personalization",
            "planner_confidence",
            "n_planner_artists",
            "n_planner_albums",
            "n_planner_genres",
            "n_planner_titles",
            "has_year_constraint",
            "artist_match",
            "album_match",
            "title_match",
            "genre_match",
            "rejected_artist_match",
            "year_in_range",
            "candidate_year",
            "artist_count",
            "album_count",
            "candidate_popularity",
            "same_artist_as_anchor",
            "same_artist_as_last_anchor",
            "same_album_as_anchor",
            "anchor_tag_overlap",
            "year_gap_to_anchors",
        ]

        names += EXTRA_FEATURES
        names += self._retriever_feature_names(
            source for source in self.retriever_names if source not in _LEGACY_RETRIEVERS and source != "dense_ft"
        )
        names += FULLPOOL_TEXT_FEATURES + STREAK_FEATURES
        names += [f"{self._safe_name(source)}_rank_pct" for source in self.retriever_names if source != "dense_ft"]
        names += POOL_AGGREGATE_FEATURES
        if "dense_ft" in self.retriever_names:
            names += self._retriever_feature_names(("dense_ft",)) + ["dense_ft_rank_pct"]

        return names

    def _retriever_feature_names(self, sources):
        return [f"{self._safe_name(source)}_{suffix}" for source in sources
                for suffix in ("present", "rank", "score", "score_norm", "rrf")]

    # =========================================================
    # Public API
    # =========================================================

    def build_candidate_features(
        self,
        retrieval_results: Dict[str, List[Any]],
        candidate_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        anchor_track_ids: Optional[List[str]] = None,
        session_track_ids: Optional[List[str]] = None,
        query_state=None,
        candidate_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        turn_index: Optional[int] = None,
        planner_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Build one feature row per candidate (plus a leading "track_id").

        retrieval_results : {retriever_name: [RetrievalResult, ...]}
        turn_index        : 1-based turn number of the current request.
        candidate_metadata: overrides the builder's `track_metadata`.
        """

        metadata_lookup = (
            candidate_metadata
            if candidate_metadata is not None
            else self.track_metadata
        )

        unknown = set(retrieval_results) - set(self.retriever_names)
        if unknown:
            raise ValueError(
                f"Retrievers {sorted(unknown)} are not in the feature schema. "
                f"Add them to KNOWN_RETRIEVERS (features.py)."
            )

        if candidate_ids is None:
            candidate_ids = self._build_candidate_union(retrieval_results)
        if not len(candidate_ids):
            return []

        indexed = self._index_results(retrieval_results)
        list_lengths = {source: len(results) for source, results in retrieval_results.items()}
        norm_scores = self._normalised_scores(indexed)

        # Pool-level quantities (same value for every candidate of the query).
        rrf_sums = {
            tid: sum(
                1.0 / (self.RRF_K + float(src[tid].rank))
                for src in indexed.values()
                if tid in src
            )
            for tid in candidate_ids
        }
        # sorted() is stable, so ties keep first-appearance order.
        order = sorted(candidate_ids, key=lambda t: -rrf_sums[t])
        rrf_rank = {tid: i + 1 for i, tid in enumerate(order)}
        pool_size = float(len(candidate_ids))

        anchor_summary = self._summarize_anchors(metadata_lookup, anchor_track_ids)
        planner_features = {}
        self._add_planner_features(planner_features, query_state)
        feature_names = self.feature_names()

        rows = []

        for track_id in candidate_ids:

            row: Dict[str, Any] = {"track_id": track_id}

            self._add_retriever_features(row, track_id, indexed, norm_scores, list_lengths)
            self._add_cross_retriever_features(row, track_id, indexed, norm_scores)

            row["rrf_rank_in_pool"] = float(rrf_rank[track_id])
            row["pool_size"] = pool_size

            self._add_context_features(
                row,
                track_id,
                anchor_track_ids,
                session_track_ids,
                user_id,
                turn_index,
            )
            row.update(planner_features)
            self._add_metadata_features(
                row,
                metadata_lookup.get(track_id),
                query_state,
            )
            self._add_anchor_features(
                row,
                metadata_lookup.get(track_id),
                anchor_summary,
            )

            # Guarantee the schema, whatever the code paths above did.
            for name in feature_names:
                row.setdefault(name, NAN)

            rows.append(row)

        self.add_extra_features(
            rows,
            user_id=user_id,
            anchor_track_ids=anchor_track_ids,
            metadata_lookup=metadata_lookup,
        )
        self._add_fullpool_text(rows, candidate_ids, query_state, anchor_track_ids, planner_fallback)
        self._add_streak_features(rows, anchor_track_ids, session_track_ids, metadata_lookup)
        self._add_pool_relative_features(rows)

        return rows

    # =========================================================
    # Helpers: candidate union / indexing / normalisation
    # =========================================================

    @staticmethod
    def _build_candidate_union(retrieval_results) -> List[str]:
        seen = set()
        out = []
        for results in retrieval_results.values():
            for result in results:
                if result.track_id not in seen:
                    seen.add(result.track_id)
                    out.append(result.track_id)
        return out

    @staticmethod
    def _index_results(retrieval_results):
        # If a retriever returns the same track twice, keep its best (first)
        # occurrence so ranks stay well defined.
        indexed = {}
        for source, results in retrieval_results.items():
            d = {}
            for result in results:
                d.setdefault(result.track_id, result)
            indexed[source] = d
        return indexed

    @staticmethod
    def _normalised_scores(indexed):
        """Per-retriever min-max normalised score in [0, 1]."""
        out = {}
        for source, d in indexed.items():
            scores = [float(r.score) for r in d.values()]
            if not scores:
                out[source] = {}
                continue
            lo, hi = min(scores), max(scores)
            span = hi - lo
            out[source] = {
                tid: (1.0 if span <= 0 else (float(r.score) - lo) / span)
                for tid, r in d.items()
            }
        return out

    # =========================================================
    # Per-retriever features
    # =========================================================

    def _add_retriever_features(self, row, track_id, indexed, norm_scores, list_lengths):

        for source in self.retriever_names:

            p = self._safe_name(source)
            result = indexed.get(source, {}).get(track_id)

            if result is None:
                row[f"{p}_present"] = 0.0
                row[f"{p}_rank"] = NAN
                row[f"{p}_score"] = NAN
                row[f"{p}_score_norm"] = NAN
                row[f"{p}_rrf"] = 0.0
                continue

            rank = float(result.rank)
            row[f"{p}_present"] = 1.0
            row[f"{p}_rank"] = rank
            row[f"{p}_score"] = float(result.score)
            row[f"{p}_score_norm"] = norm_scores[source][track_id]
            row[f"{p}_rrf"] = 1.0 / (self.RRF_K + rank)
            row[f"{p}_rank_pct"] = rank / list_lengths[source]

    # =========================================================
    # Cross-retriever features
    # =========================================================

    def _add_cross_retriever_features(self, row, track_id, indexed, norm_scores):

        ranks = []
        norms = []

        for source, d in indexed.items():
            result = d.get(track_id)
            if result is None:
                continue
            ranks.append(float(result.rank))
            norms.append(norm_scores[source][track_id])

        if ranks:
            row["num_sources"] = float(len(ranks))
            row["best_rank"] = min(ranks)
            row["worst_rank"] = max(ranks)
            row["mean_rank"] = sum(ranks) / len(ranks)
            row["median_rank"] = self._median(ranks)
            row["max_score_norm"] = max(norms)
            row["mean_score_norm"] = sum(norms) / len(norms)
            row["rrf_sum"] = sum(1.0 / (self.RRF_K + r) for r in ranks)
        else:
            row["num_sources"] = 0.0
            row["best_rank"] = NAN
            row["worst_rank"] = NAN
            row["mean_rank"] = NAN
            row["median_rank"] = NAN
            row["max_score_norm"] = NAN
            row["mean_score_norm"] = NAN
            row["rrf_sum"] = 0.0

        def any_in(names):
            return float(any(track_id in indexed.get(s, {}) for s in names))

        row["has_query_retrieval"] = any_in(QUERY_RETRIEVERS)
        row["has_personalized_retrieval"] = any_in(PERSONALIZED_RETRIEVERS)
        row["has_item_to_item"] = any_in(ITEM_TO_ITEM_RETRIEVERS)
        row["has_entity_lookup"] = any_in(ENTITY_RETRIEVERS)

    # =========================================================
    # Context features
    # =========================================================

    def _add_context_features(
        self,
        row,
        track_id,
        anchor_track_ids,
        session_track_ids,
        user_id,
        turn_index,
    ):
        anchors = set(anchor_track_ids or [])
        session_tracks = set(session_track_ids or [])

        row["anchor_count"] = float(len(anchors))
        row["session_track_count"] = float(len(session_tracks))
        row["is_anchor"] = float(track_id in anchors)
        row["seen_in_session"] = float(track_id in session_tracks)
        row["has_user"] = float(user_id is not None)
        row["turn_index"] = float(turn_index) if turn_index is not None else NAN

    # =========================================================
    # Planner features
    # =========================================================

    def _add_planner_features(self, row, query_state):

        if query_state is None:
            # No planner output at all (e.g. planner disabled). Everything is
            # "unknown", which is different from "planner said no".
            for i in INTENTS:
                row[f"intent_{i}"] = NAN
            for name in (
                "need_new_artist",
                "need_new_track",
                "need_personalization",
                "planner_confidence",
                "n_planner_artists",
                "n_planner_albums",
                "n_planner_genres",
                "n_planner_titles",
                "has_year_constraint",
            ):
                row[name] = NAN
            return

        intent = (getattr(query_state, "intent", "") or "").lower().strip()

        for i in INTENTS:
            row[f"intent_{i}"] = float(intent == i)

        row["need_new_artist"] = float(bool(getattr(query_state, "need_new_artist", False)))
        row["need_new_track"] = float(bool(getattr(query_state, "need_new_track", False)))
        row["need_personalization"] = float(
            bool(getattr(query_state, "need_personalization", False))
        )

        try:
            row["planner_confidence"] = float(
                getattr(query_state, "planner_confidence", None)
            )
        except (TypeError, ValueError):
            row["planner_confidence"] = NAN

        row["n_planner_artists"] = float(len(getattr(query_state, "artist_names", []) or []))
        row["n_planner_albums"] = float(len(getattr(query_state, "album_names", []) or []))
        row["n_planner_genres"] = float(len(getattr(query_state, "genre_tags", []) or []))
        row["n_planner_titles"] = float(len(getattr(query_state, "track_titles", []) or []))

        row["has_year_constraint"] = float(
            _to_int(getattr(query_state, "min_year", None)) is not None
            or _to_int(getattr(query_state, "max_year", None)) is not None
        )

    # =========================================================
    # Metadata / entity features
    # =========================================================

    def _add_metadata_features(self, row, metadata, query_state):

        if not metadata:
            for name in (
                "artist_match",
                "album_match",
                "title_match",
                "genre_match",
                "rejected_artist_match",
                "year_in_range",
                "candidate_year",
                "artist_count",
                "album_count",
            ):
                row[name] = NAN
            return

        artists = _norm_set(metadata.get("artist_names"))
        albums = _norm_set(metadata.get("album_names"))
        genres = _norm_set(metadata.get("genres"))
        year = metadata.get("year")

        row["artist_count"] = float(len(artists))
        row["album_count"] = float(len(albums))
        row["candidate_year"] = float(year) if year else NAN

        if query_state is None:
            row["artist_match"] = NAN
            row["album_match"] = NAN
            row["title_match"] = NAN
            row["genre_match"] = NAN
            row["rejected_artist_match"] = NAN
            row["year_in_range"] = NAN
            return

        # Same normalisation as the lookup retrievers (case, accents,
        # punctuation, leading "the"), so "Beyoncé" == "beyonce".
        def names(values):
            return {normalize_name(v) for v in (values or [])} - {""}

        c_artists = names(metadata.get("artist_names"))
        c_albums = names(metadata.get("album_names"))
        p_artists = names(getattr(query_state, "artist_names", None))
        p_albums = names(getattr(query_state, "album_names", None))
        p_titles = names(getattr(query_state, "track_titles", None))
        p_genres = _norm_set(getattr(query_state, "genre_tags", None))
        p_rejected = names(getattr(query_state, "rejected_artists", None))

        row["artist_match"] = float(bool(c_artists & p_artists))
        row["album_match"] = float(bool(c_albums & p_albums))
        row["rejected_artist_match"] = float(bool(c_artists & p_rejected))

        title = normalize_name(metadata.get("track_name"))
        if not title:
            row["title_match"] = NAN
        else:
            row["title_match"] = float(bool(p_titles) and (
                title in p_titles or base_title_norm(metadata.get("track_name")) in p_titles))

        # Genre/tag vocabularies rarely match exactly ("indie rock" vs "rock"),
        # so allow containment either way for strings of reasonable length.
        row["genre_match"] = float(
            any(
                (g == t) or (len(g) >= 3 and len(t) >= 3 and (g in t or t in g))
                for g in p_genres
                for t in genres
            )
        )

        # The LLM planner may return years as strings ("1990") or junk.
        lo = _to_int(getattr(query_state, "min_year", None))
        hi = _to_int(getattr(query_state, "max_year", None))

        if (lo is None and hi is None) or not year:
            row["year_in_range"] = NAN
        else:
            ok = (lo is None or year >= lo) and (hi is None or year <= hi)
            row["year_in_range"] = float(ok)

    # =========================================================
    # Session-anchor features (artist / album / tag / year continuity)
    # =========================================================

    @staticmethod
    def _summarize_anchors(metadata_lookup, anchor_track_ids):
        anchors = [metadata_lookup.get(tid) or {} for tid in (anchor_track_ids or ())]
        if not anchors:
            return None
        years = [m["year"] for m in anchors if m.get("year")]
        return (
            [_norm_set(m.get("artist_names")) for m in anchors],
            [_norm_set(m.get("album_names")) for m in anchors],
            [_norm_set(m.get("genres")) for m in anchors],
            sum(years) / len(years) if years else None,
        )

    def _add_anchor_features(self, row, metadata, anchor_summary):

        names = (
            "candidate_popularity",
            "same_artist_as_anchor",
            "same_artist_as_last_anchor",
            "same_album_as_anchor",
            "anchor_tag_overlap",
            "year_gap_to_anchors",
        )

        if not metadata:
            for n in names:
                row[n] = NAN
            return

        pop = metadata.get("popularity")
        row["candidate_popularity"] = float(pop) if pop is not None else NAN

        if anchor_summary is None:
            # Turn 1: no anchors, so continuity features do not apply.
            row["same_artist_as_anchor"] = NAN
            row["same_artist_as_last_anchor"] = NAN
            row["same_album_as_anchor"] = NAN
            row["anchor_tag_overlap"] = NAN
            row["year_gap_to_anchors"] = NAN
            return

        artists = _norm_set(metadata.get("artist_names"))
        albums = _norm_set(metadata.get("album_names"))
        genres = _norm_set(metadata.get("genres"))
        year = metadata.get("year")

        a_artists, a_albums, a_genres, average_year = anchor_summary

        row["same_artist_as_anchor"] = float(sum(bool(artists & x) for x in a_artists))
        row["same_artist_as_last_anchor"] = float(bool(artists & a_artists[-1]))
        row["same_album_as_anchor"] = float(sum(bool(albums & x) for x in a_albums))

        best = 0.0
        for g in a_genres:
            if genres and g:
                best = max(best, len(genres & g) / len(genres | g))
        row["anchor_tag_overlap"] = best

        row["year_gap_to_anchors"] = abs(float(year) - average_year) if year and average_year is not None else NAN

    # =========================================================
    # Extra features: similarity to the played tracks in provided embedding
    # spaces (for EVERY candidate, not only the retriever's top-k), user-track
    # affinity, artist size and pool-relative popularity.
    # =========================================================

    def attach_embeddings(self, retrievers) -> None:
        """Reuse the indexes and embedding matrices loaded for retrieval."""
        self._spaces = {}
        self._user_space = None
        self._session_bm25 = self._bge = None
        for r in retrievers:
            name = getattr(r, "name", None)
            if name == "session_bm25" and hasattr(r, "score_candidates"):
                self._session_bm25 = r
            if name == "bge" and hasattr(r, "score_candidates"):
                self._bge = r
            for key, rname in EMBEDDING_SPACES.items():
                if name == rname and hasattr(r, "track_embeddings") \
                        and hasattr(r, "track_to_index"):
                    self._spaces[key] = (r.track_embeddings, r.track_to_index)
            if name == "cf_bpr" and hasattr(r, "user_embeddings"):
                self._user_space = (
                    r.user_embeddings,
                    r.track_embeddings,
                    {t: i for i, t in enumerate(r.track_ids)},
                )

    def _get_artist_counts(self, metadata_lookup):
        if self._artist_counts is None or metadata_lookup is not self.track_metadata:
            counts: Dict[str, int] = {}
            for m in metadata_lookup.values():
                for a in _norm_set(m.get("artist_names")):
                    counts[a] = counts.get(a, 0) + 1
            if metadata_lookup is self.track_metadata:
                self._artist_counts = counts
            return counts
        return self._artist_counts

    def add_extra_features(self, rows, user_id=None, anchor_track_ids=None,
                           metadata_lookup=None) -> None:
        """
        Fill EXTRA_FEATURES in place. `rows` need "track_id" (and, for
        popularity_rank_in_pool, "candidate_popularity"). Unknown -> NaN.
        """
        for r in rows:
            for n in EXTRA_FEATURES:
                r[n] = NAN
        if not rows:
            return

        ids = [r["track_id"] for r in rows]
        anchors = list(anchor_track_ids or [])

        # ---- embedding similarity to anchors ----
        if anchors:
            for key, (emb, index) in self._spaces.items():
                a_idx = [index[a] for a in anchors if a in index]
                if not a_idx:
                    continue
                c_pos = [i for i, t in enumerate(ids) if t in index]
                if not c_pos:
                    continue
                A = _unit_rows(_take(emb, a_idx))                 # anchors x d
                C = _unit_rows(_take(emb, [index[ids[i]] for i in c_pos]))
                S = C @ A.T                                       # candidates x anchors
                centroid = A.mean(axis=0)
                norm = float(np.linalg.norm(centroid))
                cen = (C @ (centroid / norm)) if norm > 0 else np.full(len(c_pos), np.nan)
                last = S[:, -1]                                   # most recent anchor with an embedding
                mx = S.max(axis=1)
                for j, i in enumerate(c_pos):
                    rows[i][f"emb_{key}_centroid_sim"] = float(cen[j])
                    rows[i][f"emb_{key}_last_sim"] = float(last[j])
                    rows[i][f"emb_{key}_max_sim"] = float(mx[j])

        # ---- user <-> track affinity (cosine in CF-BPR space) ----
        if user_id is not None and self._user_space is not None:
            users, tmat, tindex = self._user_space
            u = users.get(user_id)
            if u is not None:
                c_pos = [i for i, t in enumerate(ids) if t in tindex]
                if c_pos:
                    U = _unit_rows(_take(u, None)[None, :])[0]
                    C = _unit_rows(_take(tmat, [tindex[ids[i]] for i in c_pos]))
                    sims = C @ U
                    for j, i in enumerate(c_pos):
                        rows[i]["user_cf_sim"] = float(sims[j])

        # ---- artist catalogue size ----
        lookup = metadata_lookup if metadata_lookup is not None else self.track_metadata
        if lookup:
            counts = self._get_artist_counts(lookup)
            for r in rows:
                m = lookup.get(r["track_id"])
                if m:
                    cs = [counts.get(a, 0) for a in _norm_set(m.get("artist_names"))]
                    if cs:
                        r["artist_track_count"] = float(max(cs))

        # ---- popularity percentile inside this pool ----
        pops = [r.get("candidate_popularity", NAN) for r in rows]
        valid = sorted(p for p in pops if p == p)
        if valid:
            import bisect
            n = float(len(valid))
            for r, p in zip(rows, pops):
                if p == p:
                    r["popularity_rank_in_pool"] = bisect.bisect_left(valid, p) / n

    def _add_fullpool_text(self, rows, candidate_ids, query_state, anchors, planner_fallback):
        if query_state is None:
            return
        message = getattr(query_state, "raw_query", "") or ""
        scores = {}
        if self._session_bm25 is not None:
            bm25 = self._session_bm25
            # Read the full session score first to reuse the catalogue scores from retrieval.
            session = bm25.session_tokens(message, anchors)
            scores["fp_bm25_session_score"] = bm25.score_candidates(session, candidate_ids)
            tokens = bm25.message_tokens(message)
            scores["fp_bm25_msg_score"] = (scores["fp_bm25_session_score"] if tokens == session else
                                             bm25.score_candidates(tokens, candidate_ids))
        semantic = getattr(query_state, "semantic_query", "") or ""
        query = semantic if semantic.strip() and not planner_fallback else message
        if self._bge is not None and query.strip():
            scores["fp_bge_cos"] = self._bge.score_candidates(query, candidate_ids)
        for name, values in scores.items():
            ranks = _descending_ranks(values)
            for row, value, rank in zip(rows, values, ranks):
                row[name] = float(value)
                row[name + "_rank_in_pool"] = float(rank)
                row[name + "_pct_in_pool"] = float(rank / len(rows))

    @staticmethod
    def _index_streak_catalogue(metadata_lookup):
        by_track, by_album = {}, defaultdict(set)
        for track_id, metadata in metadata_lookup.items():
            if not metadata:
                continue
            artists = frozenset(normalized_names(metadata.get("artist_names")))
            albums = frozenset(album_keys(metadata.get("album_names"), metadata.get("artist_names")))
            by_track[track_id] = (artists, albums)
            for album in albums:
                by_album[album].add(track_id)
        return by_track, by_album

    def _add_streak_features(self, rows, anchor_track_ids, session_track_ids, metadata_lookup):
        by_track, by_album = (self._streak_catalogue if metadata_lookup is self.track_metadata else
                              self._index_streak_catalogue(metadata_lookup))
        if not by_track:
            return
        anchors = anchor_track_ids or ()
        played = set(anchors) | set(session_track_ids or ())
        unknown = (frozenset(), frozenset())
        recent = [by_track.get(track_id, unknown) for track_id in reversed(anchors)]
        last_artists, last_albums = recent[0] if recent else unknown
        prev_artists, prev_albums = recent[1] if len(recent) >= 2 else unknown
        same_artist = float(bool(last_artists & prev_artists)) if last_artists and prev_artists else NAN
        same_album = float(bool(last_albums & prev_albums)) if last_albums and prev_albums else NAN
        known_anchor_artist = any(artists for artists, _ in recent)
        unplayed = {}
        for row in rows:
            row["last_anchors_same_artist"] = same_artist
            row["last_anchors_same_album"] = same_album
            artists, albums = by_track.get(row["track_id"], unknown)
            if artists:
                matches = [bool(artists & anchor_artists) for anchor_artists, _ in recent]
                if not anchors or last_artists:
                    row["artist_streak_last"] = float(sum(takewhile(bool, matches)))
                if known_anchor_artist:
                    row["artist_anchor_share"] = sum(matches) / len(anchors)
                row["turns_since_artist_played"] = next((float(age) for age, match in enumerate(matches, 1)
                                                         if match), NAN)
            if albums:
                matches = [bool(albums & anchor_albums) for _, anchor_albums in recent]
                if not anchors or last_albums:
                    row["album_streak_last"] = float(sum(takewhile(bool, matches)))
                if last_albums:
                    row["same_album_as_last_anchor"] = float(matches[0])
                if albums not in unplayed:
                    album_tracks = set().union(*(by_album[key] for key in albums))
                    unplayed[albums] = float(len(album_tracks - played))
                row["album_unplayed_count"] = unplayed[albums]

    @staticmethod
    def _add_pool_relative_features(rows):
        rrf = np.array([row["rrf_sum"] for row in rows])
        z = (rrf - rrf.mean()) / rrf.std() if np.ptp(rrf) > 0 else np.zeros(len(rows))
        source_pct = _descending_ranks([row["num_sources"] for row in rows]) / len(rows)
        for row, value, pct in zip(rows, z, source_pct):
            row["rrf_sum_z"] = float(value)
            row["num_sources_pct_in_pool"] = float(pct)

    # =========================================================
    # Utilities
    # =========================================================

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(
            c.lower() if (c.isalnum() or c == "_") else "_" for c in str(name)
        )

    @staticmethod
    def _median(values: List[float]) -> float:
        values = sorted(values)
        n = len(values)
        if n == 0:
            return NAN
        mid = n // 2
        if n % 2:
            return float(values[mid])
        return (float(values[mid - 1]) + float(values[mid])) / 2.0
