import math
import re
from typing import Any, Dict, List, Optional


# Every retriever the project knows about. The feature schema is built from this
# list (NOT from whichever retrievers happen to be loaded), so a dataset built
# with `--models bm25 cf` has exactly the same columns as one built with all
# retrievers. Retrievers that were not run simply have present=0 / rank=NaN.
KNOWN_RETRIEVERS = [
    "bm25",
    "bge",
    "cf_bpr",
    "clap_audio",
    "item_cf_bpr",
    "item_clap",
    "item_metadata",
    "artist_lookup",
    "album_lookup",
]

QUERY_RETRIEVERS = ["bm25", "bge", "clap_audio"]
PERSONALIZED_RETRIEVERS = ["cf_bpr"]
ITEM_TO_ITEM_RETRIEVERS = ["item_cf_bpr", "item_clap", "item_metadata"]
ENTITY_RETRIEVERS = ["artist_lookup", "album_lookup"]

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

    # =========================================================
    # Schema
    # =========================================================

    def feature_names(self) -> List[str]:

        names: List[str] = []

        for source in self.retriever_names:
            p = self._safe_name(source)
            names += [
                f"{p}_present",
                f"{p}_rank",
                f"{p}_score",
                f"{p}_score_norm",
                f"{p}_rrf",
            ]

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
            "has_year_constraint",
            "artist_match",
            "album_match",
            "genre_match",
            "rejected_artist_match",
            "year_in_range",
            "candidate_year",
            "artist_count",
            "album_count",
        ]

        return names

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

        indexed = self._index_results(retrieval_results)
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

        rows = []

        for track_id in candidate_ids:

            row: Dict[str, Any] = {"track_id": track_id}

            self._add_retriever_features(row, track_id, indexed, norm_scores)
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
            self._add_planner_features(row, query_state)
            self._add_metadata_features(
                row,
                metadata_lookup.get(track_id),
                query_state,
            )

            # Guarantee the schema, whatever the code paths above did.
            for name in self.feature_names():
                row.setdefault(name, NAN)

            rows.append(row)

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

    def _add_retriever_features(self, row, track_id, indexed, norm_scores):

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
            row["genre_match"] = NAN
            row["rejected_artist_match"] = NAN
            row["year_in_range"] = NAN
            return

        p_artists = _norm_set(getattr(query_state, "artist_names", None))
        p_albums = _norm_set(getattr(query_state, "album_names", None))
        p_genres = _norm_set(getattr(query_state, "genre_tags", None))
        p_rejected = _norm_set(getattr(query_state, "rejected_artists", None))

        row["artist_match"] = float(bool(artists & p_artists))
        row["album_match"] = float(bool(albums & p_albums))
        row["rejected_artist_match"] = float(bool(artists & p_rejected))

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
