from typing import Dict, List, Optional, Any

import math


class FeatureBuilder:
    """
    Build ranking features for candidates produced by multiple
    retrieval branches.

    One candidate becomes one feature dictionary.

    Current feature groups:

    1. Per-retriever presence/rank/score
    2. Cross-retriever agreement
    3. Best / average retrieval rank
    4. RRF-style aggregate score
    5. User-personalization signals
    6. Item-to-item signals
    7. Entity lookup signals
    8. Anchor/context signals

    Planner-derived features will be added later.
    """

    RRF_K = 60.0

    def __init__(
        self,
        retriever_names: Optional[List[str]] = None,
    ):
        self.retriever_names = retriever_names or []

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
    ) -> List[Dict[str, Any]]:
        """
        Build one feature row per candidate.

        Args:
            retrieval_results:
                {
                    "bm25": [RetrievalResult, ...],
                    "cf_bpr": [RetrievalResult, ...],
                    ...
                }

            candidate_ids:
                Optional explicit candidate ordering. If omitted,
                candidates are inferred from retrieval_results.

            user_id:
                Current user.

            anchor_track_ids:
                Tracks from previous turns.

            session_track_ids:
                Previous session tracks.

            query_state:
                Optional QueryState from the planner.

            candidate_metadata:
                Optional:
                {
                    track_id: {
                        "artist_names": [...],
                        "album_names": [...],
                        ...
                    }
                }
        """

        # -----------------------------------------------------
        # Candidate universe
        # -----------------------------------------------------

        if candidate_ids is None:
            candidate_ids = self._build_candidate_union(
                retrieval_results
            )

        # -----------------------------------------------------
        # Index retrieval results
        # -----------------------------------------------------

        indexed = self._index_results(
            retrieval_results
        )

        rows = []

        for track_id in candidate_ids:

            row = {
                "track_id": track_id,
            }

            # ---------------------------------------------
            # Per-retriever features
            # ---------------------------------------------

            self._add_retriever_features(
                row=row,
                track_id=track_id,
                indexed=indexed,
                retrieval_results=retrieval_results,
            )

            # ---------------------------------------------
            # Cross-retriever features
            # ---------------------------------------------

            self._add_cross_retriever_features(
                row=row,
                track_id=track_id,
                indexed=indexed,
            )

            # ---------------------------------------------
            # Context features
            # ---------------------------------------------

            self._add_context_features(
                row=row,
                track_id=track_id,
                anchor_track_ids=anchor_track_ids,
                session_track_ids=session_track_ids,
                user_id=user_id,
            )

            # ---------------------------------------------
            # Planner features, when available
            # ---------------------------------------------

            self._add_planner_features(
                row=row,
                track_id=track_id,
                query_state=query_state,
            )

            # ---------------------------------------------
            # Metadata/entity features
            # ---------------------------------------------

            self._add_metadata_features(
                row=row,
                track_id=track_id,
                candidate_metadata=candidate_metadata,
                query_state=query_state,
            )

            rows.append(row)

        return rows

    # =========================================================
    # Candidate union
    # =========================================================

    @staticmethod
    def _build_candidate_union(
        retrieval_results: Dict[str, List[Any]],
    ) -> List[str]:
        """
        Build a deterministic candidate union.

        First appearance order is preserved.
        """

        seen = set()
        candidates = []

        for results in retrieval_results.values():

            for result in results:

                track_id = result.track_id

                if track_id not in seen:
                    seen.add(track_id)
                    candidates.append(track_id)

        return candidates

    # =========================================================
    # Index retrieval results
    # =========================================================

    @staticmethod
    def _index_results(
        retrieval_results: Dict[str, List[Any]],
    ):
        """
        Convert:

            source -> [RetrievalResult, ...]

        into:

            source -> {
                track_id -> RetrievalResult
            }
        """

        indexed = {}

        for source, results in retrieval_results.items():

            indexed[source] = {
                result.track_id: result
                for result in results
            }

        return indexed

    # =========================================================
    # Per-retriever features
    # =========================================================

    def _add_retriever_features(
        self,
        row: Dict[str, Any],
        track_id: str,
        indexed: Dict[str, Dict[str, Any]],
        retrieval_results: Dict[str, List[Any]],
    ):

        sources = (
            self.retriever_names
            if self.retriever_names
            else list(retrieval_results.keys())
        )

        for source in sources:

            result = indexed.get(
                source,
                {},
            ).get(track_id)

            prefix = self._safe_name(
                source
            )

            if result is None:

                row[f"{prefix}_present"] = 0.0
                row[f"{prefix}_rank"] = 0.0
                row[f"{prefix}_score"] = 0.0
                row[f"{prefix}_rrf"] = 0.0

            else:

                rank = float(
                    result.rank
                )

                score = float(
                    result.score
                )

                row[f"{prefix}_present"] = 1.0
                row[f"{prefix}_rank"] = rank
                row[f"{prefix}_score"] = score
                row[f"{prefix}_rrf"] = (
                    1.0
                    / (
                        self.RRF_K
                        + rank
                    )
                )

    # =========================================================
    # Cross-retriever features
    # =========================================================

    def _add_cross_retriever_features(
        self,
        row: Dict[str, Any],
        track_id: str,
        indexed: Dict[str, Dict[str, Any]],
    ):

        ranks = []
        scores = []
        present_sources = []

        for source, source_results in indexed.items():

            result = source_results.get(
                track_id
            )

            if result is None:
                continue

            ranks.append(
                float(result.rank)
            )

            scores.append(
                float(result.score)
            )

            present_sources.append(
                source
            )

        if ranks:

            row["num_sources"] = float(
                len(ranks)
            )

            row["best_rank"] = float(
                min(ranks)
            )

            row["worst_rank"] = float(
                max(ranks)
            )

            row["mean_rank"] = (
                sum(ranks)
                / len(ranks)
            )

            row["median_rank"] = (
                self._median(ranks)
            )

            row["best_score"] = float(
                max(scores)
            )

            row["mean_score"] = (
                sum(scores)
                / len(scores)
            )

            row["rrf_sum"] = sum(
                1.0 / (
                    self.RRF_K + rank
                )
                for rank in ranks
            )

        else:

            row["num_sources"] = 0.0
            row["best_rank"] = 0.0
            row["worst_rank"] = 0.0
            row["mean_rank"] = 0.0
            row["median_rank"] = 0.0
            row["best_score"] = 0.0
            row["mean_score"] = 0.0
            row["rrf_sum"] = 0.0

        # ---------------------------------------------
        # Specific agreement features
        # ---------------------------------------------

        row["has_query_retrieval"] = float(
            any(
                source in indexed
                and track_id in indexed[source]
                for source in [
                    "bm25",
                    "bge",
                    "qwen_metadata",
                    "qwen_lyrics",
                    "qwen_attributes",
                    "clap_audio",
                ]
            )
        )

        row["has_personalized_retrieval"] = float(
            track_id in indexed.get(
                "cf_bpr",
                {}
            )
        )

        row["has_item_to_item"] = float(
            any(
                track_id in indexed.get(
                    source,
                    {}
                )
                for source in [
                    "item_cf_bpr",
                    "item_clap",
                    "item_metadata",
                ]
            )
        )

        row["has_entity_lookup"] = float(
            any(
                track_id in indexed.get(
                    source,
                    {}
                )
                for source in [
                    "artist_lookup",
                    "album_lookup",
                ]
            )
        )

    # =========================================================
    # Context features
    # =========================================================

    def _add_context_features(
        self,
        row: Dict[str, Any],
        track_id: str,
        anchor_track_ids: Optional[List[str]],
        session_track_ids: Optional[List[str]],
        user_id: Optional[str],
    ):

        anchors = set(
            anchor_track_ids or []
        )

        session_tracks = set(
            session_track_ids or []
        )

        row["anchor_count"] = float(
            len(anchors)
        )

        row["session_track_count"] = float(
            len(session_tracks)
        )

        # Should normally be zero because item-to-item retrievers
        # already exclude anchors, but keep this as a feature.
        row["is_anchor"] = float(
            track_id in anchors
        )

        row["seen_in_session"] = float(
            track_id in session_tracks
        )

        # User existence flag is useful as a simple cold/warm
        # feature. Actual user embedding score is already represented
        # by cf_bpr_score.
        row["has_user"] = float(
            user_id is not None
        )

    # =========================================================
    # Planner features
    # =========================================================

    def _add_planner_features(
        self,
        row: Dict[str, Any],
        track_id: str,
        query_state=None,
    ):

        # Planner is optional during current development.

        if query_state is None:

            row["intent_similarity"] = 0.0
            row["intent_discovery"] = 0.0
            row["intent_artist"] = 0.0
            row["intent_album"] = 0.0
            row["need_new_artist"] = 0.0
            row["need_personalization"] = 0.0

            return

        intent = (
            getattr(
                query_state,
                "intent",
                "",
            )
            or ""
        ).lower()

        row["intent_similarity"] = float(
            intent == "similarity"
        )

        row["intent_discovery"] = float(
            intent == "discovery"
        )

        row["intent_artist"] = float(
            intent == "artist_request"
        )

        row["intent_album"] = float(
            intent == "album_request"
        )

        row["need_new_artist"] = float(
            getattr(
                query_state,
                "need_new_artist",
                False,
            )
        )

        row["need_personalization"] = float(
            getattr(
                query_state,
                "need_personalization",
                False,
            )
        )

    # =========================================================
    # Metadata/entity features
    # =========================================================

    def _add_metadata_features(
        self,
        row: Dict[str, Any],
        track_id: str,
        candidate_metadata: Optional[
            Dict[str, Dict[str, Any]]
        ],
        query_state=None,
    ):

        metadata = (
            candidate_metadata or {}
        ).get(track_id, {})

        artists = metadata.get(
            "artist_names",
            []
        )

        albums = metadata.get(
            "album_names",
            []
        )

        genres = metadata.get(
            "genres",
            []
        )

        if isinstance(artists, str):
            artists = [artists]

        if isinstance(albums, str):
            albums = [albums]

        if isinstance(genres, str):
            genres = [genres]

        planner_artists = []
        planner_albums = []
        planner_genres = []

        if query_state is not None:

            planner_artists = getattr(
                query_state,
                "artist_names",
                [],
            ) or []

            planner_albums = getattr(
                query_state,
                "album_names",
                [],
            ) or []

            planner_genres = getattr(
                query_state,
                "genre_tags",
                [],
            ) or []

        candidate_artist_set = {
            str(x).lower().strip()
            for x in artists
        }

        candidate_album_set = {
            str(x).lower().strip()
            for x in albums
        }

        candidate_genre_set = {
            str(x).lower().strip()
            for x in genres
        }

        planner_artist_set = {
            str(x).lower().strip()
            for x in planner_artists
        }

        planner_album_set = {
            str(x).lower().strip()
            for x in planner_albums
        }

        planner_genre_set = {
            str(x).lower().strip()
            for x in planner_genres
        }

        row["artist_match"] = float(
            bool(
                candidate_artist_set
                & planner_artist_set
            )
        )

        row["album_match"] = float(
            bool(
                candidate_album_set
                & planner_album_set
            )
        )

        row["genre_match"] = float(
            bool(
                candidate_genre_set
                & planner_genre_set
            )
        )

        row["artist_count"] = float(
            len(candidate_artist_set)
        )

        row["album_count"] = float(
            len(candidate_album_set)
        )

    # =========================================================
    # Utility methods
    # =========================================================

    @staticmethod
    def _safe_name(name: str) -> str:
        """
        Convert source names to valid-ish feature names.
        """

        chars = []

        for char in str(name):

            if (
                char.isalnum()
                or char == "_"
            ):
                chars.append(
                    char.lower()
                )
            else:
                chars.append("_")

        return "".join(chars)

    @staticmethod
    def _median(values: List[float]) -> float:

        values = sorted(values)

        n = len(values)

        if n == 0:
            return 0.0

        middle = n // 2

        if n % 2:
            return float(values[middle])

        return (
            float(values[middle - 1])
            + float(values[middle])
        ) / 2.0