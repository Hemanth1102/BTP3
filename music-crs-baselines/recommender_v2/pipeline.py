from typing import Dict, List, Optional
from pathlib import Path

from .ranker import RRFRanker
from .ranker import LightGBMRanker

from .fusion import CandidateFusion
from .planner import QueryPlanner, QueryState
from .retrievers.base import BaseRetriever
from .features import FeatureBuilder


class RecommenderV2:
    """
    Multi-retriever conversational recommender.

    Current flow:

        conversation
            ↓
        QueryPlanner
            ↓
        QueryState
            ↓
        retrieval branches
            ↓
        candidate union

    Ranking will be added later.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        planner: Optional[QueryPlanner] = None,
        candidate_k: int = 100,
        final_k: int = 20,
    ):
        if not retrievers:
            raise ValueError(
                "At least one retriever is required."
            )

        self.retrievers = retrievers

        self.planner = planner or QueryPlanner(
            mode="mock"
        )

        self.candidate_k = candidate_k
        self.final_k = final_k

        self.fusion = CandidateFusion()

        self.feature_builder = FeatureBuilder()

        model_path = (
            Path(__file__).resolve().parent
            / "training"
            / "ranker.txt"
        )

        self.ranker = LightGBMRanker(
            str(model_path)
        )

    # ---------------------------------------------------------
    # Planner
    # ---------------------------------------------------------

    def plan(
        self,
        query: str,
        conversation_history=None,
        anchor_track_ids=None,
    ) -> QueryState:

        return self.planner.plan(
            query=query,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
        )

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    def retrieve_all(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_history=None,
        anchor_track_ids=None,
        session_track_ids=None,
    ) -> Dict[str, List]:

        # ---------------------------------------------
        # 1. Plan the request
        # ---------------------------------------------

        state = self.plan(
            query=query,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
        )

        # ---------------------------------------------
        # 2. Retrieve from every branch
        # ---------------------------------------------

        results = {}

        for retriever in self.retrievers:

            if retriever.name.startswith("bm25"):
                retrieval_query = state.bm25_query
            else:
                retrieval_query = state.semantic_query

            results[retriever.name] = retriever.retrieve(
                query=retrieval_query,
                topk=self.candidate_k,
                user_id=user_id,
                anchor_track_ids=state.anchor_track_ids,
                session_track_ids=session_track_ids,
            )

        return results

    # ---------------------------------------------------------
    # Candidate union
    # ---------------------------------------------------------

    def retrieve_candidates(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_history=None,
        anchor_track_ids=None,
        session_track_ids=None,
    ):

        retrieval_results = self.retrieve_all(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
        )

        candidates = self.fusion.fuse(
            list(retrieval_results.values())
        )

        return candidates

    # ---------------------------------------------------------
    # Feature table
    # ---------------------------------------------------------

    def retrieve_features(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_history=None,
        anchor_track_ids=None,
        session_track_ids=None,
    ):

        candidates = self.retrieve_candidates(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
        )

        return self.fusion.build_features(
            candidates
        )

    # ---------------------------------------------------------
    # Final recommendation
    # ---------------------------------------------------------

    def recommend(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_history=None,
        anchor_track_ids=None,
        session_track_ids=None,
    ) -> List[str]:

        retrieval_results = self.retrieve_all(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
        )

        candidates = self.fusion.fuse(
            list(retrieval_results.values())
        )

        features = self.build_features(
            retrieval_results=retrieval_results,
            user_id=user_id,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
            query_state=None,
        )

        scores = self.ranker.predict(features)

        scored = [
            (row["track_id"], score)
            for row, score in zip(features, scores)
        ]

        scored.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            track_id
            for track_id, _ in scored[:self.final_k]
        ]

    def build_features(
        self,
        retrieval_results,
        user_id=None,
        anchor_track_ids=None,
        session_track_ids=None,
        query_state=None,
    ):
        return self.feature_builder.build_candidate_features(
            retrieval_results=retrieval_results,
            user_id=user_id,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
            query_state=query_state,
        )

    def retrieve_for_ranking(
        self,
        query,
        user_id=None,
        conversation_history=None,
        anchor_track_ids=None,
        session_track_ids=None,
    ):
        state = self.plan(
            query=query,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
        )

        retrieval_results = {}

        for retriever in self.retrievers:
            if retriever.name.startswith("bm25"):
                retrieval_query = state.bm25_query
            else:
                retrieval_query = state.semantic_query

            retrieval_results[retriever.name] = retriever.retrieve(
                query=retrieval_query,
                topk=self.candidate_k,
                user_id=user_id,
                anchor_track_ids=state.anchor_track_ids,
                session_track_ids=session_track_ids,
            )

        return state, retrieval_results