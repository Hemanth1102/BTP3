import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from .ranker import RRFRanker
from .ranker import LightGBMRanker

from .fusion import CandidateFusion
from .planner import QueryPlanner, QueryState
from .retrievers.base import BaseRetriever
from .features import FeatureBuilder


DEFAULT_RANKER_PATH = (
    Path(__file__).resolve().parent / "training" / "ranker.txt"
)


class RecommenderV2:
    """
    Multi-retriever conversational recommender.

        conversation -> QueryPlanner -> QueryState
                     -> retrievers (candidate_k each) -> candidate union
                     -> FeatureBuilder -> Ranker -> top final_k

    Training (training/build_dataset.py) and inference (`recommend`) go through
    the exact same `plan -> retrieve -> build_features` code path, so the
    features the LightGBM model sees at inference are the ones it was trained on.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        planner: Optional[QueryPlanner] = None,
        candidate_k: int = 100,
        final_k: int = 20,
        ranker=None,
        ranker_path: Optional[str] = None,
        feature_builder: Optional[FeatureBuilder] = None,
        track_metadata=None,
    ):
        if not retrievers:
            raise ValueError("At least one retriever is required.")

        self.retrievers = retrievers

        if planner is None:
            from .planner.ollama import OllamaQwenPlanner

            planner = QueryPlanner(
                llm=OllamaQwenPlanner("qwen3:8b"),
                mode="llm",
            )

        self.planner = planner

        self.candidate_k = candidate_k
        self.final_k = final_k

        self.fusion = CandidateFusion()

        self.feature_builder = feature_builder or FeatureBuilder(
            track_metadata=track_metadata
        )

        # Ranker: explicit object > model file > RRF fallback.
        # (Building the dataset must work before any model has been trained.)
        if ranker is not None:
            self.ranker = ranker
        else:
            model_path = Path(ranker_path or DEFAULT_RANKER_PATH)

            if model_path.exists():
                self.ranker = LightGBMRanker(
                    str(model_path),
                    expected_features=self.feature_builder.feature_names(),
                )
            else:
                warnings.warn(
                    f"No ranker model at {model_path}; falling back to RRF.",
                    stacklevel=2,
                )
                self.ranker = RRFRanker()

    # ---------------------------------------------------------
    # Planner
    # ---------------------------------------------------------

    def plan(
        self,
        query: str,
        conversation_history=None,
        anchor_track_ids=None,
    ) -> QueryState:

        state = self.planner.plan(
            query=query,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
        )

        # The pipeline's anchors are authoritative: never let the LLM invent
        # or drop them.
        state.anchor_track_ids = list(anchor_track_ids or [])

        return state

    # ---------------------------------------------------------
    # Retrieval (single code path)
    # ---------------------------------------------------------

    def retrieve_for_ranking(
        self,
        query,
        user_id=None,
        conversation_history=None,
        anchor_track_ids=None,
        session_track_ids=None,
        state: Optional[QueryState] = None,
        timings: Optional[dict] = None,
    ):
        """
        Plan (unless a precomputed `state` is supplied) and run every
        retriever. Returns (state, {retriever_name: [RetrievalResult, ...]}).

        If a `timings` dict is passed it is filled with seconds per retriever
        (key = retriever name) for profiling.

        `conversation_history` must already be truncated to what is visible at
        the current turn (see training/turns.py).
        """

        if state is None:
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

            t0 = time.perf_counter()

            retrieval_results[retriever.name] = retriever.retrieve(
                query=retrieval_query,
                topk=self.candidate_k,
                user_id=user_id,
                anchor_track_ids=state.anchor_track_ids,
                session_track_ids=self._session_ids(
                    session_track_ids, anchor_track_ids
                ),
            )

            if timings is not None:
                timings[retriever.name] = time.perf_counter() - t0

        return state, retrieval_results

    def retrieve_all(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_history=None,
        anchor_track_ids=None,
        session_track_ids=None,
    ) -> Dict[str, List]:

        _, results = self.retrieve_for_ranking(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
        )

        return results

    @staticmethod
    def _session_ids(session_track_ids, anchor_track_ids):
        # Tracks already played in this session. When the caller does not say
        # otherwise, that is exactly the anchor list.
        if session_track_ids is None:
            return list(anchor_track_ids or [])
        return session_track_ids

    # ---------------------------------------------------------
    # Candidate union (legacy helpers)
    # ---------------------------------------------------------

    def retrieve_candidates(self, query, user_id=None, conversation_history=None,
                            anchor_track_ids=None, session_track_ids=None):

        retrieval_results = self.retrieve_all(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
        )

        return self.fusion.fuse(list(retrieval_results.values()))

    def retrieve_features(self, query, user_id=None, conversation_history=None,
                          anchor_track_ids=None, session_track_ids=None):

        candidates = self.retrieve_candidates(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
        )

        return self.fusion.build_features(candidates)

    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------

    def build_features(
        self,
        retrieval_results,
        user_id=None,
        anchor_track_ids=None,
        session_track_ids=None,
        query_state=None,
        turn_index=None,
    ):
        return self.feature_builder.build_candidate_features(
            retrieval_results=retrieval_results,
            user_id=user_id,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=self._session_ids(
                session_track_ids, anchor_track_ids
            ),
            query_state=query_state,
            turn_index=turn_index,
        )

    # ---------------------------------------------------------
    # Final recommendation
    # ---------------------------------------------------------

    def recommend_from_retrieval(
        self,
        state: QueryState,
        retrieval_results,
        user_id=None,
        anchor_track_ids=None,
        session_track_ids=None,
        turn_index=None,
    ) -> List[str]:
        """
        Rank an already-retrieved candidate pool. Lets an evaluator measure
        retrieval recall and final nDCG on the *same* plan.
        """

        features = self.build_features(
            retrieval_results=retrieval_results,
            user_id=user_id,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
            query_state=state,
            turn_index=turn_index,
        )

        if not features:
            return []

        scores = self.ranker.predict(features)

        order = sorted(
            range(len(features)),
            key=lambda i: -scores[i],
        )

        return [features[i]["track_id"] for i in order[: self.final_k]]

    def recommend(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_history=None,
        anchor_track_ids=None,
        session_track_ids=None,
        turn_index=None,
    ) -> List[str]:

        state, retrieval_results = self.retrieve_for_ranking(
            query=query,
            user_id=user_id,
            conversation_history=conversation_history,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
        )

        return self.recommend_from_retrieval(
            state=state,
            retrieval_results=retrieval_results,
            user_id=user_id,
            anchor_track_ids=anchor_track_ids,
            session_track_ids=session_track_ids,
            turn_index=turn_index,
        )