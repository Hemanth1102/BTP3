import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class Ranker:
    """
    Base interface for candidate ranking.

    The ranker receives candidate feature rows and returns one score per row
    (higher = more relevant).
    """

    name = "base_ranker"

    def predict(self, features: List[Dict]) -> List[float]:
        raise NotImplementedError


class RRFRanker(Ranker):
    """
    Reciprocal-rank-fusion baseline: score = sum over retrievers of
    1 / (k + rank). This is the "simple score-based ranking baseline" the
    LightGBM model has to beat.

    Uses the precomputed `rrf_sum` feature when present. The fallback only
    looks at per-retriever rank columns that are *present* (rank is NaN when a
    retriever did not return the track) and ignores the aggregate columns
    (best_rank / worst_rank / mean_rank / median_rank).
    """

    name = "rrf"

    _AGGREGATE_RANKS = {"best_rank", "worst_rank", "mean_rank", "median_rank"}

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def predict(self, features: List[Dict]) -> List[float]:

        scores = []

        for row in features:

            if "rrf_sum" in row and row["rrf_sum"] is not None:
                scores.append(float(row["rrf_sum"]))
                continue

            score = 0.0

            for key, value in row.items():

                if not key.endswith("_rank") or key in self._AGGREGATE_RANKS:
                    continue

                if value is None or (
                    isinstance(value, float) and math.isnan(value)
                ):
                    continue

                score += 1.0 / (self.rrf_k + float(value))

            scores.append(score)

        return scores


class LightGBMRanker(Ranker):
    """
    Rank candidates with a trained LightGBM LambdaRank model.

    The model's own feature list defines the input schema. Every feature the
    model was trained on must be present in the rows (missing keys raise
    instead of being silently replaced by 0). Values of None / NaN are passed
    to LightGBM as missing, exactly as in training.
    """

    name = "lightgbm"

    def __init__(
        self,
        model_path: str,
        expected_features: Optional[List[str]] = None,
    ):
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=str(model_path))
        self.feature_names = self.model.feature_name()

        # Optional second line of defence: the schema saved next to the model.
        schema_path = Path(model_path).with_name("ranker_features.json")
        if schema_path.exists():
            saved = json.loads(schema_path.read_text())
            if saved != self.feature_names:
                raise ValueError(
                    f"{schema_path} disagrees with the feature names stored "
                    f"inside {model_path}."
                )

        if expected_features is not None:
            missing = [f for f in self.feature_names if f not in expected_features]
            if missing:
                raise ValueError(
                    "The model was trained on features the current "
                    f"FeatureBuilder does not produce: {missing[:10]}"
                    f"{' ...' if len(missing) > 10 else ''}. "
                    "Retrain the ranker with the current feature schema."
                )

    def predict(self, features: List[Dict]) -> List[float]:

        if not features:
            return []

        missing = [f for f in self.feature_names if f not in features[0]]
        if missing:
            raise ValueError(
                f"Feature rows are missing model features: {missing[:10]}"
            )

        X = np.array(
            [
                [
                    np.nan if row[f] is None else float(row[f])
                    for f in self.feature_names
                ]
                for row in features
            ],
            dtype=np.float64,
        )

        return self.model.predict(X).tolist()
