from typing import Dict, List, Optional


class Ranker:
    """
    Base interface for candidate ranking.

    The ranker receives candidate feature rows and returns
    candidates ordered by predicted relevance.
    """

    name = "base_ranker"

    def __init__(self):
        pass

    def predict(
        self,
        features: List[Dict],
    ) -> List[float]:
        raise NotImplementedError


class RRFRanker(Ranker):
    """
    Temporary ranker using Reciprocal Rank Fusion.

    This is useful while we are building and validating
    the retrieval/feature pipeline.

    It will eventually be replaced by a trained
    LightGBM LambdaRank model.
    """

    name = "rrf"

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def predict(
        self,
        features: List[Dict],
    ) -> List[float]:

        scores = []

        for row in features:

            score = 0.0

            for key, value in row.items():

                if key.endswith("_rank"):

                    rank = value

                    if rank < 9999:
                        score += 1.0 / (
                            self.rrf_k + rank
                        )

            scores.append(score)

        return scores

class LightGBMRanker(Ranker):
    """
    Rank candidates using a trained LightGBM LambdaRank model.
    """

    name = "lightgbm"

    def __init__(self, model_path: str):
        import lightgbm as lgb

        self.model = lgb.Booster(
            model_file=model_path
        )

        self.feature_names = self.model.feature_name()

    def predict(
        self,
        features: List[Dict],
    ) -> List[float]:

        if not features:
            return []

        X = []

        for row in features:
            X.append([
                row.get(feature, 0.0)
                for feature in self.feature_names
            ])

        return self.model.predict(X).tolist()