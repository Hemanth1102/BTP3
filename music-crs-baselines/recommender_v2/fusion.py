from collections import defaultdict
from typing import Dict, List

from .retrievers.base import RetrievalResult


class CandidateFusion:
    """
    Combine candidates from multiple retrieval systems.

    Each track appears once in the final candidate set, while preserving
    the evidence from every retrieval source.
    """

    def __init__(self):
        pass

    def fuse(
        self,
        retrieval_lists: List[List[RetrievalResult]],
    ) -> List[Dict]:

        candidates = {}

        for results in retrieval_lists:

            for result in results:

                track_id = result.track_id

                if track_id not in candidates:
                    candidates[track_id] = {
                        "track_id": track_id,
                        "sources": {},
                    }

                candidates[track_id]["sources"][result.source] = {
                    "rank": result.rank,
                    "score": result.score,
                }

        return list(candidates.values())

    def build_features(
        self,
        candidates: List[Dict],
    ) -> List[Dict]:

        """
        Convert the source evidence into a flat feature dictionary.

        This is what we will eventually feed into LightGBM.
        """

        all_sources = set()

        for candidate in candidates:
            all_sources.update(candidate["sources"].keys())

        all_sources = sorted(all_sources)

        features = []

        for candidate in candidates:

            row = {
                "track_id": candidate["track_id"],
            }

            sources = candidate["sources"]

            for source in all_sources:

                if source in sources:

                    row[f"{source}_rank"] = sources[source]["rank"]
                    row[f"{source}_score"] = sources[source]["score"]
                    row[f"{source}_retrieved"] = 1

                else:

                    row[f"{source}_rank"] = 9999
                    row[f"{source}_score"] = 0.0
                    row[f"{source}_retrieved"] = 0

            row["num_sources"] = len(sources)

            features.append(row)

        return features