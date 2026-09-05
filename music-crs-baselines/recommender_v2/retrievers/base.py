from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalResult:
    track_id: str
    score: float
    rank: int
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "score": float(self.score),
            "rank": int(self.rank),
            "source": self.source,
        }


class BaseRetriever(ABC):
    """
    Common interface for all retrieval models.

    Retrievers can use different pieces of context:
      - query
      - user_id
      - anchor_track_ids
      - session_track_ids
    """

    name: str = "base"

    @abstractmethod
    def retrieve(
        self,
        query: str,
        topk: int = 100,
        user_id: Optional[str] = None,
        anchor_track_ids: Optional[List[str]] = None,
        session_track_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        raise NotImplementedError