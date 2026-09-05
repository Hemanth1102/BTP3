from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset

from ..base import BaseRetriever, RetrievalResult


class CFItemToItemRetriever(BaseRetriever):
    """
    Item-to-item retrieval using CF-BPR track embeddings.

    Given one or more anchor tracks, compute their centroid in CF-BPR
    space and retrieve the nearest tracks.

    This is different from user -> item CF-BPR:

        user -> item:
            user embedding -> track embedding

        item -> item:
            anchor track embedding -> track embedding
    """

    name = "item_cf_bpr"

    EMBEDDING_COLUMN = "cf-bpr"
    EMBEDDING_DIM = 128

    def __init__(
        self,
        dataset_name: str = (
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
        ),
        split: str = "all_tracks",
        device: str = "cpu",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.device = torch.device(device)

        print("Loading CF-BPR item-to-item embeddings...")

        dataset = load_dataset(dataset_name)

        if split not in dataset:
            raise ValueError(
                f"Split '{split}' not found. "
                f"Available splits: {list(dataset.keys())}"
            )

        rows = dataset[split]

        self.track_ids: List[str] = []
        vectors: List[torch.Tensor] = []

        for row in rows:
            embedding = row[self.EMBEDDING_COLUMN]

            if (
                not embedding
                or len(embedding) != self.EMBEDDING_DIM
            ):
                continue

            self.track_ids.append(row["track_id"])

            vectors.append(
                torch.tensor(
                    embedding,
                    dtype=torch.float32,
                )
            )

        if not vectors:
            raise RuntimeError(
                "No valid CF-BPR track embeddings found."
            )

        self.track_embeddings = F.normalize(
            torch.stack(vectors),
            p=2,
            dim=1,
        )

        # Fast lookup from track ID -> row in embedding matrix.
        self.track_to_index: Dict[str, int] = {
            track_id: index
            for index, track_id in enumerate(self.track_ids)
        }

        print(
            f"Item CF-BPR: loaded "
            f"{len(self.track_ids)} tracks, "
            f"dimension={self.track_embeddings.shape[1]}"
        )

    def _get_anchor_embedding(
        self,
        anchor_track_ids: List[str],
    ) -> Optional[torch.Tensor]:

        valid_embeddings = []

        for track_id in anchor_track_ids:

            index = self.track_to_index.get(track_id)

            if index is None:
                continue

            valid_embeddings.append(
                self.track_embeddings[index]
            )

        if not valid_embeddings:
            return None

        # Mean/centroid of anchor embeddings.
        centroid = torch.stack(
            valid_embeddings,
            dim=0,
        ).mean(dim=0)

        centroid = F.normalize(
            centroid,
            p=2,
            dim=0,
        )

        return centroid

    def retrieve(
        self,
        query: str,
        topk: int = 100,
        user_id: Optional[str] = None,
        anchor_track_ids: Optional[List[str]] = None,
        session_track_ids: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:

        if not anchor_track_ids:
            return []

        anchor_embedding = self._get_anchor_embedding(
            anchor_track_ids
        )

        if anchor_embedding is None:
            return []

        scores = torch.matmul(
            self.track_embeddings,
            anchor_embedding,
        )

        # Never recommend the anchor tracks themselves.
        for track_id in anchor_track_ids:

            index = self.track_to_index.get(track_id)

            if index is not None:
                scores[index] = float("-inf")

        topk = min(
            topk,
            scores.shape[0],
        )

        values, indices = torch.topk(
            scores,
            k=topk,
        )

        results = []

        rank = 1

        for score, index in zip(
            values.tolist(),
            indices.tolist(),
        ):

            # Defensive check in case the requested top-k exceeds
            # the number of usable non-anchor items.
            if score == float("-inf"):
                continue

            results.append(
                RetrievalResult(
                    track_id=self.track_ids[index],
                    score=float(score),
                    rank=rank,
                    source=self.name,
                )
            )

            rank += 1

        return results