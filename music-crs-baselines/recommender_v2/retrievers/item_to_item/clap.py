from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset

from ..base import BaseRetriever, RetrievalResult


class CLAPItemToItemRetriever(BaseRetriever):
    """
    Item-to-item retrieval using precomputed LAION-CLAP audio embeddings.

    Given one or more anchor tracks, compute their centroid in CLAP
    audio-embedding space and retrieve the most similar tracks.

    Anchor tracks themselves are excluded from the result.
    """

    name = "item_clap"

    EMBEDDING_COLUMN = "audio-laion_clap"
    EMBEDDING_DIM = 512

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

        print("Loading CLAP item-to-item embeddings...")

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
                "No valid CLAP audio embeddings found."
            )

        self.track_embeddings = F.normalize(
            torch.stack(vectors),
            p=2,
            dim=1,
        )

        self.track_to_index: Dict[str, int] = {
            track_id: index
            for index, track_id in enumerate(self.track_ids)
        }

        print(
            f"Item CLAP: loaded "
            f"{len(self.track_ids)} tracks, "
            f"dimension={self.track_embeddings.shape[1]}"
        )

    def _get_anchor_embedding(
        self,
        anchor_track_ids: List[str],
    ) -> Optional[torch.Tensor]:
        """
        Compute a normalized centroid from valid anchor tracks.
        """

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

        anchor_matrix = torch.stack(
            valid_embeddings,
            dim=0,
        )

        centroid = anchor_matrix.mean(
            dim=0
        )

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

        # Cosine similarity because both track and anchor vectors
        # are L2-normalized.
        scores = torch.matmul(
            self.track_embeddings,
            anchor_embedding,
        )

        # Never return an anchor itself.
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