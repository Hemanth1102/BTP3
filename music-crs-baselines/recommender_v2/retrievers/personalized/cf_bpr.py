from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset

from ..base import BaseRetriever, RetrievalResult

EMBEDDING_DIM = 128

class CFBPRRetriever(BaseRetriever):
    """
    Personalized retrieval using CF-BPR user/item embeddings.

    User embedding:
        user_id -> 128-dimensional vector

    Track embedding:
        track_id -> 128-dimensional vector

    Score:
        cosine similarity between user and track embeddings.
    """

    name = "cf_bpr"

    def __init__(
        self,
        user_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_split: str = "train",
        track_split: str = "all_tracks",
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.user_dataset_name = user_dataset_name
        self.track_dataset_name = track_dataset_name

        # Load user embeddings.
        user_dataset = load_dataset(user_dataset_name)

        if user_split not in user_dataset:
            raise ValueError(
                f"User split '{user_split}' not found. "
                f"Available: {list(user_dataset.keys())}"
            )

        self.user_embeddings: Dict[str, torch.Tensor] = {}

        for row in user_dataset[user_split]:
            embedding = row["cf-bpr"]

            if not embedding or len(embedding) != EMBEDDING_DIM:
                continue

            self.user_embeddings[row["user_id"]] = torch.tensor(
                embedding,
                dtype=torch.float32,
            )

        # Load track embeddings.
        track_dataset = load_dataset(track_dataset_name)

        if track_split not in track_dataset:
            raise ValueError(
                f"Track split '{track_split}' not found. "
                f"Available: {list(track_dataset.keys())}"
            )

        track_rows = track_dataset[track_split]

        self.track_ids: List[str] = []
        track_vectors = []

        for row in track_rows:
            embedding = row["cf-bpr"]

            # Some tracks may not have a CF-BPR embedding.
            if not embedding or len(embedding) != EMBEDDING_DIM:
                continue

            self.track_ids.append(row["track_id"])

            track_vectors.append(
                torch.tensor(
                    embedding,
                    dtype=torch.float32,
                )
            )

        if not track_vectors:
            raise RuntimeError(
                "No valid CF-BPR track embeddings were found."
            )

        self.track_embeddings = torch.stack(track_vectors)

        self.track_embeddings = F.normalize(
            self.track_embeddings,
            p=2,
            dim=1,
        )

    def retrieve(
        self,
        query: str,
        topk: int = 100,
        user_id: Optional[str] = None,
        anchor_track_ids=None,
        session_track_ids=None,
    ) -> List[RetrievalResult]:

        if user_id is None:
            return []

        if user_id not in self.user_embeddings:
            return []

        user_embedding = self.user_embeddings[user_id]

        user_embedding = F.normalize(
            user_embedding,
            p=2,
            dim=0,
        )

        # [N, 128] @ [128] -> [N]
        scores = torch.matmul(
            self.track_embeddings,
            user_embedding,
        )

        topk = min(topk, scores.shape[0])

        values, indices = torch.topk(
            scores,
            k=topk,
        )

        results = []

        for rank, (score, index) in enumerate(
            zip(values.tolist(), indices.tolist()),
            start=1,
        ):
            results.append(
                RetrievalResult(
                    track_id=self.track_ids[index],
                    score=float(score),
                    rank=rank,
                    source=self.name,
                )
            )

        return results