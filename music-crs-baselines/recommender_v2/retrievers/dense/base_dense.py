from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset

from ..base import BaseRetriever, RetrievalResult


class DenseEmbeddingRetriever(BaseRetriever):
    """
    Generic retriever for precomputed track embeddings.

    The query encoder must return a vector in the same embedding
    space as the track embeddings.
    """

    def __init__(
        self,
        name: str,
        dataset_name: str,
        embedding_column: str,
        split: str,
        query_encoder: Callable[[str], torch.Tensor],
        embedding_dim: int,
    ):
        self.name = name
        self.query_encoder = query_encoder
        self.embedding_column = embedding_column

        dataset = load_dataset(dataset_name)
        rows = dataset[split]

        self.track_ids = []
        vectors = []

        for row in rows:
            embedding = row[embedding_column]

            if not embedding:
                continue

            if len(embedding) != embedding_dim:
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
                f"No valid embeddings found for {embedding_column}"
            )

        self.track_embeddings = F.normalize(
            torch.stack(vectors),
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

        query_embedding = self.query_encoder(query)

        query_embedding = torch.as_tensor(
            query_embedding,
            dtype=torch.float32,
        )

        if query_embedding.ndim != 1:
            raise ValueError(
                f"{self.name} query embedding must be 1-D"
            )

        if query_embedding.shape[0] != self.track_embeddings.shape[1]:
            raise ValueError(
                f"{self.name}: query dimension "
                f"{query_embedding.shape[0]} != "
                f"track dimension {self.track_embeddings.shape[1]}"
            )

        query_embedding = F.normalize(
            query_embedding,
            p=2,
            dim=0,
        )

        scores = torch.matmul(
            self.track_embeddings,
            query_embedding,
        )

        topk = min(topk, len(scores))

        values, indices = torch.topk(
            scores,
            k=topk,
        )

        return [
            RetrievalResult(
                track_id=self.track_ids[index],
                score=float(score),
                rank=rank,
                source=self.name,
            )
            for rank, (score, index) in enumerate(
                zip(
                    values.tolist(),
                    indices.tolist(),
                ),
                start=1,
            )
        ]