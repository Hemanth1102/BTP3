from typing import List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from ..base import BaseRetriever, RetrievalResult


class CLAPTextAudioRetriever(BaseRetriever):
    """
    Text -> audio retrieval using LAION CLAP.

    Query:
        natural-language conversation/query

    Track side:
        precomputed `audio-laion_clap` embeddings
    """

    name = "clap_audio"

    MODEL_NAME = "laion/clap-htsat-unfused"
    EMBEDDING_DIM = 512

    def __init__(
        self,
        dataset_name: str,
        split: str = "all_tracks",
        embedding_column: str = "audio-laion_clap",
        device: str = "cpu",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.embedding_column = embedding_column
        self.device = torch.device(device)

        print(f"Loading {self.MODEL_NAME}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        self._load_track_embeddings()

    def _load_track_embeddings(self):
        print("Loading CLAP audio embeddings...")

        dataset = load_dataset(
            self.dataset_name
        )

        rows = dataset[self.split]

        self.track_ids = []
        vectors = []

        for row in rows:
            embedding = row[self.embedding_column]

            if not embedding:
                continue

            if len(embedding) != self.EMBEDDING_DIM:
                continue

            self.track_ids.append(
                row["track_id"]
            )

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

        print(
            f"CLAP audio: loaded "
            f"{len(self.track_ids)} tracks, "
            f"dimension={self.track_embeddings.shape[1]}"
        )

    def _encode_query(
        self,
        query: str,
    ) -> torch.Tensor:

        inputs = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            output = self.model.get_text_features(
                **inputs
            )

        if hasattr(output, "pooler_output"):
            embedding = output.pooler_output

        elif hasattr(output, "last_hidden_state"):
            embedding = output.last_hidden_state

            if embedding.ndim == 3:
                embedding = embedding[:, 0, :]

        elif torch.is_tensor(output):
            embedding = output

        else:
            raise RuntimeError(
                f"Unexpected CLAP output type: {type(output)}"
            )

        # Remove batch dimension.
        if embedding.ndim == 2:
            embedding = embedding[0]

        if embedding.ndim != 1:
            raise RuntimeError(
                f"Unexpected CLAP embedding shape: "
                f"{tuple(embedding.shape)}"
            )

        if embedding.shape[0] != self.EMBEDDING_DIM:
            raise RuntimeError(
                f"CLAP dimension mismatch: "
                f"query={embedding.shape[0]}, "
                f"expected={self.EMBEDDING_DIM}"
            )

        return F.normalize(
            embedding,
            p=2,
            dim=0,
        )

    def retrieve(
        self,
        query: str,
        topk: int = 100,
        user_id: Optional[str] = None,
        anchor_track_ids=None,
        session_track_ids=None,
    ) -> List[RetrievalResult]:

        query_embedding = self._encode_query(
            query
        )

        scores = torch.matmul(
            self.track_embeddings,
            query_embedding,
        )

        topk = min(
            topk,
            scores.shape[0],
        )

        values, indices = torch.topk(
            scores,
            k=topk,
        )

        results = []

        for rank, (score, index) in enumerate(
            zip(
                values.tolist(),
                indices.tolist(),
            ),
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