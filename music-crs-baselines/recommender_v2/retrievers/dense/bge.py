import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..base import BaseRetriever, RetrievalResult


class BGERetriever(BaseRetriever):
    """
    Dense retrieval using BAAI/bge-large-en-v1.5 over track metadata.

    Track embeddings are computed once and cached to disk.
    Query embeddings are computed at retrieval time.
    """

    name = "bge"

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIM = 1024

    QUERY_INSTRUCTION = (
        "Represent this sentence for searching relevant music tracks: "
    )

    def __init__(
        self,
        dataset_name: str,
        split_types: List[str],
        corpus_types: List[str],
        cache_dir: str = "./cache",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.cache_dir = cache_dir
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length

        self.index_dir = os.path.join(
            cache_dir,
            "bge",
            self.MODEL_NAME.replace("/", "_"),
            "_".join(corpus_types),
        )

        os.makedirs(self.index_dir, exist_ok=True)

        print(f"Loading {self.MODEL_NAME}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            dtype=torch.float32,
        )

        self.model.to(self.device)
        self.model.eval()

        self._load_corpus()
        self._load_or_build_index()

    def _load_corpus(self):
        from datasets import load_dataset, concatenate_datasets

        dataset = load_dataset(self.dataset_name)

        combined = concatenate_datasets(
            [dataset[split] for split in self.split_types]
        )

        self.metadata = {
            item["track_id"]: item
            for item in combined
        }

    def _stringify_metadata(self, metadata):
        parts = []

        for field in self.corpus_types:
            value = metadata.get(field, "")

            if isinstance(value, list):
                value = ", ".join(str(x) for x in value)

            parts.append(f"{field}: {value}")

        return "\n".join(parts)

    @staticmethod
    def _mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()

        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    def _encode_documents(
        self,
        texts: List[str],
    ) -> torch.Tensor:
        embeddings = []

        with torch.inference_mode():
            for start in range(
                0,
                len(texts),
                self.batch_size,
            ):
                batch_texts = texts[
                    start:start + self.batch_size
                ]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }

                outputs = self.model(**inputs)

                # BGE v1.5 uses CLS pooling in its model-card examples.
                pooled = outputs.last_hidden_state[:, 0]

                pooled = F.normalize(
                    pooled,
                    p=2,
                    dim=1,
                )

                embeddings.append(pooled.cpu())

        return torch.cat(
            embeddings,
            dim=0,
        ).contiguous()

    def _encode_query(self, query: str) -> torch.Tensor:
        query_text = self.QUERY_INSTRUCTION + query

        inputs = self.tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state[:, 0]

            embedding = F.normalize(
                embedding,
                p=2,
                dim=1,
            )

        return embedding[0].cpu()

    def _load_or_build_index(self):
        embeddings_path = os.path.join(
            self.index_dir,
            "embeddings.pt",
        )

        track_ids_path = os.path.join(
            self.index_dir,
            "track_ids.json",
        )

        if (
            os.path.exists(embeddings_path)
            and os.path.exists(track_ids_path)
        ):
            print("Loading cached BGE embeddings...")

            self.track_embeddings = torch.load(
                embeddings_path,
                map_location="cpu",
                weights_only=True,
            )

            with open(
                track_ids_path,
                "r",
                encoding="utf-8",
            ) as f:
                self.track_ids = json.load(f)

            return

        print(
            "BGE cache not found. "
            "Building track embeddings..."
        )

        self.track_ids = list(self.metadata.keys())

        texts = [
            self._stringify_metadata(self.metadata[track_id])
            for track_id in self.track_ids
        ]

        self.track_embeddings = self._encode_documents(texts)

        if self.track_embeddings.shape[1] != self.EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected BGE dimension: "
                f"{self.track_embeddings.shape[1]}"
            )

        torch.save(
            self.track_embeddings,
            embeddings_path,
        )

        with open(
            track_ids_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.track_ids, f)

    def retrieve(
        self,
        query: str,
        topk: int = 100,
        user_id: Optional[str] = None,
        anchor_track_ids=None,
        session_track_ids=None,
    ) -> List[RetrievalResult]:

        query_embedding = self._encode_query(query)

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