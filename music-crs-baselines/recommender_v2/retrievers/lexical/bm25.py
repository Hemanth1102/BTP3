import json
import os
from typing import List, Dict, Optional

import bm25s
from datasets import load_dataset, concatenate_datasets

from ..base import BaseRetriever, RetrievalResult


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever over track metadata.

    Unlike the original baseline, this class preserves BM25 scores so
    they can be used later as features by the reranker.
    """

    name = "bm25"

    def __init__(
        self,
        dataset_name: str,
        split_types: List[str],
        corpus_types: List[str],
        cache_dir: str = "./cache",
    ):
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.cache_dir = cache_dir

        self.corpus_name = "_".join(corpus_types)
        self.index_dir = os.path.join(
            cache_dir,
            "bm25",
            self.corpus_name,
        )

        self.metadata_dict = self._load_corpus()

        if os.path.exists(self.index_dir):
            self.bm25_model, self.track_ids = self._load_index()
        else:
            self._build_index()
            self.bm25_model, self.track_ids = self._load_index()

    def _load_corpus(self) -> Dict[str, Dict]:
        dataset = load_dataset(self.dataset_name)

        combined = concatenate_datasets(
            [dataset[split] for split in self.split_types]
        )

        return {
            item["track_id"]: item
            for item in combined
        }

    def _stringify_metadata(self, metadata: Dict) -> str:
        parts = []

        for field in self.corpus_types:
            value = metadata.get(field, "")

            if isinstance(value, list):
                value = ", ".join(str(x) for x in value)

            parts.append(f"{field}: {value}")

        return "\n".join(parts)

    def _build_index(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)

        track_ids = list(self.metadata_dict.keys())

        corpus = [
            self._stringify_metadata(self.metadata_dict[track_id])
            for track_id in track_ids
        ]

        tokens = bm25s.tokenize(corpus)

        model = bm25s.BM25()
        model.index(tokens)

        model.save(
            self.index_dir,
            corpus=corpus,
        )

        with open(
            os.path.join(self.index_dir, "track_ids.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(track_ids, f)

    def _load_index(self):
        model = bm25s.BM25.load(
            self.index_dir,
            load_corpus=True,
        )

        with open(
            os.path.join(self.index_dir, "track_ids.json"),
            "r",
            encoding="utf-8",
        ) as f:
            track_ids = json.load(f)

        return model, track_ids

    def retrieve(
        self,
        query: str,
        topk: int = 100,
        user_id: Optional[str] = None,
        anchor_track_ids=None,
        session_track_ids=None,
    ) -> List[RetrievalResult]:

        query_tokens = bm25s.tokenize([query.lower()])

        scores = self.bm25_model.retrieve(
            query_tokens,
            k=topk,
            return_as="tuple",
        )

        documents = scores.documents[0]
        score_values = scores.scores[0]

        results = []

        for rank, (doc, score) in enumerate(
            zip(documents, score_values),
            start=1,
        ):
            track_index = doc["id"]
            track_id = self.track_ids[track_index]

            results.append(
                RetrievalResult(
                    track_id=track_id,
                    score=float(score),
                    rank=rank,
                    source=self.name,
                )
            )

        return results