"""Shared item-to-item retrieval over normalized catalogue embeddings."""

import torch
import torch.nn.functional as F
from datasets import load_dataset

from ..base import BaseRetriever, RetrievalResult


TRACK_EMBEDDING_DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"


class ItemEmbeddingRetriever(BaseRetriever):
    def __init__(self, dataset_name=TRACK_EMBEDDING_DATASET, split="all_tracks", device="cpu",
                 *, mode="mean", gamma=0.6, embedding_source=None):
        suffixes = {"mean": "", "decay": "_decay", "maxsim": "_max"}
        if mode not in suffixes:
            raise ValueError(f"Unknown item-to-item mode {mode!r}.")
        if not 0 < gamma <= 1:
            raise ValueError("Item-to-item gamma must be in (0, 1].")

        self.name = type(self).name + suffixes[mode]
        self.mode = mode
        self.gamma = gamma
        self.dataset_name = dataset_name
        self.split = split
        self.device = torch.device(device)

        if embedding_source is not None:
            if (type(embedding_source) is not type(self)
                    or embedding_source.dataset_name != dataset_name
                    or embedding_source.split != split):
                raise ValueError("The shared item embeddings must use the same catalogue and space.")
            self.track_ids = embedding_source.track_ids
            self.track_embeddings = embedding_source.track_embeddings
            self.track_to_index = embedding_source.track_to_index
            return

        print(f"Loading {self.EMBEDDING_COLUMN} item-to-item embeddings...")
        dataset = load_dataset(dataset_name)
        if split not in dataset:
            raise ValueError(f"Split {split!r} not found. Available splits: {list(dataset)}")

        self.track_ids = []
        rows = dataset[split]
        vectors = torch.empty((len(rows), self.EMBEDDING_DIM), dtype=torch.float32)
        for row in rows:
            embedding = row[self.EMBEDDING_COLUMN]
            if embedding is not None and len(embedding) == self.EMBEDDING_DIM:
                vectors[len(self.track_ids)] = torch.as_tensor(embedding, dtype=torch.float32)
                self.track_ids.append(row["track_id"])
        if not self.track_ids:
            raise RuntimeError(f"No valid {self.EMBEDDING_COLUMN} track embeddings found.")

        vectors = vectors[:len(self.track_ids)]
        self.track_embeddings = F.normalize(vectors, dim=1, out=vectors)
        self.track_to_index = {track_id: i for i, track_id in enumerate(self.track_ids)}
        print(f"Loaded {len(self.track_ids)} {self.EMBEDDING_COLUMN} track embeddings "
              f"with dimension {self.EMBEDDING_DIM}.")

    def retrieve(self, query, topk=100, user_id=None, anchor_track_ids=None,
                 session_track_ids=None):
        if not anchor_track_ids or topk <= 0:
            return []

        anchors = anchor_track_ids[-4:] if self.mode == "maxsim" else anchor_track_ids
        indexed = [(age, self.track_to_index[tid])
                   for age, tid in enumerate(reversed(anchors)) if tid in self.track_to_index]
        if not indexed:
            return []

        ages, indices = zip(*indexed)
        vectors = self.track_embeddings[list(indices)]
        if self.mode == "maxsim":
            scores = (self.track_embeddings @ vectors.T).amax(dim=1)
        else:
            if self.mode == "decay":
                weights = vectors.new_tensor([self.gamma ** age for age in ages])
                centroid = weights @ vectors
            else:
                # Preserve the legacy mean's summation order.
                centroid = vectors.flip(0).mean(dim=0)
            scores = self.track_embeddings @ F.normalize(centroid, dim=0)

        played = set(anchor_track_ids)
        if self.mode != "mean":
            played.update(session_track_ids or ())
        scores[[self.track_to_index[tid] for tid in played if tid in self.track_to_index]] = -torch.inf
        values, indices = torch.topk(scores, k=min(topk, len(self.track_ids)))
        return [RetrievalResult(self.track_ids[index], score, rank, self.name)
                for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1)
                if score != -float("inf")]
