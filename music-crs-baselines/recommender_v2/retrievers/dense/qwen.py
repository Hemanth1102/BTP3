from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Qwen3EmbeddingEncoder:
    """
    Query encoder for Qwen3-Embedding-0.6B.

    Produces 1024-dimensional normalized embeddings.
    """

    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIM = 1024

    TASK_INSTRUCTION = (
        "Given a conversational music recommendation request, "
        "retrieve relevant music tracks."
    )

    def __init__(
        self,
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.device = torch.device(device)
        self.max_length = max_length

        print(
            f"Loading {self.MODEL_NAME}..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            padding_side="left",
        )

        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            dtype=torch.float32,
        )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        left_padding = (
            attention_mask[:, -1].sum()
            == attention_mask.shape[0]
        )

        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = (
            attention_mask.sum(dim=1) - 1
        )

        batch_indices = torch.arange(
            last_hidden_states.shape[0],
            device=last_hidden_states.device,
        )

        return last_hidden_states[
            batch_indices,
            sequence_lengths,
        ]

    def _format_query(
        self,
        query: str,
    ) -> str:

        return (
            f"Instruct: {self.TASK_INSTRUCTION}\n"
            f"Query:{query}"
        )

    def encode(
        self,
        queries: List[str],
    ) -> torch.Tensor:

        formatted_queries = [
            self._format_query(query)
            for query in queries
        ]

        inputs = self.tokenizer(
            formatted_queries,
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

            outputs = self.model(
                **inputs
            )

            embeddings = self._last_token_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )

        return F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

    def __call__(
        self,
        query: str,
    ) -> torch.Tensor:

        return self.encode(
            [query]
        )[0]