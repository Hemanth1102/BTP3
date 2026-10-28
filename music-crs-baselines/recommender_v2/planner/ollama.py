import re
from typing import Any, Dict, List, Optional

import ollama


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def query_state_schema() -> Dict[str, Any]:
    """
    JSON schema of the planner output (the QueryState fields). Passed to
    Ollama as `format=` so decoding is constrained to exactly these fields and
    types: no missing/extra fields, no malformed JSON.
    """
    from ..features import INTENTS

    text = {"type": "string"}
    texts = {"type": "array", "items": {"type": "string"}}
    year = {"anyOf": [{"type": "integer"}, {"type": "null"}]}

    properties = {
        "bm25_query": text,
        "semantic_query": text,
        "artist_names": texts,
        "album_names": texts,
        "track_titles": texts,
        "genre_tags": texts,
        "mood_phrases": texts,
        "sonic_descriptors": texts,
        "lyrical_themes": texts,
        "year_terms": texts,
        "min_year": year,
        "max_year": year,
        "rejected_track_ids": texts,
        "rejected_artists": texts,
        "intent": {"type": "string", "enum": list(INTENTS)},
        "need_personalization": {"type": "boolean"},
        "need_new_artist": {"type": "boolean"},
        "need_new_track": {"type": "boolean"},
        "negative_constraints": texts,
        "mentioned_entities": texts,
        "planner_confidence": {"type": "number"},
    }

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


class OllamaQwenPlanner:
    """
    Thin wrapper around a local Qwen model running through Ollama.

    This object is callable so it can be passed directly to
    QueryPlanner(llm=..., mode="llm").

    Settings chosen for a structured-extraction job on a small GPU:

    * think=False      Qwen3 otherwise writes a long reasoning block before the
                       JSON. That multiplies latency (25-130 s/query on an
                       RTX 3050) and adds nothing for field extraction.
    * format=schema    Ollama constrains decoding to the QueryState schema
                       (falls back to plain JSON mode if the installed Ollama
                       rejects the schema).
    * temperature=0    Deterministic plans -> reproducible dataset.
    * keep_alive       Keep the model in VRAM between queries.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        think: bool = False,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
        use_schema: bool = True,
    ):
        self.model = model
        self.think = think
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.use_schema = use_schema
        self._schema = query_state_schema() if use_schema else None

    def _chat(self, kwargs):
        try:
            return ollama.chat(think=self.think, **kwargs)
        except TypeError:
            # Older ollama-python without the `think` argument.
            return ollama.chat(**kwargs)

    def __call__(self, prompt: str, attempt: int = 0) -> str:
        """
        `attempt` > 0 is a retry after unparseable output: the temperature is
        raised so the model does not reproduce the same bad answer.
        """

        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=self._schema if self._schema else "json",
            options={
                "temperature": self.temperature + 0.3 * attempt,
                "num_ctx": self.num_ctx,
            },
            keep_alive=self.keep_alive,
        )

        try:
            response = self._chat(kwargs)
        except Exception:
            if self._schema is None:
                raise
            # Ollama too old for schema formats (or a transient failure):
            # use plain JSON mode from now on.
            self._schema = None
            kwargs["format"] = "json"
            response = self._chat(kwargs)

        text = response["message"]["content"]

        # Defensive: drop any reasoning block that still leaks into the answer.
        return _THINK_BLOCK.sub("", text).strip()