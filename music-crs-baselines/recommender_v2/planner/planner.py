import inspect
import json
from typing import Any, Dict, List, Optional

from .schema import QueryState


class QueryPlanner:
    """
    Query planner for the conversational recommender.

    Development mode can operate without an LLM.
    Production mode will use Qwen3-8B to generate structured JSON.
    """

    def __init__(
        self,
        llm=None,
        mode: str = "mock",
        max_attempts: int = 2,
    ):
        self.llm = llm
        self.mode = mode
        self.max_attempts = max(1, max_attempts)

        # Backends that accept `attempt` get a hotter retry after bad output.
        self._llm_takes_attempt = (
            llm is not None
            and "attempt" in inspect.signature(llm).parameters
        )

    def plan(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        anchor_track_ids: Optional[List[str]] = None,
    ) -> QueryState:

        if self.mode == "mock":
            return self._mock_plan(
                query=query,
                conversation_history=conversation_history,
                anchor_track_ids=anchor_track_ids,
            )

        if self.mode == "llm":
            return self._llm_plan(
                query=query,
                conversation_history=conversation_history,
                anchor_track_ids=anchor_track_ids,
            )

        raise ValueError(
            f"Unknown planner mode: {self.mode}"
        )

    # ---------------------------------------------------------
    # Mock planner
    # ---------------------------------------------------------

    def _mock_plan(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        anchor_track_ids: Optional[List[str]],
    ) -> QueryState:

        history_text = ""

        if conversation_history:

            history_text = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in conversation_history
            )

        combined_text = (
            history_text + "\n" + query
        ).strip()

        # Very simple development behavior.
        #
        # We intentionally do not try to reproduce an LLM here.
        # The purpose is to give every retriever a valid QueryState
        # while we develop the retrieval architecture.

        lower = combined_text.lower()

        intent = "discovery"

        if (
            "similar to" in lower
            or "like this" in lower
            or "sounds like" in lower
        ):
            intent = "similarity"

        elif (
            "by " in lower
            or "artist" in lower
        ):
            intent = "artist_request"

        elif "album" in lower:
            intent = "album_request"

        need_new_artist = any(
            phrase in lower
            for phrase in [
                "new artist",
                "new artists",
                "different artist",
                "someone i haven't heard",
                "discover artists",
            ]
        )

        return QueryState(
            raw_query=query,
            bm25_query=query,
            semantic_query=query,
            anchor_track_ids=list(
                anchor_track_ids or []
            ),
            intent=intent,
            need_personalization=True,
            need_new_artist=need_new_artist,
            planner_confidence=0.1,
        )

    # ---------------------------------------------------------
    # LLM planner
    # ---------------------------------------------------------

    def _llm_plan(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        anchor_track_ids: Optional[List[str]],
    ) -> QueryState:

        if self.llm is None:
            raise RuntimeError(
                "LLM planner requested but no LLM "
                "backend was provided."
            )

        history = conversation_history or []

        prompt = self._build_prompt(
            query=query,
            history=history,
            anchor_track_ids=anchor_track_ids or [],
        )

        last_error = None

        for attempt in range(self.max_attempts):

            try:
                if self._llm_takes_attempt:
                    response = self.llm(prompt, attempt=attempt)
                else:
                    response = self.llm(prompt)

                plan_dict = self._parse_json(response)
                break

            except ValueError as exc:       # unparseable / wrong fields
                last_error = exc

        else:
            raise last_error

        # Planner-provided anchors are allowed, but the pipeline's
        # actual known anchors take precedence.
        plan_dict["anchor_track_ids"] = list(anchor_track_ids or [])
        plan_dict["raw_query"] = query

        return QueryState(
            **plan_dict
        )

    # ---------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------

    def _build_prompt(
        self,
        query: str,
        history: List[Dict[str, str]],
        anchor_track_ids: List[str],
    ) -> str:

        # Only natural language goes to the LLM. Track ids are opaque UUIDs: they
        # cost prompt tokens, and a model asked to echo them spends ~3 s of
        # generation per id for nothing (the pipeline sets anchors itself).
        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history
            if message["role"] in ("user", "assistant")
        )

        return f"""
You are the query-planning component of a conversational music
recommendation system.

Convert the conversation into structured JSON for downstream
retrieval systems.

Extract only information supported by the conversation.

Current user query:
{query}

Conversation history:
{history_text}

Return ONLY valid JSON with exactly these fields:

{{
  "bm25_query": "...",
  "semantic_query": "...",
  "artist_names": [],
  "album_names": [],
  "track_titles": [],
  "genre_tags": [],
  "mood_phrases": [],
  "sonic_descriptors": [],
  "lyrical_themes": [],
  "year_terms": [],
  "min_year": null,
  "max_year": null,
  "rejected_track_ids": [],
  "rejected_artists": [],
  "intent": "discovery",
  "need_personalization": true,
  "need_new_artist": false,
  "need_new_track": false,
  "negative_constraints": [],
  "mentioned_entities": [],
  "planner_confidence": 1.0
}}

Possible intent values:
- discovery
- similarity
- artist_request
- album_request
- mood
- genre
- lyrical
- temporal
- conversational

Do not invent artists, albums, songs, genres, or years.

bm25_query and semantic_query are search text: they must contain only what
the user WANTS. Never put negations ("not", "without", "different from") or
the names of excluded artists in them. Put artists the user does not want in
rejected_artists, and other exclusions in negative_constraints.
"""

    # ---------------------------------------------------------
    # JSON parsing
    # ---------------------------------------------------------

    def _parse_json(
        self,
        response: Any,
    ) -> Dict[str, Any]:

        if isinstance(response, dict):
            data = response

        elif isinstance(response, str):
            text = response.strip()

            # Remove accidental markdown fences.
            if text.startswith("```"):
                lines = text.splitlines()
                lines = [
                    line
                    for line in lines
                    if not line.strip().startswith("```")
                ]
                text = "\n".join(lines).strip()

            try:
                data = json.loads(text)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Planner returned invalid JSON: {exc}"
                ) from exc

        else:
            raise ValueError(
                "Planner response must be a dict or string."
            )

        if not isinstance(data, dict):
            raise ValueError(
                "Planner JSON must be an object."
            )

        required_fields = {
            "raw_query",
            "bm25_query",
            "semantic_query",
            "artist_names",
            "album_names",
            "track_titles",
            "genre_tags",
            "mood_phrases",
            "sonic_descriptors",
            "lyrical_themes",
            "year_terms",
            "min_year",
            "max_year",
            "anchor_track_ids",
            "rejected_track_ids",
            "rejected_artists",
            "intent",
            "need_personalization",
            "need_new_artist",
            "need_new_track",
            "negative_constraints",
            "mentioned_entities",
            "planner_confidence",
        }

        # Set by the pipeline, not generated by the LLM (generating them only
        # costs time); accepted if a backend still returns them.
        pipeline_fields = {"raw_query", "anchor_track_ids"}
        required_fields = required_fields - pipeline_fields

        unknown_fields = set(data.keys()) - required_fields - pipeline_fields

        if unknown_fields:
            raise ValueError(
                f"Planner returned unexpected fields: "
                f"{sorted(unknown_fields)}"
            )

        missing_fields = required_fields - set(data.keys())

        if missing_fields:
            raise ValueError(
                f"Planner response is missing fields: "
                f"{sorted(missing_fields)}"
            )

        return data