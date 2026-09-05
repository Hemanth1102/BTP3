from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QueryState:
    """
    Structured representation of the user's current recommendation
    request and relevant conversational context.

    This object is the contract between the planner and retrievers.
    """

    # ---------------------------------------------------------
    # Core retrieval text
    # ---------------------------------------------------------

    raw_query: str = ""

    bm25_query: str = ""

    semantic_query: str = ""

    # ---------------------------------------------------------
    # Explicit entities
    # ---------------------------------------------------------

    artist_names: List[str] = field(default_factory=list)

    album_names: List[str] = field(default_factory=list)

    track_titles: List[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Musical semantics
    # ---------------------------------------------------------

    genre_tags: List[str] = field(default_factory=list)

    mood_phrases: List[str] = field(default_factory=list)

    sonic_descriptors: List[str] = field(default_factory=list)

    lyrical_themes: List[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Temporal constraints
    # ---------------------------------------------------------

    year_terms: List[str] = field(default_factory=list)

    min_year: Optional[int] = None

    max_year: Optional[int] = None

    # ---------------------------------------------------------
    # Conversational anchors
    # ---------------------------------------------------------

    anchor_track_ids: List[str] = field(default_factory=list)

    # Tracks/artists the user explicitly disliked/rejected.
    rejected_track_ids: List[str] = field(default_factory=list)

    rejected_artists: List[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Recommendation intent
    # ---------------------------------------------------------

    intent: str = "discovery"

    need_personalization: bool = True

    need_new_artist: bool = False

    need_new_track: bool = False

    # ---------------------------------------------------------
    # Query constraints
    # ---------------------------------------------------------

    negative_constraints: List[str] = field(default_factory=list)

    # ---------------------------------------------------------
    # Optional entity / planner metadata
    # ---------------------------------------------------------

    mentioned_entities: List[str] = field(default_factory=list)

    planner_confidence: float = 1.0

    def to_dict(self):
        return {
            "raw_query": self.raw_query,
            "bm25_query": self.bm25_query,
            "semantic_query": self.semantic_query,
            "artist_names": self.artist_names,
            "album_names": self.album_names,
            "track_titles": self.track_titles,
            "genre_tags": self.genre_tags,
            "mood_phrases": self.mood_phrases,
            "sonic_descriptors": self.sonic_descriptors,
            "lyrical_themes": self.lyrical_themes,
            "year_terms": self.year_terms,
            "min_year": self.min_year,
            "max_year": self.max_year,
            "anchor_track_ids": self.anchor_track_ids,
            "rejected_track_ids": self.rejected_track_ids,
            "rejected_artists": self.rejected_artists,
            "intent": self.intent,
            "need_personalization": self.need_personalization,
            "need_new_artist": self.need_new_artist,
            "need_new_track": self.need_new_track,
            "negative_constraints": self.negative_constraints,
            "mentioned_entities": self.mentioned_entities,
            "planner_confidence": self.planner_confidence,
        }