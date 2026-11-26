"""
Common code of the artist / album / title lookup retrievers.

They read the planner's structured fields (QueryState.artist_names,
album_names, track_titles) instead of searching for names inside the
semantic_query sentence, skip rejected artists and already-played tracks,
and fall back to scanning the user's own message only when the planner
extracted nothing.
"""

from typing import Dict, List, Optional, Set

from ..base import BaseRetriever, RetrievalResult
from .entity_index import EXACT, FUZZY, TEXT, EntityIndex, as_list, interleave, normalize_name

MATCH_SCORE = {EXACT: 1.0, FUZZY: 0.9, TEXT: 0.7}

# Single words that are also catalog names but almost always mean the word
# itself in a request ("I can't stop...", "something new"). Only used for the
# free-text fallback; structured planner names are matched normally.
COMMON_WORDS = set("""
a an and or but if of to in on at by for with from up down out over again back
i me my we you your he she it they them this that these those here there
is are was were be been am do does did can could will would should may might
not no yes ok okay so very really just also too more most less some any all
like love want need give play find show make let get got know think feel
song songs track tracks music album albums artist artists band singer
new old good great best better nice cool fun happy sad chill calm soft hard
night day morning evening summer winter rain sun home time life world heart
mood vibe vibes energy dance party drive road sleep study work focus relax
rock pop jazz blues soul metal folk indie country rap hip hop house techno
classical acoustic electronic live remix cover version another something else
""".split())


def load_catalog(dataset_name: str, split_types: List[str]):
    from datasets import load_dataset, concatenate_datasets

    dataset = load_dataset(dataset_name)
    return concatenate_datasets([dataset[s] for s in split_types])


def popularity_of(row) -> Optional[float]:
    try:
        v = row.get("popularity")
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class EntityLookupRetriever(BaseRetriever):
    """Subclasses fill self.index and self.track_artists, and set `field`."""

    name = "entity_lookup"
    field = ""                       # QueryState field holding the names
    uses_query_state = True          # the pipeline passes query_state=...
    text_fallback = True
    artist_preference = False        # album/title: requested artist's tracks first

    index: EntityIndex
    track_artists: Dict[str, Set[str]]

    def _names(self, query_state) -> List[str]:
        return as_list(getattr(query_state, self.field, None)) if query_state is not None else []

    def _match(self, query: str, query_state):
        hits = self.index.match_names(self._names(query_state))
        if not hits and self.text_fallback:
            raw = getattr(query_state, "raw_query", "") if query_state is not None else ""
            hits = [h for h in self.index.match_text(raw or "", query or "")
                    if " " in h[0] or h[0] not in COMMON_WORDS]
        return hits

    def retrieve(
        self,
        query: str,
        topk: int = 100,
        user_id: Optional[str] = None,
        anchor_track_ids=None,
        session_track_ids=None,
        query_state=None,
    ) -> List[RetrievalResult]:

        hits = self._match(query, query_state)
        if not hits:
            return []

        rejected = {normalize_name(a) for a in
                    as_list(getattr(query_state, "rejected_artists", None))} - {""}
        wanted = {normalize_name(a) for a in
                  as_list(getattr(query_state, "artist_names", None))} - {""} - rejected

        lists, kinds, preferred = [], [], set()
        for key, kind in hits:
            if self.field == "artist_names" and key in rejected:
                continue
            tracks = [t for t in self.index.by_name.get(key, [])
                      if not (self.track_artists.get(t, set()) & rejected)]
            if self.artist_preference and wanted:
                mine = [t for t in tracks if self.track_artists.get(t, set()) & wanted]
                preferred.update(mine)
                tracks = mine + [t for t in tracks if t not in preferred]
            if tracks:
                lists.append(tracks)
                kinds.append(kind)

        skip = set(session_track_ids or []) | set(anchor_track_ids or [])
        picked = interleave(lists, topk, skip)

        out = []
        for rank, (tid, j) in enumerate(picked, start=1):
            score = MATCH_SCORE[kinds[j]]
            if self.artist_preference and wanted and tid not in preferred:
                score *= 0.6                    # right name, but not the requested artist
            out.append(RetrievalResult(track_id=tid, score=score, rank=rank, source=self.name))
        return out
