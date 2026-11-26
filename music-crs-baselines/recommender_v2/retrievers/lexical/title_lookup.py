from collections import defaultdict
from typing import List

from ...text_norm import base_title
from .entity_index import EntityIndex, as_list, normalize_name
from .lookup_base import EntityLookupRetriever, load_catalog, popularity_of

class TitleLookupRetriever(EntityLookupRetriever):
    """
    Tracks whose title matches a song the user named (QueryState.track_titles),
    e.g. "play Creep". Titles repeat ("Intro", "Hello"), so versions by an
    artist the user also named come first. No free-text fallback: common
    titles would match ordinary words.
    """

    name = "title_lookup"
    field = "track_titles"
    artist_preference = True
    text_fallback = False

    def __init__(self, dataset_name: str, split_types: List[str]):
        self.dataset_name = dataset_name
        self.split_types = split_types
        print("Loading title lookup...")

        self.index = EntityIndex()
        self.track_artists = defaultdict(set)
        for row in load_catalog(dataset_name, split_types):
            title = row.get("track_name")
            if not title:
                continue
            pop = popularity_of(row)
            self.index.add(title, row["track_id"], pop)
            short = base_title(str(title))
            if short and normalize_name(short) != normalize_name(title):
                self.index.add(short, row["track_id"], pop)
            for artist in as_list(row.get("artist_name")):
                self.track_artists[row["track_id"]].add(normalize_name(artist))
        self.index.finalize()
        print(f"Title lookup: {len(self.index)} titles")
