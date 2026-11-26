from collections import defaultdict
from typing import List

from .entity_index import EntityIndex, as_list, normalize_name
from .lookup_base import EntityLookupRetriever, load_catalog, popularity_of


class ArtistLookupRetriever(EntityLookupRetriever):
    """
    Tracks of the artists the user asked for (QueryState.artist_names),
    most popular first; several artists share the slots. Falls back to
    artist names written in the user's message.
    """

    name = "artist_lookup"
    field = "artist_names"

    def __init__(self, dataset_name: str, split_types: List[str]):
        self.dataset_name = dataset_name
        self.split_types = split_types
        print("Loading artist lookup...")

        self.index = EntityIndex()
        self.track_artists = defaultdict(set)
        for row in load_catalog(dataset_name, split_types):
            pop = popularity_of(row)
            for artist in as_list(row.get("artist_name")):
                self.index.add(artist, row["track_id"], pop)
                self.track_artists[row["track_id"]].add(normalize_name(artist))
        self.index.finalize()
        print(f"Artist lookup: {len(self.index)} artists")
