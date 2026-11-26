from collections import defaultdict
from typing import List

from .entity_index import EntityIndex, as_list, normalize_name
from .lookup_base import EntityLookupRetriever, load_catalog, popularity_of


class AlbumLookupRetriever(EntityLookupRetriever):
    """
    Tracks of the albums the user asked for (QueryState.album_names). Album
    names repeat across artists ("Greatest Hits"), so tracks by an artist the
    user also named come first.
    """

    name = "album_lookup"
    field = "album_names"
    artist_preference = True

    def __init__(self, dataset_name: str, split_types: List[str]):
        self.dataset_name = dataset_name
        self.split_types = split_types
        print("Loading album lookup...")

        # free-text fallback only for longer album names: short ones ("Home",
        # "Love") match ordinary words in a request
        self.index = EntityIndex(min_text_len=6)
        self.track_artists = defaultdict(set)
        for row in load_catalog(dataset_name, split_types):
            pop = popularity_of(row)
            for album in as_list(row.get("album_name")):
                self.index.add(album, row["track_id"], pop)
            for artist in as_list(row.get("artist_name")):
                self.track_artists[row["track_id"]].add(normalize_name(artist))
        self.index.finalize()
        print(f"Album lookup: {len(self.index)} albums")
