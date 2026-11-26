"""Name normalisation shared by the lookup retrievers and the feature builder."""

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s&']")
_SPACES = re.compile(r"\s+")


def normalize_name(text) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("’", "'")
    text = _PUNCT.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    if text.startswith("the "):
        text = text[4:]
    return text


def normalized_names(values):
    if not isinstance(values, (list, tuple, set, frozenset)):
        values = (values,)
    return {normalize_name(value) for value in values} - {""}


def album_keys(album_names, artist_names):
    """Identify each album by its normalized title and a credited artist."""
    albums = normalized_names(album_names)
    artists = normalized_names(artist_names)
    return {(album, artist) for album in albums for artist in artists}


# "Song (Remastered 2011)", "Song [Live]", "Song - Radio Edit" -> "Song"
_SUFFIX = re.compile(r"\s*(\(.*?\)|\[.*?\]|\s-\s.*)$")


def base_title(title) -> str:
    title = str(title or "")
    prev = None
    while prev != title:
        prev, title = title, _SUFFIX.sub("", title).strip()
    return title


def base_title_norm(title) -> str:
    return normalize_name(base_title(title))
