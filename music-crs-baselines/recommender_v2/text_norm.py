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
