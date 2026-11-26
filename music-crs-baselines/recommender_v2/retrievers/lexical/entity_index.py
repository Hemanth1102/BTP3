"""
Name -> tracks index shared by the artist, album and title lookup retrievers.

Matching, in order of trust:
  1. exact     the planner's structured name equals a catalog name after
               normalisation (case, accents, punctuation, a leading "the")
  2. fuzzy     a close spelling of a structured name ("Beyonse" -> "Beyonce"),
               only for names of 4+ characters
  3. text      a catalog name appears as whole words in the user's message
               (fallback when the planner extracted nothing)

Tracks of a name are stored most-popular first, so cutting a big artist at
`topk` keeps their best-known tracks instead of an arbitrary catalog order.
"""

import difflib
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

EXACT, FUZZY, TEXT = "exact", "fuzzy", "text"

from ...text_norm import normalize_name  # noqa: E402


def as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    value = str(value).strip()
    return [value] if value else []


class EntityIndex:

    def __init__(self, min_text_len: int = 2, max_ngram: int = 8,
                 fuzzy_cutoff: float = 0.85):
        self.min_text_len = min_text_len
        self.max_ngram = max_ngram
        self.fuzzy_cutoff = fuzzy_cutoff
        self._tracks: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        self.by_name: Dict[str, List[str]] = {}
        self._by_first: Dict[str, List[str]] = defaultdict(list)

    # ---- building ----
    def add(self, name, track_id: str, popularity: Optional[float]) -> None:
        key = normalize_name(name)
        if key:
            self._tracks[key].append((popularity if popularity is not None else -1.0, track_id))

    def finalize(self) -> None:
        for key, pairs in self._tracks.items():
            seen, ordered = set(), []
            for _, tid in sorted(pairs, key=lambda p: -p[0]):     # stable: ties keep catalog order
                if tid not in seen:
                    seen.add(tid)
                    ordered.append(tid)
            self.by_name[key] = ordered
            self._by_first[key[0]].append(key)
        self._tracks = None

    def __len__(self):
        return len(self.by_name)

    # ---- matching ----
    def match_names(self, names: Iterable[str]) -> List[Tuple[str, str]]:
        """Structured names -> [(catalog key, EXACT|FUZZY)], in the given order."""
        out, seen = [], set()
        for raw in names or []:
            key = normalize_name(raw)
            if not key:
                continue
            if key in self.by_name:
                hit = (key, EXACT)
            else:
                hit = self._fuzzy(key)
            if hit and hit[0] not in seen:
                seen.add(hit[0])
                out.append(hit)
        return out

    def _fuzzy(self, key: str) -> Optional[Tuple[str, str]]:
        if len(key) < 4:
            return None
        slack = max(2, int(len(key) * 0.2))
        pool = [k for k in self._by_first.get(key[0], ()) if abs(len(k) - len(key)) <= slack]
        best = difflib.get_close_matches(key, pool, n=1, cutoff=self.fuzzy_cutoff)
        return (best[0], FUZZY) if best else None

    def match_text(self, *texts: str) -> List[Tuple[str, str]]:
        """Catalog names appearing as whole words in free text (longest first)."""
        found: List[str] = []
        for text in texts:
            tokens = normalize_name(text).split()
            for n in range(min(self.max_ngram, len(tokens)), 0, -1):
                for i in range(len(tokens) - n + 1):
                    gram = " ".join(tokens[i:i + n])
                    if len(gram) >= self.min_text_len and gram in self.by_name and gram not in found:
                        # skip a name that is only part of a longer name already found
                        if not any(f" {gram} " in f" {g} " for g in found):
                            found.append(gram)
        return [(k, TEXT) for k in found]


def interleave(lists: List[List[str]], topk: int, skip: Set[str]) -> List[Tuple[str, int]]:
    """Round-robin over several matched names so each gets slots; returns (track, list index)."""
    out, seen = [], set(skip)
    pos = [0] * len(lists)
    while len(out) < topk:
        progressed = False
        for j, lst in enumerate(lists):
            while pos[j] < len(lst) and lst[pos[j]] in seen:
                pos[j] += 1
            if pos[j] < len(lst):
                tid = lst[pos[j]]
                pos[j] += 1
                seen.add(tid)
                out.append((tid, j))
                progressed = True
                if len(out) >= topk:
                    break
        if not progressed:
            break
    return out
