"""
Does 'which artist follows which' (learned from the train sessions) predict the
gold artist when it is NOT an artist already played in the session?

Offline check, no retrievers / planner / GPU. For every query (turn >= 2) whose
gold artist is new to the session it measures how often the gold artist is
among the top-K artists predicted from transition statistics, and how often the
gold TRACK lies in the first 100 tracks produced by those artists (the size of
one candidate list in our pipeline). A global-popularity baseline shows how much
the transitions add beyond "famous artists".

    python analyze_transitions.py                  # build on train, score on dev ("test" split)
    python analyze_transitions.py --heldout-train  # build on 80% of train, score on the other 20%
"""

import argparse
from collections import Counter, defaultdict

DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"


def gold_sequence(session):
    by_turn = {m["turn_number"]: m["content"]
               for m in session["conversations"] if m["role"] == "music"}
    return [by_turn[t] for t in sorted(by_turn)]


def build_transitions(sequences, track_artists):
    """trans[a][b] = recency-weighted count of artist b appearing j-i turns after artist a (a != b)."""
    trans = defaultdict(Counter)
    pop = Counter()
    for seq in sequences:
        arts = [set(track_artists.get(t, ())) for t in seq]
        for j, bj in enumerate(arts):
            for b in bj:
                pop[b] += 1
            for i in range(j):
                w = 1.0 / (j - i)
                for a in arts[i]:
                    for b in bj:
                        if a != b:
                            trans[a][b] += w
    totals = {a: sum(c.values()) for a, c in trans.items()}
    return trans, totals, pop


def predict_artists(anchor_artist_lists, trans, totals, k):
    """Top-k artists by recency-weighted transition probability from the session's artists."""
    n = len(anchor_artist_lists)
    played = set().union(*anchor_artist_lists) if anchor_artist_lists else set()
    score = Counter()
    for idx, arts in enumerate(anchor_artist_lists):
        age = n - idx                      # most recent anchor has age 1
        for a in arts:
            tot = totals.get(a)
            if not tot:
                continue
            for b, c in trans[a].items():
                if b not in played:
                    score[b] += (1.0 / age) * c / tot
    return [a for a, _ in score.most_common(k)]


def tracks_for(artists, artist_tracks, popularity, limit=100):
    out, seen = [], set()
    for a in artists:
        for t in sorted(artist_tracks.get(a, ()), key=lambda x: -popularity.get(x, 0.0)):
            if t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= limit:
                    return out
    return out


def evaluate(queries, trans, totals, pop, track_artists, artist_tracks, popularity):
    ks = (10, 20, 50)
    top_pop = [a for a, _ in pop.most_common(max(ks))]
    n = 0
    hit = Counter()
    hit_pop = Counter()
    track_hit = 0
    track_hit_pop = 0
    pop_tracks = tracks_for(top_pop[:50], artist_tracks, popularity)
    for anchors, gold in queries:
        a_arts = [set(track_artists.get(t, ())) for t in anchors]
        g_arts = set(track_artists.get(gold, ()))
        if not g_arts or g_arts & set().union(*a_arts):
            continue                       # gold artist already in session: other source covers it
        n += 1
        pred = predict_artists(a_arts, trans, totals, max(ks))
        for k in ks:
            hit[k] += int(bool(g_arts & set(pred[:k])))
            hit_pop[k] += int(bool(g_arts & set(top_pop[:k])))
        cand = tracks_for(pred, artist_tracks, popularity)
        track_hit += int(gold in cand)
        track_hit_pop += int(gold in pop_tracks)
    return n, hit, hit_pop, track_hit, track_hit_pop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-train", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    meta = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_artists, artist_tracks, popularity = {}, defaultdict(list), {}
    for row in meta:
        arts = row["artist_name"]
        arts = [arts] if isinstance(arts, str) else list(arts or [])
        arts = [a.strip().lower() for a in arts if a and a.strip()]
        track_artists[row["track_id"]] = arts
        try:
            popularity[row["track_id"]] = float(row.get("popularity") or 0.0)
        except (TypeError, ValueError):
            popularity[row["track_id"]] = 0.0
        for a in arts:
            artist_tracks[a].append(row["track_id"])

    ds = load_dataset(DATASET)
    train = list(ds["train"])

    if args.heldout_train:
        fit = [s for i, s in enumerate(train) if i % 5 != 0]
        score_sessions = [s for i, s in enumerate(train) if i % 5 == 0]
        label = "held-out 20% of train"
    else:
        fit, score_sessions, label = train, list(ds["test"]), "dev (test split)"

    trans, totals, pop = build_transitions([gold_sequence(s) for s in fit], track_artists)
    print(f"transition table from {len(fit)} sessions: {len(trans)} source artists")

    queries = []
    for s in score_sessions:
        seq = gold_sequence(s)
        for t in range(1, len(seq)):
            queries.append((seq[:t], seq[t]))

    n, hit, hit_pop, th, thp = evaluate(queries, trans, totals, pop, track_artists, artist_tracks, popularity)
    total = len(queries)
    print(f"\nScored on {label}: {total} queries from turn 2 on; "
          f"{n} ({n / total:.1%}) have a gold artist that is NEW to the session.")
    print("\nAmong those new-artist queries, gold artist in the top-K predicted artists:")
    print(f"  {'K':>4}{'transitions':>14}{'global popularity':>20}")
    for k in (10, 20, 50):
        print(f"  {k:>4}{hit[k] / n:>14.1%}{hit_pop[k] / n:>20.1%}")
    print(f"\nGold TRACK within the first 100 tracks of the predicted artists: "
          f"{th / n:.1%}  (popular artists baseline: {thp / n:.1%})")


if __name__ == "__main__":
    main()