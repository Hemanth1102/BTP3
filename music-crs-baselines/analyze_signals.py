"""
How much of the gold track can be predicted WITHOUT the conversation text?

Uses only the challenge dataset (train split = history, test split = the
sessions we are scored on) and the track metadata. Run:

    python analyze_signals.py                     # test split vs train history
    python analyze_signals.py --split train --max-sessions 2000   # sanity check

For every (session, turn) it asks whether the gold track is:
  * among the globally most frequent gold tracks of the train split
    (the starter kit's "popularity" lower bound),
  * among the tracks this same user received in OTHER (train) sessions,
  * by an artist that user received in train sessions,
  * by an artist already played earlier in the same session (anchors),
and prints the nDCG@20 each of those signals alone would give.
Nothing here uses the planner or the retrievers.
"""

import argparse
import math
from collections import Counter, defaultdict

from datasets import load_dataset

DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"


def gold_tracks(session):
    """Ground-truth track per turn (the 'music' message of each turn)."""
    return {m["turn_number"]: m["content"]
            for m in session["conversations"] if m["role"] == "music"}


def ndcg_from_rank(rank, k=20):
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--max-sessions", type=int, default=0)
    args = ap.parse_args()

    from recommender_v2.training.metadata import load_track_metadata
    meta = load_track_metadata()

    def artists(t):
        return set(meta.get(t, {}).get("artist_names", []))

    print("Loading dataset...")
    ds = load_dataset(DATASET)
    train = ds["train"]
    target = ds[args.split]

    # ---- history from the train split ----
    pop = Counter()
    user_tracks = defaultdict(Counter)       # user -> track counts
    user_session_ids = defaultdict(set)
    for s in train:
        for t in gold_tracks(s).values():
            pop[t] += 1
            user_tracks[s["user_id"]][t] += 1
        user_session_ids[s["user_id"]].add(s["session_id"])

    print(f"train: {len(train)} sessions, {len(pop)} distinct gold tracks, "
          f"{len(user_tracks)} users")
    top_pop = [t for t, _ in pop.most_common()]
    pop_rank = {t: i + 1 for i, t in enumerate(top_pop)}

    n = 0
    stat = Counter()
    ndcg = Counter()
    users_known = 0
    hist_sizes = []
    n_sessions = 0

    for s in target:
        if args.max_sessions and n_sessions >= args.max_sessions:
            break
        n_sessions += 1

        user = s["user_id"]
        golds = gold_tracks(s)

        # For the train sanity check, remove this session from its own user's history.
        hist = Counter(user_tracks.get(user, {}))
        if args.split == "train":
            for t in golds.values():
                hist[t] -= 1
            hist = +hist
        if hist:
            users_known += 1
        hist_sizes.append(len(hist))

        hist_ranked = [t for t, _ in hist.most_common()]
        hist_rank = {t: i + 1 for i, t in enumerate(hist_ranked)}
        hist_artists = Counter(a for t in hist for a in artists(t))

        played = []
        for turn in sorted(golds):
            g = golds[turn]
            n += 1

            r = pop_rank.get(g)
            ndcg["popularity_global"] += ndcg_from_rank(r)
            for N in (20, 100, 1000):
                stat[f"gold in global top-{N}"] += int(r is not None and r <= N)

            stat["gold in user's train history"] += int(g in hist)
            ndcg["user_history_by_freq"] += ndcg_from_rank(hist_rank.get(g))

            ga = artists(g)
            stat["gold artist in user's train history"] += int(bool(ga & set(hist_artists)))
            stat["gold artist = artist of an earlier track in this session"] += int(
                any(ga & artists(p) for p in played))
            stat["gold already played earlier in this session"] += int(g in played)

            played.append(g)

    print(f"\nSplit: {args.split}   sessions: {n_sessions}   queries: {n}")
    print(f"Users with any train history: {users_known / n_sessions:.1%}   "
          f"mean distinct history tracks per user: {sum(hist_sizes) / len(hist_sizes):.1f}")

    print("\nShare of queries where the gold track ...")
    for k, v in stat.items():
        print(f"  {k:<58}{v / n:7.1%}")

    print("\nnDCG@20 of each signal used alone (no conversation, no retrieval):")
    for k, v in ndcg.items():
        print(f"  {k:<30}{v / n:8.4f}")


if __name__ == "__main__":
    main()