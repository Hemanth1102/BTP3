"""
Single source of truth for "what is the recommender allowed to see at turn t?".

Conversation layout in TalkPlayData (see music-crs-evaluator/make_ground_truth.py):

    turn t:  user  ->  music (the ground-truth track)  ->  assistant (response
                                                            that talks about
                                                            that track)

So when predicting the track for turn t, the recommender may use

    * every user + assistant message from turns  < t
    * the music tracks of turns                   < t   (the "anchors")
    * the user message of turn                    == t

and must NOT see

    * the assistant message of turn t  (it describes the answer)
    * the music of turn t              (it IS the answer)
    * anything from turns > t

Both the dataset builder and the evaluator should use this module so that the
training distribution and the test-time distribution are identical.
"""

from typing import Any, Dict, List, NamedTuple


class TurnContext(NamedTuple):
    query: str                           # "user: ...\nassistant: ...\nuser: ..."
    history: List[Dict[str, Any]]        # messages visible at turn t (planner input)
    ground_truth: str                    # track id of turn t
    anchor_track_ids: List[str]          # track ids of turns < t, in order
    turn_number: int
    latest_user_message: str = ""        # the user message of turn t only

    @property
    def planner_query(self) -> str:
        """
        What is passed to the planner as "Current user query". The planner
        prompt already contains the full visible conversation as history, so
        repeating it here would double the prompt (and the LLM latency).
        """
        return self.latest_user_message


def get_turn_context(
    conversations: List[Dict[str, Any]],
    target_turn: int,
) -> TurnContext:

    history: List[Dict[str, Any]] = []
    anchors: List[str] = []
    ground_truth = None
    user_seen_for_target = False
    latest_user = ""

    for message in conversations:

        turn = message["turn_number"]
        role = message["role"]

        if turn > target_turn:
            continue

        if role == "music":

            if turn < target_turn:
                anchors.append(message["content"])
                history.append(message)

            else:
                if ground_truth is not None:
                    raise ValueError(
                        f"Turn {target_turn} has more than one music message."
                    )
                ground_truth = message["content"]

            continue

        if turn < target_turn:
            history.append(message)

        elif role == "user":
            history.append(message)
            user_seen_for_target = True
            latest_user = message["content"]

        # role == "assistant" and turn == target_turn -> deliberately dropped.

    if not user_seen_for_target:
        raise ValueError(f"No user message found for turn {target_turn}.")

    if ground_truth is None:
        raise ValueError(f"No ground-truth track found for turn {target_turn}.")

    # The query text only contains natural-language turns. Track ids are
    # opaque strings and carry no information for text retrievers/planner.
    query = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in history
        if m["role"] in ("user", "assistant")
    )

    return TurnContext(
        query=query,
        history=history,
        ground_truth=ground_truth,
        anchor_track_ids=anchors,
        turn_number=target_turn,
        latest_user_message=latest_user,
    )