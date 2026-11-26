"""Format the same visible conversation and catalogue text for training and inference."""

from dataclasses import dataclass

from ..splits import DEV_SPLIT, DEV_TRAIN, TRAIN, TRAIN_SPLIT, TURNS
from ..training.turns import get_turn_context, session_query_id


def _join(values, separator=", "):
    if isinstance(values, str):
        return values
    return separator.join(str(value) for value in (values or ()) if value)


def _track_fields(track):
    return (track.get("track_name") or "", _join(track.get("artist_names")),
            _join(track.get("album_names")), str(track.get("year") or ""))


def document_text(track):
    name, artist, album, year = _track_fields(track)
    return f"{name} \u2013 {artist} | album: {album} | year: {year} | tags: {_join(track.get('genres'))}"


@dataclass(frozen=True)
class QueryText:
    context: str
    request: str

    def __str__(self):
        return self.context + self.request


def query_text(message, history, anchors, turn, metadata):
    listens = []
    for track_id in reversed((anchors or ())[-3:]):
        track = metadata.get(track_id)
        if track:
            name, artist, album, year = _track_fields(track)
            listens.append(f"{name} \u2013 {artist} ({album}, {year})")
    words = [word for item in (history or ())
             if item["role"] == "user" and item["turn_number"] < turn
             for word in item["content"].split()]
    recent = f"Recent listens: {' | '.join(listens)}\n" if listens and turn > 1 else ""
    return QueryText(recent + f"Session so far: {' '.join(words[-25:])}\n", f"Looking for: {message}")


def query_tokens(tokenizer, text, instruction, max_length):
    """Shorten the context first. A request alone can exceed the nominal budget."""
    def encode(value):
        return tokenizer.encode(value, add_special_tokens=False)

    raw = instruction + str(text)
    complete = tokenizer.encode(raw, add_special_tokens=True)
    if len(complete) <= max_length:
        return complete

    plain = encode(raw)
    special = len(complete) - len(plain)
    # Fast tokenizers can add CLS/SEP through their backend rather than their Python class.
    offset = next((i for i in range(special + 1) if complete[i:i + len(plain)] == plain), None)
    if offset is None:
        raise ValueError("Unexpected special-token layout for the dense encoder.")

    head = encode(instruction)
    tail = encode("\n" + text.request)
    room = max(0, max_length - len(head) - len(tail) - special)
    context = encode(text.context.rstrip("\n"))[:room]
    return complete[:offset] + head + context + tail + complete[offset + len(plain):]


@dataclass(frozen=True)
class TurnQuery:
    session_id: str
    turn_number: int
    text: QueryText
    anchors: tuple
    gold: str = None

    @property
    def query_id(self):
        return session_query_id(self.session_id, self.turn_number)


def session_queries(sessions, metadata, *, blind=False):
    for session in sessions:
        conversation = session["conversations"]
        if blind and conversation[-1]["role"] != "user":
            raise ValueError("A blind conversation must end at the current user message.")
        turns = (conversation[-1]["turn_number"],) if blind else TURNS
        for turn in turns:
            ctx = get_turn_context(conversation, turn, require_ground_truth=not blind)
            yield TurnQuery(str(session["session_id"]), turn,
                            query_text(ctx.latest_user_message, ctx.history, ctx.anchor_track_ids,
                                       turn, metadata), tuple(ctx.anchor_track_ids), ctx.ground_truth)


def training_queries(dataset_name, metadata):
    from datasets import load_dataset

    train = load_dataset(dataset_name, split=TRAIN_SPLIT)
    dev = load_dataset(dataset_name, split=DEV_SPLIT)
    yield from session_queries(train.select(TRAIN), metadata)
    for window in DEV_TRAIN:
        yield from session_queries(dev.select(window), metadata)
