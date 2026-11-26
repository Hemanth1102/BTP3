"""Read ranked predictions and attach replies without touching their rankings."""

import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from ..training.turns import session_query_id


def prediction_id(row):
    return session_query_id(row["session_id"], int(row["turn_number"]))


def read_predictions(path):
    content = Path(path).read_text(encoding="utf-8-sig")
    if content.lstrip().startswith("["):
        predictions = json.loads(content)
    else:
        predictions = [json.loads(line) for line in content.splitlines() if line.strip()]
    seen = set()
    for row in predictions:
        key = prediction_id(row)
        if key in seen:
            raise ValueError(f"Duplicate prediction for {key}.")
        seen.add(key)
        if not isinstance(row["predicted_track_ids"], list):
            raise ValueError(f"Expected a ranked track list for {key}.")
    return predictions


def write_predictions(path, predictions, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        pending = Path(temp) / path.name
        with pending.open("w", encoding="utf-8") as out:
            if jsonl:
                for row in predictions:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(predictions, out, ensure_ascii=False)
        os.replace(pending, path)


def generate_for_prediction(generator, prediction, conversations):
    return generator.generate(prediction_id(prediction), conversations, int(prediction["turn_number"]),
                              prediction["predicted_track_ids"])


def timing_summary(replies):
    times = [reply.seconds for reply in replies]
    return (median(times), max(times)) if times else (0.0, 0.0)
