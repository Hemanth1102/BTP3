"""Generate replies for saved diagnostic rankings and report local proxy metrics."""

import argparse
import re
from dataclasses import dataclass
from itertools import islice, zip_longest
from pathlib import Path
from statistics import mean
from typing import Optional, Tuple

import torch  # Initialize the existing recommender's dependencies before NumPy on Windows.

from recommender_v2.response.generator import DEFAULT_MODEL, GeneratedResponse, ResponseGenerator, mentions_title
from recommender_v2.response.predictions import generate_for_prediction, prediction_id, read_predictions, timing_summary
from recommender_v2.splits import DEV_SPLIT, DIAGNOSTIC, TURNS
from recommender_v2.training.build_dataset import DATASET_NAME
from recommender_v2.training.metadata import load_track_metadata
from recommender_v2.training.turns import session_query_id


CATALOGUE_SIZE = 47_071
_WORD = re.compile(r"\w+(?:['’]\w+)*")


def distinct_2(replies):
    unique = set()
    total = 0
    for reply in replies:
        words = _WORD.findall(reply.casefold())
        bigrams = list(zip(words, words[1:]))
        unique.update(bigrams)
        total += len(bigrams)
    return len(unique) / total if total else 0.0


@dataclass
class EvaluatedReply:
    prediction: dict
    generated: GeneratedResponse
    judge_scores: Optional[Tuple[int, int]] = None


def evaluate_sessions(sessions, predictions, generator, judge=False):
    rankings = {prediction_id(row): row for row in predictions}
    evaluated = []
    for session in sessions:
        for turn in TURNS:
            key = session_query_id(session["session_id"], turn)
            prediction = rankings.get(key)
            if prediction is None:
                continue
            conversation = session["conversations"]
            result = generate_for_prediction(generator, prediction, conversation)
            scores = None
            if judge and result.text:
                scores = generator.judge(key, conversation, turn, prediction["predicted_track_ids"], result.text)
            evaluated.append(EvaluatedReply(prediction, result, scores))
            print(f"[{len(evaluated)}] {key} {result.seconds:.2f} s", flush=True)
    if not evaluated:
        raise ValueError("The prediction file has no rankings for the selected diagnostic sessions.")
    return evaluated


def report_metrics(evaluated, generator, judge=False):
    replies = [row.generated for row in evaluated]
    rankings = [row.prediction["predicted_track_ids"] for row in evaluated]
    median_sec, max_sec = timing_summary(replies)
    total = len(replies)
    metrics = {
        "Distinct-2": distinct_2(reply.text for reply in replies),
        "Catalogue diversity": len({track_id for tracks in rankings for track_id in tracks}) / CATALOGUE_SIZE,
        "Top-1 mention rate": sum(bool(tracks) and mentions_title(reply.text, generator.title(tracks[0]))
                                 for reply, tracks in zip(replies, rankings)) / total,
        "Template fallback rate": sum(reply.used_template for reply in replies) / total,
        "Generation time, median (s)": median_sec,
        "Generation time, maximum (s)": max_sec,
    }
    if judge:
        valid = [row.judge_scores for row in evaluated if row.judge_scores is not None]
        for i, label in enumerate(("Personalisation", "Explanation")):
            metrics[f"Local Qwen judge proxy / {label}"] = (mean(row[i] for row in valid) - 1) / 4 if valid else None
        metrics["Local Qwen judge proxy / Valid scores"] = f"{len(valid)} / {total}"
    return metrics


def _quote(text):
    return "\n".join(f"> {line}" for line in text.splitlines())


def write_report(directory, tag, window, evaluated, generator, sessions, judge):
    metrics = report_metrics(evaluated, generator, judge)
    lines = [f"# Local response evaluation · {tag}", "",
             f"Dev sessions {window.start}–{window.stop - 1}, {len(evaluated)} of {len(window) * len(TURNS)} ranked queries. "
             f"Response model `{generator.model}`.", "",
             "| Metric | Value |", "| --- | ---: |"]
    for name, value in metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else (value or "Unavailable")
        lines.append(f"| {name} | {formatted} |")
    if judge:
        lines.extend(["", "The local Qwen judge scores each dimension from 1 to 5 and reports (mean − 1) / 4. "
                      "Generation time excludes the judge call."])

    contexts = {str(session["session_id"]): session["conversations"] for session in sessions}
    lines.extend(["", "## Example replies"])
    by_session = {}
    for row in evaluated:
        by_session.setdefault(str(row.prediction["session_id"]), []).append(row)
    examples = (row for group in zip_longest(*by_session.values()) for row in group if row is not None)
    for row in islice(examples, 5):
        prediction = row.prediction
        conversation = contexts[str(prediction["session_id"])]
        excerpt, _ = generator.conversation_excerpt(conversation, int(prediction["turn_number"]))
        tracks = generator.recommended_tracks(prediction["predicted_track_ids"])
        lines.extend(["", f"### Query `{prediction_id(prediction)}`", "", "**Conversation excerpt**", "", _quote(excerpt),
                      "", "**Top five**", "", tracks, "", "**Generated reply**", "", _quote(row.generated.text)])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"responses_{tag}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7 + len(metrics)]))
    print(f"Wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Rankings as JSONL or a JSON array.")
    parser.add_argument("--num-sessions", type=int, default=len(DIAGNOSTIC))
    parser.add_argument("--tag", default="diagnostic")
    parser.add_argument("--response-model", default=DEFAULT_MODEL)
    parser.add_argument("--num-demos", type=int, default=1)
    parser.add_argument("--judge", action="store_true", help="Score replies with the optional local Qwen proxy.")
    args = parser.parse_args(argv)
    if not 1 <= args.num_sessions <= len(DIAGNOSTIC):
        parser.error(f"--num-sessions must be between 1 and {len(DIAGNOSTIC)}")
    if args.num_demos < 0 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("use a nonnegative --num-demos and an alphanumeric --tag (hyphens and underscores are allowed)")

    from datasets import load_dataset

    predictions = read_predictions(args.predictions)
    window = DIAGNOSTIC[:args.num_sessions]
    sessions = load_dataset(DATASET_NAME, split=DEV_SPLIT).select(window)
    generator = ResponseGenerator(load_track_metadata(), model=args.response_model, num_demos=args.num_demos)
    evaluated = evaluate_sessions(sessions, predictions, generator, judge=args.judge)
    write_report(Path("results"), args.tag, window, evaluated, generator, sessions, args.judge)


if __name__ == "__main__":
    main()
