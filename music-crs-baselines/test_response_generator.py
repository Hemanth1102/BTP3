"""Offline checks for response prompts, retries and proxy metrics."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import run_blind_inference
from evaluate_responses import EvaluatedReply, distinct_2, report_metrics, write_report
from recommender_v2.response.generator import GeneratedResponse, ResponseGenerator, mentions_title, query_seed
from recommender_v2.response.predictions import read_predictions
from recommender_v2.splits import DIAGNOSTIC, TRAIN_SPLIT
from recommender_v2.training.build_dataset import DATASET_NAME
from test_training_pipeline import check


def message(turn, role, content):
    return {"turn_number": turn, "role": role, "content": content}


def response(text):
    return {"message": {"content": text}}


TRACKS = [f"track{i}" for i in range(1, 8)]
METADATA = {
    track_id: {"track_name": f"Song {i}", "artist_names": [f"Artist {i}"], "album_names": [f"Album {i}"],
               "year": 2000 + i, "genres": ["folk", "acoustic", "soft", "fourth-tag"]}
    for i, track_id in enumerate(TRACKS, 1)
}
METADATA.update({f"played{i}": {"track_name": f"Earlier {i}", "artist_names": [f"Singer {i}"]}
                 for i in range(1, 6)})
METADATA["FORBIDDEN_MUSIC"] = {"track_name": "FORBIDDEN_MUSIC_TITLE"}
METADATA["track1"]["track_name"] = "['Song 1']"
METADATA["literal"] = {"track_name": "[Live]"}
CONVERSATION = [item for turn in range(1, 6)
                for item in [message(turn, "user", f"Request {turn}"), message(turn, "music", f"played{turn}"),
                             message(turn, "assistant", f"Reply {turn}")]] + [
    message(6, "user", "Something acoustic for a quiet evening"),
    message(6, "music", "FORBIDDEN_MUSIC"), message(6, "assistant", "FORBIDDEN_ASSISTANT"),
    message(7, "user", "FORBIDDEN_LATER_USER"), message(7, "music", "FORBIDDEN_LATER_MUSIC"),
]
TRAIN = [{"conversations": [message(1, "user", "Example request"),
                            message(1, "assistant", "Example response")]}]


def check_prompt_and_generation():
    chat = Mock(return_value=response("Try song 1 by Artist 1. Its acoustic folk suits a quiet evening."))
    with patch("datasets.load_dataset", return_value=TRAIN) as load:
        generator = ResponseGenerator(METADATA, chat=chat)
        result = generator.generate("diag:6", CONVERSATION, 6, TRACKS)
        generator.generate("diag:6", CONVERSATION, 6, TRACKS)
        load.assert_called_once_with(DATASET_NAME, split=TRAIN_SPLIT)

    first, second = (call.kwargs for call in chat.call_args_list)
    system, user = [m["content"] for m in first["messages"]]
    check(first == second and first["options"]["seed"] == query_seed("diag:6"),
          "same query ID produces the same conversation, demonstration and seed")
    check("Example request\nAssistant: Example response" in system, "one deterministic train demonstration is reused")
    check(system.index("Example response") < system.index("Name the exact title and artist from rank 1") and
          "Do not use any Markdown" in system, "rank and style instructions take priority over the demonstration")
    check("FORBIDDEN" not in user and user.count("\nUser:") + user.count("\nAssistant:") == 6
          and "Reply 3" in user and "Request 3" not in user,
          "prompt only includes the last six visible user and assistant messages")
    played, recommended = user.split("Previously played:\n")[1].split("\n\nRecommended tracks:\n")
    check("Earlier 2" not in played and "Earlier 3" in played and "Earlier 5" in played,
          "previously played lists the last three visible anchors")
    check(len(recommended.splitlines()) == 5 and
          "1. Song 1 \u2013 Artist 1 (Album 1, 2001) [tags: folk, acoustic, soft]" in recommended and
          not any(value in recommended for value in ("Song 6", "Song 7", "fourth-tag", "Earlier 5")),
          "only the five ranked titles, with the first three tags, are offered as recommendations")
    check(first["model"] == "qwen3:8b" and first["think"] is False and first["keep_alive"] == "30m" and
          "format" not in first and first["options"]["temperature"] == .7 and first["options"]["num_predict"] == 160
          and not result.used_template, "valid case-insensitive mentions use one plain-text Qwen request")
    check(generator.title("track1") == "Song 1" and generator.title("literal") == "[Live]",
          "serialized catalogue title lists are unwrapped while literal bracketed titles are preserved")

    chat = Mock(side_effect=[response("An unrelated suggestion."), response("")])
    generated = ResponseGenerator(METADATA, num_demos=0, chat=chat).generate("diag:6", CONVERSATION, 6, TRACKS)
    check(generated.used_template and generated.text ==
          "You might enjoy Song 1 by Artist 1, which fits what you asked for. I also added Song 2 by Artist 2.",
          "two invalid replies use the top-two template")
    check([call.kwargs["options"]["temperature"] for call in chat.call_args_list] == [.7, .3],
          "an invalid reply is retried once at temperature 0.3")

    chat = Mock(side_effect=[response("Song 1 " + "word " * 120), response("Song 1 sounds right here.")])
    generator = ResponseGenerator(METADATA, num_demos=0, chat=chat)
    check(not generator.generate("diag:6", CONVERSATION, 6, TRACKS[:1]).used_template and chat.call_count == 2,
          "an overlong reply is retried and a valid second reply is used")
    check(generator.generate("diag:6", CONVERSATION, 6, []).text == "" and chat.call_count == 2,
          "an empty ranked list makes no Ollama call")


def check_mentions_and_retries():
    check(not mentions_title("it will fit", "I", "Will") and not mentions_title("Homecoming", "Home"),
          "substring fragments do not count as whole titles")
    check(mentions_title("Try ‘Straße—Nacht’.", "STRASSE Nacht"),
          "title matching casefolds and treats Unicode punctuation as word boundaries")
    check(mentions_title("Try ‘I’ by Beyoncé.", "I", "Beyoncé") and
          not mentions_title("Try ‘I’ by another artist.", "I", "Beyoncé") and
          not mentions_title("Try ‘I’ by Beyoncéish.", "I", "Beyoncé"),
          "short titles require a whole artist-name match")

    chat = Mock(side_effect=[response("Try Song 2."), response("Try Song 3.")])
    generator = ResponseGenerator(METADATA, num_demos=0, chat=chat)
    reply = generator.generate("diag:6", CONVERSATION, 6, TRACKS)
    check(reply.text == "Try Song 3." and not reply.used_template and chat.call_count == 2,
          "only the retry can accept a top-five title below rank one")

    chat = Mock(side_effect=[response("Song 1 sounds like Earlier 1."), response("Try Song 2 and Earlier 5.")])
    generator = ResponseGenerator(METADATA, num_demos=0, chat=chat)
    reply = generator.generate("diag:6", CONVERSATION, 6, TRACKS)
    check(reply.used_template and "Earlier" not in reply.text and chat.call_count == 2,
          "an unranked anchor title causes retry and fallback, including anchors outside the prompt's last three")

    chat = Mock(return_value=response("Try Song 1 and Earlier 1."))
    generator = ResponseGenerator(METADATA, num_demos=0, chat=chat)
    reply = generator.generate("diag:6", CONVERSATION, 6, [*TRACKS[:4], "played1"])
    check(not reply.used_template and chat.call_count == 1 and
          generator.mentions_anchor(reply.text, ["played1"]),
          "a top-five anchor is allowed during generation and counted by the anchor metric")


def check_inference(directory):
    blind = {"session_id": "blind", "user_id": "user", "conversations": CONVERSATION[:-4]}
    old = {"session_id": "blind", "user_id": "user", "turn_number": 6,
           "predicted_track_ids": TRACKS.copy(), "predicted_response": ""}
    recommender = SimpleNamespace(recommend=Mock(return_value=TRACKS))
    out = run_blind_inference.predict_rows([blind], recommender, log=lambda *_: None)
    check(out == [old], "response generation off preserves the blind prediction output")

    generator = Mock()
    generator.generate.return_value = GeneratedResponse("Song 1 suits the request.", 1.0, False)
    stats = []
    out = run_blind_inference.predict_rows([blind], recommender, log=lambda *_: None,
                                           response_generator=generator, response_stats=stats)
    check(out == [{**old, "predicted_response": "Song 1 suits the request."}] and len(stats) == 1,
          "opt-in generation only fills the response and records time")

    source = Path(directory) / "predictions.jsonl"
    destination = Path(directory) / "replies.jsonl"
    prior = {**old, "predicted_response": "Old text", "extra": {"provenance": "untouched"}}
    source.write_text(json.dumps(prior) + "\n", encoding="utf-8")
    with patch("datasets.load_dataset", return_value={"test": [blind]}), \
         patch("recommender_v2.training.metadata.load_track_metadata", return_value=METADATA), \
         patch("recommender_v2.response.generator.ResponseGenerator", return_value=generator) as factory, \
         patch("recommender_v2.training.build_dataset.build_retrievers", side_effect=AssertionError("ranking ran")):
        run_blind_inference.main(["--responses-only", "--predictions", str(source), "--output", str(destination),
                                  "--planner-model", "planner-model"])
    check(read_predictions(destination) == [{**prior, "predicted_response": "Song 1 suits the request."}]
          and read_predictions(source) == [prior], "responses-only reads JSONL and changes no fields besides the reply")
    check(factory.call_args.kwargs["model"] == "planner-model", "response model defaults to the planner model")


def check_metrics(directory):
    check(distinct_2(["A b a.", "a b!"]) == 2 / 3 and distinct_2(["one", "two"]) == 0,
          "Distinct-2 counts corpus bigrams without crossing reply boundaries")
    generator = ResponseGenerator(METADATA, num_demos=0)
    prediction = {"session_id": "diag", "turn_number": 6, "predicted_track_ids": TRACKS[:2]}
    sample = [EvaluatedReply(prediction, GeneratedResponse("Song 1 fits.", 1.0, False), (5, 4)),
              EvaluatedReply(prediction, GeneratedResponse("Song 2 fits.", 3.0, True), (1, 2), mentions_anchor=True)]
    metrics = report_metrics(sample, generator, judge=True)
    check(metrics["Catalogue diversity"] == 2 / 47_071 and metrics["Top-1 mention rate"] == .5 and
          metrics["Any-top-5 mention rate"] == 1 and metrics["Anchor mention rate"] == .5 and
          metrics["Template fallback rate"] == .5 and metrics["Generation time, median (s)"] == 2 and
          metrics["Generation time, maximum (s)"] == 3 and metrics["Local Qwen judge proxy / Personalisation"] == .5,
          "proxy metrics use the catalogue denominator, mention and fallback rates, timing and normalized judge scores")
    write_report(Path(directory), "test", DIAGNOSTIC[:1], sample, generator,
                 [{"session_id": "diag", "conversations": CONVERSATION}], True)
    report = (Path(directory) / "responses_test.md").read_text(encoding="utf-8")
    check("Local Qwen judge proxy" in report and "Song 1 fits." in report and "FORBIDDEN" not in report,
          "Markdown report includes visible examples and labels the judge as a local proxy")


if __name__ == "__main__":
    check_prompt_and_generation()
    check_mentions_and_retries()
    with TemporaryDirectory() as directory:
        check_inference(directory)
        check_metrics(directory)
    if check.failed:
        raise SystemExit(1)
    print("All response generation checks passed.")
