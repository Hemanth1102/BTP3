"""Export catalogue embeddings and top-100 retrieval results for each model."""

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .artifacts import (FOLD_ENSEMBLE, FULL_MODEL, INFERENCE_SPLITS, N_FOLDS, TOPK, Settings,
                        catalogue_fingerprint, create_manifest, load_manifest, model_dir, model_record,
                        open_rankings, parse_folds,
                        pending_output, ranking_schema, write_json)
from .text import document_text, query_tokens, session_queries, training_queries
from ..splits import DEV_SPLIT, DIAGNOSTIC, FINAL_HOLDOUT
from ..training.build_dataset import DATASET_NAME
from ..training.metadata import load_track_metadata


class Encoder:
    def __init__(self, model, tokenizer, settings, device):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.device = torch.device(device)
        self.fp16 = self.device.type == "cuda" and settings.fp16

    def tokenize_query(self, text):
        return query_tokens(self.tokenizer, text, self.settings.query_instruction, self.settings.query_length)

    def tokenize_documents(self, texts):
        return self.tokenizer(list(texts), truncation=True, max_length=self.settings.document_length)["input_ids"]

    def embed(self, token_ids):
        if any(len(ids) > self.model.config.max_position_embeddings for ids in token_ids):
            raise ValueError("A current request exceeds the model context window and cannot be kept intact.")
        inputs = self.tokenizer.pad({"input_ids": list(token_ids)}, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
            outputs = self.model(**inputs).last_hidden_state
        if self.settings.family == "qwen":
            mask = inputs["attention_mask"]
            positions = torch.arange(mask.shape[1], device=self.device).expand_as(mask)
            last = positions.masked_fill(mask == 0, -1).amax(dim=1)
            pooled = outputs[torch.arange(len(last), device=self.device), last]
        else:
            pooled = outputs[:, 0]
        return F.normalize(pooled.float(), dim=-1)


def load_encoder(settings, device, *, checkpoint=None, training=False):
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(device)
    is_lora = settings.family == "qwen"
    from_base = checkpoint is None or is_lora
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint if checkpoint else settings.model_name,
        **({} if checkpoint else {"revision": settings.revision}),
        padding_side="left" if is_lora else "right",
    )
    base = AutoModel.from_pretrained(
        settings.model_name if from_base else checkpoint,
        **({"revision": settings.revision} if from_base else {}),
        torch_dtype=torch.float16 if is_lora and device.type == "cuda" and settings.fp16 else torch.float32,
    )
    if is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        base.config.use_cache = False
        if checkpoint is not None:
            base = PeftModel.from_pretrained(base, checkpoint)
        elif training:
            base = get_peft_model(base, LoraConfig(
                r=settings.lora_rank, lora_alpha=2 * settings.lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="FEATURE_EXTRACTION",
            ))
    model = base.to(device)
    model.train(training)
    return Encoder(model, tokenizer, settings, device)


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


@dataclass
class LiveIndex:
    encoder: Encoder
    track_ids: list
    matrix: torch.Tensor

    def __post_init__(self):
        self.positions = {track_id: offset for offset, track_id in enumerate(self.track_ids)}

    @torch.inference_mode()
    def search(self, texts, played, topk=TOPK):
        query_vectors = self.encoder.embed([self.encoder.tokenize_query(text) for text in texts])
        scores = query_vectors @ self.matrix.T
        for i, history in enumerate(played):
            cols = [self.positions[track_id] for track_id in history if track_id in self.positions]
            scores[i, cols] = -torch.inf
        values, indices = scores.topk(min(topk, len(self.track_ids)), dim=1)
        return [[(self.track_ids[index], score) for index, score in zip(row_indices, row_scores)
                 if math.isfinite(score)] for row_indices, row_scores in zip(indices.tolist(), values.tolist())]


@torch.inference_mode()
def load_index(root, manifest, fold, device, *, catalogue=None, batch_size=64):
    checkpoint = model_dir(root, manifest, fold)
    model_record(root, manifest, fold)
    settings = Settings(**manifest["settings"])
    encoder = load_encoder(settings, device, checkpoint=checkpoint)
    path = checkpoint / "catalogue.pt"
    if not path.is_file():
        if catalogue is None:
            raise FileNotFoundError(f"Dense catalogue index is missing {path}. Run dense_ft.encode first.")
        track_ids = list(catalogue)
        pieces = [encoder.embed(encoder.tokenize_documents(batch)).cpu()
                  for batch in chunks(list(catalogue.values()), batch_size)]
        index = {"run_id": manifest["run_id"], "fold": fold,
                 "track_ids": track_ids, "matrix": torch.cat(pieces)}
        with pending_output(path) as pending:
            torch.save(index, pending)
    else:
        index = torch.load(path, map_location="cpu", weights_only=True)
    if (index["run_id"] != manifest["run_id"] or index["fold"] != fold or
            len(index["track_ids"]) != index["matrix"].shape[0]):
        raise ValueError(f"Invalid catalogue provenance in {path}.")
    return LiveIndex(encoder, index["track_ids"], index["matrix"].to(device))


def rrf(rankings):
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, (track_id, _) in enumerate(ranking, 1):
            fused[track_id] += 1 / (60 + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:TOPK]


def _write_rankings(path, manifest, source, batches):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = ranking_schema(manifest, source)
    with pending_output(path) as pending, pq.ParquetWriter(pending, schema) as writer:
        for queries, predictions, fold in batches:
            rows = [{"query_id": q.query_id, "session_id": q.session_id, "turn_number": q.turn_number,
                     "track_id": track_id, "rank": rank, "score": score, "fold": fold}
                    for q, results in zip(queries, predictions)
                    for rank, (track_id, score) in enumerate(results, 1)]
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _predict_batches(index, queries, batch_size, fold):
    for batch in chunks(queries, batch_size):
        predictions = index.search([q.text for q in batch], [q.anchors for q in batch])
        yield batch, predictions, fold


def _exists(path, manifest, source):
    if not Path(path).is_file():
        return False
    open_rankings(path, manifest, source).close()
    return True


def _merge_oof(root, manifest):
    import pyarrow.parquet as pq

    shards = [root / f"fold_{fold}_top100.parquet" for fold in range(N_FOLDS)]
    destination = root / "oof_top100.parquet"
    if all(shard.is_file() for shard in shards) and not _exists(destination, manifest, "oof"):
        with pending_output(destination) as pending, pq.ParquetWriter(pending, ranking_schema(manifest, "oof")) as writer:
            for fold, shard in enumerate(shards):
                with open_rankings(shard, manifest, fold) as source:
                    for i in range(source.num_row_groups):
                        writer.write_table(source.read_row_group(i))


def encode_oof(root, manifest, queries, catalogue, folds, device, batch_size):
    create_manifest(root, Settings(**manifest["settings"]), queries, catalogue)
    for fold in folds:
        if fold == FULL_MODEL:
            raise ValueError("Only numbered folds can produce out-of-fold rows.")
        path = root / f"fold_{fold}_top100.parquet"
        if _exists(path, manifest, fold):
            print(f"Fold {fold} output exists. Skipping encoding.", flush=True)
            continue
        test = [q for q in queries if manifest["training_sessions"][q.session_id] == fold]
        index = load_index(root, manifest, fold, device, catalogue=catalogue, batch_size=batch_size)
        _write_rankings(path, manifest, fold, _predict_batches(index, test, batch_size, fold))
        print(f"Fold {fold} wrote {len(test)} out-of-fold queries.", flush=True)
        del index
    _merge_oof(root, manifest)


def encode_inference(root, manifest, split, queries, catalogue, strategy, device, batch_size):
    for query in queries:
        if query.session_id in manifest["training_sessions"]:
            raise ValueError(f"Inference split {split} includes training session {query.session_id}.")
    source = FULL_MODEL if strategy == "full" else FOLD_ENSEMBLE
    path = root / f"full_top100_{split}.parquet"
    if _exists(path, manifest, source):
        return path, source
    if source == FULL_MODEL:
        index = load_index(root, manifest, source, device, catalogue=catalogue, batch_size=batch_size)
        _write_rankings(path, manifest, source, _predict_batches(index, queries, batch_size, source))
    else:
        by_query = defaultdict(list)
        for fold in range(N_FOLDS):
            index = load_index(root, manifest, fold, device, catalogue=catalogue, batch_size=batch_size)
            for batch, predictions, _ in _predict_batches(index, queries, batch_size, fold):
                for query, results in zip(batch, predictions):
                    by_query[query.query_id].append(results)
            del index
        _write_rankings(path, manifest, source, (
            (batch, [rrf(by_query[q.query_id]) for q in batch], source) for batch in chunks(queries, batch_size)
        ))
    return path, source


def diagnostic_metrics(path, manifest, source, queries):
    truth = {q.query_id: q.gold for q in queries}
    ranks = {}
    with open_rankings(path, manifest, source) as rankings:
        for batch in rankings.iter_batches(columns=["query_id", "track_id", "rank"]):
            for row in batch.to_pylist():
                if truth[row["query_id"]] == row["track_id"]:
                    ranks[row["query_id"]] = row["rank"]
    n = len(truth)
    return {"queries": n, "recall@100": len(ranks) / n,
            "recall@20": sum(rank <= 20 for rank in ranks.values()) / n,
            "ndcg@20": sum(1 / math.log2(rank + 1) for rank in ranks.values() if rank <= 20) / n}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/dense_ft"))
    parser.add_argument("--split", choices=("oof", *INFERENCE_SPLITS), required=True)
    parser.add_argument("--folds", nargs="+")
    parser.add_argument("--inference", choices=("full", "rrf"), default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--blind-dataset", default="talkpl-ai/TalkPlayData-Challenge-Blind-B")
    parser.add_argument("--final-holdout-i-mean-it", action="store_true")
    args = parser.parse_args(argv)
    if args.split == "holdout" and not args.final_holdout_i_mean_it:
        parser.error("encoding holdout requires --final-holdout-i-mean-it at final evaluation")
    if args.batch_size <= 0 or (args.folds and args.split != "oof"):
        parser.error("use a positive batch size and select folds only for oof")

    from datasets import load_dataset

    root = args.output_dir
    manifest = load_manifest(root)
    metadata = load_track_metadata()
    catalogue = {track_id: document_text(track) for track_id, track in metadata.items()}
    if catalogue_fingerprint(catalogue) != manifest["catalogue_sha256"]:
        raise ValueError("The catalogue does not match the training manifest.")
    if args.split == "oof":
        folds = parse_folds(args.folds) if args.folds else range(N_FOLDS)
        queries = list(training_queries(DATASET_NAME, metadata))
        encode_oof(root, manifest, queries, catalogue, folds, args.device, args.batch_size)
        return
    if args.split == "blind":
        sessions = load_dataset(args.blind_dataset)[DEV_SPLIT]
    else:
        indices = DIAGNOSTIC if args.split == "diagnostic" else FINAL_HOLDOUT
        sessions = load_dataset(DATASET_NAME, split=DEV_SPLIT).select(indices)
    queries = list(session_queries(sessions, metadata, blind=(args.split == "blind")))
    path, source = encode_inference(root, manifest, args.split, queries, catalogue, args.inference,
                                    args.device, args.batch_size)
    if args.split == "diagnostic":
        metrics = diagnostic_metrics(path, manifest, source, queries)
        write_json(root / "diagnostic_metrics.json", metrics)
        print(metrics)


if __name__ == "__main__":
    main()
