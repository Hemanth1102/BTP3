"""Train one independent bi-encoder per held-out fold, then the full model."""

import argparse
import math
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from .artifacts import (DEFAULT_MODEL, FULL_MODEL, N_FOLDS, Settings, create_manifest, load_manifest,
                        model_dir, model_record, parse_folds, pending_output, write_json)
from .encode import chunks, load_encoder
from .text import document_text, training_queries
from ..training.build_dataset import DATASET_NAME
from ..training.metadata import load_track_metadata


def contrastive_loss(queries, tracks, gold_ids, log_q, temperature):
    """In-batch InfoNCE with log-frequency correction and all duplicate positives."""
    logits = queries.float() @ tracks.float().T / temperature - log_q
    positives = gold_ids[:, None] == gold_ids[None, :]
    normalizer = logits.logsumexp(dim=1)
    positive_mass = logits.masked_fill(~positives, -torch.inf).logsumexp(dim=1)
    return (normalizer - positive_mass).mean()


def _detach_batches(encoder, token_ids, micro_batch_size):
    gpu = encoder.device.index if encoder.device.type == "cuda" else None
    if encoder.device.type == "cuda" and gpu is None:
        gpu = torch.cuda.current_device()
    features, states = [], []
    with torch.no_grad():
        for batch in chunks(token_ids, micro_batch_size):
            cpu_state = torch.get_rng_state()
            gpu_state = torch.cuda.get_rng_state(gpu) if gpu is not None else None
            states.append((batch, cpu_state, gpu_state))
            features.append(encoder.embed(batch))
    return torch.cat(features).requires_grad_(), states, gpu


def _replay_gradients(encoder, gradients, states, gpu):
    offset = 0
    for batch, cpu_state, gpu_state in states:
        with torch.random.fork_rng(devices=[gpu] if gpu is not None else []):
            torch.set_rng_state(cpu_state)
            if gpu is not None:
                torch.cuda.set_rng_state(gpu_state, gpu)
            encoder.embed(batch).backward(gradients[offset:offset + len(batch)])
        offset += len(batch)


def batch_backward(encoder, queries, tracks, ids, log_q, settings, scaler):
    if len(ids) <= settings.micro_batch_size:
        loss = contrastive_loss(encoder.embed(queries), encoder.embed(tracks), ids, log_q, settings.temperature)
        scaler.scale(loss).backward()
    else:
        # Keep all logical-batch negatives. Replay each forward with its original dropout state.
        q, q_states, gpu = _detach_batches(encoder, queries, settings.micro_batch_size)
        d, d_states, _ = _detach_batches(encoder, tracks, settings.micro_batch_size)
        loss = contrastive_loss(q, d, ids, log_q, settings.temperature)
        scaler.scale(loss).backward()
        _replay_gradients(encoder, q.grad, q_states, gpu)
        _replay_gradients(encoder, d.grad, d_states, gpu)
    return float(loss.detach())


def fit(encoder, queries, catalogue, settings):
    device = encoder.device
    params = [weight for weight in encoder.model.parameters() if weight.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=settings.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=encoder.fp16)
    counts = Counter(query.gold for query in queries)
    ids = {track_id: i for i, track_id in enumerate(counts)}
    log_probs = {tid: math.log(count / len(queries)) for tid, count in counts.items()}
    track_tokens = dict(zip(catalogue, encoder.tokenize_documents(catalogue.values())))
    query_ids = [encoder.tokenize_query(query.text) for query in queries]

    steps = 0
    for epoch in range(settings.epochs):
        order = list(range(len(queries)))
        random.Random(settings.seed + epoch).shuffle(order)
        for batch in chunks(order, settings.batch_size):
            golds = [queries[i].gold for i in batch]
            id_tensor = torch.tensor([ids[track] for track in golds], device=device)
            log_q = torch.tensor([log_probs[track] for track in golds], device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            value = batch_backward(encoder, [query_ids[i] for i in batch], [track_tokens[track] for track in golds],
                                   id_tensor, log_q, settings, scaler)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            if steps == 1 or steps % 100 == 0:
                print(f"epoch={epoch + 1} step={steps} loss={value:.5f}", flush=True)
    return steps


def train_fold(root, manifest, queries, catalogue, fold, device):
    checkpoint = model_dir(root, manifest, fold)
    if checkpoint.exists():
        record = model_record(root, manifest, fold)
        print(f"Model {fold} is complete. Training took {record['training_seconds']:.1f} s.", flush=True)
        return record
    settings = Settings(**manifest["settings"])
    train = [query for query in queries
             if fold == FULL_MODEL or manifest["training_sessions"][query.session_id] != fold]
    if not train:
        raise ValueError(f"No training pairs for fold {fold}.")
    torch.manual_seed(settings.seed)
    started = time.perf_counter()
    encoder = load_encoder(settings, device, training=True)
    if hasattr(encoder.model, "gradient_checkpointing_enable") and encoder.device.type == "cuda":
        encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    steps = fit(encoder, train, catalogue, settings)
    record = {"run_id": manifest["run_id"], "excluded_fold": fold, "queries": len(train), "steps": steps,
              "training_seconds": time.perf_counter() - started,
              "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
              "settings": asdict(settings)}
    with pending_output(checkpoint) as pending:
        encoder.model.save_pretrained(pending)
        encoder.tokenizer.save_pretrained(pending)
        write_json(pending / "training.json", record)
    print(f"Model {fold} trained on {len(train)} pairs in {record['training_seconds']:.1f} s "
          f"on {record['gpu'] or 'CPU'}.", flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("cache/dense_ft"))
    parser.add_argument("--folds", nargs="+", default=[*(str(i) for i in range(N_FOLDS)), "full"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--family", choices=("qwen", "bge"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--query-length", type=int, default=256)
    parser.add_argument("--document-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args(argv)
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Enable the Kaggle GPU accelerator.")

    from transformers import AutoConfig

    family = args.family or ("bge" if "bge" in args.model.lower() else "qwen")
    manifest_path = args.output_dir / "manifest.json"
    revision = args.revision
    if manifest_path.exists() and revision == "main":
        revision = load_manifest(args.output_dir)["settings"]["revision"]
    model_config = AutoConfig.from_pretrained(args.model, revision=revision)
    revision = getattr(model_config, "_commit_hash", None) or revision
    settings = Settings(model_name=args.model, family=family, revision=revision, lora_rank=args.lora_rank,
                        learning_rate=args.lr if args.lr is not None else (2e-4 if family == "qwen" else 2e-5),
                        temperature=args.temperature, batch_size=args.batch_size, micro_batch_size=args.micro_batch_size,
                        epochs=args.epochs, query_length=args.query_length, document_length=args.document_length,
                        seed=args.seed, fp16=not args.fp32)
    metadata = load_track_metadata()
    queries = list(training_queries(DATASET_NAME, metadata))
    catalogue = {track_id: document_text(track) for track_id, track in metadata.items()}
    missing = {query.gold for query in queries} - catalogue.keys()
    if missing:
        raise ValueError(f"Catalogue is missing {len(missing)} training tracks.")
    manifest = create_manifest(args.output_dir, settings, queries, catalogue)
    for fold in parse_folds(args.folds):
        train_fold(args.output_dir, manifest, queries, catalogue, fold, args.device)
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
