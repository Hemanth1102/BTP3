"""Artifact identity, fold provenance and atomic writes for dense retrieval."""

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ..retrievers.lexical.transition_artist import N_FOLDS, session_fold


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
FULL_MODEL = -1
FOLD_ENSEMBLE = -2
TOPK = 100
INFERENCE_SPLITS = ("diagnostic", "holdout", "blind")


@dataclass(frozen=True)
class Settings:
    model_name: str = DEFAULT_MODEL
    family: str = "qwen"
    revision: str = "main"
    lora_rank: int = 16
    learning_rate: float = 2e-4
    temperature: float = 0.05
    batch_size: int = 64
    micro_batch_size: int = 8
    epochs: int = 1
    query_length: int = 256
    document_length: int = 96
    seed: int = 42
    fp16: bool = True

    def __post_init__(self):
        if self.family not in ("qwen", "bge"):
            raise ValueError("Select the qwen or bge encoder family.")
        if min(self.lora_rank, self.batch_size, self.micro_batch_size, self.epochs,
               self.query_length, self.document_length, self.learning_rate, self.temperature) <= 0:
            raise ValueError("Training and token limits must be positive.")

    @property
    def query_instruction(self):
        if self.family == "bge":
            return "Represent this sentence for searching relevant passages: "
        return ("Instruct: Given a conversation about music, retrieve a track that satisfies "
                "the user's latest request.\nQuery:")


@contextmanager
def pending_output(destination):
    """Publish a file or new directory only after its contents have been written."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".pending-") as work:
        path = Path(work) / destination.name
        yield path
        os.replace(path, destination)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with pending_output(path) as pending:
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def catalogue_fingerprint(catalogue):
    digest = hashlib.sha256()
    for track_id, text in catalogue.items():
        digest.update(json.dumps((track_id, text), ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def create_manifest(root, settings, queries, catalogue):
    digest = hashlib.sha256()
    sessions = {}
    seen = set()
    for query in queries:
        if query.query_id in seen:
            raise ValueError(f"Duplicate training query {query.query_id}.")
        seen.add(query.query_id)
        sessions[query.session_id] = session_fold(query.session_id)
        digest.update(json.dumps((query.query_id, str(query.text), query.gold,
                                  query.anchors), ensure_ascii=False).encode("utf-8"))
    manifest = {"version": 1, "settings": asdict(settings), "fold_count": N_FOLDS,
                "catalogue_sha256": catalogue_fingerprint(catalogue),
                "training_sessions": sessions, "data_sha256": digest.hexdigest()}
    manifest["run_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    path = Path(root) / "manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError(f"Different training data or settings already exist in {root}. Choose another output directory.")
    if not path.exists():
        write_json(path, manifest)
    return manifest


def load_manifest(root):
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Dense retriever artifacts are missing {path}. Use the Kaggle output or omit dense_ft.")
    manifest = read_json(path)
    fields = {key: value for key, value in manifest.items() if key != "run_id"}
    run_id = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:20]
    if (manifest["run_id"] != run_id or manifest["version"] != 1 or manifest["fold_count"] != N_FOLDS or
            any(fold != session_fold(sid) for sid, fold in manifest["training_sessions"].items())):
        raise ValueError(f"Invalid training fold provenance in {path}.")
    return manifest


def model_dir(root, manifest, fold):
    name = "full" if fold == FULL_MODEL else f"fold_{fold}"
    return Path(root) / "models" / manifest["run_id"] / name


def model_record(root, manifest, fold):
    folder = model_dir(root, manifest, fold)
    record = read_json(folder / "training.json")
    if record["run_id"] != manifest["run_id"] or record["excluded_fold"] != fold:
        raise ValueError(f"Unexpected training provenance in {folder}.")
    return record


def parse_folds(values):
    allowed = {str(fold): fold for fold in range(N_FOLDS)}
    allowed["full"] = FULL_MODEL
    try:
        return list(dict.fromkeys(allowed[value] for value in values))
    except KeyError as error:
        raise ValueError(f"Unknown fold {error.args[0]!r}. Choose 0–{N_FOLDS - 1} or full.") from error


def ranking_schema(manifest, source):
    import pyarrow as pa

    return pa.schema([
        ("query_id", pa.string()), ("session_id", pa.string()), ("turn_number", pa.int16()),
        ("track_id", pa.string()), ("rank", pa.int16()), ("score", pa.float32()), ("fold", pa.int8()),
    ], metadata={b"dense_ft_run": manifest["run_id"].encode(), b"dense_ft_source": str(source).encode()})


def open_rankings(path, manifest, source):
    import pyarrow.parquet as pq

    if not Path(path).is_file():
        raise FileNotFoundError(f"Dense retrieval results are missing {path}.")
    file = pq.ParquetFile(path)
    if not file.schema_arrow.equals(ranking_schema(manifest, source), check_metadata=True):
        file.close()
        raise ValueError(f"Unexpected model provenance or ranking schema in {path}.")
    return file
