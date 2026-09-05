from .retrievers.lexical.bm25 import BM25Retriever
from .retrievers.personalized.cf_bpr import CFBPRRetriever
from .retrievers.dense.bge import BGERetriever
from .retrievers.dense.base_dense import DenseEmbeddingRetriever
from .retrievers.dense.qwen import Qwen3EmbeddingEncoder
from .retrievers.dense.clap import CLAPTextAudioRetriever
from .retrievers.item_to_item.cf import CFItemToItemRetriever
from .retrievers.item_to_item.clap import CLAPItemToItemRetriever
from .retrievers.item_to_item.metadata import MetadataItemToItemRetriever
from .retrievers.lexical.artist_lookup import ArtistLookupRetriever
from .retrievers.lexical.album_lookup import AlbumLookupRetriever


ITEM_DATASET = (
    "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
)

TRACK_EMBEDDING_DATASET = (
    "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
)

USER_EMBEDDING_DATASET = (
    "talkpl-ai/TalkPlayData-Challenge-User-Embeddings"
)


def build_bm25():
    return BM25Retriever(
        dataset_name=ITEM_DATASET,
        split_types=["all_tracks"],
        corpus_types=[
            "track_name",
            "artist_name",
            "album_name",
            "release_date",
        ],
        cache_dir="./cache",
    )


def build_cf_bpr():
    return CFBPRRetriever(
        user_dataset_name=USER_EMBEDDING_DATASET,
        track_dataset_name=TRACK_EMBEDDING_DATASET,
        user_split="train",
        track_split="all_tracks",
        device="cpu",
    )


def build_bge(device="cpu"):
    return BGERetriever(
        dataset_name=ITEM_DATASET,
        split_types=["all_tracks"],
        corpus_types=[
            "track_name",
            "artist_name",
            "album_name",
            "release_date",
        ],
        cache_dir="./cache",
        device=device,
        batch_size=16,
        max_length=512,
    )

def build_qwen_retrievers(device="cpu"):
    encoder = Qwen3EmbeddingEncoder(
        device=device
    )

    metadata = DenseEmbeddingRetriever(
        name="qwen_metadata",
        dataset_name=TRACK_EMBEDDING_DATASET,
        embedding_column="metadata-qwen3_embedding_0.6b",
        split="all_tracks",
        query_encoder=encoder,
        embedding_dim=1024,
    )

    lyrics = DenseEmbeddingRetriever(
        name="qwen_lyrics",
        dataset_name=TRACK_EMBEDDING_DATASET,
        embedding_column="lyrics-qwen3_embedding_0.6b",
        split="all_tracks",
        query_encoder=encoder,
        embedding_dim=1024,
    )

    attributes = DenseEmbeddingRetriever(
        name="qwen_attributes",
        dataset_name=TRACK_EMBEDDING_DATASET,
        embedding_column="attributes-qwen3_embedding_0.6b",
        split="all_tracks",
        query_encoder=encoder,
        embedding_dim=1024,
    )

    return [
        metadata,
        lyrics,
        attributes,
    ]

def build_clap(device="cpu"):
    return CLAPTextAudioRetriever(
        dataset_name=TRACK_EMBEDDING_DATASET,
        split="all_tracks",
        embedding_column="audio-laion_clap",
        device=device,
    )

def build_item_cf():
    return CFItemToItemRetriever(
        dataset_name=TRACK_EMBEDDING_DATASET,
        split="all_tracks",
        device="cpu",
    )

def build_item_clap():
    return CLAPItemToItemRetriever(
        dataset_name=TRACK_EMBEDDING_DATASET,
        split="all_tracks",
        device="cpu",
    )

def build_item_metadata():
    return MetadataItemToItemRetriever(
        dataset_name=TRACK_EMBEDDING_DATASET,
        split="all_tracks",
        device="cpu",
    )

def build_artist_lookup():
    return ArtistLookupRetriever(
        dataset_name=ITEM_DATASET,
        split_types=["all_tracks"],
    )


def build_album_lookup():
    return AlbumLookupRetriever(
        dataset_name=ITEM_DATASET,
        split_types=["all_tracks"],
    )