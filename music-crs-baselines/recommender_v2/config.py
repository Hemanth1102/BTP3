from .retrievers.lexical.bm25 import BM25Retriever
from .retrievers.personalized.cf_bpr import CFBPRRetriever
from .retrievers.dense.bge import BGERetriever
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


DEFAULT_TEXT_FIELDS = ["track_name", "artist_name", "album_name", "release_date"]


def build_bm25(text_fields=None):
    return BM25Retriever(
        dataset_name=ITEM_DATASET,
        split_types=["all_tracks"],
        corpus_types=list(text_fields or DEFAULT_TEXT_FIELDS),
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


def build_bge(device="cpu", text_fields=None):
    return BGERetriever(
        dataset_name=ITEM_DATASET,
        split_types=["all_tracks"],
        corpus_types=list(text_fields or DEFAULT_TEXT_FIELDS),
        cache_dir="./cache",
        device=device,
        batch_size=16,
        max_length=512,
    )


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