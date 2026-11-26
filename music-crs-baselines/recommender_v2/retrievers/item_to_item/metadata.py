from .embedding import ItemEmbeddingRetriever


class MetadataItemToItemRetriever(ItemEmbeddingRetriever):
    """Item-to-item retrieval using precomputed Qwen3 metadata embeddings."""

    name = "item_metadata"
    EMBEDDING_COLUMN = "metadata-qwen3_embedding_0.6b"
    EMBEDDING_DIM = 1024
