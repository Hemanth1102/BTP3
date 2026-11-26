from .embedding import ItemEmbeddingRetriever


class CLAPItemToItemRetriever(ItemEmbeddingRetriever):
    """Item-to-item retrieval using precomputed LAION-CLAP audio embeddings."""

    name = "item_clap"
    EMBEDDING_COLUMN = "audio-laion_clap"
    EMBEDDING_DIM = 512
