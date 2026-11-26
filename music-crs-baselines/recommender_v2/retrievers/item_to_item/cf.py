from .embedding import ItemEmbeddingRetriever


class CFItemToItemRetriever(ItemEmbeddingRetriever):
    """Item-to-item retrieval using precomputed CF-BPR track embeddings."""

    name = "item_cf_bpr"
    EMBEDDING_COLUMN = "cf-bpr"
    EMBEDDING_DIM = 128
