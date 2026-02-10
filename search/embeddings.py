import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model(model_name=DEFAULT_MODEL):
    """Load a sentence-transformers model."""
    return SentenceTransformer(model_name)


def embed_text(model, text):
    """Embed a single text string. Returns a float32 numpy array."""
    return model.encode(text, convert_to_numpy=True).astype(np.float32)


def embed_batch(model, texts, batch_size=256, show_progress=False):
    """Embed a list of texts in batches. Returns a (n, dims) float32 numpy array."""
    return model.encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True
    ).astype(np.float32)
