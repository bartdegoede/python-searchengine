import numpy as np

from search.embeddings import get_embedding_model, embed_text, embed_batch


def test_get_embedding_model():
    model = get_embedding_model()
    assert model is not None


def test_embed_text_returns_numpy_array():
    model = get_embedding_model()
    vector = embed_text(model, "hello world")
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert vector.ndim == 1
    assert vector.shape[0] > 0


def test_embed_text_different_texts_differ():
    model = get_embedding_model()
    v1 = embed_text(model, "the cat sat on the mat")
    v2 = embed_text(model, "quantum physics is fascinating")
    # cosine similarity should be low for unrelated texts
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert similarity < 0.8


def test_embed_text_similar_texts_similar():
    model = get_embedding_model()
    v1 = embed_text(model, "the dog chased the cat")
    v2 = embed_text(model, "a canine pursued a feline")
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert similarity > 0.4


def test_embed_batch():
    model = get_embedding_model()
    texts = ["hello world", "foo bar", "search engine"]
    vectors = embed_batch(model, texts, batch_size=2)
    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2
    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0
    assert vectors.dtype == np.float32


def test_embed_batch_consistent_with_single():
    """Batch and single embedding should produce the same vectors."""
    model = get_embedding_model()
    text = "consistency check"
    single = embed_text(model, text)
    batch = embed_batch(model, [text])
    assert batch.shape == (1, single.shape[0])
    np.testing.assert_allclose(single, batch[0], atol=1e-5)
