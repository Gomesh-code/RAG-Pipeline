"""Steps 3 & 4: Turn chunks into vectors and retrieve the most similar ones."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfRetriever:
    """Keyword-based retrieval. Fast, offline, no downloads needed."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        # Each row of this matrix is one chunk's vector (L2-normalized by default)
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in chunks)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q_vec = self.vectorizer.transform([query])
        # Vectors are normalized, so dot product == cosine similarity
        scores = (self.matrix @ q_vec.T).toarray().ravel()
        return _top_k(self.chunks, scores, k)


class EmbeddingRetriever:
    """Neural retrieval using sentence-transformers (understands synonyms)."""

    def __init__(self, chunks: list[dict], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.matrix = self.model.encode(
            [c["text"] for c in chunks], normalize_embeddings=True
        )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q_vec
        return _top_k(self.chunks, scores, k)


def _top_k(chunks, scores, k):
    best = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in best if scores[i] > 0]
