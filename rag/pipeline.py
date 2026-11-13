"""Glue code: wires the five steps together."""
from .loader import load_documents
from .chunker import chunk_documents
from .retriever import TfidfRetriever, EmbeddingRetriever
from .generator import get_generator


class RAGPipeline:
    def __init__(self, data_dir="data", retriever="tfidf", use_llm=False,
                 chunk_size=80, overlap=15, k=3):
        docs = load_documents(data_dir)                          # 1. load
        self.chunks = chunk_documents(docs, chunk_size, overlap)  # 2. chunk
        cls = EmbeddingRetriever if retriever == "embedding" else TfidfRetriever
        self.retriever = cls(self.chunks)                         # 3. embed/index
        self.generator = get_generator(use_llm)
        self.k = k

    def ask(self, question: str) -> dict:
        hits = self.retriever.retrieve(question, self.k)          # 4. retrieve
        answer = self.generator.generate(question, hits)          # 5. generate
        return {"answer": answer, "sources": hits}
