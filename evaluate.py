"""Tiny retrieval evaluation: does the right document appear in the top-k?"""
from rag.pipeline import RAGPipeline

TESTS = [
    ("What are the steps of a RAG pipeline?", "what_is_rag.txt"),
    ("How many words should a chunk have?", "chunking_and_embeddings.txt"),
    ("What does cosine similarity measure?", "chunking_and_embeddings.txt"),
    ("Name some vector databases", "vector_stores.txt"),
    ("What is TF-IDF?", "vector_stores.txt"),
    ("How do I measure retrieval quality?", "evaluation.txt"),
]

if __name__ == "__main__":
    rag = RAGPipeline()
    hits = 0
    for question, expected in TESTS:
        sources = [h["source"] for h in rag.ask(question)["sources"]]
        ok = expected in sources
        hits += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question}  -> {sources[0] if sources else None}")
    print(f"\nHit rate @k: {hits}/{len(TESTS)}")
