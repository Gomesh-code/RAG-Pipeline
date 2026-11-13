"""Step 2: Split documents into overlapping word-based chunks."""


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 15) -> list[str]:
    """Split text into chunks of `chunk_size` words, sharing `overlap` words."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(docs: list[dict], chunk_size: int = 80, overlap: int = 15) -> list[dict]:
    """Chunk every document, keeping track of where each chunk came from."""
    chunks = []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append({"source": doc["source"], "chunk_id": i, "text": piece})
    return chunks
