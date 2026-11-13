"""Step 1: Load documents from a folder of .txt files."""
from pathlib import Path


def load_documents(folder: str) -> list[dict]:
    """Return a list of {"source": filename, "text": file contents}."""
    docs = []
    for path in sorted(Path(folder).glob("*.txt")):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    if not docs:
        raise FileNotFoundError(f"No .txt files found in '{folder}'")
    return docs
