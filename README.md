# Beginner RAG Pipeline

A minimal Retrieval-Augmented Generation project in plain Python.
It runs **offline with no API key**, and you can upgrade each piece later.

## How it works

```
documents -> chunks -> vectors -> retrieve top-k -> generate answer
 loader.py  chunker.py retriever.py   retriever.py   generator.py
```

| Step | File | Default | Optional upgrade |
|------|------|---------|------------------|
| Load | `rag/loader.py` | `.txt` files in `data/` | PDFs, web pages |
| Chunk | `rag/chunker.py` | 80 words, 15 overlap | sentence/semantic chunking |
| Embed + retrieve | `rag/retriever.py` | TF-IDF | `--retriever embedding` |
| Generate | `rag/generator.py` | best-matching sentence | `--llm` (Claude) |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py -q "What is cosine similarity?"   # one question
python main.py                                    # interactive mode
python evaluate.py                                # retrieval test
```

Upgrades:

```bash
pip install sentence-transformers
python main.py --retriever embedding -q "How do I reduce hallucinations?"

pip install anthropic
export ANTHROPIC_API_KEY=your_key
python main.py --llm -q "Why use chunk overlap?"
```

## Use your own documents

Drop `.txt` files into `data/` (or pass `--data my_folder`) and re-run.

## Exercises

1. Change `chunk_size` and `overlap` in `rag/pipeline.py`; watch the evaluate hit rate change.
2. Add 5 more test questions to `evaluate.py`.
3. Add a `PDFLoader` using `pypdf`.
4. Add mean reciprocal rank (MRR) to `evaluate.py`.
5. Compare TF-IDF vs embeddings on questions that use synonyms.
