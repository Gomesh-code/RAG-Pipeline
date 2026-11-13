import argparse
from rag.pipeline import RAGPipeline


def show(result):
    print(f"\nAnswer: {result['answer']}\n")
    print("Sources:")
    for h in result["sources"]:
        print(f"  - {h['source']} (chunk {h['chunk_id']}, score {h['score']:.3f})")
    print()


def main():
    p = argparse.ArgumentParser(description="Beginner RAG pipeline")
    p.add_argument("-q", "--question", help="Ask one question and exit")
    p.add_argument("--data", default="data", help="Folder with .txt files")
    p.add_argument("--retriever", choices=["tfidf", "embedding"], default="tfidf")
    p.add_argument("--llm", action="store_true", help="Generate answers with Claude")
    p.add_argument("--k", type=int, default=3, help="Chunks to retrieve")
    args = p.parse_args()

    rag = RAGPipeline(args.data, args.retriever, args.llm, k=args.k)
    print(f"Indexed {len(rag.chunks)} chunks.")

    if args.question:
        show(rag.ask(args.question))
        return
    print("Ask questions (type 'quit' to exit).")
    while True:
        q = input("> ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        show(rag.ask(q))


if __name__ == "__main__":
    main()
