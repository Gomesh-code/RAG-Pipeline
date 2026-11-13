"""Step 5: Produce an answer from the retrieved chunks."""
import os
import re

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know based on the documents."

Context:
{context}

Question: {question}
Answer:"""


class ExtractiveGenerator:
    """No LLM: returns the sentence from the retrieved chunks that best overlaps the question."""

    def generate(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return "I don't know based on the documents."
        q_words = set(re.findall(r"\w+", question.lower()))
        best, best_score = "", -1
        for chunk in chunks:
            for sent in re.split(r"(?<=[.!?])\s+", chunk["text"]):
                score = len(q_words & set(re.findall(r"\w+", sent.lower())))
                if score > best_score:
                    best, best_score = sent, score
        return best


class ClaudeGenerator:
    """Uses the Anthropic API. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-5-5"):
        import anthropic  # lazy import

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model

    def generate(self, question: str, chunks: list[dict]) -> str:
        context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        )
        return msg.content[0].text


def get_generator(use_llm: bool):
    if use_llm:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError("Set ANTHROPIC_API_KEY to use --llm")
        return ClaudeGenerator()
    return ExtractiveGenerator()
