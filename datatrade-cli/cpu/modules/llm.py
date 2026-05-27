"""
modules/llm.py — llama.cpp streaming wrapper (llama-cpp-python).

Loads a GGUF-quantised model in-process via llama.cpp. No external server.
10-20 tok/s on CPU. The chat template is read from the GGUF metadata, so
swapping models (Qwen / Phi / Llama) needs no prompt-format changes here.

Model resolution (see config.py):
  1. LLM_GGUF_PATH points to an existing .gguf → load directly (offline).
  2. else download LLM_GGUF_FILE from LLM_GGUF_REPO on HuggingFace Hub.
"""

import os
from typing import Generator
from config import (
    LLM_GGUF_PATH, LLM_GGUF_REPO, LLM_GGUF_FILE,
    LLM_N_CTX, LLM_N_THREADS,
    LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_CONTEXT_CHUNKS,
)


SYSTEM_PROMPT = """You are a multilingual legal assistant specialising in data protection law.
Answer only based on the provided context. If the context does not contain enough information,
say so clearly. Always cite the section_id of the relevant law.
Respond in the same language as the user's question."""


class LLM:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "Run: pip install llama-cpp-python"
            )

        common = dict(n_ctx=LLM_N_CTX, n_threads=LLM_N_THREADS, verbose=False)

        if LLM_GGUF_PATH and os.path.exists(LLM_GGUF_PATH):
            self._model = Llama(model_path=LLM_GGUF_PATH, **common)
        else:
            self._model = Llama.from_pretrained(
                repo_id=LLM_GGUF_REPO,
                filename=LLM_GGUF_FILE,
                **common,
            )

    # ── Public API ──────────────────────────────────────────────────────────────

    def stream(
        self,
        question: str,
        context_chunks: list[str],
        system: str = SYSTEM_PROMPT,
    ) -> Generator[str, None, None]:
        self._load()

        context_block = "\n\n---\n\n".join(
            f"[{i+1}] {c}" for i, c in enumerate(context_chunks[:LLM_CONTEXT_CHUNKS])
        )
        user_msg = f"Context:\n{context_block}\n\nQuestion: {question}"

        stream = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content

    def complete(self, question: str, context_chunks: list[str]) -> str:
        return "".join(self.stream(question, context_chunks))
