"""
modules/llm.py — Qwen2.5-7B-Instruct streaming wrapper.

Uses 4-bit quantisation via bitsandbytes (unsloth bnb model).
Falls back to full precision float16 if bitsandbytes unavailable.
Runs on GPU if available; CPU is slow but functional for testing.

Usage:
    from modules.llm import LLM
    llm = LLM()

    context = ["มาตรา 28 ผู้ควบคุมข้อมูล...", "Article 7 The controller..."]
    for token in llm.stream("ต้องแต่งตั้ง DPO ไหม", context_chunks=context):
        print(token, end="", flush=True)
    print()
"""

import threading
from queue import Queue, Empty
from typing import Generator, Optional
from config import (
    DEVICE, LLM_MODEL_ID,
    LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LLM_CONTEXT_CHUNKS,
)


SYSTEM_PROMPT = """You are a multilingual legal assistant specialising in data protection law.
Answer only based on the provided context. If the context does not contain enough information,
say so clearly. Always cite the section_id of the relevant law.
Respond in the same language as the user's question."""


class LLM:
    def __init__(self, model_id: str = LLM_MODEL_ID):
        self.model_id  = model_id
        self._model    = None
        self._tokeniser = None

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        self._tokeniser = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        # Try 4-bit quant; fall back to float16
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
        except (ImportError, ValueError):
            # bitsandbytes not available → float16 on GPU, float32 on CPU
            dtype = torch.float16 if DEVICE != "cpu" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto" if DEVICE != "cpu" else None,
                trust_remote_code=True,
            )
            if DEVICE == "cpu":
                self._model = self._model.to("cpu")

    # ── Public API ──────────────────────────────────────────────────────────────

    def stream(
        self,
        question: str,
        context_chunks: list[str],
        system: str = SYSTEM_PROMPT,
    ) -> Generator[str, None, None]:
        """
        Streams tokens one-by-one using a background thread + Queue.
        Compatible with both CPU and GPU.

        Yields individual string tokens as they are generated.
        """
        self._load()
        from transformers import TextIteratorStreamer

        prompt   = self._build_prompt(question, context_chunks, system)
        inputs   = self._tokeniser(prompt, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(
            self._tokeniser,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=(LLM_TEMPERATURE > 0),
            pad_token_id=self._tokeniser.eos_token_id,
        )

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    def complete(self, question: str, context_chunks: list[str]) -> str:
        """Blocking version — collects full output then returns."""
        return "".join(self.stream(question, context_chunks))

    # ── Prompt builder ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(question: str, chunks: list[str], system: str) -> str:
        context_block = "\n\n---\n\n".join(
            f"[{i+1}] {c}" for i, c in enumerate(chunks[:LLM_CONTEXT_CHUNKS])
        )
        return (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\nContext:\n{context_block}\n\nQuestion: {question}<|end|>\n"
            f"<|assistant|>\n"
        )
