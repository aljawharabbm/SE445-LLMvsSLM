from __future__ import annotations

import json
import time
from dataclasses import dataclass

import psutil
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ug9_benchmark.schema import extract_json_object


@dataclass
class RunRecord:
    label: str
    model_id: str
    raw_text: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    peak_rss_mb: float
    parsed: dict | None
    cost_usd: float


class SLMRunner:
    """Edge-style path: small seq2seq model running locally (CPU/GPU)."""

    def __init__(self, model_id: str = "google/flan-t5-small") -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()

    def run(self, prompt: str, max_new_tokens: int = 160) -> RunRecord:
        proc = psutil.Process()
        rss_before = proc.memory_info().rss

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_tokens = int(inputs["input_ids"].shape[1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
            )
        latency_s = time.perf_counter() - t0

        gen_ids = out[0]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        output_tokens = int(gen_ids.shape[0])

        rss_after = proc.memory_info().rss
        peak_rss_mb = max(rss_before, rss_after) / (1024 * 1024)

        parsed = extract_json_object(raw_text)

        return RunRecord(
            label="SLM (local)",
            model_id=self.model_id,
            raw_text=raw_text,
            latency_s=latency_s,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            peak_rss_mb=peak_rss_mb,
            parsed=parsed,
            cost_usd=0.0,
        )


def load_examples(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
