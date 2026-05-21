"""
Local GGUF inference (llama.cpp) for tiny instruct models.

We use SmolLM2-135M-Instruct in two quantizations:
  - IQ3_XS (~84 MB on disk) — aggressive compression / "edge" tier
  - Q4_K_M (~101 MB) — higher-fidelity quantization (~100 MB tier)

A coherent *instruct* generative model below ~50 MB is not realistically available in
standard GGUF catalogs for this JSON task; IQ3_XS is the smallest practical default.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ug9_benchmark.schema import extract_json_object


@dataclass
class GGUFRunRecord:
    label: str
    model_path: str
    raw_text: str
    latency_s: float
    parsed: dict | None


def default_gguf_dir(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[1]
    return root / "models" / "gguf"


def resolve_model_path(project_root: Path, filename: str) -> Path:
    return default_gguf_dir(project_root) / filename


def format_file_mb(path: Path) -> float | None:
    if not path.is_file():
        return None
    return path.stat().st_size / (1024 * 1024)


def load_gguf(
    model_path: Path,
    *,
    n_ctx: int | None = None,
    n_threads: int | None = None,
) -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install llama-cpp-python: pip install llama-cpp-python"
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(f"GGUF not found: {model_path}")

    ctx = n_ctx if n_ctx is not None else int(os.environ.get("UG9_GGUF_CTX", "1024"))
    threads = n_threads if n_threads is not None else int(
        os.environ.get("UG9_GGUF_THREADS", str(min(8, os.cpu_count() or 4)))
    )

    return Llama(
        model_path=str(model_path),
        n_ctx=ctx,
        n_threads=threads,
        verbose=False,
    )


def gguf_chat_infer(
    llm: Any,
    *,
    label: str,
    model_path: Path,
    user_prompt: str,
    system_prompt: str = "Return only valid JSON for the IoT command. No prose.",
    max_tokens: int = 120,
) -> GGUFRunRecord:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        repeat_penalty=1.15,
        top_p=0.9,
    )
    latency_s = time.perf_counter() - t0

    raw_text = (out["choices"][0]["message"].get("content") or "").strip()
    parsed = extract_json_object(raw_text)

    return GGUFRunRecord(
        label=label,
        model_path=str(model_path),
        raw_text=raw_text,
        latency_s=latency_s,
        parsed=parsed,
    )


# Defaults documented for demos / download script
DEFAULT_GGUF_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
DEFAULT_SLM_GGUF = "SmolLM2-135M-Instruct-IQ3_XS.gguf"
DEFAULT_LLM_GGUF = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
