#!/usr/bin/env python3
"""Single entry point for UG-9 benchmarks (HF SLM + GGUF comparison)."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TeeTextIO(io.TextIOBase):
    """Mirror stdout to an optional report file."""

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except ValueError:
                pass


def cmd_download_gguf(_args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    from ug9_benchmark.gguf_runner import (
        DEFAULT_GGUF_REPO,
        DEFAULT_LLM_GGUF,
        DEFAULT_SLM_GGUF,
        default_gguf_dir,
    )

    dest = default_gguf_dir(ROOT)
    dest.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("IQ3_XS (~84 MB tier)", DEFAULT_SLM_GGUF),
        ("Q4_K_M (~100 MB tier)", DEFAULT_LLM_GGUF),
    ]
    for desc, fname in pairs:
        print(f"Downloading {desc}: {fname} …")
        path = hf_hub_download(
            repo_id=DEFAULT_GGUF_REPO,
            filename=fname,
            local_dir=str(dest),
        )
        p = Path(path)
        mb = p.stat().st_size / (1024 * 1024)
        print(f"  → {p} ({mb:.1f} MB)\n")

    print("Done. Example: python run.py gguf -o reports/gguf.txt")


def cmd_gguf(args: argparse.Namespace) -> None:
    from ug9_benchmark.gguf_runner import (
        DEFAULT_LLM_GGUF,
        DEFAULT_SLM_GGUF,
        format_file_mb,
        gguf_chat_infer,
        load_gguf,
    )
    from ug9_benchmark.schema import build_prompt, score_prediction

    slm_path = ROOT / "models" / "gguf" / DEFAULT_SLM_GGUF
    llm_path = ROOT / "models" / "gguf" / DEFAULT_LLM_GGUF

    if not slm_path.is_file() or not llm_path.is_file():
        print("GGUF files missing. Run: python run.py download-gguf", file=sys.stderr)
        sys.exit(1)

    out_file: IO[str] | None = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(args.output, "w", encoding="utf-8")

    streams: list[IO[str]] = [sys.stdout]
    if out_file is not None:
        streams.append(out_file)
    tee = TeeTextIO(*streams)
    old_stdout = sys.stdout
    try:
        sys.stdout = tee

        mb_s = format_file_mb(slm_path)
        mb_l = format_file_mb(llm_path)
        print("=== Local GGUF comparison ===\n")
        print(f"IQ3_XS:  {slm_path.name}  ({mb_s:.1f} MB)")
        print(f"Q4_K_M: {llm_path.name} ({mb_l:.1f} MB)\n")

        cmd = args.command
        gold = {
            "device": "office_lights",
            "action": "set_brightness",
            "value": 60,
            "unit": "percent",
        }
        prompt = build_prompt(cmd)

        print(f'Command: "{cmd}"\n')
        print("Loading IQ3_XS…")
        eng_s = load_gguf(slm_path)
        print("Running IQ3_XS inference…")
        rec_s = gguf_chat_infer(eng_s, label="IQ3_XS", model_path=slm_path, user_prompt=prompt)
        del eng_s

        print("Loading Q4_K_M…")
        eng_l = load_gguf(llm_path)
        print("Running Q4_K_M inference…")
        rec_l = gguf_chat_infer(eng_l, label="Q4_K_M", model_path=llm_path, user_prompt=prompt)
        del eng_l

        acc_s, hits_s = score_prediction(rec_s.parsed, gold)
        acc_l, hits_l = score_prediction(rec_l.parsed, gold)

        print("\n--- Results ---\n")
        print(f"{'Metric':<22} {'IQ3_XS':<20} {'Q4_K_M':<24}")
        print("-" * 66)
        print(f"{'Inference time (s)':<22} {rec_s.latency_s:<20.3f} {rec_l.latency_s:<24.3f}")
        print(f"{'Field accuracy':<22} {acc_s:<20.2f} {acc_l:<24.2f}")
        print("\nGold JSON:", json.dumps(gold))
        print("\nIQ3_XS output:\n", rec_s.raw_text or "(empty)")
        print("\nQ4_K_M output:\n", rec_l.raw_text or "(empty)")
        print("\nPer-field (IQ3_XS):", hits_s)
        print("Per-field (Q4_K_M):", hits_l)
    finally:
        sys.stdout = old_stdout
        if out_file is not None:
            out_file.close()

    if args.output is not None:
        print(f"Wrote report to {args.output}", file=sys.stderr)


def _stdout_tee(output_path: Path | None) -> tuple[IO[str] | None, IO[str]]:
    if output_path is None:
        return None, sys.stdout
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(output_path, "w", encoding="utf-8")
    return fh, TeeTextIO(sys.stdout, fh)


DEFAULT_GOLD_SINGLE = {
    "device": "office_lights",
    "action": "set_brightness",
    "value": 60,
    "unit": "percent",
}

DATA_PATH = ROOT / "ug9_benchmark" / "data" / "iot_commands.json"


def cmd_slm_single(args: argparse.Namespace) -> None:
    from ug9_benchmark.schema import build_prompt, score_prediction
    from ug9_benchmark.slm_runner import SLMRunner

    gold_path = args.gold_file
    if gold_path is not None:
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        if not isinstance(gold, dict):
            print("--gold-file must contain a JSON object", file=sys.stderr)
            sys.exit(1)
    else:
        gold = DEFAULT_GOLD_SINGLE

    out_fh, stdout_target = _stdout_tee(args.output)
    old_stdout = sys.stdout
    try:
        sys.stdout = stdout_target
        print("=== HF SLM single command ===\n")
        print(f"Model: {args.slm}\n")
        cmd = args.command.strip()
        prompt = build_prompt(cmd)
        print(f'Command: "{cmd}"\n')

        print("Loading model (first run may download weights)…")
        slm = SLMRunner(model_id=args.slm)
        print("Running inference…")
        s_rec = slm.run(prompt)

        acc_s, hits_s = score_prediction(s_rec.parsed, gold)

        print("\n--- Results ---")
        print(f"Latency (s):        {s_rec.latency_s:.3f}")
        print(f"Peak RSS (MB):      {s_rec.peak_rss_mb:.0f}")
        print(f"Tokens in / out:    {s_rec.input_tokens} / {s_rec.output_tokens}")
        print(f"Raw output:\n{s_rec.raw_text or '(empty)'}\n")
        print("Gold JSON:", json.dumps(gold))
        print("Field accuracy:", f"{acc_s:.2f}", hits_s)
    finally:
        sys.stdout = old_stdout
        if out_fh is not None:
            out_fh.close()

    if args.output is not None:
        print(f"Wrote report to {args.output}", file=sys.stderr)


def cmd_slm_bench(args: argparse.Namespace) -> None:
    from ug9_benchmark.schema import build_prompt, score_prediction
    from ug9_benchmark.slm_runner import SLMRunner, load_examples

    examples = load_examples(str(DATA_PATH))
    if not examples:
        print(f"No examples in {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    subset = examples[: max(1, args.limit)]

    out_fh, stdout_target = _stdout_tee(args.output)
    old_stdout = sys.stdout
    rows: list[dict[str, object]] = []
    fieldnames = ["id", "latency_s", "field_accuracy", "output_tokens", "peak_rss_mb"]

    try:
        sys.stdout = stdout_target
        print("=== HF SLM labeled benchmark ===\n")
        print(f"Dataset: {DATA_PATH}")
        print(f"Model: {args.slm}")
        print(f"Examples: {len(subset)}\n")

        print("Loading model…")
        slm = SLMRunner(model_id=args.slm)

        for i, ex in enumerate(subset):
            prompt = build_prompt(ex["text"])
            gold = ex["gold"]
            s_rec = slm.run(prompt)
            acc_s, _ = score_prediction(s_rec.parsed, gold)
            row = {
                "id": ex["id"],
                "latency_s": round(s_rec.latency_s, 6),
                "field_accuracy": round(acc_s, 4),
                "output_tokens": s_rec.output_tokens,
                "peak_rss_mb": round(s_rec.peak_rss_mb, 2),
            }
            rows.append(row)
            print(
                f"[{i + 1}/{len(subset)}] id={ex['id']}  acc={acc_s:.2f}  lat={s_rec.latency_s:.3f}s",
                flush=True,
            )

        print("\n--- Per-row ---")
        for r in rows:
            print(
                f"id={r['id']}\tlat={r['latency_s']}s\tacc={r['field_accuracy']}\t"
                f"tok_out={r['output_tokens']}\trss_mb={r['peak_rss_mb']}",
            )

        mean_lat = sum(float(r["latency_s"]) for r in rows) / len(rows)
        mean_acc = sum(float(r["field_accuracy"]) for r in rows) / len(rows)
        print("\n--- Summary ---")
        print(f"Mean latency (s): {mean_lat:.3f}")
        print(f"Mean field accuracy: {mean_acc:.2f}")

        if args.csv is not None:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.csv, "w", encoding="utf-8", newline="") as cf:
                w = csv.DictWriter(cf, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            print(f"\nWrote CSV to {args.csv}")
    finally:
        sys.stdout = old_stdout
        if out_fh is not None:
            out_fh.close()

    if args.output is not None:
        print(f"Wrote report to {args.output}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="UG-9 local benchmarks.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download-gguf", help="Download SmolLM2 GGUF weights into models/gguf/")
    p_dl.set_defaults(func=cmd_download_gguf)

    p_g = sub.add_parser("gguf", help="Compare IQ3_XS vs Q4_K_M on one command")
    p_g.add_argument(
        "command",
        nargs="?",
        default="Dim office lights to 60 percent.",
        help="Natural-language IoT command",
    )
    p_g.add_argument("-o", "--output", type=Path, metavar="FILE", help="Save transcript")
    p_g.set_defaults(func=cmd_gguf)

    p_ss = sub.add_parser("slm-single", help="One HF SLM inference + scoring")
    p_ss.add_argument("--command", "-c", default="Dim office lights to 60 percent.")
    p_ss.add_argument("--slm", default="google/flan-t5-small")
    p_ss.add_argument("--gold-file", type=Path, metavar="PATH")
    p_ss.add_argument("-o", "--output", type=Path, metavar="FILE")
    p_ss.set_defaults(func=cmd_slm_single)

    p_sb = sub.add_parser("slm-bench", help="First N rows from iot_commands.json")
    p_sb.add_argument("--limit", type=int, default=8)
    p_sb.add_argument("--slm", default="google/flan-t5-small")
    p_sb.add_argument("--csv", type=Path, metavar="FILE")
    p_sb.add_argument("-o", "--output", type=Path, metavar="FILE")
    p_sb.set_defaults(func=cmd_slm_bench)

    ns = parser.parse_args()
    ns.func(ns)


if __name__ == "__main__":
    main()
