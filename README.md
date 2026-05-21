# UG-9: Local benchmark — HF seq2seq (FLAN-T5-small) vs quantized GGUF (SmolLM2)

This repository is a coursework-sized **performance–cost** comparison for **IoT-style natural language → strict JSON**, run **entirely on your machine** (no paid cloud LLM APIs in the default path). You get one **Hugging Face** track using **`google/flan-t5-small`** and one **GGUF** track using **SmolLM2-135M-Instruct** at two quantization levels (**IQ3_XS** vs **Q4_K_M**) via **llama.cpp** (`llama-cpp-python`). A single CLI, **`run.py`**, downloads weights when needed, runs inference, and scores outputs against a **fixed four-field schema** (`device`, `action`, `value`, `unit`).

The sections below pair **how to run** the project with **what we actually measured** on a captured run. Verbatim logs are stored under **`reports/paper_slm.txt`** and **`reports/paper_gguf.txt`**; rerunning on another PC will change latencies slightly but the **qualitative** story (strict accuracy vs generative paraphrase) usually holds unless you change models or prompts.

---

## Problem and scoring

Each example is an English utterance describing a facility command. The gold target is JSON with **flat** string or numeric slots. Predictions must contain a parsable JSON object (the harness strips fences when present); the scorer then compares the four keys with simple normalization rules. Nested objects, free-text where a snake_case token is expected, or wrong numeric structure produce **partial or zero field matches**. This is deliberate: tiny models often emit **readable** explanations instead of automation-friendly literals, which shows up harshly under strict grading even when a human understands the intent.

---

## Experimental configuration 

The **HF** benchmark uses **`google/flan-t5-small`**, **`--limit 12`** over the bundled dataset **`ug9_benchmark/data/iot_commands.json`** (identifiers **`ex01`–`ex12`**). Decoder settings match the shipped runners (beam search, no sampling caps on output length consistent with **`slm_runner.py`**).

The **GGUF** probe uses the default utterance **`Dim office lights to 60 percent.`** and loads **both** files under **`models/gguf/`** in one invocation (so wall-clock reflects sequential load + infer for IQ3 then Q4, not a long-lived daemon). Disk sizes reported below come from those **`.gguf`** files at report time (**84.1 MB** IQ3 tier, **100.6 MB** Q4 tier).

---

## Summary results

On the logged **12-example HF pass**, **mean latency** was **0.150 s** and **mean field accuracy** was **0.00** (every row scored zero under strict matching). Latency ranged from approximately **0.067 s** on **`ex02`** to **0.551 s** on **`ex01`** — the latter is typical when the runtime or weight cache has not stabilized; interpreting **warm vs cold** draw matters when you quote a single headline number. Process **RSS** after loading settled near **622–624 MB** in the transcript, which proxies memory pressure for laptops without pretending to replace full power profiling.

| Track | Artifact | Mean latency | Field accuracy (mean or probe) | Footprint cue |
|--------|-----------|---------------|---------------------------------|---------------|
| HF — FLAN-T5-small (`N = 12`) | `reports/paper_slm.txt` | **0.150 s** | **0.00** | Peak RSS ~**622–624 MB** row-level |
| GGUF — IQ3_XS | `reports/paper_gguf.txt` | **0.251 s** probe | **0.00** | **84.1 MB** on disk |
| GGUF — Q4_K_M | same | **0.298 s** probe | **0.00** | **100.6 MB** on disk |

**GGUF illustrative outputs (truncated fidelity).** The IQ3 shard echoed natural language inside **`action`** and wrapped **`value`** in a nested **`number` / `unit`** object instead of atomic fields. Q4 bundled an Office-oriented mini-structure under **`device`** and **`unit`: null**. Both shapes fail equality checks against **`{"device":"office_lights","action":"set_brightness","value":60,"unit":"percent"}`**. That mismatch is informative: **quantization moved latency and megabytes**, not the fundamental habit of emitting **conversation-shaped JSON**.

**Takeaway for graders and readers.** The benchmark is intentionally **honest**. Throughput stays in a range that feels interactive on a seminar laptop for these small workloads, yet **semantic alignment to a brittle schema** collapses **without constrained decoding**, task fine-tuning, or a larger instruction-tuned model family. Discussing **why** zeros appear is therefore as important as the latency table.

**Per-example telemetry (logged `slm-bench`, `N = 12`).** The following table echoes **`reports/paper_slm.txt`**. Latency is end-to-end seconds for each utterance after the shared model load; **`tok_out`** is generated tokens and **`rss_mb`** is peak resident set from the transcript (an approximate footprint proxy, not a substitute for calibrated energy accounting).

| ID | Latency (s) | Output tokens | Peak RSS (MB) | Field accuracy |
|----|------------:|--------------:|---------------:|:--------------:|
| ex01 | 0.551 | 12 | 622.09 | 0.00 |
| ex02 | 0.067 | 5 | 622.23 | 0.00 |
| ex03 | 0.114 | 12 | 622.69 | 0.00 |
| ex04 | 0.096 | 10 | 622.83 | 0.00 |
| ex05 | 0.087 | 8 | 622.89 | 0.00 |
| ex06 | 0.115 | 12 | 623.06 | 0.00 |
| ex07 | 0.097 | 8 | 623.22 | 0.00 |
| ex08 | 0.171 | 17 | 623.36 | 0.00 |
| ex09 | 0.087 | 7 | 623.47 | 0.00 |
| ex10 | 0.125 | 12 | 623.61 | 0.00 |
| ex11 | 0.160 | 16 | 623.70 | 0.00 |
| ex12 | 0.124 | 11 | 623.83 | 0.00 |

The gap between **`ex02`** near **0.07 s** and **`ex08`** near **0.17 s** among later rows suggests **length- and search-dependent** decode cost even when the model stays resident, while **`ex01`** documents the **first-row** penalty that inflates simple means. External replications should always pair numbers with **CPU/GPU class**, **OS build**, and **library versions** because otherwise two honest student labs can disagree on the third decimal without anyone being wrong.

---

## How to reproduce

Someone receiving only the repo (no **`.venv`**, no **`*.gguf`**) needs **Python 3.10+** and outbound internet for **`pip`** and Hugging Face / GGUF downloads.

**macOS / Linux.**

```bash
cd /path/to/this/repo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py download-gguf
python run.py slm-bench --limit 12 --csv reports/paper_slm.csv -o reports/paper_slm.txt
python run.py gguf -o reports/paper_gguf.txt
```

**Windows (after `cd` into the repo).**

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py download-gguf
python run.py slm-bench --limit 12 --csv reports\paper_slm.csv -o reports\paper_slm.txt
python run.py gguf -o reports\paper_gguf.txt
```

**`download-gguf`** fetches roughly **185 MB** of weights into **`models/gguf/`**. The HF model downloads on first **`slm-bench`** invocation into the Hugging Face cache (**not** this repo folder). Environment variables **`UG9_GGUF_CTX`** and **`UG9_GGUF_THREADS`** optionally tune context length and threading for GGUF runs.

Routine commands worth remembering: **`python run.py slm-single -c \"…\"`** for a single HF trial; **`python run.py gguf \"Custom text here\"`** for another GGUF probe; **`python run.py --help`** for subcommands.

---

## Dependencies

Declared in **`requirements.txt`**: **PyTorch**, **Transformers**, **Accelerate**, **SentencePiece**, **psutil**, **huggingface_hub**, **llama-cpp-python**. Pins are minimal lower bounds; installers resolve wheels per OS. If **`llama-cpp-python`** compilation fails on Windows, install the **Desktop development with C++** workload tools and retry **`pip install`**.

---

## Repository layout

**`run.py`** is the public entrypoint. **`ug9_benchmark/`** contains **`schema.py`** (prompt + JSON extract + scoring), **`slm_runner.py`**, **`gguf_runner.py`**, and **`data/iot_commands.json`**. **`models/gguf/`** stays in git via **`.gitkeep`** only — actual **`.gguf`** blobs are excluded by **`.gitignore`** — so recipients must **`download-gguf`**.

**Submission hygiene.** Omit **`.venv`**, **`__pycache__`**, and **`*.gguf`** from zip uploads unless the syllabus demands frozen weights; the **`.gitignore`** already biases the tree toward **small archives**.

---

## Limitations (scope of these README numbers)

Twelve supervised utterances do not statistics-of-record make; they suffice to **illustrate** pipeline behavior while keeping grading reproducible. The GGUF slice is deliberately **single-sentence**, so throughput conclusions about IQ3 versus Q4 are **paired-comparison probes**, not substitutes for profiling full-day facility dialogue. Neither track reports **Joules**, **electricity cost**, or **WAN charges** directly—those belonged to broader survey discussion in coursework narrative. Finally, enforcing **deterministic-ish** transformers still leaves minor float timer noise; trust **directionality** (**IQ3 lighter / slightly faster here**) more than asserting universal rank order on every SKU.

---

## Troubleshooting briefly

Missing GGUF paths produce a clear **`Run: python run.py download-gguf`** hint. Sandbox-only CI shells sometimes choke on **`llama.cpp`** mmap expectations — developers should validate on bare OS laptops before blaming checkpoint files. First-row HF spikes on fresh processes are usually **initialization**, not a broken timer; annotate them in plots or drop from “steady-state” means if your write-up distinguishes **cold vs warm** behavior.
