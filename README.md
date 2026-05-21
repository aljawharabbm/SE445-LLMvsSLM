# UG-9 LLM vs SLM (local benchmark)

**What this is:** You type smart-building style commands. Two small models try to answer in **JSON**. Everything runs on **your computer** (no OpenAI-style API in the default flow).

- **Track A — Hugging Face:** `google/flan-t5-small` on 12 examples in `ug9_benchmark/data/iot_commands.json`.
- **Track B — GGUF:** `SmolLM2` in two file sizes (**IQ3_XS** and **Q4_K_M**), one test sentence.

**How to run:** open a terminal **in this folder** (where `run.py` lives), then:

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py download-gguf            # downloads ~185 MB into models/gguf/

python run.py slm-bench --limit 12 -o reports/paper_slm.txt
python run.py gguf -o reports/paper_gguf.txt
```

You need **Python 3.10+** and **internet** the first time (`pip` + model download). On Windows, if `llama-cpp-python` fails to install, you may need Microsoft **C++ Build Tools**.

**Other useful commands:**

```bash
python run.py slm-single -c "Dim office lights to 60 percent."
python run.py gguf "Turn the hall lights to thirty percent."
python run.py --help
```

---

## Results from our saved runs (your numbers may differ a bit)

Rough summary from `reports/paper_slm.txt` and `reports/paper_gguf.txt`:

| What we ran | Speed | Match score (strict JSON) |
|-------------|-------|---------------------------|
| FLAN-T5-small, 12 commands | about **0.15 s** average | **0** (no perfect rows) |
| SmolLM2 IQ3_XS, one sentence | about **0.25 s** | **0** |
| SmolLM2 Q4_K_M, same sentence | about **0.30 s** | **0** |

**Why the score is often zero:** We check JSON **exactly** (flat keys like `office_lights`). Small models often answer in “human” JSON (nested fields, long text in `action`). That can be **wrong for the checker** even if a person understands it.

**First row slow?** The **first** HF run can be much slower (model warming up). Later rows are quicker.

---

## What’s in the folder

- **`run.py`** — main script.
- **`ug9_benchmark/`** — code + `data/iot_commands.json`.
- **`models/gguf/`** — empty in git; run `download-gguf` to fill it.
- **`reports/`** — text logs if you use `-o`.

**Small zip for class:** do **not** zip `.venv` or the big `.gguf` files; the grader can run `pip install` and `download-gguf` like you did.
