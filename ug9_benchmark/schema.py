from __future__ import annotations

import json
import re
from typing import Any


def _relax_trailing_commas(s: str) -> str:
    """Allow a single trailing comma before } or ] (common in model output)."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",(\s*})", r"\1", s)
        s = re.sub(r",(\s*\])", r"\1", s)
    return s


def _loads_json_dict(chunk: str) -> dict[str, Any] | None:
    chunk = chunk.strip()
    for candidate in (chunk, _relax_trailing_commas(chunk)):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _first_balanced_json_object(text: str) -> dict[str, Any] | None:
    """Find first `{ ... }` slice that parses as JSON object (handles strings/braces)."""
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[i : j + 1]
                    obj = _loads_json_dict(chunk)
                    if isinstance(obj, dict):
                        return obj
                    break
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object from model output (handles fences / chatter)."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    obj = _loads_json_dict(cleaned)
    if isinstance(obj, dict):
        return obj
    balanced = _first_balanced_json_object(cleaned)
    if balanced is not None:
        return balanced
    return None


def normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    if isinstance(v, float):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("null", "none", ""):
            return None
        try:
            return float(s) if "." in s else float(int(s))
        except ValueError:
            return s.strip()
    return v


def score_prediction(pred: dict[str, Any] | None, gold: dict[str, Any]) -> tuple[float, dict[str, bool]]:
    """Per-field fuzzy equality; returns (accuracy in [0,1], field hits)."""
    if not pred:
        return 0.0, {"device": False, "action": False, "value": False, "unit": False}
    fields = ("device", "action", "value", "unit")
    hits: dict[str, bool] = {}
    for f in fields:
        pv = pred.get(f, None)
        gv = gold.get(f, None)
        pn = normalize_value(pv)
        gn = normalize_value(gv)
        if f == "device":
            ok = isinstance(pn, str) and isinstance(gn, str) and pn.strip().lower() == gn.strip().lower()
        elif f == "action":
            ok = isinstance(pn, str) and isinstance(gn, str) and pn.strip().lower() == gn.strip().lower()
        elif f == "unit":
            ok = (pn is None and gn is None) or (
                isinstance(pn, str) and isinstance(gn, str) and pn.strip().lower() == gn.strip().lower()
            )
        else:  # value — allow numeric tolerance and case-insensitive strings
            if pn is None and gn is None:
                ok = True
            elif pn is not None and gn is not None:
                if isinstance(pn, (int, float)) and isinstance(gn, (int, float)):
                    ok = float(pn) == float(gn)
                else:
                    ok = str(pn).strip().lower() == str(gn).strip().lower()
            else:
                ok = False
        hits[f] = bool(ok)
    acc = sum(hits.values()) / len(fields)
    return acc, hits


def build_prompt(user_text: str) -> str:
    return (
        "You convert smart-building / IoT voice commands into STRICT JSON.\n"
        "Keys only: device (snake_case string), action (snake_case string), "
        "value (number or null), unit (string or null).\n"
        "Respond with ONE JSON object only, no markdown.\n\n"
        f'Command: "{user_text}"'
    )
