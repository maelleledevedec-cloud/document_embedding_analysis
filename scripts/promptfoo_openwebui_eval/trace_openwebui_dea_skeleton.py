#!/usr/bin/env python3
"""Experimental Trace/NewTrace optimizer for OpenWebUI tool/model evaluation.

This version is intentionally standalone and more generic than the first draft.
It can optimize either:

1. an OpenWebUI tool/pipe target (``OpenWebUISummarizerAgent``), typically the
   ``large_thematic_summarizer`` reached through a pipe model such as
   ``summarizer---kohaku-OR``;
2. an OpenWebUI model target (``OpenWebUIModel``), i.e. a workspace-model-like
   wrapper with a trainable system prompt plus request-time model parameters.

Generation always goes through the same local bridge endpoint. The script simply
changes the target model id that the bridge should call and how the row is
prepared before sending it.

Evaluation backends:
- ``dea``: native DEA evaluation via ``common.doc_eval.evaluate_document``.
- ``promptfoo``: batch evaluation by invoking a Promptfoo command on a generated
  CSV and parsing the resulting JSON.
- ``llm_judge``: local/custom LLM-as-a-Judge with a user-provided prompt.

The script is also usable for non-summarization / QA-style CSV tasks as long as
rows contain enough information to build a request prompt and a reference.
If the input CSV does not exist yet, the script can create it by running an
external builder command (for example ``build_promptfoo_csvs.py`` or another
Promptfoo-oriented dataset preparation script).

Notes / limitations:
- The current OpenWebUI bridge forwards per-call system prompts and generation
  parameters without mutating shared OpenWebUI model settings. For strict model
  behavior constraints, ``OpenWebUIModel`` can still wrap the system prompt into
  the user prompt because some models treat native system prompts as guidance
  rather than a hard output contract.
- The currently uploaded ``large_thematic_summarizer`` tool exposes only a
  subset of its runtime controls via method arguments / UserValves. Per-call
  optimization of admin-only valves is not guaranteed unless the tool/pipe is
  extended accordingly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests

# ---------------------------------------------------------------------------
# Optional bundle/dea imports
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
for _candidate in (
    _HERE.parent,
    _HERE.parent.parent,
    Path.cwd(),
):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:
    from lib.bundle_common import parse_jsonish as _bundle_parse_jsonish  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    _bundle_parse_jsonish = None

# Trace / NewTrace experimental
from opto import trace
from opto.optimizers import OptoPrime
from opto.trainer.algorithms.algorithm import Trainer
from opto.trainer.guide import Guide
from opto.trainer.loader import DataLoader
from opto.trainer.objectives import ObjectiveConfig, apply_minimize, weighted_scalarize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGORITHMS = {"raptor", "dtcrs", "kohaku", "lightrag"}
LENGTHS = {"short", "medium", "long"}
MODEL_PROMPT_MODES = {"wrapped", "bridge_field", "both", "none"}
GENERATION_MODES = {"bridge", "prepared_rows"}
OPTIMIZATION_METHODS = {
    "direct_dea",
    "direct_dea_judge",
    "direct_llm_judge",
    "step2_promptfoo",
    "step3_promptfoo",
}
TOOL_MODEL_IDS_BY_ALGORITHM = {
    "dtcrs": "summarizer---dtcrs",
    "kohaku": "summarizer---kohaku",
    "kohaku_openrouter": "summarizer---kohaku-OR",
    "lightrag": "summarizer---lightrag",
    "raptor": "summarizer---raptor",
}
TOOL_EXPOSED_PER_CALL_KEYS = {
    # Current uploaded tool: method args and UserValves that can plausibly be
    # influenced per call through the common pipe.
    "algorithm",
    "target_length",
    "structure",
    "instruction",
    "summarizer_model_id",
    "target_language",
    "default_target_length",
    "summary_structure",
    "max_context_chars_per_llm_call",
    "max_final_summary_chars",
    "enable_map_reduce",
    "max_concurrent_llm_calls",
    "diagnostic_status",
    "emit_diagnostics",
}
REFERENCE_FIELD_CANDIDATES = (
    "gold_summary",
    "reference_answer",
    "expected_answer",
    "reference",
    "expected",
    "answer",
    "target",
    "gold",
)
PROMPT_FIELD_CANDIDATES = (
    "request_prompt",
    "question",
    "prompt",
    "input",
    "query",
    "instruction",
    "title",
)
DEFAULT_OBJECTIVE_WEIGHTS = {
    "score": 1.0,
    "execution_duration_score": 0.15,
    "llm_calls_score": 0.05,
    "tool_calls_score": 0.03,
}
DEFAULT_MINIMIZE_METRICS = (
    "execution_duration_s",
    "duration_s",
    "latency_s",
    "llm_calls",
    "llm_rounds",
    "tool_calls",
)
QUALITY_OBJECTIVE_KEYS = {
    "score",
    "dea_score",
    "dea_composite",
    "plan",
    "content",
    "resources",
    "length_alignment",
    "rouge_l_f",
    "promptfoo_score",
    "judge_score",
}
RUNTIME_OBJECTIVE_SCORE_KEYS = {
    "execution_duration_score",
    "duration_score",
    "latency_score",
    "llm_calls_score",
    "llm_rounds_score",
    "tool_calls_score",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def parse_jsonish(value: Any, default: Any = None) -> Any:
    if _bundle_parse_jsonish is not None:
        try:
            return _bundle_parse_jsonish(value, default=default)
        except Exception:
            pass
    if value is None:
        return default
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# Allow large CSV cells (long prompts / references / outputs).
def raise_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    raise_csv_field_size_limit()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(r) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    all_fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def mean_dict(dicts: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({k for d in dicts for k in d.keys()})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(d[key]) for d in dicts if key in d]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def row_batches(rows: Sequence[Mapping[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """Group rows into deterministic optimization batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    normalized = [dict(row) for row in rows]
    return [normalized[i : i + batch_size] for i in range(0, len(normalized), batch_size)]


def task_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return stable identifiers for reporting a row batch."""
    ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        ids.append(str(row.get("task_id") or row.get("id") or index))
    return ids


def trace_metrics(outputs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate runtime metrics from bridge responses for optimization."""
    latencies = [float(item.get("latency_s") or 0.0) for item in outputs]
    rounds = []
    tool_calls = []
    reasoning_chars = []
    for item in outputs:
        trace_data = ((item.get("raw_response") or {}).get("trace") or {}) if isinstance(item.get("raw_response"), dict) else {}
        trace_rounds = trace_data.get("rounds") or []
        trace_tools = trace_data.get("tool_calls") or []
        rounds.append(float(len(trace_rounds)))
        tool_calls.append(float(len(trace_tools)))
        total_reasoning = 0
        for round_item in trace_rounds:
            message = (round_item or {}).get("message") or {}
            for key in ("reasoning", "reasoning_content", "thinking"):
                total_reasoning += len(str(message.get(key) or ""))
        reasoning_chars.append(float(total_reasoning))
    metrics: dict[str, float] = {}
    if latencies:
        duration_s = sum(latencies) / len(latencies)
        duration_score = 1.0 / (1.0 + duration_s)
        metrics["execution_duration_s"] = duration_s
        metrics["execution_duration_score"] = duration_score
        metrics["duration_s"] = duration_s
        metrics["duration_score"] = duration_score
        metrics["latency_s"] = duration_s
        metrics["latency_score"] = duration_score
    if rounds:
        llm_calls = sum(rounds) / len(rounds)
        llm_calls_score = 1.0 / (1.0 + llm_calls)
        metrics["llm_calls"] = llm_calls
        metrics["llm_calls_score"] = llm_calls_score
        metrics["llm_rounds"] = llm_calls
        metrics["llm_rounds_score"] = llm_calls_score
    if tool_calls:
        metrics["tool_calls"] = sum(tool_calls) / len(tool_calls)
        metrics["tool_calls_score"] = 1.0 / (1.0 + metrics["tool_calls"])
    if reasoning_chars:
        metrics["reasoning_chars"] = sum(reasoning_chars) / len(reasoning_chars)
    return metrics


def total_duration_metrics(duration_s: float) -> dict[str, float]:
    """Return normalized metrics for the full optimization step runtime."""
    safe_duration = max(0.0, float(duration_s))
    return {
        "execution_duration_s": safe_duration,
        "execution_duration_score": 1.0 / (1.0 + safe_duration),
        "optimization_step_duration_s": safe_duration,
        "optimization_step_duration_score": 1.0 / (1.0 + safe_duration),
    }


def merge_step_runtime_metrics(score_dict: Mapping[str, float], duration_s: float) -> dict[str, float]:
    """Add full-step runtime and finite no-call defaults without hiding bridge traces."""
    merged = dict(score_dict)
    merged.update(total_duration_metrics(duration_s))
    # Online Process PromptFoo may not expose bridge traces; keep the default
    # objective finite while preserving real call counts when traces exist.
    merged.setdefault("llm_calls", 0.0)
    merged.setdefault("llm_calls_score", 1.0)
    merged.setdefault("llm_rounds", merged["llm_calls"])
    merged.setdefault("llm_rounds_score", merged["llm_calls_score"])
    merged.setdefault("tool_calls", 0.0)
    merged.setdefault("tool_calls_score", 1.0)
    return merged


def clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _ratio_alignment(value: Any) -> float | None:
    """Score a ratio by closeness to 1.0, capped to the 0-1 optimization range."""
    ratio = numeric(value)
    if ratio is None:
        return None
    return clamp01(1.0 - min(abs(ratio - 1.0), 1.0))


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        preferred = []
        for key in (
            "embedding2_cosine_similarity",
            "embedding1_cosine_similarity",
            "cosine_similarity",
            "score",
            "similarity",
            "f",
        ):
            v = value.get(key)
            if isinstance(v, (int, float)):
                preferred.append(float(v))
        if preferred:
            return sum(preferred) / len(preferred)
        vals = [float(v) for v in value.values() if isinstance(v, (int, float))]
        if vals:
            return sum(vals) / len(vals)
    return None


def pick(scores: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in scores:
            value = numeric(scores.get(key))
            if value is not None:
                return value
    return None


def trim_text(text: str, limit: int = 4000) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def repo_root() -> Path:
    root = os.environ.get("DEA_REPO_ROOT", "").strip() or os.environ.get("DEA_REPO_PATH", "").strip()
    if not root:
        raise RuntimeError("DEA_REPO_ROOT or DEA_REPO_PATH must be set")
    return Path(root).resolve()


def ensure_repo_on_path() -> Path:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(str(path_str))
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def first_non_empty(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    for key in fields:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def resolve_reference_text(row: Mapping[str, Any]) -> str:
    direct = first_non_empty(row, REFERENCE_FIELD_CANDIDATES)
    if direct:
        return direct
    for key in ("expected_path", "reference_path", "gold_path"):
        path = resolve_repo_path(str(row.get(key) or "").strip())
        if path and path.exists() and path.is_file():
            try:
                return read_text(path)
            except Exception:
                pass
    return ""


def derive_request_prompt(row: Mapping[str, Any]) -> str:
    prompt = first_non_empty(row, PROMPT_FIELD_CANDIDATES)
    if prompt:
        return prompt
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or row.get("context") or "").strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(abstract)
    return "\n\n".join(parts).strip()


def normalize_generation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    current = dict(row)
    if not str(current.get("request_prompt") or "").strip():
        current["request_prompt"] = derive_request_prompt(current)
    if "candidate_answer" not in current:
        current["candidate_answer"] = ""
    return current


def run_shell(command: str, *, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        shell=True,
        cwd=str(cwd) if cwd else None,
        env=dict(os.environ, **dict(env or {})),
        check=False,
        text=True,
        capture_output=True,
    )


def maybe_build_input_csv(input_csv: Path, build_cmd_template: str | None, *, force: bool = False) -> None:
    if input_csv.exists() and not force:
        return
    if not build_cmd_template:
        raise FileNotFoundError(
            f"Input CSV does not exist: {input_csv}. Provide --build-input-csv-cmd to create it."
        )
    command = build_cmd_template.format(output_csv=shlex.quote(str(input_csv)))
    proc = run_shell(command, cwd=Path.cwd())
    if proc.returncode != 0:
        raise RuntimeError(
            f"CSV builder command failed ({proc.returncode}).\nCOMMAND: {command}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if not input_csv.exists():
        raise FileNotFoundError(
            f"CSV builder command completed but did not create {input_csv}."
        )


# ---------------------------------------------------------------------------
# Parameter sanitation and row application
# ---------------------------------------------------------------------------


def sanitize_tool_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(params or {})
    algo = str(out.get("algorithm") or "kohaku").strip().lower()
    out["algorithm"] = algo if algo in ALGORITHMS else "kohaku"

    target_length = str(out.get("target_length") or "long").strip().lower()
    out["target_length"] = target_length if target_length in LENGTHS else "long"

    def as_int(name: str, low: int, high: int, default: int) -> None:
        try:
            value = int(float(out.get(name, default)))
        except Exception:
            value = default
        out[name] = max(low, min(high, value))

    def as_float(name: str, low: float, high: float, default: float) -> None:
        try:
            value = float(out.get(name, default))
        except Exception:
            value = default
        out[name] = max(low, min(high, value))

    as_int("max_context_chars_per_llm_call", 8000, 48000, 24000)
    as_int("max_final_summary_chars", 1000, 12000, 4000)
    as_int("max_concurrent_llm_calls", 1, 8, 4)
    as_int("kohaku_n_axes", 2, 12, 6)
    as_int("kohaku_top_sections_per_axis", 1, 8, 3)
    as_int("kohaku_top_paragraphs_per_axis", 2, 16, 8)
    as_int("kohaku_ensemble_size", 1, 6, 3)
    as_float("kohaku_ensemble_temperature", 0.0, 1.0, 0.4)

    out["enable_map_reduce"] = bool(out.get("enable_map_reduce", False))
    out["structure"] = str(out.get("structure") or out.get("summary_structure") or "thematic")
    return out


def sanitize_model_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(params or {})

    def as_int(name: str, low: int, high: int, default: int) -> None:
        try:
            value = int(float(out.get(name, default)))
        except Exception:
            value = default
        out[name] = max(low, min(high, value))

    def as_float(name: str, low: float, high: float, default: float) -> None:
        try:
            value = float(out.get(name, default))
        except Exception:
            value = default
        out[name] = max(low, min(high, value))

    # Current bridge supports these fields directly.
    as_float("generation_temperature", 0.0, 2.0, 0.2)
    as_float("generation_top_p", 0.0, 1.0, 0.95)
    as_int("generation_max_tokens", 64, 64000, 4096)

    if "stop_sequences" in out and not isinstance(out["stop_sequences"], list):
        parsed = parse_jsonish(out["stop_sequences"], default=[])
        out["stop_sequences"] = parsed if isinstance(parsed, list) else []
    if "model_extra_payload_json" in out:
        extra_payload = parse_jsonish(out["model_extra_payload_json"], default=out["model_extra_payload_json"])
        if not isinstance(extra_payload, dict):
            raise ValueError("model_extra_payload_json must be a JSON object")
        out["model_extra_payload_json"] = extra_payload
    return out


def sanitize_system_prompt(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    if len(s) > 12000:
        s = s[:12000].rstrip() + "\n\n[TRUNCATED]"
    return s


def merge_tool_params_into_row(
    row: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    target_model_id: str,
    strict_exposed_keys: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    current = normalize_generation_row(row)
    warnings: list[str] = []
    current["openwebui_pipe_model"] = target_model_id

    tool_parameters = parse_jsonish(current.get("tool_parameters_json"), default={}) or {}
    if not isinstance(tool_parameters, dict):
        tool_parameters = {}

    for key, value in dict(params).items():
        if key in {"algorithm", "target_length", "structure", "summary_structure", "summarizer_model_id"}:
            current["structure" if key == "summary_structure" else key] = value
            continue

        if key in TOOL_EXPOSED_PER_CALL_KEYS:
            tool_parameters[key] = value
            continue

        msg = (
            f"Tool parameter `{key}` is not known to be exposed per call by the current pipe/tool; "
            "the bridge row was updated, but the live tool may ignore it unless the tool exposes it "
            "as a method argument or UserValve."
        )
        warnings.append(msg)
        if not strict_exposed_keys:
            tool_parameters[key] = value

    current["tool_parameters_json"] = json.dumps(tool_parameters, ensure_ascii=False)
    return current, warnings


def wrap_request_with_system_prompt(system_prompt: str, request_prompt: str) -> str:
    sp = sanitize_system_prompt(system_prompt)
    rp = str(request_prompt or "").strip()
    if not sp:
        return rp
    return (
        "<optimized_system_prompt>\n"
        f"{sp}\n"
        "</optimized_system_prompt>\n\n"
        "<user_task>\n"
        f"{rp}\n"
        "</user_task>\n\n"
        "Follow the optimized system prompt above as the behavioral contract for this answer."
    ).strip()


def apply_model_params_to_row(
    row: Mapping[str, Any],
    system_prompt: str,
    params: Mapping[str, Any],
    *,
    target_model_id: str,
    prompt_mode: str,
    keep_existing_tool_fields: bool = False,
) -> dict[str, Any]:
    current = normalize_generation_row(row)
    current["openwebui_pipe_model"] = target_model_id
    current["openwebui_target_kind"] = "model"
    current["openwebui_system_prompt"] = sanitize_system_prompt(system_prompt)
    current["openwebui_model_params_json"] = json.dumps(dict(params), ensure_ascii=False)

    prompt_mode = prompt_mode if prompt_mode in MODEL_PROMPT_MODES else "wrapped"
    if prompt_mode == "wrapped":
        current["request_prompt"] = wrap_request_with_system_prompt(
            current["openwebui_system_prompt"],
            derive_request_prompt(current),
        )
        current["openwebui_system_prompt"] = ""
    elif prompt_mode == "bridge_field":
        current["request_prompt"] = derive_request_prompt(current)
    elif prompt_mode == "both":
        current["request_prompt"] = wrap_request_with_system_prompt(
            current["openwebui_system_prompt"],
            derive_request_prompt(current),
        )
    else:
        current["request_prompt"] = derive_request_prompt(current)
        current["openwebui_system_prompt"] = ""

    if not keep_existing_tool_fields:
        current["tool_parameters_json"] = "{}"
        current["algorithm"] = ""
        current["target_length"] = current.get("target_length") or ""
        current["structure"] = current.get("structure") or ""

    for src, dst in (
        ("generation_temperature", "generation_temperature"),
        ("generation_top_p", "generation_top_p"),
        ("generation_max_tokens", "generation_max_tokens"),
    ):
        if src in params:
            current[dst] = params[src]

    return current


# ---------------------------------------------------------------------------
# Bridge generation
# ---------------------------------------------------------------------------


def bridge_generate(
    row: Mapping[str, Any],
    bridge_url: str,
    timeout_seconds: int = 1800,
) -> tuple[str, float, dict[str, Any]]:

    payload = {
        k: v
        for k, v in dict(row).items()
        if not str(k).startswith("__metadata:")
    }

    print("\n================ BRIDGE REQUEST ================")
    print("URL :", bridge_url)
    print("Model :", payload.get("openwebui_pipe_model"))
    print("Target kind :", payload.get("openwebui_target_kind"))
    print("System prompt :", payload.get("openwebui_system_prompt"))
    print("Request prompt :", str(payload.get("request_prompt"))[:500])
    print("Generation params :")
    print("  temperature =", payload.get("generation_temperature"))
    print("  top_p       =", payload.get("generation_top_p"))
    print("  max_tokens  =", payload.get("generation_max_tokens"))
    print("Payload keys :", sorted(payload.keys()))
    print("===============================================\n")

    t0 = time.time()

    response = requests.post(
        bridge_url,
        json=payload,
        timeout=timeout_seconds,
    )

    print("Bridge status :", response.status_code)

    if response.status_code >= 400:
        print("Bridge response:")
        print(response.text)

    response.raise_for_status()

    data = response.json()

    return (
        str(data.get("output") or "").strip(),
        time.time() - t0,
        data,
    )


def generate_batch_via_bridge(
    rows: Sequence[Mapping[str, Any]],
    *,
    bridge_url: str,
    row_transformer: Callable[[Mapping[str, Any]], dict[str, Any]],
    timeout_seconds: int = 1800,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    total = len(rows)
    for idx, raw_row in enumerate(rows, start=1):
        active_row = row_transformer(raw_row)
        answer, latency_s, raw = bridge_generate(active_row, bridge_url, timeout_seconds)
        current = dict(active_row)
        current["candidate_answer"] = answer
        current["generated_latency_s"] = latency_s
        outputs.append(
            {
                "row": current,
                "candidate_answer": answer,
                "latency_s": latency_s,
                "raw_response": raw,
                "row_index": idx - 1,
                "task_id": current.get("task_id") or current.get("id") or idx,
            }
        )
        print(f"[{idx}/{total}] generated {current.get('task_id') or current.get('id') or idx}")
    return outputs


def prepare_batch_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_transformer: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prepare rows for external live generation without calling the bridge."""
    outputs: list[dict[str, Any]] = []
    for idx, raw_row in enumerate(rows, start=1):
        active_row = row_transformer(raw_row)
        answer = str(active_row.get("candidate_answer") or "").strip()
        outputs.append(
            {
                "row": active_row,
                "candidate_answer": answer,
                "latency_s": 0.0,
                "raw_response": {"trace": {"mode": "prepared_rows", "message_count": 0, "rounds": [], "tool_calls": []}},
                "row_index": idx - 1,
                "task_id": active_row.get("task_id") or active_row.get("id") or idx,
            }
        )
    return outputs


# ---------------------------------------------------------------------------
# DEA evaluation
# ---------------------------------------------------------------------------


def _load_dea_solution(row: Mapping[str, Any]) -> dict[str, Any]:
    ensure_repo_on_path()
    path_str = str(row.get("dea_solution_path") or row.get("solution_path") or "").strip()
    if not path_str:
        raise RuntimeError("Missing dea_solution_path in row")
    path = resolve_repo_path(path_str)
    if not path or not path.exists():
        raise FileNotFoundError(path_str)
    return json.loads(read_text(path))


def evaluate_candidate_with_dea(row: Mapping[str, Any], output: str, *, use_dea_judge: bool = False) -> dict[str, Any]:
    ensure_repo_on_path()
    from common.doc_eval import evaluate_document

    solution = _load_dea_solution(row)
    result = evaluate_document(
        document_content=output,
        solution=solution,
        content_type=str(row.get("candidate_content_type") or "markdown"),
        use_enhanced_metrics=False,
        skip_dea=False,
        openai_model=os.environ.get("OPENAI_MODEL") or None,
        dea_embedding_backend=os.environ.get("DEA_EMBEDDING_BACKEND") or None,
        dea_embedding_model=os.environ.get("DEA_EMBEDDING_MODEL") or None,
        skip_entity_recall=True,
        use_dea_judge=use_dea_judge,
        dea_judge_model=os.environ.get("DEA_JUDGE_MODEL") or None,
    )

    dea_status = result.get("dea_evaluation_status", {}) or {}
    dea_scores = result.get("dea_evaluation_scores", {}) or {}
    article_metrics = result.get("article_metrics", {}) or {}
    rouge_scores = article_metrics.get("rouge_scores", {}) or {}
    rouge_l = rouge_scores.get("rouge-l", {}) or rouge_scores.get("rougeL", {}) or {}
    rouge_l_f = clamp01(numeric(rouge_l.get("f")))
    entity_recall = clamp01(numeric(article_metrics.get("entity_recall")))
    citation_count = numeric(article_metrics.get("citation_count")) or 0.0

    plan = clamp01(
        pick(
            dea_scores,
            [
                "plan_embedding_similarity",
                "section_total_similarity",
                "plan_total_similarity",
                "global_plan_embedding_similarity",
                "global_section_total_similarity",
                "global_plan_total_similarity",
            ],
        )
    )
    content = clamp01(
        pick(
            dea_scores,
            [
                "plan_contents_embedding_similarity",
                "content_total_similarity",
                "global_plan_contents_embedding_similarity",
                "global_content_total_similarity",
            ],
        )
    )
    resources = clamp01(
        pick(
            dea_scores,
            [
                "plan_resources_embedding_similarity",
                "resources_citation_coverage_score",
                "bibliography_coverage1",
                "bibliography_coverage2",
                "global_plan_resources_embedding_similarity",
            ],
        )
    )
    length_alignment = _ratio_alignment(
        pick(
            dea_scores,
            [
                "content_length_ratio_to_target",
                "global_content_length_ratio_to_target",
                "sections contents length (top:1, <1:too short, >1:too long)",
                "global_sections contents length (top:1, <1:too short, >1:too long)",
            ],
        )
    )

    weights = {"plan": 0.30, "content": 0.45, "resources": 0.15, "rouge_l": 0.10}
    available = {"plan": plan, "content": content, "resources": resources, "rouge_l": rouge_l_f}
    weighted_terms = {k: v for k, v in available.items() if v is not None and weights[k] > 0}
    if weighted_terms:
        composite = sum(weights[k] * weighted_terms[k] for k in weighted_terms) / sum(weights[k] for k in weighted_terms)
    else:
        composite = 0.0

    judge = result.get("dea_judge", {}) or {}
    judge_status = str(judge.get("status", "skipped"))
    judge_problem_count = float(len(judge.get("problems", []) or []))

    named_scores = {
        "dea_status_ok": 1.0 if dea_status.get("status") not in {"error"} else 0.0,
        "plan": plan if plan is not None else 0.0,
        "content": content if content is not None else 0.0,
        "resources": resources if resources is not None else 0.0,
        "length_alignment": length_alignment if length_alignment is not None else 0.0,
        "rouge_l_f": rouge_l_f if rouge_l_f is not None else 0.0,
        "entity_recall": entity_recall if entity_recall is not None else 0.0,
        "citation_count": citation_count,
        "dea_composite": composite,
        "dea_judge_ran": 0.0 if judge_status in {"skipped", "error"} else 1.0,
        "dea_judge_problem_count": judge_problem_count,
    }
    for key, value in dea_scores.items():
        v = numeric(value)
        if v is not None:
            named_scores.setdefault(str(key), float(v))

    reason_bits = [
        f"plan={plan:.3f}" if plan is not None else "plan=n/a",
        f"content={content:.3f}" if content is not None else "content=n/a",
        f"resources={resources:.3f}" if resources is not None else "resources=n/a",
        f"length={length_alignment:.3f}" if length_alignment is not None else "length=n/a",
        f"rouge_l_f={rouge_l_f:.3f}" if rouge_l_f is not None else "rouge_l_f=n/a",
        f"score={composite:.3f}",
        f"dea_status={dea_status.get('status', 'unknown')}",
        f"judge_status={judge_status}",
    ]
    return {
        "pass": composite >= 0.70,
        "score": composite,
        "reason": " | ".join(reason_bits),
        "namedScores": named_scores,
    }


def aggregate_dea_batch(outputs: Sequence[Mapping[str, Any]], *, use_dea_judge: bool = False) -> tuple[dict[str, float], str]:
    per_row = []
    reasons = []
    for item in outputs:
        row = dict(item["row"])
        scored = evaluate_candidate_with_dea(row, str(item["candidate_answer"]), use_dea_judge=use_dea_judge)
        named = dict(scored.get("namedScores", {}))
        named.setdefault("dea_score", float(scored.get("score") or 0.0))
        named["latency_s"] = float(item.get("latency_s") or 0.0)
        per_row.append(named)
        reasons.append((named.get("dea_composite", named.get("dea_score", 0.0)), row.get("task_id"), scored.get("reason", "")))
    batch = mean_dict(per_row)
    batch.setdefault("score", batch.get("dea_composite", batch.get("dea_score", 0.0)))
    batch.update(trace_metrics(outputs))
    worst = sorted(reasons, key=lambda x: x[0])[:3]
    feedback = "\n".join(
        f"- task_id={task_id}: score={score:.3f} | {reason}"
        for score, task_id, reason in worst
    )
    return batch, feedback or "DEA batch evaluation completed."


# ---------------------------------------------------------------------------
# Promptfoo batch evaluation
# ---------------------------------------------------------------------------


def extract_promptfoo_records(payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            score = obj.get("score")
            named = obj.get("namedScores")
            if isinstance(score, (int, float)) and (
                isinstance(named, dict)
                or any(k in obj for k in ("pass", "reason", "provider", "vars", "description"))
            ):
                records.append(obj)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)
    return records


def parse_promptfoo_result_json(result_path: Path) -> tuple[dict[str, float], str]:
    payload = json.loads(read_text(result_path))
    records = extract_promptfoo_records(payload)
    if not records:
        top_score = numeric(payload.get("score")) if isinstance(payload, dict) else None
        if top_score is not None:
            return {"promptfoo_score": float(top_score), "score": float(top_score)}, "Promptfoo summary score parsed from top-level JSON."
        raise RuntimeError(
            f"Could not parse Promptfoo row scores from {result_path}. Provide a different result JSON or override the Promptfoo command/template."
        )

    named_scores = []
    reasons = []
    for record in records:
        row_named = dict(record.get("namedScores", {})) if isinstance(record.get("namedScores"), dict) else {}
        row_named.setdefault("promptfoo_score", float(record.get("score") or 0.0))
        row_named.setdefault("promptfoo_pass", 1.0 if record.get("pass") else 0.0)
        named_scores.append(row_named)
        reasons.append((row_named["promptfoo_score"], str(record.get("description") or ""), str(record.get("reason") or "")))

    batch = mean_dict(named_scores)
    batch.setdefault("score", batch.get("promptfoo_score", 0.0))
    worst = sorted(reasons, key=lambda x: x[0])[:3]
    feedback = "\n".join(
        f"- {desc or 'row'}: score={score:.3f} | {reason}"
        for score, desc, reason in worst
    )
    return batch, feedback or "Promptfoo batch evaluation completed."


def run_promptfoo_eval_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    promptfoo_config: Path,
    promptfoo_eval_cmd: str,
    working_dir: Path | None = None,
    artifact_dir: Path | None = None,
    csv_name: str = "candidate_rows.csv",
) -> tuple[dict[str, float], str, dict[str, str]]:
    if not promptfoo_eval_cmd:
        raise RuntimeError("Promptfoo evaluation mode requires --promptfoo-eval-cmd")

    workdir = working_dir or Path.cwd()
    artifact_dir = artifact_dir or Path(tempfile.mkdtemp(prefix="trace_promptfoo_eval_"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifact_dir / csv_name
    result_json = artifact_dir / "promptfoo_result.json"
    write_csv_rows(csv_path, rows)

    command = promptfoo_eval_cmd.format(
        config=shlex.quote(str(promptfoo_config)),
        csv=shlex.quote(str(csv_path)),
        result_json=shlex.quote(str(result_json)),
        workdir=shlex.quote(str(workdir)),
    )
    t0 = time.time()
    proc = run_shell(command, cwd=workdir)
    promptfoo_duration_s = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Promptfoo command failed ({proc.returncode}).\nCOMMAND: {command}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    if not result_json.exists():
        # Some command templates may dump JSON to stdout.
        try:
            maybe_json = json.loads(proc.stdout)
            result_json.write_text(json.dumps(maybe_json, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            raise FileNotFoundError(
                f"Promptfoo command did not create {result_json}. Override --promptfoo-eval-cmd so it writes a JSON result. ({e})"
            )

    score_dict, feedback = parse_promptfoo_result_json(result_json)
    score_dict["promptfoo_duration_s"] = promptfoo_duration_s
    score_dict["promptfoo_duration_score"] = 1.0 / (1.0 + promptfoo_duration_s)
    return score_dict, feedback, {"csv": str(csv_path), "result_json": str(result_json)}


def evaluate_batch_with_promptfoo(
    outputs: Sequence[Mapping[str, Any]],
    *,
    promptfoo_config: Path,
    promptfoo_eval_cmd: str,
    working_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> tuple[dict[str, float], str, dict[str, str]]:
    rows = [dict(item["row"], candidate_answer=item["candidate_answer"]) for item in outputs]
    score_dict, feedback, artifacts = run_promptfoo_eval_rows(
        rows,
        promptfoo_config=promptfoo_config,
        promptfoo_eval_cmd=promptfoo_eval_cmd,
        working_dir=working_dir,
        artifact_dir=artifact_dir,
        csv_name="step2_candidate_rows.csv",
    )
    score_dict.update(trace_metrics(outputs))
    return score_dict, feedback, artifacts


def evaluate_batch_with_promptfoo_live(
    outputs: Sequence[Mapping[str, Any]],
    *,
    promptfoo_config: Path,
    promptfoo_eval_cmd: str,
    working_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> tuple[dict[str, float], str, dict[str, str]]:
    rows = [dict(item["row"]) for item in outputs]
    score_dict, feedback, artifacts = run_promptfoo_eval_rows(
        rows,
        promptfoo_config=promptfoo_config,
        promptfoo_eval_cmd=promptfoo_eval_cmd,
        working_dir=working_dir,
        artifact_dir=artifact_dir,
        csv_name="step3_live_rows.csv",
    )
    score_dict.update(total_duration_metrics(score_dict["promptfoo_duration_s"]))
    return score_dict, feedback, artifacts


# ---------------------------------------------------------------------------
# Custom LLM-as-a-Judge
# ---------------------------------------------------------------------------


class OpenAICompatJudgeClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 600,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        if not self.base_url:
            raise RuntimeError("judge base_url is empty")
        if not self.model:
            raise RuntimeError("judge model is empty")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_text(payload: Any) -> str:
        if isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                message = first.get("message") if isinstance(first, dict) else None
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict):
                                parts.append(str(item.get("text") or item.get("content") or ""))
                            elif item:
                                parts.append(str(item))
                        return "\n".join(p for p in parts if p.strip()).strip()
                if isinstance(first, dict) and first.get("text"):
                    return str(first["text"])
            for key in ("response", "output", "content"):
                if payload.get(key):
                    return str(payload[key])
        raise RuntimeError(f"Could not extract text from judge response: {payload}")

    def complete(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "temperature": temperature,
        }
        body.update(self.extra_body)
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._extract_text(response.json()).strip()


def load_optional_text(path_or_text: str | None) -> str:
    if not path_or_text:
        return ""
    candidate = Path(path_or_text)
    if candidate.exists() and candidate.is_file():
        return read_text(candidate)
    return str(path_or_text)


DEFAULT_LLM_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Always return strict JSON and nothing else. "
    "Score must be a float between 0 and 1."
)

DEFAULT_LLM_JUDGE_PROMPT = """
Evaluate the candidate answer for the task below.

Task:
{task_text}

Reference answer:
{reference_text}

Candidate answer:
{candidate_answer}

Optional context:
{context}

Return strict JSON only with this schema:
{{"score": 0.0, "feedback": "...", "metrics": {{"correctness": 0.0, "faithfulness": 0.0, "coverage": 0.0}}}}

Rules:
- For QA tasks, correctness should dominate.
- For summarization tasks, faithfulness and coverage should dominate.
- If the reference is partial, reward semantically equivalent answers.
- Penalize unsupported inventions, contradiction, missing key facts, and scope drift.
""".strip()


def parse_judge_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()

    print("===== RAW JUDGE OUTPUT =====")
    print(raw)
    print("============================")

    candidates = []

    # 1) JSON dans bloc markdown ```json ... ```
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S | re.I):
        candidates.append(m.group(1))

    # 2) Premier gros objet JSON dans le texte
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        candidates.append(m.group(0))

    last_error = None
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception as e:
            last_error = e

    # Fallback si le juge ne respecte pas le JSON
    print(">>> Fallback parser utilisé <<<")
    return {
        "score": 0.2,
        "feedback": (
            "Improve the system prompt by requiring a structured literature review "
            "with clear sections: context, techniques, evaluation metrics, limitations, "
            "and conclusion. Keep all claims grounded in the provided abstracts and "
            "avoid generic explanations."
        ),
        "metrics": {
            "correctness": 0.2,
            "faithfulness": 0.2,
            "coverage": 0.2,
        },
    }


def build_llm_judge_prompt(row: Mapping[str, Any], candidate_answer: str, template: str) -> str:
    reference_text = resolve_reference_text(row)
    task_text = derive_request_prompt(row)
    mapping = SafeFormatDict(
        {
            **{k: ("" if v is None else str(v)) for k, v in dict(row).items()},
            "candidate_answer": candidate_answer,
            "reference_text": reference_text,
            "task_text": task_text,
            "context": str(row.get("context") or row.get("abstract") or ""),
        }
    )
    return template.format_map(mapping)


def aggregate_llm_judge_batch(
    outputs: Sequence[Mapping[str, Any]],
    *,
    client: OpenAICompatJudgeClient,
    judge_system_prompt: str,
    judge_prompt_template: str,
) -> tuple[dict[str, float], str]:
    per_row = []
    reasons = []
    for item in outputs:
        row = dict(item["row"])
        prompt = build_llm_judge_prompt(row, str(item["candidate_answer"]), judge_prompt_template)
        raw = client.complete(system_prompt=judge_system_prompt, user_prompt=prompt, temperature=0.0)
        parsed = parse_judge_json(raw)
        metrics = dict(parsed.get("metrics", {})) if isinstance(parsed.get("metrics"), dict) else {}
        metrics = {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        metrics.setdefault("judge_score", float(parsed.get("score") or 0.0))
        metrics["latency_s"] = float(item.get("latency_s") or 0.0)
        per_row.append(metrics)
        reasons.append((metrics["judge_score"], row.get("task_id"), str(parsed.get("feedback") or "")))
    batch = mean_dict(per_row)
    batch.setdefault("score", batch.get("judge_score", 0.0))
    batch.update(trace_metrics(outputs))
    worst = sorted(reasons, key=lambda x: x[0])[:3]
    feedback = "\n".join(
        f"- task_id={task_id}: score={score:.3f} | {reason}"
        for score, task_id, reason in worst
    )
    return batch, feedback or "LLM judge batch evaluation completed."


# ---------------------------------------------------------------------------
# Batch guides
# ---------------------------------------------------------------------------


class BatchEvaluationGuide(Guide):
    def __init__(
        self,
        *,
        mode: str,
        use_dea_judge: bool = False,
        promptfoo_config: str = "",
        promptfoo_eval_cmd: str = "",
        promptfoo_workdir: str = "",
        judge_base_url: str = "",
        judge_api_key: str = "",
        judge_model: str = "",
        judge_timeout_seconds: int = 600,
        judge_extra_body: Mapping[str, Any] | None = None,
        judge_system_prompt: str = DEFAULT_LLM_JUDGE_SYSTEM_PROMPT,
        judge_prompt_template: str = DEFAULT_LLM_JUDGE_PROMPT,
        artifact_dir: str = "",
    ) -> None:
        self.mode = mode
        self.use_dea_judge = use_dea_judge
        self.promptfoo_config = promptfoo_config
        self.promptfoo_eval_cmd = promptfoo_eval_cmd
        self.promptfoo_workdir = promptfoo_workdir
        self.artifact_dir = artifact_dir
        self.last_artifacts: dict[str, str] = {}
        self._judge_client = None
        self.judge_system_prompt = judge_system_prompt
        self.judge_prompt_template = judge_prompt_template
        if mode == "llm_judge":
            self._judge_client = OpenAICompatJudgeClient(
                base_url=judge_base_url,
                api_key=judge_api_key,
                model=judge_model,
                timeout=judge_timeout_seconds,
                extra_body=judge_extra_body or {},
            )

    def _evaluate_outputs(self, outputs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], str]:
        if self.mode == "dea":
            return aggregate_dea_batch(outputs, use_dea_judge=self.use_dea_judge)
        if self.mode == "promptfoo":
            score_dict, feedback, _artifacts = evaluate_batch_with_promptfoo(
                outputs,
                promptfoo_config=Path(self.promptfoo_config),
                promptfoo_eval_cmd=self.promptfoo_eval_cmd,
                working_dir=Path(self.promptfoo_workdir) if self.promptfoo_workdir else Path.cwd(),
                artifact_dir=Path(self.artifact_dir) if self.artifact_dir else None,
            )
            self.last_artifacts = _artifacts
            return score_dict, feedback
        if self.mode == "promptfoo_step3":
            score_dict, feedback, _artifacts = evaluate_batch_with_promptfoo_live(
                outputs,
                promptfoo_config=Path(self.promptfoo_config),
                promptfoo_eval_cmd=self.promptfoo_eval_cmd,
                working_dir=Path(self.promptfoo_workdir) if self.promptfoo_workdir else Path.cwd(),
                artifact_dir=Path(self.artifact_dir) if self.artifact_dir else None,
            )
            self.last_artifacts = _artifacts
            return score_dict, feedback
        if self.mode == "llm_judge":
            return aggregate_llm_judge_batch(
                outputs,
                client=self._judge_client,
                judge_system_prompt=self.judge_system_prompt,
                judge_prompt_template=self.judge_prompt_template,
            )
        raise ValueError(f"Unknown evaluation mode: {self.mode}")

    def get_feedback(self, query: list[dict[str, Any]], response: list[dict[str, Any]], reference: Any = None, **kwargs):
        score_dict, feedback = self._evaluate_outputs(response)
        return score_dict, feedback

    def get_score_dict(self, query: list[dict[str, Any]], response: list[dict[str, Any]], reference: Any = None, **kwargs):
        score_dict, _feedback = self._evaluate_outputs(response)
        return score_dict


# ---------------------------------------------------------------------------
# Trace agents
# ---------------------------------------------------------------------------


@trace.model
class OpenWebUISummarizerAgent:
    """Optimize per-call tool/pipe parameters through the common bridge.

    This agent assumes the selected OpenWebUI target is a pipe/model that routes
    to the tool under test. The bridge model id is therefore usually a pipe id
    such as ``summarizer---kohaku-OR``.
    """

    def __init__(
        self,
        initial_param_json: str,
        bridge_url: str,
        target_model_id: str,
        strict_exposed_keys: bool = False,
        include_bridge_trace: bool = True,
        generation_mode: str = "bridge",
        bridge_timeout_seconds: int = 1800,
    ) -> None:
        self.bridge_url = bridge_url
        self.target_model_id = target_model_id
        self.strict_exposed_keys = strict_exposed_keys
        self.include_bridge_trace = include_bridge_trace
        self.generation_mode = generation_mode if generation_mode in GENERATION_MODES else "bridge"
        self.bridge_timeout_seconds = bridge_timeout_seconds
        self.param_json = trace.node(initial_param_json, trainable=True)

    @trace.bundle(trainable=True)
    def build_param_json(self, raw_json: str) -> str:
        try:
            data = json.loads(raw_json)
        except Exception:
            data = {}
        safe = sanitize_tool_params(data)
        return json.dumps(safe, ensure_ascii=False)

    @trace.bundle()
    def call_bridge(self, rows: list[dict[str, Any]], param_json: str) -> list[dict[str, Any]]:
        params = json.loads(param_json)

        def _row_transformer(row: Mapping[str, Any]) -> dict[str, Any]:
            active_row, warnings = merge_tool_params_into_row(
                row,
                params,
                target_model_id=self.target_model_id,
                strict_exposed_keys=self.strict_exposed_keys,
            )
            if warnings:
                active_row["trace_tool_param_warnings"] = " | ".join(warnings)
            active_row["openwebui_target_kind"] = "tool"
            active_row["openwebui_include_trace"] = str(self.include_bridge_trace).lower()
            return active_row

        if self.generation_mode == "prepared_rows":
            return prepare_batch_rows(rows, row_transformer=_row_transformer)
        return generate_batch_via_bridge(rows, bridge_url=self.bridge_url, row_transformer=_row_transformer, timeout_seconds=self.bridge_timeout_seconds)

    def __call__(self, rows: list[dict[str, Any]]):
        safe_json = self.build_param_json(self.param_json)
        return self.call_bridge(rows, safe_json)


@trace.model
class OpenWebUIModel:
    """Optimize a model/workspace-model-like request layer through the same bridge.

    The selected target model id can be a workspace model or a plain base model.
    The trainable system prompt is applied with the configured prompt mode.
    Current bridge-compatible request-time parameters are temperature/top_p/
    max_tokens. Extra fields are still written to the row for future bridge
    revisions but may be ignored by the current bridge.
    """

    def __init__(
        self,
        initial_system_prompt: str,
        initial_param_json: str,
        bridge_url: str,
        target_model_id: str,
        prompt_mode: str = "wrapped",
        keep_existing_tool_fields: bool = False,
        include_bridge_trace: bool = True,
        generation_mode: str = "bridge",
        bridge_timeout_seconds: int = 1800,
    ) -> None:
        self.bridge_url = bridge_url
        self.target_model_id = target_model_id
        self.prompt_mode = prompt_mode if prompt_mode in MODEL_PROMPT_MODES else "wrapped"
        self.keep_existing_tool_fields = keep_existing_tool_fields
        self.include_bridge_trace = include_bridge_trace
        self.generation_mode = generation_mode if generation_mode in GENERATION_MODES else "bridge"
        self.bridge_timeout_seconds = bridge_timeout_seconds
        self.system_prompt = trace.node(initial_system_prompt, trainable=True)
        self.param_json = trace.node(initial_param_json, trainable=True)

    @trace.bundle(trainable=True)
    def build_system_prompt(self, raw_prompt: str) -> str:
        return sanitize_system_prompt(raw_prompt)

    @trace.bundle(trainable=True)
    def build_param_json(self, raw_json: str) -> str:
        try:
            data = json.loads(raw_json)
        except Exception:
            data = {}
        safe = sanitize_model_params(data)
        return json.dumps(safe, ensure_ascii=False)

    @trace.bundle()
    def call_bridge(self, rows: list[dict[str, Any]], system_prompt: str, param_json: str) -> list[dict[str, Any]]:
        params = json.loads(param_json)

        def _row_transformer(row: Mapping[str, Any]) -> dict[str, Any]:
            active_row = apply_model_params_to_row(
                row,
                system_prompt,
                params,
                target_model_id=self.target_model_id,
                prompt_mode=self.prompt_mode,
                keep_existing_tool_fields=self.keep_existing_tool_fields,
            )
            active_row["openwebui_include_trace"] = str(self.include_bridge_trace).lower()
            return active_row

        if self.generation_mode == "prepared_rows":
            return prepare_batch_rows(rows, row_transformer=_row_transformer)
        return generate_batch_via_bridge(rows, bridge_url=self.bridge_url, row_transformer=_row_transformer, timeout_seconds=self.bridge_timeout_seconds)

    def __call__(self, rows: list[dict[str, Any]]):
        safe_prompt = self.build_system_prompt(self.system_prompt)
        safe_json = self.build_param_json(self.param_json)
        return self.call_bridge(rows, safe_prompt, safe_json)


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------


def scalarize(score_dict: Mapping[str, float], config: ObjectiveConfig) -> float:
    minimized = apply_minimize(dict(score_dict), config.minimize)
    return float(weighted_scalarize(minimized, config.weights, config.missing_value))


def build_objective_config(args: argparse.Namespace) -> ObjectiveConfig:
    """Build an objective that always balances answer quality and execution cost."""
    raw_weights = parse_jsonish(args.weights, default=None)
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError("--weights must be a non-empty JSON object")

    weights: dict[str, float] = {}
    for key, value in raw_weights.items():
        try:
            weight = float(value)
        except Exception as exc:
            raise ValueError(f"Objective weight for `{key}` must be numeric") from exc
        if weight < 0:
            raise ValueError(f"Objective weight for `{key}` must be non-negative")
        weights[str(key)] = weight

    # Keep quality and runtime visible in the ObjectiveConfig even when callers
    # pass a narrow custom weights JSON for one DEA sub-metric.
    if not QUALITY_OBJECTIVE_KEYS.intersection(weights):
        weights["score"] = DEFAULT_OBJECTIVE_WEIGHTS["score"]
    if not RUNTIME_OBJECTIVE_SCORE_KEYS.intersection(weights):
        weights["execution_duration_score"] = DEFAULT_OBJECTIVE_WEIGHTS["execution_duration_score"]

    minimize = set(str(item) for item in (args.minimize or []))
    minimize.update(DEFAULT_MINIMIZE_METRICS)
    return ObjectiveConfig(
        mode=args.objective_mode,
        weights=weights,
        minimize=frozenset(minimize),
        tie_break="weighted",
        scalarize_dict="weighted",
        score_key="score",
    )


def objective_config_to_dict(config: ObjectiveConfig) -> dict[str, Any]:
    """Serialize objective settings for result artifacts."""
    return {
        "mode": config.mode,
        "weights": dict(config.weights),
        "minimize": sorted(config.minimize),
        "tie_break": config.tie_break,
        "scalarize_dict": config.scalarize_dict,
        "score_key": config.score_key,
    }


def dominates_vector(a: Mapping[str, float], b: Mapping[str, float], minimize: frozenset[str]) -> bool:
    a2 = apply_minimize(dict(a), minimize)
    b2 = apply_minimize(dict(b), minimize)
    metrics = sorted(set(a2.keys()) | set(b2.keys()))
    at_least_one_better = False
    for metric in metrics:
        av = a2.get(metric, float("-inf"))
        bv = b2.get(metric, float("-inf"))
        if av < bv:
            return False
        if av > bv:
            at_least_one_better = True
    return at_least_one_better


def update_pareto_front(front: list[dict[str, Any]], snapshot: dict[str, Any], config: ObjectiveConfig) -> list[dict[str, Any]]:
    kept = []
    dominated = False
    for existing in front:
        if dominates_vector(existing["mean_scores"], snapshot["mean_scores"], config.minimize):
            dominated = True
            kept.append(existing)
        elif dominates_vector(snapshot["mean_scores"], existing["mean_scores"], config.minimize):
            continue
        else:
            kept.append(existing)
    if not dominated:
        kept.append(snapshot)
    return kept


def numeric_score_dict(score_dict: Mapping[str, Any]) -> dict[str, float]:
    """Keep only finite numeric score values for scalarization and artifacts."""
    out: dict[str, float] = {}
    for key, value in score_dict.items():
        if isinstance(value, bool):
            out[str(key)] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[str(key)] = float(value)
    return out


def current_agent_state(agent: Any) -> dict[str, str]:
    """Return trainable parameter state in a stable artifact shape."""
    param_json = str(getattr(agent, "param_json", None).data) if hasattr(agent, "param_json") else ""
    system_prompt = str(getattr(agent, "system_prompt", None).data) if hasattr(agent, "system_prompt") else ""
    return {
        "param_json": param_json,
        "tool_param_json": param_json if isinstance(agent, OpenWebUISummarizerAgent) else "",
        "model_param_json": param_json if isinstance(agent, OpenWebUIModel) else "",
        "system_prompt": system_prompt,
    }


def output_trace_records(outputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract compact per-row trace records without duplicating full answers."""
    records: list[dict[str, Any]] = []
    for item in outputs:
        raw_response = item.get("raw_response") if isinstance(item.get("raw_response"), dict) else {}
        records.append(
            {
                "task_id": item.get("task_id"),
                "row_index": item.get("row_index"),
                "latency_s": item.get("latency_s"),
                "answer_chars": len(str(item.get("candidate_answer") or "")),
                "trace": raw_response.get("trace") if isinstance(raw_response, dict) else {},
            }
        )
    return records


def save_step_artifacts(
    artifact_dir: Path,
    snapshot: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
) -> None:
    """Persist one optimization step with candidates and bridge traces."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    step = int(snapshot.get("global_step") or 0)
    prefix = f"step_{step:04d}_epoch_{int(snapshot.get('epoch') or 0):03d}_batch_{int(snapshot.get('batch_index') or 0):03d}"
    write_csv_rows(
        artifact_dir / f"{prefix}.candidates.csv",
        [dict(item["row"], candidate_answer=item["candidate_answer"]) for item in outputs],
    )
    write_text(artifact_dir / f"{prefix}.traces.json", json.dumps(output_trace_records(outputs), indent=2, ensure_ascii=False))
    write_text(artifact_dir / f"{prefix}.summary.json", json.dumps(snapshot, indent=2, ensure_ascii=False))


class NoOpOptimizer:
    """Optimizer adapter for generation/evaluation calibration runs."""

    def zero_feedback(self) -> None:
        return None

    def backward(self, target: Any, feedback: Any) -> None:
        return None

    def step(self) -> dict[str, Any]:
        return {}


class TraceBatchTrainer(Trainer):
    """Train a Trace agent on row batches and keep PromptFoo/DEA feedback artifacts."""

    def __init__(
        self,
        agent: Any,
        optimizer: Any,
        *,
        objective_config: ObjectiveConfig,
        artifact_dir: Path | None = None,
        optimize_target: str = "",
        eval_backend: str = "",
        optimization_method: str = "",
        optimizer_mode: str = "optoprime",
        logger: Any = None,
    ) -> None:
        super().__init__(agent, logger=logger)
        self.optimizer = optimizer
        self.objective_config = objective_config
        self.artifact_dir = artifact_dir
        self.optimize_target = optimize_target
        self.eval_backend = eval_backend
        self.optimization_method = optimization_method
        self.optimizer_mode = optimizer_mode
        self.history: list[dict[str, Any]] = []
        self.pareto_front: list[dict[str, Any]] = []
        self.best_weighted: dict[str, Any] | None = None

    def _run_batch(
        self,
        *,
        guide: BatchEvaluationGuide,
        rows: list[dict[str, Any]],
        epoch: int,
        batch_index: int,
        global_step: int,
    ) -> dict[str, Any]:
        t0 = time.time()
        print("INITIAL PROMPT =", current_agent_state(self.agent)["system_prompt"])
        try:
            target = self.agent(rows)
            outputs = target.data
            if not isinstance(outputs, list):
                raise RuntimeError(f"Agent returned {type(outputs).__name__}, expected list of generated rows")
            score_dict_raw, feedback = guide.get_feedback(rows, outputs, rows)
            score_dict = numeric_score_dict(score_dict_raw)
            # Full runtime covers generation plus DEA/PromptFoo/judge work.
            score_dict = merge_step_runtime_metrics(score_dict, time.time() - t0)
            scalar_score = scalarize(score_dict, self.objective_config)
        except trace.ExecutionError as e:
            target = e.exception_node
            outputs = []
            score_dict = numeric_score_dict(merge_step_runtime_metrics({}, time.time() - t0))
            score_dict.setdefault("score", 0.0)
            scalar_score = 0.0
            feedback = target.create_feedback("full")

        self.optimizer.zero_feedback()
        self.optimizer.backward(target, feedback)

        print("\n========== OPTIMIZER DEBUG ==========")
        print("BEFORE PROMPT =", current_agent_state(self.agent)["system_prompt"])

        step_result = self.optimizer.step()
        print("STEP RESULT =", step_result)

        print("AFTER PROMPT =", current_agent_state(self.agent)["system_prompt"])
        print("=====================================\n")

        snapshot = {
            "epoch": epoch,
            "batch_index": batch_index,
            "global_step": global_step,
            "batch_size": len(rows),
            "batch_task_ids": task_ids(rows),
            "optimize_target": self.optimize_target,
            "eval_backend": self.eval_backend,
            "optimization_method": self.optimization_method,
            "optimizer_mode": self.optimizer_mode,
            "objective_config": objective_config_to_dict(self.objective_config),
            "mean_scores": score_dict,
            "scalar_objective": scalar_score,
            "feedback": trim_text(str(feedback), limit=4000),
            "current_state": current_agent_state(self.agent),
            "evaluation_artifacts": dict(getattr(guide, "last_artifacts", {}) or {}),
        }
        self.history.append(snapshot)
        if self.artifact_dir is not None:
            save_step_artifacts(self.artifact_dir, snapshot, outputs)
        print(json.dumps(snapshot, ensure_ascii=False))

        if self.best_weighted is None or scalar_score > self.best_weighted["scalar_objective"]:
            self.best_weighted = snapshot
        if self.objective_config.mode == "pareto":
            self.pareto_front = update_pareto_front(self.pareto_front, snapshot, self.objective_config)
        return snapshot

    def train(
        self,
        guide: BatchEvaluationGuide,
        train_dataset: Mapping[str, Sequence[list[dict[str, Any]]]],
        num_threads: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run optimization over deterministic row batches for each epoch."""
        num_epochs = int(kwargs.get("num_epochs", 1))
        shuffle_batches = bool(kwargs.get("shuffle_batches", False))
        global_step = 0
        for epoch in range(num_epochs):
            loader = DataLoader(dict(train_dataset), batch_size=1, randomize=shuffle_batches, shuffle=shuffle_batches)
            for batch_index, (xs, _infos) in enumerate(loader):
                rows = [dict(row) for row in xs[0]]
                self._run_batch(
                    guide=guide,
                    rows=rows,
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=global_step,
                )
                global_step += 1
        return {
            "objective_config": objective_config_to_dict(self.objective_config),
            "best_weighted": self.best_weighted,
            "pareto_front": self.pareto_front if self.objective_config.mode == "pareto" else [],
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental Trace/NewTrace optimizer for OpenWebUI tool/model targets")

    # Dataset / CSV
    parser.add_argument("--input-csv", required=True)
    parser.add_argument(
        "--build-input-csv-cmd",
        default="",
        help="Optional shell command used to build the CSV when --input-csv is missing. Use {output_csv} as placeholder.",
    )
    parser.add_argument("--force-build-input-csv", action="store_true")
    parser.add_argument("--max-rows", type=int, default=5)

    # Target selection
    parser.add_argument(
        "--optimization-method",
        choices=sorted(OPTIMIZATION_METHODS),
        default="",
        help=(
            "High-level method preset. step2_promptfoo is the Offline Process: generate through the bridge, then evaluate with PromptFoo. "
            "step3_promptfoo is the Online Process: let PromptFoo call the bridge live. direct_dea_judge uses native DEA with its LLM judge."
        ),
    )
    parser.add_argument("--optimize-target", choices=["tool", "model"], default="tool")
    parser.add_argument(
        "--target-model-id",
        default=os.environ.get("OPENWEBUI_PIPE_MODEL", ""),
        help="OpenWebUI model id to hit through the bridge. For tool mode this is usually a pipe model id. For model mode this can be a workspace model or a base model id.",
    )
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8001/generate")
    parser.add_argument("--bridge-timeout-seconds", type=int, default=int(os.environ.get("BRIDGE_REQUEST_TIMEOUT_SECONDS", "1800")))
    parser.add_argument("--generation-mode", choices=sorted(GENERATION_MODES), default="bridge")
    parser.add_argument("--disable-bridge-trace", action="store_true", help="Do not request trace/timing metadata from the bridge.")

    # Tool target init
    parser.add_argument("--initial-tool-params-json", default="")
    parser.add_argument("--strict-exposed-tool-params", action="store_true")

    # Model target init
    parser.add_argument("--initial-system-prompt", default="")
    parser.add_argument("--initial-system-prompt-file", default="")
    parser.add_argument("--initial-model-params-json", default="")
    parser.add_argument("--model-prompt-mode", choices=sorted(MODEL_PROMPT_MODES), default="wrapped")
    parser.add_argument("--model-keep-existing-tool-fields", action="store_true")

    # Evaluation
    parser.add_argument("--eval-backend", choices=["dea", "promptfoo", "promptfoo_step3", "llm_judge"], default="dea")
    parser.add_argument("--use-dea-judge", action="store_true")
    parser.add_argument("--promptfoo-config", default="")
    parser.add_argument(
        "--promptfoo-eval-cmd",
        default="",
        help=(
            "Command template to evaluate a generated CSV with Promptfoo. Placeholders: {config}, {csv}, {result_json}, {workdir}. "
            "Example: promptfoo eval -c {config} -t {csv} --output {result_json} --no-progress-bar"
        ),
    )
    parser.add_argument("--promptfoo-workdir", default="")
    parser.add_argument("--judge-base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--judge-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--judge-model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--judge-timeout-seconds", type=int, default=int(os.environ.get("DEA_JUDGE_TIMEOUT_SECONDS", "600")))
    parser.add_argument("--judge-extra-body-json", default=os.environ.get("DEA_JUDGE_EXTRA_BODY_JSON", ""))
    parser.add_argument("--judge-system-prompt", default="")
    parser.add_argument("--judge-system-prompt-file", default="")
    parser.add_argument("--judge-prompt-template", default="")
    parser.add_argument("--judge-prompt-file", default="")

    # Optimization / objective
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=3, help="Number of examples per optimization feedback batch.")
    parser.add_argument("--shuffle-batches", action="store_true")
    parser.add_argument("--optimizer-mode", choices=["optoprime", "noop"], default="optoprime")
    parser.add_argument("--optimizer-max-tokens", type=int, default=1024, help="Max tokens for OptoPrime optimizer suggestions.")
    parser.add_argument("--objective-mode", choices=["weighted", "pareto"], default="weighted")
    parser.add_argument(
        "--weights",
        default=json.dumps(DEFAULT_OBJECTIVE_WEIGHTS),
        help=(
            "JSON dict of objective weights. Defaults include quality score plus normalized runtime costs. "
            "Use detailed DEA keys such as plan/content/resources/length_alignment when needed."
        ),
    )
    parser.add_argument("--minimize", nargs="*", default=list(DEFAULT_MINIMIZE_METRICS))

    # Artifacts
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--output-json", default="")
    return parser


def apply_optimization_method(args: argparse.Namespace) -> None:
    """Map high-level experiment methods to generation and evaluation modes."""
    if args.optimization_method == "direct_dea":
        args.eval_backend = "dea"
        args.use_dea_judge = False
        args.generation_mode = "bridge"
    elif args.optimization_method == "direct_dea_judge":
        args.eval_backend = "dea"
        args.use_dea_judge = True
        args.generation_mode = "bridge"
    elif args.optimization_method == "direct_llm_judge":
        args.eval_backend = "llm_judge"
        args.generation_mode = "bridge"
    elif args.optimization_method == "step2_promptfoo":
        args.eval_backend = "promptfoo"
        args.generation_mode = "bridge"
    elif args.optimization_method == "step3_promptfoo":
        args.eval_backend = "promptfoo_step3"
        args.generation_mode = "prepared_rows"
    else:
        if args.eval_backend == "promptfoo_step3":
            args.optimization_method = "step3_promptfoo"
            args.generation_mode = "prepared_rows"
        elif args.eval_backend == "promptfoo":
            args.optimization_method = "step2_promptfoo"
        elif args.eval_backend == "llm_judge":
            args.optimization_method = "direct_llm_judge"
        else:
            args.optimization_method = "direct_dea_judge" if args.use_dea_judge else "direct_dea"


def validate_runtime_args(args: argparse.Namespace) -> None:
    """Fail early for incompatible method/runtime settings."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.judge_timeout_seconds <= 0:
        raise ValueError("--judge-timeout-seconds must be positive")
    if args.optimizer_max_tokens <= 0:
        raise ValueError("--optimizer-max-tokens must be positive")
    if args.bridge_timeout_seconds <= 0:
        raise ValueError("--bridge-timeout-seconds must be positive")
    if args.eval_backend in {"promptfoo", "promptfoo_step3"}:
        if not args.promptfoo_config:
            raise ValueError("--promptfoo-config is required for PromptFoo evaluation modes")
        if not args.promptfoo_eval_cmd:
            raise ValueError("--promptfoo-eval-cmd is required for PromptFoo evaluation modes")
    if args.eval_backend == "llm_judge" and not args.judge_model:
        raise ValueError("--judge-model or OPENAI_MODEL is required for --eval-backend llm_judge")
    if args.generation_mode == "prepared_rows" and args.eval_backend != "promptfoo_step3":
        raise ValueError("prepared_rows generation mode is only valid with step3_promptfoo/promptfoo_step3")


def build_training_dataset(rows: Sequence[Mapping[str, Any]], batch_size: int) -> dict[str, list[list[dict[str, Any]]]]:
    """Build the Trace trainer dataset where one sample is one row batch."""
    batches = row_batches(rows, batch_size)
    return {"inputs": batches, "infos": batches}


def build_optimizer(args: argparse.Namespace, agent: Any) -> Any:
    """Create the configured optimizer adapter for training or calibration."""
    if args.optimizer_mode == "noop":
        return NoOpOptimizer()
    return OptoPrime(agent.parameters(), max_tokens=args.optimizer_max_tokens)


def default_tool_params_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0] if rows else {}
    return {
        "algorithm": str(first.get("algorithm") or "kohaku"),
        "target_length": str(first.get("target_length") or "long"),
        "structure": str(first.get("structure") or "thematic"),
        "max_context_chars_per_llm_call": 24000,
        "max_final_summary_chars": 4000,
        "enable_map_reduce": False,
        "kohaku_n_axes": 6,
        "kohaku_top_sections_per_axis": 3,
        "kohaku_top_paragraphs_per_axis": 8,
        "kohaku_ensemble_size": 3,
        "kohaku_ensemble_temperature": 0.4,
    }


def default_model_params() -> dict[str, Any]:
    return {
        "generation_temperature": 0.2,
        "generation_top_p": 0.95,
        "generation_max_tokens": 4096,
    }


def build_agent(args, rows: list[dict[str, Any]]):
    if args.optimize_target == "tool":
        initial = parse_jsonish(args.initial_tool_params_json, default={}) or default_tool_params_from_rows(rows)
        return OpenWebUISummarizerAgent(
            json.dumps(sanitize_tool_params(initial), ensure_ascii=False),
            args.bridge_url,
            args.target_model_id,
            strict_exposed_keys=args.strict_exposed_tool_params,
            include_bridge_trace=not args.disable_bridge_trace,
            generation_mode=args.generation_mode,
            bridge_timeout_seconds=args.bridge_timeout_seconds,
        )

    system_prompt = load_optional_text(args.initial_system_prompt_file or args.initial_system_prompt)
    if not system_prompt:
        system_prompt = "You are a precise assistant. Follow the task carefully, stay faithful to the provided evidence, and avoid unsupported claims."
    model_params = parse_jsonish(args.initial_model_params_json, default={}) or default_model_params()
    return OpenWebUIModel(
        initial_system_prompt=sanitize_system_prompt(system_prompt),
        initial_param_json=json.dumps(sanitize_model_params(model_params), ensure_ascii=False),
        bridge_url=args.bridge_url,
        target_model_id=args.target_model_id,
        prompt_mode=args.model_prompt_mode,
        keep_existing_tool_fields=args.model_keep_existing_tool_fields,
        include_bridge_trace=not args.disable_bridge_trace,
        generation_mode=args.generation_mode,
        bridge_timeout_seconds=args.bridge_timeout_seconds,
    )


def build_guide(args) -> BatchEvaluationGuide:
    judge_system = load_optional_text(args.judge_system_prompt_file or args.judge_system_prompt) or DEFAULT_LLM_JUDGE_SYSTEM_PROMPT
    judge_template = load_optional_text(args.judge_prompt_file or args.judge_prompt_template) or DEFAULT_LLM_JUDGE_PROMPT
    judge_extra_body = parse_jsonish(args.judge_extra_body_json, default={}) or {}
    if not isinstance(judge_extra_body, dict):
        raise ValueError("--judge-extra-body-json must be a JSON object")
    return BatchEvaluationGuide(
        mode=args.eval_backend,
        use_dea_judge=args.use_dea_judge,
        promptfoo_config=args.promptfoo_config,
        promptfoo_eval_cmd=args.promptfoo_eval_cmd,
        promptfoo_workdir=args.promptfoo_workdir,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model,
        judge_timeout_seconds=args.judge_timeout_seconds,
        judge_extra_body=judge_extra_body,
        judge_system_prompt=judge_system,
        judge_prompt_template=judge_template,
        artifact_dir=args.artifact_dir,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_optimization_method(args)
    validate_runtime_args(args)

    input_csv = Path(args.input_csv).resolve()
    maybe_build_input_csv(input_csv, args.build_input_csv_cmd or None, force=args.force_build_input_csv)
    rows = [normalize_generation_row(r) for r in read_csv_rows(input_csv)[: args.max_rows]]
    if not rows:
        raise SystemExit("No rows found in input CSV")
    if not args.target_model_id:
        raise SystemExit("--target-model-id is required (pipe model id or workspace/base model id)")

    objective_config = build_objective_config(args)

    artifact_dir = Path(args.artifact_dir).resolve() if args.artifact_dir else None
    agent = build_agent(args, rows)
    optimizer = build_optimizer(args, agent)
    guide = build_guide(args)
    trainer = TraceBatchTrainer(
        agent,
        optimizer,
        objective_config=objective_config,
        artifact_dir=artifact_dir,
        optimize_target=args.optimize_target,
        eval_backend=args.eval_backend,
        optimization_method=args.optimization_method,
        optimizer_mode=args.optimizer_mode,
    )
    payload = trainer.train(
        guide,
        build_training_dataset(rows, args.batch_size),
        num_epochs=args.num_epochs,
        shuffle_batches=args.shuffle_batches,
    )
    if args.output_json:
        write_text(Path(args.output_json), json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
