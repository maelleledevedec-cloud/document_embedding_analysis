from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import requests


def _extract_answer(data: Any) -> str:
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content"):
                    return str(message["content"])
                if first.get("text"):
                    return str(first["text"])

        for key in ["output", "response", "content", "answer", "text"]:
            if data.get(key):
                return str(data[key])

        return json.dumps(data, ensure_ascii=False)

    return str(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/redteam/step2_input.csv")
    parser.add_argument("--output", default="datasets/redteam/step2_output.csv")
    parser.add_argument(
        "--bridge-url",
        default=os.getenv("OWUI_BRIDGE_URL", "http://127.0.0.1:8003/generate"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENWEBUI_PIPE_MODEL", "summarizer---kohaku"),
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Fichier introuvable : {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("CSV vide.")

    output_rows = []

    for index, row in enumerate(rows, start=1):
        task_id = row.get("task_id", f"row_{index}")
        request_prompt = row.get("request_prompt", "")

        model = row.get("openwebui_pipe_model") or args.model

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": request_prompt,
                }
            ],
            "temperature": float(row.get("generation_temperature") or 0),
            "top_p": float(row.get("generation_top_p") or 1),
            "max_tokens": int(float(row.get("generation_max_tokens") or 700)),
        }

        print(f"[{index}/{len(rows)}] {task_id} -> {args.bridge_url}")

        try:
            response = requests.post(
                args.bridge_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            data = response.json()
            candidate_answer = _extract_answer(data)
        except Exception as exc:
            candidate_answer = f"[GENERATION_ERROR] {type(exc).__name__}: {exc}"

        row["candidate_answer"] = candidate_answer
        output_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(output_rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()