from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/redteam/tasks.jsonl")
    parser.add_argument("--dataset-dir", default="datasets/redteam")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = ROOT / args.input
    dataset_dir = EVAL_DIR / args.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Fichier introuvable : {input_path}")

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    rows = rows[: args.n]

    if not rows:
        raise SystemExit("Aucune ligne redteam trouvée.")

    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "task_id": row["task_id"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "instruction": row["instruction"],
                "request_prompt": row["request_prompt"],
                "gold_summary": row["gold_summary"],
                "candidate_answer": "",
                "min_chars": 20,
                "max_chars": 1200,
                "source_paths_json": "[]",
                "kb_ids_json": "[]",
                "source_document_mode": "none",
                "openwebui_pipe_model": "",
                "tool_parameters_json": "{}",
                "summarizer_model_id": "",
                "algorithm": "",
                "target_length": "",
                "structure": "",
                "generation_temperature": 0,
                "generation_top_p": 1,
                "generation_max_tokens": 700,
                "openwebui_extra_instructions": "",
                "openwebui_model_params_json": "{}",
            }
        )

    fieldnames = list(csv_rows[0].keys())

    for filename in ["step2_input.csv"]:
        output_path = dataset_dir / filename

        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"{output_path} existe déjà. Ajoute --overwrite pour remplacer.")

        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"Wrote {len(csv_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()