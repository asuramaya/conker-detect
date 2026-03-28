#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


API_URL = "https://api.github.com/repos/openai/parameter-golf/pulls?state=all&sort=created&direction=desc&per_page={limit}"

PATCH_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
ARTIFACT_EXTS = (".pt", ".pth", ".bin", ".npz", ".safetensors", ".ckpt")
def patch_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for match in PATCH_FILE_RE.finditer(patch_text):
        path = match.group(1)
        if path != "/dev/null":
            files.append(path)
    return files


def classify_pr(pr: dict[str, Any], patch_text: str, files: list[str]) -> dict[str, Any]:
    lower_patch = patch_text.lower()
    lower_title = str(pr.get("title", "")).lower()
    record_dirs = sorted({path.split("/", 3)[2] for path in files if path.startswith("records/") and path.count("/") >= 3})
    adds_submission_json = any(path.endswith("/submission.json") for path in files)
    adds_train_gpt = any(path.endswith("/train_gpt.py") for path in files)
    touches_core = any(path in {"train_gpt.py", "train_gpt_mlx.py"} for path in files)
    docs_only = bool(files) and all(
        path.endswith((".md", ".txt", ".json", ".log")) or path.startswith(".github/")
        for path in files
    )
    has_artifact = any(path.endswith(ARTIFACT_EXTS) for path in files)
    category = "other"
    if adds_submission_json and adds_train_gpt:
        category = "record_submission"
    elif touches_core:
        category = "core_code"
    elif docs_only:
        category = "docs_or_logs"
    elif record_dirs:
        category = "record_misc"

    ttt_signal = "ttt" in lower_title or "test-time" in lower_patch or "ttt_" in lower_patch or "ttt" in lower_patch
    score_first_signal = "score-first" in lower_title or "score-first" in lower_patch
    inference_mode_signal = "inference_mode" in lower_patch
    optimizer_in_eval_signal = (
        "optimizer.step()" in patch_text
        or ".step()" in patch_text and "eval" in lower_patch and "optimizer" in lower_patch
    )
    replayable_runtime = category == "record_submission" and has_artifact
    triage_ready = category in {"record_submission", "core_code", "record_misc"}

    return {
        "number": pr["number"],
        "state": pr["state"],
        "created_at": pr["created_at"],
        "title": pr["title"],
        "author": pr["user"]["login"],
        "html_url": pr["html_url"],
        "category": category,
        "record_dirs": record_dirs,
        "file_count": len(files),
        "adds_submission_json": adds_submission_json,
        "adds_train_gpt": adds_train_gpt,
        "touches_core": touches_core,
        "has_artifact_blob": has_artifact,
        "ttt_signal": ttt_signal,
        "score_first_signal": score_first_signal,
        "inference_mode_signal": inference_mode_signal,
        "optimizer_in_eval_signal": optimizer_in_eval_signal,
        "triage_ready": triage_ready,
        "runtime_replay_ready": replayable_runtime,
        "sample_files": files[:12],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["category"] for row in rows)
    return {
        "pr_count": len(rows),
        "category_counts": dict(sorted(counts.items())),
        "ttt_signal_count": sum(bool(row["ttt_signal"]) for row in rows),
        "score_first_signal_count": sum(bool(row["score_first_signal"]) for row in rows),
        "inference_mode_signal_count": sum(bool(row["inference_mode_signal"]) for row in rows),
        "runtime_replay_ready_count": sum(bool(row["runtime_replay_ready"]) for row in rows),
        "triage_ready_count": sum(bool(row["triage_ready"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the latest parameter-golf PRs for conker-detect triage readiness.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--prs-json", help="Path to a pre-fetched GitHub pulls API response")
    parser.add_argument("--patch-dir", help="Directory containing <pr_number>.patch files")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    if args.prs_json:
        prs = json.loads(Path(args.prs_json).read_text(encoding="utf-8"))
    else:
        raise ValueError(
            "This environment expects --prs-json with a pre-fetched pulls API response. "
            f"Fetch it first with: curl -sS '{API_URL.format(limit=args.limit)}' > prs.json"
        )
    rows: list[dict[str, Any]] = []
    for pr in prs:
        try:
            if not args.patch_dir:
                raise ValueError("Provide --patch-dir with pre-fetched <pr_number>.patch files")
            patch_path = Path(args.patch_dir) / f"{pr['number']}.patch"
            if not patch_path.exists():
                raise FileNotFoundError(f"Missing patch file: {patch_path}")
            patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
            if not patch_text.strip():
                raise ValueError(f"Patch file is empty: {patch_path}")
            files = patch_files(patch_text)
            rows.append(classify_pr(pr, patch_text, files))
        except Exception as exc:  # pragma: no cover - best effort sweep
            rows.append(
                {
                    "number": pr["number"],
                    "state": pr["state"],
                    "created_at": pr["created_at"],
                    "title": pr["title"],
                    "author": pr["user"]["login"],
                    "html_url": pr["html_url"],
                    "category": "fetch_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "triage_ready": False,
                    "runtime_replay_ready": False,
                }
            )

    result = {"summary": summarize(rows), "prs": rows}
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
