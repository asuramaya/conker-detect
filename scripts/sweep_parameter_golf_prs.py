#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conker_detect.submission_profiles import classify_patch_context, patch_files


API_URL = "https://api.github.com/repos/openai/parameter-golf/pulls?state=all&sort=created&direction=desc&per_page={limit}"


def classify_pr(pr: dict[str, Any], patch_text: str, files: list[str]) -> dict[str, Any]:
    lower_title = str(pr.get("title", "")).lower()
    lower_patch = patch_text.lower()
    triage = classify_patch_context("parameter-golf", patch_text, files)
    triage["ttt_signal"] = bool(triage["ttt_signal"] or "ttt" in lower_title or "test-time" in lower_title)
    triage["score_first_signal"] = bool(triage["score_first_signal"] or "score-first" in lower_title)

    return {
        "number": pr["number"],
        "state": pr["state"],
        "created_at": pr["created_at"],
        "title": pr["title"],
        "author": pr["user"]["login"],
        "html_url": pr["html_url"],
        **triage,
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
