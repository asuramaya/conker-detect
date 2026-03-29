from __future__ import annotations

import re
from typing import Any


PATCH_FILE_RE = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
ARTIFACT_EXTS = (".pt", ".pth", ".bin", ".npz", ".safetensors", ".ckpt", ".ptz")


PARAMETER_GOLF_PROFILE: dict[str, Any] = {
    "required_evidence": ("readme", "submission_json"),
    "claim_fields": (
        "name",
        "track",
        "val_bpb",
        "pre_quant_val_bpb",
        "bytes_total",
        "bytes_model_int6_zlib",
    ),
    "protocol_risk_markers": (
        ("optimizer.step()", "optimizer_step"),
        (".backward(", "backward_call"),
        ("model.train()", "model_train_call"),
        ("set_grad_enabled(true)", "grad_enabled_true"),
    ),
    "protocol_reassurance_markers": (
        ("inference_mode", "inference_mode"),
        ("no_grad", "no_grad"),
        ("score-first", "score_first"),
    ),
    "data_boundary_risk_markers": (
        ("fineweb_val", "validation_path_reference"),
        ("urllib", "network_call"),
        ("requests.", "network_call"),
        ("http://", "network_call"),
        ("https://", "network_call"),
        ("wget ", "network_call"),
        ("curl ", "network_call"),
    ),
}


def get_profile_rules(profile: str) -> dict[str, Any]:
    if profile != "parameter-golf":
        raise ValueError(f"Unknown submission profile: {profile}")
    return PARAMETER_GOLF_PROFILE


def patch_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for match in PATCH_FILE_RE.finditer(patch_text):
        path = match.group(1)
        if path != "/dev/null":
            files.append(path)
    return files


def classify_patch_context(profile: str, patch_text: str, files: list[str]) -> dict[str, Any]:
    if profile != "parameter-golf":
        raise ValueError(f"Unknown submission profile: {profile}")
    lower_patch = patch_text.lower()
    record_dirs = sorted(
        {
            path.split("/", 3)[2]
            for path in files
            if path.startswith("records/") and path.count("/") >= 3
        }
    )
    adds_submission_json = any(path.endswith("/submission.json") for path in files)
    adds_train_gpt = any(path.endswith("/train_gpt.py") or path.endswith("/train_gpt_mlx.py") for path in files)
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
    return {
        "category": category,
        "record_dirs": record_dirs,
        "file_count": len(files),
        "adds_submission_json": adds_submission_json,
        "adds_train_gpt": adds_train_gpt,
        "touches_core": touches_core,
        "has_artifact_blob": has_artifact,
        "ttt_signal": "ttt" in lower_patch or "test-time" in lower_patch,
        "score_first_signal": "score-first" in lower_patch,
        "inference_mode_signal": "inference_mode" in lower_patch,
        "optimizer_in_eval_signal": "optimizer.step()" in patch_text
        or (".step()" in patch_text and "eval" in lower_patch and "optimizer" in lower_patch),
        "triage_ready": category in {"record_submission", "core_code", "record_misc"},
        "runtime_replay_ready": category == "record_submission" and has_artifact,
        "sample_files": files[:12],
    }
