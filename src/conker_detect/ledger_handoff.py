from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_ledger_bundle_manifest(
    out_path: Path,
    *,
    bundle_id: str,
    claim: str | None = None,
    metrics: str | None = None,
    provenance: str | None = None,
    audits: str | None = None,
    submission_report: str | None = None,
    provenance_report: str | None = None,
    legality_report: str | None = None,
    replay_report: str | None = None,
) -> dict[str, Any]:
    attachments: list[dict[str, str]] = []
    if submission_report:
        attachments.append({"source": submission_report, "dest": "audits/tier1/submission.json"})
    if provenance_report:
        attachments.append({"source": provenance_report, "dest": "audits/tier1/provenance.json"})
    if legality_report:
        attachments.append({"source": legality_report, "dest": "audits/tier3/legality.json"})
    if replay_report:
        attachments.append({"source": replay_report, "dest": "audits/tier3/replay.json"})

    manifest: dict[str, Any] = {"bundle_id": bundle_id, "attachments": attachments}
    if claim:
        manifest["claim"] = claim
    if metrics:
        manifest["metrics"] = metrics
    if provenance:
        manifest["provenance"] = provenance
    if audits:
        manifest["audits"] = audits

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
