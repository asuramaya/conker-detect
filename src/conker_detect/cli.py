from __future__ import annotations

import argparse
import json
from pathlib import Path

from .attack import load_campaign, run_attack_campaign
from .audit import (
    audit_artifact,
    audit_bundle,
    audit_matrix,
    carve_safetensors_slice,
    compare_bundles,
    inspect_tensor_bundle,
    load_matrix,
    load_tensor_bundle,
    mask_geometry_stats,
    summarize_tensor_families,
)
from .handoff import prepare_ledger_handoff
from .legality import audit_legality, load_adapter, load_json_config, load_token_array
from .ledger_handoff import write_ledger_bundle_manifest
from .provenance import audit_provenance
from .replay import replay_runtime
from .submission import audit_submission
from .trigger import (
    MUTATION_FAMILIES,
    activation_diff,
    chat_diff,
    cross_model_compare,
    load_case,
    load_probe_config,
    load_provider,
    minimize_trigger,
    mutate_case,
    sweep_variants,
)


def _write_output(text: str, json_path: str | None) -> None:
    print(text)
    if json_path:
        out_path = Path(json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


def _write_text(text: str, path: str | None) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")


def _submission_markdown(result: dict) -> str:
    lines = [
        f"# Submission Audit: {result['submission'].get('name') or '<submission>'}",
        "",
        f"- profile: `{result['profile']}`",
        f"- verdict: `{result['verdict']}`",
    ]
    track = result["submission"].get("track")
    if track:
        lines.append(f"- track: `{track}`")
    lines.extend(["", "## Checks", ""])
    for name, row in result.get("checks", {}).items():
        lines.append(f"- `{name}`: `{row.get('pass')}`")
    findings = result.get("findings", [])
    if findings:
        lines.extend(["", "## Findings", ""])
        for row in findings:
            path = f" ({row['path']})" if "path" in row else ""
            lines.append(f"- `{row['severity']}` `{row['kind']}`: {row['message']}{path}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect hidden side channels and structural anomalies in matrices and tensor bundles.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_matrix = sub.add_parser("matrix", help="Audit a single .npy or .csv matrix")
    p_matrix.add_argument("matrix")
    p_matrix.add_argument("--reference")
    p_matrix.add_argument("--topk", type=int, default=16)
    p_matrix.add_argument("--expect-causal-mask", action="store_true")
    p_matrix.add_argument("--json")

    p_geom = sub.add_parser("geometry", help="Square-matrix lag / Toeplitz geometry report")
    p_geom.add_argument("matrix")
    p_geom.add_argument("--json")

    p_bundle = sub.add_parser("bundle", help="Audit all 2D tensors in a tensor bundle (.npz, .safetensors, or local HF safetensors repo)")
    p_bundle.add_argument("bundle")
    p_bundle.add_argument("--topk", type=int, default=16)
    p_bundle.add_argument("--only-square", action="store_true")
    p_bundle.add_argument("--name-regex")
    p_bundle.add_argument("--expect-causal", action="append", default=[], help="Substring marking tensors that should be strict-lower causal")
    p_bundle.add_argument("--json")

    p_catalog = sub.add_parser("catalog", help="Catalog a safetensors file, shard slice, or local HF repo without loading tensors")
    p_catalog.add_argument("source")
    p_catalog.add_argument("--name-regex")
    p_catalog.add_argument("--limit", type=int)
    p_catalog.add_argument("--json")

    p_carve = sub.add_parser("carve", help="Carve fully available tensors from a safetensors file or range slice into a standalone bundle")
    p_carve.add_argument("source")
    p_carve.add_argument("out")
    p_carve.add_argument("--name-regex")
    p_carve.add_argument("--only-2d", action="store_true")
    p_carve.add_argument("--only-square", action="store_true")
    p_carve.add_argument("--max-tensors", type=int)
    p_carve.add_argument("--json")

    p_family = sub.add_parser("family", help="Summarize loaded tensors by projection family")
    p_family.add_argument("bundle")
    p_family.add_argument("--topk", type=int, default=16)
    p_family.add_argument("--only-square", action="store_true")
    p_family.add_argument("--name-regex")
    p_family.add_argument("--json")

    p_artifact = sub.add_parser("artifact", help="Audit a packed .ptz artifact for boundary and payload anomalies")
    p_artifact.add_argument("artifact")
    p_artifact.add_argument("--json")

    p_submission = sub.add_parser("submission", help="Tier 1 manifest-first submission audit")
    p_submission.add_argument("source", help="Path to a submission manifest JSON file")
    p_submission.add_argument("--json")
    p_submission.add_argument("--md", help="Optional Markdown summary output path")

    p_provenance = sub.add_parser("provenance", help="Manifest-first provenance and selection audit")
    p_provenance.add_argument("source", help="Path to a provenance manifest JSON file")
    p_provenance.add_argument("--json")

    p_handoff = sub.add_parser("handoff", help="Prepare detector reports and a conker-ledger manifest from one run directory")
    p_handoff.add_argument("run_dir", help="Submission run directory containing README, submission.json, logs, and artifacts")
    p_handoff.add_argument("out_dir", help="Output directory for synthesized reports and ledger_manifest.json")
    p_handoff.add_argument("--bundle-id")
    p_handoff.add_argument("--profile", choices=["parameter-golf"], default="parameter-golf")
    p_handoff.add_argument("--repo-root", help="Optional repo root; defaults to the nearest parent containing records/ or .git")
    p_handoff.add_argument("--patch", help="Optional patch file for Tier 1 submission audit context")
    p_handoff.add_argument("--provenance-source", help="Optional provenance manifest JSON path")
    p_handoff.add_argument("--adapter", help="Python adapter module or .py file exporting build_adapter(config)")
    p_handoff.add_argument("--adapter-config", help="JSON object or path to a JSON config file passed to build_adapter(config)")
    p_handoff.add_argument("--tokens", help="1D token array (.npy, .npz, or .csv) for legality and replay")
    p_handoff.add_argument("--tokens-key", help="Array name when --tokens points at an .npz bundle")
    p_handoff.add_argument("--trust-level", choices=["basic", "traced", "strict"], default="basic")
    p_handoff.add_argument("--chunk-size", type=int, default=32768)
    p_handoff.add_argument("--max-chunks", type=int, help="Only replay the first N chunks for a cheap prefix pass")
    p_handoff.add_argument("--sample-chunks", type=int, default=4)
    p_handoff.add_argument("--future-probes-per-chunk", type=int, default=2)
    p_handoff.add_argument("--answer-probes-per-chunk", type=int, default=2)
    p_handoff.add_argument("--positions-per-future-probe", type=int, default=4)
    p_handoff.add_argument("--position-batch-size", type=int, default=256)
    p_handoff.add_argument("--seed", type=int, default=0)
    p_handoff.add_argument("--vocab-size", type=int)
    p_handoff.add_argument("--atol", type=float, default=1e-7)
    p_handoff.add_argument("--rtol", type=float, default=1e-7)
    p_handoff.add_argument("--json")

    p_handoff = sub.add_parser("ledger-manifest", help="Write a conker-ledger bundle manifest from detector outputs")
    p_handoff.add_argument("out", help="Output manifest JSON path")
    p_handoff.add_argument("--bundle-id", required=True)
    p_handoff.add_argument("--claim")
    p_handoff.add_argument("--metrics")
    p_handoff.add_argument("--provenance")
    p_handoff.add_argument("--audits")
    p_handoff.add_argument("--submission-report")
    p_handoff.add_argument("--provenance-report")
    p_handoff.add_argument("--legality-report")
    p_handoff.add_argument("--replay-report")
    p_handoff.add_argument("--json")

    p_compare = sub.add_parser("compare", help="Compare matching 2D tensors across two tensor bundles")
    p_compare.add_argument("lhs_bundle")
    p_compare.add_argument("rhs_bundle")
    p_compare.add_argument("--topk", type=int, default=16)
    p_compare.add_argument("--only-square", action="store_true")
    p_compare.add_argument("--name-regex")
    p_compare.add_argument("--strip-prefix", action="append", default=[], help="Prefix to strip before name matching, e.g. student.")
    p_compare.add_argument("--json")

    p_chatdiff = sub.add_parser("chatdiff", help="Compare chat completions for two prompt cases on one model")
    p_chatdiff.add_argument("--provider", required=True, help="Python provider module or .py file exporting build_provider(config)")
    p_chatdiff.add_argument("--provider-config", help="JSON object or path to a JSON config file passed to build_provider(config)")
    p_chatdiff.add_argument("--lhs", required=True, help="JSON file describing the left prompt case")
    p_chatdiff.add_argument("--rhs", required=True, help="JSON file describing the right prompt case")
    p_chatdiff.add_argument("--model", required=True)
    p_chatdiff.add_argument("--repeats", type=int, default=1, help="Number of repeated chat probes to aggregate per side")
    p_chatdiff.add_argument("--json")

    p_actdiff = sub.add_parser("actdiff", help="Compare activations for two prompt cases on one model")
    p_actdiff.add_argument("--provider", required=True, help="Python provider module or .py file exporting build_provider(config)")
    p_actdiff.add_argument("--provider-config", help="JSON object or path to a JSON config file passed to build_provider(config)")
    p_actdiff.add_argument("--lhs", required=True, help="JSON file describing the left prompt case")
    p_actdiff.add_argument("--rhs", required=True, help="JSON file describing the right prompt case")
    p_actdiff.add_argument("--model", required=True)
    p_actdiff.add_argument("--json")

    p_crossmodel = sub.add_parser("crossmodel", help="Compare one prompt case across multiple models")
    p_crossmodel.add_argument("--provider", required=True, help="Python provider module or .py file exporting build_provider(config)")
    p_crossmodel.add_argument("--provider-config", help="JSON object or path to a JSON config file passed to build_provider(config)")
    p_crossmodel.add_argument("--case", required=True, help="JSON file describing the shared prompt case")
    p_crossmodel.add_argument("--model", action="append", required=True, dest="models")
    p_crossmodel.add_argument("--repeats", type=int, default=1, help="Number of repeated chat probes to aggregate per model")
    p_crossmodel.add_argument("--json")

    p_mutate = sub.add_parser("mutate", help="Generate prompt mutations for trigger hunting")
    p_mutate.add_argument("case", help="JSON file describing the base prompt case")
    p_mutate.add_argument("--family", action="append", dest="families", choices=MUTATION_FAMILIES, help="Mutation family to apply")
    p_mutate.add_argument("--json")

    p_sweep = sub.add_parser("sweep", help="Score mutated prompt variants against a base case")
    p_sweep.add_argument("--provider", required=True, help="Python provider module or .py file exporting build_provider(config)")
    p_sweep.add_argument("--provider-config", help="JSON object or path to a JSON config file passed to build_provider(config)")
    p_sweep.add_argument("--case", required=True, help="JSON file describing the base prompt case")
    p_sweep.add_argument("--model", action="append", required=True, dest="models")
    p_sweep.add_argument("--family", action="append", dest="families", choices=MUTATION_FAMILIES, help="Mutation family to apply")
    p_sweep.add_argument("--repeats", type=int, default=1, help="Number of repeated chat probes to aggregate per scored variant")
    p_sweep.add_argument("--json")

    p_minimize = sub.add_parser("minimize", help="Greedily minimize a trigger candidate against a control case")
    p_minimize.add_argument("--provider", required=True, help="Python provider module or .py file exporting build_provider(config)")
    p_minimize.add_argument("--provider-config", help="JSON object or path to a JSON config file passed to build_provider(config)")
    p_minimize.add_argument("--control", required=True, help="JSON file describing the control prompt case")
    p_minimize.add_argument("--candidate", required=True, help="JSON file describing the trigger candidate case")
    p_minimize.add_argument("--model", required=True)
    p_minimize.add_argument("--metric", choices=["chat", "activation"], default="chat")
    p_minimize.add_argument("--unit", choices=["token", "line", "char"], default="token")
    p_minimize.add_argument("--threshold", type=float)
    p_minimize.add_argument("--repeats", type=int, default=1, help="Number of repeated chat probes to aggregate when metric=chat")
    p_minimize.add_argument("--json")

    p_attack = sub.add_parser("attack", help="Run a ranked trigger-hunting campaign against a provider")
    p_attack.add_argument("--provider", required=True, help="Python provider module or .py file exporting build_provider(config)")
    p_attack.add_argument("--provider-config", help="JSON object or path to a JSON config file passed to build_provider(config)")
    p_attack.add_argument("campaign", help="JSON campaign file defining models, cases, and mutation families")
    p_attack.add_argument("--repeats", type=int, help="Override campaign chat_repeats for live probing")
    p_attack.add_argument("--json")

    p_legality = sub.add_parser("legality", help="Behavioral legality audit via adapter-backed runtime probes")
    p_legality.add_argument("--adapter", required=True, help="Python adapter module or .py file exporting build_adapter(config)")
    p_legality.add_argument("--adapter-config", help="JSON object or path to a JSON config file passed to build_adapter(config)")
    p_legality.add_argument("--tokens", required=True, help="1D token array (.npy, .npz, or .csv)")
    p_legality.add_argument("--tokens-key", help="Array name when --tokens points at an .npz bundle")
    p_legality.add_argument("--profile", choices=["parameter-golf"], default="parameter-golf")
    p_legality.add_argument("--trust-level", choices=["basic", "traced", "strict"], default="basic")
    p_legality.add_argument("--chunk-size", type=int, default=32768)
    p_legality.add_argument("--max-chunks", type=int, help="Only audit the first N chunks for a cheap prefix pass")
    p_legality.add_argument("--sample-chunks", type=int, default=4)
    p_legality.add_argument("--future-probes-per-chunk", type=int, default=2)
    p_legality.add_argument("--answer-probes-per-chunk", type=int, default=2)
    p_legality.add_argument("--positions-per-future-probe", type=int, default=4)
    p_legality.add_argument("--seed", type=int, default=0)
    p_legality.add_argument("--vocab-size", type=int)
    p_legality.add_argument("--atol", type=float, default=1e-7)
    p_legality.add_argument("--rtol", type=float, default=1e-7)
    p_legality.add_argument("--json")

    p_replay = sub.add_parser("replay", help="Finalist-strength replay summary via adapter-backed runtime")
    p_replay.add_argument("--adapter", required=True, help="Python adapter module or .py file exporting build_adapter(config)")
    p_replay.add_argument("--adapter-config", help="JSON object or path to a JSON config file passed to build_adapter(config)")
    p_replay.add_argument("--tokens", required=True, help="1D token array (.npy, .npz, or .csv)")
    p_replay.add_argument("--tokens-key", help="Array name when --tokens points at an .npz bundle")
    p_replay.add_argument("--profile", choices=["parameter-golf"], default="parameter-golf")
    p_replay.add_argument("--chunk-size", type=int, default=32768)
    p_replay.add_argument("--max-chunks", type=int)
    p_replay.add_argument("--sample-chunks", type=int, default=4)
    p_replay.add_argument("--position-batch-size", type=int, default=256)
    p_replay.add_argument("--seed", type=int, default=0)
    p_replay.add_argument("--atol", type=float, default=1e-7)
    p_replay.add_argument("--rtol", type=float, default=1e-7)
    p_replay.add_argument("--json")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "matrix":
        matrix = load_matrix(Path(args.matrix))
        ref = load_matrix(Path(args.reference)) if args.reference else None
        result = audit_matrix(
            matrix,
            name=Path(args.matrix).name,
            topk=args.topk,
            reference=ref,
            expect_causal_mask=args.expect_causal_mask,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "bundle":
        result = audit_bundle(
            Path(args.bundle),
            topk=args.topk,
            only_square=args.only_square,
            name_regex=args.name_regex,
            expect_causal=tuple(args.expect_causal),
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "catalog":
        result = inspect_tensor_bundle(
            Path(args.source),
            name_regex=args.name_regex,
            limit=args.limit,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "carve":
        result = carve_safetensors_slice(
            Path(args.source),
            Path(args.out),
            only_2d=args.only_2d,
            only_square=args.only_square,
            name_regex=args.name_regex,
            max_tensors=args.max_tensors,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "family":
        tensors = load_tensor_bundle(
            Path(args.bundle),
            only_2d=True,
            only_square=args.only_square,
            name_regex=args.name_regex,
        )
        result = summarize_tensor_families(tensors, topk=args.topk)
        result["bundle"] = str(Path(args.bundle))
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "artifact":
        result = audit_artifact(Path(args.artifact))
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "submission":
        result = audit_submission(Path(args.source))
        _write_output(json.dumps(result, indent=2), args.json)
        _write_text(_submission_markdown(result), args.md)
        return

    if args.command == "provenance":
        result = audit_provenance(Path(args.source))
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "handoff":
        result = prepare_ledger_handoff(
            Path(args.run_dir),
            Path(args.out_dir),
            bundle_id=args.bundle_id,
            profile=args.profile,
            repo_root=Path(args.repo_root) if args.repo_root else None,
            patch=Path(args.patch) if args.patch else None,
            provenance_source=Path(args.provenance_source) if args.provenance_source else None,
            adapter_ref=args.adapter,
            adapter_config_raw=args.adapter_config,
            tokens_path=Path(args.tokens) if args.tokens else None,
            tokens_key=args.tokens_key,
            trust_level=args.trust_level,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            sample_chunks=args.sample_chunks,
            future_probes_per_chunk=args.future_probes_per_chunk,
            answer_probes_per_chunk=args.answer_probes_per_chunk,
            positions_per_future_probe=args.positions_per_future_probe,
            position_batch_size=args.position_batch_size,
            seed=args.seed,
            vocab_size=args.vocab_size,
            atol=args.atol,
            rtol=args.rtol,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "ledger-manifest":
        result = write_ledger_bundle_manifest(
            Path(args.out),
            bundle_id=args.bundle_id,
            claim=args.claim,
            metrics=args.metrics,
            provenance=args.provenance,
            audits=args.audits,
            submission_report=args.submission_report,
            provenance_report=args.provenance_report,
            legality_report=args.legality_report,
            replay_report=args.replay_report,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "geometry":
        matrix = load_matrix(Path(args.matrix))
        result = {
            "matrix": args.matrix,
            "shape": list(matrix.shape),
            "geometry": mask_geometry_stats(matrix),
        }
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "compare":
        result = compare_bundles(
            Path(args.lhs_bundle),
            Path(args.rhs_bundle),
            topk=args.topk,
            only_square=args.only_square,
            name_regex=args.name_regex,
            strip_prefixes=tuple(args.strip_prefix),
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "chatdiff":
        provider = load_provider(args.provider, load_probe_config(args.provider_config))
        result = chat_diff(
            provider,
            load_case(args.lhs, default_id="lhs"),
            load_case(args.rhs, default_id="rhs"),
            model=args.model,
            repeats=args.repeats,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "actdiff":
        provider = load_provider(args.provider, load_probe_config(args.provider_config))
        result = activation_diff(
            provider,
            load_case(args.lhs, default_id="lhs"),
            load_case(args.rhs, default_id="rhs"),
            model=args.model,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "crossmodel":
        provider = load_provider(args.provider, load_probe_config(args.provider_config))
        result = cross_model_compare(
            provider,
            load_case(args.case, default_id="case"),
            models=list(args.models),
            repeats=args.repeats,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "mutate":
        result = mutate_case(load_case(args.case, default_id="case"), families=args.families)
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "sweep":
        provider = load_provider(args.provider, load_probe_config(args.provider_config))
        result = sweep_variants(
            provider,
            load_case(args.case, default_id="case"),
            models=list(args.models),
            families=args.families,
            chat_repeats=args.repeats,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "minimize":
        provider = load_provider(args.provider, load_probe_config(args.provider_config))
        result = minimize_trigger(
            provider,
            load_case(args.control, default_id="control"),
            load_case(args.candidate, default_id="candidate"),
            model=args.model,
            metric=args.metric,
            unit=args.unit,
            threshold=args.threshold,
            chat_repeats=args.repeats,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "attack":
        provider = load_provider(args.provider, load_probe_config(args.provider_config))
        campaign = load_campaign(args.campaign)
        if args.repeats is not None:
            campaign = dict(campaign)
            campaign["chat_repeats"] = int(args.repeats)
        result = run_attack_campaign(provider, campaign)
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "legality":
        adapter = load_adapter(args.adapter, load_json_config(args.adapter_config))
        tokens = load_token_array(Path(args.tokens), key=args.tokens_key)
        result = audit_legality(
            adapter,
            tokens,
            profile=args.profile,
            trust_level=args.trust_level,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            sample_chunks=args.sample_chunks,
            future_probes_per_chunk=args.future_probes_per_chunk,
            answer_probes_per_chunk=args.answer_probes_per_chunk,
            positions_per_future_probe=args.positions_per_future_probe,
            seed=args.seed,
            vocab_size=args.vocab_size,
            atol=args.atol,
            rtol=args.rtol,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    if args.command == "replay":
        adapter = load_adapter(args.adapter, load_json_config(args.adapter_config))
        tokens = load_token_array(Path(args.tokens), key=args.tokens_key)
        result = replay_runtime(
            adapter,
            tokens,
            profile=args.profile,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            sample_chunks=args.sample_chunks,
            position_batch_size=args.position_batch_size,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
        )
        _write_output(json.dumps(result, indent=2), args.json)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
