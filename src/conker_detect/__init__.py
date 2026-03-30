"""Conker detect."""

from .attack import load_campaign, normalize_campaign, run_attack_campaign
from .audit import (
    audit_artifact,
    audit_bundle,
    audit_matrix,
    carve_safetensors_slice,
    compare_bundles,
    compare_stats,
    inspect_safetensors_file,
    inspect_tensor_bundle,
    load_matrix,
    load_npz_tensors,
    load_safetensors_repo_tensors,
    load_safetensors_tensors,
    load_tensor_bundle,
    mask_geometry_stats,
    region_stats,
    spectral_stats,
)
from .handoff import prepare_ledger_handoff
from .legality import audit_legality, audit_parameter_golf_legality, load_adapter, load_token_array
from .leakage import build_fuzzy_trigger_case_suite, build_fuzzy_trigger_string_suite, build_leakage_probe_suite
from .ledger_handoff import write_ledger_bundle_manifest
from .priors import load_prior_source, summarize_static_priors
from .provenance import audit_provenance
from .replay import replay_parameter_golf, replay_runtime
from .submission import audit_submission
from .trigger import (
    activation_diff,
    activation_probe_report,
    chat_diff,
    cross_model_compare,
    describe_provider,
    load_case,
    load_probe_config,
    load_provider,
    minimize_trigger,
    mutate_case,
    normalize_case,
    score_case_suite,
    sweep_variants,
)

__all__ = [
    "audit_artifact",
    "audit_bundle",
    "audit_matrix",
    "activation_diff",
    "load_campaign",
    "audit_legality",
    "audit_parameter_golf_legality",
    "audit_provenance",
    "audit_submission",
    "carve_safetensors_slice",
    "chat_diff",
    "compare_bundles",
    "compare_stats",
    "cross_model_compare",
    "describe_provider",
    "build_fuzzy_trigger_case_suite",
    "build_fuzzy_trigger_string_suite",
    "build_leakage_probe_suite",
    "inspect_safetensors_file",
    "inspect_tensor_bundle",
    "load_case",
    "load_adapter",
    "load_matrix",
    "load_npz_tensors",
    "load_probe_config",
    "load_provider",
    "load_prior_source",
    "load_safetensors_repo_tensors",
    "load_safetensors_tensors",
    "load_token_array",
    "load_tensor_bundle",
    "mask_geometry_stats",
    "activation_probe_report",
    "normalize_campaign",
    "minimize_trigger",
    "mutate_case",
    "normalize_case",
    "prepare_ledger_handoff",
    "replay_parameter_golf",
    "replay_runtime",
    "region_stats",
    "score_case_suite",
    "spectral_stats",
    "run_attack_campaign",
    "summarize_static_priors",
    "sweep_variants",
    "write_ledger_bundle_manifest",
]
