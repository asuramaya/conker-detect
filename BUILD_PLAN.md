# Build Plan

This document turns the current "missing at large" list into an execution plan for `conker-detect`.

The short version:

- keep `conker-detect` as the detector
- make the missing boundaries explicit in code and reports
- add the missing Tier 1 / provenance / testing layers incrementally
- split implementation across independent tracks so multiple agents can work in parallel without stepping on each other

## Current Gaps

The main missing capabilities are:

1. Tier 1 submission audit
- code / README / `submission.json` / logs are not audited by the tool yet

2. Adapter trust and scoring traces
- legality probes still depend on the honesty of the adapter surface
- score accounting is only partially visible through optional `sample_gold_logprobs`

3. Provenance outside the artifact
- best-of-`k` outcome selection, train/eval contamination, and declared dataset boundaries are not represented

4. Power and replay depth
- runtime probes are sampled and adapter-backed
- full finalist replay and outer-protocol checks are still missing

5. Hardening
- there was no test tree or CI
- there is no public fixture corpus of known legal / illegal cases

## Parallel Workstreams

These tracks can be owned by separate agents with disjoint write sets.

### Track A: Tier 1 Submission Audit

Goal:
- add a first-class `submission` audit command

Suggested files:
- `src/conker_detect/submission.py`
- `src/conker_detect/submission_checks.py`
- `src/conker_detect/submission_extract.py`
- `src/conker_detect/submission_profiles.py`
- `src/conker_detect/cli.py`
- `tests/test_submission.py`
- `examples/submission_fixtures/`

CLI shape:

```bash
conker-detect submission manifest.json --json out/tier1.json --md out/tier1.md
```

Convenience form:

```bash
conker-detect submission \
  --profile parameter-golf \
  --repo-root /path/to/checkout \
  --submission-root records/.../2026-03-27_Conker5_TandemResidual_MLX \
  --patch /path/to/998.patch \
  --json out/tier1.json
```

Inputs:
- submission folder or explicit manifest
- `README.md`
- `submission.json`
- logs
- optional patch file or file list

Checks:
- claim consistency across README / JSON / logs
- artifact size consistency
- declared protocol shape markers
- reproducibility completeness
- suspicious eval-path markers such as optimizer steps in scored regions

Outputs:
- `pass` / `warn` / `fail`
- structured findings with file references

### Track B: Legality Trace Extension

Goal:
- reduce adapter trust by extending the legality contract

Suggested files:
- `src/conker_detect/legality.py`
- `src/conker_detect/trace_schema.py`
- `examples/packed_cache_demo_adapter.py`
- `tests/test_legality.py`
- `tests/test_legality_trace.py`

Contract additions:
- `sample_gold_logprobs`
- optional `sample_trace`
- optional `score_trace_version`

Suggested trace fields:
- `gold_logprobs`
- `loss_nats`
- `weights`
- `counted`
- `path_ids`
- `state_hash_before`
- `state_hash_after`

CLI shape:
- `--trust-level basic|traced|strict`

Checks:
- returned scalar gold scores match returned distributions
- answer-dependent inclusion / weighting mismatches are visible
- hidden loss-path mismatches are testable
- score-path mutation is testable in `strict` mode
- known cheating modes fail deterministically in fixtures

### Track C: Provenance and Selection Manifests

Goal:
- represent what a single artifact cannot prove

Suggested files:
- `src/conker_detect/provenance.py`
- `src/conker_detect/cli.py`
- `tests/test_provenance.py`
- `examples/provenance_fixtures/`

Inputs:
- run manifest
- dataset fingerprint manifest
- declared training-run lineage

Checks:
- explicit declaration of selected run vs sweep population
- dataset split fingerprints and overlap markers
- declared held-out set identity

This track should not pretend to prove cleanliness; it should package the missing evidence explicitly.

### Track D: Finalist Replay / Power Upgrades

Goal:
- make Tier 3 stronger when finalists provide enough material

Suggested files:
- `src/conker_detect/replay.py`
- `tests/test_replay.py`

Features:
- full-chunk replay mode
- stronger suffix perturbation schedules
- repeated-run consistency checks
- optional stricter tolerances and aggregate drift summaries

### Track E: Fixtures, Tests, and CI

Goal:
- make regressions cheap to catch

Suggested files:
- `tests/conftest.py`
- `tests/test_audit.py`
- `tests/test_legality.py`
- `tests/test_cli.py`
- `tests/fixtures/`
- `.github/workflows/tests.yml`

## Suggested Parallel-Agent Breakdown

If multiple agents are working concurrently, use this split:

1. Agent 1 owns Track A.
Files:
- `src/conker_detect/submission.py`
- `tests/test_submission.py`
- `examples/submission_fixtures/`

2. Agent 2 owns Track B.
Files:
- `src/conker_detect/legality.py`
- `examples/packed_cache_demo_adapter.py`
- `tests/test_legality.py`

3. Agent 3 owns Track C.
Files:
- `src/conker_detect/provenance.py`
- `tests/test_provenance.py`
- `examples/provenance_fixtures/`

4. Agent 4 owns Track D.
Files:
- `src/conker_detect/replay.py`
- `tests/test_replay.py`

5. Main branch owner integrates Track E and resolves CLI / docs stitching.
Files:
- `src/conker_detect/cli.py`
- `README.md`
- `AUDIT_STANDARD.md`
- `.github/workflows/tests.yml`

## Phase Plan

### Phase 1

Land immediately:

- test tree and CI
- stronger legality/accounting fixtures
- basic build-plan documentation

Exit criteria:
- every current CLI mode has at least one regression test
- legal and illegal demo adapters are covered in CI

### Phase 2

Land next:

- Tier 1 `submission` audit
- provenance manifest schema

Exit criteria:
- code-only `parameter-golf` PRs can be audited without artifacts
- claim / artifact / log mismatches produce machine-readable findings

### Phase 3

Land for finalists:

- replay-strength upgrades
- expanded accounting traces
- explicit run-selection manifests

Exit criteria:
- finalist submissions can receive a stronger Tier 3 report
- missing evidence is surfaced explicitly rather than hidden behind a binary pass/fail

## Test Matrix

The repo should maintain a public corpus of small, deterministic fixtures.

### Unit Tests

- spectral stats on tiny matrices
- shape-mismatch compare behavior
- artifact boundary classification
- legality obligation coverage fields
- gold-logprob consistency
- known self-include / future-peek / hidden-score cheats
- submission claim and artifact-byte mismatch checks once Track A lands

### Fixture Tests

- one known-legal packed-cache adapter
- one same-position leak adapter
- one future-suffix leak adapter
- one hidden gold-score mismatch adapter
- one hidden loss-path mismatch adapter once Track B lands
- one artifact carrying deterministic substrate
- one artifact carrying structural-control tensors

### CLI Smoke Tests

- `matrix`
- `geometry`
- `bundle`
- `artifact`
- `compare`
- `legality`
- `submission` once Track A lands

### CI Policy

Run on every push and PR:

- `pytest -q`
- `python -m compileall src`

Run optionally in a slower job once Tier 1 and replay modules land:

- CLI smoke tests over example fixtures
- round-trip JSON output checks

## Non-Goals

This plan still does not claim that a single artifact can prove:

- absence of train/eval contamination
- absence of best-of-`k` run selection without provenance manifests
- fairness of the competition protocol itself

Those need policy and provenance layers, not just better probes.
