# conker-detect

`conker-detect` is a small defensive audit tool for model weights, checkpoint bundles, packed artifacts, and runtime legality probes.

It grew out of the `Conker-6` failure mode: a matrix that looked almost identical to a clean causal mask, but changed behavior massively because it had learned illegal diagonal and upper-triangle structure. The immediate lesson was:

- some side channels are easiest to catch with architecture-aware invariants
- others are easier to catch with spectral-tail or bundle-to-bundle comparison
- artifact-boundary bugs can be just as important as model bugs

This tool packages both.

## What It Does

`conker-detect` supports eleven audit modes:

1. `matrix`
- inspect a single `.npy` or `.csv` matrix
- compute spectral concentration and decay statistics
- compute region energy for square matrices
- optionally compare against a clean reference

2. `geometry`
- inspect square-mask lag structure
- compute lag-profile summary
- compare the strict-lower part to its Toeplitz lag-mean approximation
- quantify residual microstructure without needing the original training script

3. `bundle`
- inspect all 2D tensors in an `.npz` checkpoint bundle, a single `.safetensors` file, or a local Hugging Face safetensors repo
- tolerate `BF16` and `F8_E4M3` safetensors payloads through a PyTorch fallback when NumPy cannot decode them directly
- report per-tensor spectral and structural metrics
- optionally flag tensors that are expected to be strict-lower causal
- useful for telling raw replay checkpoints apart from packed submission payloads

4. `carve`
- carve fully available tensors out of a safetensors file or a byte-range shard slice
- write a smaller standalone `.safetensors` bundle that `bundle` and `compare` can inspect normally
- useful for probing giant Hugging Face shards without mirroring the whole file

5. `compare`
- compare matching 2D tensors across two tensor bundles
- useful for poisoned-vs-clean or trained-vs-reference analysis

6. `artifact`
- inspect a packed `.ptz` artifact
- classify payload entries as deterministic substrate, structural control, or learned payload
- surface raw-vs-compressed byte stories and suspicious packed names directly

7. `submission`
- audit a submission manifest and its attached evidence
- check claim consistency across `README.md`, `submission.json`, logs, artifacts, and patch context
- emit Tier 1 findings without pretending to prove runtime legality

8. `provenance`
- audit selection disclosure and dataset fingerprints from a provenance manifest
- surface best-of-`k` selection risk and missing held-out identity explicitly

9. `legality`
- run behavioral legality probes against a live submission adapter
- check score-phase repeatability and causal invariance
- grade the adapter surface at `basic`, `traced`, or `strict` trust levels
- support challenge-specific profiles such as score-first `parameter-golf` TTT

10. `replay`
- run finalist-strength chunked replay against a live submission adapter
- compute aggregate loss / bpb summaries and stronger repeated-run drift statistics
- keep this distinct from narrow legality probes

11. `handoff`
- generate Tier 1 detector reports and optional runtime reports from one run directory
- synthesize `claim.json`, `metrics.json`, `provenance.json`, and `audits.json`
- write a ready-to-edit `conker-ledger` manifest in the same output directory

12. `ledger-manifest`
- write a ready-to-edit `conker-ledger` bundle manifest from detector outputs
- prewire `submission`, `provenance`, `legality`, and `replay` attachments into the expected bundle paths

## Why This Exists

The main `Conker-6` lesson was not just “this branch was invalid.” It was:

- trained-model audits matter more than init audits
- a tiny forbidden-region perturbation can matter more than broad geometric similarity
- cosine and visual similarity can completely miss a functional leak
- small smooth structural tensors can carry too much behavior while staying cheap enough to hide inside the artifact budget

So `conker-detect` is deliberately simple:

- architecture-aware invariant checks for structured matrices
- spectral/tail checks for dense unrestricted weights
- reference comparison when you have two binaries
- protocol-aware runtime probes when weights alone are not enough

It is not a universal proof of cleanliness. It is a fast triage tool.

For a proposed community submission-review policy, see [AUDIT_STANDARD.md](./AUDIT_STANDARD.md).
To package audit outputs into a portable validity bundle, pair this repo with the sibling `conker-ledger` repository.

## Artifact Boundary

The `Conker` postmortem exposed three different artifact stories that should not be conflated:

1. raw replay checkpoints
- huge `.npz` files used for debugging and faithful replay
- often contain deterministic substrate that should never count as stored submission payload

2. broken packed artifacts
- compressed payloads that accidentally serialized regenerated deterministic tensors
- these are artifact-boundary bugs even if the model itself were otherwise clean

3. corrected packed artifacts
- deterministic substrate omitted correctly
- still need separate auditing for illegal learned structure inside the remaining payload

In the old tandem line, the broken packed artifact inflated to `11.87 MB` because it incorrectly stored regenerated tensors such as `base.linear_kernel`. The corrected packed artifact dropped to about `3.72 MB`, and the strict packed artifact landed at about `3.73 MB`.

That split matters:

- one bug counted things that should have stayed on the code side of the boundary
- another bug let supposedly fixed structural tensors become trainable and dominate the score

## Brutal Example

The `conker` branches that motivated this tool are not subtle cautionary tales. They are blunt ones.

The cleanest public example is [parameter-golf PR #998](https://github.com/openai/parameter-golf/pull/998), opened on March 28, 2026. It packaged `Conker-5 Tandem Residual Exact Experts (MLX, non-record)` and claimed:

- pre-quant full held-out `val_bpb = 0.57180453`
- packaged int6 full held-out `val_bpb = 0.57546632`
- artifact bytes `= 3,720,359`
- a supposedly "boringly valid" packaged run

- `conker4b_tandem` still looked good after full eval at `0.5718232495381582 bpb`, but its extracted causal mask carried forbidden-region mass with `upper_plus_diag_frac = 0.04358700722704721`.
- `conker4b_strict` zeroed that illegal structure and immediately collapsed to `2.0971244136143423 bpb`.
- `conker6_mask_geometry` looked visually close to a clean causal operator, but its saved mask still had `upper_frac = 0.011201489739837839` and `diag_frac = 0.017354798229237627`.
- In the attached `conker6` source summary, swapping that learned mask for its Toeplitz mean blows `full_test_bpb` from `0.07209327818598087` to `5.752106388513692`.

That is the point: if a result survives only while the forbidden structure is left in place, the performance was never clean. The side channel was doing the work.

## Install

```bash
pip install .
```

or run directly:

```bash
python -m conker_detect.cli ...
```

## Usage

### Audit one matrix

```bash
conker-detect matrix mask.npy --expect-causal-mask
```

With a clean reference:

```bash
conker-detect matrix suspect.npy --reference clean.npy --json out/report.json
```

### Inspect square-mask geometry

```bash
conker-detect geometry mask.npy
```

### Audit a checkpoint bundle

```bash
conker-detect bundle model.npz --expect-causal mask --expect-causal causal
```

This mode also works on a single `.safetensors` file or a local Hugging Face safetensors repo directory.
That makes it usable as a probe over exported HF weights without first repackaging the whole model.
For very large sharded repos, pass `--name-regex` so you only load the tensor families you actually want to inspect.
If NumPy does not understand a safetensors dtype such as `BF16` or `F8_E4M3`, `conker-detect` falls back to PyTorch decoding when it is installed.

Examples:

```bash
conker-detect bundle /path/to/model.safetensors --name-regex 'down_proj|q_proj'
conker-detect bundle /path/to/hf-repo --name-regex 'model\\.layers\\.(0|1)\\..*down_proj'
```

This mode is also the quickest way to check whether a raw replay checkpoint is carrying large deterministic tensors that should have been regenerated instead of packed.

Only square tensors:

```bash
conker-detect bundle model.npz --only-square
```

### Carve a partial safetensors slice

```bash
conker-detect carve shard-head.bin shard-mini.safetensors --name-regex 'model\\.layers\\.10\\.'
conker-detect bundle shard-mini.safetensors --name-regex 'weight_scale_inv$'
```

This is the intended workflow for very large Hugging Face models when you only have a ranged download of the front of one shard.

### Compare two checkpoint bundles

```bash
conker-detect compare model_a.npz model_b.npz --only-square
```

If one checkpoint wraps another under prefixes like `student.`:

```bash
conker-detect compare base.npz wrapped.npz --strip-prefix student.
```

You can also compare local Hugging Face safetensors repos directly:

```bash
conker-detect compare /path/to/dormant-model-1 /path/to/dormant-model-2 --name-regex 'down_proj|q_proj'
```

### Audit a packed artifact

```bash
conker-detect artifact model.int6.ptz
```

This is the quickest way to answer:

- did regenerated deterministic substrate get packed by mistake?
- which tiny structural tensors are crossing the artifact boundary?
- how much of the raw payload is actually structural control?

### Audit a submission claim

```bash
conker-detect submission submission_manifest.json --json out/submission.json --md out/submission.md
```

Minimal manifest shape:

```json
{
  "profile": "parameter-golf",
  "repo_root": "/abs/path/to/repo",
  "submission_root": "records/track_non_record_16mb/2026-03-27_Demo",
  "evidence": {
    "readme": "README.md",
    "submission_json": "submission.json",
    "results_json": "results.json",
    "logs": ["train.log"],
    "artifacts": ["model.int6.ptz"],
    "code": ["train_gpt.py"],
    "patch": "/abs/path/to/998.patch"
  }
}
```

This is the Tier 1 layer:

- claim consistency
- artifact-byte consistency
- protocol-shape risk markers
- data-boundary risk markers
- reproducibility surface
- patch triage context

It is a credibility and completeness audit, not a proof that the trained runtime is legal.

### Audit provenance and selection disclosure

```bash
conker-detect provenance provenance_manifest.json --json out/provenance.json
```

Minimal manifest shape:

```json
{
  "profile": "parameter-golf",
  "selection": {
    "submitted_run_id": "run-43",
    "selection_mode": "single_run",
    "candidate_run_count": 1
  },
  "datasets": {
    "train": {"name": "fineweb_train", "fingerprint": "train-sha"},
    "validation": {"name": "fineweb_val", "fingerprint": "val-sha"},
    "held_out_test": {"name": "fineweb_test", "fingerprint": "test-sha"}
  }
}
```

This layer answers provenance questions the artifact itself cannot answer cleanly:

- was the submitted run identified explicitly?
- was best-of-`k` selection disclosed?
- do train / validation / held-out fingerprints collide?
- was the held-out test set named at all?

### Audit behavioral legality

The runtime layer uses a small Python adapter that exposes:

```python
def build_adapter(config: dict) -> Runner: ...

class Runner:
    def fork(self) -> "Runner": ...
    def score_chunk(self, tokens, sample_positions=None) -> dict: ...
    def adapt_chunk(self, tokens) -> None: ...
```

When `sample_positions` are requested, `score_chunk()` must return:

```python
{"sample_predictions": ...}
```

where `sample_predictions[i]` is the prediction tensor for `sample_positions[i]`.
For legality work, this should be the actual full distribution used for scoring.

Optionally, adapters may also return:

```python
{"sample_predictions": ..., "sample_gold_logprobs": ...}
```

where `sample_gold_logprobs[i]` is the actual gold-token log-probability used for scoring `sample_positions[i]`.
If present, `conker-detect` will compare that scalar path against the returned full distribution and flag any mismatch.

For stronger accounting checks, adapters may also return:

```python
{
  "sample_predictions": ...,
  "sample_trace": {
    "gold_logprobs": ...,
    "loss_nats": ...,
    "weights": ...,
    "counted": ...,
    "path_ids": ...,
    "state_hash_before": ...,
    "state_hash_after": ...
  }
}
```

That trace lets the legality layer compare:

- reported gold-token scores vs the returned full distribution
- additive loss contributions vs `-weight * gold_logprob`
- counted flags / weights / path tags under perturbation
- repeated score-path state hashes when exposed

`parameter-golf` profile example:

```bash
python -m conker_detect.cli legality \
  --profile parameter-golf \
  --adapter examples/causal_demo_adapter.py \
  --adapter-config '{"vocab_size": 8}' \
  --tokens /tmp/demo_tokens.npy \
  --max-chunks 8
```

This profile checks:

- normalization
- optional trace coverage
- repeatability from the same pre-score state
- score-phase invariance when future suffix tokens are randomized
- answer-mask invariance when the token being scored is randomized too
- optional gold-logprob consistency when the adapter exposes scored gold-token logprobs
- optional accounting-contribution consistency when the adapter exposes `sample_trace`
- optional accounting-path invariance when the adapter exposes `sample_trace`
- optional state-hash consistency when the adapter exposes `sample_trace`

Those probes are deliberately narrower than a full legality proof. In particular:

- answer-mask invariance is a same-position leakage probe, not the whole causal contract
- future-suffix invariance is stronger, but still sampled rather than exhaustive
- normalization now also checks that sampled prediction vectors match the declared vocabulary size
- score accounting is only partially covered unless the adapter exposes `sample_gold_logprobs` and `sample_trace`
- best-of-`k` outcome selection remains out of scope for the current adapter contract

Packed-cache demo:

```bash
python -m conker_detect.cli legality \
  --profile parameter-golf \
  --trust-level strict \
  --adapter examples/packed_cache_demo_adapter.py \
  --adapter-config '{"mode":"legal","vocab_size":8}' \
  --tokens /tmp/conker_detect_tokens.npy
```

Switch `mode` to `self_include` or `future_peek` to see the detector catch score-after-update and future-leak behavior.
Switch `mode` to `reported_gold_cheat` to see the detector catch a hidden gold-token scoring path that does not match the returned full distribution.
Switch `mode` to `reported_loss_cheat`, `counted_flag_cheat`, `path_id_cheat`, or `state_hash_cheat` to see the trace-backed checks catch accounting-path cheats.

Use `--max-chunks` for a cheap prefix-only sweep across many submissions. That is a triage pass, not a full legality certificate.
Use `--trust-level basic|traced|strict` to tell the detector how much adapter evidence you require before calling the runtime surface credible.

It follows the score-first TTT contract that emerged around `parameter-golf` PRs `#461` and `#549`: score a chunk first, then adapt on that already-scored chunk.

Trust levels mean:

- `basic`: normalized sampled distributions plus causal repeatability and invariance probes
- `traced`: `basic`, plus explicit vocabulary size and trace-backed gold-score/accounting/path checks
- `strict`: `traced`, plus score-time state-hash consistency

If the requested trust level is not met, the legality report says so explicitly in its top-level `trust` block and adds an alert.

### Run finalist-strength replay

```bash
python -m conker_detect.cli replay \
  --profile parameter-golf \
  --adapter examples/packed_cache_demo_adapter.py \
  --adapter-config '{"mode":"legal","vocab_size":8}' \
  --tokens /tmp/conker_detect_tokens.npy \
  --chunk-size 4096 \
  --sample-chunks 4
```

This mode is for stronger replay summaries, not narrow legality probes. It reports:

- aggregate `total_loss_nats`, `mean_loss_nats`, and `mean_bpb`
- per-chunk replay summaries
- repeated-run drift on selected chunks
- gold-logprob drift summaries
- state-hash mismatch counts when trace-backed adapters expose them

Use this when a contender provides enough material for a replay-strength audit but you still want the simpler `legality` report for causal probe failures.

### Prepare a one-shot detector-to-ledger handoff

```bash
python -m conker_detect.cli handoff \
  records/track_non_record_16mb/2026-03-27_Demo \
  out/handoff \
  --bundle-id parameter-golf-pr-998 \
  --provenance-source provenance_manifest.json \
  --trust-level strict \
  --adapter examples/packed_cache_demo_adapter.py \
  --adapter-config '{"mode":"legal","vocab_size":8}' \
  --tokens /tmp/conker_detect_tokens.npy \
  --max-chunks 8
```

This is the convenience path when you already have one run directory and want `conker-ledger` inputs without hand-writing intermediate manifests.
By default, `conker-detect` infers the repo root from the nearest parent containing `records/` or `.git`; use `--repo-root` only when that inference would be wrong.

It writes:

- `reports/submission.json`
- optional `reports/provenance.json`
- optional `reports/legality.json`
- optional `reports/replay.json`
- `claim.json`
- `metrics.json`
- `provenance.json`
- `audits.json`
- `ledger_manifest.json`

The runtime half stays conservative on purpose. Even if the sampled legality and replay checks look clean, the synthesized bundle still stops at `Tier-1 reviewed` and marks Tier 3 as `warn` with scope `one_shot_runtime_handoff`. That keeps the convenience path from over-claiming full legality certification.
When runtime reports are present, `audits.json` also records the requested and achieved legality trust level so `conker-ledger` can surface whether the adapter evidence was merely basic or actually trace-backed.

### Write a `conker-ledger` manifest

```bash
python -m conker_detect.cli ledger-manifest out/bundle_manifest.json \
  --bundle-id parameter-golf-pr-998 \
  --claim claim.json \
  --metrics metrics.json \
  --provenance provenance.json \
  --audits audits.json \
  --submission-report out/submission.json \
  --provenance-report out/provenance.json \
  --legality-report out/legality.json \
  --replay-report out/replay.json
```

This writes the cross-repo handoff file for `conker-ledger bundle ...`, with the detector report attachments already placed under:

- `audits/tier1/submission.json`
- `audits/tier1/provenance.json`
- `audits/tier3/legality.json`
- `audits/tier3/replay.json`

### Package Audit Outputs

`conker-detect` is the detector layer. If you want a portable claim package, write the audit JSONs here and hand them to `conker-ledger bundle`:

```bash
python -m conker_detect.cli legality \
  --profile parameter-golf \
  --adapter submission_adapter.py \
  --tokens tokens.npy \
  --json out/legality.json

conker-ledger bundle manifest.json out/validity-bundle
```

That keeps structural and behavioral audit logic here, while `conker-ledger` handles claim metadata, provenance, copied reports, and public README packaging.

In practice, the manifest can point its `attachments` at files like `out/legality.json`, `out/matrix.json`, or `out/bundle.json` produced here.

## What To Look For

### Structured matrices

For matrices that are supposed to be causal or masked:

- `upper_frac`
- `diag_frac`
- `upper_plus_diag_frac`

If those are nontrivial where they should be zero, that is often more informative than any spectral statistic.

### Dense unrestricted weights

For ordinary dense matrices:

- `top{k}_energy_frac`
- `decay_1_to_last`
- `decay_1_to_{k}`

The absolute values are model-dependent. The main use is:

- compare suspicious vs reference
- find unexpectedly flat singular-value tails

### Lag / Toeplitz geometry

For square matrices that are supposed to behave like causal or near-causal lag operators:

- `active_mean`, `active_std`
- `mean_lag_std`
- Toeplitz deviation metrics
- residual norm / mean-absolute residual

This came directly from the `Conker-6` failure analysis, where a mask could look visually simple and still hide functionally important forbidden-region structure.

## Limitations

- not every side channel is spectral
- not every dense tail anomaly is malicious
- not every architecture has the same invariants
- a single-file scan is weaker than a clean-reference comparison
- runtime legality probes are only as strong as the adapter contract you give them
- challenge-specific legality still needs a clearly stated protocol
- passing the current runtime probes does not by itself rule out x_t-dependent accounting or best-of-`k` outcome selection

This is a detection aid, not a proof system.

## Provenance

This repository was spun out of the `Conker` research line, especially:

- `Conker-6` invalid causal-mask branch
- trained-model legality / normalization audits
- matrix-geometry and residual-substitution attacks

The original branch docs remain useful because they include the failed cases, not only the successful ones.
