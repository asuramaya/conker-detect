# conker-detect

`conker-detect` is a small defensive audit tool for model weights, checkpoint bundles, packed artifacts, and runtime legality probes.

It grew out of the `Conker-6` failure mode: a matrix that looked almost identical to a clean causal mask, but changed behavior massively because it had learned illegal diagonal and upper-triangle structure. The immediate lesson was:

- some side channels are easiest to catch with architecture-aware invariants
- others are easier to catch with spectral-tail or bundle-to-bundle comparison
- artifact-boundary bugs can be just as important as model bugs

This tool packages both.

## What It Does

`conker-detect` supports six audit modes:

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
- inspect all 2D tensors in an `.npz` checkpoint bundle
- report per-tensor spectral and structural metrics
- optionally flag tensors that are expected to be strict-lower causal
- useful for telling raw replay checkpoints apart from packed submission payloads

4. `compare`
- compare matching 2D tensors across two `.npz` bundles
- useful for poisoned-vs-clean or trained-vs-reference analysis

5. `artifact`
- inspect a packed `.ptz` artifact
- classify payload entries as deterministic substrate, structural control, or learned payload
- surface raw-vs-compressed byte stories and suspicious packed names directly

6. `legality`
- run behavioral legality probes against a live submission adapter
- check score-phase repeatability and causal invariance
- support challenge-specific profiles such as score-first `parameter-golf` TTT

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

This mode is also the quickest way to check whether a raw replay checkpoint is carrying large deterministic tensors that should have been regenerated instead of packed.

Only square tensors:

```bash
conker-detect bundle model.npz --only-square
```

### Compare two checkpoint bundles

```bash
conker-detect compare model_a.npz model_b.npz --only-square
```

If one checkpoint wraps another under prefixes like `student.`:

```bash
conker-detect compare base.npz wrapped.npz --strip-prefix student.
```

### Audit a packed artifact

```bash
conker-detect artifact model.int6.ptz
```

This is the quickest way to answer:

- did regenerated deterministic substrate get packed by mistake?
- which tiny structural tensors are crossing the artifact boundary?
- how much of the raw payload is actually structural control?

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
- repeatability from the same pre-score state
- score-phase invariance when future suffix tokens are randomized
- answer-mask invariance when the token being scored is randomized too
- optional gold-logprob consistency when the adapter exposes scored gold-token logprobs

Those probes are deliberately narrower than a full legality proof. In particular:

- answer-mask invariance is a same-position leakage probe, not the whole causal contract
- future-suffix invariance is stronger, but still sampled rather than exhaustive
- normalization now also checks that sampled prediction vectors match the declared vocabulary size
- score accounting is only covered if the adapter exposes `sample_gold_logprobs`
- best-of-`k` outcome selection remains out of scope for the current adapter contract

Packed-cache demo:

```bash
python -m conker_detect.cli legality \
  --profile parameter-golf \
  --adapter examples/packed_cache_demo_adapter.py \
  --adapter-config '{"mode":"legal","vocab_size":8}' \
  --tokens /tmp/conker_detect_tokens.npy
```

Switch `mode` to `self_include` or `future_peek` to see the detector catch score-after-update and future-leak behavior.
Switch `mode` to `reported_gold_cheat` to see the detector catch a hidden gold-token scoring path that does not match the returned full distribution.

Use `--max-chunks` for a cheap prefix-only sweep across many submissions. That is a triage pass, not a full legality certificate.

It follows the score-first TTT contract that emerged around `parameter-golf` PRs `#461` and `#549`: score a chunk first, then adapt on that already-scored chunk.

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
