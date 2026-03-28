# conker-detect

`conker-detect` is a small defensive audit tool for model weights and checkpoint bundles.

It grew out of the `Conker-6` failure mode: a matrix that looked almost identical to a clean causal mask, but changed behavior massively because it had learned illegal diagonal and upper-triangle structure. The immediate lesson was:

- some side channels are easiest to catch with architecture-aware invariants
- others are easier to catch with spectral-tail or bundle-to-bundle comparison

This tool packages both.

## What It Does

`conker-detect` supports three audit modes:

1. `matrix`
- inspect a single `.npy` or `.csv` matrix
- compute spectral concentration and decay statistics
- compute region energy for square matrices
- optionally compare against a clean reference

2. `bundle`
- inspect all 2D tensors in an `.npz` checkpoint bundle
- report per-tensor spectral and structural metrics
- optionally flag tensors that are expected to be strict-lower causal

3. `compare`
- compare matching 2D tensors across two `.npz` bundles
- useful for poisoned-vs-clean or trained-vs-reference analysis

## Why This Exists

The main `Conker-6` lesson was not just “this branch was invalid.” It was:

- trained-model audits matter more than init audits
- a tiny forbidden-region perturbation can matter more than broad geometric similarity
- cosine and visual similarity can completely miss a functional leak

So `conker-detect` is deliberately simple:

- architecture-aware invariant checks for structured matrices
- spectral/tail checks for dense unrestricted weights
- reference comparison when you have two binaries

It is not a universal proof of cleanliness. It is a fast triage tool.

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

### Audit a checkpoint bundle

```bash
conker-detect bundle model.npz --expect-causal mask --expect-causal causal
```

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

## Limitations

- not every side channel is spectral
- not every dense tail anomaly is malicious
- not every architecture has the same invariants
- a single-file scan is weaker than a clean-reference comparison

This is a detection aid, not a proof system.

## Provenance

This repository was spun out of the `Conker` research line, especially:

- `Conker-6` invalid causal-mask branch
- trained-model legality / normalization audits
- matrix-geometry and residual-substitution attacks

The original branch docs remain useful because they include the failed cases, not only the successful ones.
