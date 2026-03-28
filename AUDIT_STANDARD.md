# Audit Standard

`conker-detect` is an audit tool, not a leaderboard oracle. The right audit depends on what evidence a submission actually provides.

This document proposes a three-tier community audit standard for challenge submissions such as `parameter-golf`.

## Principle

There are two different questions:

1. Is this exact runtime artifact behaviorally legal?
2. Is this submission internally consistent and protocol-safe enough to take seriously?

Most public submission PRs only provide enough evidence for question 2. The audit standard should reflect that instead of pretending code review alone proves runtime legality.

## Tier 1

Required for every submission PR.

Goal:
- audit the submission claim, protocol shape, and reproducibility surface

Evidence expected:
- code
- README
- `submission.json`
- logs

Checks:
- score happens before any adaptation on the same scored region
- train and validation sources are separated
- no hidden mutable state in the score path
- no evaluation-time access to forbidden data sources
- claimed `val_bpb`, timing, chunk counts, and artifact bytes are internally consistent
- the PR is sufficiently reproducible to inspect and rerun

Outputs:
- `pass`, `warn`, or `fail`
- concrete file references
- JSON and short Markdown summary

What Tier 1 can answer:
- whether a submission is protocol-consistent on paper and in code
- whether the claim is credible enough to escalate

What Tier 1 cannot answer:
- whether the trained/exported runtime is actually legal
- whether the claimed score reproduces

## Tier 2

Required when a PR ships tensors, masks, checkpoints, or other serialized model state.

Goal:
- audit structural side channels and architecture-aware anomalies

Evidence expected:
- committed `.npy`, `.npz`, `.csv`, or similar tensor artifacts

Checks:
- forbidden upper-triangle or diagonal energy
- suspicious causal-mask geometry
- lag / Toeplitz residual structure
- dense spectral-tail anomalies
- poisoned-vs-reference or trained-vs-reference comparison when available

`conker-detect` support:
- `matrix`
- `geometry`
- `bundle`
- `compare`

What Tier 2 can answer:
- whether committed tensors contain obvious structural leakage or side channels

What Tier 2 cannot answer:
- whether the final runtime protocol is legal
- whether adaptation code mutates state illegally during scoring

## Tier 3

Required for leaderboard contention, explicit legality claims, or any submission asking to be treated as behaviorally verified.

Goal:
- audit real runtime behavior under the declared evaluation protocol

Evidence expected:
- a replayable artifact, or
- a standard adapter exposing score/adapt hooks, or
- a deterministic export path that rebuilds the eval artifact

Checks:
- score-phase repeatability from the same pre-score state
- future-suffix invariance for already-scored positions
- answer-mask invariance where applicable
- fresh-process rerun consistency
- challenge-specific protocol checks

For `parameter-golf`, the relevant contract is score-first TTT:
- score chunk `N`
- then adapt on chunk `N`
- chunk `N+1` may depend on chunks `0..N`, but not on any unscored tokens

`conker-detect` support:
- `legality --profile parameter-golf`
- `--max-chunks` for a cheap prefix triage pass

What Tier 3 can answer:
- whether the declared runtime protocol appears behaviorally legal

What Tier 3 cannot answer:
- whether the submission is globally optimal or scientifically interesting

## Required Labels

Suggested community labels:

- `Tier-1 reviewed`
- `Tier-2 structurally audited`
- `Tier-3 behaviorally audited`
- `Legally verified`

Only Tier 3 should justify a `Legally verified` label.

## Minimal Adapter Contract

Behavioral audits should not require one shared training stack. A small replay interface is enough:

```python
def build_adapter(config: dict) -> Runner: ...

class Runner:
    def fork(self) -> "Runner": ...
    def score_chunk(self, tokens, sample_positions=None) -> dict: ...
    def adapt_chunk(self, tokens) -> None: ...
```

When `sample_positions` are supplied, `score_chunk()` should return:

```python
{"sample_predictions": ...}
```

The predictions may be logits, log-probabilities, or probabilities, but they must be consistent across repeated calls.

## Practical Policy

Recommended policy for open submission repos:

1. Run Tier 1 on every PR.
2. Run Tier 2 whenever tensors or checkpoints are committed.
3. Require Tier 3 only for top contenders or explicit legality claims.
4. Treat cheap prefix legality checks as triage, not proof.

This keeps community auditing scalable without collapsing all review into prose scrutiny or, in the other direction, pretending every PR already includes enough material for full behavioral replay.
