## PR #998 Update Packet

This example is a modest, explicit update packet for the still-open [openai/parameter-golf PR #998](https://github.com/openai/parameter-golf/pull/998).

It does not try to preserve the invalid historical `Conker-5` headline. It replaces that claim with a fresh legal `Conker-11` rerun and supporting audit material.

Current attached run:

- run id: `conker11_seed43_replacement_2026-03-29`
- fresh-process full `bpb`: `2.0124189795892984`
- `bpt`: `4.902112352537433`
- eval loss: `3.397885355949402`
- train time: `157.92236804962158s`
- raw replay checkpoint bytes: `709480922`

Why this exists:

- PR `#998` is still open and still advertises an invalid `Conker-5` score.
- the update should be honest, smaller in scope, and validity-first
- the packet points back to the living repos instead of pretending the old record folder is still the right source of truth

Companion repos:

- `conker`: <https://github.com/asuramaya/conker>
- `conker-detect`: <https://github.com/asuramaya/conker-detect>
- `conker-ledger`: <https://github.com/asuramaya/conker-ledger>
- `giddy-up`: <https://github.com/asuramaya/giddy-up>

This directory contains:

- `source_submission/`: a detector-friendly replacement submission packet
- `provenance_manifest.json`: explicit disclosure that the reported row was selected as the best of `2` fresh legal runs
- `handoff/`: synthesized detector outputs and ledger manifest
