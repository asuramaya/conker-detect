# Conker-11 legal replacement evidence for PR #998

This packet updates the still-open PR `#998` by replacing the invalid March 28, 2026 `Conker-5 tandem residual exact experts` claim with a fresh legal `Conker-11` rerun.

- track: `track_non_record_16mb`
- pre_quant_val_bpb: `2.0124189795892984`
- test_bits_per_token: `4.902112352537433`
- test_eval_loss: `3.397885355949402`
- seed: `43`
- recipe: `steps=1000 seq_len=256 batch_size=16 global_lag_cap=0.5`
- raw checkpoint bytes: `709480922`

Important scope note:

- this is a raw replay checkpoint and validity packet
- it is not a new packed `int6+zlib` artifact claim
- it is intended to update PR `#998` in place, not to open a different pull or pretend the old score survives

Canonical code and writeups now live in:

- `conker`: <https://github.com/asuramaya/conker>
- `conker-detect`: <https://github.com/asuramaya/conker-detect>
- `conker-ledger`: <https://github.com/asuramaya/conker-ledger>
- `giddy-up`: <https://github.com/asuramaya/giddy-up>
