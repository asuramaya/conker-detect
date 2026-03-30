"""Update-packet wrapper for the canonical Conker-11 runner.

This file exists only to give the detector a concrete reproducibility surface.
The source of truth is the canonical `conker` repo and its `run_conker11_golf_bridge.py`
runner.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Use the canonical Conker-11 runner in the conker repo; this packet is an audit wrapper, not the source of truth."
    )


if __name__ == "__main__":
    main()
