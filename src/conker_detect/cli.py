from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import (
    audit_bundle,
    audit_matrix,
    compare_bundles,
    load_matrix,
    mask_geometry_stats,
)


def _write_output(text: str, json_path: str | None) -> None:
    print(text)
    if json_path:
        out_path = Path(json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect hidden side channels and structural anomalies in matrices and NPZ checkpoint bundles."
    )
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

    p_bundle = sub.add_parser("bundle", help="Audit all 2D tensors in an .npz checkpoint")
    p_bundle.add_argument("bundle")
    p_bundle.add_argument("--topk", type=int, default=16)
    p_bundle.add_argument("--only-square", action="store_true")
    p_bundle.add_argument("--name-regex")
    p_bundle.add_argument("--expect-causal", action="append", default=[], help="Substring marking tensors that should be strict-lower causal")
    p_bundle.add_argument("--json")

    p_compare = sub.add_parser("compare", help="Compare matching 2D tensors across two .npz checkpoints")
    p_compare.add_argument("lhs_bundle")
    p_compare.add_argument("rhs_bundle")
    p_compare.add_argument("--topk", type=int, default=16)
    p_compare.add_argument("--only-square", action="store_true")
    p_compare.add_argument("--name-regex")
    p_compare.add_argument("--strip-prefix", action="append", default=[], help="Prefix to strip before name matching, e.g. student.")
    p_compare.add_argument("--json")

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

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
