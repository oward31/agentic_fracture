"""CLI: build / inspect the RAG indices.

    python -m fracture_agent.rag build               # build skeleton + snippet
    python -m fracture_agent.rag query  "<query>"     # top-K skeleton + snippet matches
    python -m fracture_agent.rag stats               # one-line stats per index
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from . import (build_all_indices, build_error_fix_index,
                build_skeleton_index, build_snippet_index, index_path,
                retrieve_skeleton, retrieve_snippets, retrieve_error_fix,
                sigma_gate)


def cmd_build(args) -> int:
    paths = build_all_indices()
    for k, p in paths.items():
        print(f"{k:10s} -> {p}")
    return 0


def cmd_query(args) -> int:
    q = args.query
    print(f"=== Skeleton (k={args.k}) ===")
    sk = retrieve_skeleton(q, k=args.k)
    top, all_ = sigma_gate(sk, threshold=args.threshold)
    for r in sk:
        marker = "*" if (top is not None and r.id == top.id) else " "
        print(f" {marker} sim={r.score:.3f}  {r.id}  "
              f"variant={r.metadata.get('variant')}")
    if top is None:
        print(f"   (no match >= sim={args.threshold:.2f})")
    print(f"\n=== Snippet (k={args.k}) ===")
    for r in retrieve_snippets(q, k=args.k):
        print(f"   sim={r.score:.3f}  {r.id}")
    print(f"\n=== Error-Fix (k={args.k}) ===")
    for r in retrieve_error_fix(q, k=args.k):
        print(f"   sim={r.score:.3f}  {r.id}")
    return 0


def cmd_stats(args) -> int:
    for name in ("skeleton", "snippet", "error_fix"):
        p = index_path(name)
        if not p.exists():
            print(f"{name:10s}: (not built)")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            n = len(data.get("items") or [])
            model = data.get("model")
            print(f"{name:10s}: {n} items  model={model}  hash={data.get('corpus_hash')}")
        except Exception as e:
            print(f"{name:10s}: <error reading: {e}>")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fracture_agent.rag")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build", help="Build/refresh skeleton + snippet + error_fix indices.")
    qp = sp.add_parser("query", help="Top-K matches for a query string.")
    qp.add_argument("query")
    qp.add_argument("-k", type=int, default=5)
    qp.add_argument("--threshold", type=float, default=0.85)
    sp.add_parser("stats", help="Print one-line stats per index.")
    args = ap.parse_args(argv)
    return {"build": cmd_build, "query": cmd_query,
             "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
