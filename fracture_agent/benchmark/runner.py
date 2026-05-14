"""Benchmark runner — execute a prompt suite at every ablation level.

Reads JSON files from ``fracture_agent/benchmark/suite/*.json`` and for each
problem runs the four ablation levels (B1 one-shot → B4 full fracture_agent).
Emits a single ``benchmark_results.csv`` with one row per (problem,
level, repetition) carrying the columns reviewers expect:

    problem | level | rep | rc | diverged | n_steps | min_z | peak_F |
    health_total | health_verdict | wall_solver_s | wall_total_s |
    n_llm_calls | prompt_tokens | output_tokens | cost_usd |
    architect_rounds | debugger_attempts | reflect_revise_cycles

Multi-run consistency (B3 from the SOTA report) is supported via the
``--reps`` flag — each (problem, level) pair runs ``reps`` times and
aggregation is done by the caller (e.g. pandas).

Usage::

    python -m fracture_agent.benchmark.runner --suite tier1 --reps 3 \\
            --levels 1,2,3,4 --out benchmark_results.csv

Each run lives in its own ``agentic_simulations/<slug>/`` folder so the
state, telemetry and cost tables are inspectable per case.

Schema of a benchmark JSON entry::

    {
      "id": "miehe_sent",
      "name": "Miehe SENT (single-edge notched tension)",
      "tier": 1,
      "tags": ["brittle", "mode_I"],
      "prompts": {
        "tier1_full": "10 mm by 20 mm... E=210 GPa, nu=0.3, ...",
        "tier2_partial": "steel SENT plate, mode-I, ...",
        "tier3_figure_only": "reproduce Fig X of Miehe 2010 (DOI ...)"
      },
      "expected": {
        "peak_F_range": [...],
        "crack_initiation_step_range": [40, 80],
        "min_z_below": 0.1
      }
    }

The ``expected`` block is consulted only by post-hoc analysis tools; the
runner itself just records the observed values.  Fully-empty
``expected: {}`` is fine.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..ablation import AblationLevel, run_at_level
from ..config import RUNS_DIR, WSL_MPI_DEFAULT
from ..events import set_sink
from ..state import SessionState
from ..telemetry import detach_telemetry, set_active


def _silent_asker(qs):
    return [""] * len(qs)


SUITE_DIR = Path(__file__).resolve().parent / "suite"


def list_suite(suite_name: Optional[str] = None) -> List[Path]:
    """Return JSON files in the named suite directory.

    ``suite_name`` is a sub-directory name (e.g. ``tier1``); if ``None``,
    every JSON in ``benchmark/suite/`` is loaded recursively.
    """
    if suite_name:
        d = SUITE_DIR / suite_name
        if not d.exists():
            raise FileNotFoundError(f"benchmark suite not found: {d}")
        return sorted(d.glob("*.json"))
    return sorted(SUITE_DIR.rglob("*.json"))


def load_problem(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_for(state: SessionState,
             rec, health, *,
             problem_id: str, level_name: str, rep: int,
             tier_key: str, wall_total_s: float) -> Dict[str, Any]:
    """One CSV row — a flat projection of state.{telemetry, runs, …}."""
    tot = state.telemetry.totals()
    iters = state.telemetry.iters
    summary = state.final_result
    return {
        "problem":             problem_id,
        "tier":                tier_key,
        "level":               level_name,
        "rep":                 rep,
        "rc":                  rec.returncode if rec else None,
        "diverged":            (rec.diverged if rec else None),
        "n_steps":             (summary.n_steps if summary else None),
        "min_z":               (summary.min_z if summary else None),
        "peak_F":              (summary.peak_reaction if summary else None),
        "cracked":             (summary.cracked if summary else None),
        "health_total":        (health.total if health else None),
        "health_verdict":      (health.verdict if health else None),
        "health_flags":        ("|".join(sorted(set(health.flags)))
                                if health else ""),
        "wall_solver_s":       round(state.telemetry.wall_time_solver_s, 3),
        "wall_total_s":        round(wall_total_s, 3),
        "n_llm_calls":         tot["n_llm_calls"],
        "prompt_tokens":       tot["prompt_tokens"],
        "output_tokens":       tot["output_tokens"],
        "thinking_tokens":     tot["thinking_tokens"],
        "cost_usd":            tot["cost_usd"],
        "architect_rounds":    iters.architect_rounds,
        "debugger_attempts":   iters.debugger_attempts,
        "mesh_rescales":       iters.mesh_rescales,
        "reflect_revise":      iters.reflect_revise_cycles,
        "session_id":          state.session_id,
    }


def run_one(problem: Dict[str, Any],
             *, tier_key: str, level: AblationLevel, rep: int,
             nprocs: int, execute: bool) -> Dict[str, Any]:
    prompt = problem["prompts"][tier_key]
    state = SessionState.new()
    set_active(state.telemetry)
    t0 = time.monotonic()
    try:
        script, rec, health = run_at_level(
            state, [("text", prompt)],
            level=level, ask_user=_silent_asker,
            nprocs=nprocs, execute=execute)
    except Exception as e:
        # Failure during the run itself — record a row with error.
        rec = None
        health = None
        state.save()
        wall = time.monotonic() - t0
        return {
            **_row_for(state, rec, health,
                        problem_id=problem["id"],
                        level_name=level.name,
                        rep=rep, tier_key=tier_key,
                        wall_total_s=wall),
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        detach_telemetry()
    wall = time.monotonic() - t0
    row = _row_for(state, rec, health,
                    problem_id=problem["id"],
                    level_name=level.name,
                    rep=rep, tier_key=tier_key,
                    wall_total_s=wall)
    row["error"] = ""
    return row


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="fracture_agent.benchmark.runner")
    ap.add_argument("--suite", default=None,
                    help="Subdirectory of benchmark/suite/ to load "
                         "(default: every JSON in suite/).")
    ap.add_argument("--levels", default="1,2,3,4",
                    help="Comma-separated ablation levels to execute.")
    ap.add_argument("--tiers", default="tier1_full",
                    help="Comma-separated tier keys to run "
                         "(default: tier1_full only).")
    ap.add_argument("--reps", type=int, default=1,
                    help="Repetitions per (problem, level, tier) — for "
                         "multi-run consistency (default 1).")
    ap.add_argument("--nprocs", type=int, default=WSL_MPI_DEFAULT,
                    help="MPI ranks for WSL (default 1).")
    ap.add_argument("--no-execute", action="store_true",
                    help="Generate scripts only; do not run in WSL.")
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: ./benchmark_results.csv "
                         "next to the runs directory).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional: cap number of problems (for smoke tests).")
    args = ap.parse_args(argv)

    # Silence event-stream printing so the runner's own status dominates.
    set_sink(None)

    problem_files = list_suite(args.suite)
    if args.limit:
        problem_files = problem_files[: args.limit]
    if not problem_files:
        print(f"[benchmark] no problems found in suite={args.suite!r} "
              f"({SUITE_DIR}); nothing to do.", file=sys.stderr)
        return 1

    levels = [AblationLevel(int(s)) for s in args.levels.split(",")]
    tiers = args.tiers.split(",")
    out_csv = Path(args.out) if args.out else (RUNS_DIR.parent
                                                / "benchmark_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] {len(problem_files)} problems × {len(tiers)} tiers × "
          f"{len(levels)} levels × {args.reps} reps = "
          f"{len(problem_files) * len(tiers) * len(levels) * args.reps} runs")
    print(f"[benchmark] writing CSV to {out_csv}")

    fieldnames = [
        "problem", "tier", "level", "rep", "rc", "diverged",
        "n_steps", "min_z", "peak_F", "cracked",
        "health_total", "health_verdict", "health_flags",
        "wall_solver_s", "wall_total_s",
        "n_llm_calls", "prompt_tokens", "output_tokens", "thinking_tokens",
        "cost_usd", "architect_rounds", "debugger_attempts",
        "mesh_rescales", "reflect_revise",
        "session_id", "error",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        idx = 0
        n_total = len(problem_files) * len(tiers) * len(levels) * args.reps
        for pf in problem_files:
            problem = load_problem(pf)
            for tier in tiers:
                if tier not in problem.get("prompts", {}):
                    print(f"[benchmark] skip {problem['id']} "
                          f"(no tier '{tier}')")
                    continue
                for level in levels:
                    for rep in range(1, args.reps + 1):
                        idx += 1
                        print(f"[{idx}/{n_total}] {problem['id']} "
                              f"tier={tier} level={level.name} rep={rep}")
                        row = run_one(problem,
                                       tier_key=tier, level=level, rep=rep,
                                       nprocs=args.nprocs,
                                       execute=not args.no_execute)
                        w.writerow(row)
                        f.flush()
    print(f"[benchmark] done -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
