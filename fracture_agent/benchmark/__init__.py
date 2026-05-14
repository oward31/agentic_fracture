"""Benchmark suite for fracture_agent.

Two pieces:
  * ``runner.py``  — execute a JSON-defined prompt suite at the four
                     ablation levels (B1/B2/B3/B4); aggregate per-case
                     and per-level metrics.
  * ``suite/``     — JSON files, one per problem, with prompt(s) and
                     expected outputs (load-disp range, crack-path,
                     energy-balance %).
"""
