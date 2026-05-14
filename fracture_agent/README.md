# fracture_agent — agentic phase-field fracture FEA

Natural-language driver for the `modular/` phase-field codes.
Type (or sketch, or say) what you want — the agent distils a canonical
spec, selects the right modular variant, synthesises a runnable DOLFINx
script, executes it inside WSL, watches for divergence, retries on errors,
then answers free-form questions about the result.

```
                  ┌───────────┐
  text / image ──▶│ Recept.   │──fragments──┐
  / audio         └───────────┘             │
                                            ▼
                  ┌───────────┐      ┌───────────┐
                  │ Architect │◀────▶│  User Q&A │   (clarifications, if any)
                  └─────┬─────┘      └───────────┘
                        │ CanonicalSpec
                        ▼
                  ┌───────────┐
                  │ Strategist│─────────▶ Action (pick 1 of 9 variants)
                  └─────┬─────┘
                        │
                        ▼
                  ┌───────────┐
                  │ Synthesiser│────────▶ runs/<sid>/run_<variant>.py
                  └─────┬─────┘
                        │
                        ▼
                  ┌───────────┐
                  │ Inspector │──AST + legacy-API check
                  └─────┬─────┘
                        │
                        ▼
                  ┌───────────┐              ┌──────────┐
                  │ Executor  │─fail─▶─────▶│ Debugger │──patch──┐
                  │ (WSL+MPI) │               └──────────┘         │
                  └─────┬─────┘                                    │
                        │ok                                        ▼
                        ▼                                   re-run / rescale-eps
                  ┌───────────┐              ┌───────────┐
                  │ Advisor   │──verdict────▶│  Q&A loop │
                  └───────────┘              └───────────┘
```

## Install

On Windows (PowerShell or bash):

```
pip install requests pydantic flask
```

FEniCSx itself is **not** a Python dep of the agent — it runs inside WSL via
`wsl.exe`.  The agent only needs Python 3.9+ and an active conda env
`fenicsx` on the WSL side (same as `modular/README.md`).

## Usage

### Browser UI (recommended)

```
python -m fracture_agent.main --ui
```

Open http://127.0.0.1:7860 — a single page with:

* **Prompt box** plus optional overrides (material, geometry kind, physics
  mode, fracture on/off).
* **Agent activity** — live status: "Parsing prompt", "Architect picking
  variant", "Running in WSL", "Uh-oh, debugging", ...
* **Decisions & assumptions** — what the agent inferred (plane stress vs
  strain, catalog match vs handbook lookup, mesh plan, BCs).
* **Console** — streaming stdout from the DOLFINx solver.
* **Verdict + metrics** — plain-English summary from the Advisor, plus
  cracked/min-z/peak-Fy cards.
* **Visualisations** — initial mesh, final deformed config coloured by the
  phase field, load-displacement curve.
* **Chat box** — ask follow-up questions grounded in the run's log and
  metrics ("did it crack?", "what's the peak reaction force?").

### CLI

```
# Interactive:
python -m fracture_agent.main

# With a single prompt:
python -m fracture_agent.main --prompt "3 mm steel circle pulled at one end by 3 N"

# Text + sketch:
python -m fracture_agent.main --prompt "plate with a hole" --image ./sketch.png

# Parallel (4 MPI ranks):
python -m fracture_agent.main --prompt "..." --nprocs 4

# Stop before running the generated script:
python -m fracture_agent.main --prompt "..." --no-execute

# Resume a run (re-open Q&A, or continue after a crash):
python -m fracture_agent.main --resume run_20260424_120000
```

All artefacts land in `agentic_simulations/<slug>/`, one subfolder per run.
The slug is `<YYYYMMDD_HHMMSS>_<material>_<shape>_<loading>`, e.g.

```
agentic_simulations/
├── 20260424_141530_copper_rectangle_tension/
├── 20260424_142012_graphite_custom_sympull/
└── 20260424_142530_alumina_l_shape_tension/
```

Inside each session folder:

* `state.json`           — full session state (resumable via `--resume <slug>`)
* `run_<variant>.py`     — generated driver
* `run_<variant>.attempt1.py`, ... — debugger patches
* `custom_mesh.py`       — LLM-generated gmsh module (custom geometries only)
* `material.json`        — ad-hoc material entry (when no catalog match)
* `paraview_<...>/`      — XDMF snapshots (per accepted step)
* `output_<...>.txt`     — per-step log (step, time, Δt, u_res, z_res,
                           min_z, disp, Fy)

## Supported problem variants (the RAG skeletons)

| variant                 | dim | constitutive        | mode         |
|-------------------------|:---:|---------------------|--------------|
| `linear_elastic_2d_pe`  |  2  | linear elastic      | quasistatic  |
| `linear_elastic_2d_ps`  |  2  | linear elastic      | quasistatic  |
| `linear_elastic_3d`     |  3  | linear elastic      | quasistatic  |
| `dynamic_2d`            |  2  | linear elastic      | dynamic HHT-α|
| `ductile_2d_pe`         |  2  | J2 plasticity       | quasistatic  |
| `ductile_3d`            |  3  | J2 plasticity       | quasistatic  |
| `finite_elastic_2d_pe`  |  2  | Lopez-Pamies Ogden  | quasistatic  |
| `finite_elastic_2d_ps`  |  2  | Lopez-Pamies Ogden  | quasistatic  |
| `finite_elastic_3d`     |  3  | Lopez-Pamies Ogden  | quasistatic  |

The Strategist picks exactly one variant by metadata match.  Fracture can be
disabled (deformation-only) on any variant by saying "no crack" / "just
deformation" — `Gc` is inflated by 1e8 so the phase field stays at z ≈ 1.

## Mesh sizing rule

The agent obeys the user-mandated rule exactly:

1. Start with `h0 = 2·eps` derived from the material (same convention as
   `modular/materials/loader.py`).
2. After the first cold run the template prints `[mesh-audit] n_cells_global
   = N`.  If N < `target_min_cells` (default 5000), rescale
      `eps_new = (N / target)**dim * eps_old`
   then re-synthesise and re-run.  Up to 3 rescales.

No box- or tip-refinement is added by default — the modular AMR machinery is
still live inside the solver and handles the damage zone automatically.

## LLM backend

API keys are loaded from environment variables (`GEMINI_API_KEY`,
`GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`). Up to three keys are rotated per
request to soak up free-tier RPM limits. See `.env.example` at the repo root.

**Model routing** (edit `config.py` to override):

| agent | model | why |
|---|---|---|
| Receptionist, Architect, Debugger, Mesh-LLM, Material-Helper | `FAST_MODEL` = `gemini-2.5-flash` | pure structured extraction; Pro's thinking loops cause deadlocks here |
| Advisor (plain-English verdict + Q&A) | `PRIMARY_MODEL` = `gemini-2.5-pro` | reasoning over physics |
| Strategist / Synthesizer / Inspector | no LLM | deterministic code |

To force Pro for everything, set `FAST_MODEL = PRIMARY_MODEL` in `config.py`.

**Material lookup cache** lives at `.fracture_agent_cache/materials.json` — any
handbook lookup is saved and re-used across sessions.  Delete the file to
force a fresh lookup.

**Key health**: on transient 429/503 a key is cooled down for the server-
suggested retry-delay; terminal errors (`monthly spending cap`, `billing`,
`project suspended`, `API key is invalid`) disable the key permanently.  If
every key is permanently disabled the run exits with a clear message.

## Extending

* **New variant**: add an entry in `knowledge.CATALOG` plus a row in
  `templates.VARIANT_META` — the rest of the pipeline adapts automatically.
* **New geometry kind**: add a branch in `templates._mesh_call` and a
  `make_*` row in `knowledge.CATALOG.mesh_builder`.
* **Custom BCs**: today the template expects `fixed_regions` + a single
  `loaded_region`.  For more exotic BC schedules, edit `templates._bc_block`.

## Files

```
fracture_agent/
├── __init__.py
├── README.md                 ← this file
├── config.py                 ← API keys, model name, paths, WSL env
├── llm.py                    ← Gemini REST client with key cycling
├── schema.py                 ← Pydantic canonical spec
├── state.py                  ← Resumable session state
├── knowledge.py              ← 9-variant catalog (metadata RAG)
├── templates.py              ← Runnable-script renderer
├── mesh.py                   ← Mesh-sizing rule + custom gmsh LLM helper
├── executor.py               ← wsl.exe streaming runner + divergence watchdog
├── results.py                ← Log / XDMF parser, ResultSummary
├── orchestrator.py           ← State-machine gluing agents together
├── main.py                   ← argparse CLI
└── agents/
    ├── receptionist.py
    ├── architect.py
    ├── material_helper.py
    ├── strategist.py
    ├── synthesizer.py
    ├── inspector.py
    ├── debugger.py
    └── advisor.py
```
