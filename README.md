# agentic_fracture

**Type a sentence. Get a verified phase-field fracture simulation.**

`agentic_fracture` is a small team of LLM-driven agents that take a plain
English description of a fracture mechanics problem, infer the geometry,
material properties, boundary conditions, and the right physics variant,
synthesise a runnable [DOLFINx](https://github.com/FEniCS/dolfinx) script,
execute it (in WSL), watch for divergence, retry on errors, score the result
against a 100-point physics-aware rubric, and answer follow-up questions
about what happened.

> _"10 mm × 20 mm graphite sheet, plane stress, with a crack from the top
> centre edge to the middle, bottom fixed, top half of the left and right
> edges pulled apart by 1 mm each."_

End-to-end in roughly **1–3 minutes**: a generated driver script, a custom
gmsh mesh, the per-step log, XDMF snapshots for ParaView, a load-displacement
curve, the deformed damage field, and a chat box to ask questions about the
run.

---

## Why this exists

Phase-field fracture is one of the most expressive tools in computational
mechanics, but writing a correct DOLFINx driver is genuinely hard:

`agentic_fracture` takes a guess at every one of the decisions, writes
the assumption down so you can audit it, runs the simulation, and tells you
whether the result is physically plausible.

---

## Features

| | |
|---|---|
| **Plain-English input** | Text, sketches, or voice clips. A vision-language pass extracts dimensions from hand-drawn figures and cross-checks them against the prompt. |
| **Multi-agent pipeline** | Nine single-purpose agents (Receptionist, Architect, Material Helper, Strategist, Synthesiser, Inspector, Executor, Debugger, Advisor) wired together as a plain Python state machine — no LangGraph, no LangChain. |
| **Autonomous decisions** | Plane stress vs strain, catalog vs handbook material, mesh kind, time stepping, multi-stage loading windows. The agent only asks for clarification when it genuinely cannot decide. |
| **Material resolution** | Four-tier hierarchy: built-in catalog → user-supplied numbers → on-disk handbook cache → live Gemini + Google-Search lookup → typed backfill with audit trail. |
| **Geometry handling** | Five built-in mesh builders (notched plate, slant plate, dogbone, 2D and 3D variants) plus an LLM-written `custom_mesh.py` for arbitrary shapes. Marker names are AST-validated against what the BCs expect. |
| **9 physics variants** | 2D plane strain / 2D plane stress / 3D × linear elastic / J2 plasticity / Lopez–Pamies finite elasticity × quasistatic / dynamic. |
| **Boundary conditions** | Multi-region, partial-edge, and multi-stage `(t_start, t_end)` loading windows. Automatic rigid-body suppression when the user's BCs underdetermine the problem. Directional sign correction from prompt verbs ("squeezed" vs "pulled apart"). |
| **Adaptive mesh sizing** | `h₀ = 2·ε`; if the cold mesh has fewer than 5000 cells the agent rescales `ε ← (n/target)^(1/dim) · ε` and retries up to 6 times. |
| **Adaptive time stepping** | Halve and revert the step on staggered residual blow-up or SNES variational-inequality divergence. Floor at `Δt₁/10`. |
| **Self-debugging** | Up to 5 LLM-driven patch attempts. Successful patches are archived in an Error-Fix RAG index so the next run gets the fix for free. |
| **Reflect-Revise outer loop** | A 100-point physics rubric drives a deterministic Reviser that nudges scalar knobs (loading magnitude, step count, stagger tolerance, ε override). Up to two outer cycles. |
| **Hierarchical RAG** | Three indices over the user's verified physics codes: `skeleton` (whole files, queried by Strategist), `snippet` (AST function-level, queried by Mesh-LLM and Synthesiser), and `error_fix` (self-populating, queried by Debugger). |
| **4-level ablation harness** | One CLI flag toggles between (1) raw one-shot LLM, (2) +Inspector, (3) full inner loop, (4) full pipeline including Reflect-Revise. One CSV row per case. |
| **Telemetry built in** | Every Gemini call is tagged with the active agent. Tokens, USD cost, latency, and wall-time aggregate per session and per agent into a CSV next to every run. |
| **Browser UI** | A single page that streams agent activity, decisions, console output, the verdict, metric cards, the final mesh, the deformed damage field, and a load-displacement plot — plus a chat box for follow-ups. |
| **Resumable** | Every run is a self-contained slug-named folder with a `state.json`. Re-open a run with `--resume <slug>` to ask more questions or re-render visuals. |
| **Key rotation** | Up to three API keys are cycled per request; transient 429/503s are cooled down, terminal billing/auth errors permanently disable the offending key. |

---

## Quick start

### 1. Get the code

```bash
git clone https://github.com/<your-username>/agentic_fracture.git
cd agentic_fracture
```

### 2. Configure your API key

The agent talks to Google's Gemini API. Grab a free key at
[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey),
then:

```bash
cp .env.example .env
# Edit .env and paste your key into GEMINI_API_KEY
```

One key is enough to start. If you have two or three, drop them in
`GEMINI_API_KEY_2` / `GEMINI_API_KEY_3` to soak up free-tier RPM limits.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs only the agent-side dependencies. The DOLFINx side (FEniCSx,
PETSc, gmsh, h5py, matplotlib) lives in a WSL conda environment — see the
WSL setup section below.

### 4. Run

```bash
# Browser UI (recommended):
python -m fracture_agent.main --ui
# then open http://127.0.0.1:7860

# CLI, single prompt:
python -m fracture_agent.main --prompt "50 mm by 10 mm copper bar, plane stress, pulled vertically by 0.5 mm"

# CLI, interactive:
python -m fracture_agent.main
```

Every run lands in
`agentic_simulations/<YYYYMMDD_HHMMSS>_<material>_<shape>_<loading>/`.

---

## CLI reference

```bash
python -m fracture_agent.main --prompt "..."          # single shot
python -m fracture_agent.main                         # interactive
python -m fracture_agent.main --prompt "..." --image sketch.png       # multimodal
python -m fracture_agent.main --prompt "..." --audio note.wav         # voice input
python -m fracture_agent.main --prompt "..." --nprocs 4               # MPI-parallel solve
python -m fracture_agent.main --prompt "..." --no-execute             # stop after script gen
python -m fracture_agent.main --resume 20260424_141530_graphite_custom_sympull
python -m fracture_agent.main --prompt "..." --ablation 3             # 1=raw LLM, 4=full pipeline
python -m fracture_agent.main --ui --port 7860
```

---

## Prerequisites

| where  | what |
| ------ | ---- |
| Windows | Python 3.9+, the deps in `requirements.txt`, and WSL2. |
| WSL    | A conda environment named `fenicsx` (override with `FRACTURE_AGENT_CONDA_ENV`) containing DOLFINx 0.9, petsc4py, gmsh, h5py, and matplotlib. |
| Account | A Gemini API key. Free-tier RPM is enough for roughly one full run per minute. |

The agent currently shells out to `wsl.exe` for execution, so it is
Windows-first. Linux/macOS support is straightforward (replace the WSL
launcher with a direct conda call) but not yet packaged.

### Setting up the WSL side

Inside WSL (Ubuntu 22.04 or later):

```bash
# Install miniconda if you don't already have it.
# Then:
conda create -n fenicsx -c conda-forge fenics-dolfinx=0.9 mpich pyvista gmsh
conda activate fenicsx
pip install h5py matplotlib
```

Verify with:

```bash
conda activate fenicsx
python -c "import dolfinx; print(dolfinx.__version__)"
```

---

## How it works

The pipeline is a plain state machine — each box is one Python function
that reads and mutates a shared `SessionState` (persisted as `state.json`,
so any run is fully resumable):

```
        user prompt + optional image / audio
                       │
                       ▼
                 Receptionist        ── multimodal → fragments
                       │
                       ▼
                  Architect          ── fragments → CanonicalSpec
                       │
        ┌──────────────┴── open questions? ──── ask user ──┐
        │                                                  │
        ▼                                                  ▼
  Material Helper                            (loop until clean)
        │ catalog → cache → handbook → backfill
        ▼
   Strategist            ── pick one of 9 physics variants
        │
        ▼
   Synthesiser           ── render driver script + custom_mesh.py
        │
        ▼
    Inspector            ── AST + grep + region validation
        │
        ▼
    Executor             ── wsl.exe → conda → mpirun
        │
        ├── rc != 0 ──── Debugger (LLM patches script, ≤5 retries)
        │                                  │
        ▼                                  └─── back to Inspector
    Advisor              ── Result + Health 100-pt rubric
        │
        ├── verdict revise ── Reviser ── back to Synthesiser (≤2 cycles)
        │
        ▼
     Q&A loop            ── follow-up questions about the run
```

Every step records its decisions and assumptions into `state.json`. You can
inspect the trail in the browser UI or by opening the JSON directly.

---

## Repository layout

```
agentic_fracture/
├── README.md              ← this file
├── LICENSE                ← MIT
├── pyproject.toml         ← package metadata, console script
├── requirements.txt       ← Python dependencies for the agent
├── .env.example           ← copy to .env, fill in your Gemini key
├── .gitignore
│
├── fracture_agent/        ← the agent (this is the package you import)
│   ├── config.py          ← env-var-driven config, model routing, paths
│   ├── llm.py             ← Gemini REST client with key cycling
│   ├── schema.py          ← Pydantic CanonicalSpec, Action, ResultSummary
│   ├── state.py           ← resumable session state
│   ├── orchestrator.py    ← state-machine glue
│   ├── main.py            ← argparse CLI + --ui launcher
│   ├── templates.py       ← single parametric driver-script renderer
│   ├── mesh.py            ← mesh sizing + custom-gmsh LLM helper
│   ├── executor.py        ← wsl.exe streaming runner + divergence watchdog
│   ├── results.py         ← log → ResultSummary parser
│   ├── health.py          ← 100-point physics-aware rubric
│   ├── ablation.py        ← B1 / B2 / B3 / B4 ablation harness
│   ├── telemetry.py       ← per-session token / cost / latency aggregator
│   ├── agents/            ← receptionist, architect, material_helper,
│   │                        strategist, synthesizer, inspector, debugger,
│   │                        advisor, reviser
│   ├── rag/               ← three-index hierarchical retrieval
│   ├── benchmark/         ← per-level / per-tier benchmark harness
│   ├── ui/                ← Flask + SSE browser UI
│   └── _tests/            ← unit tests
│
├── modular/               ← DOLFINx phase-field building blocks
│   ├── materials/         ← JSON catalog (8 calibrated materials) + loader
│   ├── meshes/            ← 5 gmsh builders (plate, slant, dogbone, 2D/3D)
│   ├── constitutive/      ← linear, J2 plasticity, Lopez–Pamies, Drucker–Prager
│   ├── problems/          ← 9 problem builders (the RAG skeletons)
│   ├── solvers/           ← quasistatic, dynamic (HHT-α), ductile, finite-elastic
│   ├── common/            ← SNES wrappers, AMR, BCs, IO utilities
│   ├── post/              ← XDMF writer + reaction-force form
│   └── examples/          ← 9 runnable reference drivers
│
├── agentic_simulations/   ← runs land here (slug-named, gitignored)
│
└── papers/                ← research plan, reference papers (citations)
```

---

## Configuration

| variable | default | what it controls |
|----------|---------|------------------|
| `GEMINI_API_KEY` | — (required) | Your Gemini key. |
| `GEMINI_API_KEY_2` / `GEMINI_API_KEY_3` | — | Optional rotation pool. |
| `FRACTURE_AGENT_PRIMARY_MODEL` | `gemini-2.5-pro` | Reasoning-heavy calls (Strategist rationale, Advisor Q&A). |
| `FRACTURE_AGENT_FAST_MODEL` | `gemini-2.5-flash` | Structured extraction (Receptionist, Architect, Debugger, Material lookup). |
| `FRACTURE_AGENT_VISION_MODEL` | `gemini-2.5-pro` | Image-aware calls. |
| `FRACTURE_AGENT_EMBED_MODEL` | `gemini-embedding-001` | RAG index embeddings. |
| `FRACTURE_AGENT_CONDA_ENV` | `fenicsx` | WSL conda env name. |
| `FRACTURE_AGENT_MPI_DEFAULT` | `1` | Default MPI ranks for a run. |

---

## Extending

| if you want to... | edit |
|---|---|
| Add a material | `modular/materials/materials.json` |
| Add a built-in geometry | a `make_*` in `modular/meshes/`, then wire into `fracture_agent/knowledge.py:CATALOG` and `fracture_agent/templates.py:_mesh_call` |
| Add a constitutive model | a module in `modular/constitutive/` and `modular/problems/`, then register in `CATALOG` and `templates.py:VARIANT_META` |
| Add a Reviser rule | append a branch in `fracture_agent/agents/reviser.py:_propose_revision` keyed on a Health flag |
| Add a Health metric | extend `fracture_agent/health.py` (keep total points = 100) |
| Switch LLM models | set `FRACTURE_AGENT_PRIMARY_MODEL` / `FRACTURE_AGENT_FAST_MODEL` in your `.env` |

---

## Running the tests

A small unit-test suite covers the deterministic pieces (region naming,
prompt parsing, telemetry, RAG indices, health scoring, ablation routing):

```bash
cd agentic_fracture
python -m pytest fracture_agent/_tests/ -q
```

The full integration tests need WSL + a working FEniCSx env and a live
Gemini key, so they're not part of the default suite.

---

## Known limitations

1. **Windows-only execution path.** The executor shells out to `wsl.exe`;
   Linux/macOS users would need to swap that for a direct `conda` invocation.
2. **Physics Advisor is log-only** — no XDMF readback, no energy balance, no
   pointwise irreversibility check, no h/2 mesh-independence rerun. Catches
   obvious failures (NaN, divergence, no damage evolution); silent physics
   errors can slip through.
3. **Vague prompts with no dimensions** can lead the mesh-LLM to invent
   geometry. The architect tries to refuse, but a "compress an alumina
   disc" with no size will get a 1 mm radius guess.
4. **σ_ts backfill** assumes 0.5%·E, which is too high for high-modulus
   ceramics (alumina ends up around 1.9 GPa vs the real ~300 MPa).
5. **Velocity boundary conditions** for dynamic problems are mis-coerced
   to a `traction` magnitude — the schema lacks a velocity BC type.
6. **Free-tier Gemini RPM** caps end-to-end at roughly 1–2 minutes per
   prompt. Add a second or third key to lift the throttle.
7. **3D visualisation in the UI** renders 2D slices; true 3D is on the to-do
   list.

---

## Status

This started as the engineering substrate for an academic paper on
physics-aware agentic code synthesis for regularised-fracture continua —
see `papers/agent_plan.md` for the full research plan. The code is
research-grade: it works on the problems it was tested on, but expect rough
edges on novel geometries.

If you find a case where it goes off the rails, please open an issue with
the prompt and (if possible) the `state.json` from the failing run.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

The physics modular layer is informed by the Bourdin–Francfort–Marigo
phase-field variational fracture line, Pham–Marigo–Maurini gradient damage,
and the Drucker–Prager driving force formulation from Kumar–Francfort–
Lopez-Pamies (2018) and Kamarei–Lopez-Pamies (2025). The agent design
borrows from ATHENA's policy-implementation-execution separation, Foam-Agent
2.0's hierarchical RAG, and ALL-FEM's coder-executor-corrector inner loop.
A more complete citation list lives in `papers/agent_plan.md`.
