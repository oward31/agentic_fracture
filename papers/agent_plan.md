# An agentic FEniCSx system for phase-field fracture

A multi-agent LLM system for phase-field fracture mechanics in FEniCSx is feasible today and publishable at a top venue, provided it is built on **three unexploited leverage points**: (1) a *hierarchical multi-index RAG over the user's 6–7 modular FEniCSx codes* (Foam-Agent pattern, proven to deliver +12.7 pp retrieval quality over flat RAG); (2) a *physics-aware Inspector + Advisor pair* that enforces the Markov property on generated code and diagnoses phase-field–specific silent failures (ATHENA pattern, absent from every existing FEA agent); and (3) a *multimodal canonical problem spec* that fuses text, hand-drawn sketches, reference figures, and speech into a Pydantic-validated JSON schema (extending FeaGPT's single-modality approach). No prior system addresses the unique automation hazards of phase-field fracture — length-scale/mesh coupling, staggered-loop non-convergence, split choice, irreversibility violations, snap-back — and no prior system targets DOLFINx (all existing FEniCS agents emit legacy `dolfin`). This is the publishable gap. The plan below sequences an MVP in ~10 weeks, a benchmark-validated full system in 5–7 months, and a journal-grade deliverable for CMAME or JMPS in 9–12 months.

## 1. What the literature actually shows

Six systems dominate the current (April 2026) landscape of LLM agents for numerical PDE solvers: **FeaGPT** (FreeCAD+Gmsh+CalculiX, 432 NACA cases, σ=0.85 similarity-gated knowledge-augmentation), **ALL-FEM** (7 agents in an AutoGen group chat with a nested Coder↔Executor↔Corrector inner loop, 71.79% code-success on 39 legacy-FEniCS benchmarks using a fine-tuned GPT-OSS-120B), **MCP-SIM** (6 agents plus a memory-centric orchestrator implementing Plan-Act-Reflect-Revise, 100% success on 12 curriculum levels including a phase-field crack at Level 12 reproducing Storvik et al. 2021), **MechAgents** (two-agent vs multi-agent AutoGen on FEniCS linear/hyperelasticity with documented failure modes — wrong weak forms, missing traction terms), **AutoFEA** (GCN-Transformer link-prediction retrieval + GPT-4o achieving 90.2% success on 512 CalculiX projects), and in CFD **Foam-Agent 2.0** (6 agents, LangGraph + MCP, hierarchical 4-index FAISS RAG, 88.2% on 110 cases with Claude 3.5 Sonnet). Outside the FEM world, **ATHENA** (arXiv 2512.03476) frames scientific computing as a contextual bandit with a Policy-Implementation-Execution loop, adds a cell-by-cell refactoring Planner and an **Inspector** that verifies generated code faithfully implements the proposed action — a role that prevents the autoregressive correctness decay P(correct) ≈ p^N on long generations and is absent from every FEM agent surveyed.

**Five design patterns stand out as essential** (appearing in the highest-performing systems): hierarchical multi-index RAG (+12.7 pp in Foam-Agent ablation), dependency-aware generation with Pydantic schemas on every agent boundary (stops ~5–8 pp regressions and cuts reviewer loops ~50%), a dedicated Reviewer/Corrector loop with history memory (the single largest ablation effect across all systems, +30–50 pp), proposer-critic debate before execution (cheapest validator — no simulation needed), and visual inspection of simulation outputs by a VLM Advisor (catches silent physics failures invisible to exit-code checks). **Three patterns are bloat to avoid**: redundant \"Scientist-1 / Scientist-2\" elaboration chains (SciAgents), dataset-augmentation RAG done at fine-tuning time rather than inference time (ALL-FEM — user has excluded fine-tuning), and novelty/patent checks that add tokens without value for a simulation-focused product.

## 2. Why phase-field fracture breaks every existing agent

Phase-field fracture is not merely harder than linear elasticity — it exhibits failure modes that would be invisible to the error-correction loops in ALL-FEM, MCP-SIM, and Foam-Agent because **it fails silently, in physics, not in Python**. Six hazards dominate:

The **length-scale/mesh coupling** is physical, not numerical: for AT1 the peak stress σ_c = √(3 G_c E / (8ℓ)) fixes the effective material strength as a function of the regularization length ℓ, and the rule h ≤ ℓ/4 near the crack forces a mesh resolution an agent cannot choose independently of material parameters. A naive uniform mesh either produces physically wrong strength or blows up the O(10^6)-element budget. **Staggered convergence** (the default in phasefieldx and comet-fenicsx) can require 10^3–10^4 iterations at initiation steps; without Ambati's energy-based stopping criterion the loop silently terminates at the wrong answer. **Irreversibility** fails in three common LLM-induced ways: SNES without bounds, history-variable reset between steps, or forgetting the `alpha_lb.x.array[:] = alpha.x.array[:]` update at end-of-step — all produce plausible but thermodynamically impossible healing cracks. **Energy-split choice is mode-dependent** (Bilgen–Homberger–Weinberg 2019 demonstrated volumetric-deviatoric gives the *wrong* crack path on the Brazilian disc); no LLM today knows to switch to a stress-based or star-convex split. **Monolithic Newton is non-convex**; vanilla `dolfinx.nls.petsc.NewtonSolver` without line search and active-set hits max iterations and returns garbage. **Snap-back** needs arc-length continuation that DOLFINx does not ship.

Layered on top is the **DOLFINx API shift** that most web-scraped FEniCS code violates: `dolfin.FunctionSpace` → `dolfinx.fem.functionspace`; `ufl.MixedElement` → `basix.ufl.mixed_element`; `NonlinearProblem` is now a high-level SNES interface in v0.10; bounded variational inequalities require a custom `SNESProblem` wrapper with `vinewtonrsls`; quadrature storage of the history field needs `basix.ufl.element(\"Quadrature\", ...)` and `dolfinx.fem.Expression` rather than direct `interpolate`. Every existing FEniCS agent emits legacy `from dolfin import *` code. **This alone is a publishable gap.**

## 3. The novelty story and why a top venue will take this

No paper in the 2024–2026 corpus combines (a) multimodal input (sketch/photo/voice/text) for a specialist physics domain, (b) user-owned modular code blocks as first-class retrieval primitives (distinct from generic tutorial RAG), (c) DOLFINx-specific code generation with API-version filtering, and (d) phase-field-aware validation that checks energy balance, damage bounds, irreversibility, and mesh/ℓ ratios as first-class reward components. The closest analogs each address one dimension: FeaGPT handles parametric geometry but only CalculiX and single-modality; ALL-FEM handles FEniCS breadth but requires fine-tuning, targets legacy FEniCS, and treats all problems identically; MCP-SIM includes one phase-field task but has no RAG, no multimodality, and no physics-specific checks; Foam-Agent perfects hierarchical RAG but for OpenFOAM. The proposed system is the first to integrate all four dimensions in one specialist domain where **silent physics failures** dominate over syntactic errors. A CMAME or JMPS paper can frame this as a methodological contribution ("physics-aware agentic code synthesis for regularized-fracture continua") with the modular-blocks RAG and the Inspector-as-Markov-enforcer as the core technical novelties; npj Computational Materials and EML are viable alternative venues with a slight shift of emphasis toward application breadth and materials-aware validation.

## 4. System architecture

The proposed architecture, named here **PF-Agent**, adopts ATHENA's three-layer Policy–Implementation–Execution separation, inherits Foam-Agent 2.0's LangGraph + MCP orchestration, borrows ALL-FEM's nested Coder↔Executor↔Corrector inner loop, and adds two domain-specific agents (a Physics Advisor and a PF-Critic) that exist in no prior FEM system.

The **Conceptualization layer** has three agents. A *Receptionist* consumes the raw user query (text / speech / image / hybrid), routes it through modality-specific encoders, and produces a raw JSON fragment list. An *Architect* merges fragments into a canonical problem spec (schema in §6), asking targeted clarification questions when VLM and text disagree — this is where FeaGPT's semantic-location pattern (e.g., "left edge", "notch tip") replaces unreliable coordinate read-off from sketches. A *Gatekeeper* enforces Pydantic validation, well-posedness (dimensional consistency, non-empty BC set, load-history presence) and rejects the request to the user if fields are unsupplied, preventing silent downstream failures.

The **Policy layer** contains the phase-field–specific intelligence and is the principal contribution. A *Strategist* selects the formulation (AT1 vs AT2 vs PF-CZM vs hyperelastic vs ductile), the split (none / Amor vol–dev / Miehe spectral / stress-based / star-convex), the irreversibility mechanism (history variable vs SNES-VI vs penalty vs augmented Lagrangian), the solver scheme (staggered-alternate-minimization vs monolithic BFGS vs line-search Newton vs arc-length), and the mesh/ℓ strategy. Its output is a structured Action A_n in the ATHENA sense — a discrete tuple over the finite combinatorial space of formulation choices. A *PF-Critic* (an LLM-judge with a phase-field-aware system prompt) reviews A_n against five blueprints encoded as "Conceptual Scaffolding" (variational fracture, regularization and length-scale selection rules, degradation-and-split taxonomy, irreversibility enforcement, solver continuation strategies) and either approves or returns A_n for revision. The Critic rejects dangerous plans before any simulation runs — e.g., volumetric-deviatoric split on a Brazilian disc, ℓ=2h uniform mesh on a 3D case, monolithic Newton without line search. A *Physics Advisor* — the domain-specific analog of ATHENA's Advisor — runs *after* each simulation attempt, consuming multimodal observations (load–displacement curves, damage maps, energy histories, ParaView screenshots via a VLM) and assigns a composite reward that decomposes into integrity (code ran, exit 0), accuracy (residuals, energy balance ∫G_c dΓ vs Π_ext − Ψ), physical admissibility (damage bounds d ∈ [0,1], irreversibility max_t d(·,t) non-decreasing, no spurious far-field damage), and mesh-independence (agreement between h and h/2 when automatic).

The **Implementation layer** translates a Strategist-approved action into DOLFINx Python. A *Template Retriever* executes the hierarchical multi-index RAG described in §5 and returns the best-matching user modular code block plus supporting documentation and API signatures. A *Cell-by-Cell Planner* (ATHENA's invention, adapted) performs surgical edits on the retrieved template rather than free generation — swapping a weak form, changing a split function, editing BC locations, updating solver parameters — which counters the autoregressive correctness decay that gives MechAgents its wrong-weak-form failure mode. An *Inspector* (unique to ATHENA among prior systems) verifies the emitted code faithfully implements A_n by cross-checking imports against expected DOLFINx modules, AST-parsing for forbidden legacy-FEniCS symbols (`from dolfin import *`, `FunctionSpace(...)` capitalized), checking that the Strategist's declared irreversibility mechanism is actually present in the source, and confirming the declared split function appears literally in the UFL form. A *Debugger* runs the nested Coder↔Executor↔Corrector loop from ALL-FEM, capped at 5 iterations (Foam-Agent found gains beyond ~5 loops marginal), with ChatCFD's dual-model split: a reasoning LLM (Claude Opus 4 or GPT-5 Thinking) localizes errors, a fast LLM (Claude 3.5 Sonnet, GPT-4.1) applies the edit, cutting token cost ~3× at equivalent success.

The **Execution layer** is a sandboxed Python process with preinstalled DOLFINx v0.10, petsc4py, gmsh, pyvista, MPI, and the user's modular codes. Outputs are structured: XDMF fields, CSV load–displacement, PNG screenshots at user-configured damage thresholds, and a JSON run summary. A *Visualization Agent* (Foam-Agent 2.0's addition) produces the plots the Physics Advisor consumes.

Crosscutting is a *Memory Orchestrator* (MCP-SIM pattern) managing a shared JSON state machine: raw input artifacts, canonical spec, A_n history, code versions, execution logs, error→fix mappings, reward R_n, and long-term `<reflexion>` blocks from ChatCFD's pattern ("For the L-shaped panel I chose vol–dev split but the crack went wrong; next, for corner-singularity problems under mode-I-dominant loading with high ν, I will default to spectral split"). Successful runs are archived back into the KB (FeaGPT's self-growing principle; ATHENA's Storage Group).

**Agent-count discipline**: 11 agents is the upper limit that yields diminishing returns in the ablations surveyed. The Novelty/Safety checks from SciAgents and ChemCrow are deliberately omitted as bloat. The Planner/Critic/Advisor trio and the Inspector are the four non-negotiable roles; everything else is a specialist on top.

## 5. RAG and knowledge base design

The user's 6–7 modular phase-field FEniCSx codes are the crown jewel and must be treated as **first-class retrieval primitives**, not as one bucket among tutorials. The Foam-Agent 2.0 hierarchical multi-index FAISS architecture is adopted with a domain-specific reshaping:

**Seven parallel indices**, each with its own embedding, chunking, and retrieval strategy:

1. *User-Code-Skeleton index* — each of the 6–7 user codes indexed at **whole-file** level with an LLM-generated summary and a structured metadata record: `{dimensionality: 2D|3D, kinematics: small-strain|finite-strain, constitutive: linear-elastic|NeoHookean|Ogden|..., formulation: AT1|AT2|PF-CZM, split: none|volDev|spectral|star-convex, irreversibility: history|SNES-VI|penalty, solver: staggered|monolithic-BFGS|line-search, AMR: yes|no, loading: quasistatic|dynamic, dolfinx_version: 0.10}`. Retrieval key is the Strategist's declared A_n.
2. *User-Code-Snippet index* — the same codes chunked via **cAST** (AST-recursive tree-sitter chunking, max 1024 tokens, merge threshold 768), function-level, with parent-function metadata and canonical signatures. This lets the Cell-by-Cell Planner retrieve exactly the function needed (e.g., just the history-field update, just the spectral-split UFL expression).
3. *DOLFINx-Official-Demos index* — official demos from `FEniCS/dolfinx/python/demo/` and Dokken's `dolfinx-tutorial`, cAST-chunked, with a **hard metadata filter `fenics_version ≥ 0.10` enforced at retrieval time**, rejecting any legacy `from dolfin import` content. Crawled targets: `jsdokken.com/dolfinx-tutorial`, `bleyerj.github.io/comet-fenicsx`, `docs.fenicsproject.org/dolfinx/main/python/demos.html`, plus the ALL-FEM curated corpus filtered for DOLFINx-only.
4. *PF-Benchmark-Reference index* — curated implementations of canonical benchmarks from `phasefieldx` (Castillón, JOSS 2025), `farhadkama/FEniCSx_Kamarei_*` (nine-circles, Kamarei–Lopez-Pamies 2025), `jhale/cism-2024-varfrac-code` (CISM 2024), `newfrac/fenicsx-fracture`. Used by Template Retriever when a benchmark is the target.
5. *API-and-Signature index* — every public DOLFINx, UFL, Basix docstring with canonical usage, auto-extracted from source. Used by Debugger for API recall.
6. *Prose-Pedagogy index* — paragraph-level chunks (800 tokens, overlap 120) from textbook material, Bleyer tours narrative, course notes, and the Kamarei–Lopez-Pamies arXiv papers. Used by Strategist for formulation rationales and by the multilingual Insight Agent at the end.
7. *Error-Fix Memory* — self-populating from Debugger experiences, chunked as `{error_message, stacktrace, offending_snippet, applied_fix, outcome}`. Bootstrap with the ~30 common DOLFINx phase-field errors catalogued from the FEniCS Discourse.

**Embedding stack**: code chunks use **voyage-code-3** at 1024-dim int8 (13.8% better than text-embedding-3-large on 238-benchmark code suite; 32K context handles whole demos); prose chunks use **voyage-3-large** or fallback `bge-m3` for on-prem. Cosine threshold 0.35 floor; top-k=20 per index; reciprocal rank fusion across indices with k=60; **Cohere rerank-3.5** (or bge-reranker-v2-m3 on-prem) compresses to top-8 passed to the generator. Total retrieved context is budgeted at ~30K tokens of a ~200K-token model window, leaving room for the Strategist's action JSON, prior trial history, and the generation target.

**Retrieval routing is stage-aware** (the Foam-Agent Algorithm-2 pattern): the Architect queries only index 1 to pick a skeleton; the Cell-by-Cell Planner queries indices 2+3+4+5 with hybrid BM25+dense fusion; the Debugger queries indices 5+7 with a stacktrace; the Physics Advisor queries index 6 when a physical anomaly needs literature anchoring. The **composition-vs-generation ratio is controlled by a FeaGPT-style similarity gate at σ_threshold = 0.85**: if the best User-Code-Skeleton match exceeds 0.85 in the metadata-constrained cosine space, the pipeline enters "skeleton-mutation mode" (Cell-by-Cell edits only); below 0.85 it enters "novel synthesis mode" (free generation of a new block, but still with retrieved API context). This is the user's "modular codes as building primitives" idea formalized.

## 6. Multimodal pipeline

The canonical problem spec is a strict Pydantic schema with namespaced fields — the single biggest reliability lever found across every surveyed system. Separate encoders handle each modality then write into fragments:

```
CanonicalSpec {
  source_modalities: List[text|speech|image|figure]
  domain: "phase_field_fracture"
  geometry: GeometrySpec { kind: SENT|SENS|3PB|L_panel|Brazilian|notched_rubber|custom
                           dimensions_mm: {...}; notch_location: SemanticRegion; mesh_hint }
  material: MaterialSpec { E, nu, Gc, ell_0, eta?, constitutive, regime: brittle|quasi-brittle|rubbery|ductile }
  loading: LoadingSpec { mode: quasistatic|dynamic, control: displacement|force|arc-length,
                         history: [{t, dof_set: SemanticRegion, value}] }
  formulation_pref?: AT1|AT2|PF-CZM  # optional; Strategist picks if absent
  split_pref?: none|volDev|spectral|stress|star-convex
  irreversibility_pref?: history|SNES-VI|penalty|aug-lagrangian
  amr: {enabled: bool, strategy: damage-threshold|hwh-predictor-corrector}
  QoI: [load-disp | crack-path | energy-dissipation | fracture-toughness]
  reference_figure?: {source_doi, fig_id, bbox, target: shape|crack-topology}
  output_language: "en" | "de" | "ko" | "ja" | ...   # MCP-SIM pattern
}
```

**Text** flows directly into the Receptionist LLM. **Speech** is transcribed by **gpt-4o-transcribe** (2.46% WER, 99.2% on technical jargon per the April 2026 benchmarks) with a prepended domain glossary ("phase-field", "quasi-brittle", "Gc", "AT1", "AT2", "spectral split", "notch-tip", "three-point bend", "Brazilian disc"); offline fallback is Whisper-large-v3 + an LLM glossary-check pass. **Images** branch by sub-type: hand-drawn sketches go through Claude 3.5 Sonnet v2 (primary, best structured-JSON extraction) and GPT-4o (cross-check, better small-numeral OCR); disagreement on any numeric dimension triggers Architect clarification. Scanned technical drawings with dimension call-outs are handled the same way; given that DesignQA and Businessware benchmarks show 25–40% dimension-reading errors even for frontier VLMs, **a confirmation echo to the user** is built into the clarification loop. Photographs of cracked specimens feed a specialized pipeline: VLM description → Gmsh geometry JSON → DOLFINx mesh (following CAD-MLLM's image→JSON→code template). Reference figures from papers — e.g., "match Miehe 2010 Fig 6 crack path" — are extracted with PyMuPDF + layout parsing (Nougat or PaperMage), bounding-boxed by VLM-Grounder pattern, stored as a *target observation* that the Physics Advisor later compares against the simulation output using an LLM-judge or cosine-on-VLM-embeddings approach.

Fragments from all modalities are assembled by the Architect into a single canonical spec, Pydantic-validated, with any conflict (VLM says 3 mm notch, user typed 2 mm) flagged for user confirmation before dispatch.

## 7. Memory, orchestration, and tooling

LangGraph hosts the stateful graph with Pydantic I/O on every edge (the Foam-Agent 2.0 pattern); LangSmith provides traceability. Eleven agent nodes plus conditional edges route on A_n approval, execution outcome, and reward value. The Memory Orchestrator is a separate service owning the shared JSON state and exposing reads/writes to all agents — this is MCP-SIM's memory-centric pattern chosen over AutoGen's conversation-transcript approach because JSON structure enables targeted queries (e.g., "have I already tried spectral split + SNES-VI on this case?") without parsing prose. All external capabilities are wrapped as **MCP tools** (`classify_problem`, `retrieve_skeleton`, `propose_action`, `critique_action`, `edit_cell`, `inspect_code`, `run_simulation`, `review_errors`, `analyze_physics`, `render_plot`, `generate_report`) so a developer can drive the system from Claude Code or Cursor and an outer AI-scientist loop (turbulence.ai pattern) can be added later without refactoring.

LLM selection follows ChatCFD's dual-model philosophy: **Claude Opus 4** or **GPT-5 Thinking** as the reasoning LLM for Strategist, PF-Critic, Physics Advisor, and error localization; **Claude 3.5 Sonnet v2** or **GPT-4.1** as the fast LLM for Cell-by-Cell Planner, Inspector, Debugger, Visualization Agent. Temperature 0.01 for code generators (MetaOpenFOAM found 85%→48% degradation at T=0.99), 0.3 for Strategist (needs exploration), 0.0 for Inspector (strict determinism).

## 8. Error correction and physics validation

The correction loop is layered. The **Corrector inner loop** (ALL-FEM) handles syntactic errors, runtime exceptions, and solver non-convergence at max 5 iterations with dual-model cost economy. The **Reflect-Revise outer loop** (MCP-SIM) handles cases where the error persists after 5 inner iterations — here the Input Rewriter returns to Architect to revise the canonical spec (e.g., "your mesh was too coarse; increasing element count") or the Strategist revises A_n (e.g., switch staggered to monolithic-BFGS with line search). The **Physics Advisor checks** run after every successful execution and enforce:

- **Energy balance**: the externally supplied work Π_ext must equal the elastic stored energy Ψ plus the crack-surface dissipation ∫G_c dΓ within 5% tolerance; larger imbalance flags a split-choice or history-field bug.
- **Damage admissibility**: pointwise d ∈ [0,1], violation indicates missing penalty/VI or numerical overshoot.
- **Irreversibility**: d(x, t_2) ≥ d(x, t_1) for t_2 > t_1 at all nodes; violation indicates missing `alpha_lb` update or penalty weight too low.
- **Mesh-independence spot check**: when wall-time budget permits, rerun at h/2 on the last load step; load–displacement curves must agree within 2% — failure flags ℓ/h under-resolution.
- **Mesh/ℓ ratio audit**: static check that `max_cell_in(crack_neighborhood).h ≤ ℓ/4`; auto-refines if violated.
- **Crack-path plausibility**: for benchmarks with a known reference (Miehe SENT, SENS), VLM-judge compares the final damage contour to the reference figure with a structured similarity prompt.
- **Load–displacement sanity**: no negative stiffness before peak (except in snap-back cases flagged by Strategist); no healing dips.

These checks are encoded as reward components summing to 100 (following ATHENA): integrity 25, physical admissibility 30 (irreversibility 12, damage bounds 8, energy balance 10), accuracy 25 (residual, reference match), mesh-independence 10, efficiency 10. Rewards below 60 trigger a Reflect-Revise cycle; reward 85+ accepts the solution and archives the run in Storage.

## 9. Benchmark design — the 22 problems

The benchmark suite is organized in five difficulty tiers spanning 22 problems, designed as a union of MCP-SIM's curriculum (Levels 1–12), Kamarei–Lopez-Pamies 2025's Nine Circles (CMAME arXiv 2507.00266), and the Miehe–Ambati canon. Expert-authored, disjoint from training corpora, scored per §8 metrics.

**Tier 1 — Foundation (fully specified, single-physics, 2D small-strain AT2, vol-dev or spectral split, history-variable irreversibility):**
1. 1D bar in tension with analytical AT1/AT2 benchmark (Pham 2011) — parameter calibration.
2. Single-Edge-Notched Tension, mode-I, Miehe 2010 geometry — canonical.
3. Double-Edge-Notched Tension — crack coalescence.
4. Compact Tension — Griffith calibration.

**Tier 2 — Path selection (mode-mix and split sensitivity):**
5. Single-Edge-Notched Shear, Miehe 2010 — vol-dev gives wrong path; tests Strategist's split decision.
6. Symmetric three-point bending, Bittencourt 1996 — symmetry.
7. Asymmetric three-point bending with off-center hole — curved crack path.
8. L-shaped panel, Winkler 2001 — corner singularity, high-ν handling.
9. Brazilian disc, Bilgen–Homberger–Weinberg 2019 — demands stress-based or star-convex split; vol-dev fails.

**Tier 3 — Model and formulation selection:**
10. Sneddon 2D pressurized line crack — analytical verification; pressure BC.
11. Notched quasi-brittle concrete beam — demands PF-CZM (Wu 2018) for length-scale insensitivity.
12. Surfing test, Hossain 2014 — effective G_c extraction.
13. Kamarei Nine-Circles *poker-chip* — nucleation under triaxial hydrostatic stress.
14. Kamarei Nine-Circles *indentation* — strength-surface nucleation.

**Tier 4 — 3D and finite strain:**
15. Sneddon 3D penny-shaped pressurized crack — analytical TCV verification.
16. 3D torsion-tension bar, Nine-Circles — helical crack.
17. 3D SENT plate with bending — dimensional extension.
18. Notched rubber specimen, Miehe–Schänzel 2014 — finite-strain Neo-Hookean.
19. DCB on rubber — mode-I hyperelastic.
20. Trousers test — mode-III tearing in hyperelastic regime.

**Tier 5 — Dynamic and ductile (research-grade):**
21. Kalthoff–Winkler impact, Borden 2012 — dynamic, 70° branching, Newmark-β.
22. Miehe–Aldakheel ductile shear-compression specimen — gradient-extended plasticity-damage, two length scales.

Three ambiguity variants (per problem): fully specified (Level 1), partially specified requiring Strategist inference (Level 2), problem-described-from-a-paper-figure only (Level 3 — tests the multimodal path). This yields 66 effective test cases. Against the literature, 22 problems split 9/5/5/3 brittle-quasibrittle/benchmarks/3D-finite/dynamic-ductile beats ALL-FEM's 16 solid benchmarks and MCP-SIM's single PF-level on rigor and uniquely spans regimes.

## 10. Evaluation metrics

Five orthogonal axes are reported per benchmark run:

**Code-execution success** (ALL-FEM metric): binary — did the generated Python run to completion without exception on the sandbox? Target ≥ 85% on Tier 1–3, ≥ 60% on Tier 4–5, matching Foam-Agent's 88.2% on CFD and beating ALL-FEM's 71.79%.

**Physical admissibility** (novel to this system): composite 0–1 score from energy balance, damage bounds, irreversibility, and mesh/ℓ audit. Target ≥ 0.9 on accepted runs.

**Reference accuracy**: for benchmarks with published results, L2 error on load–displacement curve; crack-path Hausdorff distance on damage-contour polyline; peak-load relative error. Target within 10% of reference for Tiers 1–3.

**Iteration efficiency** (MCP-SIM metric): number of Plan-Act-Reflect-Revise cycles; target median ≤ 3 on Tier 1, ≤ 6 on Tier 4.

**Cost and wall-time**: tokens, API cost, wall-time. Target $2–5/case on Tier 1–2 following ChatCFD's $0.21/case CFD cost, accounting for longer PF simulations.

Two baselines are run for publication: B1 = one-shot Claude Opus 4 with the user's codes in context; B2 = full PF-Agent without the Physics Advisor (ablates the physics-specific reward). Ablations include removing: each index from the hierarchical RAG (expected +12.7 pp from Foam-Agent priors); the Inspector (expected +15–25 pp from ATHENA priors on long generations); the Critic pre-execution debate; and the skeleton-vs-novel-synthesis similarity gate.

## 11. Implementation phases

The project is structured in four phases with hard deliverables.

**Phase 0 — Infrastructure (weeks 1–3).** Containerize DOLFINx with PETSc/MPI/Gmsh/PyVista; set up LangGraph + MCP server scaffolding; implement the canonical Pydantic schema; ingest and index the user's 6–7 codes (Indices 1 and 2) with voyage-code-3 and cAST, using astchunk (`pip install astchunk`); crawl and index jsdokken, comet-fenicsx, official demos, phasefieldx, Kamarei repos into Indices 3 and 4 with DOLFINx-version filtering. Deliverable: queryable 7-index KB with retrieval latency < 500 ms top-8 hybrid.

**Phase 1 — MVP single-modality, Tier 1 (weeks 4–10).** Implement Architect, Strategist, PF-Critic, Template Retriever, Cell-by-Cell Planner, Inspector, Debugger, Execution, Physics Advisor with Tier-1-only checks (execution + damage bounds). Target: ≥ 80% success on problems 1–4 from text input only. Deliverable: publishable workshop paper (ML4PS @ NeurIPS, AIPhy @ ICML).

**Phase 2 — Full agent roster and multimodality (months 3–5).** Add Receptionist with VLM + ASR; add reference-figure grounding; add Gatekeeper; implement the full Physics Advisor (energy, irreversibility, mesh-independence, reference comparison); add multilingual Insight Agent; bootstrap Error-Fix Memory (Index 7) from Discourse. Target: ≥ 80% Tier 1–3, ≥ 65% Tier 4 on text-only; ≥ 70% Tier 1–2 on sketch-input. Deliverable: full-system arXiv preprint.

**Phase 3 — Advanced physics and AMR (months 6–9).** Port Heister–Wheeler–Wick predictor-corrector AMR to DOLFINx as a reusable module (this is a publishable contribution in itself given no dolfinx implementation exists) and integrate with the Strategist. Add dynamic fracture and ductile phase-field support. Add arc-length continuation for snap-back problems. Run full 22-problem × 3-ambiguity benchmark with all baselines and ablations. Deliverable: CMAME or JMPS submission.

**Phase 4 — AI-Scientist outer loop (optional, months 10–12).** Wrap PF-Agent in a turbulence.ai-style Idea–Simulate–Write triple, enabling autonomous exploration of the Gc–ℓ parameter space or toughness-anisotropy scans. Deliverable: npj Computational Materials submission.

Dependencies: Phase 1 blocks Phase 2; the AMR port in Phase 3 is the tallest technical risk — a 2–4 week sub-project with fallback to damage-threshold-based refine-reinterpolate if Heister–Wheeler–Wick proves slow to port.

## 12. Publication strategy and risk

The central publishable claim is that *physics-aware agentic code synthesis with user-owned modular codes as retrieval primitives* solves a problem no prior agent addresses: silent physics failures in regularized-fracture continua. The paper should lead with the ablation showing each of (hierarchical RAG, Inspector, Physics Advisor, similarity-gated skeleton/novel mode) is individually load-bearing and together deliver the ≥ 80% Tier 1–3 / ≥ 65% Tier 4 numbers that beat ALL-FEM's fine-tuned 71.79%, MCP-SIM's single-PF-task, and Foam-Agent's CFD-domain 88.2%. Secondary contributions — the DOLFINx API-filtered RAG, the canonical multimodal PF spec, the Heister–Wheeler–Wick AMR port, and the 22-benchmark suite with three ambiguity levels — are each citable in follow-on work. The biggest risk is that frontier LLMs at the time of submission already solve Tier 1 problems zero-shot; this is mitigated by focusing metrics on Tier 3–5 (where all existing systems fail) and by the novelty of the multimodal and Physics-Advisor components, which no foundation-model improvement subsumes.

Target venues in decreasing fit: **CMAME** (methodological weight, FEM audience, ALL-FEM precedent), **JMPS** (if framed around physics-admissibility and fracture-mechanics contributions), **npj Computational Materials** (if multimodal and materials-discovery framing is emphasized), **EML** (MechAgents precedent, faster turnaround).

## Conclusion

The architecture above is not a synthesis of everything in the literature — it is a deliberate selection of the three or four mechanisms each system independently proved load-bearing, organized around two original components (a phase-field-aware Physics Advisor and a similarity-gated modular-code RAG) that answer the unique hazards of regularized-fracture simulation. The critical insight from the survey is that **prior FEM agents have exhausted the easy wins from AutoGen-style group chats and single-index RAG**; the next 15–30 points of benchmark success come from physics-aware validation (ATHENA shows this in SciML, no one has shown it in FEA) and from treating the user's verified codes as a different kind of knowledge than tutorials. The DOLFINx-specific targeting is not a nuisance detail but a moat: every published FEniCS agent to date emits legacy code, and an agent that reliably produces runnable DOLFINx phase-field code is, for the next 12–18 months, a category-defining artifact in computational mechanics.