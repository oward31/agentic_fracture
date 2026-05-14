# Modular phase-field fracture (DOLFINx)

Plug-and-play refactor of [`../fea_codes/`](../fea_codes/). A simulation is
composed from four layers: **material → mesh → problem builder → solver**,
plus shared infrastructure (SNES wrapper, AMR, XDMF writer).

## Run

```bash
cd /mnt/c/Users/adahal8/Downloads/agent_full_2
wsl
conda activate fenicsx
python modular/examples/linear_elastic_2d_pe.py
# or in parallel:
mpirun -n 4 python modular/examples/linear_elastic_2d_pe.py
```

Each example writes a per-step `paraview_*/step_<NNNNNN>.xdmf` (u, z, σ_vm,
optionally p) and a line-per-step `output_*.txt` log.

## Available cases (`examples/`)

| File                         | Kinematics                          | Loading                   |
| ---------------------------- | ----------------------------------- | ------------------------- |
| `linear_elastic_2d_pe.py`    | 2D plane strain                     | quasistatic displacement  |
| `linear_elastic_2d_ps.py`    | 2D plane stress                     | quasistatic displacement  |
| `linear_elastic_3d.py`       | 3D                                  | quasistatic displacement  |
| `dynamic_2d_ps.py`           | 2D plane stress                     | dynamic (HHT-α), traction |
| `ductile_2d_pe.py`           | 2D plane strain, J2 plasticity      | quasistatic displacement  |
| `ductile_3d.py`              | 3D, J2 plasticity                   | quasistatic displacement  |
| `finite_elastic_2d_ps.py`    | 2D plane stress, Lopez-Pamies Ogden | quasistatic displacement  |
| `finite_elastic_2d_pe.py`    | 2D plane strain, Lopez-Pamies Ogden | quasistatic displacement  |
| `finite_elastic_3d.py`       | 3D, Lopez-Pamies Ogden              | quasistatic displacement  |

## Fixed conventions

| quantity                    | value                              |
| --------------------------- | ---------------------------------- |
| Irwin length `lch`          | `3·Gc / (16·Wts)`                  |
| regularisation length `eps` | `lch`                              |
| coarse element `h0`         | `2·eps = 2·lch`                    |
| minimum element `h_min`     | `h0 / 8`                           |
| AMR refine radius           | `3·eps`                            |
| AMR flag threshold          | `-0.1` (phase-field)               |
| stagger tolerance           | `1e-7`                             |
| max stagger iterations      | `20`                               |
| quasistatic/ductile steps   | `200`                              |
| adaptive-dt rule            | halve & revert if `z_res > 10·tol` |
| adaptive-dt floor           | `dt_first / 10`                    |
| dynamic Δt                  | `0.1 · L_char / c_R` (Rayleigh)    |
| `bcs_z`                     | empty (linear / ductile / dynamic); z=1 top/bottom (finite elasticity, matches slant_amr) |

## Layout

```
modular/
├── materials/      JSON database + loader (derives mu, lmbda, kappa, Wts, Whs,
│                   lch, eps, h0, h_min).
├── meshes/         Geometry generators. Each exports `make_*(…)` returning
│                   (msh, markers_spec, geom).
├── constitutive/   UFL helpers: linear elasticity (2D-PS, 2D-PE, 3D),
│                   Drucker-Prager driving force, J2 plasticity (radial return,
│                   Voigt helpers), Lopez-Pamies Ogden.
├── common/         SNES wrapper, AMR machinery, Dirichlet/facet-tag helpers,
│                   non-matching interpolation, I/O banners.
├── problems/       `make_*_builder(mat, markers, geom, …) -> build_problem(msh)`.
│                   Nine variants — these are the RAG "skeletons".
├── solvers/        Time-marching drivers: quasistatic adaptive, dynamic HHT-α,
│                   finite-elasticity hand Newton (adaptive dt + ω regularisation),
│                   ductile (adaptive dt + QP→QP plastic state transfer).
├── post/           Per-step XDMF writer, reaction-force form factories.
└── examples/       Nine complete runnable drivers.
```

## Composition pattern

```python
from modular.materials import load_material
from modular.meshes    import make_notched_plate_2d
from modular.problems  import make_linear_elastic_2d_pe_builder
from modular.solvers   import run_quasistatic
from modular.post      import XDMFWriter, reaction_form_from_sigma_2d

mat = load_material("Steel_bench_2D_PE")
msh, markers, geom = make_notched_plate_2d(
    W=1.0, L=1.0, ac=0.5, cw=0.001, h0=mat["h0"])

build = make_linear_elastic_2d_pe_builder(
    mat=mat, markers_spec=markers, geom=geom,
    eps=mat["eps"], h0=mat["h0"], h_min=mat["h_min"])
P = build(msh)

writer = XDMFWriter("paraview_run")
run_quasistatic(
    P, build, msh,
    T_total=1.0, steps=200, max_stag=20, tol_stag=1e-7, max_disp=0.006,
    on_output=lambda P, t, s, dt, e: writer.write(P, t, s),
    reaction_form=reaction_form_from_sigma_2d("top", component=1),
)
```

Swap plane strain → plane stress by changing one import
(`make_linear_elastic_2d_pe_builder` → `make_linear_elastic_2d_ps_builder`).
Swap 2D → 3D by changing two (the mesh and the builder). Switch to dynamic
by swapping the solver and adding a `pressure_ramp` callable.

## Extending

* **New material**: add an entry to `materials/materials.json`.
* **New geometry**: add a file in `meshes/` returning `(msh, markers_spec, geom)`.
* **New constitutive law**: add a module to `constitutive/`, then a new
  builder in `problems/`; re-use the Drucker-Prager helpers.
* **New solver**: add to `solvers/`. `common/amr.try_amr` supports arbitrary
  extra field transfers via `extra_V_fields`, `extra_Y_fields`,
  `after_rebuild` hooks.

## Relationship to the originals

Physics is line-for-line identical to [`../fea_codes/`](../fea_codes/) —
verified block-by-block. The only deliberate change is the length-scale
convention (`eps`, `h0`, `h_min` derived from the material via `lch` rather
than hard-coded per script).
