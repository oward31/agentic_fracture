"""PETSc SNES wrapper with optional variable bounds.

Identical to the SNESSolver used across the user's quasistatic, surfing and
dynamic codes. Kept verbatim so a RAG retrieval of this block returns a
drop-in replacement.

Usage
-----
    problem = SNESSolver(R_u, u, bcs=bcs_u, petsc_options=OPTS_U)
    its, reason = problem.solve()

Bounded (variational inequality, for the phase-field):
    problem = SNESSolver(R_z, z, bcs=bcs_z, petsc_options=OPTS_Z_VI,
                         bounds=(z_lb, z_ub))
"""

from __future__ import annotations

from dolfinx import fem
from dolfinx.cpp.log import LogLevel, log
import dolfinx.fem.petsc
from petsc4py import PETSc
from ufl import TrialFunction, derivative


# Canonical SNES options. Override selectively per caller.
SNES_VI_COMMON = {
    "snes_type":                "vinewtonrsls",
    "snes_linesearch_type":     "basic",
    "ksp_type":                 "preonly",
    "pc_type":                  "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_atol": 1.0e-8, "snes_rtol": 1.0e-9, "snes_stol": 1.0e-9,
    "snes_monitor": "",
}
SNES_NEWTON_COMMON = {**SNES_VI_COMMON, "snes_type": "newtonls"}

OPTS_U       = {**SNES_VI_COMMON,     "snes_max_it": 50}
OPTS_Z_VI    = {**SNES_VI_COMMON,     "snes_max_it": 10}
OPTS_U_DYN   = {**SNES_NEWTON_COMMON, "snes_max_it": 50}
OPTS_Z_VI_DYN = {**SNES_VI_COMMON,    "snes_max_it": 10}


class SNESSolver:
    """PETSc SNES-based nonlinear solver with optional variable bounds (VI)."""

    def __init__(self, F_form, u, bcs=None, J_form=None, bounds=None,
                 petsc_options=None, prefix=None):
        self.u       = u
        self.bcs     = bcs or []
        self.bounds  = bounds
        self.prefix  = prefix or f"snes_{str(id(self))[:4]}"
        if bounds is not None:
            self.lb, self.ub = bounds
        V = u.function_space
        self.comm = V.mesh.comm
        self.F_form = fem.form(F_form)
        self.J_form = fem.form(J_form if J_form is not None
                               else derivative(F_form, u, TrialFunction(V)))
        self.petsc_options = petsc_options or {}
        self.solver = self._setup()

    # ------------------------------------------------------------------ #
    def _set_petsc_options(self):
        opts = PETSc.Options()
        opts.prefixPush(self.prefix)
        for k, v in self.petsc_options.items():
            opts[k] = v
        opts.prefixPop()

    def _setup(self):
        snes = PETSc.SNES().create(self.comm)
        snes.setOptionsPrefix(self.prefix)
        self._set_petsc_options()
        snes.setFromOptions()
        self.b = fem.petsc.create_vector(self.F_form)
        self.a = fem.petsc.create_matrix(self.J_form)
        snes.setFunction(self._F, self.b)
        snes.setJacobian(self._J, self.a)
        if self.bounds is not None:
            snes.setVariableBounds(self.lb.x.petsc_vec, self.ub.x.petsc_vec)
        return snes

    def _F(self, snes, x, b):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        with b.localForm() as bl:
            bl.set(0.0)
        fem.petsc.assemble_vector(b, self.F_form)
        fem.petsc.apply_lifting(b, [self.J_form], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def _J(self, snes, x, A, P):
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, self.J_form, self.bcs)
        A.assemble()

    # ------------------------------------------------------------------ #
    def solve(self):
        log(LogLevel.INFO, f"Solving {self.prefix}")
        self.solver.solve(None, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        return self.solver.getIterationNumber(), self.solver.getConvergedReason()

    def destroy(self):
        self.solver.destroy()
        self.b.destroy()
        self.a.destroy()


# ---------------------------------------------------------------------------
# SNES converged-reason helpers — used by the staggered solver to decide
# when to halve dt (true divergence) vs. accept (genuine convergence).
# ---------------------------------------------------------------------------
# PETSc's SNES converged-reason codes: positive = converged variants,
# negative = diverged variants, 0 = "iterating".  See PETSc docs
# https://petsc.org/release/manualpages/SNES/SNESConvergedReason/.
def is_diverged(reason: int) -> bool:
    """True iff PETSc reports a hard divergence we should react to.

    Excludes 0 (still iterating, never returned post-solve) and positive
    converged codes.  Treats SNES_DIVERGED_MAX_IT as a soft warning rather
    than a hard divergence — the staggered loop's outer iterations may
    still drive convergence even when one inner SNES hits its iteration
    cap.
    """
    if reason is None:
        return False
    return int(reason) < 0 and int(reason) != PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT


def reason_label(reason: int) -> str:
    """Short label for a PETSc SNES reason — used in stagger-step traces."""
    try:
        return str(PETSc.SNES.ConvergedReason(int(reason)).name)
    except Exception:
        return f"reason={reason}"
