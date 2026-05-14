"""Central configuration: API keys, model names, paths.

API keys are loaded from environment variables. You can either export them in
your shell or drop them into a ``.env`` file at the repo root (see
``.env.example``). At least one ``GEMINI_API_KEY`` is required; up to three
keys can be supplied to enable round-robin rotation under free-tier RPM
limits.
"""
from __future__ import annotations
import os
from pathlib import Path

# Optional: load .env if python-dotenv is installed. Soft-fail if not.
try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass


def _collect_gemini_keys() -> list[str]:
    """Read up to three rotating Gemini keys from the environment.

    Recognised names (in order):
        GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3
        plus the legacy alias GOOGLE_API_KEY.
    """
    names = ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
             "GOOGLE_API_KEY")
    keys: list[str] = []
    for n in names:
        v = os.environ.get(n, "").strip()
        if v and v not in keys:
            keys.append(v)
    return keys


GEMINI_KEYS = _collect_gemini_keys()

if not GEMINI_KEYS:
    # Defer the hard failure to first use so tooling like `python -c
    # "import fracture_agent"` still works without a configured key.
    import warnings
    warnings.warn(
        "No Gemini API key found. Set GEMINI_API_KEY (and optionally "
        "GEMINI_API_KEY_2 / GEMINI_API_KEY_3) in your environment, or "
        "create a .env file from .env.example.",
        stacklevel=2,
    )

# Model routing. Reasoning-heavy calls (Advisor physics answers, Strategist
# rationale) use PRIMARY_MODEL. Extraction tasks (Receptionist, Architect,
# Debugger, Mesh-LLM, Material handbook lookup) use FAST_MODEL: Flash matches
# Pro on structured extraction while cutting latency ~3x and being far less
# likely to thinking-deadlock. Set FAST_MODEL = PRIMARY_MODEL to use Pro
# everywhere.
PRIMARY_MODEL = os.environ.get("FRACTURE_AGENT_PRIMARY_MODEL", "gemini-2.5-pro")
FAST_MODEL    = os.environ.get("FRACTURE_AGENT_FAST_MODEL",    "gemini-2.5-flash")
VISION_MODEL  = os.environ.get("FRACTURE_AGENT_VISION_MODEL",  "gemini-2.5-pro")
EMBED_MODEL   = os.environ.get("FRACTURE_AGENT_EMBED_MODEL",   "gemini-embedding-001")

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# ---- Paths ---------------------------------------------------------------- #
PKG_ROOT   = Path(__file__).resolve().parent
REPO_ROOT  = PKG_ROOT.parent
MODULAR    = REPO_ROOT / "modular"

# Every simulation session lives under RUNS_DIR in its own slug-named
# subfolder (material_shape_loading). Script, mesh module, material json,
# xdmf snapshots, log, and state.json all land inside the same subfolder.
RUNS_DIR = REPO_ROOT / "agentic_simulations"
RUNS_DIR.mkdir(exist_ok=True)

# ---- WSL execution -------------------------------------------------------- #
# Mount path inside WSL that maps to REPO_ROOT on Windows.
def repo_root_wsl() -> str:
    """Return the repo path as WSL sees it (/mnt/c/... style)."""
    p = REPO_ROOT.as_posix()
    if len(p) >= 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p

WSL_CONDA_ENV   = os.environ.get("FRACTURE_AGENT_CONDA_ENV", "fenicsx")
WSL_MPI_DEFAULT = int(os.environ.get("FRACTURE_AGENT_MPI_DEFAULT", "1"))

# ---- Agent loop limits ---------------------------------------------------- #
MAX_DEBUG_ITERS = 5            # Code -> run -> fix retries
MAX_MESH_ITERS  = 3            # mesh-size rescale retries
EXECUTION_TIMEOUT_S = 60 * 60  # 1 hour per run
