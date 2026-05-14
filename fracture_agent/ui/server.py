"""Flask server wrapping the fracture_agent orchestrator with a live web UI.

Architecture:

  * ``POST /run``                - start a new simulation.  Accepts JSON or
                                   ``multipart/form-data`` for image upload.
                                   Returns a ``session_id`` immediately and
                                   spawns a worker thread that streams events
                                   into an in-memory queue.
  * ``GET  /stream/<sid>``       - Server-Sent Events stream; the browser
                                   consumes it with EventSource.
  * ``POST /clarify/<sid>``      - the UI's response to an agent
                                   clarification request.  Worker is parked
                                   on a threading.Event until this comes in
                                   (with a 10-minute timeout fallback).
  * ``POST /ask/<sid>``          - chat endpoint; calls the Advisor.
  * ``GET  /img/<dir>/<name>``   - serve a rendered PNG.
  * ``GET  /file/<dir>/<name>``  - serve any file from the session folder
                                   (download path: state.json, run_*.py,
                                   custom_mesh.py, output log, ...).
  * ``GET  /sessions``           - list recent sessions (history panel).

Stateful per-session data lives in ``SESSIONS`` (an in-memory dict).  When
the orchestrator completes, a small render subprocess runs inside WSL to
dump initial_mesh.png / final_damage.png / load_disp.png.

The UI side never modifies orchestrator behaviour — every change here is
either a new endpoint, a new event-dict field, or a richer payload on the
DONE event.  The CLI (``fracture_agent.main``) is untouched and still uses the
original ``_cli_asker``.
"""
from __future__ import annotations
import json
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from flask import (Flask, Response, abort, jsonify, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

from ..agents.advisor import answer_question
from ..config import RUNS_DIR
from ..events import (CONSOLE, DECISION, DONE, ERROR, IMAGE, METADATA,
                       RESULT, STATUS, emit, set_sink)
from ..executor import _path_to_wsl
from ..orchestrator import (advise, attach_telemetry, conceptualise,
                             fill_material, plan_and_build,
                             run_with_reflect_revise)
from ..state import SessionState

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload cap

SESSIONS: Dict[str, Dict[str, Any]] = {}
CLARIFY_TIMEOUT_S = 600  # 10-minute user timeout per clarification round

# Quick-load examples surfaced in the UI's "Examples" dropdown.
EXAMPLES = [
    {
        "label": "Steel SENT (Miehe 2010)",
        "tags": ["brittle", "mode-I", "fully-specified"],
        "prompt": (
            "1 mm by 2 mm steel plate in plane strain. Single-edge notched "
            "tension specimen with a horizontal crack from the left edge at "
            "mid-height of length 0.5 mm. Bottom edge fully fixed. Top edge "
            "displaced vertically upward by 0.005 mm. E = 210 GPa, nu = 0.3, "
            "Gc = 2.7 N/mm, sigma_ts = 2000 MPa, sigma_cs = 5000 MPa. Run 100 "
            "steps."
        ),
    },
    {
        "label": "Graphite shear (top-edge horizontal pull)",
        "tags": ["mode-II", "shear", "fully-specified"],
        "prompt": (
            "25 mm by 25 mm graphite plate in plane stress with a horizontal "
            "pre-crack from the left edge at mid-height, 12.5 mm long. "
            "Bottom edge fully fixed (u_x = u_y = 0). Top edge pulled "
            "horizontally to the right by 0.08 mm (in-plane shear loading). "
            "Material: E = 9.8 GPa, nu = 0.13, Gc = 91 N/m, sigma_ts = 27 MPa, "
            "sigma_cs = 77 MPa. Run 100 steps."
        ),
    },
    {
        "label": "Concrete two-crack multi-stage",
        "tags": ["multi-stage", "two-crack", "custom"],
        "prompt": (
            "200 mm by 200 mm concrete plate, plane stress, with two "
            "horizontal edge cracks: one 25 mm long entering from the left "
            "edge at y = 102.5 mm, and one 25 mm long entering from the "
            "right edge at y = 97.5 mm. Bottom edge fully fixed. Left edge "
            "first shifted horizontally to the right by 0.005 mm "
            "(pre-shear, held constant). Top edge then pulled vertically "
            "upward by 0.02 mm. Material: E = 30 GPa, nu = 0.2, "
            "Gc = 11 N/m, sigma_ts = 5 MPa, sigma_cs = 15 MPa."
        ),
    },
    {
        "label": "Marble 45° centre notch + roller",
        "tags": ["compression", "internal-notch", "roller"],
        "prompt": (
            "80 mm wide by 104 mm tall marble plate in plane stress with a "
            "20 mm long straight notch located at the geometric centre and "
            "oriented at 45 degrees from the horizontal axis. Bottom edge "
            "constrained vertically only (u_y = 0, u_x free — roller). Top "
            "edge compressed downward by 0.1 mm. Material: E = 63.5 GPa, "
            "nu = 0.21, Gc = 11 N/m, sigma_ts = 10 MPa, sigma_cs = 175 MPa."
        ),
    },
    {
        "label": "Graphite sheet — top half pulled apart",
        "tags": ["partial-edge", "sym-pull"],
        "prompt": (
            "10 mm wide by 20 mm tall graphite sheet, plane stress, with a "
            "crack from top centre edge to centre. Bottom fixed. Top half of "
            "left and right edges pulled apart, displaced by 1 mm each."
        ),
    },
    {
        "label": "L-shape alumina, top-left pulled outward",
        "tags": ["L-shape", "outward"],
        "prompt": (
            "L shaped domain of alumina with outer lengths of 15 mm, inner "
            "of 10 mm. Bottom edge fixed. Top half of left edge pulled "
            "outside by 2 mm."
        ),
    },
    {
        "label": "Vague: copper bar tensile test",
        "tags": ["vague", "fill-the-gaps"],
        "prompt": "uniaxial tensile test of a rectangular bar 50 mm by 10 mm of copper plane stress",
    },
    {
        "label": "3D steel cube",
        "tags": ["3D", "tension"],
        "prompt": (
            "10 mm by 10 mm by 10 mm steel cube in 3D. Bottom face (z = 0) "
            "fully fixed. Top face (z = 10) pulled upward in z by 0.1 mm. "
            "E = 200 GPa, nu = 0.3, Gc = 5 N/mm, sigma_ts = 500 MPa."
        ),
    },
]


# ---------------------------------------------------------------------------
def _push(sess: Dict[str, Any], **payload) -> None:
    """Append an event to the session's SSE queue, defaulting category."""
    payload.setdefault("category", STATUS)
    payload.setdefault("ts", time.time())
    sess["queue"].put(payload)


def _make_ui_asker(session_id: str) -> Callable[[List[str]], List[str]]:
    """Build a Clarifier closure that round-trips through the UI.

    On call: emits a ``clarify`` event with the questions, then blocks on
    ``clarify_event`` until ``/clarify/<sid>`` POSTs answers (or the
    ``CLARIFY_TIMEOUT_S`` fallback fires, in which case empty answers are
    returned and the orchestrator proceeds with whatever defaults it has —
    same fallback shape as the CLI's ``_silent_asker``).
    """
    def asker(questions: List[str]) -> List[str]:
        sess = SESSIONS.get(session_id)
        if sess is None:
            return [""] * len(questions)
        sess["clarify_event"].clear()
        sess["clarify_questions"] = list(questions)
        sess["clarify_answers"] = None
        _push(sess, category="clarify", questions=list(questions),
              prompt_count=len(questions))
        # Block until UI POSTs to /clarify/<sid> or timeout.
        if not sess["clarify_event"].wait(timeout=CLARIFY_TIMEOUT_S):
            _push(sess, category="status",
                  message=("Clarification timed out after "
                           f"{CLARIFY_TIMEOUT_S}s — proceeding with defaults."))
            return [""] * len(questions)
        ans = sess.get("clarify_answers") or [""] * len(questions)
        # Pad / truncate so the orchestrator always gets the right shape.
        if len(ans) < len(questions):
            ans = list(ans) + [""] * (len(questions) - len(ans))
        return list(ans[:len(questions)])
    return asker


def _silent_asker(questions: List[str]) -> List[str]:
    """Legacy fallback used only if the UI hasn't initialised the session."""
    return [""] * len(questions)


# ---------------------------------------------------------------------------
def _push_assumptions(sess: Dict[str, Any], state: SessionState,
                       phase: str) -> None:
    """Push any newly-added spec.assumptions since the last snapshot."""
    if state.spec is None:
        return
    seen = sess.setdefault("seen_assumptions", set())
    new = []
    for a in state.spec.assumptions:
        if a not in seen:
            seen.add(a); new.append(a)
    if new:
        _push(sess, category="assumption", phase=phase, items=new)


def _push_telemetry(sess: Dict[str, Any], state: SessionState,
                     phase: str = "") -> None:
    """Push a snapshot of the per-session telemetry."""
    try:
        t = state.telemetry
        totals = t.totals()
        by_agent = t.by_agent()
        iters = t.iters.model_dump()
    except Exception:
        return
    _push(sess, category="telemetry", phase=phase, totals=totals,
          by_agent=by_agent, iters=iters)


def _list_session_files(session_dir: Path) -> List[Dict[str, Any]]:
    """Enumerate downloadable artefacts in a session folder."""
    out: List[Dict[str, Any]] = []
    if not session_dir.exists():
        return out
    # Show interesting files in a deterministic order.
    interesting = ("state.json", "cost_table.csv", "material.json",
                    "custom_mesh.py")
    for name in interesting:
        p = session_dir / name
        if p.exists():
            out.append({"name": name, "size": p.stat().st_size,
                        "kind": _file_kind(p)})
    for p in sorted(session_dir.glob("run_*.py")):
        out.append({"name": p.name, "size": p.stat().st_size, "kind": "code"})
    for p in sorted(session_dir.glob("output_*.txt")):
        out.append({"name": p.name, "size": p.stat().st_size, "kind": "log"})
    for p in sorted(session_dir.glob("*.png")):
        out.append({"name": p.name, "size": p.stat().st_size, "kind": "image"})
    return out


def _file_kind(p: Path) -> str:
    s = p.suffix.lower()
    return {
        ".py": "code", ".json": "data", ".csv": "data",
        ".txt": "log", ".png": "image", ".xdmf": "data", ".h5": "data",
    }.get(s, "data")


# ---------------------------------------------------------------------------
def _worker(session_id: str, prompt: str, overrides: Dict[str, Any],
             image_paths: List[str]) -> None:
    """Orchestrator thread.  Translates emit() calls into queue.put()."""
    sess = SESSIONS[session_id]
    q: "queue.Queue[Dict[str, Any]]" = sess["queue"]

    def sink(category: str, message: str) -> None:
        q.put({"category": category, "message": message, "ts": time.time()})

    set_sink(sink)
    try:
        # ---- Build a prompt that carries the user's UI overrides ---- #
        extras: List[str] = [prompt]
        if overrides.get("material"):
            extras.append(f"Material: {overrides['material']}.")
        if overrides.get("geometry"):
            extras.append(f"Geometry kind: {overrides['geometry']}.")
        if overrides.get("physics"):
            extras.append(f"Physics: {overrides['physics']}.")
        if overrides.get("fracture") is False:
            extras.append(
                "Fracture should be DISABLED (elastic deformation only, no crack).")
        if overrides.get("fracture") is True:
            extras.append("Fracture must be ENABLED (phase-field fracture).")
        full_prompt = "\n".join(extras).strip()

        state = SessionState.new()
        sess["state"] = state
        sess["session_dir"] = str(state.dir)
        sess["session_id"] = state.session_id
        attach_telemetry(state)
        asker = _make_ui_asker(session_id)

        emit(STATUS, f"Session opened at {state.session_id}")

        user_inputs: List[Any] = [("text", full_prompt)]
        for p in image_paths:
            user_inputs.append(("image", p))

        spec = conceptualise(state, user_inputs, asker)
        _push_assumptions(sess, state, phase="architect")
        _push_telemetry(sess, state, phase="architect")

        spec = fill_material(spec, asker)
        _push_assumptions(sess, state, phase="material")
        _push_telemetry(sess, state, phase="material")

        state.spec = spec; state.save()
        state.rename_to_slug()
        sess["session_id"] = state.session_id
        sess["session_dir"] = str(state.dir)
        emit(STATUS, f"Session renamed to {state.session_id}")

        action, script = plan_and_build(state, spec)
        _push_assumptions(sess, state, phase="strategist")
        _push_telemetry(sess, state, phase="strategist")
        # Push the canonical spec snapshot so the UI can show the spec preview.
        _push(sess, category="spec",
              spec=state.spec.model_dump(mode="json") if state.spec else None,
              action=state.action.model_dump(mode="json") if state.action else None,
              mesh_plan=state.mesh_plan or {})

        emit(STATUS, "Running simulation in WSL (mesh -> solver -> output)")
        script, rec, _health = run_with_reflect_revise(
            state, spec, action, script)
        _push_assumptions(sess, state, phase="reviser")
        _push_telemetry(sess, state, phase="reviser")

        summary = advise(state, spec, script, rec)
        _push_telemetry(sess, state, phase="advisor")

        emit(STATUS, "Rendering mesh + damage + load-displacement plots...")
        _render_images(Path(state.dir), sess)

        # Final DONE event with the full payload for the UI.
        files = _list_session_files(Path(state.dir))
        q.put({
            "category": DONE,
            "message": state.session_id,
            "ts": time.time(),
            "session_dir": str(state.dir),
            "session_dir_name": Path(state.dir).name,
            "summary": summary.model_dump(),
            "health": (state.health_report.model_dump()
                        if state.health_report else None),
            "telemetry": state.telemetry.totals(),
            "by_agent": state.telemetry.by_agent(),
            "iters": state.telemetry.iters.model_dump(),
            "files": files,
            "revision_history": state.revision_history,
        })
    except Exception as e:
        emit(ERROR, f"{type(e).__name__}: {e}")
        q.put({"category": ERROR, "message": traceback.format_exc(),
               "ts": time.time()})
        q.put({"category": DONE, "message": "errored",
               "ts": time.time()})
    finally:
        set_sink(None)


# ---------------------------------------------------------------------------
def _render_images(session_dir: Path, sess: Dict[str, Any]) -> None:
    """Spawn the render subprocess inside WSL to produce the PNGs."""
    import subprocess
    wsl_dir = _path_to_wsl(session_dir.resolve())
    repo_wsl = _path_to_wsl(session_dir.resolve().parent.parent)   # agent_v2/
    cmd = [
        "wsl.exe", "--", "bash", "-ilc",
        (f"cd {repo_wsl} && "
         "if ! command -v conda >/dev/null 2>&1; then "
         "  for p in \"$HOME/miniconda3\" \"$HOME/miniforge3\" \"$HOME/anaconda3\"; do "
         "    [ -f \"$p/etc/profile.d/conda.sh\" ] && . \"$p/etc/profile.d/conda.sh\" && break; "
         "  done; fi; "
         "conda activate fenicsx 2>&1 >/dev/null; "
         f"python -m fracture_agent.ui.render '{wsl_dir}' 2>&1")
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        for line in (proc.stdout or "").splitlines():
            if line.strip():
                emit(CONSOLE, line)
        for fn in ("final_mesh.png", "final_damage.png", "load_disp.png"):
            p = session_dir / fn
            if p.exists():
                emit(IMAGE, fn)
    except Exception as e:
        emit(ERROR, f"Render failed: {e}")


# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/examples")
def list_examples():
    return jsonify(examples=EXAMPLES)


@app.route("/sessions")
def list_sessions():
    """Return the most recent N session folders, newest first."""
    if not RUNS_DIR.exists():
        return jsonify(sessions=[])
    out = []
    for p in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not p.is_dir() or p.name.startswith("pending_"):
            continue
        st = p / "state.json"
        if not st.exists():
            continue
        try:
            data = json.loads(st.read_text(encoding="utf-8"))
            spec = data.get("spec") or {}
            mat = (spec.get("material") or {}).get("display_name") or "?"
            kind = (spec.get("geometry") or {}).get("kind") or "?"
            const = spec.get("constitutive") or "?"
            out.append({
                "id": p.name,
                "material": mat,
                "kind": kind,
                "constitutive": const,
                "fracture": spec.get("fracture_enabled"),
                "verdict": (data.get("health_report") or {}).get("verdict"),
            })
        except Exception:
            continue
        if len(out) >= 30:
            break
    return jsonify(sessions=out)


@app.route("/run", methods=["POST"])
def run_sim():
    """Start a simulation.  Accepts either:
      * Content-Type: application/json   →  ``{prompt, material, ...}``
      * Content-Type: multipart/form-data →  prompt + overrides + image files
    """
    image_paths: List[str] = []
    if request.content_type and request.content_type.startswith("multipart/"):
        prompt = (request.form.get("prompt") or "").strip()
        overrides = {
            "material":  (request.form.get("material") or "").strip() or None,
            "geometry":  (request.form.get("geometry") or "").strip() or None,
            "physics":   (request.form.get("physics") or "").strip() or None,
            "fracture":  _truthy(request.form.get("fracture")),
        }
        for f in request.files.getlist("images"):
            if not f or not f.filename:
                continue
            safe = secure_filename(f.filename)
            tmp_dir = Path(tempfile.gettempdir()) / "fracture_agent_uploads"
            tmp_dir.mkdir(exist_ok=True)
            dest = tmp_dir / f"{int(time.time()*1000)}_{safe}"
            f.save(dest)
            image_paths.append(str(dest))
    else:
        data = request.get_json(silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        overrides = {
            "material":  (data.get("material") or "").strip() or None,
            "geometry":  (data.get("geometry") or "").strip() or None,
            "physics":   (data.get("physics") or "").strip() or None,
            "fracture":  data.get("fracture"),
        }
    if not prompt:
        return jsonify(error="Prompt is required"), 400

    session_id = f"sess_{int(time.time() * 1000)}"
    SESSIONS[session_id] = {
        "queue":              queue.Queue(),
        "state":              None,
        "session_id":         session_id,
        "session_dir":        None,
        "clarify_event":      threading.Event(),
        "clarify_questions":  [],
        "clarify_answers":    None,
        "seen_assumptions":   set(),
    }
    threading.Thread(
        target=_worker, args=(session_id, prompt, overrides, image_paths),
        daemon=True, name=f"sim-{session_id}",
    ).start()
    return jsonify(session_id=session_id)


def _truthy(v) -> Optional[bool]:
    """Coerce form-data string to ``True``/``False``/``None``."""
    if v is None: return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "on"):  return True
    if s in ("false", "0", "no", "off"): return False
    if s == "": return None
    return None


@app.route("/stream/<session_id>")
def stream(session_id: str):
    if session_id not in SESSIONS:
        return jsonify(error="unknown session"), 404
    sess = SESSIONS[session_id]

    def gen():
        q: queue.Queue = sess["queue"]
        while True:
            try:
                evt = q.get(timeout=60)
            except queue.Empty:
                # keep-alive comment so the browser doesn't close
                yield ": keepalive\n\n"; continue
            evt.setdefault("category", STATUS)
            evt["session_id"] = sess.get("session_id", session_id)
            evt["session_dir_name"] = (Path(sess["session_dir"]).name
                                        if sess.get("session_dir") else None)
            yield f"data: {json.dumps(evt, default=str)}\n\n"
            if evt["category"] == DONE:
                break

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/clarify/<session_id>", methods=["POST"])
def clarify(session_id: str):
    """Receive the user's answers to the architect's clarification round."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        return jsonify(error="unknown session"), 404
    data = request.get_json(silent=True) or {}
    answers = data.get("answers") or []
    if not isinstance(answers, list):
        return jsonify(error="answers must be a list of strings"), 400
    sess["clarify_answers"] = [str(a) for a in answers]
    sess["clarify_event"].set()
    return jsonify(ok=True, n=len(answers))


@app.route("/ask/<session_id>", methods=["POST"])
def ask(session_id: str):
    if session_id not in SESSIONS:
        return jsonify(error="unknown session"), 404
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="question required"), 400
    sess = SESSIONS[session_id]
    state: SessionState = sess.get("state")
    if state is None or state.spec is None or state.final_result is None:
        return jsonify(error="simulation not finished"), 409
    log_name = f"output_{state.spec.geometry.kind}_{state.action.variant}.txt"
    log_path = Path(state.dir) / log_name
    try:
        answer = answer_question(state.spec, state.final_result,
                                 log_path, question,
                                 history=state.conversation)
    except Exception as e:
        return jsonify(error=f"advisor failed: {e}"), 500
    state.conversation.append({"q": question, "a": answer})
    state.save()
    return jsonify(answer=answer)


@app.route("/img/<session_dir_name>/<filename>")
def image(session_dir_name: str, filename: str):
    folder = RUNS_DIR / session_dir_name
    if not folder.exists():
        return jsonify(error="not found"), 404
    return send_from_directory(folder, filename)


@app.route("/file/<session_dir_name>/<path:filename>")
def session_file(session_dir_name: str, filename: str):
    """Serve any file from a session folder for download."""
    folder = RUNS_DIR / session_dir_name
    if not folder.exists() or ".." in filename:
        return jsonify(error="not found"), 404
    target = folder / filename
    if not target.exists() or not target.is_file():
        return jsonify(error="not found"), 404
    # Force download for code / log / data files; let the browser preview images.
    as_attachment = target.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif")
    return send_from_directory(folder, filename, as_attachment=as_attachment)


# ---------------------------------------------------------------------------
def main(host: str = "127.0.0.1", port: int = 7860):
    print(f"[fracture_agent.ui] open http://{host}:{port} in your browser")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
