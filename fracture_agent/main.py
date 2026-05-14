"""Command-line entry point.

    python -m fracture_agent.main  [--image path.png]* [--audio clip.wav]* [--nprocs 4]
                             [--prompt "..."]  [--resume run_id]
                             [--no-execute]

If no --prompt is given the user is prompted interactively.  Multiple
--image and --audio flags may be repeated.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple


# Ensure the console can encode the non-ASCII glyphs the modular solvers
# print (``Δt``, ``✓``, box-drawing).  Windows defaults to cp1252 which
# crashes on these; reconfiguring stdout to UTF-8 with replacement char
# preserves the run-summary output even on legacy consoles.  No-op on
# Linux/macOS terminals which already default to UTF-8.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from .config import WSL_MPI_DEFAULT
from .llm import AllKeysExhausted
from .orchestrator import (advise, conceptualise, fill_material,
                           plan_and_build, qa, run_with_debug,
                           run_with_reflect_revise)
from .state import SessionState


def _cli_asker(questions: List[str]) -> List[str]:
    """Default Clarifier — prompt the user at the terminal."""
    answers: List[str] = []
    for q in questions:
        print(f"\n  Q: {q}")
        try:
            a = input("  A: ").strip()
        except EOFError:
            a = ""
        answers.append(a)
    return answers


def build_user_inputs(prompt: str | None,
                      images: List[Path],
                      audios: List[Path]) -> List[Tuple[str, object]]:
    parts: List[Tuple[str, object]] = []
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read()
    elif prompt is None:
        print("Describe the phase-field fracture problem you want to solve")
        print("(multi-line; end with Ctrl-D or Ctrl-Z+Enter):")
        try:
            prompt = sys.stdin.read()
        except KeyboardInterrupt:
            sys.exit(0)
    parts.append(("text", prompt.strip()))
    for p in images:
        parts.append(("image", str(p)))
    for p in audios:
        parts.append(("audio", str(p)))
    return parts


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fracture_agent")
    ap.add_argument("--prompt", type=str, default=None,
                    help="User description of the problem.")
    ap.add_argument("--image", action="append", default=[],
                    help="Path to a sketch / photograph / figure.")
    ap.add_argument("--audio", action="append", default=[],
                    help="Path to a voice-clip input.")
    ap.add_argument("--nprocs", type=int, default=WSL_MPI_DEFAULT,
                    help="MPI ranks for the WSL execution.")
    ap.add_argument("--resume", type=str, default=None,
                    help="Resume an existing session by id (run_YYYYMMDD_HHMMSS).")
    ap.add_argument("--no-execute", action="store_true",
                    help="Stop after generating the script; do not run it.")
    ap.add_argument("--ui", action="store_true",
                    help="Launch the browser UI instead of the CLI.")
    ap.add_argument("--port", type=int, default=7860,
                    help="Port for --ui (default 7860).")
    ap.add_argument("--ablation", type=int, default=4, choices=(1, 2, 3, 4),
                    help="Ablation level: 1=B1 one-shot LLM, 2=B2 +Inspector, "
                         "3=B3 full inner loop (no Reflect-Revise), "
                         "4=B4 full fracture_agent (default).")
    args = ap.parse_args(argv)

    if args.ui:
        from .ui.server import main as ui_main
        return ui_main(port=args.port) or 0

    if args.resume:
        state = SessionState.load(args.resume)
        print(f"[resume] session {state.session_id}")
    else:
        state = SessionState.new()
        print(f"[new session] {state.session_id}")
    # Wire telemetry sink so per-LLM-call costs/tokens land in this session.
    from .orchestrator import attach_telemetry
    attach_telemetry(state)

    images = [Path(p) for p in args.image]
    audios = [Path(p) for p in args.audio]
    user_inputs = build_user_inputs(args.prompt, images, audios)
    state.save()

    try:
        # Ablation-level routing (A3).  Level 4 is the default, full fracture_agent.
        if args.ablation < 4:
            from .ablation import AblationLevel, run_at_level
            level = AblationLevel(args.ablation)
            print(f"[ablation] running at level {level.name}")
            script, rec, health = run_at_level(
                state, user_inputs,
                level=level,
                ask_user=_cli_asker,
                nprocs=args.nprocs,
                execute=not args.no_execute,
            )
            if args.no_execute or rec is None:
                print(f"[ablation] script: {script}")
                return 0
            if health is not None:
                print(f"[health] {health.total:.1f}/100 [{health.verdict}]")
            if rec.returncode != 0 or rec.diverged:
                print(f"\n[ablation] level {level.name} did not cleanly "
                      f"complete (rc={rec.returncode}, diverged={rec.diverged})")
                return 1
            return 0

        spec = conceptualise(state, user_inputs, _cli_asker)
        spec = fill_material(spec, _cli_asker)
        state.spec = spec; state.save()
        # Rename the session folder to a descriptive slug now that the spec
        # is finalised.  Every downstream artefact lands inside this folder.
        new_dir = state.rename_to_slug()
        print(f"[session] folder: {new_dir}")

        # When the user passes --no-execute we don't have WSL available
        # (or they're just inspecting the generated artefacts), so skip the
        # WSL-side mesh probe too.  Without this, --no-execute still hangs
        # for ~20s on a wsl.exe spin-up before bailing.
        action, script = plan_and_build(state, spec, probe_mesh=not args.no_execute)
        print(f"\n[strategist] {action.rationale}")
        print(f"[synthesizer] wrote {script}")
        if args.no_execute:
            print("--no-execute given; stopping before run.")
            return 0

        script, rec, _health = run_with_reflect_revise(
            state, spec, action, script, nprocs=args.nprocs)
        summary = advise(state, spec, script, rec)
        if summary.diverged or rec.returncode != 0:
            print("\n[advisor] The run did not cleanly complete — inspect "
                  f"{script.parent} for details.")
            return 1
        qa(state, spec, script)
        return 0

    except AllKeysExhausted as e:
        print(f"\n[FATAL] {e}\n"
              "All configured Gemini keys are rate-limited on the "
              "primary model.  As requested, we are not falling back to a "
              "smaller model.  Wait for the quota to reset, or add another "
              "key.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
