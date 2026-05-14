"""Gemini REST client with transparent API-key cycling.

We never downgrade from PRIMARY_MODEL — on rate-limit / quota errors we
rotate keys; if every key is rate-limited we raise ``AllKeysExhausted``,
which the orchestrator turns into a terminal user-facing message.
"""
from __future__ import annotations
import base64
import json
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from .config import API_ROOT, EMBED_MODEL, GEMINI_KEYS, PRIMARY_MODEL


class AllKeysExhausted(RuntimeError):
    """Raised when every configured key has hit its quota for the target model."""


class LLMError(RuntimeError):
    """Raised on non-transient failure (malformed content, safety block, etc.)."""


# ---- Utility: attach one inline blob (image / audio / pdf) to a request ---- #
def _inline_part(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {"inline_data": {"mime_type": mime, "data": data}}


@dataclass
class KeyState:
    key: str
    cooldown_until: float = 0.0     # monotonic seconds — avoid this key until
    disabled: bool = False          # permanent (auth fail, bad key fmt)
    last_error: str = ""


@dataclass
class Gemini:
    """Stateful client that rotates across ``GEMINI_KEYS`` per request.

    Usage:
        g = Gemini()
        text = g.complete(system="...", user="...", temperature=0.2)
        json_obj = g.complete_json(system="...", user="...", schema={...})
        text = g.complete(system="...", user="...", files=["sketch.png"])
    """
    keys: List[KeyState] = field(default_factory=list)
    max_retries_per_call: int = 8
    default_model: str = PRIMARY_MODEL
    total_calls: int = 0

    def __post_init__(self):
        if not self.keys:
            if not GEMINI_KEYS:
                raise AllKeysExhausted(
                    "No Gemini API key configured. Set GEMINI_API_KEY "
                    "(and optionally GEMINI_API_KEY_2 / GEMINI_API_KEY_3) "
                    "in your environment, or copy .env.example to .env and "
                    "fill in your key. Get a key at "
                    "https://aistudio.google.com/app/apikey.")
            self.keys = [KeyState(k) for k in GEMINI_KEYS]

    # ------------------------------------------------------------------ #
    def _pick_key(self) -> KeyState:
        now = time.monotonic()
        live = [k for k in self.keys if (not k.disabled) and k.cooldown_until <= now]
        if not live:
            soonest = min((k.cooldown_until for k in self.keys
                           if not k.disabled), default=None)
            if soonest is None:
                raise AllKeysExhausted(
                    "All Gemini API keys are permanently disabled "
                    f"(auth / format errors). Last errors: "
                    + "; ".join(k.last_error for k in self.keys))
            wait = max(0.0, soonest - now)
            if wait > 120:
                raise AllKeysExhausted(
                    f"All {len(self.keys)} keys rate-limited on {self.default_model}; "
                    f"earliest retry in {wait:.0f}s. Stopping per user instruction "
                    f"(best models only, no downgrade).")
            time.sleep(wait + 0.2)
            return self._pick_key()
        # Prefer the key with the oldest (lowest) cooldown release; i.e. simple
        # round-robin among live keys.
        live.sort(key=lambda k: k.cooldown_until)
        return live[0]

    # ------------------------------------------------------------------ #
    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Rotating POST with quota handling.  Returns parsed JSON.

        Side-effect: on a 200 response, records a telemetry row for the
        active session via ``fracture_agent.telemetry.record_llm_call``.  The
        telemetry sink is a no-op when no session is active.
        """
        from .telemetry import record_llm_call

        # Best-effort introspection of the model & google_search flag for
        # the telemetry tag — pulled from the URL and payload.
        model = url.rsplit("/", 1)[-1].split(":", 1)[0]
        used_search = bool(payload.get("tools"))

        attempt = 0
        while attempt < self.max_retries_per_call:
            attempt += 1
            ks = self._pick_key()
            t0 = time.monotonic()
            try:
                r = requests.post(
                    url, params={"key": ks.key}, json=payload,
                    timeout=300,
                    headers={"Content-Type": "application/json"})
            except requests.RequestException as e:
                ks.cooldown_until = time.monotonic() + 5 * attempt
                ks.last_error = f"network: {e}"
                continue

            latency_ms = (time.monotonic() - t0) * 1000.0

            # --- Decode --------------------------------------------------- #
            if r.status_code == 200:
                self.total_calls += 1
                try:
                    j = r.json()
                except json.JSONDecodeError as e:
                    raise LLMError(f"Gemini returned non-JSON: {e}; text={r.text[:400]}")
                # ---- telemetry ------------------------------------------ #
                meta = j.get("usageMetadata", {}) or {}
                # Gemini reports thoughts separately on 2.5 ("thoughts" or
                # "thoughtsTokenCount"); fall back to 0 if missing.
                thinking_tokens = int(
                    meta.get("thoughtsTokenCount")
                    or meta.get("thoughts_token_count") or 0)
                cands = j.get("candidates") or [{}]
                finish_reason = (cands[0].get("finishReason")
                                 or cands[0].get("finish_reason") or "")
                try:
                    record_llm_call(
                        model=model,
                        prompt_tokens=int(meta.get("promptTokenCount") or 0),
                        output_tokens=int(meta.get("candidatesTokenCount") or 0),
                        thinking_tokens=thinking_tokens,
                        latency_ms=latency_ms,
                        used_search=used_search,
                        finish_reason=str(finish_reason),
                        attempt=attempt,
                    )
                except Exception:
                    pass  # never let telemetry break the call
                return j

            body = r.text[:600]
            if r.status_code in (429, 503):                         # rate limit / overloaded
                # Some 429 / PERMISSION_DENIED errors are TERMINAL for this
                # key (billing cap / project suspended).  Mark permanent.
                body_lower = body.lower()
                terminal_markers = (
                    "monthly spending cap", "billing", "quota exceeded",
                    "project has been suspended", "permission_denied",
                    "api key is invalid",
                )
                if any(m in body_lower for m in terminal_markers):
                    ks.disabled = True
                    ks.last_error = f"terminal 429: {body[:200]}"
                    continue
                # Otherwise transient — put on cooldown for retryDelay or 30 s.
                retry = 30.0
                try:
                    j = r.json()
                    for d in j.get("error", {}).get("details", []):
                        if isinstance(d, dict) and "retryDelay" in d:
                            s = d["retryDelay"].rstrip("s")
                            retry = float(s) if s else retry
                except Exception:
                    pass
                ks.cooldown_until = time.monotonic() + retry
                ks.last_error = f"429/503: {body[:200]}"
                continue
            if r.status_code in (401, 403):                         # bad / revoked key
                ks.disabled = True
                ks.last_error = f"auth {r.status_code}: {body[:200]}"
                continue
            if 500 <= r.status_code < 600:                          # transient server error
                ks.cooldown_until = time.monotonic() + 15
                ks.last_error = f"{r.status_code}: {body[:200]}"
                continue
            raise LLMError(f"Gemini {r.status_code}: {body}")

        raise AllKeysExhausted(
            f"Exceeded {self.max_retries_per_call} retries across keys; "
            f"last errors: {[k.last_error for k in self.keys]}")

    # ---- public API ---------------------------------------------------- #
    def complete(self,
                 system: str,
                 user: Union[str, List[Any]],
                 *,
                 files: Optional[List[Union[str, Path]]] = None,
                 temperature: float = 0.2,
                 max_output_tokens: int = 16384,
                 thinking_budget: Optional[int] = None,
                 use_google_search: bool = False,
                 model: Optional[str] = None) -> str:
        """Plain-text completion.  `files` attaches images / audio / pdf.

        ``thinking_budget``:
          * ``None`` (default) — model decides (Gemini 2.5 Pro default).
          * ``0``              — disable thinking entirely (use for extraction).
          * positive int       — cap thinking tokens at this value.

        ``use_google_search``:
          * ``True`` attaches Gemini's ``google_search`` grounding tool —
            the model can issue real-time web searches for up-to-date or
            obscure facts (e.g. handbook material properties for exotic
            materials).  Incompatible with ``responseSchema``.
        """
        model = model or self.default_model
        parts: List[Dict[str, Any]] = []
        if isinstance(user, str):
            parts.append({"text": user})
        else:
            for p in user:
                parts.append({"text": p} if isinstance(p, str) else p)
        for f in (files or []):
            parts.append(_inline_part(f))

        gen: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
        if thinking_budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen,
        }
        if use_google_search:
            payload["tools"] = [{"google_search": {}}]
        url = f"{API_ROOT}/models/{model}:generateContent"

        # Same MAX_TOKENS safety net as complete_json: on overrun, double
        # the budget and cap thinking.  Up to 3 attempts total.
        for doubling in range(3):
            try:
                j = self._post(url, payload)
                return self._extract_text(j)
            except LLMError as e:
                if "MAX_TOKENS" in str(e) and doubling < 2:
                    gen["maxOutputTokens"] = int(gen["maxOutputTokens"] * 2)
                    cur = gen.get("thinkingConfig", {}).get("thinkingBudget", -1)
                    if cur in (-1, None):
                        gen["thinkingConfig"] = {"thinkingBudget": 1024}
                    else:
                        gen["thinkingConfig"] = {
                            "thinkingBudget": max(128, int(cur) // 2)}
                    continue
                raise
        # Unreachable except for exhausted retries.
        raise LLMError("complete() retry loop exhausted without response")

    def complete_json(self,
                      system: str,
                      user: Union[str, List[Any]],
                      *,
                      schema: Optional[Dict[str, Any]] = None,
                      files: Optional[List[Union[str, Path]]] = None,
                      temperature: float = 0.1,
                      max_output_tokens: int = 16384,
                      thinking_budget: Optional[int] = -1,
                      model: Optional[str] = None) -> Dict[str, Any]:
        """JSON-mode completion.

        Defaults:
          * ``thinking_budget = -1`` — Gemini 2.5 Pro *requires* thinking
            mode.  -1 means "model decides"; positive ints cap the budget.
            With generous ``max_output_tokens`` this avoids MAX_TOKENS
            truncation even when the model thinks a lot.
          * If ``schema`` is provided, Gemini is told to conform to it
            exactly (responseSchema).  Pydantic models can be converted with
            :func:`pydantic_to_gemini_schema` below.
        """
        model = model or self.default_model
        parts: List[Dict[str, Any]] = []
        if isinstance(user, str):
            parts.append({"text": user})
        else:
            for p in user:
                parts.append({"text": p} if isinstance(p, str) else p)
        for f in (files or []):
            parts.append(_inline_part(f))

        gen: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
        }
        if schema is not None:
            gen["responseSchema"] = schema
        if thinking_budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen,
        }
        url = f"{API_ROOT}/models/{model}:generateContent"

        # Three auto-retries covering two distinct failure modes:
        #   1. MAX_TOKENS  -> double maxOutputTokens, cap thinking budget
        #   2. Malformed JSON (truncated response, stray prose) -> double
        #      maxOutputTokens, and on the final retry also drop the
        #      responseSchema (Gemini 2.5 Flash sometimes emits partial JSON
        #      when the schema is deeply nested).
        last_err: Optional[Exception] = None
        for doubling in range(3):
            try:
                j = self._post(url, payload)
                txt = self._extract_text(j)
                return _parse_json_loose(txt)
            except LLMError as e:
                last_err = e
                if "MAX_TOKENS" in str(e) and doubling < 2:
                    gen["maxOutputTokens"] = int(gen["maxOutputTokens"] * 2)
                    cur = gen.get("thinkingConfig", {}).get("thinkingBudget", -1)
                    if cur in (-1, None):
                        gen["thinkingConfig"] = {"thinkingBudget": 2048}
                    else:
                        gen["thinkingConfig"] = {
                            "thinkingBudget": max(128, int(cur) // 2)}
                    continue
                raise
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                if doubling >= 2:
                    raise LLMError(
                        f"Gemini returned malformed JSON after 3 attempts: {e}"
                    ) from e
                # Budget bump + on the FINAL retry drop the responseSchema.
                gen["maxOutputTokens"] = int(gen["maxOutputTokens"] * 2)
                if doubling == 1 and "responseSchema" in gen:
                    del gen["responseSchema"]
                continue
        # unreachable — every branch returns or raises
        raise LLMError(f"complete_json exhausted retries: {last_err}")

    def embed(self, text: str, *, model: str = EMBED_MODEL) -> List[float]:
        url = f"{API_ROOT}/models/{model}:embedContent"
        payload = {"content": {"parts": [{"text": text}]}}
        j = self._post(url, payload)
        return j["embedding"]["values"]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_text(j: Dict[str, Any]) -> str:
        try:
            cands = j["candidates"]
            if not cands:
                fb = j.get("promptFeedback", {})
                raise LLMError(f"No candidates returned (feedback={fb})")
            finish = cands[0].get("finishReason", "?")
            parts = cands[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            # Two failure shapes trigger the retry path:
            #   - completely empty text (we already surface this)
            #   - non-STOP finishReason with partial text (Gemini truncated
            #     mid-response; caller gets garbled JSON otherwise).
            if not text:
                raise LLMError(f"Empty text; finishReason={finish}; full={j}")
            if finish and finish not in ("STOP", "?", None):
                raise LLMError(
                    f"Truncated response; finishReason={finish}; "
                    f"partial_len={len(text)}; full={j}")
            return text
        except KeyError as e:
            raise LLMError(f"Unexpected response shape: missing {e}; got {j}") from e


def _parse_json_loose(txt: str) -> Dict[str, Any]:
    """Parse JSON tolerantly: strip ``` fences, allow leading/trailing noise."""
    s = txt.strip()
    if s.startswith("```"):
        # ``` [json]? \n ... \n ```
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rsplit("```", 1)[0]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Find the first balanced { ... } or [ ... ] block.
        start = min((s.find(c) for c in "{[" if s.find(c) >= 0), default=-1)
        if start >= 0:
            depth = 0
            for i, ch in enumerate(s[start:], start):
                if ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        return json.loads(s[start:i + 1])
        raise


# ---------------------------------------------------------------------------
# Pydantic → Gemini responseSchema converter.
# Gemini accepts an OpenAPI-3-subset JSONSchema:
#   - only these "type" values: object, array, string, number, integer, boolean
#   - no $defs / $refs (we inline)
#   - no oneOf / anyOf (we flatten nullable by union of base type)
#   - `required` must name keys in `properties`
# ---------------------------------------------------------------------------
def pydantic_to_gemini_schema(model_cls) -> Dict[str, Any]:
    """Return a Gemini-compatible responseSchema for a Pydantic BaseModel."""
    raw = model_cls.model_json_schema()
    return _clean_schema(raw, raw.get("$defs", {}))


def _clean_schema(node: Any, defs: Dict[str, Any]) -> Any:
    if not isinstance(node, dict):
        return node
    # Resolve $ref.
    if "$ref" in node:
        ref = node["$ref"].rsplit("/", 1)[-1]
        target = defs.get(ref, {})
        return _clean_schema(target, defs)
    # anyOf — pick the first non-null variant.
    if "anyOf" in node:
        variants = [v for v in node["anyOf"]
                    if not (isinstance(v, dict) and v.get("type") == "null")]
        chosen = variants[0] if variants else node["anyOf"][0]
        merged = {k: v for k, v in node.items() if k != "anyOf"}
        merged.update({k: v for k, v in chosen.items()})
        node = merged
    # Recurse into children.
    out: Dict[str, Any] = {}
    for k, v in node.items():
        if k in ("$defs", "definitions", "title", "additionalProperties",
                 "discriminator", "const"):
            continue
        if k == "default":
            # Gemini doesn't use defaults — drop.
            continue
        if k == "enum":
            out[k] = v
            continue
        if isinstance(v, dict):
            out[k] = _clean_schema(v, defs)
        elif isinstance(v, list):
            out[k] = [_clean_schema(item, defs) for item in v]
        else:
            out[k] = v
    # Gemini requires type for objects.
    if "properties" in out and "type" not in out:
        out["type"] = "object"
    # Strip `required` down to fields that are actually in `properties`.
    if "required" in out and isinstance(out["required"], list):
        props = out.get("properties") or {}
        out["required"] = [r for r in out["required"] if r in props]
        if not out["required"]:
            out.pop("required")
    return out


# Module-level singleton for convenience.
_GLOBAL: Optional[Gemini] = None


def llm() -> Gemini:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = Gemini()
    return _GLOBAL
