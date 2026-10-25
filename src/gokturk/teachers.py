"""Teacher-model API adapter.

Contract
--------
* Credentials come only from an environment variable named in configuration
  (``provider.api_key_env``) or from Kaggle Secrets placed into the environment.
  The key is never written to a file, a log line, or a dataset record.
* Only an OpenAI-compatible chat-completions wire format is implemented
  (``POST {base_url}/chat/completions``). EVREN's public documentation
  describes an OpenAI-compatible inference layer, but the concrete model ids
  and base URL for Kimi / MiMo must be verified by the operator against the
  provider's own model list before use -- this module never guesses them.
* Transport failures are classified. Retryable failures (429, 5xx, timeouts,
  empty bodies) are retried with exponential backoff and jitter. Fatal
  failures (401/403, missing key, unresolvable model id) abort the run
  immediately instead of producing bogus data.
* If the teacher is unreachable the generator stops; it never substitutes a
  local placeholder response for a teacher answer.

The implementation uses only the standard library for HTTP so that the data
pipeline can run anywhere (including Kaggle) without extra dependencies.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import (
    ConfigError,
    ProviderConfig,
    TeacherModelConfig,
    TeachersConfig,
)

LOGGER = logging.getLogger(__name__)

_USER_AGENT = "GokTurk-Ultra/0.1 (+data-generation)"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class TeacherError(RuntimeError):
    """Base class for all teacher adapter failures."""


class TeacherConfigError(TeacherError):
    """Misconfiguration: nothing was sent over the network."""


class TeacherAuthError(TeacherError):
    """401/403 or missing credentials. Fatal; retrying cannot help."""


class TeacherRequestError(TeacherError):
    """4xx caused by the request itself (bad model id, bad parameters). Fatal."""


class TeacherRateLimitError(TeacherError):
    """429 or provider-specific rate limiting. Retryable."""


class TeacherServerError(TeacherError):
    """5xx from the provider. Retryable."""


class TeacherTimeoutError(TeacherError):
    """Socket timeout / connection error. Retryable."""


class TeacherEmptyResponseError(TeacherError):
    """HTTP 200 with no usable content. Retryable, but never accepted as data."""


class TeacherUnavailableError(TeacherError):
    """Too many consecutive failures: the run must stop safely."""


RETRYABLE_ERRORS = (
    TeacherRateLimitError,
    TeacherServerError,
    TeacherTimeoutError,
    TeacherEmptyResponseError,
)


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------
@dataclass
class TeacherResponse:
    text: str
    model: str
    teacher_name: str
    latency_s: float
    attempts: int
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None

    def is_truncated(self) -> bool:
        return self.finish_reason == "length"

    def as_metadata(self) -> dict[str, Any]:
        """Non-secret metadata safe for dataset records and logs."""
        return {
            "teacher": self.teacher_name,
            "model": self.model,
            "latency_s": round(self.latency_s, 3),
            "attempts": self.attempts,
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
        }


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------
class RateLimiter:
    """Simple pacing limiter: enforces a minimum interval between calls."""

    def __init__(self, requests_per_minute: float, clock=time.monotonic):
        if requests_per_minute <= 0:
            raise TeacherConfigError("requests_per_minute must be > 0")
        self.min_interval = 60.0 / requests_per_minute
        self._clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                time.sleep(remaining)
        self._last = self._clock()


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------
class TeacherClient:
    """Client for a single OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        transport=None,
        sleep=time.sleep,
        jitter: random.Random | None = None,
    ) -> None:
        self.provider = provider
        self._transport = transport or self._http_post_json
        self._sleep = sleep
        self._rng = jitter or random.Random(0)
        self._limiter = RateLimiter(provider.requests_per_minute)

    # -- public API --------------------------------------------------------
    def health_check(self, model: str) -> dict[str, Any]:
        """Minimal round-trip that proves the endpoint and model id work.

        Returns a dict with ``ok`` plus a human-readable ``detail``. Never
        raises for auth/network problems: the caller decides what to do.
        """
        provider = self.provider
        if not provider.base_url:
            return {
                "ok": False,
                "detail": (
                    "provider.base_url is empty. Copy the verified base URL from "
                    "the provider documentation into configs/teachers.yaml."
                ),
            }
        if provider.api_key() is None:
            return {
                "ok": False,
                "detail": (
                    f"environment variable {provider.api_key_env} is not set; "
                    "no request was sent"
                ),
            }
        try:
            resp = self.chat(
                model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=8,
                temperature=0.0,
                max_attempts=1,
            )
        except TeacherError as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": bool(resp.text.strip()),
            "detail": f"model={resp.model} latency={resp.latency_s:.2f}s",
        }

    def chat(
        self,
        model: str,
        *,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        max_attempts: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> TeacherResponse:
        provider = self.provider
        if not provider.base_url:
            raise TeacherConfigError(
                "provider.base_url is empty: refusing to send requests to an "
                "unverified endpoint. Fill configs/teachers.yaml first."
            )
        try:
            api_key = provider.require_api_key()
        except ConfigError as exc:
            raise TeacherConfigError(str(exc)) from exc
        attempts_allowed = max_attempts or provider.max_attempts_per_sample
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if extra_body:
            payload.update(dict(extra_body))

        url = provider.base_url.rstrip("/") + "/chat/completions"
        last_error: TeacherError | None = None

        for attempt in range(1, attempts_allowed + 1):
            self._limiter.wait()
            started = time.monotonic()
            try:
                raw = self._transport(url, payload, api_key, provider.timeout_s)
            except TeacherAuthError:
                raise
            except TeacherRequestError:
                raise
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                LOGGER.warning(
                    "teacher request failed (attempt %d/%d, model=%s): %s: %s",
                    attempt,
                    attempts_allowed,
                    model,
                    type(exc).__name__,
                    exc,
                )
                if attempt < attempts_allowed:
                    self._sleep(self._backoff(attempt))
                continue

            latency = time.monotonic() - started
            text, usage, finish_reason = self._extract_content(raw, model)
            return TeacherResponse(
                text=text,
                model=str(raw.get("model") or model),
                teacher_name=model,
                latency_s=latency,
                attempts=attempt,
                usage=usage,
                finish_reason=finish_reason,
            )

        raise TeacherUnavailableError(
            f"model {model!r} failed after {attempts_allowed} attempts; "
            f"last error: {type(last_error).__name__}: {last_error}"
        )

    # -- internals ---------------------------------------------------------
    def _backoff(self, attempt: int) -> float:
        p = self.provider
        delay = min(p.backoff_max_s, p.backoff_base_s * (2 ** (attempt - 1)))
        return delay * (0.5 + self._rng.random() / 2.0)  # full-ish jitter

    def _http_post_json(
        self, url: str, payload: dict[str, Any], api_key: str, timeout: float
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("User-Agent", _USER_AGENT)
        for k, v in self.provider.extra_headers.items():
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - body may be unreadable
                pass
            raise _classify_http_error(exc.code, body) from exc
        except TimeoutError as exc:
            raise TeacherTimeoutError(f"timeout after {timeout}s: {exc}") from exc
        except urllib.error.URLError as exc:
            raise TeacherTimeoutError(f"connection error: {exc.reason}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise TeacherEmptyResponseError(
                f"response was not valid JSON ({len(body)} bytes)"
            ) from exc
        if not isinstance(parsed, dict):
            raise TeacherEmptyResponseError("response JSON is not an object")
        return parsed

    @staticmethod
    def _extract_content(
        raw: Mapping[str, Any], model: str
    ) -> tuple[str, dict[str, Any], str | None]:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            # Some providers return {"error": {...}} with HTTP 200.
            if "error" in raw:
                detail = str(raw.get("error"))[:400]
                raise TeacherServerError(f"provider returned an error object: {detail}")
            raise TeacherEmptyResponseError("response has no 'choices' array")
        first = choices[0] if isinstance(choices[0], Mapping) else {}
        message = first.get("message") or {}
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, list):  # content-part array
            parts = []
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            content = "".join(parts)
        if not isinstance(content, str) or not content.strip():
            raise TeacherEmptyResponseError(
                f"empty content for model {model!r} (finish_reason="
                f"{first.get('finish_reason')!r})"
            )
        usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
        finish_reason = first.get("finish_reason")
        return content.strip(), dict(usage), (
            finish_reason if isinstance(finish_reason, str) else None
        )


def _classify_http_error(status: int, body: str) -> TeacherError:
    snippet = " ".join(body.strip().split())[:300]
    if status in (401, 403):
        return TeacherAuthError(f"HTTP {status}: authentication/authorization failed. {snippet}")
    if status == 429:
        return TeacherRateLimitError(f"HTTP 429: rate limited. {snippet}")
    if 500 <= status < 600:
        return TeacherServerError(f"HTTP {status}: provider error. {snippet}")
    if status in (408, 504):
        return TeacherTimeoutError(f"HTTP {status}: gateway timeout. {snippet}")
    return TeacherRequestError(f"HTTP {status}: request rejected. {snippet}")


# ---------------------------------------------------------------------------
# pool
# ---------------------------------------------------------------------------
def normalize_roles(role: Any) -> tuple[str, ...]:
    """Teachers may declare one role or several (``"reasoning,code"`` / list)."""
    if isinstance(role, str):
        parts = [part.strip() for part in role.replace(";", ",").split(",")]
    elif isinstance(role, (list, tuple)):
        parts = [str(part).strip() for part in role]
    else:  # pragma: no cover - defensive
        parts = []
    cleaned = tuple(part for part in parts if part)
    return cleaned or ("general",)


class TeacherPool:
    """Assigns a teacher model to each request according to configured weights.

    All teachers share one provider block (endpoint, key, retry policy).
    """

    def __init__(
        self,
        config: TeachersConfig,
        *,
        client: TeacherClient | None = None,
        rng: random.Random | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.models: list[TeacherModelConfig] = config.models()
        self.client = client or TeacherClient(config.provider)
        self._rng = rng or random.Random(config.provider.name and 0)
        self._weighted: list[str] = []
        for m in self.models:
            # weight is a float; expand to integer slots with 100 slots per unit
            self._weighted.extend([m.name] * max(1, int(round(m.weight * 100))))
        self._by_name = {m.name: m for m in self.models}
        self._roles: dict[str, tuple[str, ...]] = {
            m.name: normalize_roles(m.role) for m in self.models
        }
        self._failures_in_a_row = 0

    def health_check(self) -> dict[str, Any]:
        report = {
            "provider": self.config.provider.name,
            "base_url": self.config.provider.base_url,
            "api_key_env": self.config.provider.api_key_env,
            "key_present": self.config.provider.api_key() is not None,
            "teachers": {},
        }
        for m in self.models:
            report["teachers"][m.name] = self.client.health_check(m.model)
        report["ok"] = all(v.get("ok") for v in report["teachers"].values())
        return report

    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[TeacherResponse, TeacherModelConfig]:
        """Send one request, letting the pool choose the teacher.

        ``role`` constrains the choice to teachers whose configured ``role``
        list contains that semantic role (``reasoning``, ``code``, ``general``).
        When no teacher declares the role, every teacher stays eligible and the
        fallback is logged -- the pool never invents a model.
        """
        teacher, _excluded = self.pick(role=role, exclude=set())
        if teacher is None:  # pragma: no cover - defensive
            raise TeacherConfigError("no teacher available for this request")
        resp = self.client.chat(
            teacher.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=teacher.temperature if temperature is None else temperature,
            top_p=teacher.top_p,
            max_tokens=teacher.max_tokens if max_tokens is None else max_tokens,
        )
        resp.teacher_name = teacher.name
        self._failures_in_a_row = 0
        return resp, teacher

    def ask_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[TeacherResponse, TeacherModelConfig]:
        """Try the role-preferred teacher first, then any other teacher once."""
        tried: set[str] = set()
        last_error: Exception | None = None
        for _ in range(len(self.models)):
            teacher, _ = self.pick(role=role, exclude=tried)
            if teacher is None:
                break
            tried.add(teacher.name)
            try:
                resp = self.client.chat(
                    teacher.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=teacher.temperature if temperature is None else temperature,
                    top_p=teacher.top_p,
                    max_tokens=teacher.max_tokens if max_tokens is None else max_tokens,
                )
            except (TeacherAuthError, TeacherRequestError):
                raise
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                LOGGER.warning("öğretmen %s başarısız (%s); diğeri denenecek", teacher.name, exc)
                continue
            resp.teacher_name = teacher.name
            self._failures_in_a_row = 0
            return resp, teacher
        if last_error is not None:
            raise last_error
        raise TeacherConfigError("no teacher available for this request")

    def pick(
        self, *, role: str | None = None, exclude: set[str] | None = None
    ) -> tuple[TeacherModelConfig | None, list[str]]:
        """Choose a teacher honouring ``role``; returns (teacher, excluded_names)."""
        excluded = exclude or set()
        pool = [m for m in self.models if m.name not in excluded]
        if not pool:
            return None, sorted(excluded)
        roles = getattr(self, "_roles", {})
        if role:
            matching = [m for m in pool if role in roles.get(m.name, ())]
            if matching:
                weights = [max(1, int(round(m.weight * 100))) for m in matching]
                return self._rng.choices(matching, weights=weights, k=1)[0], sorted(excluded)
        weights = [max(1, int(round(m.weight * 100))) for m in pool]
        return self._rng.choices(pool, weights=weights, k=1)[0], sorted(excluded)

    def register_failure(self) -> int:
        self._failures_in_a_row += 1
        return self._failures_in_a_row

    def reset_failures(self) -> None:
        self._failures_in_a_row = 0

    @property
    def consecutive_failures(self) -> int:
        return self._failures_in_a_row

    def _pick(self) -> TeacherModelConfig:
        teacher, _ = self.pick()
        if teacher is None:  # pragma: no cover - defensive
            raise TeacherConfigError("no teacher configured")
        return teacher
