"""Teacher-model API adapter.

Contract
--------
* Credentials come only from an environment variable named in configuration
  (``provider.api_key_env``) or from Kaggle Secrets placed into the environment.
  The key is never written to a file, a log line, or a dataset record.
* Only an OpenAI-compatible chat-completions wire format is implemented
  (``POST {base_url}/chat/completions``). EVREN's public documentation
  describes an OpenAI-compatible inference layer, but the concrete model ids
  and base URL for GLM / MiMo must be verified by the operator against the
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
import socket
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

#: Reasoning modelleri bütçeyi düşünme zincirinde harcayıp boş ``content``
#: döndürebilir; bu durumda bir kez bu sınıra kadar yükseltilir.
MAX_ESCALATED_TOKENS = 8192

#: Sağlık denetiminin ping bütçesi. "Uç nokta ayakta mı" sorusu için yeterli;
#: reasoning modeli bunu da düşünmede harcarsa bu bir HATA değildir (bkz.
#: :meth:`TeacherClient.health_check`).
HEALTH_CHECK_MAX_TOKENS = 64


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


class TeacherTruncatedResponseError(TeacherEmptyResponseError):
    """The model stopped at the token limit before writing any answer.

    ``finish_reason="length"`` with an empty ``content`` means the whole budget
    was spent inside the reasoning trace. Retrying with the *same* budget is
    deterministic waste: 2026-10-26 Kaggle run burned 118 such retries and
    ~14 minutes per slot. The client therefore retries **once** with a larger
    ``max_tokens`` instead (see :meth:`TeacherClient.chat`).
    """

    def __init__(self, message: str, *, finish_reason: str | None = None) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason


class TeacherDnsError(TeacherTimeoutError):
    """The host could not be resolved (``gaierror``).

    Subclass of :class:`TeacherTimeoutError` on purpose: it stays retryable, but
    the *diagnosis* is different. A resolver hiccup and a slow model need
    different actions, and until 2026-10-26 both were reported as
    "timeout ... connection error: [Errno -3] Temporary failure in name
    resolution", which sent the reader looking for the wrong cause (and the
    notebook even blamed a missing API key).
    """


class TeacherUnavailableError(TeacherError):
    """Every retry and every fallback teacher failed for one request."""


#: ``TeacherUnavailableError`` belongs here: without it a failed request never
#: triggers the fallback teacher and never increments the consecutive-failure
#: counter, so a dead endpoint looks like "slow progress" forever instead of
#: aborting (exactly what happened on 2026-10-26).
RETRYABLE_ERRORS = (
    TeacherRateLimitError,
    TeacherServerError,
    TeacherTimeoutError,
    TeacherEmptyResponseError,
    TeacherUnavailableError,
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
        transport_get=None,
        sleep=time.sleep,
        jitter: random.Random | None = None,
    ) -> None:
        self.provider = provider
        self._transport = transport or self._http_post_json
        self._get_transport = transport_get or self._http_get_json
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
                # 2026-10-26: burada 8 token vardı. Reasoning modelleri bütün
                # bütçeyi düşünme zincirinde harcayıp boş `content` döndürdü,
                # sağlık denetimi de her seferinde "başarısız" dedi.
                max_tokens=HEALTH_CHECK_MAX_TOKENS,
                temperature=0.0,
                max_attempts=1,
                escalate_truncation=False,
            )
        except TeacherTruncatedResponseError:
            # Uç nokta YANIT VERDİ: HTTP 200, geçerli `choices`, model adı doğru.
            # Yalnızca ping bütçesi düşünme zincirine yetmedi — bu bir sağlık
            # sorunu değil, reasoning modelinin normal davranışı.
            return {
                "ok": True,
                "detail": (
                    f"yanıt verdi (ping bütçesi {HEALTH_CHECK_MAX_TOKENS} token; "
                    "finish_reason='length', reasoning modeli olabilir)"
                ),
            }
        except TeacherError as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": bool(resp.text.strip()),
            "detail": f"model={resp.model} latency={resp.latency_s:.2f}s",
        }

    def list_models(
        self, *, timeout: float | None = None, max_attempts: int | None = None
    ) -> list[str]:
        """Return the model ids the provider itself advertises (``GET /models``).

        This is the only supported way to learn a model id: the caller reads
        what the server reports instead of trusting a hard-coded guess. Raises
        :class:`TeacherConfigError` when the configuration or the key is
        missing, and the usual HTTP taxonomy otherwise.

        Retries like :meth:`chat` does: one DNS hiccup must not fail the whole
        cell. On 2026-10-26 a single resolution failure made the notebook print
        "model list could not be fetched" and then blame the API key.
        """
        provider = self.provider
        if not provider.base_url:
            raise TeacherConfigError(
                "provider.base_url is empty: cannot list models without an endpoint."
            )
        try:
            api_key = provider.require_api_key()
        except ConfigError as exc:
            raise TeacherConfigError(str(exc)) from exc
        url = provider.base_url.rstrip("/") + "/models"
        attempts_allowed = max_attempts or max(2, provider.max_attempts_per_sample)
        last_error: TeacherError | None = None
        for attempt in range(1, attempts_allowed + 1):
            self._limiter.wait()
            try:
                raw = self._get_transport(url, api_key, timeout or provider.timeout_s)
            except (TeacherAuthError, TeacherConfigError, TeacherRequestError):
                raise
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                LOGGER.warning(
                    "model listesi alınamadı (deneme %d/%d): %s: %s",
                    attempt,
                    attempts_allowed,
                    type(exc).__name__,
                    exc,
                )
                if attempt < attempts_allowed:
                    self._sleep(self._backoff(attempt))
                continue
            return _extract_model_ids(raw)
        raise TeacherUnavailableError(
            f"model listesi {attempt} denemede alınamadı; son hata: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

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
        escalate_truncation: bool = True,
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
        current_max_tokens = int(max_tokens)
        # Deneme bütçesi normalde ``attempts_allowed``. Yalnızca yanıt boş dönerse
        # (reasoning token bütçesini yedi) TEK bir ek deneme verilir ve o deneme
        # daha büyük ``max_tokens`` ile yapılır: aynı parametrelerle tekrar denemek
        # deterministik olarak aynı sonucu verir.
        max_attempts_total = attempts_allowed
        attempt = 0
        # Bütçe yükseltmesi EN FAZLA BİR KEZ yapılır. 2026-10-26 logunda ping
        # 8 token'la atıldığı için 8->16->32->64->128->... diye 11 denemeye
        # tırmandı; "tek kademeli" tasarım küçük bütçelerde kontrolden çıktı.
        escalated = False

        while attempt < max_attempts_total:
            attempt += 1
            # Yeni dict: transport kayıtları sonradan mutasyonla bozulmasın.
            request_payload = {**payload, "max_tokens": current_max_tokens}
            self._limiter.wait()
            started = time.monotonic()
            try:
                raw = self._transport(url, request_payload, api_key, provider.timeout_s)
            except TeacherAuthError:
                raise
            except TeacherRequestError:
                raise
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                LOGGER.warning(
                    "teacher request failed (attempt %d/%d, model=%s): %s: %s",
                    attempt,
                    max_attempts_total,
                    model,
                    type(exc).__name__,
                    exc,
                )
                if attempt < attempts_allowed:
                    self._sleep(self._backoff(attempt))
                continue

            latency = time.monotonic() - started
            try:
                text, usage, finish_reason = self._extract_content(raw, model)
            except TeacherTruncatedResponseError as exc:
                last_error = exc
                bigger = min(MAX_ESCALATED_TOKENS, current_max_tokens * 2)
                if not escalate_truncation:
                    # Yükseltme kapalı: bütçe kesilmesi deterministiktir, aynı
                    # isteği tekrarlamak sadece zaman yakar. Çağırana söyle.
                    raise
                if not escalated and bigger > current_max_tokens:
                    LOGGER.warning(
                        "model %s: boş içerik (finish_reason=length), max_tokens "
                        "%d -> %d ile bir kez daha denenecek",
                        model,
                        current_max_tokens,
                        bigger,
                    )
                    current_max_tokens = bigger
                    max_attempts_total += 1
                    escalated = True
                elif bigger > current_max_tokens:
                    LOGGER.warning(
                        "model %s: bütçe bir kez yükseltildi, ikinci kez "
                        "yükseltilmiyor (finish_reason=length)",
                        model,
                    )
                    break
                else:
                    LOGGER.warning(
                        "model %s: max_tokens %d bile yetmedi (finish_reason=length)",
                        model,
                        current_max_tokens,
                    )
                    break
                continue
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
            f"model {model!r} failed after {attempt} attempts; "
            f"last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

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
        except socket.gaierror as exc:
            raise TeacherDnsError(f"DNS resolution failed: {exc}") from exc
        except urllib.error.URLError as exc:
            raise _classify_connection_error(exc, timeout) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise TeacherEmptyResponseError(
                f"response was not valid JSON ({len(body)} bytes)"
            ) from exc
        if not isinstance(parsed, dict):
            raise TeacherEmptyResponseError("response JSON is not an object")
        return parsed

    def _http_get_json(
        self, url: str, api_key: str, timeout: float
    ) -> dict[str, Any]:
        req = urllib.request.Request(url, method="GET")
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
        except socket.gaierror as exc:
            raise TeacherDnsError(f"DNS resolution failed: {exc}") from exc
        except urllib.error.URLError as exc:
            raise _classify_connection_error(exc, timeout) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise TeacherEmptyResponseError(
                f"/models response was not valid JSON ({len(body)} bytes)"
            ) from exc
        return parsed if isinstance(parsed, dict) else {"data": parsed}

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
        finish_reason = first.get("finish_reason")
        reasoning = ""
        if isinstance(message, Mapping):
            # Sağlayıcılar düşünme zincirini ayrı alanda döndürebilir.
            for key in ("reasoning_content", "reasoning"):
                value = message.get(key)
                if isinstance(value, str) and value.strip():
                    reasoning = value.strip()
                    break

        if not isinstance(content, str) or not content.strip():
            if finish_reason == "length":
                # Bütçe düşünme zincirinde bitti: yanıt HİÇ yazılmadı. Aynı
                # parametrelerle tekrar denemek zaman kaybıdır (2026-10-26'da
                # 118 kez denendi); çağıran daha büyük bütçeyle bir kez dener.
                raise TeacherTruncatedResponseError(
                    f"empty content for model {model!r} (finish_reason='length', "
                    f"reasoning={'var' if reasoning else 'yok'})",
                    finish_reason="length",
                )
            if reasoning:
                LOGGER.warning(
                    "model %s: 'content' boş, 'reasoning' alanı içerik olarak kullanılıyor "
                    "(finish_reason=%r)",
                    model,
                    finish_reason,
                )
                content = reasoning
            else:
                raise TeacherEmptyResponseError(
                    f"empty content for model {model!r} (finish_reason="
                    f"{first.get('finish_reason')!r})"
                )
        usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
        return content.strip(), dict(usage), (
            finish_reason if isinstance(finish_reason, str) else None
        )


_DNS_MARKERS = (
    "name resolution",
    "name or service not known",
    "nodename nor servname",
    "no address associated with hostname",
    "temporary failure in name resolution",
    "getaddrinfo failed",
)


def _is_dns_error(exc: BaseException | None) -> bool:
    """True when *exc* (or its ``reason``) is a DNS resolution failure."""
    if exc is None:
        return False
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            return True
        text = str(current).casefold()
        if any(marker in text for marker in _DNS_MARKERS):
            return True
        nxt = getattr(current, "reason", None)
        current = nxt if isinstance(nxt, BaseException) else None
    return False


#: Türkçe, eyleme dönük teşhis metinleri: hata tipini doğru adlandırmak,
#: yanlış yeri kurcalamayı önler (2026-10-26: DNS hatası "anahtar yok" sanıldı).
def describe_error(exc: BaseException | None) -> str:
    """One-line diagnosis for a teacher failure, walking the ``__cause__`` chain."""
    if exc is None:
        return "bilinmeyen hata"
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        nxt = current.__cause__ or current.__context__
        current = nxt if isinstance(nxt, BaseException) else None
    kinds = {type(item) for item in chain}

    if TeacherDnsError in kinds or any(_is_dns_error(item) for item in chain):
        return (
            "DNS: ad çözümlenemedi (resolver). API anahtarı veya model kimliği sorunu DEĞİL. "
            "Aynı isteği tekrar deneyin; sık tekrarlarsa oturumu yeniden başlatın."
        )
    if TeacherAuthError in kinds:
        return "401/403: anahtar geçersiz veya süresi geçmiş. Secrets'taki değeri yenileyin."
    if TeacherConfigError in kinds:
        return "Yapılandırma eksik (base_url/anahtar değişkeni). configs/teachers.yaml."
    if TeacherRateLimitError in kinds:
        return "429: oran sınırı. requests_per_minute düşürün veya bekleyin."
    if TeacherServerError in kinds:
        return "5xx: sağlayıcı tarafındaki hata. Birkaç dakika sonra tekrar deneyin."
    if TeacherTruncatedResponseError in kinds:
        return (
            "Yanıt token bütçesinde kesildi (finish_reason=length): max_tokens yetersiz."
            " configs/teachers.yaml içinde yükseltin."
        )
    if TeacherEmptyResponseError in kinds:
        return "Boş yanıt: sağlayıcı 200 döndü ama içerik yok. Tekrar deneyin."
    if TeacherTimeoutError in kinds:
        return "Zaman aşımı: istek süresi doldu. timeout_s değerini ölçerek artırın."
    return f"{type(exc).__name__}: {exc}"


@dataclass
class EndpointDiagnosis:
    """Result of :func:`diagnose_endpoint` -- facts, not guesses."""

    host: str
    port: int
    addresses: list[str] = field(default_factory=list)
    dns_attempts: int = 0
    dns_failures: int = 0
    dns_error: str | None = None
    tcp_ok: bool = False
    tcp_error: str | None = None
    key_present: bool = False

    @property
    def dns_ok(self) -> bool:
        return bool(self.addresses)

    @property
    def dns_intermittent(self) -> bool:
        """True when some resolutions worked and some failed."""
        return bool(self.addresses) and self.dns_failures > 0

    @property
    def ok(self) -> bool:
        return self.dns_ok and self.tcp_ok and self.key_present

    def render(self) -> str:
        lines = [f"Uç nokta tanısı: {self.host}:{self.port}"]
        if self.dns_ok:
            lines.append(f"  DNS  : ÇÖZÜLDÜ -> {', '.join(self.addresses)}")
        else:
            lines.append(f"  DNS  : BAŞARISIZ ({self.dns_error or 'bilinmeyen'})")
        if self.dns_attempts:
            lines.append(
                f"  DNS  : {self.dns_attempts} denemede {self.dns_failures} başarısız"
                + ("  <-- ARALIKLI (resolver kararsız)" if self.dns_intermittent else "")
            )
        if not self.dns_ok:
            lines.append("  TCP  : denenmedi (ad çözümlenemedi)")
        elif self.tcp_ok:
            lines.append("  TCP  : AÇIK")
        else:
            lines.append(f"  TCP  : BAŞARISIZ ({self.tcp_error})")
        lines.append(f"  ANAHTAR: {'VAR' if self.key_present else 'YOK'}")
        if not self.dns_ok:
            lines.append(
                "  YORUM: Bu bir API anahtarı sorunu DEĞİL. Ad çözümlemesi (DNS) düşüyor;"
                " anahtarı kontrol etmek zaman kaybıdır."
            )
        elif self.dns_intermittent:
            lines.append(
                "  YORUM: DNS ARALIKLI bozuk: bazı sorgular çözüldü, bazıları düşmedi."
                " Retry'lar tek bir sıçramayı atlatır; sık tekrarlarsa oturumu yeniden"
                " başlatın (bozulma Kaggle resolver'ına özgü olabiliyor)."
            )
        elif not self.tcp_ok:
            lines.append(
                "  YORUM: Ad çözümlendi ama TCP açılmıyor: güvenlik duvarı/port veya"
                " uç nokta kapalı. Internet açık mı?"
            )
        elif not self.key_present:
            lines.append(
                "  YORUM: Ağ sağlam; eksik olan EVREN_API_KEY"
                " (Add-ons > Secrets > Attach)."
            )
        return "\n".join(lines)


def _default_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def diagnose_endpoint(
    provider,
    *,
    resolver=None,
    connect=None,
    attempts: int = 3,
    timeout: float = 8.0,
    sleep=None,
    pause_s: float = 0.5,
) -> EndpointDiagnosis:
    """Check DNS, TCP and key presence *before* spending hours on generation.

    2026-10-26: the run spent 5,7 hours and produced nothing, while the very first
    minute already showed "Temporary failure in name resolution". This function
    turns that signal into an explicit, actionable diagnosis.

    All I/O is injectable (``resolver``/``connect``/``sleep``) so it is testable
    without a network.
    """
    url = str(getattr(provider, "base_url", "") or "")
    host = ""
    if "://" in url:
        host = url.split("://", 1)[1].split("/", 1)[0].split(":", 1)[0]
    port = 443
    diagnosis = EndpointDiagnosis(
        host=host,
        port=port,
        key_present=bool(getattr(provider, "api_key", lambda: None)()),
    )
    if not host:
        diagnosis.dns_error = "provider.base_url boş"
        return diagnosis

    resolve = resolver or _default_resolver
    wait = sleep or time.sleep
    for attempt in range(1, max(1, attempts) + 1):
        diagnosis.dns_attempts = attempt
        try:
            found = [str(a) for a in resolve(host) if a]
        except Exception as exc:  # gaierror, hermetik test hataları, ...
            diagnosis.dns_failures += 1
            diagnosis.dns_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                wait(pause_s)
            continue
        if not found:
            diagnosis.dns_failures += 1
            diagnosis.dns_error = "boş yanıt (ad yok)"
            if attempt < attempts:
                wait(pause_s)
            continue
        diagnosis.addresses = found
        diagnosis.dns_error = None
        if attempt < attempts:
            wait(pause_s)

    if not diagnosis.dns_ok:
        return diagnosis

    opener = connect or (lambda address, **kw: socket.create_connection(address, **kw))
    try:
        sock = opener((diagnosis.addresses[0], port), timeout=timeout)
        closer = getattr(sock, "close", None)
        if callable(closer):
            closer()
        diagnosis.tcp_ok = True
    except Exception as exc:
        diagnosis.tcp_error = f"{type(exc).__name__}: {exc}"
    return diagnosis


def _classify_connection_error(exc: urllib.error.URLError, timeout: float) -> TeacherError:
    """Map a URLError onto the right error type instead of calling everything a timeout."""
    if _is_dns_error(exc):
        return TeacherDnsError(
            f"DNS resolution failed for {getattr(exc, 'filename', '') or 'the endpoint'}: "
            f"{exc.reason}. Bu bir zaman aşımı DEĞİL: ad çözümlenemedi (resolver/Kaggle ağı)."
        )
    return TeacherTimeoutError(f"connection error: {exc.reason}")


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


def _extract_model_ids(raw: Mapping[str, Any]) -> list[str]:
    """Pull model ids out of an OpenAI-style ``/models`` payload.

    Accepts ``{"data": [{"id": ...}]}`` (the documented shape), a bare list of
    strings, and the ``{"models": [...]}`` variant some gateways return. Ids
    are returned in server order with duplicates removed.
    """
    items = raw.get("data")
    if items is None:
        items = raw.get("models")
    if not isinstance(items, list):
        raise TeacherEmptyResponseError(
            "unexpected /models payload: expected a 'data' or 'models' list"
        )
    ids: list[str] = []
    for item in items:
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, Mapping):
            candidate = item.get("id") or item.get("name") or item.get("model")
        else:
            candidate = None
        if isinstance(candidate, str) and candidate.strip():
            name = candidate.strip()
            if name not in ids:
                ids.append(name)
    if not ids:
        raise TeacherEmptyResponseError("/models returned no usable model ids")
    return ids


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

    def list_models(self) -> dict[str, Any]:
        """Compare the configured teacher ids with the provider's own list.

        Returns ``available`` (server order), per-teacher ``configured`` /
        ``present`` flags and ``missing`` ids. One request, no generation.
        """
        available = self.client.list_models()
        configured = {
            m.name: {"model": m.model, "present": m.model in available}
            for m in self.models
        }
        missing = sorted(
            {entry["model"] for entry in configured.values() if not entry["present"]}
        )
        return {
            "provider": self.config.provider.name,
            "base_url": self.config.provider.base_url,
            "available": available,
            "configured": configured,
            "missing": missing,
            "ok": not missing,
        }

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
                LOGGER.warning("öğretmen %s başarısız (%s); kalan öğretmen sayısı: %d",
                               teacher.name, exc, len(self.models) - len(tried))
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
