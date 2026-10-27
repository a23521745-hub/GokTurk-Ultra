"""Teacher adapter: retries, error taxonomy, secret handling, real HTTP path."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import FakeTransport, chat_payload
from gokturk.config import ProviderConfig
from gokturk.teachers import (
    RateLimiter,
    TeacherAuthError,
    TeacherClient,
    TeacherConfigError,
    TeacherEmptyResponseError,
    TeacherPool,
    TeacherRateLimitError,
    TeacherRequestError,
    TeacherServerError,
    TeacherUnavailableError,
)


def _client(provider: ProviderConfig, responses, monkeypatch=None):
    transport = FakeTransport(responses)
    sleeps: list[float] = []
    client = TeacherClient(provider, transport=transport, sleep=lambda s: sleeps.append(s))
    return client, transport, sleeps


def test_successful_request_and_metadata(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, _ = _client(provider, [chat_payload("Merhaba dünya")])
    resp = client.chat("mimo-test", messages=[{"role": "user", "content": "selam"}])
    assert resp.text == "Merhaba dünya"
    assert resp.model == "test-model"
    assert resp.finish_reason == "stop"
    assert resp.usage["total_tokens"] == 30
    assert transport.calls[0]["url"] == "https://example.invalid/v1/chat/completions"
    meta = resp.as_metadata()
    assert "api_key" not in json.dumps(meta).lower()


def test_missing_key_sends_no_request(provider, monkeypatch):
    monkeypatch.delenv(provider.api_key_env, raising=False)
    client, transport, _ = _client(provider, [chat_payload("x")])
    with pytest.raises(TeacherConfigError, match="GTU_TEST_API_KEY"):
        client.chat("m", messages=[{"role": "user", "content": "hi"}])
    assert transport.calls == []


def test_empty_base_url_is_refused(monkeypatch):
    provider = ProviderConfig(base_url="", api_key_env="GTU_EMPTY_KEY")
    monkeypatch.setenv("GTU_EMPTY_KEY", "k")
    client, transport, _ = _client(provider, [])
    with pytest.raises(TeacherConfigError, match="base_url"):
        client.chat("m", messages=[{"role": "user", "content": "hi"}])
    assert transport.calls == []


def test_auth_error_is_fatal_and_not_retried(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, sleeps = _client(provider, [TeacherAuthError("HTTP 401")])
    with pytest.raises(TeacherAuthError):
        client.chat("m", messages=[{"role": "user", "content": "hi"}])
    assert len(transport.calls) == 1
    assert sleeps == []


def test_rate_limit_is_retried_then_succeeds(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    provider.backoff_base_s = 0.01
    client, transport, sleeps = _client(
        provider, [TeacherRateLimitError("429"), chat_payload("tamam")]
    )
    resp = client.chat("m", messages=[{"role": "user", "content": "hi"}])
    assert resp.text == "tamam"
    assert resp.attempts == 2
    assert len(transport.calls) == 2
    assert sleeps


def test_server_error_exhausts_attempts(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, _ = _client(
        provider, [TeacherServerError("500"), TeacherServerError("502"), TeacherServerError("503")]
    )
    with pytest.raises(TeacherUnavailableError):
        client.chat("m", messages=[{"role": "user", "content": "hi"}])
    assert len(transport.calls) == 3


def test_empty_content_is_not_accepted(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    payload = {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]}
    client, _, _ = _client(provider, [payload])
    with pytest.raises(TeacherEmptyResponseError):
        client.chat("m", messages=[{"role": "user", "content": "hi"}], max_attempts=1)


def test_error_object_with_http_200_raises(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    client, _, _ = _client(provider, [{"error": {"message": "quota exceeded"}}])
    with pytest.raises(TeacherServerError):
        client.chat("m", messages=[{"role": "user", "content": "hi"}], max_attempts=1)


def test_content_part_array_is_joined(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    payload = {
        "choices": [
            {
                "message": {"content": [{"type": "text", "text": "bir "}, {"text": "iki"}]},
                "finish_reason": "stop",
            }
        ]
    }
    client, _, _ = _client(provider, [payload])
    resp = client.chat("m", messages=[{"role": "user", "content": "hi"}])
    assert resp.text == "bir iki"


def test_truncated_response_flagged(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    payload = chat_payload("uzun")
    payload["choices"][0]["finish_reason"] = "length"
    client, _, _ = _client(provider, [payload])
    resp = client.chat("m", messages=[{"role": "user", "content": "hi"}])
    assert resp.is_truncated() is True
    assert resp.as_metadata()["finish_reason"] == "length"


def test_backoff_is_bounded_and_jittered(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    provider.backoff_base_s = 2.0
    provider.backoff_max_s = 10.0
    client, _, _ = _client(provider, [])
    delays = [client._backoff(i) for i in range(1, 8)]
    assert all(0 <= d <= 10.0 for d in delays)
    assert len({round(d, 3) for d in delays}) > 1  # jitter present


def test_rate_limiter_uses_injected_clock():
    now = [0.0]

    def clock():
        return now[0]

    limiter = RateLimiter(60.0, clock=clock)  # 1 request per second
    assert limiter.min_interval == pytest.approx(1.0)
    limiter.wait()


def test_rate_limiter_sleeps(monkeypatch):
    limiter = RateLimiter(60.0)
    calls = {"n": 0}

    def fake_sleep(seconds):
        calls["n"] += 1

    monkeypatch.setattr("time.sleep", fake_sleep)
    limiter.wait()
    limiter.wait()
    assert calls["n"] == 1


def test_health_check_without_credentials_sends_nothing(monkeypatch):
    provider = ProviderConfig(base_url="https://example.invalid/v1", api_key_env="GTU_NO_KEY")
    monkeypatch.delenv("GTU_NO_KEY", raising=False)
    transport = FakeTransport([])
    client = TeacherClient(provider, transport=transport)
    report = client.health_check("m")
    assert report["ok"] is False
    assert "GTU_NO_KEY" in report["detail"]
    assert transport.calls == []


def test_health_check_with_empty_base_url():
    client = TeacherClient(ProviderConfig(api_key_env="X_KEY"))
    report = client.health_check("m")
    assert report["ok"] is False
    assert "base_url" in report["detail"]


def test_pool_reports_provider_state(provider, monkeypatch, teachers_config):
    monkeypatch.setenv(provider.api_key_env, "k")
    transport = FakeTransport([chat_payload("pong"), chat_payload("pong")])
    pool = TeacherPool(teachers_config, client=TeacherClient(provider, transport=transport))
    report = pool.health_check()
    assert report["provider"] == "test"
    assert report["key_present"] is True
    assert set(report["teachers"]) == {"mimo", "glm"}
    assert report["ok"] is True


# ---------------------------------------------------------------------------
# real HTTP path (localhost server, no external network)
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    status_plan: list = []
    received: list = []
    delay_s: float = 0.0
    models_payload: dict = {"data": [{"id": "mimo-test"}, {"id": "glm-test"}]}

    def do_POST(self):  # noqa: N802
        if type(self).delay_s:
            import time as _time

            _time.sleep(type(self).delay_s)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        type(self).received.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": json.loads(body)}
        )
        status = type(self).status_plan.pop(0) if type(self).status_plan else 200
        payload = json.dumps(chat_payload("yerel yanıt")).encode()
        if status == 401:
            payload = json.dumps({"error": {"message": "invalid key"}}).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # The timeout test intentionally abandons the connection.
            pass

    def do_GET(self):  # noqa: N802
        type(self).received.append(
            {"path": self.path, "auth": self.headers.get("Authorization")}
        )
        status = type(self).status_plan.pop(0) if type(self).status_plan else 200
        if status == 401:
            payload = json.dumps({"error": {"message": "API anahtarı eksik"}}).encode()
        else:
            payload = json.dumps(type(self).models_payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence test output
        return


@pytest.fixture
def http_server():
    _Handler.status_plan = []
    _Handler.received = []
    _Handler.delay_s = 0.0
    _Handler.models_payload = {"data": [{"id": "mimo-test"}, {"id": "glm-test"}]}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", _Handler
    server.shutdown()
    server.server_close()


def test_real_http_success(http_server, monkeypatch):
    base_url, handler = http_server
    monkeypatch.setenv("GTU_HTTP_KEY", "test-key-value")
    provider = ProviderConfig(base_url=base_url, api_key_env="GTU_HTTP_KEY", timeout_s=5)
    client = TeacherClient(provider)
    resp = client.chat("m", messages=[{"role": "user", "content": "selam"}], max_tokens=16)
    assert resp.text == "yerel yanıt"
    assert handler.received[0]["path"] == "/v1/chat/completions"
    assert handler.received[0]["auth"] == "Bearer test-key-value"
    assert handler.received[0]["body"]["model"] == "m"


def test_real_http_401_is_auth_error(http_server, monkeypatch):
    base_url, handler = http_server
    handler.status_plan = [401]
    monkeypatch.setenv("GTU_HTTP_KEY", "bad")
    provider = ProviderConfig(base_url=base_url, api_key_env="GTU_HTTP_KEY", timeout_s=5)
    client = TeacherClient(provider)
    with pytest.raises(TeacherAuthError):
        client.chat("m", messages=[{"role": "user", "content": "selam"}])
    assert len(handler.received) == 1


def test_real_http_429_is_retryable(http_server, monkeypatch):
    base_url, handler = http_server
    handler.status_plan = [429, 200]
    monkeypatch.setenv("GTU_HTTP_KEY", "k")
    provider = ProviderConfig(
        base_url=base_url, api_key_env="GTU_HTTP_KEY", timeout_s=5, backoff_base_s=0.0
    )
    client = TeacherClient(provider, sleep=lambda s: None)
    resp = client.chat("m", messages=[{"role": "user", "content": "selam"}])
    assert resp.text == "yerel yanıt"
    assert resp.attempts == 2


def test_real_http_400_is_request_error(http_server, monkeypatch):
    base_url, handler = http_server
    handler.status_plan = [400]
    monkeypatch.setenv("GTU_HTTP_KEY", "k")
    provider = ProviderConfig(base_url=base_url, api_key_env="GTU_HTTP_KEY", timeout_s=5)
    client = TeacherClient(provider)
    with pytest.raises(TeacherRequestError):
        client.chat("m", messages=[{"role": "user", "content": "selam"}])


def test_real_http_timeout(http_server, monkeypatch):
    base_url, handler = http_server
    handler.delay_s = 0.5
    monkeypatch.setenv("GTU_HTTP_KEY", "k")
    provider = ProviderConfig(base_url=base_url, api_key_env="GTU_HTTP_KEY", timeout_s=0.15)
    client = TeacherClient(provider, sleep=lambda s: None)
    # Exhausting the retry budget surfaces as TeacherUnavailableError whose
    # cause is the timeout: the caller must not receive a bogus answer.
    with pytest.raises(TeacherUnavailableError) as excinfo:
        client.chat("m", messages=[{"role": "user", "content": "selam"}], max_attempts=1)
    assert "TeacherTimeoutError" in str(excinfo.value)


# ---------------------------------------------------------------------------
# GET /models -- the only supported way to learn a model id
# ---------------------------------------------------------------------------
def test_list_models_reads_provider_list(http_server, monkeypatch):
    base_url, handler = http_server
    monkeypatch.setenv("GTU_HTTP_KEY", "test-key-value")
    provider = ProviderConfig(base_url=base_url, api_key_env="GTU_HTTP_KEY", timeout_s=5)
    ids = TeacherClient(provider).list_models()
    assert ids == ["mimo-test", "glm-test"]
    assert handler.received[0]["path"] == "/v1/models"
    assert handler.received[0]["auth"] == "Bearer test-key-value"


def test_list_models_401_is_auth_error(http_server, monkeypatch):
    base_url, handler = http_server
    handler.status_plan = [401]
    monkeypatch.setenv("GTU_HTTP_KEY", "bad")
    provider = ProviderConfig(base_url=base_url, api_key_env="GTU_HTTP_KEY", timeout_s=5)
    with pytest.raises(TeacherAuthError):
        TeacherClient(provider).list_models()


def test_list_models_without_key_sends_no_request(provider, monkeypatch):
    monkeypatch.delenv(provider.api_key_env, raising=False)
    client, transport, _ = _client(provider, [])
    with pytest.raises(TeacherConfigError, match=provider.api_key_env):
        client.list_models()
    assert transport.calls == []


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"data": [{"id": "a"}, {"id": "b"}]}, ["a", "b"]),
        ({"models": ["a", "b"]}, ["a", "b"]),
        ({"data": [{"name": "a"}, {"model": "b"}]}, ["a", "b"]),
        ({"data": [{"id": "a"}, {"id": "a"}]}, ["a"]),
        ({"data": ["a", " "], "models": ["zzz"]}, ["a"]),
    ],
)
def test_extract_model_ids_shapes(payload, expected):
    from gokturk.teachers import _extract_model_ids

    assert _extract_model_ids(payload) == expected


def test_extract_model_ids_rejects_unusable_payload():
    from gokturk.teachers import _extract_model_ids

    with pytest.raises(TeacherEmptyResponseError):
        _extract_model_ids({"object": "list"})
    with pytest.raises(TeacherEmptyResponseError):
        _extract_model_ids({"data": []})


def test_pool_list_models_flags_missing_ids(teachers_config, http_server, monkeypatch):
    base_url, handler = http_server
    monkeypatch.setenv("GTU_HTTP_KEY", "k")
    teachers_config.provider.base_url = base_url
    teachers_config.provider.api_key_env = "GTU_HTTP_KEY"
    report = TeacherPool(teachers_config).list_models()
    assert report["ok"] is True
    assert report["missing"] == []
    assert report["configured"]["glm"]["present"] is True

    handler.models_payload = {"data": [{"id": "mimo-test"}]}
    report = TeacherPool(teachers_config).list_models()
    assert report["ok"] is False
    assert report["missing"] == ["glm-test"]


# ---------------------------------------------------------------------------
# 2026-10-26 Kaggle koşusu: boş yanıt, ölü uç nokta ve sessiz döngü
# ---------------------------------------------------------------------------
def _truncated_payload(model: str = "test-model") -> dict:
    """Gerçek logdaki durum: finish_reason='length', content boş, reasoning dolu."""
    return {
        "id": "chatcmpl-trunc",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "", "reasoning_content": "düşünme..."},
                "finish_reason": "length",
            }
        ],
        "usage": {"total_tokens": 4096},
    }


def test_truncated_empty_content_retries_with_bigger_budget(provider, monkeypatch):
    """Boş yanıtta aynı bütçeyle kör tekrar yok: bütçe yükseltilir."""
    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, _ = _client(
        provider, [_truncated_payload(), chat_payload("Gerçek yanıt")]
    )

    resp = client.chat("m", messages=[{"role": "user", "content": "soru"}], max_tokens=2048)

    assert resp.text == "Gerçek yanıt"
    assert resp.attempts == 2
    assert transport.calls[0]["payload"]["max_tokens"] == 2048
    assert transport.calls[1]["payload"]["max_tokens"] == 4096, "bütçe iki katına çıkmalı"


def test_truncated_budget_is_capped(provider, monkeypatch):
    from gokturk.teachers import MAX_ESCALATED_TOKENS

    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, _ = _client(
        provider,
        [_truncated_payload(), _truncated_payload(), chat_payload("olmaz")],
    )

    with pytest.raises(TeacherUnavailableError):
        client.chat(
            "m", messages=[{"role": "user", "content": "s"}], max_tokens=6000, max_attempts=1
        )

    assert [c["payload"]["max_tokens"] for c in transport.calls] == [
        6000,
        MAX_ESCALATED_TOKENS,
    ], "sınırın üstüne çıkılmamalı ve sonsuz tekrar olmamalı"


def test_reasoning_content_is_used_when_content_is_empty_but_finished(provider, monkeypatch):
    """finish_reason='stop' + boş content + dolu reasoning -> içerik kabul edilir."""
    monkeypatch.setenv(provider.api_key_env, "k")
    payload = {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": "Nihai cevap metni."},
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    client, _transport, _ = _client(provider, [payload])

    resp = client.chat("m", messages=[{"role": "user", "content": "s"}])

    assert resp.text == "Nihai cevap metni."


def test_unavailable_error_triggers_fallback_teacher(provider, monkeypatch):
    """TeacherUnavailableError retryable olmalı; yoksa diğer öğretmen hiç denenmez."""
    from gokturk.config import TeacherModelConfig, TeachersConfig
    from gokturk.teachers import RETRYABLE_ERRORS

    assert TeacherUnavailableError in RETRYABLE_ERRORS

    monkeypatch.setenv(provider.api_key_env, "k")
    from conftest import FakeTransport

    # "bad" modeli her denemede 500 verir, "good" modeli çalışır.
    class SwitchingTransport(FakeTransport):
        def __call__(self, url, payload, api_key, timeout):
            self.calls.append({"url": url, "payload": payload, "api_key": api_key, "timeout": timeout})
            if payload["model"] == "bad":
                from gokturk.teachers import TeacherServerError

                raise TeacherServerError("500")
            return chat_payload("iyi yanıt")

    config = TeachersConfig(
        provider=provider,
        teachers=[
            {"name": "bad", "model": "bad", "role": "general", "weight": 1.0},
            {"name": "good", "model": "good", "role": "general", "weight": 1.0},
        ],
    )
    config.validate()
    client = TeacherClient(config.provider, transport=SwitchingTransport([]), sleep=lambda s: None)
    pool = TeacherPool(config, client=client)
    pool.pick = lambda role=None, exclude=None: (  # type: ignore[assignment]
        next(
            (m for m in [TeacherModelConfig(name="bad", model="bad", role="general"),
                         TeacherModelConfig(name="good", model="good", role="general")]
             if m.name not in (exclude or set())),
            None,
        ),
        sorted(exclude or set()),
    )

    resp, teacher = pool.ask_with_fallback("sys", "user")

    assert resp.text == "iyi yanıt"
    assert teacher.name == "good", "ilk öğretmen tükendiğinde diğeri denenmeli"


# ---------------------------------------------------------------------------
# 3b olayı: DNS hatası "timeout" değildir, tek denemede pes edilmez
# ---------------------------------------------------------------------------
def _raise_dns_on_urlopen(monkeypatch):
    """Gerçek taşıma yolunu (urllib) DNS hatası verecek şekilde taklit et."""
    import socket
    import urllib.error
    import urllib.request

    dns = urllib.error.URLError(socket.gaierror(-3, "Temporary failure in name resolution"))

    def fake_urlopen(req, timeout=None):
        raise dns

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_dns_error_is_classified_separately(provider, monkeypatch):
    """gaierror -> TeacherDnsError; teşhis "DNS" der, "zaman aşımı" demez."""
    from gokturk.teachers import RETRYABLE_ERRORS, TeacherClient, TeacherDnsError, describe_error

    monkeypatch.setenv(provider.api_key_env, "k")
    _raise_dns_on_urlopen(monkeypatch)
    client = TeacherClient(provider, sleep=lambda s: None)

    with pytest.raises(TeacherUnavailableError) as excinfo:
        client.chat("m", messages=[{"role": "user", "content": "s"}], max_attempts=1)

    # Denemeler tükendiğinde sarmalayıcı hata çıkar, ama nedeni DNS olmalı.
    assert isinstance(excinfo.value.__cause__, TeacherDnsError)
    diagnosis = describe_error(excinfo.value)
    assert diagnosis.startswith("DNS:")
    assert "anahtarı veya model kimliği sorunu DEĞİL" in diagnosis
    assert "zaman aşımı" not in diagnosis.casefold()
    # Retryable kalmalı (TeacherTimeoutError alt sınıfı): aksi hâlde tek sıçrama
    # koşuyu düşürür. Tuple üyeliği kimlik kontrolü olduğu için issubclass kullanılır.
    assert any(issubclass(TeacherDnsError, err) for err in RETRYABLE_ERRORS)


def test_describe_error_covers_other_kinds(provider):
    from gokturk.teachers import (
        TeacherAuthError,
        TeacherConfigError,
        TeacherRateLimitError,
        TeacherServerError,
        describe_error,
    )

    assert describe_error(TeacherAuthError("401")).startswith("401/403")
    assert describe_error(TeacherConfigError("x")).startswith("Yapılandırma")
    assert describe_error(TeacherRateLimitError("429")).startswith("429")
    assert describe_error(TeacherServerError("502")).startswith("5xx")
    assert describe_error(ValueError("?")) == "ValueError: ?"


def test_is_dns_error_markers():
    import socket

    from gokturk.teachers import _is_dns_error

    assert _is_dns_error(socket.gaierror(-3, "Temporary failure in name resolution"))
    assert _is_dns_error(OSError("[Errno -5] No address associated with hostname"))
    assert not _is_dns_error(TimeoutError("timed out"))
    assert not _is_dns_error(None)


class FakeGetTransport:
    """GET taşıması: imza (url, api_key, timeout)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, api_key, timeout):
        self.calls.append({"url": url, "timeout": timeout})
        if not self.responses:
            raise AssertionError("FakeGetTransport exhausted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _dns_error():
    from gokturk.teachers import TeacherDnsError

    return TeacherDnsError("DNS resolution failed: Temporary failure in name resolution")


def test_list_models_retries_transient_dns_failure(provider, monkeypatch):
    """Tek DNS sıçraması 3b hücresini düşürmemeli."""
    monkeypatch.setenv(provider.api_key_env, "k")
    models_payload = {"data": [{"id": "mimo-v2.6-pro"}, {"id": "glm-5.3"}]}
    get = FakeGetTransport([_dns_error(), models_payload])
    client = TeacherClient(provider, transport=FakeTransport([]), transport_get=get,
                           sleep=lambda s: None)

    ids = client.list_models(max_attempts=2)

    assert ids == ["mimo-v2.6-pro", "glm-5.3"]
    assert len(get.calls) == 2


def test_list_models_raises_after_retries(provider, monkeypatch):
    from gokturk.teachers import TeacherDnsError, TeacherUnavailableError

    monkeypatch.setenv(provider.api_key_env, "k")
    get = FakeGetTransport([_dns_error(), _dns_error(), _dns_error()])
    client = TeacherClient(provider, transport=FakeTransport([]), transport_get=get,
                           sleep=lambda s: None)

    with pytest.raises(TeacherUnavailableError) as excinfo:
        client.list_models(max_attempts=3)

    assert isinstance(excinfo.value.__cause__, TeacherDnsError)
    assert len(get.calls) == 3, "tek deneme yapmamalı"


def test_list_models_does_not_retry_auth_errors(provider, monkeypatch):
    from gokturk.teachers import TeacherAuthError

    monkeypatch.setenv(provider.api_key_env, "k")
    get = FakeGetTransport([TeacherAuthError("401")])
    client = TeacherClient(provider, transport=FakeTransport([]), transport_get=get,
                           sleep=lambda s: None)

    with pytest.raises(TeacherAuthError):
        client.list_models(max_attempts=3)
    assert len(get.calls) == 1, "401 tekrar denenmez"


# ---------------------------------------------------------------------------
# diagnose_endpoint — ağ olmadan test edilebilir
# ---------------------------------------------------------------------------
def _diag(provider, **kwargs):
    from gokturk.teachers import diagnose_endpoint

    return diagnose_endpoint(provider, sleep=lambda s: None, **kwargs)


def test_diagnose_endpoint_dns_failure_is_reported_as_dns(provider):
    def boom(host):
        raise OSError("[Errno -3] Temporary failure in name resolution")

    result = _diag(provider, resolver=boom)

    assert result.dns_ok is False
    assert result.dns_failures == 3
    assert "name resolution" in (result.dns_error or "")
    text = result.render()
    assert "BAŞARISIZ" in text
    assert "API anahtarı sorunu DEĞİL" in text, "yanlış teşhis bir daha yazılmamalı"
    assert "TCP  : denenmedi" in text


def test_diagnose_endpoint_reports_intermittent_dns(provider):
    calls = {"n": 0}

    def flaky(host):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("Temporary failure in name resolution")
        return ["195.142.26.68"]

    result = _diag(provider, resolver=flaky, connect=lambda *a, **k: object())

    assert result.dns_ok is True
    assert result.dns_intermittent is True
    assert "ARALIKLI" in result.render()


def test_diagnose_endpoint_tcp_failure(provider):
    def refuse(address, **kwargs):
        raise ConnectionRefusedError("refused")

    result = _diag(provider, resolver=lambda h: ["195.142.26.68"], connect=refuse)

    assert result.dns_ok and not result.tcp_ok
    assert "TCP  : BAŞARISIZ" in result.render()


def test_diagnose_endpoint_missing_key_is_the_only_key_complaint(provider, monkeypatch):
    monkeypatch.delenv(provider.api_key_env, raising=False)

    result = _diag(provider, resolver=lambda h: ["195.142.26.68"], connect=lambda *a, **k: object())
    text = result.render()

    assert result.key_present is False
    assert result.ok is False
    assert "eksik olan EVREN_API_KEY" in text


def test_diagnose_endpoint_healthy(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")

    result = _diag(provider, resolver=lambda h: ["195.142.26.68"], connect=lambda *a, **k: object())

    assert result.ok is True
    assert "ÇÖZÜLDÜ" in result.render()


def test_diagnose_endpoint_without_base_url(provider):
    from gokturk.config import ProviderConfig

    empty = ProviderConfig(base_url="")
    result = _diag(empty)

    assert result.dns_ok is False
    assert "base_url" in (result.dns_error or "")


# ---------------------------------------------------------------------------
# 2026-10-26 (ikinci koşu): sağlık denetimi token fırtınası
#   "boş içerik (finish_reason=length), max_tokens 8 -> 16 ... 64 -> 128"
# ---------------------------------------------------------------------------
def test_health_check_accepts_a_truncated_ping(provider, monkeypatch):
    """Reasoning modeli ping bütçesini düşünmede harcarsa uç nokta SAĞLIKLIDIR."""
    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, _ = _client(provider, [_truncated_payload(), _truncated_payload()])

    report = client.health_check("mimo-test")

    assert report["ok"] is True, "finish_reason='length' bir sağlık arızası değil"
    assert len(transport.calls) == 1, "ping tek istek atmalı, yükseltme yapmamalı"


def test_health_check_uses_a_budget_that_reasoning_models_can_survive(provider, monkeypatch):
    from gokturk.teachers import HEALTH_CHECK_MAX_TOKENS

    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, _ = _client(provider, [chat_payload("pong")])
    client.health_check("mimo-test")

    assert transport.calls[0]["payload"]["max_tokens"] == HEALTH_CHECK_MAX_TOKENS
    assert HEALTH_CHECK_MAX_TOKENS >= 32, "8 token reasoning için anlamsızdı"


def test_truncation_escalates_at_most_once(provider, monkeypatch):
    """8->16->32->64->... tırmanışı olmamalı: en fazla iki istek."""
    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, _ = _client(
        provider, [_truncated_payload(), _truncated_payload(), _truncated_payload()]
    )

    with pytest.raises(TeacherUnavailableError):
        client.chat("m", messages=[{"role": "user", "content": "s"}], max_tokens=8, max_attempts=1)

    assert [c["payload"]["max_tokens"] for c in transport.calls] == [8, 16], (
        "bütçe bir kez yükseltilir, sonra vazgeçilir"
    )


def test_escalation_can_be_disabled(provider, monkeypatch):
    """Ölçüm hücresi hangi bütçenin yettiğini görmek için yükseltmeyi kapatır."""
    from gokturk.teachers import TeacherTruncatedResponseError

    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, _ = _client(provider, [_truncated_payload()])

    with pytest.raises(TeacherTruncatedResponseError):
        client.chat(
            "m",
            messages=[{"role": "user", "content": "s"}],
            max_tokens=128,
            max_attempts=1,
            escalate_truncation=False,
        )

    assert len(transport.calls) == 1


def test_incident_doc_covers_the_health_check_storm():
    from pathlib import Path

    body = (Path(__file__).resolve().parents[1] / "docs" / "uretim-olay-2026-10-26.md").read_text(
        encoding="utf-8"
    )
    for needle in ("2.5 Sağlık denetimi", "HEALTH_CHECK_MAX_TOKENS", "escalate_truncation"):
        assert needle in body, f"olay raporunda eksik: {needle}"


def test_truncation_does_not_spend_remaining_transport_retries(provider, monkeypatch):
    monkeypatch.setenv(provider.api_key_env, "k")
    client, transport, _ = _client(provider, [_truncated_payload()] * 6)
    with pytest.raises(TeacherUnavailableError):
        client.chat("m", messages=[{"role": "user", "content": "s"}],
                    max_tokens=8, max_attempts=5)
    assert [c["payload"]["max_tokens"] for c in transport.calls] == [8, 16]


@pytest.mark.parametrize("role", ["general", "reasoning", "code"])
def test_v1_repo_pool_routes_every_role_to_mimo(role, monkeypatch):
    from pathlib import Path

    from gokturk.config import load_teachers_config

    config = load_teachers_config(Path(__file__).resolve().parents[1] / "configs/teachers.yaml")
    monkeypatch.setenv(config.provider.api_key_env, "test-key")
    client, transport, _ = _client(config.provider, [chat_payload("Türkçe yanıt")])
    pool = TeacherPool(config, client=client)
    _, teacher = pool.ask_with_fallback("sistem", "istek", role=role)
    assert teacher.name == "mimo-v26-pro"
    assert len(transport.calls) == 1
    assert transport.calls[0]["payload"]["model"] == "mimo-v2.6-pro"


def test_v1_failed_mimo_does_not_call_another_model(monkeypatch):
    from pathlib import Path

    from gokturk.config import load_teachers_config

    config = load_teachers_config(Path(__file__).resolve().parents[1] / "configs/teachers.yaml")
    monkeypatch.setenv(config.provider.api_key_env, "test-key")
    client, transport, _ = _client(config.provider, [TeacherServerError("capacity")] * 8)
    pool = TeacherPool(config, client=client)
    with pytest.raises(TeacherUnavailableError):
        pool.ask_with_fallback("sistem", "istek", role="code")
    assert transport.calls
    assert {c["payload"]["model"] for c in transport.calls} == {"mimo-v2.6-pro"}


def test_diagnosis_checks_all_dns_samples_even_after_success(provider):
    from gokturk.teachers import diagnose_endpoint

    calls = []

    def resolver(host):
        calls.append(host)
        if len(calls) == 2:
            raise OSError("intermittent DNS")
        return ["192.0.2.1"]

    result = diagnose_endpoint(provider, resolver=resolver,
                               connect=lambda *a, **k: object(), attempts=3, sleep=lambda _: None)
    assert len(calls) == 3
    assert result.dns_attempts == 3
    assert result.dns_failures == 1
    assert result.dns_intermittent
