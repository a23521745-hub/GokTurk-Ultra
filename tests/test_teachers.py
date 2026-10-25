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
    assert set(report["teachers"]) == {"mimo", "kimi"}
    assert report["ok"] is True


# ---------------------------------------------------------------------------
# real HTTP path (localhost server, no external network)
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    status_plan: list = []
    received: list = []
    delay_s: float = 0.0

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

    def log_message(self, *args):  # silence test output
        return


@pytest.fixture
def http_server():
    _Handler.status_plan = []
    _Handler.received = []
    _Handler.delay_s = 0.0
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
