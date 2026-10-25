"""Read a training log (or a run manifest) and say what it means, in plain Turkish.

Motivation
----------
HuggingFace ``Trainer`` progress bars are dense:

    42%|████▏     | 25/60 [04:12<05:52, 10.08s/it]
    {'loss': 1.8342, 'grad_norm': 0.612, 'learning_rate': 4.8e-05, 'epoch': 0.12}

A non-specialist cannot tell a healthy run from a diverging one by looking at
that. This module turns such text into:

* a **verdict** (``saglikli`` / ``dikkat`` / ``bozuk`` / ``basarisiz``),
* concrete **findings** with evidence lines,
* and an **action** list in Turkish.

What it does *not* do
---------------------
It judges **mechanics only** (did it crash? is the loss finite? is it
diverging? did it stop early?). It says nothing about model quality: a run can
be perfectly healthy and still produce a useless adapter. Quality needs
:mod:`gokturk.evaluate` and human review.

Usage
-----
    python -m gokturk.logcheck --log train.log
    python -m gokturk.logcheck --manifest outputs/run/run_manifest.json
    pbpaste | python -m gokturk.logcheck --stdin
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
OK = "saglikli"
WARN = "dikkat"
BROKEN = "bozuk"
FAILED = "basarisiz"

SEVERITY_ORDER = {OK: 0, WARN: 1, BROKEN: 2, FAILED: 3}

_TITLES = {
    OK: "SAĞLIKLI — eğitim mekanik olarak düzgün görünüyor",
    WARN: "DİKKAT — çalışıyor ama izlenmesi gereken bir şey var",
    BROKEN: "BOZUK — kayıp/ıraksama sorunu var, koşu güvenilir değil",
    FAILED: "BAŞARISIZ — koşu tamamlanmadı",
}


@dataclass
class Finding:
    code: str
    severity: str
    message: str
    evidence: list[str] = field(default_factory=list)
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence,
            "action": self.action,
        }


# ---------------------------------------------------------------------------
# error dictionary: ordered (pattern, code, severity, message, action)
# ---------------------------------------------------------------------------
_ERROR_RULES: list[tuple[str, str, str, str, str]] = [
    (
        r"torch\.OutOfMemoryError|CUDA out of memory|CUDA error: out of memory",
        "cuda-oom",
        FAILED,
        "GPU belleği yetmedi (CUDA out of memory).",
        "max_seq_len'i 512'ye düşürün, per_device_train_batch_size=1 yapın, "
        "gradient_accumulation_steps'i artırın. Unsloth'ta 4-bit açık mı kontrol edin.",
    ),
    (
        r"^\s*Killed\s*$|(?:system|host) (?:RAM|memory) (?:exhausted|exceeded)|"
        r"(?<![A-Za-z])MemoryError(?::|$)",
        "ram-oom",
        FAILED,
        "Sistem RAM'i tükendi (Kaggle 29 GB). Süreç öldürüldü.",
        "Veri dosyasını küçültün, tokenizer/dataset örneklerini parça parça yükleyin.",
    ),
    # Unsloth'un füzyonlu (fused) kayıp çekirdeği, kurulu torch/transformers
    # sürümüyle uyuşmazsa ilk adımda patlar. Kullanıcı için tek anlamı var:
    # veri/model hatası değil, çekirdek uyumsuzluğu → ya kapatılır ya da
    # füzyonsuz yola geçilir.
    (
        r"grad_and_value\(f\)\(\*args\): Expected f\(\*args\) to return a scalar\s+Tensor"
        r"|unsloth_fused_ce_loss|UnslothFusedLoss\.apply",
        "unsloth-fused-loss",
        FAILED,
        "Unsloth'un füzyonlu kayıp (fused loss) çekirdeği bu torch/transformers "
        "sürümüyle uyumsuz: kayıp skaler değil, 2 boyutlu tensör döndü. "
        "Veri veya config hatası DEĞİL.",
        "En hızlı çözüm: `--backend transformers` (Qwen3.5-4B için "
        "configs/training_qwen3_5_4b_t4.yaml; 4B'de Unsloth gerekmez). "
        "Unsloth'ta kalınacaksa: `import unsloth` satırından ÖNCE "
        "os.environ['UNSLOTH_RETURN_LOGITS']='1' (bu repoda `--unsloth-plain-loss`) "
        "veya TrainingArguments'ta average_tokens_across_devices=False; ayrıca "
        "unsloth/unsloth_zoo'yu aynı sürüme sabitleyin.",
    ),
    (
        r"(?<![\d.])(?:HTTP\s*)?40[13](?![\d.])|Unauthorized|Forbidden|Invalid API key|"
        r"authentication (?:failed|error)",
        "auth",
        FAILED,
        "Kimlik doğrulama başarısız (401/403).",
        "Kaggle Secrets'ta EVREN_API_KEY doğru mu? Anahtarı hücreye yazmayın, "
        "Secrets üzerinden ortam değişkenine aktarın.",
    ),
    (
        r"(?<![\d.])404(?![\d.])|model_not_found|does not exist.*model|Repository Not Found|"
        r"gated repo",
        "model-not-found",
        FAILED,
        "Model kimliği bulunamadı (404 / gated).",
        "config içindeki name_or_path'i doğrulayın; HF'de gated model ise HF_TOKEN secret'ı ekleyin.",
    ),
    (
        r"ModuleNotFoundError|ImportError: cannot import name",
        "missing-package",
        FAILED,
        "Bir Python paketi eksik.",
        "Notebook'taki paket kurulum hücresini (INSTALL=True) çalıştırıp Runtime → Restart yapın.",
    ),
    (
        r"UnslothUnavailableError|No module named 'unsloth'",
        "no-unsloth",
        FAILED,
        "Unsloth kurulu değil ama Unsloth arka ucu istendi.",
        "`pip install unsloth` çalıştırın; ya da --backend transformers kullanın "
        "(gpt-oss-20b 16 GB'da generic yolla eğitilemez).",
    ),
    (
        r"429|rate limit|Too Many Requests|quota exceeded",
        "rate-limit",
        WARN,
        "Sağlayıcı oran sınırı (429).",
        "requests_per_minute'u düşürün; kod geri çekilme uygular ama uzun sürer.",
    ),
    (
        r"ConnectionError|NewConnectionError|Max retries exceeded|timed out|Timeout",
        "network",
        FAILED,
        "Ağ bağlantısı kurulamadı.",
        "Kaggle'da Settings → Internet açık mı? Sağlayıcı adresi erişilebilir mi?",
    ),
    (
        r"FileNotFoundError|No such file or directory",
        "missing-file",
        FAILED,
        "Bir dosya bulunamadı (çoğunlukla veri yolu).",
        "Kaggle Dataset eklendi mi? Yolu /kaggle/input/... olarak kontrol edin.",
    ),
    (
        r"PermissionError|Read-only file system",
        "permission",
        FAILED,
        "Yazma izni yok.",
        "Çıktıları /kaggle/working altına yazın; /kaggle/input salt okunurdur.",
    ),
    (
        r"ValueError: .*load_in_4bit|requires bitsandbytes|No GPU|CUDA is not available",
        "no-gpu",
        FAILED,
        "GPU/CUDA yok ya da 4-bit yolu bu ortamda çalışmıyor.",
        "Settings → Accelerator: GPU T4 x2 seçili mi?",
    ),
    (
        r"loss.*(nan|inf)|(nan|inf).*loss",
        "nan-loss",
        BROKEN,
        "Kayıp NaN/sonsuz oldu: eğitim sayısal olarak bozuldu.",
        "Learning rate'i düşürün (2e-5), max_grad_norm=0.5 tutun, fp16 yerine "
        "bf16/float32 deneyin. Bu checkpoint kullanılmaz.",
    ),
]

_COMPILED_ERRORS = [
    (re.compile(pattern, re.IGNORECASE | re.MULTILINE), code, severity, message, action)
    for pattern, code, severity, message, action in _ERROR_RULES
]

# {"loss": 1.23, "grad_norm": 0.5, "learning_rate": 5e-05, "epoch": 0.12}
_LOSS_VALUE = r"[-+]?(?:[\d.]+(?:[eE][-+]?\d+)?|nan|inf)"
_STEP_RE = re.compile(
    rf"\{{[^{{}}]*'loss'\s*:\s*(?P<loss>{_LOSS_VALUE})[^{{}}]*\}}", re.IGNORECASE
)
_EVAL_RE = re.compile(
    rf"\{{[^{{}}]*'eval_loss'\s*:\s*(?P<eval_loss>{_LOSS_VALUE})[^{{}}]*\}}", re.IGNORECASE
)
_GRAD_RE = re.compile(rf"'grad_norm'\s*:\s*({_LOSS_VALUE})", re.IGNORECASE)
_LR_RE = re.compile(r"'learning_rate'\s*:\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)")
_PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\[")
_TRAIN_LOSS_RE = re.compile(rf"'?train_loss'?\s*[:=]\s*({_LOSS_VALUE})", re.IGNORECASE)
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")


def _to_float(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


@dataclass
class LogStats:
    steps: int = 0
    losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    final_train_loss: float | None = None
    progress: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "first_loss": self.losses[0] if self.losses else None,
            "last_loss": self.losses[-1] if self.losses else None,
            "min_loss": min(self.losses) if self.losses else None,
            "eval_losses": self.eval_losses,
            "final_train_loss": self.final_train_loss,
            "median_grad_norm": sorted(self.grad_norms)[len(self.grad_norms) // 2]
            if self.grad_norms
            else None,
            "first_learning_rate": self.learning_rates[0] if self.learning_rates else None,
            "progress": self.progress,
        }


def parse_log(text: str) -> LogStats:
    """Extract numeric signals from a Trainer-style log."""
    stats = LogStats()
    for match in _STEP_RE.finditer(text):
        value = _to_float(match.group("loss"))
        if value is None:
            continue
        stats.losses.append(value)
        grad = _GRAD_RE.search(match.group(0))
        if grad and _to_float(grad.group(1)) is not None:
            stats.grad_norms.append(float(grad.group(1)))
        lr = _LR_RE.search(match.group(0))
        if lr and _to_float(lr.group(1)) is not None:
            stats.learning_rates.append(float(lr.group(1)))
    stats.steps = len(stats.losses)

    for match in _EVAL_RE.finditer(text):
        value = _to_float(match.group("eval_loss"))
        if value is not None:
            stats.eval_losses.append(value)

    train_loss = _TRAIN_LOSS_RE.findall(text)
    if train_loss:
        stats.final_train_loss = _to_float(train_loss[-1])

    progress = _PROGRESS_RE.findall(text)
    if progress:
        done, total = progress[-1]
        stats.progress = (int(done), int(total))
    return stats


def _loss_findings(stats: LogStats) -> list[Finding]:
    """Judge the loss curve: NaN, divergence, plateau, late rise, grad spikes."""
    findings: list[Finding] = []
    losses = stats.losses
    if not losses:
        return findings

    if any(value != value or value in (float("inf"), float("-inf")) for value in losses):
        findings.append(
            Finding(
                "nan-loss",
                BROKEN,
                "Kayıp değerleri arasında NaN/sonsuz var: eğitim sayısal olarak bozuldu.",
                [f"ilk: {losses[0]}", f"son: {losses[-1]}"],
                "Learning rate'i düşürün (2e-5), max_grad_norm=0.5 tutun, "
                "fp16 yerine bf16/float32 deneyin. Bu checkpoint kullanılmaz.",
            )
        )
        return findings

    first, last = losses[0], losses[-1]
    best = min(losses)
    n = len(losses)

    if last > first * 1.2 and last > best * 1.05:
        findings.append(
            Finding(
                "divergence",
                BROKEN,
                "Kayıp yükseliyor: eğitim ıraksıyor (diverge).",
                [
                    f"adım sayısı: {n}",
                    f"ilk kayıp: {first:.4f}",
                    f"en iyi: {best:.4f}",
                    f"son: {last:.4f}",
                ],
                "Erken durdurun. Learning rate'i yarıya düşürün (ör. 5e-5 → 2e-5), "
                "gradient_accumulation_steps'i artırın, T4'te fp16 yerine "
                "float32/bf16 yolunu deneyin.",
            )
        )
    elif n >= 10 and (first - last) / max(first, 1e-9) < 0.02:
        findings.append(
            Finding(
                "flat-loss",
                WARN,
                "Kayıp neredeyse hiç azalmıyor.",
                [f"ilk: {first:.4f}", f"son: {last:.4f}"],
                "Learning rate çok düşük olabilir veya veri/etiketler hatalı olabilir. "
                "Koşuyu 2 kat uzatın; düzelmezse veriyi inceleyin.",
            )
        )
    elif n >= 5 and last > best * 1.15:
        findings.append(
            Finding(
                "late-rise",
                WARN,
                "Kayıp en iyi noktasından belirgin yükseldi (aşırı öğrenme veya kararsızlık).",
                [f"en iyi: {best:.4f}", f"son: {last:.4f}"],
                "Adım sayısını azaltın veya daha erken checkpoint kullanın.",
            )
        )
    else:
        findings.append(
            Finding(
                "loss-decreasing",
                OK,
                "Kayıp genel olarak düşüyor.",
                [f"adım: {n}", f"ilk: {first:.4f}", f"en iyi: {best:.4f}", f"son: {last:.4f}"],
                "",
            )
        )

    finite_grads = [g for g in stats.grad_norms if g == g and abs(g) != float("inf")]
    if finite_grads:
        worst = max(finite_grads)
        if worst > 10:
            findings.append(
                Finding(
                    "grad-spike",
                    WARN,
                    f"Gradyan normu çok yükseldi (en yüksek {worst:.1f}).",
                    [f"medyan: {sorted(finite_grads)[len(finite_grads) // 2]:.3f}"],
                    "max_grad_norm=0.5 tutun; learning rate'i düşürün.",
                )
            )
    return findings


def _error_findings(text: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()
    for pattern, code, severity, message, action in _COMPILED_ERRORS:
        match = pattern.search(text)
        if not match:
            continue
        line_no = text[: match.start()].count("\n")
        evidence = [lines[line_no].strip()[:200]] if line_no < len(lines) else []
        findings.append(Finding(code, severity, message, evidence, action))
    if _TRACEBACK_RE.search(text) and not any(f.severity == FAILED for f in findings):
        findings.append(
            Finding(
                "traceback",
                FAILED,
                "Logda yakalanmamış bir Python hatası (traceback) var.",
                [],
                "Traceback'in son satırındaki hata mesajına bakın; genelde eksik "
                "paket, yanlış yol veya bellek sorunudur.",
            )
        )
    return findings


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Keep one finding per code, preferring the most severe, then the first."""
    best: dict[str, Finding] = {}
    for finding in findings:
        current = best.get(finding.code)
        if current is None or SEVERITY_ORDER[finding.severity] > SEVERITY_ORDER[current.severity]:
            best[finding.code] = finding
    return list(best.values())


def analyze_log(text: str) -> dict[str, Any]:
    """Return a full, JSON-serialisable analysis of a training log."""
    stats = parse_log(text)
    findings = _dedupe(_error_findings(text) + _loss_findings(stats))

    hard_failures = [f for f in findings if f.severity == FAILED]
    broken = [f for f in findings if f.severity == BROKEN]
    warnings = [f for f in findings if f.severity == WARN]

    if hard_failures:
        verdict = FAILED
    elif broken:
        verdict = BROKEN
    elif warnings:
        verdict = WARN
    else:
        verdict = OK

    # A run that stopped before its declared step budget without an error is
    # suspicious even if what we saw looked healthy.
    if (
        verdict in (OK, WARN)
        and stats.progress
        and stats.progress[0] < stats.progress[1]
        and stats.steps > 0
    ):
        done, total = stats.progress
        findings.append(
            Finding(
                "incomplete",
                WARN,
                f"Koşu {done}/{total} adımda görünüyor (tamamlanmamış olabilir).",
                [],
                "Oturum süresi dolduysa checkpoint'ten devam edin; log kesilmiş olabilir.",
            )
        )
        if verdict == OK:
            verdict = WARN

    return {
        "verdict": verdict,
        "title": _TITLES[verdict],
        "stats": stats.to_dict(),
        "findings": [f.to_dict() for f in findings],
        "note": (
            "Bu araç yalnızca eğitimin MEKANİĞİNİ denetler (çökme, NaN, ıraksama). "
            "Model kalitesi hakkında hiçbir şey söylemez; kalite için "
            "gokturk.evaluate ve insan incelemesi gerekir."
        ),
    }


def analyze_manifest(data: dict[str, Any], source: str = "") -> dict[str, Any]:
    """Summarise a ``run_manifest.json`` written by ``gokturk.train``."""
    findings: list[Finding] = []
    metrics = data.get("metrics") or {}
    kind = data.get("kind", "?")
    backend = data.get("backend", "?")
    dtype = data.get("resolved_dtype", "?")
    loss = metrics.get("train_loss") or metrics.get("train_runtime") and metrics.get("train_loss")
    eval_loss = metrics.get("eval_loss") or metrics.get("eval_eval_loss")

    if data.get("preflight", {}).get("ok") is False:
        findings.append(
            Finding(
                "preflight-failed",
                FAILED,
                "Manifest ön denetimin geçmediğini söylüyor.",
                [],
                "Eğitim gerçekten çalışmadı; ön denetim bulgularını okuyun.",
            )
        )

    if kind == "smoke":
        findings.append(
            Finding(
                "smoke-run",
                WARN,
                "Bu bir DUMAN TESTİ: rastgele başlatılmış küçük model.",
                [str(data.get("note", ""))[:180]],
                "Sonuçları model kalitesi sanmayın; yalnızca boru hattı testidir.",
            )
        )

    if isinstance(loss, (int, float)):
        if loss != loss or loss in (float("inf"), float("-inf")):
            findings.append(
                Finding("nan-loss", BROKEN, "Kayıp NaN/sonsuz.", [f"train_loss={loss}"], "Koşuyu kullanmayın.")
            )
        elif loss > 6:
            findings.append(
                Finding(
                    "high-loss",
                    WARN,
                    f"Train loss yüksek ({loss:.3f}).",
                    [],
                    "Domain'e göre değişir; 60 adımlık kısa koşuda normal olabilir. "
                    "Daha uzun koşuda da düşmüyorsa learning rate/veri kontrol edin.",
                )
            )
        else:
            findings.append(
                Finding("loss-present", OK, f"Train loss kaydedildi: {loss:.4f}", [], "")
            )
    else:
        findings.append(
            Finding(
                "no-loss",
                WARN,
                "Manifestte kayıp metriği yok.",
                [],
                "Eğitim tamamlanmamış veya metrikler kaydedilmemiş olabilir.",
            )
        )

    if isinstance(eval_loss, (int, float)) and isinstance(loss, (int, float)) and loss > 0:
        if eval_loss > loss * 1.5:
            findings.append(
                Finding(
                    "eval-gap",
                    WARN,
                    f"Değerlendirme kaybı train kaybından belirgin yüksek "
                    f"({eval_loss:.3f} vs {loss:.3f}).",
                    [],
                    "Aşırı öğrenme veya train/eval dağılım farkı olabilir; "
                    "veri setlerinin sızıntısız bölündüğünü doğrulayın.",
                )
            )

    train_examples = (data.get("data") or {}).get("train_examples")
    if isinstance(train_examples, int) and train_examples < 50:
        findings.append(
            Finding(
                "tiny-dataset",
                WARN,
                f"Eğitim seti çok küçük ({train_examples} örnek).",
                [],
                "10.000 örneklik hedefe göre bu bir deneme koşusudur; "
                "sonuçları genel kalite sanmayın.",
            )
        )

    verdict = OK
    for finding in findings:
        if SEVERITY_ORDER[finding.severity] > SEVERITY_ORDER[verdict]:
            verdict = finding.severity

    return {
        "source": source,
        "verdict": verdict,
        "title": _TITLES[verdict],
        "run": {
            "kind": kind,
            "backend": backend,
            "dtype": dtype,
            "duration_s": data.get("duration_s"),
            "train_examples": train_examples,
            "metrics": metrics,
        },
        "findings": [f.to_dict() for f in findings],
        "note": (
            "Manifest, koşunun kendi kaydettiği sayılardır. Kalite iddiası taşımaz; "
            "değerlendirme ayrı bir adımdır."
        ),
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
_ICONS = {OK: "✓", WARN: "!", BROKEN: "✗", FAILED: "✗"}


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"KARAR: {report['title']}")
    lines.append("=" * 60)

    stats = report.get("stats") or report.get("run") or {}
    if stats:
        lines.append("")
        lines.append("--- Özet sayılar ---")
        for key, value in stats.items():
            if value is None or value == [] or value == {}:
                continue
            if isinstance(value, float):
                lines.append(f"  {key:<22} {value:.5g}")
            else:
                lines.append(f"  {key:<22} {value}")

    findings = report.get("findings", [])
    if findings:
        lines.append("")
        lines.append("--- Bulgular ---")
        for finding in findings:
            icon = _ICONS.get(finding["severity"], "?")
            lines.append(f"  [{icon}] {finding['code']}: {finding['message']}")
            for item in finding.get("evidence", []):
                if item:
                    lines.append(f"        kanıt: {item}")
            if finding.get("action"):
                lines.append(f"        yapılacak: {finding['action']}")

    lines.append("")
    lines.append(f"NOT: {report['note']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Eğitim logunu/manifesti oku ve sade Türkçe karar ver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python -m gokturk.logcheck --log train.log\n"
            "  python -m gokturk.logcheck --manifest outputs/run/run_manifest.json\n"
            "  python -m gokturk.logcheck --stdin < kopyalanan_log.txt\n"
        ),
    )
    parser.add_argument("--log", help="Eğitim log dosyası (metin)")
    parser.add_argument("--manifest", help="run_manifest.json yolu")
    parser.add_argument("--stdin", action="store_true", help="Logu standart girdiden oku")
    parser.add_argument("--json", action="store_true", help="JSON çıktısı")
    parser.add_argument(
        "--fail-on",
        choices=["warning", "broken", "failure"],
        default="failure",
        help="Hangi seviyede sıfırdan farklı çıkış kodu dönsün (CI için)",
    )
    args = parser.parse_args(argv)

    if args.manifest:
        path = Path(args.manifest)
        if not path.is_file():
            print(f"Dosya bulunamadı: {path}", file=sys.stderr)
            return 4
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Geçersiz JSON: {exc}", file=sys.stderr)
            return 4
        report = analyze_manifest(data, source=str(path))
    elif args.log:
        path = Path(args.log)
        if not path.is_file():
            print(f"Dosya bulunamadı: {path}", file=sys.stderr)
            return 4
        report = analyze_log(path.read_text(encoding="utf-8", errors="replace"))
    elif args.stdin:
        report = analyze_log(sys.stdin.read())
    else:
        parser.error("--log, --manifest veya --stdin gerekli")

    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))

    thresholds = {"warning": WARN, "broken": BROKEN, "failure": FAILED}
    return 1 if SEVERITY_ORDER[report["verdict"]] >= SEVERITY_ORDER[thresholds[args.fail_on]] else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
