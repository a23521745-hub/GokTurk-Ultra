"""Çevrimdışı eğitim: DNS/ağ sorunu eğitimi yakmasın.

Neden var
---------
2026-10-26'da Kaggle'ın resolver'ı aralıklı düştü (``Temporary failure in name
resolution``) ve 5,7 saatlik üretim koşusu boşa gitti. Aynı risk **eğitim** için
de geçerli görünüyor, ama değil: eğitim yalnızca taban ağırlıkları indirmek için
ağa ihtiyaç duyar. Ağırlıklar ``/kaggle/input`` altında bir Dataset/Model olarak
eklendiğinde eğitim **tamamen çevrimdışı** koşabilir.

Bu modül üç şeyi yapar:

* :func:`resolve_model_source` — verilen yol/kimlik yerel bir dizine denk geliyor
  mu, yoksa Hub'dan mı indirilecek, söyler (``/kaggle/input`` altında arar).
* :func:`enable_offline_mode` — ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``
  değişkenlerini kurar; kütüphaneler ağa hiç çıkmaz, DNS'e hiç dokunmaz.
* :func:`offline_report` — eğitimden önce "bu koşu ağa muhtaç mı" sorusunu
  tek ekranda yanıtlar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Yerel bir model dizini olduğunu kanıtlayan dosyalar (en az biri bulunmalı).
MODEL_DIR_MARKERS = ("config.json", "model.safetensors", "pytorch_model.bin")

#: Kaggle'da veri setleri/modeller buraya bağlanır.
DEFAULT_SEARCH_ROOTS = ("/kaggle/input",)

#: Yerel ağırlık arama köklerini geçici olarak değiştirmek için (ör. test veya
#: başka bir bağlama noktası): ``GTU_MODEL_SEARCH_ROOTS=/a:/b``.
SEARCH_ROOTS_ENV = "GTU_MODEL_SEARCH_ROOTS"

OFFLINE_ENV_VARS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def is_local_model_dir(path: str | Path) -> bool:
    """True when *path* is a directory that looks like a saved HF model."""
    candidate = Path(path)
    if not candidate.is_dir():
        return False
    return any((candidate / marker).exists() for marker in MODEL_DIR_MARKERS)


def search_roots_from_env() -> tuple[str, ...]:
    """``GTU_MODEL_SEARCH_ROOTS`` varsa onu, yoksa :data:`DEFAULT_SEARCH_ROOTS`."""
    raw = os.environ.get(SEARCH_ROOTS_ENV, "").strip()
    if not raw:
        return DEFAULT_SEARCH_ROOTS
    return tuple(part for part in raw.split(os.pathsep) if part)


def find_local_model(
    name_or_path: str, search_roots: tuple[str, ...] | list[str] | None = None
) -> str | None:
    """Return a local directory for *name_or_path*, or ``None``.

    Accepts either a direct path or a Hub id such as ``Qwen/Qwen3.5-4B`` (whose
    local mirror would be ``.../Qwen/Qwen3.5-4B`` or ``.../Qwen3.5-4B``). The
    search is shallow on purpose: Kaggle mounts datasets one level deep.
    """
    if is_local_model_dir(name_or_path):
        return str(Path(name_or_path))
    roots = DEFAULT_SEARCH_ROOTS if search_roots is None else search_roots

    repository = name_or_path.rstrip("/").split("/")
    wanted = [repository[-1]]
    if len(repository) > 1:
        wanted.append("/".join(repository))

    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for name in wanted:
            for candidate in (base / name, base / "models" / name):
                if is_local_model_dir(candidate):
                    return str(candidate)
        # Derin arama: /kaggle/input/<slug>/Qwen3.5-4B
        for candidate in sorted(base.rglob(repository[-1])):
            if is_local_model_dir(candidate):
                return str(candidate)
    return None


def resolve_model_source(
    name_or_path: str,
    *,
    search_roots: tuple[str, ...] | list[str] | None = None,
    offline: bool = False,
) -> str:
    """Return the path/id to hand to ``from_pretrained``.

    ``offline=True`` turns a missing local copy into a :class:`FileNotFoundError`
    *before* training starts, instead of a confusing 9 GB download attempt on a
    machine that has no network.
    """
    local = find_local_model(name_or_path, search_roots)
    if local:
        return local
    if offline:
        raise FileNotFoundError(
            f"Çevrimdışı mod istendi ama '{name_or_path}' için yerel kopya bulunamadı. "
            f"Aranan kökler: {list(search_roots or DEFAULT_SEARCH_ROOTS)}. "
            "Ağırlıkları bir Kaggle Dataset/Model "
            "olarak ekleyin ya da çevrimdışı modu kapatın."
        )
    return name_or_path


def enable_offline_mode(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Set the HF/transformers offline switches; returns what was set.

    Must run **before** ``transformers``/``huggingface_hub`` are imported, hence
    the notebook calls it at the top of the training cell.
    """
    target = os.environ if environ is None else environ
    for key, value in OFFLINE_ENV_VARS.items():
        target.setdefault(key, value)
    return {key: target[key] for key in OFFLINE_ENV_VARS}


@dataclass
class OfflineReport:
    """Whether this run needs the network, and why."""

    requested_model: str
    resolved_model: str
    local: bool
    data_files: dict[str, str] = field(default_factory=dict)
    missing_data: list[str] = field(default_factory=list)

    @property
    def network_needed(self) -> bool:
        """Only the base weights can force a download; data is always local."""
        return not self.local

    @property
    def ok(self) -> bool:
        return self.local and not self.missing_data

    def render(self) -> str:
        lines = ["Çevrimdışı eğitim denetimi"]
        if self.local:
            lines.append(f"  TABAN AĞIRLIK : YEREL -> {self.resolved_model}")
            lines.append("  AĞ İHTİYACI   : YOK (eğitim internet olmadan koşabilir)")
        else:
            lines.append(f"  TABAN AĞIRLIK : HUB -> {self.resolved_model}")
            lines.append(
                "  AĞ İHTİYACI   : VAR (yalnızca ilk indirme). DNS düşerse eğitim başlamaz;"
                " ağırlıkları Kaggle Dataset/Model olarak ekleyin."
            )
        for split, path in self.data_files.items():
            lines.append(f"  VERİ {split:<6} : {path}")
        for item in self.missing_data:
            lines.append(f"  VERİ EKSİK    : {item}")
        if self.ok:
            lines.append("  SONUÇ         : Bu koşu tamamen çevrimdışı çalışabilir.")
        return "\n".join(lines)


def offline_report(
    model: str,
    *,
    data_files: dict[str, str] | None = None,
    search_roots: tuple[str, ...] | list[str] | None = None,
) -> OfflineReport:
    """Build an :class:`OfflineReport` for a planned training run."""
    resolved = resolve_model_source(
        model, search_roots=search_roots or search_roots_from_env(), offline=False
    )
    local = is_local_model_dir(resolved)
    report = OfflineReport(
        requested_model=model,
        resolved_model=resolved,
        local=local,
    )
    for split, path in (data_files or {}).items():
        if path and Path(path).is_file():
            report.data_files[split] = path
        else:
            report.missing_data.append(f"{split}: {path or '(tanımsız)'}")
    return report
