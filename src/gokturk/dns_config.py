"""Explicit, reversible Linux resolver configuration for a disposable Kaggle session.

No IP pinning, TLS bypass, API-key forwarding or global socket monkeypatching.
Writing resolv.conf does not prove public resolvers are reachable.
"""
from pathlib import Path

PUBLIC_RESOLVERS = ("1.1.1.1", "8.8.8.8")
PUBLIC_CONFIG = "nameserver 1.1.1.1\nnameserver 8.8.8.8\noptions timeout:2 attempts:2 rotate\n"


def configure_public_dns(
    *,
    resolv_path: str | Path = "/etc/resolv.conf",
    backup_path: str | Path = "/kaggle/working/resolv.conf.original",
) -> str:
    """Back up once, write both resolvers, verify readback; raise on failure.

    Requires write permission. Does not replace the file (it may be a bind mount).
    Backup is session-local: never reuse it across containers.
    """
    target, backup = Path(resolv_path), Path(backup_path)
    original = target.read_text(encoding="utf-8")
    backup.parent.mkdir(parents=True, exist_ok=True)
    try:
        with backup.open("x", encoding="utf-8") as fh:
            fh.write(original)
    except FileExistsError:
        pass  # Re-running a cell must not overwrite the original resolver.
    try:
        target.write_text(PUBLIC_CONFIG, encoding="utf-8")
        if target.read_text(encoding="utf-8") != PUBLIC_CONFIG:
            raise OSError("DNS ayarı geri okuma doğrulamasını geçemedi")
    except OSError as exc:
        try:
            target.write_text(original, encoding="utf-8")
        except OSError:
            raise OSError("DNS yazılamadı; geri alma da başarısız. Oturumu yeniden başlatın.") from exc
        raise OSError("DNS ayarı uygulanamadı; önceki içerik korundu/geri yüklendi.") from exc
    return PUBLIC_CONFIG


def restore_dns(
    *,
    resolv_path: str | Path = "/etc/resolv.conf",
    backup_path: str | Path = "/kaggle/working/resolv.conf.original",
) -> None:
    """Restore this session's backup; fail visibly if it cannot be restored."""
    original = Path(backup_path).read_text(encoding="utf-8")
    target = Path(resolv_path)
    target.write_text(original, encoding="utf-8")
    if target.read_text(encoding="utf-8") != original:
        raise OSError("Özgün DNS ayarı geri yüklenemedi")
