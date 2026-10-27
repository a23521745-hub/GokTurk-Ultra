from pathlib import Path

import pytest

from gokturk.dns_config import PUBLIC_CONFIG, configure_public_dns, restore_dns


def test_dns_backed_up_idempotently_and_restored(tmp_path):
    target, backup = tmp_path / "resolv.conf", tmp_path / "original"
    original = "nameserver 10.0.0.1\nsearch internal\n"
    target.write_text(original)
    for _ in range(2):
        assert configure_public_dns(resolv_path=target, backup_path=backup) == PUBLIC_CONFIG
    assert target.read_text() == PUBLIC_CONFIG
    assert backup.read_text() == original
    restore_dns(resolv_path=target, backup_path=backup)
    assert target.read_text() == original


def test_backup_failure_leaves_dns_untouched(tmp_path):
    target = tmp_path / "resolv.conf"
    target.write_text("original")
    blocker = tmp_path / "not-directory"
    blocker.write_text("file")
    with pytest.raises(OSError):
        configure_public_dns(resolv_path=target, backup_path=blocker / "backup")
    assert target.read_text() == "original"


def test_dns_write_failure_is_visible(tmp_path, monkeypatch):
    target, backup = tmp_path / "resolv.conf", tmp_path / "backup"
    target.write_text("original")
    real = Path.write_text

    def fail_public(self, text, **kwargs):
        if text == PUBLIC_CONFIG:
            raise PermissionError("read-only")
        return real(self, text, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_public)
    with pytest.raises(OSError, match="uygulanamadı"):
        configure_public_dns(resolv_path=target, backup_path=backup)
    assert target.read_text() == "original"
