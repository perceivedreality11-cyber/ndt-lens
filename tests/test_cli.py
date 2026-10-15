import os
import subprocess
import sys


def test_cli_chinese_diagnostics_are_utf8_under_legacy_terminal_encoding(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "ndt.cli", "check-data", "--data", str(tmp_path / "missing.yaml")],
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "错误" in result.stderr.decode("utf-8")
