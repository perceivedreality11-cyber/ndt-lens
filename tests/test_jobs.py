import json
from pathlib import Path

from ndt.jobs import read_job, run, write_status


def test_dead_training_process_becomes_actionable_failure(tmp_path):
    path = tmp_path / "job.json"
    write_status(path, {"status": "queued"})
    write_status(tmp_path / "process.json", {"pid": 2147483647, "process_started": 1})
    assert read_job(path)["status"] == "failed"
    assert "退出" in json.loads(path.read_text(encoding="utf-8"))["error"]


def test_training_failure_is_saved_not_left_running(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    # Avoid importing/downloading a real model in unit tests.
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=object))
    path = tmp_path / "job.json"
    write_status(path, {"status": "queued", "data": str(Path(tmp_path) / "missing.yaml")})
    run(path)
    job = read_job(path)
    assert job["status"] == "failed"
    assert "error" in job
