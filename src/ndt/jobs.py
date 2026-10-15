"""Training runs in a detached local process, surviving Streamlit reruns."""

import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ndt.dataset import validate_dataset
from ndt.provenance import environment_report


def write_status(path, value):
    path = Path(path)
    temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_job(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    launch = Path(path).with_name("process.json")
    if value["status"] == "queued" and launch.is_file():
        value.update(json.loads(launch.read_text(encoding="utf-8")))
    if value["status"] == "queued" and not value.get("pid") and time.time() - Path(path).stat().st_mtime > 60:
        value.update(status="failed", error="训练进程未能启动，请查看日志。")
        write_status(Path(path), value)
    if value["status"] in ("queued", "running") and value.get("pid"):
        import psutil

        try:
            process = psutil.Process(value["pid"])
            if abs(process.create_time() - value.get("process_started", process.create_time())) > 2:
                raise psutil.NoSuchProcess(value["pid"])
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                raise psutil.NoSuchProcess(value["pid"])
        except psutil.NoSuchProcess:
            # Re-read: the worker may have completed between the first read and the check.
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            if value["status"] in ("queued", "running"):
                value.update(status="failed", error="训练进程已退出，请查看日志后重新启动。")
                write_status(Path(path), value)
    return value


def list_jobs(root):
    folder = Path(root) / "jobs"
    return [read_job(p) for p in sorted(folder.glob("*/job.json"), reverse=True)]


def start_training(root, data, *, epochs=100, imgsz=640, batch=8, device="cpu", model="yolo11n.pt"):
    validate_dataset(Path(data))
    if not (1 <= epochs <= 1000 and 64 <= imgsz <= 2048 and 1 <= batch <= 128):
        raise ValueError("训练参数超出允许范围。")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".training-start.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError("另一个页面正在启动训练，请稍后刷新。") from exc
    try:
        if any(job["status"] in ("queued", "running") for job in list_jobs(root)):
            raise ValueError("已有训练任务运行中。")
        job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
        folder = root / "jobs" / job_id
        folder.mkdir(parents=True)
        path = folder / "job.json"
        job = dict(
            id=job_id,
            status="queued",
            data=str(Path(data).resolve()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            model=model,
            directory=str(folder),
        )
        write_status(path, job)
        with (folder / "train.log").open("wb") as log:
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "ndt.jobs", str(path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    cwd=str(root),
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                import psutil

                try:
                    started = psutil.Process(process.pid).create_time()
                except psutil.NoSuchProcess:
                    started = time.time()
                write_status(folder / "process.json", {"pid": process.pid, "process_started": started})
            except Exception as exc:
                job.update(status="failed", error=str(exc))
                write_status(path, job)
                raise
        return path
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def run(path):
    import psutil

    job = read_job(path)
    job.update(
        status="running",
        pid=os.getpid(),
        process_started=psutil.Process().create_time(),
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_status(path, job)
    try:
        write_status(path.with_name("environment.json"), environment_report())
        from ultralytics import YOLO

        validate_dataset(Path(job["data"]))
        model = YOLO(job["model"])
        model.train(
            data=job["data"],
            epochs=job["epochs"],
            imgsz=job["imgsz"],
            batch=job["batch"],
            device=job["device"],
            workers=0,
            seed=42,
            deterministic=True,
            project=str(Path(job["directory"]) / "result"),
            name="ndt",
            exist_ok=False,
            hsv_h=0,
            hsv_s=0,
            hsv_v=0.1,
            fliplr=0.5,
            flipud=0,
            degrees=3,
            mosaic=0,
            mixup=0,
        )
        weights = Path(model.trainer.best).resolve()
        if not weights.is_file():
            raise RuntimeError("训练完成但未找到 best.pt。")
        job.update(status="completed", weights=str(weights), results=str(model.trainer.save_dir))
    except Exception as exc:
        traceback.print_exc()
        job.update(status="failed", error=str(exc))
    job["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_status(path, job)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
