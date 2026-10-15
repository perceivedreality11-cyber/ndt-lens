# NDT Lens

**A local, human-in-the-loop X-ray weld defect inspection workbench.**

[中文](README.md) · [Architecture](docs/ARCHITECTURE.md) · [Experiments](docs/EXPERIMENTS.md)

![Procedural phantom and morphology baseline](docs/demo.png)

NDT Lens connects image import, bounding-box annotation, human confirmation, YOLO training, automatic annotation and review. Initial classes are porosity, crack and inclusion.

## Try it

```bash
git clone https://github.com/perceivedreality11-cyber/ndt-lens.git
cd ndt-lens
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -e ".[ui,dev]"
python -m streamlit run app.py
```

Click **打开合成演示项目** in the sidebar. It creates 15 procedural images in a separate workspace, including 12 generated ground-truth annotations and 3 unreviewed images. The interface is currently in Chinese.

Install `python -m pip install -e ".[train]"` for YOLO. The demo workspace defaults to an offline, one-epoch CPU run from random initialization. This verifies the training workflow; it does not produce a useful industrial detector.

## Engineering features

- Canvas bounding-box editor, table editing and SQLite persistence.
- Pixel hashing, workpiece-group split isolation and optimistic annotation revisions.
- Training snapshots contain confirmed samples only. Automatic predictions never become ground truth without confirmation.
- Background training processes, logs, model provenance and failure detection.
- Portable YOLO exports containing images without overlays, labels, config and review manifests.
- A reproducible morphology-versus-YOLO experiment with one-to-one IoU matching and independent test evaluation.
- pytest and Streamlit page tests; GitHub Actions configured for Windows/Linux and Python 3.10/3.12.

```bash
python -m pip install -e ".[train]"
python scripts/run_experiment.py --output outputs/my-experiment --epochs 100 --count 120 --size 256 --seed 2026
```

## Scope

This is a research and engineering project. Repository examples and experiments use procedural phantoms. No real industrial benchmark, field accuracy, or automatic acceptance decision is claimed. The UI targets local single-user use, not a public multi-user service. See [experiment evidence and limitations](docs/EXPERIMENTS.md).

AGPL-3.0-only. Ultralytics and model assets are subject to their respective licenses. Data, weights, databases and runtime outputs are excluded from version control.
