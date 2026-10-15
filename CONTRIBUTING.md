# Contributing

Use Python 3.10 or newer. The CI matrix covers Python 3.10 / 3.12 on Linux and Windows.

```bash
python -m pip install -e ".[ui,dev]"
ruff format --check .
ruff check .
pytest -q
```

Keep image loading and preprocessing consistent across annotation, training and inference. Prediction output must remain unreviewed until explicitly confirmed. Preserve group isolation and optimistic annotation revisions. Tests should exercise observable behavior and failure modes; default tests must not download weights or train a model.

Use a fresh output directory for each experiment. Record the source of data, split policy, initialization, seed, thresholds and environment. Report synthetic and real-data results separately. Add regression tests for changed validation, persistence, export or evaluation behavior.

Never commit private images, databases, weights, logs containing local data, virtual environments or secrets. Repository images should be generated examples or have clear redistribution permission.
