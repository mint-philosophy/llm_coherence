# Scripts By Pipeline Stage

The `scripts/` folders mirror the experiment order. They contain command notes
for running the package modules from the repo root.

Use `validate_artifacts.py` to check that manifests, comparison indexes, and
model-run indexes are in sync:

```bash
PYTHONPATH=src python scripts/validate_artifacts.py
PYTHONPATH=src python scripts/validate_artifacts.py --write-indexes
```
