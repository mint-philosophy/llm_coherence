# Scripts By Pipeline Stage

The `scripts/` folders mirror the experiment order. They contain command notes
for running the package modules from the repo root.

Use `validate_artifacts.py` to check that manifests, comparison indexes, and
model-run indexes are in sync:

```bash
PYTHONPATH=src python scripts/validate_artifacts.py
PYTHONPATH=src python scripts/validate_artifacts.py --write-indexes
```

Use `prepare_hf_artifact_bundle.py` to stage the external Hugging Face artifact
archive for raw model runs, derived analysis outputs, and final figures/tables:

```bash
python scripts/prepare_hf_artifact_bundle.py /path/to/fresh/hf_bundle --checksums
```
