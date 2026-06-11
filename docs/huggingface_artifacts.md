# Hugging Face Artifact Archive

Use GitHub for code, canonical inputs, documentation, and compact summaries.
Use a Hugging Face Dataset repository for the generated artifacts that are too
large and noisy for GitHub.

The local publication payload currently has this shape:

- `outputs/07_model_runs/`: raw model choices and reasoning traces, about
  2.7 GB locally.
- `outputs/08_analysis/`: derived coherence, predictive-utility, forced-choice,
  and reasoning-summary outputs, about 329 MB locally.
- `outputs/09_figures_tables/`: final generated paper figures and tables when
  they exist. At the time this workflow was added, only README/placeholders
  were present locally.
- `results/`: compact public result summaries.

## Recommended Archive Structure

Create one HF Dataset repository, for example:

```text
mint-philosophy/llm-coherence-aies-2026-artifacts
```

Upload an artifact bundle with this layout:

```text
README.md
artifact_manifest.json
SHA256SUMS
outputs/
  07_model_runs/
  08_analysis/
  09_figures_tables/
results/
```

Optionally include the tracked `data/` stages in the HF archive as well. They
are already tracked in GitHub, but including them makes the HF archive easier to
inspect on its own.

## Prepare The Bundle

From the repository root, stage a fresh bundle outside the Git checkout:

```bash
python scripts/prepare_hf_artifact_bundle.py \
  /Users/elenaajayi/Downloads/llm_coherence_hf_artifacts \
  --include-inputs \
  --checksums
```

The script writes:

- a Hugging Face Dataset card at `README.md`;
- `artifact_manifest.json` with the source Git commit SHA, file counts, sizes,
  and per-file hashes;
- `SHA256SUMS` for file verification;
- hard links to the local artifact files by default, so it usually does not
  duplicate 3 GB of local data.

If hard links are not possible on the destination filesystem, the script falls
back to copying. Use `--link-mode copy` if you explicitly want a fully separate
copy.

If the bundle already exists and you intentionally want to overwrite matching
files, pass `--force`. For final publication, prefer a fresh directory so stale
files cannot remain from an older upload attempt.

## Upload To Hugging Face

Create a Hugging Face account and a Dataset repository through the web UI. Then
install the current Hub CLI and log in:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

Upload the prepared bundle:

```bash
HF_XET_HIGH_PERFORMANCE=1 hf upload-large-folder \
  mint-philosophy/llm-coherence-aies-2026-artifacts \
  --repo-type=dataset \
  /Users/elenaajayi/Downloads/llm_coherence_hf_artifacts
```

`upload-large-folder` is preferred here because the raw model-run archive has
thousands of files. It is resumable, uses multiple workers, and records upload
progress locally inside the bundle directory.

If the upload is interrupted, rerun the same command. The cached upload state
lets Hugging Face resume completed hashing/pre-upload steps.

## After Upload

1. Check the HF repository file tree and confirm these paths exist:
   `outputs/07_model_runs/`, `outputs/08_analysis/`, `outputs/09_figures_tables/`,
   `results/`, `artifact_manifest.json`, and `SHA256SUMS`.
2. Generate a DOI from the HF repository settings when the artifact is final.
3. Add the HF URL or DOI to `README.md` and `docs/replication.md`.
4. Commit that README/replication update to GitHub.

For camera-ready release, regenerate and stage final figures/tables before
running the bundle script if `outputs/09_figures_tables/` is still empty.

## Hugging Face References

- Upload files and large folders:
  <https://huggingface.co/docs/huggingface_hub/guides/upload>
- Create and upload datasets:
  <https://huggingface.co/docs/hub/datasets-adding>
- Repository storage guidance:
  <https://huggingface.co/docs/hub/storage-limits>
- Generate a DOI:
  <https://huggingface.co/docs/hub/doi>
