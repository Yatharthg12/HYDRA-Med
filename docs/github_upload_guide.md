# GitHub Upload Guide

## Release principle

Publish the authored implementation, tests, documentation, and reviewed
aggregate evidence. Exclude third-party datasets and papers, row-level derived
data, private Office documents, local environments, and rebuildable model
binaries unless a separate reviewed release has a clear reason to include them.

The repository documentation does not establish a redistribution grant for the
locally stored UCI CSV files. They should therefore be excluded from GitHub.

## Acquire and regenerate data

1. Visit the official UCI Diabetes 130-US Hospitals dataset page:
   <https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008>.
2. Review and accept the current source terms.
3. Place `diabetic_data.csv` and `IDS_mapping.csv` under `data/raw/`.
4. Run:

   ```powershell
   python run_experiments.py --dataset reduced --research
   ```

Reduced data and patient split assignments are regenerated deterministically.

## Commit

- `app.py`, `run_experiments.py`
- `src/urop_healthgraph/`
- `web/templates/`, `web/static/`
- `tests/`
- `README.md`, `LICENSE`, `.gitignore`
- `requirements.txt`, `pytest.ini`
- `data/README.md`
- authored Markdown and JSON documentation, including the reproducibility
  manifest
- reviewed aggregate metric JSON/CSV files
- reviewed aggregate figures
- aggregate graph statistics and the five-node Warshall artifact
- `.gitkeep` directory markers

## Do not commit

- `.venv/`, `venv/`
- `__pycache__/`, `.pytest_cache/`, coverage output
- `.vscode/`, `.idea/`, OS metadata
- logs and runtime test directories
- `data/raw/*.csv`
- `data/processed/*.csv`, including split assignments
- `artifacts/predictions/`
- `artifacts/graphs/sample_graph.json`
- `docs/*.docx`, `docs/*.pptx`
- `docs/References/*.pdf`

## Commit only after an explicit release decision

- `artifacts/models/`: small in this run, but rebuildable, Python-version
  dependent, and unnecessary for source reproduction;
- model-specific training-history JSON embedded in metrics: acceptable when
  aggregate and de-identified, but review size and contents;
- any future figure containing row-level labels or encounter identifiers;
- datasets only if redistribution permission is independently confirmed and
  documented for the exact files and version;
- large future binaries through Git LFS only when a release genuinely needs
  them.

## Review workflow

Stage explicit paths rather than `git add .`, then inspect:

```powershell
git status --short
git diff --cached --stat
git diff --cached
```

Search the staged source for local paths, secrets, row identifiers, draft
metadata, and unreviewed third-party material before publishing.

## License notice

The project license permits educational inspection and authorized evaluation
only; it is not an open-source license. Public visibility cannot technically
prevent copying or forking. Use a private repository with time-limited evaluator
access when view control is important. See `LICENSE` for the controlling terms.
