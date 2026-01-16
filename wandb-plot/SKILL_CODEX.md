---
name: wandb-plot
description: |
  Codex-focused variant for scripted, deterministic CLI usage:
  - List runs, list metrics
  - Download existing plot images
  - Generate plots from metric history
---

# W&B Plot Skill (Codex)

## Prereqs

- Auth: `export WANDB_API_KEY=...` (non-interactive).
- Run from `wandb-plot/` so `scripts/` is on disk.

## Tools (Scripts)

### `scripts/list_runs.py`

**Inputs**: `<entity/project>`, optional `--state`, `--limit`, `--json`.

**Outputs**
- Default: human-readable table to stdout.
- With `--json`: JSON list of runs, each containing `id`, `name`, `state`, `created_at`, `summary_metrics`, `tags`.

### `scripts/list_metrics.py`

**Inputs**: `<entity/project> <run_id>`, optional `--include-system`, `--json`.

**Outputs**
- Default: human-readable table to stdout.
- With `--json`: JSON dict keyed by metric name, values include `type`, `count`, `non_null_count` and (numeric) `min/max/mean/std`.

### `scripts/download_plots.py`

**Inputs**: `<entity/project> <run_id>`, optional `--pattern`, `--output`, `--force`.

**Outputs**
- Writes downloaded images and `metadata.json` to the output directory.
- Returns “No plot files found…” when there are no matching images.

### `scripts/generate_plots.py`

**Inputs**: `<entity/project> <run_id> --metrics <m1,m2>`, optional `--full-res`, `--smooth`, `--output`.

**Outputs**
- Writes `<metric>.png` per metric plus `metadata.json` to the output directory.

## Commands (Copy/Paste)

```bash
python3 scripts/list_runs.py <entity/project> [--state finished] [--limit 100] [--json]
python3 scripts/list_metrics.py <entity/project> <run_id> [--include-system] [--json]
python3 scripts/download_plots.py <entity/project> <run_id> [--pattern "*.png"] [--output <dir>] [--force]
python3 scripts/generate_plots.py <entity/project> <run_id> --metrics <m1,m2> [--full-res] [--smooth N] [--output <dir>]
```

## Notes

- Prefer `--json` when you need structured output.
- Default outputs go to `wandb_plots/<entity>_<project>/<run_name>_<run_id>/` (files: `*.png`, `metadata.json`).
- If `download_plots.py` returns no files, run `generate_plots.py`.
