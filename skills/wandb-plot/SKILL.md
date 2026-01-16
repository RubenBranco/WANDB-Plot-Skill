---
name: wandb-plot
description: |
  Use when the user wants to analyze W&B experiments by:
  - listing runs in a project
  - discovering available metrics
  - downloading existing plot images from a run
  - generating plots from metric history
version: 0.1.0
---

# W&B Plot Skill (Claude Plugin)

This plugin ships the Python scripts under `${CLAUDE_PLUGIN_ROOT}/wandb-plot/`.
Run them from that directory so imports resolve correctly.

## Prereqs

- Auth: set `WANDB_API_KEY` (recommended) or run `wandb login`.

## Tools (Scripts)

All commands assume:

```bash
cd "${CLAUDE_PLUGIN_ROOT}/wandb-plot"
```

### `scripts/list_runs.py`

**Inputs**
- `<entity/project>` (required)
- `--state <state>` (optional)
- `--limit <n>` (optional, default: 100)
- `--json` (optional)

**Output**
- Stdout table (default) or JSON list (with `--json`) containing: `id`, `name`, `state`, `created_at`, `summary_metrics`, `tags`.

### `scripts/list_metrics.py`

**Inputs**
- `<entity/project>` (required)
- `<run_id>` (required; run id or run name)
- `--include-system` (optional)
- `--json` (optional)

**Output**
- Stdout table (default) or JSON dict (with `--json`) keyed by metric name.

### `scripts/download_plots.py`

**Inputs**
- `<entity/project>` (required)
- `<run_id>` (required)
- `--pattern "<glob>"` (optional)
- `--output <dir>` (optional)
- `--force` (optional)

**Output**
- Writes downloaded images and `metadata.json` to the output directory.
- Prints “No plot files found…” when nothing matches (use `generate_plots.py`).

### `scripts/generate_plots.py`

**Inputs**
- `<entity/project>` (required)
- `<run_id>` (required)
- `--metrics "<m1,m2,...>"` (required)
- `--full-res` (optional)
- `--smooth <n>` (optional)
- `--output <dir>` (optional)

**Output**
- Writes `<metric>.png` per metric plus `metadata.json` to the output directory.

## Workflow

```bash
cd "${CLAUDE_PLUGIN_ROOT}/wandb-plot"
python3 scripts/list_runs.py <entity/project> --limit 10
python3 scripts/list_metrics.py <entity/project> <run_id>
python3 scripts/download_plots.py <entity/project> <run_id>
python3 scripts/generate_plots.py <entity/project> <run_id> --metrics loss,accuracy
```
