# W&B Plot Skill

This skill comes from a personal necessity. I use Claude and Codex a lot to crunch experiment analysis, and I couldn't find a skill that could download plots from W&B. W&B does have an MCP but it doesn't download plots, only data, as far as I know.

Disclaimer: This was 99.9% vibe-coded using Claude Code and Codex.

Download and generate plots from Weights & Biases runs. This repository ships a
skill and CLI scripts for:
- Listing projects in an entity
- Listing runs in a project
- Discovering available metrics for a run
- Downloading existing plot images
- Generating line plots from raw metric data (including multi-run comparison and EMA smoothing)

The implementation lives in `wandb-plot/`.

## Installation

### Claude Code (Plugin)

This repo is a Claude Code plugin (manifest at `.claude-plugin/plugin.json`) and includes a plugin skill at `skills/wandb-plot/SKILL.md`.

- Install from this repo (no publish required):
  - `git clone <repo-url> && cc --plugin-dir /path/to/WANDB-Plot-Skill`
- Marketplace install:
  - Run `/plugin install wandb-plot` inside Claude Code (marketplace installs don’t take arbitrary repo URLs).
- Runtime deps:
  - Install Python deps from `wandb-plot/` (see “Python Package” below).

### Codex (Skill)

Codex loads skills from `$CODEX_HOME/skills` (typically `~/.codex/skills`). To install this skill, copy the `wandb-plot/` directory into your Codex skills directory:

```bash
mkdir -p ~/.codex/skills
cp -R wandb-plot ~/.codex/skills/wandb-plot
```

Restart Codex after installing to pick up the new skill.
Install Python deps from `wandb-plot/` (see “Python Package” below).

### Python Package (Optional)

Using **uv** (recommended):

```bash
cd wandb-plot
uv pip install -e .
```

Using pip (fallback):

```bash
cd wandb-plot
pip install -r requirements.txt
```

## Authentication

Use a non-interactive auth method:

```bash
export WANDB_API_KEY=your_key_here
```

## Quick Start

```bash
# Run scripts from within the package directory
cd wandb-plot

# 1. List projects for your default entity
python3 scripts/list_projects.py --limit 10

# 2. List available runs in a project
python3 scripts/list_runs.py my-org/my-project --limit 10

# 3. View available metrics for a specific run
python3 scripts/list_metrics.py my-org/my-project run-id-123

# 4. Try downloading existing plots (faster)
python3 scripts/download_plots.py my-org/my-project run-id-123

# 5. Generate plots from raw data
python3 scripts/generate_plots.py my-org/my-project run-id-123 --metrics loss,accuracy

# 6. Compare runs with W&B-style EMA smoothing (default)
python3 scripts/generate_plots.py my-org/my-project run-a,run-b --metrics reward/total_mean --ema-weight 0.99 --viewport-scale 1000

# 7. Group outputs by metric prefix
python3 scripts/generate_plots.py my-org/my-project run-a,run-b --metrics rewards/total_mean,rewards/total_std --output /path/to/folder --group-by-prefix

# 8. Plot all metrics (excludes system metrics unless --include-system is set)
python3 scripts/generate_plots.py my-org/my-project run-a,run-b --all-metrics --output /path/to/folder --group-by-prefix
```

## Output

Default output directory:

```
wandb_plots/
└── <entity>_<project>/
    └── <run_name>_<run_id>/
        ├── loss.png
        ├── accuracy.png
        └── metadata.json
```

## Testing

```bash
# Run tests from within the package directory
cd wandb-plot

# Install dev dependencies
uv pip install -e ".[dev]"

# Run unit tests (no network)
pytest tests/ -v -m "not integration"

# Run integration tests (requires WANDB_API_KEY)
pytest tests/ -v -m integration
```

## Skills

- Claude plugin skill: `skills/wandb-plot/SKILL.md`
- Codex skill: `wandb-plot/SKILL.md` (and `wandb-plot/SKILL_CODEX.md`)
