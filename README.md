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

The implementation lives in `skills/wandb-plot/`.

## Installation

### Claude Code (Plugin)

This repo is a Claude Code plugin (manifest at `.claude-plugin/plugin.json`) and uses the skill file at `skills/wandb-plot/SKILL.md`.

#### Option 1: Install from Marketplace (Recommended)

The easiest way to install this plugin is through the marketplace:

1. **Add the marketplace** (one-time setup):
   ```
   /plugin marketplace add RubenBranco/WANDB-Plot-Skill
   ```

2. **Install the plugin**:
   ```
   /plugin install wandb-plot
   ```

That's it! The plugin will be automatically installed and ready to use.

#### Option 2: Install from Source

If you want to install directly from this repository:

```bash
git clone https://github.com/RubenBranco/WANDB-Plot-Skill.git
cc --plugin-dir /path/to/WANDB-Plot-Skill
```

#### Runtime Dependencies

After installing the plugin (either method), you'll need to install Python dependencies from `skills/wandb-plot/` (see "Python Package" section below).

### Codex

See [.codex/INSTALL.md](.codex/INSTALL.md) for detailed Codex installation instructions.

### Python Package (Optional)

Using **uv** (recommended):

```bash
cd skills/wandb-plot
uv pip install -e .
```

Using pip (fallback):

```bash
cd skills/wandb-plot
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
cd skills/wandb-plot

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
cd skills/wandb-plot

# Install dev dependencies
uv pip install -e ".[dev]"

# Run unit tests (no network)
pytest tests/ -v -m "not integration"

# Run integration tests (requires WANDB_API_KEY)
pytest tests/ -v -m integration
```

## Skills

- Skill file: `skills/wandb-plot/SKILL.md`
