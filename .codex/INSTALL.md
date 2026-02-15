# Installing WANDB-Plot-Skill for Codex

## Step 1: Clone the repository

```bash
mkdir -p ~/.codex
git clone https://github.com/RubenBranco/WANDB-Plot-Skill.git ~/.codex/wandb-plot-skill
```

## Step 2: Add to your AGENTS.md

Add this section to `~/.codex/AGENTS.md`:

```markdown
### W&B Plot Skill
For plotting and analyzing Weights & Biases runs, see:
~/.codex/wandb-plot-skill/skills/wandb-plot/SKILL.md
```

## Step 3: Install dependencies

```bash
cd ~/.codex/wandb-plot-skill/skills/wandb-plot
pip install -e .
# or: uv pip install -e .
```

## Step 4: Set up authentication

```bash
export WANDB_API_KEY=your_key_here
```
