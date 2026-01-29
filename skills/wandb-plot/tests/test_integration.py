"""Integration tests for full W&B plot workflow."""

import os

import pytest

from scripts.list_runs import list_runs
from scripts.list_metrics import list_metrics
from scripts.download_plots import download_plots
from scripts.generate_plots import generate_plots


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("WANDB_API_KEY"), reason="No W&B API key")
def test_full_workflow(tmp_path):
    """Test full workflow: list runs -> metrics -> download -> generate."""
    runs = list_runs("wandb/test-project", limit=1)
    assert len(runs) > 0

    run_id = runs[0]["id"]

    metrics = list_metrics("wandb/test-project", run_id)
    assert len(metrics) > 0

    downloaded = download_plots(
        "wandb/test-project",
        run_id,
        output_dir=str(tmp_path / "downloads")
    )
    assert isinstance(downloaded, list)

    metric_name = list(metrics.keys())[0]
    generated = generate_plots(
        "wandb/test-project",
        run_id,
        [metric_name],
        output_dir=str(tmp_path / "generated")
    )
    assert len(generated) == 1
