"""Shared pytest fixtures for W&B Plot Skill tests."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def mock_wandb_api(mocker):
    """Mock W&B API instance with viewer info."""
    api = mocker.Mock()
    api.viewer = {"entity": "test-entity", "username": "test-user"}
    return api


@pytest.fixture
def mock_run(mocker):
    """Mock W&B run object with common attributes."""
    run = mocker.Mock()
    run.id = "test-run-123"
    run.name = "test-run"
    run.state = "finished"
    run.entity = "test-entity"
    run.project = "test-project"
    run.created_at = "2024-01-01T00:00:00"
    run.summary = {"loss": 0.234, "accuracy": 0.945}
    run.config = {"learning_rate": 0.001, "batch_size": 32}
    return run


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing plot generation."""
    np.random.seed(42)  # For reproducibility
    n_points = 100

    return pd.DataFrame({
        '_step': range(n_points),
        '_timestamp': range(n_points),
        'loss': np.random.random(n_points) * 2,  # Random loss values
        'accuracy': np.random.random(n_points) * 0.5 + 0.5,  # 0.5-1.0 range
        'learning_rate': [0.001] * n_points,  # Constant value
        'val_loss': np.random.random(n_points) * 2.5,
        'val_accuracy': np.random.random(n_points) * 0.4 + 0.6,
    })


@pytest.fixture
def sample_sparse_metrics_data():
    """Sample metrics data with missing values for testing."""
    np.random.seed(42)
    n_points = 100

    data = {
        '_step': range(n_points),
        'loss': np.random.random(n_points),
        'accuracy': np.random.random(n_points),
    }

    df = pd.DataFrame(data)
    # Add some NaN values
    df.loc[10:20, 'loss'] = np.nan
    df.loc[30:35, 'accuracy'] = np.nan

    return df


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    output_dir = tmp_path / "wandb_plots"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def mock_run_list():
    """Mock list of runs with various states."""
    return [
        {
            "id": "run1",
            "name": "experiment-1",
            "state": "finished",
            "created_at": "2024-01-01T00:00:00",
            "summary_metrics": {"loss": 0.234, "accuracy": 0.945}
        },
        {
            "id": "run2",
            "name": "experiment-2",
            "state": "running",
            "created_at": "2024-01-02T00:00:00",
            "summary_metrics": {"loss": 0.456, "accuracy": 0.912}
        },
        {
            "id": "run3",
            "name": "experiment-3",
            "state": "crashed",
            "created_at": "2024-01-03T00:00:00",
            "summary_metrics": {}
        }
    ]


@pytest.fixture
def mock_file():
    """Mock W&B file object."""
    file_mock = Mock()
    file_mock.name = "media/images/plot.png"
    file_mock.size = 12345

    def download_side_effect(root=None, replace=False):
        """Simulate file download."""
        if root:
            download_path = Path(root) / file_mock.name
            download_path.parent.mkdir(parents=True, exist_ok=True)
            download_path.touch()
            return str(download_path)
        return file_mock.name

    file_mock.download = Mock(side_effect=download_side_effect)
    return file_mock


@pytest.fixture
def sample_history_df():
    """Sample history dataframe as returned by run.history()."""
    np.random.seed(42)
    n_points = 500  # run.history() typically returns ~500 points

    return pd.DataFrame({
        '_step': range(n_points),
        '_timestamp': np.linspace(1704067200, 1704070800, n_points),  # Unix timestamps
        '_runtime': np.linspace(0, 3600, n_points),  # Runtime in seconds
        'epoch': np.repeat(range(10), 50),
        'train/loss': np.exp(-np.linspace(0, 3, n_points)) + np.random.random(n_points) * 0.1,
        'train/accuracy': 1 - np.exp(-np.linspace(0, 3, n_points)) * 0.5 + np.random.random(n_points) * 0.05,
        'val/loss': np.exp(-np.linspace(0, 3, n_points)) + np.random.random(n_points) * 0.15,
        'val/accuracy': 1 - np.exp(-np.linspace(0, 3, n_points)) * 0.5 + np.random.random(n_points) * 0.07,
        'learning_rate': np.linspace(0.001, 0.0001, n_points),
    })


@pytest.fixture
def mock_empty_run(mocker):
    """Mock W&B run with no metrics."""
    run = mocker.Mock()
    run.id = "empty-run-123"
    run.name = "empty-run"
    run.state = "finished"
    run.entity = "test-entity"
    run.project = "test-project"
    run.summary = {}
    run.history = mocker.Mock(return_value=pd.DataFrame())
    return run
