"""Unit tests for generate_plots module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from scripts.generate_plots import generate_plots


class TestGeneratePlots:
    """Tests for generate_plots function."""

    @patch("scripts.generate_plots.get_run")
    def test_generate_basic_plots(self, mock_get_run, sample_history_df, tmp_path):
        """Test generating plots with sampled history data."""
        run = Mock()
        run.id = "run-123"
        run.name = "test-run"
        run.entity = "test-entity"
        run.project = "test-project"
        run.history.return_value = sample_history_df
        mock_get_run.return_value = run

        output_dir = tmp_path / "out"

        generated = generate_plots(
            "test-entity/test-project",
            "run-123",
            ["train/loss", "val/accuracy"],
            output_dir=str(output_dir)
        )

        assert len(generated) == 2
        assert Path(generated[0]).exists()
        assert (output_dir / "metadata.json").exists()

        with open(output_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        assert metadata["plot_count"] == 2
        assert "train/loss" in metadata["metrics_plotted"]

    @patch("scripts.generate_plots.get_run")
    def test_generate_with_full_resolution(self, mock_get_run, sample_history_df, tmp_path):
        """Test generating plots using full resolution scan_history."""
        run = Mock()
        run.id = "run-123"
        run.name = "test-run"
        run.entity = "test-entity"
        run.project = "test-project"
        run.scan_history.return_value = sample_history_df.to_dict(orient="records")
        mock_get_run.return_value = run

        output_dir = tmp_path / "out"

        generated = generate_plots(
            "test-entity/test-project",
            "run-123",
            ["train/loss"],
            full_resolution=True,
            output_dir=str(output_dir)
        )

        assert len(generated) == 1
        assert Path(generated[0]).exists()

    @patch("scripts.generate_plots.get_run")
    def test_generate_with_smoothing(self, mock_get_run, sample_history_df, tmp_path):
        """Test generating plots with smoothing enabled."""
        run = Mock()
        run.id = "run-123"
        run.name = "test-run"
        run.entity = "test-entity"
        run.project = "test-project"
        run.history.return_value = sample_history_df
        mock_get_run.return_value = run

        output_dir = tmp_path / "out"

        generated = generate_plots(
            "test-entity/test-project",
            "run-123",
            ["train/loss"],
            smooth=5,
            output_dir=str(output_dir)
        )

        assert len(generated) == 1
        assert Path(generated[0]).exists()

    @patch("scripts.generate_plots.get_run")
    def test_generate_with_multiple_runs(self, mock_get_run, sample_history_df, tmp_path):
        """Test generating plots across multiple runs."""
        run_a = Mock()
        run_a.id = "run-aaa"
        run_a.name = "test-run-a"
        run_a.entity = "test-entity"
        run_a.project = "test-project"
        run_a.history.return_value = sample_history_df

        run_b = Mock()
        run_b.id = "run-bbb"
        run_b.name = "test-run-b"
        run_b.entity = "test-entity"
        run_b.project = "test-project"
        run_b.history.return_value = sample_history_df

        mock_get_run.side_effect = [run_a, run_b]

        output_dir = tmp_path / "out"

        generated = generate_plots(
            "test-entity/test-project",
            "run-aaa,run-bbb",
            ["train/loss"],
            output_dir=str(output_dir)
        )

        assert len(generated) == 1
        assert Path(generated[0]).exists()
        assert (output_dir / "metadata.json").exists()

        with open(output_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        assert metadata["run_ids"] == ["run-aaa", "run-bbb"]

    @patch("scripts.generate_plots.get_run")
    def test_missing_metrics_raises(self, mock_get_run, sample_history_df):
        """Test error when requested metrics are missing."""
        run = Mock()
        run.history.return_value = sample_history_df
        mock_get_run.return_value = run

        with pytest.raises(ValueError, match="Metrics not found"):
            generate_plots(
                "test-entity/test-project",
                "run-123",
                ["missing/metric"]
            )

    @patch("scripts.generate_plots.get_run")
    def test_empty_history_raises(self, mock_get_run):
        """Test error when run has no history data."""
        run = Mock()
        run.history.return_value = pd.DataFrame()
        mock_get_run.return_value = run

        with pytest.raises(ValueError, match="no history data"):
            generate_plots(
                "test-entity/test-project",
                "run-123",
                ["loss"]
            )
