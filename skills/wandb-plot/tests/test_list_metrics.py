"""Unit tests for list_metrics module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from scripts.list_metrics import list_metrics, format_metrics_table


class TestListMetrics:
    """Tests for list_metrics function."""

    @patch('scripts.list_metrics.get_run')
    def test_list_basic_metrics(self, mock_get_run, sample_metrics_data):
        """Test listing basic numeric metrics."""
        mock_run = Mock()
        mock_run.history.return_value = sample_metrics_data
        mock_get_run.return_value = mock_run

        metrics = list_metrics("test-entity/test-project", "abc123")

        # Should have filtered out system columns
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "learning_rate" in metrics
        assert "_step" not in metrics
        assert "_timestamp" not in metrics

    @patch('scripts.list_metrics.get_run')
    def test_list_metrics_with_system_columns(self, mock_get_run, sample_metrics_data):
        """Test including system columns."""
        mock_run = Mock()
        mock_run.history.return_value = sample_metrics_data
        mock_get_run.return_value = mock_run

        metrics = list_metrics(
            "test-entity/test-project",
            "abc123",
            include_system=True
        )

        # Should include system columns
        assert "_step" in metrics
        assert "_timestamp" in metrics
        assert "loss" in metrics

    @patch('scripts.list_metrics.get_run')
    def test_metric_statistics(self, mock_get_run):
        """Test that statistics are calculated correctly."""
        # Create simple test data
        test_df = pd.DataFrame({
            'metric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'metric2': [10, 20, 30, 40, 50]
        })

        mock_run = Mock()
        mock_run.history.return_value = test_df
        mock_get_run.return_value = mock_run

        metrics = list_metrics("test-entity/test-project", "abc123")

        # Check metric1 statistics
        assert metrics["metric1"]["min"] == 1.0
        assert metrics["metric1"]["max"] == 5.0
        assert metrics["metric1"]["mean"] == 3.0
        assert metrics["metric1"]["count"] == 5
        assert metrics["metric1"]["non_null_count"] == 5
        assert metrics["metric1"]["type"] == "numeric"

    @patch('scripts.list_metrics.get_run')
    def test_empty_history(self, mock_get_run):
        """Test handling of empty history."""
        mock_run = Mock()
        mock_run.history.return_value = pd.DataFrame()
        mock_get_run.return_value = mock_run

        metrics = list_metrics("test-entity/test-project", "abc123")

        assert metrics == {}

    @patch('scripts.list_metrics.get_run')
    def test_metrics_with_missing_values(self, mock_get_run):
        """Test handling metrics with NaN values."""
        test_df = pd.DataFrame({
            'metric1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'metric2': [10, np.nan, np.nan, 40, 50]
        })

        mock_run = Mock()
        mock_run.history.return_value = test_df
        mock_get_run.return_value = mock_run

        metrics = list_metrics("test-entity/test-project", "abc123")

        assert metrics["metric1"]["count"] == 5
        assert metrics["metric1"]["non_null_count"] == 4
        assert metrics["metric2"]["non_null_count"] == 3

    @patch('scripts.list_metrics.get_run')
    def test_skip_all_null_columns(self, mock_get_run):
        """Test that columns with all NaN values are skipped."""
        test_df = pd.DataFrame({
            'metric1': [1.0, 2.0, 3.0],
            'all_null': [np.nan, np.nan, np.nan]
        })

        mock_run = Mock()
        mock_run.history.return_value = test_df
        mock_get_run.return_value = mock_run

        metrics = list_metrics("test-entity/test-project", "abc123")

        assert "metric1" in metrics
        assert "all_null" not in metrics

    @patch('scripts.list_metrics.get_run')
    def test_string_metrics(self, mock_get_run):
        """Test handling of string-type metrics."""
        test_df = pd.DataFrame({
            'status': ['running', 'finished', 'running'],
            'model_name': ['model_v1', 'model_v2', 'model_v1']
        })

        mock_run = Mock()
        mock_run.history.return_value = test_df
        mock_get_run.return_value = mock_run

        metrics = list_metrics("test-entity/test-project", "abc123")

        assert metrics["status"]["type"] == "string"
        assert metrics["model_name"]["type"] == "string"
        assert "min" not in metrics["status"]  # String metrics shouldn't have min/max

    @patch('scripts.list_metrics.get_run')
    def test_boolean_metrics(self, mock_get_run):
        """Test handling of boolean-type metrics."""
        test_df = pd.DataFrame({
            'is_best': [True, False, True, False],
            'converged': [False, False, True, True]
        })

        mock_run = Mock()
        mock_run.history.return_value = test_df
        mock_get_run.return_value = mock_run

        metrics = list_metrics("test-entity/test-project", "abc123")

        # In pandas, boolean dtype is often treated as numeric
        # Both "boolean" and "numeric" are acceptable
        assert metrics["is_best"]["type"] in ["boolean", "numeric"]
        assert metrics["converged"]["type"] in ["boolean", "numeric"]

    @patch('scripts.list_metrics.get_run')
    def test_filter_system_prefixes(self, mock_get_run):
        """Test filtering of various system prefixes."""
        test_df = pd.DataFrame({
            '_step': [1, 2, 3],
            '_timestamp': [100, 200, 300],
            'system/cpu': [50, 60, 70],
            'gradients/layer1': [0.1, 0.2, 0.3],
            'user_metric': [1.0, 2.0, 3.0]
        })

        mock_run = Mock()
        mock_run.history.return_value = test_df
        mock_get_run.return_value = mock_run

        metrics = list_metrics("test-entity/test-project", "abc123")

        # Only user_metric should be included
        assert "user_metric" in metrics
        assert "_step" not in metrics
        assert "_timestamp" not in metrics
        assert "system/cpu" not in metrics
        assert "gradients/layer1" not in metrics

    @patch('scripts.list_metrics.get_run')
    def test_history_fetch_error(self, mock_get_run):
        """Test error handling when fetching history fails."""
        mock_run = Mock()
        mock_run.history.side_effect = Exception("Network error")
        mock_get_run.return_value = mock_run

        with pytest.raises(ValueError, match="Error fetching run history"):
            list_metrics("test-entity/test-project", "abc123")


class TestFormatMetricsTable:
    """Tests for format_metrics_table function."""

    def test_format_empty_metrics(self):
        """Test formatting empty metrics dict."""
        result = format_metrics_table({})
        assert result == "No metrics found."

    def test_format_single_metric(self):
        """Test formatting single metric."""
        metrics = {
            "loss": {
                "type": "numeric",
                "count": 100,
                "non_null_count": 100,
                "min": 0.1,
                "max": 2.5,
                "mean": 0.8
            }
        }

        result = format_metrics_table(metrics)

        assert "loss" in result
        assert "numeric" in result
        assert "100" in result
        assert "0.1" in result
        assert "2.5" in result

    def test_format_multiple_metrics(self):
        """Test formatting multiple metrics."""
        metrics = {
            "loss": {
                "type": "numeric",
                "count": 100,
                "non_null_count": 100,
                "min": 0.1,
                "max": 2.5,
                "mean": 0.8
            },
            "accuracy": {
                "type": "numeric",
                "count": 100,
                "non_null_count": 98,
                "min": 0.5,
                "max": 0.95,
                "mean": 0.85
            }
        }

        result = format_metrics_table(metrics)

        assert "loss" in result
        assert "accuracy" in result
        assert "Total metrics: 2" in result

    def test_format_non_numeric_metric(self):
        """Test formatting non-numeric metrics."""
        metrics = {
            "status": {
                "type": "string",
                "count": 50,
                "non_null_count": 50
            }
        }

        result = format_metrics_table(metrics)

        assert "status" in result
        assert "string" in result
        assert "N/A" in result  # No min/max for string

    def test_format_includes_summary(self):
        """Test that summary statistics are included."""
        metrics = {
            "metric1": {
                "type": "numeric",
                "count": 100,
                "non_null_count": 100,
                "min": 0,
                "max": 1
            },
            "metric2": {
                "type": "numeric",
                "count": 100,
                "non_null_count": 100,
                "min": 0,
                "max": 1
            },
            "metric3": {
                "type": "string",
                "count": 100,
                "non_null_count": 100
            }
        }

        result = format_metrics_table(metrics)

        assert "Total metrics: 3" in result
        assert "By type:" in result
        assert "numeric: 2" in result
        assert "string: 1" in result

    def test_format_sorted_by_name(self):
        """Test that metrics are sorted alphabetically."""
        metrics = {
            "zebra": {"type": "numeric", "count": 1, "non_null_count": 1, "min": 0, "max": 1},
            "apple": {"type": "numeric", "count": 1, "non_null_count": 1, "min": 0, "max": 1},
            "banana": {"type": "numeric", "count": 1, "non_null_count": 1, "min": 0, "max": 1}
        }

        result = format_metrics_table(metrics)

        # Find positions of metric names in output
        apple_pos = result.find("apple")
        banana_pos = result.find("banana")
        zebra_pos = result.find("zebra")

        # Should be in alphabetical order
        assert apple_pos < banana_pos < zebra_pos

    def test_format_with_sparse_data(self):
        """Test formatting metrics with different null counts."""
        metrics = {
            "sparse_metric": {
                "type": "numeric",
                "count": 1000,
                "non_null_count": 50,
                "min": 0.1,
                "max": 1.0,
                "mean": 0.5
            }
        }

        result = format_metrics_table(metrics)

        assert "sparse_metric" in result
        assert "50" in result  # Non-null count should be shown
