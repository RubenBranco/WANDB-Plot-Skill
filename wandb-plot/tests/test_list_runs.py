"""Unit tests for list_runs module."""

import pytest
from unittest.mock import Mock, patch
import json

from scripts.list_runs import list_runs, format_run_table


class TestListRuns:
    """Tests for list_runs function."""

    @patch('scripts.list_runs.get_api')
    @patch('scripts.list_runs.parse_entity_project')
    def test_list_all_runs(self, mock_parse, mock_get_api):
        """Test listing all runs without filters."""
        mock_parse.return_value = ("test-entity", "test-project")

        # Create mock runs
        mock_run1 = Mock()
        mock_run1.id = "run1"
        mock_run1.name = "experiment-1"
        mock_run1.state = "finished"
        mock_run1.created_at = "2024-01-01T00:00:00"
        mock_run1.summary = {"loss": 0.234, "accuracy": 0.945}
        mock_run1.tags = ["baseline"]

        mock_run2 = Mock()
        mock_run2.id = "run2"
        mock_run2.name = "experiment-2"
        mock_run2.state = "running"
        mock_run2.created_at = "2024-01-02T00:00:00"
        mock_run2.summary = {"loss": 0.456}
        mock_run2.tags = []

        mock_api = Mock()
        mock_api.runs.return_value = [mock_run1, mock_run2]
        mock_get_api.return_value = mock_api

        runs = list_runs("test-entity/test-project")

        assert len(runs) == 2
        assert runs[0]["id"] == "run1"
        assert runs[0]["name"] == "experiment-1"
        assert runs[0]["state"] == "finished"
        assert runs[0]["summary_metrics"] == {"loss": 0.234, "accuracy": 0.945}
        assert runs[1]["id"] == "run2"

    @patch('scripts.list_runs.get_api')
    @patch('scripts.list_runs.parse_entity_project')
    def test_list_runs_with_state_filter(self, mock_parse, mock_get_api):
        """Test filtering runs by state."""
        mock_parse.return_value = ("test-entity", "test-project")

        mock_run = Mock()
        mock_run.id = "run1"
        mock_run.name = "finished-run"
        mock_run.state = "finished"
        mock_run.created_at = "2024-01-01T00:00:00"
        mock_run.summary = {}
        mock_run.tags = []

        mock_api = Mock()
        mock_api.runs.return_value = [mock_run]
        mock_get_api.return_value = mock_api

        runs = list_runs("test-entity/test-project", state="finished")

        assert len(runs) == 1
        assert runs[0]["state"] == "finished"
        mock_api.runs.assert_called_once()
        # Verify filters were passed
        call_args = mock_api.runs.call_args
        assert call_args[1]["filters"]["state"] == "finished"

    @patch('scripts.list_runs.get_api')
    @patch('scripts.list_runs.parse_entity_project')
    def test_list_runs_with_limit(self, mock_parse, mock_get_api):
        """Test limiting number of runs returned."""
        mock_parse.return_value = ("test-entity", "test-project")

        # Create 10 mock runs
        mock_runs = []
        for i in range(10):
            mock_run = Mock()
            mock_run.id = f"run{i}"
            mock_run.name = f"experiment-{i}"
            mock_run.state = "finished"
            mock_run.created_at = "2024-01-01T00:00:00"
            mock_run.summary = {}
            mock_run.tags = []
            mock_runs.append(mock_run)

        mock_api = Mock()
        mock_api.runs.return_value = mock_runs
        mock_get_api.return_value = mock_api

        runs = list_runs("test-entity/test-project", limit=5)

        assert len(runs) == 5
        assert runs[0]["id"] == "run0"
        assert runs[4]["id"] == "run4"

    @patch('scripts.list_runs.get_api')
    @patch('scripts.list_runs.parse_entity_project')
    def test_list_runs_empty_project(self, mock_parse, mock_get_api):
        """Test listing runs from empty project."""
        mock_parse.return_value = ("test-entity", "test-project")

        mock_api = Mock()
        mock_api.runs.return_value = []
        mock_get_api.return_value = mock_api

        runs = list_runs("test-entity/test-project")

        assert len(runs) == 0
        assert isinstance(runs, list)

    @patch('scripts.list_runs.get_api')
    @patch('scripts.list_runs.parse_entity_project')
    def test_list_runs_project_not_found(self, mock_parse, mock_get_api):
        """Test error when project doesn't exist."""
        mock_parse.return_value = ("test-entity", "nonexistent")

        mock_api = Mock()
        mock_api.runs.side_effect = Exception("Project not found")
        mock_get_api.return_value = mock_api

        with pytest.raises(ValueError, match="Error accessing project"):
            list_runs("test-entity/nonexistent")

    @patch('scripts.list_runs.get_api')
    @patch('scripts.list_runs.parse_entity_project')
    def test_list_runs_no_summary_metrics(self, mock_parse, mock_get_api):
        """Test handling runs with no summary metrics."""
        mock_parse.return_value = ("test-entity", "test-project")

        mock_run = Mock()
        mock_run.id = "run1"
        mock_run.name = "no-metrics"
        mock_run.state = "running"
        mock_run.created_at = "2024-01-01T00:00:00"
        mock_run.summary = None
        mock_run.tags = []

        mock_api = Mock()
        mock_api.runs.return_value = [mock_run]
        mock_get_api.return_value = mock_api

        runs = list_runs("test-entity/test-project")

        assert len(runs) == 1
        assert runs[0]["summary_metrics"] == {}

    @patch('scripts.list_runs.get_api')
    @patch('scripts.list_runs.parse_entity_project')
    def test_list_runs_with_tags(self, mock_parse, mock_get_api):
        """Test that run tags are included in output."""
        mock_parse.return_value = ("test-entity", "test-project")

        mock_run = Mock()
        mock_run.id = "run1"
        mock_run.name = "tagged-run"
        mock_run.state = "finished"
        mock_run.created_at = "2024-01-01T00:00:00"
        mock_run.summary = {}
        mock_run.tags = ["baseline", "v1"]

        mock_api = Mock()
        mock_api.runs.return_value = [mock_run]
        mock_get_api.return_value = mock_api

        runs = list_runs("test-entity/test-project")

        assert len(runs) == 1
        assert runs[0]["tags"] == ["baseline", "v1"]


class TestFormatRunTable:
    """Tests for format_run_table function."""

    def test_format_empty_list(self):
        """Test formatting empty run list."""
        result = format_run_table([])
        assert result == "No runs found."

    def test_format_single_run(self):
        """Test formatting single run."""
        runs = [{
            "id": "abc123",
            "name": "experiment-1",
            "state": "finished",
            "created_at": "2024-01-01T12:00:00.000Z",
            "summary_metrics": {"loss": 0.234, "accuracy": 0.945}
        }]

        result = format_run_table(runs)

        assert "abc123" in result
        assert "experiment-1" in result
        assert "finished" in result
        assert "2024-01-01" in result
        assert "loss" in result
        assert "accuracy" in result

    def test_format_multiple_runs(self):
        """Test formatting multiple runs."""
        runs = [
            {
                "id": "run1",
                "name": "exp-1",
                "state": "finished",
                "created_at": "2024-01-01T00:00:00",
                "summary_metrics": {"loss": 0.1}
            },
            {
                "id": "run2",
                "name": "exp-2",
                "state": "running",
                "created_at": "2024-01-02T00:00:00",
                "summary_metrics": {"loss": 0.2}
            },
            {
                "id": "run3",
                "name": "exp-3",
                "state": "crashed",
                "created_at": "2024-01-03T00:00:00",
                "summary_metrics": {}
            }
        ]

        result = format_run_table(runs)

        assert "run1" in result
        assert "run2" in result
        assert "run3" in result
        assert "Total: 3 runs" in result
        assert "finished: 1" in result or "By state:" in result

    def test_format_run_with_many_metrics(self):
        """Test formatting run with many summary metrics."""
        runs = [{
            "id": "abc123",
            "name": "many-metrics",
            "state": "finished",
            "created_at": "2024-01-01T00:00:00",
            "summary_metrics": {
                "loss": 0.1,
                "accuracy": 0.9,
                "f1": 0.85,
                "precision": 0.88,
                "recall": 0.82
            }
        }]

        result = format_run_table(runs)

        assert "loss" in result
        assert "accuracy" in result
        assert "f1" in result
        assert "+2 more" in result  # Should indicate more metrics

    def test_format_run_no_metrics(self):
        """Test formatting run with no summary metrics."""
        runs = [{
            "id": "abc123",
            "name": "no-metrics",
            "state": "running",
            "created_at": "2024-01-01T00:00:00",
            "summary_metrics": {}
        }]

        result = format_run_table(runs)

        assert "None" in result

    def test_format_includes_summary_statistics(self):
        """Test that summary statistics are included."""
        runs = [
            {
                "id": "run1",
                "name": "exp-1",
                "state": "finished",
                "created_at": "2024-01-01T00:00:00",
                "summary_metrics": {}
            },
            {
                "id": "run2",
                "name": "exp-2",
                "state": "finished",
                "created_at": "2024-01-02T00:00:00",
                "summary_metrics": {}
            },
            {
                "id": "run3",
                "name": "exp-3",
                "state": "running",
                "created_at": "2024-01-03T00:00:00",
                "summary_metrics": {}
            }
        ]

        result = format_run_table(runs)

        assert "Total: 3 runs" in result
        assert "finished: 2" in result
        assert "running: 1" in result

    def test_format_handles_missing_created_at(self):
        """Test formatting when created_at is None."""
        runs = [{
            "id": "abc123",
            "name": "no-date",
            "state": "finished",
            "created_at": None,
            "summary_metrics": {}
        }]

        result = format_run_table(runs)

        assert "N/A" in result
