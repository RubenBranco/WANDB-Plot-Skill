"""Unit tests for download_plots module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from scripts.download_plots import download_plots


class TestDownloadPlots:
    """Tests for download_plots function."""

    @patch("scripts.download_plots.get_run")
    @patch("scripts.download_plots.resolve_output_dir")
    def test_download_with_default_patterns(
        self,
        mock_resolve_output_dir,
        mock_get_run,
        mock_file,
        tmp_path
    ):
        """Test download with default pattern matching."""
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        mock_resolve_output_dir.return_value = output_dir

        run = Mock()
        run.id = "run-123"
        run.name = "test-run"
        run.entity = "test-entity"
        run.project = "test-project"

        def files_side_effect(pattern=None):
            if pattern == "media/images/*.png":
                return [mock_file]
            return []

        run.files.side_effect = files_side_effect
        mock_get_run.return_value = run

        downloaded = download_plots("test-entity/test-project", "run-123")

        assert len(downloaded) == 1
        assert Path(downloaded[0]).exists()
        assert (output_dir / "metadata.json").exists()

    @patch("scripts.download_plots.get_run")
    @patch("scripts.download_plots.resolve_output_dir")
    def test_download_with_custom_pattern(
        self,
        mock_resolve_output_dir,
        mock_get_run,
        mock_file,
        tmp_path
    ):
        """Test download with custom pattern."""
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        mock_resolve_output_dir.return_value = output_dir

        run = Mock()
        run.id = "run-123"
        run.name = "test-run"
        run.entity = "test-entity"
        run.project = "test-project"
        run.files.return_value = [mock_file]
        mock_get_run.return_value = run

        downloaded = download_plots(
            "test-entity/test-project",
            "run-123",
            pattern="custom/*.png"
        )

        assert len(downloaded) == 1
        run.files.assert_called_once_with(pattern="custom/*.png")

    @patch("scripts.download_plots.get_run")
    @patch("scripts.download_plots.resolve_output_dir")
    def test_download_no_files(
        self,
        mock_resolve_output_dir,
        mock_get_run,
        tmp_path
    ):
        """Test when no plot files are found."""
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        mock_resolve_output_dir.return_value = output_dir

        run = Mock()
        run.id = "run-123"
        run.name = "test-run"
        run.files.return_value = []
        mock_get_run.return_value = run

        downloaded = download_plots("test-entity/test-project", "run-123")

        assert downloaded == []
        assert not (output_dir / "metadata.json").exists()

    @patch("scripts.download_plots.get_run")
    @patch("scripts.download_plots.resolve_output_dir")
    def test_skip_existing_files(
        self,
        mock_resolve_output_dir,
        mock_get_run,
        mock_file,
        tmp_path
    ):
        """Test skipping already-downloaded files."""
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        mock_resolve_output_dir.return_value = output_dir

        existing_path = output_dir / "plot.png"
        existing_path.write_text("existing")

        run = Mock()
        run.id = "run-123"
        run.name = "test-run"
        run.entity = "test-entity"
        run.project = "test-project"
        run.files.return_value = [mock_file]
        mock_get_run.return_value = run

        mock_file.download.reset_mock()
        downloaded = download_plots("test-entity/test-project", "run-123")

        assert len(downloaded) == 1
        mock_file.download.assert_not_called()

    @patch("scripts.download_plots.get_run")
    @patch("scripts.download_plots.resolve_output_dir")
    def test_force_redownload(
        self,
        mock_resolve_output_dir,
        mock_get_run,
        mock_file,
        tmp_path
    ):
        """Test forcing re-download of existing files."""
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        mock_resolve_output_dir.return_value = output_dir

        existing_path = output_dir / "plot.png"
        existing_path.write_text("existing")

        run = Mock()
        run.id = "run-123"
        run.name = "test-run"
        run.entity = "test-entity"
        run.project = "test-project"
        run.files.return_value = [mock_file]
        mock_get_run.return_value = run

        mock_file.download.reset_mock()
        downloaded = download_plots(
            "test-entity/test-project",
            "run-123",
            force=True
        )

        assert len(downloaded) == 1
        mock_file.download.assert_called_once()

    @patch("scripts.download_plots.get_run")
    @patch("scripts.download_plots.resolve_output_dir")
    def test_metadata_contents(
        self,
        mock_resolve_output_dir,
        mock_get_run,
        mock_file,
        tmp_path
    ):
        """Test metadata.json content after downloads."""
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        mock_resolve_output_dir.return_value = output_dir

        run = Mock()
        run.id = "run-123"
        run.name = "test-run"
        run.entity = "test-entity"
        run.project = "test-project"
        run.files.return_value = [mock_file]
        mock_get_run.return_value = run

        downloaded = download_plots("test-entity/test-project", "run-123")

        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["run_id"] == "run-123"
        assert metadata["file_count"] == 1
        assert metadata["files_downloaded"] == [Path(downloaded[0]).name]
