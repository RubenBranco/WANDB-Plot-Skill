"""Unit tests for wandb_utils module."""

import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import wandb

from scripts.wandb_utils import (
    WandBAuthError,
    get_api,
    parse_entity_project,
    get_run,
    ensure_output_dir,
    format_entity_project,
)


class TestWandBAuthError:
    """Tests for WandBAuthError exception."""

    def test_error_message_content(self):
        """Test that error message contains helpful instructions."""
        error = WandBAuthError()
        error_msg = str(error)

        assert "authentication required" in error_msg.lower()
        assert "wandb login" in error_msg
        assert "WANDB_API_KEY" in error_msg
        assert "~/.netrc" in error_msg


class TestGetApi:
    """Tests for get_api function."""

    @patch('scripts.wandb_utils.wandb.Api')
    def test_successful_authentication_with_api_key(self, mock_api_class):
        """Test successful API initialization with API key."""
        mock_api_instance = Mock()
        mock_api_instance.viewer = {"username": "test-user"}
        mock_api_class.return_value = mock_api_instance

        api = get_api()

        assert api is not None
        mock_api_class.assert_called_once()

    @patch('scripts.wandb_utils.wandb.Api')
    def test_authentication_failure_usage_error(self, mock_api_class):
        """Test authentication failure with UsageError."""
        mock_api_class.side_effect = wandb.errors.UsageError(
            "Could not find login credentials"
        )

        with pytest.raises(WandBAuthError):
            get_api()

    @patch('scripts.wandb_utils.wandb.Api')
    def test_authentication_failure_viewer_access(self, mock_api_class):
        """Test authentication failure when accessing viewer."""
        mock_api_instance = Mock()
        mock_api_instance.viewer
        mock_api_class.return_value = mock_api_instance

        # Make viewer property raise an error
        type(mock_api_instance).viewer = property(
            lambda self: (_ for _ in ()).throw(
                wandb.errors.UsageError("Please login")
            )
        )

        with pytest.raises(WandBAuthError):
            get_api()

    @patch('scripts.wandb_utils.wandb.Api')
    def test_authentication_with_netrc(self, mock_api_class):
        """Test API initialization falls back to .netrc."""
        mock_api_instance = Mock()
        mock_api_instance.viewer = {"username": "test-user"}
        mock_api_class.return_value = mock_api_instance

        api = get_api()

        assert api.viewer["username"] == "test-user"

    @patch('scripts.wandb_utils.wandb.Api')
    def test_authentication_with_viewer_object(self, mock_api_class):
        """Test API initialization when viewer is an object."""
        mock_api_instance = Mock()
        mock_api_instance.viewer = SimpleNamespace(username="test-user")
        mock_api_class.return_value = mock_api_instance

        api = get_api()

        assert api.viewer.username == "test-user"


class TestParseEntityProject:
    """Tests for parse_entity_project function."""

    @patch('scripts.wandb_utils.get_api')
    def test_parse_with_entity_and_project(self, mock_get_api):
        """Test parsing 'entity/project' format."""
        entity, project = parse_entity_project("my-org/my-project")

        assert entity == "my-org"
        assert project == "my-project"
        # Should not need to call API when entity is provided
        mock_get_api.assert_not_called()

    @patch('scripts.wandb_utils.get_api')
    def test_parse_project_only(self, mock_get_api):
        """Test parsing 'project' format without entity."""
        mock_api = Mock()
        mock_api.viewer = {"entity": "default-entity", "username": "test-user"}
        mock_get_api.return_value = mock_api

        entity, project = parse_entity_project("my-project")

        assert entity == "default-entity"
        assert project == "my-project"
        mock_get_api.assert_called_once()

    @patch('scripts.wandb_utils.get_api')
    def test_parse_project_only_with_username_fallback(self, mock_get_api):
        """Test parsing with username fallback when entity not in viewer."""
        mock_api = Mock()
        mock_api.viewer = {"username": "test-user"}  # No entity field
        mock_get_api.return_value = mock_api

        entity, project = parse_entity_project("my-project")

        assert entity == "test-user"
        assert project == "my-project"

    @patch('scripts.wandb_utils.get_api')
    def test_parse_project_only_with_viewer_object(self, mock_get_api):
        """Test parsing with viewer object instead of dict."""
        mock_api = Mock()
        mock_api.viewer = SimpleNamespace(entity="default-entity", username="test-user")
        mock_get_api.return_value = mock_api

        entity, project = parse_entity_project("my-project")

        assert entity == "default-entity"
        assert project == "my-project"

    def test_parse_empty_string(self):
        """Test parsing empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_entity_project("")

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_entity_project("   ")

    def test_parse_invalid_format_multiple_slashes(self):
        """Test parsing string with multiple slashes raises ValueError."""
        with pytest.raises(ValueError, match="Invalid format"):
            parse_entity_project("org/project/extra")

    def test_parse_empty_entity(self):
        """Test parsing with empty entity raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_entity_project("/project")

    def test_parse_empty_project(self):
        """Test parsing with empty project raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_entity_project("entity/")

    def test_parse_strips_whitespace(self):
        """Test that whitespace is stripped from entity and project."""
        entity, project = parse_entity_project("  my-org  /  my-project  ")

        assert entity == "my-org"
        assert project == "my-project"


class TestGetRun:
    """Tests for get_run function."""

    @patch('scripts.wandb_utils.get_api')
    @patch('scripts.wandb_utils.parse_entity_project')
    def test_get_run_by_id(self, mock_parse, mock_get_api):
        """Test retrieving run by ID."""
        mock_parse.return_value = ("test-entity", "test-project")
        mock_api = Mock()
        mock_run = Mock()
        mock_run.id = "abc123"
        mock_run.name = "test-run"
        mock_api.run.return_value = mock_run
        mock_get_api.return_value = mock_api

        run = get_run("test-entity/test-project", "abc123")

        assert run.id == "abc123"
        mock_api.run.assert_called_once_with("test-entity/test-project/abc123")

    @patch('scripts.wandb_utils.get_api')
    @patch('scripts.wandb_utils.parse_entity_project')
    def test_get_run_by_name(self, mock_parse, mock_get_api):
        """Test retrieving run by name."""
        mock_parse.return_value = ("test-entity", "test-project")
        mock_api = Mock()
        mock_run = Mock()
        mock_run.name = "experiment-1"
        mock_api.run.return_value = mock_run
        mock_get_api.return_value = mock_api

        run = get_run("test-entity/test-project", "experiment-1")

        assert run.name == "experiment-1"
        mock_api.run.assert_called_once_with("test-entity/test-project/experiment-1")

    @patch('scripts.wandb_utils.get_api')
    @patch('scripts.wandb_utils.parse_entity_project')
    def test_get_run_not_found(self, mock_parse, mock_get_api):
        """Test error when run is not found."""
        mock_parse.return_value = ("test-entity", "test-project")
        mock_api = Mock()
        mock_api.run.side_effect = wandb.errors.CommError("Run not found")
        mock_get_api.return_value = mock_api

        with pytest.raises(ValueError, match="not found"):
            get_run("test-entity/test-project", "nonexistent")

    def test_get_run_empty_id(self):
        """Test error when run_id is empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_run("test-entity/test-project", "")

    def test_get_run_whitespace_id(self):
        """Test error when run_id is whitespace only."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_run("test-entity/test-project", "   ")

    @patch('scripts.wandb_utils.get_api')
    @patch('scripts.wandb_utils.parse_entity_project')
    def test_get_run_strips_whitespace(self, mock_parse, mock_get_api):
        """Test that whitespace is stripped from run_id."""
        mock_parse.return_value = ("test-entity", "test-project")
        mock_api = Mock()
        mock_run = Mock()
        mock_api.run.return_value = mock_run
        mock_get_api.return_value = mock_api

        get_run("test-entity/test-project", "  abc123  ")

        mock_api.run.assert_called_once_with("test-entity/test-project/abc123")


class TestEnsureOutputDir:
    """Tests for ensure_output_dir function."""

    @patch('scripts.wandb_utils.parse_entity_project')
    def test_create_output_dir_with_run_name(self, mock_parse, tmp_path):
        """Test creating output directory with run name."""
        mock_parse.return_value = ("test-entity", "test-project")

        output_dir = ensure_output_dir(
            "test-entity/test-project",
            "abc123",
            run_name="experiment-1",
            base_dir=str(tmp_path)
        )

        expected_path = tmp_path / "test-entity_test-project" / "experiment-1_abc123"
        assert output_dir == expected_path
        assert output_dir.exists()
        assert output_dir.is_dir()

    @patch('scripts.wandb_utils.parse_entity_project')
    def test_create_output_dir_without_run_name(self, mock_parse, tmp_path):
        """Test creating output directory without run name."""
        mock_parse.return_value = ("test-entity", "test-project")

        output_dir = ensure_output_dir(
            "test-entity/test-project",
            "abc123",
            base_dir=str(tmp_path)
        )

        expected_path = tmp_path / "test-entity_test-project" / "abc123"
        assert output_dir == expected_path
        assert output_dir.exists()

    @patch('scripts.wandb_utils.parse_entity_project')
    def test_create_output_dir_already_exists(self, mock_parse, tmp_path):
        """Test that existing directory doesn't raise error."""
        mock_parse.return_value = ("test-entity", "test-project")

        # Create directory first time
        output_dir1 = ensure_output_dir(
            "test-entity/test-project",
            "abc123",
            base_dir=str(tmp_path)
        )

        # Create same directory again
        output_dir2 = ensure_output_dir(
            "test-entity/test-project",
            "abc123",
            base_dir=str(tmp_path)
        )

        assert output_dir1 == output_dir2
        assert output_dir2.exists()

    @patch('scripts.wandb_utils.parse_entity_project')
    def test_create_nested_directories(self, mock_parse, tmp_path):
        """Test that parent directories are created."""
        mock_parse.return_value = ("test-entity", "test-project")

        output_dir = ensure_output_dir(
            "test-entity/test-project",
            "abc123",
            run_name="nested-run",
            base_dir=str(tmp_path / "deep" / "nested" / "path")
        )

        assert output_dir.exists()
        assert output_dir.parent.exists()
        assert output_dir.parent.parent.exists()

    @patch('scripts.wandb_utils.parse_entity_project')
    def test_default_base_dir(self, mock_parse, tmp_path, monkeypatch):
        """Test that default base_dir is 'wandb_plots'."""
        mock_parse.return_value = ("test-entity", "test-project")

        # Change to temp directory for test
        monkeypatch.chdir(tmp_path)

        output_dir = ensure_output_dir(
            "test-entity/test-project",
            "abc123"
        )

        assert "wandb_plots" in str(output_dir)
        assert output_dir.exists()


class TestFormatEntityProject:
    """Tests for format_entity_project function."""

    def test_format_basic(self):
        """Test basic entity/project formatting."""
        result = format_entity_project("my-entity", "my-project")
        assert result == "my-entity/my-project"

    def test_format_with_special_characters(self):
        """Test formatting with special characters."""
        result = format_entity_project("my_entity-123", "my.project.name")
        assert result == "my_entity-123/my.project.name"

    def test_format_preserves_input(self):
        """Test that input strings are not modified."""
        entity = "Test-Entity"
        project = "Test-Project"
        result = format_entity_project(entity, project)

        assert result == "Test-Entity/Test-Project"
