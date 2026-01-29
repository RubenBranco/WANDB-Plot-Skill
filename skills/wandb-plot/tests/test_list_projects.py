"""Unit tests for list_projects module."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts.list_projects import list_projects, format_project_table


class TestListProjects:
    """Tests for list_projects function."""

    @patch("scripts.list_projects.get_api")
    def test_list_projects_default_entity_dict(self, mock_get_api):
        """Uses viewer dict to pick default entity."""
        mock_api = Mock()
        mock_api.viewer = {"entity": "test-entity", "username": "test-user"}

        mock_project = Mock()
        mock_project.name = "project-a"
        mock_project.description = "Test project"
        mock_project.created_at = "2024-01-01T00:00:00"

        mock_api.projects.return_value = [mock_project]
        mock_get_api.return_value = mock_api

        projects = list_projects()

        assert len(projects) == 1
        assert projects[0]["name"] == "project-a"
        assert projects[0]["entity"] == "test-entity"
        assert projects[0]["url"] == "https://wandb.ai/test-entity/project-a"

    @patch("scripts.list_projects.get_api")
    def test_list_projects_default_entity_object(self, mock_get_api):
        """Uses viewer object attributes when viewer is not a dict."""
        mock_api = Mock()
        mock_api.viewer = SimpleNamespace(entity="org-entity", username="user-1")

        mock_project = Mock()
        mock_project.name = "project-b"
        mock_project.description = None
        mock_project.created_at = None

        mock_api.projects.return_value = [mock_project]
        mock_get_api.return_value = mock_api

        projects = list_projects()

        assert projects[0]["entity"] == "org-entity"
        assert projects[0]["description"] is None
        assert projects[0]["created_at"] is None

    @patch("scripts.list_projects.get_api")
    def test_list_projects_with_entity_param(self, mock_get_api):
        """Passes explicit entity to API."""
        mock_api = Mock()
        mock_api.viewer = {"entity": "ignored"}
        mock_api.projects.return_value = []
        mock_get_api.return_value = mock_api

        list_projects(entity="my-org")

        mock_api.projects.assert_called_once_with("my-org")

    @patch("scripts.list_projects.get_api")
    def test_list_projects_limit(self, mock_get_api):
        """Limits number of projects returned."""
        mock_api = Mock()
        mock_api.viewer = {"entity": "test-entity"}

        projects = []
        for i in range(3):
            mock_project = Mock()
            mock_project.name = f"project-{i}"
            mock_project.description = None
            mock_project.created_at = None
            projects.append(mock_project)

        mock_api.projects.return_value = projects
        mock_get_api.return_value = mock_api

        result = list_projects(limit=2)

        assert len(result) == 2
        assert result[0]["name"] == "project-0"
        assert result[1]["name"] == "project-1"

    @patch("scripts.list_projects.get_api")
    def test_list_projects_error(self, mock_get_api):
        """Raises ValueError when API fails."""
        mock_api = Mock()
        mock_api.viewer = {"entity": "test-entity"}
        mock_api.projects.side_effect = Exception("API error")
        mock_get_api.return_value = mock_api

        with pytest.raises(ValueError, match="Error accessing projects"):
            list_projects()

    @patch("scripts.list_projects.get_api")
    def test_list_projects_missing_entity(self, mock_get_api):
        """Raises ValueError when entity cannot be determined."""
        mock_api = Mock()
        mock_api.viewer = SimpleNamespace(entity=None, username=None)
        mock_get_api.return_value = mock_api

        with pytest.raises(ValueError, match="Could not determine default entity"):
            list_projects()


class TestFormatProjectTable:
    """Tests for format_project_table function."""

    def test_format_empty_list(self):
        """Formats empty project list."""
        result = format_project_table([])
        assert result == "No projects found."

    def test_format_single_project(self):
        """Formats single project with description."""
        projects = [{
            "name": "project-a",
            "entity": "test-entity",
            "description": "A test project",
            "created_at": "2024-01-01T12:00:00",
            "url": "https://wandb.ai/test-entity/project-a",
        }]

        result = format_project_table(projects)

        assert "project-a" in result
        assert "2024-01-01" in result
        assert "A test project" in result

    def test_format_truncates_description(self):
        """Truncates long descriptions for table output."""
        long_description = "x" * 200
        projects = [{
            "name": "project-long",
            "entity": "test-entity",
            "description": long_description,
            "created_at": "2024-01-01T12:00:00",
            "url": "https://wandb.ai/test-entity/project-long",
        }]

        result = format_project_table(projects)

        assert "project-long" in result
        assert "..." in result
