"""Tests for the GitHub ingestion module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from conftest import create_fake_zip_bytes
from requests import HTTPError

from git_grok.ingestion.github import read_repo_markdown_files

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestReadRepoMarkdownFiles:
    """Tests for read_repo_markdown_files function."""

    def test_success_parses_markdown_files(self, mock_requests_get: MagicMock) -> None:
        """Parses markdown files from ZIP with frontmatter and content."""
        files = {
            "repo-main/docs/readme.md": "---\ntitle: Hello\n---\nContent here",
            "repo-main/guide.md": "---\nauthor: Test\n---\nGuide content",
        }
        mock_response = Mock()
        mock_response.content = create_fake_zip_bytes(files)
        mock_requests_get.return_value = mock_response

        result = read_repo_markdown_files("owner", "repo")

        assert len(result) == 2
        readme_doc = next(
            doc for doc in result if doc["filename"] == "repo-main/docs/readme.md"
        )
        assert readme_doc["title"] == "Hello"
        assert readme_doc["content"] == "Content here"

    def test_raises_error_on_http_failure(self, mock_requests_get: MagicMock) -> None:
        """Raises RuntimeError when HTTP request fails."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_requests_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="Failed to download repository"):
            read_repo_markdown_files("owner", "repo")

    def test_returns_empty_list_for_repo_without_markdown(
        self, mock_requests_get: MagicMock
    ) -> None:
        """Returns empty list when ZIP contains no markdown files."""
        files = {
            "repo-main/readme.txt": "Just a text file",
            "repo-main/config.json": '{"key": "value"}',
        }
        mock_response = Mock()
        mock_response.content = create_fake_zip_bytes(files)
        mock_requests_get.return_value = mock_response

        result = read_repo_markdown_files("owner", "repo")

        assert result == []

    def test_uses_custom_branch_in_url(self, mock_requests_get: MagicMock) -> None:
        """Includes custom branch name in GitHub URL."""
        files = {"repo-develop/readme.md": "---\n---\nContent"}
        mock_response = Mock()
        mock_response.content = create_fake_zip_bytes(files)
        mock_requests_get.return_value = mock_response

        read_repo_markdown_files("owner", "repo", branch="develop")

        expected_url = "https://codeload.github.com/owner/repo/zip/refs/heads/develop"
        mock_requests_get.assert_called_once_with(expected_url, timeout=30)

    def test_includes_mdx_files(self, mock_requests_get: MagicMock) -> None:
        """Processes both .md and .mdx file extensions."""
        files = {
            "repo-main/doc.md": "---\ntype: md\n---\nMarkdown content",
            "repo-main/component.mdx": "---\ntype: mdx\n---\nMDX content",
        }
        mock_response = Mock()
        mock_response.content = create_fake_zip_bytes(files)
        mock_requests_get.return_value = mock_response

        result = read_repo_markdown_files("owner", "repo")

        assert len(result) == 2
        types = {doc["type"] for doc in result}
        assert types == {"md", "mdx"}

    def test_handles_invalid_utf8(self, mock_requests_get: MagicMock) -> None:
        """Handles files with invalid UTF-8 bytes without crashing."""
        buffer = b"---\ntitle: Test\n---\nContent with invalid bytes: \xff\xfe"
        files = {"repo-main/doc.md": buffer}

        # Create ZIP with raw bytes
        import io
        import zipfile

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)

        mock_response = Mock()
        mock_response.content = zip_buffer.getvalue()
        mock_requests_get.return_value = mock_response

        result = read_repo_markdown_files("owner", "repo")

        assert len(result) == 1
        assert result[0]["title"] == "Test"
