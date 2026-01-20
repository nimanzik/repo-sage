"""Tests for the ingestion module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from conftest import create_fake_zip_bytes
from requests import HTTPError

from repo_sage.ingestion import chunk_document, read_repo_markdown_files

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
        mock_requests_get.assert_called_once_with(expected_url)

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


class TestChunkDocument:
    """Tests for chunk_document function."""

    def test_success_splits_on_delimiter(self, mock_genai_client: MagicMock) -> None:
        """Splits document into chunks based on --- delimiter."""
        mock_response = Mock()
        mock_response.text = "## Section 1\nFirst content\n---\n## Section 2\nSecond"
        mock_genai_client.models.generate_content.return_value = mock_response

        result = chunk_document("Some document text")

        assert len(result) == 2
        assert "Section 1" in result[0]
        assert "Section 2" in result[1]

    def test_returns_empty_list_on_empty_response(
        self, mock_genai_client: MagicMock
    ) -> None:
        """Returns empty list when LLM response has no text."""
        mock_response = Mock()
        mock_response.text = None
        mock_genai_client.models.generate_content.return_value = mock_response

        result = chunk_document("Some document text")

        assert result == []

    def test_single_chunk_without_delimiter(self, mock_genai_client: MagicMock) -> None:
        """Returns single chunk when response has no delimiter."""
        mock_response = Mock()
        mock_response.text = "## Single Section\nAll the content here"
        mock_genai_client.models.generate_content.return_value = mock_response

        result = chunk_document("Some document text")

        assert len(result) == 1
        assert "Single Section" in result[0]

    def test_strips_whitespace_from_chunks(self, mock_genai_client: MagicMock) -> None:
        """Strips leading and trailing whitespace from chunks."""
        mock_response = Mock()
        mock_response.text = "  \n## Section 1\nContent\n  \n---\n  \n## Section 2\n  "
        mock_genai_client.models.generate_content.return_value = mock_response

        result = chunk_document("Some document text")

        assert len(result) == 2
        assert result[0] == "## Section 1\nContent"
        assert result[1] == "## Section 2"

    def test_passes_model_to_client(self, mock_genai_client: MagicMock) -> None:
        """Passes model parameter to generate_content call."""
        mock_response = Mock()
        mock_response.text = "## Section\nContent"
        mock_genai_client.models.generate_content.return_value = mock_response

        chunk_document("Some document", model="gemini-pro")

        mock_genai_client.models.generate_content.assert_called_once()
        call_kwargs = mock_genai_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-pro"
