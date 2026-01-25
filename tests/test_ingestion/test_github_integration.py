"""Integration tests for the GitHub ingestion module.

These tests hit real external services (GitHub).

Requirements:
    - Network access

Run with:
    uv run pytest tests/test_ingestion/ -m integration -v
To skip them:
    uv run pytest tests/test_ingestion/ -m "not integration" -v
"""

import pytest

from git_grok.ingestion.github import read_repo_markdown_files

pytestmark = pytest.mark.integration


class TestReadRepoMarkdownFilesIntegration:
    """Integration tests for read_repo_markdown_files against real GitHub."""

    def test_downloads_and_parses_real_repo(self) -> None:
        """Downloads markdown files from a real GitHub repository."""
        result = read_repo_markdown_files("DataTalksClub", "faq")

        assert len(result) > 0
        assert all("filename" in doc for doc in result)
        assert all("content" in doc for doc in result)

    def test_downloads_with_custom_branch(self) -> None:
        """Downloads from a specific branch."""
        result = read_repo_markdown_files("evidentlyai", "evidently", branch="main")

        assert len(result) > 0

    def test_raises_error_for_nonexistent_repo(self) -> None:
        """Raises RuntimeError for a repository that does not exist."""
        with pytest.raises(RuntimeError, match="Failed to download repository"):
            read_repo_markdown_files(
                "this-owner-does-not-exist-12345",
                "this-repo-does-not-exist-67890",
            )
