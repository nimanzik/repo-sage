"""Integration tests for the ingestion module.

These tests hit real external services (GitHub, Google Gemini).

Requirements:
    - Network access
    - GEMINI_API_KEY environment variable set (for Gemini tests)

Run with:
    uv run pytest tests/integration/ -m integration -v
To skip them:
    uv run pytest tests/integration/ -m "not integration" -v
"""

import os

import pytest

from repo_sage.ingestion import chunk_document, read_repo_markdown_files

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


class TestChunkDocumentIntegration:
    """Integration tests for chunk_document against real Gemini API."""

    @pytest.fixture
    def sample_document(self) -> str:
        """Sample document for chunking tests."""
        return """
Artificial Intelligence (AI) is a branch of computer science that aims to create
machines capable of intelligent behaviour. It encompasses various subfields,
including machine learning (ML), natural language processing (NLP), and robotics.

Machine Learning is a subset of AI that focuses on developing algorithms that
allow computers to learn from and make predictions based on data. Common ML
techniques include supervised learning, unsupervised learning, and reinforcement
learning.

Natural Language Processing enables machines to understand and interpret human
language, facilitating better human-computer interactions. Applications include
chatbots, translation services, and sentiment analysis.
        """

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable not set",
    )
    def test_chunks_document_with_real_llm(self, sample_document: str) -> None:
        """Chunks a document using the real Gemini API."""
        result = chunk_document(sample_document)

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)
        assert all(len(chunk) > 0 for chunk in result)

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable not set",
    )
    def test_chunks_preserve_content(self, sample_document: str) -> None:
        """Chunks should contain text from the original document."""
        result = chunk_document(sample_document)

        # At least some key terms should appear in the chunks
        all_chunks_text = " ".join(result)
        assert "AI" in all_chunks_text or "Artificial Intelligence" in all_chunks_text
