"""Tests for the spliting module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

from git_grok.ingestion.splitting import llm_split

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestChunkDocument:
    """Tests for llm_split function."""

    def test_success_splits_on_delimiter(self, mock_genai_client: MagicMock) -> None:
        """Splits document into chunks based on delimiter."""
        mock_response = Mock()
        mock_response.text = (
            "## Section 1\nFirst content\n<<<SECTION_BREAK>>>\n## Section 2\nSecond"
        )
        mock_genai_client.return_value.models.generate_content.return_value = (
            mock_response
        )

        result = llm_split("Some document text")

        assert len(result) == 2
        assert "Section 1" in result[0]
        assert "Section 2" in result[1]

    def test_returns_empty_list_on_empty_response(
        self, mock_genai_client: MagicMock
    ) -> None:
        """Returns empty list when LLM response has no text."""
        mock_response = Mock()
        mock_response.text = None
        mock_genai_client.return_value.models.generate_content.return_value = (
            mock_response
        )

        result = llm_split("Some document text")

        assert result == []

    def test_single_chunk_without_delimiter(self, mock_genai_client: MagicMock) -> None:
        """Returns single chunk when response has no delimiter."""
        mock_response = Mock()
        mock_response.text = "## Single Section\nAll the content here"
        mock_genai_client.return_value.models.generate_content.return_value = (
            mock_response
        )

        result = llm_split("Some document text")

        assert len(result) == 1
        assert "Single Section" in result[0]

    def test_strips_whitespace_from_chunks(self, mock_genai_client: MagicMock) -> None:
        """Strips leading and trailing whitespace from chunks."""
        mock_response = Mock()
        mock_response.text = (
            "  \n## Section 1\nContent\n  \n<<<SECTION_BREAK>>>\n  \n## Section 2\n  "
        )
        mock_genai_client.return_value.models.generate_content.return_value = (
            mock_response
        )

        result = llm_split("Some document text")

        assert len(result) == 2
        assert result[0] == "## Section 1\nContent"
        assert result[1] == "## Section 2"

    def test_passes_model_to_client(self, mock_genai_client: MagicMock) -> None:
        """Passes model parameter to generate_content call."""
        mock_response = Mock()
        mock_response.text = "## Section\nContent"
        mock_genai_client.return_value.models.generate_content.return_value = (
            mock_response
        )

        llm_split("Some document", gemini_model_id="gemini-pro")

        mock_genai_client.return_value.models.generate_content.assert_called_once()
        call_kwargs = mock_genai_client.return_value.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-pro"
