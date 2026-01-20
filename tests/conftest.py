"""Shared pytest fixtures for repo_sage tests."""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def create_fake_zip_bytes(files: dict[str, str]) -> bytes:
    """
    Create an in-memory ZIP archive.

    Parameters
    ----------
    files : dict[str, str]
        Mapping of filename to file content.

    Returns
    -------
    bytes
        ZIP archive as bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture
def mock_requests_get(mocker: MockerFixture) -> MockerFixture:
    """Mock requests.get for GitHub ZIP downloads."""
    return mocker.patch("repo_sage.ingestion.requests.get")


@pytest.fixture
def mock_genai_client(mocker: MockerFixture) -> MockerFixture:
    """Mock the genai client for LLM calls."""
    return mocker.patch("repo_sage.ingestion.client")
