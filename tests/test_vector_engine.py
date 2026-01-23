"""Tests for the vector_engine module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from repo_sage.defaults import DEFAULT_EMBEDDING_MODEL_ID
from repo_sage.vector_engine import (
    QdrantEngine,
    _get_embedding_model,
    _get_qdrant_client,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestGetEmbeddingModel:
    """Tests for _get_embedding_model function."""

    def test_returns_sentence_transformer(self, mocker: MockerFixture) -> None:
        """Returns a SentenceTransformer instance."""
        mock_transformer = mocker.patch("repo_sage.vector_engine.SentenceTransformer")
        mock_model = mocker.Mock()
        mock_transformer.return_value = mock_model

        # Clear cache to ensure fresh call
        _get_embedding_model.cache_clear()

        result = _get_embedding_model("test-model-id")

        mock_transformer.assert_called_once_with(
            "test-model-id", model_kwargs={"torch_dtype": "float16"}
        )
        assert result == mock_model

    def test_caches_model_by_id(self, mocker: MockerFixture) -> None:
        """Caches model instances by model ID."""
        mock_transformer = mocker.patch("repo_sage.vector_engine.SentenceTransformer")
        mock_model = mocker.Mock()
        mock_transformer.return_value = mock_model

        _get_embedding_model.cache_clear()

        # Call twice with same ID
        result1 = _get_embedding_model("cached-model")
        result2 = _get_embedding_model("cached-model")

        # Should only create one instance
        assert mock_transformer.call_count == 1
        assert result1 is result2

    def test_creates_different_instances_for_different_ids(
        self, mocker: MockerFixture
    ) -> None:
        """Creates separate instances for different model IDs."""
        mock_transformer = mocker.patch("repo_sage.vector_engine.SentenceTransformer")
        mock_model_a = mocker.Mock()
        mock_model_b = mocker.Mock()
        mock_transformer.side_effect = [mock_model_a, mock_model_b]

        _get_embedding_model.cache_clear()

        result1 = _get_embedding_model("model-a")
        result2 = _get_embedding_model("model-b")

        assert mock_transformer.call_count == 2
        assert result1 == mock_model_a
        assert result2 == mock_model_b


class TestGetQdrantClient:
    """Tests for _get_qdrant_client function."""

    def test_returns_qdrant_client(self, mocker: MockerFixture) -> None:
        """Returns a QdrantClient instance with given path."""
        mock_client_class = mocker.patch("repo_sage.vector_engine.QdrantClient")
        mock_client = mocker.Mock()
        mock_client_class.return_value = mock_client

        _get_qdrant_client.cache_clear()

        result = _get_qdrant_client("/path/to/db")

        mock_client_class.assert_called_once_with(path="/path/to/db")
        assert result == mock_client

    def test_caches_client_by_path(self, mocker: MockerFixture) -> None:
        """Caches client instances by path."""
        mock_client_class = mocker.patch("repo_sage.vector_engine.QdrantClient")
        mock_client = mocker.Mock()
        mock_client_class.return_value = mock_client

        _get_qdrant_client.cache_clear()

        result1 = _get_qdrant_client("/cached/path")
        result2 = _get_qdrant_client("/cached/path")

        assert mock_client_class.call_count == 1
        assert result1 is result2

    def test_creates_different_clients_for_different_paths(
        self, mocker: MockerFixture
    ) -> None:
        """Creates separate clients for different paths."""
        mock_client_class = mocker.patch("repo_sage.vector_engine.QdrantClient")
        mock_client_a = mocker.Mock()
        mock_client_b = mocker.Mock()
        mock_client_class.side_effect = [mock_client_a, mock_client_b]

        _get_qdrant_client.cache_clear()

        result1 = _get_qdrant_client("/path/a")
        result2 = _get_qdrant_client("/path/b")

        assert mock_client_class.call_count == 2
        assert result1 == mock_client_a
        assert result2 == mock_client_b


class TestQdrantEngineInit:
    """Tests for QdrantEngine.__init__."""

    def test_initialises_with_required_params(self) -> None:
        """Initialises with collection name and path."""
        engine = QdrantEngine(collection_name="test-collection", path="/db/path")

        assert engine.collection_name == "test-collection"
        assert engine.path == "/db/path"
        assert engine.model_id == DEFAULT_EMBEDDING_MODEL_ID
        assert engine._client is None
        assert engine._embedding_model is None
        assert engine._collection_created is False

    def test_accepts_custom_model_id(self) -> None:
        """Accepts custom model ID parameter."""
        engine = QdrantEngine(
            collection_name="test",
            path="/db",
            model_id="custom-model-id",
        )

        assert engine.model_id == "custom-model-id"

    def test_uses_default_model_id_when_none(self) -> None:
        """Uses default model ID when None is provided."""
        engine = QdrantEngine(
            collection_name="test",
            path="/db",
            model_id=None,
        )

        assert engine.model_id == DEFAULT_EMBEDDING_MODEL_ID


class TestQdrantEngineClientProperty:
    """Tests for QdrantEngine.client property."""

    def test_creates_client_on_first_access(self, mocker: MockerFixture) -> None:
        """Creates client lazily on first access."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_client = mocker.Mock()
        mock_get_client.return_value = mock_client

        engine = QdrantEngine(collection_name="test", path="/db/path")

        result = engine.client

        mock_get_client.assert_called_once_with("/db/path")
        assert result == mock_client

    def test_returns_cached_client_on_subsequent_access(
        self, mocker: MockerFixture
    ) -> None:
        """Returns cached client on subsequent accesses."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_client = mocker.Mock()
        mock_get_client.return_value = mock_client

        engine = QdrantEngine(collection_name="test", path="/db/path")

        _ = engine.client
        result = engine.client

        assert mock_get_client.call_count == 1
        assert result == mock_client


class TestQdrantEngineEmbeddingModelProperty:
    """Tests for QdrantEngine.embedding_model property."""

    def test_creates_model_on_first_access(self, mocker: MockerFixture) -> None:
        """Creates embedding model lazily on first access."""
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")
        mock_model = mocker.Mock()
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(
            collection_name="test",
            path="/db",
            model_id="my-model",
        )

        result = engine.embedding_model

        mock_get_model.assert_called_once_with("my-model")
        assert result == mock_model

    def test_returns_cached_model_on_subsequent_access(
        self, mocker: MockerFixture
    ) -> None:
        """Returns cached model on subsequent accesses."""
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")
        mock_model = mocker.Mock()
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db", model_id="my-model")

        _ = engine.embedding_model
        result = engine.embedding_model

        assert mock_get_model.call_count == 1
        assert result == mock_model


class TestQdrantEngineEnsureCollection:
    """Tests for QdrantEngine._ensure_collection."""

    def test_creates_collection_when_not_exists(self, mocker: MockerFixture) -> None:
        """Creates collection when it does not exist."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_client = mocker.Mock()
        mock_client.collection_exists.return_value = False
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="new-collection", path="/db")

        engine._ensure_collection()

        mock_client.collection_exists.assert_called_once_with("new-collection")
        mock_client.create_collection.assert_called_once()

        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "new-collection"

    def test_skips_creation_when_collection_exists(self, mocker: MockerFixture) -> None:
        """Skips creation when collection already exists."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")

        mock_client = mocker.Mock()
        mock_client.collection_exists.return_value = True
        mock_get_client.return_value = mock_client

        engine = QdrantEngine(collection_name="existing", path="/db")

        engine._ensure_collection()

        mock_client.collection_exists.assert_called_once_with("existing")
        mock_client.create_collection.assert_not_called()

    def test_skips_check_when_already_created(self, mocker: MockerFixture) -> None:
        """Skips existence check when collection already created in session."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")

        mock_client = mocker.Mock()
        mock_get_client.return_value = mock_client

        engine = QdrantEngine(collection_name="test", path="/db")
        engine._collection_created = True

        engine._ensure_collection()

        mock_client.collection_exists.assert_not_called()

    def test_raises_error_for_unknown_embedding_size(
        self, mocker: MockerFixture
    ) -> None:
        """Raises ValueError when embedding dimension is unknown."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_client = mocker.Mock()
        mock_client.collection_exists.return_value = False
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_model.get_sentence_embedding_dimension.return_value = None
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(
            collection_name="test", path="/db", model_id="unknown-model"
        )

        with pytest.raises(ValueError, match="Unknown embedding size"):
            engine._ensure_collection()

    def test_sets_collection_created_flag(self, mocker: MockerFixture) -> None:
        """Sets _collection_created flag after ensuring collection."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")

        mock_client = mocker.Mock()
        mock_client.collection_exists.return_value = True
        mock_get_client.return_value = mock_client

        engine = QdrantEngine(collection_name="test", path="/db")

        assert engine._collection_created is False
        engine._ensure_collection()
        assert engine._collection_created is True


class TestQdrantEngineStore:
    """Tests for QdrantEngine.store method."""

    def test_stores_single_text_as_string(self, mocker: MockerFixture) -> None:
        """Accepts single text string and stores it."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_client = mocker.Mock()
        mock_client.collection_exists.return_value = True
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embedding
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db")

        engine.store("single text")

        mock_model.encode.assert_called_once_with(
            ["single text"], convert_to_numpy=True
        )
        mock_client.upload_points.assert_called_once()

    def test_stores_list_of_texts(self, mocker: MockerFixture) -> None:
        """Stores multiple texts from a list."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_client = mocker.Mock()
        mock_client.collection_exists.return_value = True
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db")
        texts = ["text one", "text two", "text three"]

        engine.store(texts)

        mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)
        mock_client.upload_points.assert_called_once()

        call_kwargs = mock_client.upload_points.call_args
        points = call_kwargs.kwargs["points"]
        assert len(points) == 3

    def test_creates_points_with_correct_payload(self, mocker: MockerFixture) -> None:
        """Creates points with text in payload."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_client = mocker.Mock()
        mock_client.collection_exists.return_value = True
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_embeddings = np.array([[0.1, 0.2]])
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db")

        engine.store(["my text content"])

        call_kwargs = mock_client.upload_points.call_args
        points = call_kwargs.kwargs["points"]
        assert points[0].payload == {"text": "my text content"}

    def test_ensures_collection_before_storing(self, mocker: MockerFixture) -> None:
        """Ensures collection exists before storing."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_client = mocker.Mock()
        mock_client.collection_exists.return_value = False
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="new-coll", path="/db")

        engine.store(["text"])

        mock_client.create_collection.assert_called_once()


class TestQdrantEngineSearch:
    """Tests for QdrantEngine.search method."""

    def test_returns_matching_texts(self, mocker: MockerFixture) -> None:
        """Returns texts from search results."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_result_1 = mocker.Mock()
        mock_result_1.payload = {"text": "result one"}
        mock_result_2 = mocker.Mock()
        mock_result_2.payload = {"text": "result two"}

        mock_query_response = mocker.Mock()
        mock_query_response.points = [mock_result_1, mock_result_2]

        mock_client = mocker.Mock()
        mock_client.query_points.return_value = mock_query_response
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db")

        results = engine.search("my query")

        assert results == ["result one", "result two"]

    def test_uses_correct_collection_and_limit(self, mocker: MockerFixture) -> None:
        """Queries with correct collection name and limit."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_query_response = mocker.Mock()
        mock_query_response.points = []

        mock_client = mocker.Mock()
        mock_client.query_points.return_value = mock_query_response
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="my-collection", path="/db")

        engine.search("query", top_k=10)

        call_kwargs = mock_client.query_points.call_args
        assert call_kwargs.kwargs["collection_name"] == "my-collection"
        assert call_kwargs.kwargs["limit"] == 10
        assert call_kwargs.kwargs["with_payload"] is True

    def test_default_top_k_is_five(self, mocker: MockerFixture) -> None:
        """Uses default top_k of 5."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_query_response = mocker.Mock()
        mock_query_response.points = []

        mock_client = mocker.Mock()
        mock_client.query_points.return_value = mock_query_response
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_model.encode.return_value = np.array([0.1])
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db")

        engine.search("query")

        call_kwargs = mock_client.query_points.call_args
        assert call_kwargs.kwargs["limit"] == 5

    def test_skips_results_without_payload(self, mocker: MockerFixture) -> None:
        """Skips results that have no payload."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_result_1 = mocker.Mock()
        mock_result_1.payload = {"text": "valid result"}
        mock_result_2 = mocker.Mock()
        mock_result_2.payload = None

        mock_query_response = mocker.Mock()
        mock_query_response.points = [mock_result_1, mock_result_2]

        mock_client = mocker.Mock()
        mock_client.query_points.return_value = mock_query_response
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_model.encode.return_value = np.array([0.1])
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db")

        results = engine.search("query")

        assert results == ["valid result"]

    def test_skips_results_without_text_key(self, mocker: MockerFixture) -> None:
        """Skips results that have payload but no 'text' key."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_result_1 = mocker.Mock()
        mock_result_1.payload = {"text": "valid result"}
        mock_result_2 = mocker.Mock()
        mock_result_2.payload = {"other_key": "no text here"}

        mock_query_response = mocker.Mock()
        mock_query_response.points = [mock_result_1, mock_result_2]

        mock_client = mocker.Mock()
        mock_client.query_points.return_value = mock_query_response
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_model.encode.return_value = np.array([0.1])
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db")

        results = engine.search("query")

        assert results == ["valid result"]

    def test_encodes_query_with_model(self, mocker: MockerFixture) -> None:
        """Encodes query string using embedding model."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_query_response = mocker.Mock()
        mock_query_response.points = []

        mock_client = mocker.Mock()
        mock_client.query_points.return_value = mock_query_response
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db")

        engine.search("my search query")

        mock_model.encode.assert_called_once_with(
            "my search query", convert_to_numpy=True
        )

    def test_returns_empty_list_when_no_results(self, mocker: MockerFixture) -> None:
        """Returns empty list when search finds no results."""
        mock_get_client = mocker.patch("repo_sage.vector_engine._get_qdrant_client")
        mock_get_model = mocker.patch("repo_sage.vector_engine._get_embedding_model")

        mock_query_response = mocker.Mock()
        mock_query_response.points = []

        mock_client = mocker.Mock()
        mock_client.query_points.return_value = mock_query_response
        mock_get_client.return_value = mock_client

        mock_model = mocker.Mock()
        mock_model.encode.return_value = np.array([0.1])
        mock_get_model.return_value = mock_model

        engine = QdrantEngine(collection_name="test", path="/db")

        results = engine.search("query")

        assert results == []
