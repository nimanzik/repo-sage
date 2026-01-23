from functools import lru_cache
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .defaults import DEFAULT_EMBEDDING_MODEL_ID
from .utils.logging import get_logger

logger = get_logger(__name__, level="info")


@lru_cache(maxsize=8)
def _get_embedding_model(model_id: str) -> SentenceTransformer:
    """Load and cache the sentence transformer model."""
    return SentenceTransformer(model_id, model_kwargs={"torch_dtype": "float16"})


@lru_cache(maxsize=32)
def _get_qdrant_client(path: str) -> QdrantClient:
    """Get or create a Qdrant client with local file persistence."""
    return QdrantClient(path=path)


class QdrantEngine:
    def __init__(
        self, collection_name: str, path: str, model_id: str | None = None
    ) -> None:
        self.collection_name = collection_name
        self.path = path
        self.model_id = model_id or DEFAULT_EMBEDDING_MODEL_ID
        self._client: QdrantClient | None = None
        self._embedding_model: SentenceTransformer | None = None
        self._collection_created: bool = False

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            logger.info("Creating Qdrant client")
            self._client = _get_qdrant_client(self.path)
        return self._client

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info("Loading embedding model")
            self._embedding_model = _get_embedding_model(self.model_id)
        return self._embedding_model

    def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists, creating it if necessary."""
        if self._collection_created:
            return

        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating Qdrant collection '{self.collection_name}'")

            vector_size = self.embedding_model.get_sentence_embedding_dimension()
            if not vector_size:
                raise ValueError(f"Unknown embedding size for model '{self.model_id}'")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        self._collection_created = True

    def store(self, texts: list[str] | str) -> None:
        """Store texts and their embeddings in a Qdrant collection."""
        self._ensure_collection()

        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)

        # Prepare points for Qdrant
        points = [
            PointStruct(id=uuid4(), vector=embed.tolist(), payload={"text": text})
            for text, embed in zip(texts, embeddings, strict=True)
        ]

        # Upload points to Qdrant
        self.client.upload_points(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Perform semantic search in a Qdrant collection."""
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedding_model.encode(query, convert_to_numpy=True).tolist(),
            limit=top_k,
            with_payload=True,
        ).points

        retrieved_texts = [
            result.payload["text"]
            for result in search_results
            if result.payload and "text" in result.payload
        ]

        return retrieved_texts
