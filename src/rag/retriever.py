"""Vector retrieval using Qdrant."""

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any]


class QdrantRetriever:
    """Retrieve relevant documents from Qdrant vector database."""

    def __init__(
        self,
        collection_name: str = "vietnamese_docs",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        host: str = "localhost",
        port: int = 6333,
    ):
        """Initialize the retriever.

        Args:
            collection_name: Name of the Qdrant collection.
            embedding_model: HuggingFace model for embeddings.
            host: Qdrant server host.
            port: Qdrant server port.
        """
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.encoder = SentenceTransformer(embedding_model)
        self._embedding_dim = self.encoder.get_sentence_embedding_dimension()

    def create_collection(self, recreate: bool = False) -> None:
        """Create the vector collection if it doesn't exist.

        Args:
            recreate: If True, delete and recreate the collection.
        """
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in collections:
            if recreate:
                self.client.delete_collection(self.collection_name)
            else:
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self._embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        batch_size: int = 100,
    ) -> None:
        """Add documents to the collection.

        Args:
            documents: List of document texts.
            metadatas: Optional metadata for each document.
            batch_size: Number of documents to process at once.
        """
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Process in batches for memory efficiency
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]

            embeddings = self.encoder.encode(batch_docs, show_progress_bar=False)

            points = [
                models.PointStruct(
                    id=i + j,
                    vector=embedding.tolist(),
                    payload={"content": doc, **meta},
                )
                for j, (doc, embedding, meta) in enumerate(
                    zip(batch_docs, embeddings, batch_meta)
                )
            ]

            self.client.upsert(collection_name=self.collection_name, points=points)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[RetrievedDocument]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query text.
            top_k: Maximum number of documents to return.
            score_threshold: Minimum similarity score threshold.

        Returns:
            List of retrieved documents with scores.
        """
        query_embedding = self.encoder.encode(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
        )

        return [
            RetrievedDocument(
                content=hit.payload.get("content", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "content"},
            )
            for hit in results
        ]

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection.

        Returns:
            Collection statistics and configuration.
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
