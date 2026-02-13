"""Reranker for improving retrieval quality.

Uses cross-encoder models to rerank retrieved documents based on
relevance to the query, improving precision over vector similarity alone.
"""

from sentence_transformers import CrossEncoder

from .retriever import RetrievedDocument


class DocumentReranker:
    """Rerank retrieved documents using cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ):
        """Initialize the reranker.

        Args:
            model_name: HuggingFace cross-encoder model name.
            batch_size: Batch size for reranking.
        """
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Rerank documents based on cross-encoder scores.

        Args:
            query: The search query.
            documents: List of retrieved documents to rerank.
            top_k: Number of top documents to return (default: all).

        Returns:
            Reranked list of documents with updated scores.
        """
        if not documents:
            return []

        # Prepare query-document pairs for cross-encoder
        pairs = [[query, doc.content] for doc in documents]

        # Get cross-encoder scores
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Create new RetrievedDocument objects with updated scores
        reranked = [
            RetrievedDocument(
                content=doc.content,
                score=float(score),
                metadata={**doc.metadata, "original_score": doc.score},
            )
            for doc, score in zip(documents, scores)
        ]

        # Sort by new scores (descending)
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Return top_k if specified
        if top_k is not None:
            return reranked[:top_k]

        return reranked


class HybridReranker:
    """Hybrid reranker combining vector similarity and cross-encoder scores."""

    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha: float = 0.5,
    ):
        """Initialize hybrid reranker.

        Args:
            cross_encoder_model: Cross-encoder model name.
            alpha: Weight for cross-encoder score (0-1).
                   Final score = alpha * cross_encoder + (1-alpha) * vector_sim
        """
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.alpha = alpha

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Rerank using hybrid approach.

        Args:
            query: The search query.
            documents: List of retrieved documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            Reranked list with hybrid scores.
        """
        if not documents:
            return []

        # Get cross-encoder scores
        pairs = [[query, doc.content] for doc in documents]
        ce_scores = self.cross_encoder.predict(pairs)

        # Normalize original scores to 0-1 range for fair combination
        original_scores = [doc.score for doc in documents]
        min_score = min(original_scores)
        max_score = max(original_scores)
        score_range = max_score - min_score

        reranked = []
        for doc, ce_score in zip(documents, ce_scores):
            # Normalize original score
            norm_original = (doc.score - min_score) / score_range if score_range > 0 else 1.0

            # Combine scores
            hybrid_score = self.alpha * float(ce_score) + (1 - self.alpha) * norm_original

            reranked.append(
                RetrievedDocument(
                    content=doc.content,
                    score=hybrid_score,
                    metadata={
                        **doc.metadata,
                        "vector_score": doc.score,
                        "cross_encoder_score": float(ce_score),
                        "hybrid_alpha": self.alpha,
                    },
                )
            )

        # Sort by hybrid scores
        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k is not None:
            return reranked[:top_k]

        return reranked
