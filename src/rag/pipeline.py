"""End-to-end RAG pipeline combining retrieval and generation."""

from dataclasses import dataclass
from typing import Any

from .generator import GenerationResult, RAGGenerator
from .retriever import QdrantRetriever, RetrievedDocument


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""

    question: str
    answer: str
    retrieved_docs: list[RetrievedDocument]
    generation_result: GenerationResult
    metadata: dict[str, Any]


class RAGPipeline:
    """End-to-end RAG pipeline for Vietnamese Q&A."""

    def __init__(
        self,
        retriever: QdrantRetriever | None = None,
        generator: RAGGenerator | None = None,
        top_k: int = 3,
        score_threshold: float = 0.5,
    ):
        """Initialize the RAG pipeline.

        Args:
            retriever: Document retriever instance.
            generator: Response generator instance.
            top_k: Number of documents to retrieve.
            score_threshold: Minimum retrieval score.
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.score_threshold = score_threshold

    def initialize(
        self,
        collection_name: str = "vietnamese_docs",
        model_name: str = "vinai/PhoGPT-4B-Chat",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        """Initialize retriever and generator with default settings.

        Args:
            collection_name: Qdrant collection name.
            model_name: LLM model name.
            embedding_model: Embedding model name.
        """
        self.retriever = QdrantRetriever(
            collection_name=collection_name,
            embedding_model=embedding_model,
        )

        self.generator = RAGGenerator(
            model_name=model_name,
            load_in_8bit=True,
        )

    def query(
        self,
        question: str,
        top_k: int | None = None,
        **generation_kwargs,
    ) -> RAGResponse:
        """Process a question through the RAG pipeline.

        Args:
            question: User's question.
            top_k: Override default top_k for retrieval.
            **generation_kwargs: Additional arguments for generation.

        Returns:
            Complete RAG response with answer and sources.
        """
        if self.retriever is None or self.generator is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Retrieve relevant documents
        retrieved = self.retriever.retrieve(
            query=question,
            top_k=top_k or self.top_k,
            score_threshold=self.score_threshold,
        )

        # Extract document contents
        context_docs = [doc.content for doc in retrieved]

        # Handle case with no relevant documents
        if not context_docs:
            return RAGResponse(
                question=question,
                answer="Xin lỗi, tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này.",
                retrieved_docs=[],
                generation_result=GenerationResult(
                    answer="",
                    context_used=[],
                    metadata={"no_context": True},
                ),
                metadata={"retrieval_count": 0},
            )

        # Generate response
        generation_result = self.generator.generate(
            question=question,
            context_docs=context_docs,
            **generation_kwargs,
        )

        return RAGResponse(
            question=question,
            answer=generation_result.answer,
            retrieved_docs=retrieved,
            generation_result=generation_result,
            metadata={
                "retrieval_count": len(retrieved),
                "avg_retrieval_score": sum(d.score for d in retrieved) / len(retrieved),
            },
        )

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the knowledge base.

        Args:
            documents: List of document texts.
            metadatas: Optional metadata for each document.
        """
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized.")
        self.retriever.add_documents(documents, metadatas)
