"""Demo script for Phase 3: RAG System.

Demonstrates:
- Document ingestion into Qdrant
- Vector retrieval
- RAG query with generation
- Reranking (optional)
- Caching
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


def print_header(title: str):
    """Print a formatted header."""
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False))


def demo_basic_rag():
    """Demonstrate basic RAG functionality without requiring full setup."""
    print_header("Phase 3 Demo: RAG System")

    console.print("\n[yellow]Note: This demo requires Qdrant running on localhost:6333[/yellow]")
    console.print("[yellow]Start services with: make services-up[/yellow]\n")

    # Check if we can import modules
    try:
        from src.rag import QdrantRetriever, RAGCache, RAGPipeline
        from src.rag.config import RAGConfig
        from src.rag.reranker import HybridReranker

        console.print("✅ RAG modules imported successfully\n")
    except ImportError as e:
        console.print(f"[red]❌ Failed to import modules: {e}[/red]")
        console.print("[yellow]Install dependencies with: make install[/yellow]")
        return

    # Demo 1: Configuration
    print_header("1. Configuration Management")

    try:
        config = RAGConfig.from_yaml("configs/rag_config.yaml")
        console.print("✅ Configuration loaded from configs/rag_config.yaml")

        # Display config
        config_table = Table(title="RAG Configuration")
        config_table.add_column("Component", style="cyan")
        config_table.add_column("Setting", style="magenta")
        config_table.add_column("Value", style="green")

        config_table.add_row("Retriever", "Embedding Model", config.retriever.embedding_model)
        config_table.add_row("Retriever", "Top K", str(config.retriever.top_k))
        config_table.add_row("Generator", "Model Name", config.generator.model_name)
        config_table.add_row("Generator", "Temperature", str(config.generator.temperature))
        config_table.add_row("Reranker", "Enabled", str(config.reranker.enabled))
        config_table.add_row("Cache", "Enabled", str(config.cache.enabled))
        config_table.add_row("Cache", "TTL", f"{config.cache.ttl}s")

        console.print(config_table)
        console.print()

    except Exception as e:
        console.print(f"[yellow]⚠ Config loading failed: {e}[/yellow]")
        console.print("Using default configuration\n")
        config = RAGConfig()

    # Demo 2: Document Retrieval (without LLM)
    print_header("2. Document Retrieval")

    try:
        retriever = QdrantRetriever(
            collection_name="demo_collection",
            embedding_model=config.retriever.embedding_model,
        )
        console.print("✅ Retriever initialized")

        # Create collection
        retriever.create_collection(recreate=True)
        console.print("✅ Qdrant collection created")

        # Add sample documents
        sample_docs = [
            "Việt Nam là một quốc gia ở Đông Nam Á với dân số hơn 98 triệu người.",
            "Hà Nội là thủ đô của Việt Nam, nổi tiếng với Hồ Hoàn Kiếm và phố cổ.",
            "Phở là món ăn truyền thống của Việt Nam, được yêu thích trên toàn thế giới.",
            "Machine Learning là một nhánh của trí tuệ nhân tạo giúp máy tính học từ dữ liệu.",
            "Python là ngôn ngữ lập trình phổ biến cho data science và AI.",
        ]

        retriever.add_documents(sample_docs)
        console.print(f"✅ Added {len(sample_docs)} documents to collection\n")

        # Retrieve documents
        query = "Thủ đô của Việt Nam"
        console.print(f"[bold]Query:[/bold] {query}")

        retrieved = retriever.retrieve(query, top_k=3)

        results_table = Table(title="Retrieved Documents")
        results_table.add_column("#", style="cyan", width=5)
        results_table.add_column("Score", style="magenta", width=8)
        results_table.add_column("Content", style="white")

        for i, doc in enumerate(retrieved, 1):
            results_table.add_row(
                str(i),
                f"{doc.score:.3f}",
                doc.content[:80] + "..." if len(doc.content) > 80 else doc.content,
            )

        console.print(results_table)
        console.print()

    except Exception as e:
        console.print(f"[red]❌ Retrieval demo failed: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running: docker compose up qdrant[/yellow]\n")
        return

    # Demo 3: Reranking
    print_header("3. Document Reranking")

    try:
        console.print("[yellow]Loading reranker model (this may take a moment)...[/yellow]")
        reranker = HybridReranker(alpha=0.5)
        console.print("✅ Reranker initialized")

        reranked = reranker.rerank(query, retrieved, top_k=3)

        rerank_table = Table(title="Reranked Documents")
        rerank_table.add_column("#", style="cyan", width=5)
        rerank_table.add_column("New Score", style="green", width=12)
        rerank_table.add_column("Original", style="magenta", width=12)
        rerank_table.add_column("Content", style="white")

        for i, doc in enumerate(reranked, 1):
            original_score = doc.metadata.get("vector_score", 0)
            rerank_table.add_row(
                str(i),
                f"{doc.score:.3f}",
                f"{original_score:.3f}",
                doc.content[:60] + "..." if len(doc.content) > 60 else doc.content,
            )

        console.print(rerank_table)
        console.print()

    except Exception as e:
        console.print(f"[yellow]⚠ Reranking demo skipped: {e}[/yellow]\n")

    # Demo 4: Cache
    print_header("4. Redis Caching")

    try:
        cache = RAGCache(enabled=True)

        if cache.enabled:
            console.print("✅ Redis cache connected")

            # Cache a result
            test_result = {"answer": "Hà Nội", "sources": ["doc1"]}
            cache.set("test_query", test_result, top_k=3)
            console.print("✅ Cached a test result")

            # Retrieve from cache
            cached = cache.get("test_query", top_k=3)
            if cached:
                console.print("✅ Retrieved from cache successfully")
                console.print(f"   Cached data: {cached}\n")

            # Get stats
            stats = cache.get_stats()
            console.print(f"📊 Cache stats: {stats}\n")

        else:
            console.print("[yellow]⚠ Redis cache not available[/yellow]\n")

    except Exception as e:
        console.print(f"[yellow]⚠ Cache demo skipped: {e}[/yellow]\n")

    # Summary
    print_header("Phase 3 Features Summary")

    features = Table()
    features.add_column("Feature", style="cyan")
    features.add_column("Status", style="green")
    features.add_column("Description", style="white")

    features.add_row(
        "Vector Retrieval",
        "✅ Implemented",
        "Qdrant-based semantic search",
    )
    features.add_row(
        "Document Reranking",
        "✅ Implemented",
        "Cross-encoder + hybrid scoring",
    )
    features.add_row(
        "Redis Caching",
        "✅ Implemented",
        "Query result caching with TTL",
    )
    features.add_row(
        "Configuration",
        "✅ Implemented",
        "YAML-based config management",
    )
    features.add_row(
        "FastAPI Endpoints",
        "✅ Implemented",
        "/query, /documents, /health, /metrics",
    )
    features.add_row(
        "Streaming Responses",
        "✅ Implemented",
        "Server-sent events for /query/stream",
    )

    console.print(features)
    console.print()

    console.print("[bold green]Phase 3 Demo Complete![/bold green]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  • Start all services: [yellow]make services-up[/yellow]")
    console.print("  • Start API server: [yellow]make api[/yellow]")
    console.print("  • Test API: [yellow]curl http://localhost:8000/health[/yellow]")
    console.print()


if __name__ == "__main__":
    try:
        demo_basic_rag()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        import traceback

        traceback.print_exc()
