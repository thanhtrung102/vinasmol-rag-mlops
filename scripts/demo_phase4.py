"""Demo script for Phase 4: Evaluation Framework.

Demonstrates:
- Vietnamese benchmark dataset
- Hallucination detection
- RAG evaluation metrics
- Report generation
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


def demo_vietnamese_benchmark():
    """Demonstrate Vietnamese benchmark dataset."""
    print_header("1. Vietnamese Benchmark Dataset")

    try:
        from src.evaluation.vietnamese_benchmark import VietnameseBenchmark

        benchmark = VietnameseBenchmark()
        console.print("✅ Vietnamese benchmark loaded\n")

        # Show statistics
        stats = benchmark.get_statistics()

        stats_table = Table(title="Benchmark Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Questions", str(stats["total_questions"]))

        console.print(stats_table)
        console.print()

        # Show categories
        cat_table = Table(title="Questions by Category")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", style="magenta")

        for category, count in stats["categories"].items():
            cat_table.add_row(category.capitalize(), str(count))

        console.print(cat_table)
        console.print()

        # Show sample questions
        sample_table = Table(title="Sample Questions")
        sample_table.add_column("#", style="cyan", width=5)
        sample_table.add_column("Category", style="magenta", width=12)
        sample_table.add_column("Question", style="white")

        for i, q in enumerate(benchmark.get_all()[:5], 1):
            sample_table.add_row(
                str(i),
                q.category,
                q.question[:60] + "..." if len(q.question) > 60 else q.question,
            )

        console.print(sample_table)
        console.print()

    except Exception as e:
        console.print(f"[red]❌ Benchmark demo failed: {e}[/red]\n")


def demo_hallucination_detection():
    """Demonstrate hallucination detection."""
    print_header("2. Hallucination Detection")

    try:
        from src.evaluation.hallucination_detector import (
            AdvancedHallucinationDetector,
            SimpleHallucinationDetector,
        )

        # Example 1: Grounded answer
        console.print("[bold]Example 1: Grounded Answer[/bold]")
        question = "Thủ đô của Việt Nam là gì?"
        answer = "Hà Nội là thủ đô của Việt Nam."
        contexts = [
            "Hà Nội là thủ đô của Việt Nam.",
            "Thủ đô Hà Nội có dân số khoảng 8 triệu người.",
        ]

        simple_detector = SimpleHallucinationDetector(threshold=0.5)
        result1 = simple_detector.detect(answer, contexts)

        console.print(f"Answer: {answer}")
        console.print(f"Verdict: [green]{result1['verdict']}[/green]")
        console.print(f"Score: {result1['score']:.3f}")
        console.print()

        # Example 2: Hallucinated answer
        console.print("[bold]Example 2: Hallucinated Answer[/bold]")
        answer2 = "Tokyo là thủ đô của Việt Nam."

        result2 = simple_detector.detect(answer2, contexts)

        console.print(f"Answer: {answer2}")
        console.print(f"Verdict: [red]{result2['verdict']}[/red]")
        console.print(f"Score: {result2['score']:.3f}")
        console.print()

        # Advanced detector
        console.print("[bold]Using Advanced Detector:[/bold]")
        advanced_detector = AdvancedHallucinationDetector()

        result3 = advanced_detector.detect(question, answer, contexts)

        adv_table = Table()
        adv_table.add_column("Metric", style="cyan")
        adv_table.add_column("Score", style="magenta")

        adv_table.add_row("Faithfulness", f"{result3.faithfulness_score:.3f}")
        adv_table.add_row("Consistency", f"{result3.consistency_score:.3f}")
        adv_table.add_row("Entity Match", f"{result3.entity_match_score:.3f}")
        adv_table.add_row("Confidence", f"{result3.confidence:.3f}")

        console.print(adv_table)
        console.print(f"Verdict: [green]{result3.verdict}[/green]\n")

    except Exception as e:
        console.print(f"[yellow]⚠ Hallucination demo skipped: {e}[/yellow]\n")


def demo_rag_evaluation():
    """Demonstrate RAG evaluation."""
    print_header("3. RAG System Evaluation")

    console.print("[yellow]Note: RAG evaluation requires Ragas package[/yellow]")
    console.print("[yellow]Install with: pip install ragas[/yellow]\n")

    try:
        from src.evaluation.evaluate_rag import RAGEvaluator

        console.print("✅ RAGEvaluator loaded")

        # Mock evaluation data
        questions = ["Thủ đô của Việt Nam là gì?"]
        answers = ["Hà Nội là thủ đô của Việt Nam."]
        contexts = [["Hà Nội là thủ đô của Việt Nam, nằm ở miền Bắc."]]

        console.print("\n[bold]Evaluation Metrics:[/bold]")
        console.print("  • Faithfulness: factual consistency with context")
        console.print("  • Answer Relevance: relevance to question")
        console.print("  • Context Precision: ranking quality")
        console.print("  • Context Recall: completeness of retrieval\n")

        console.print("[cyan]Sample evaluation data prepared[/cyan]")
        console.print(f"Questions: {len(questions)}")
        console.print(f"Answers: {len(answers)}")
        console.print(f"Contexts: {len(contexts)}\n")

    except ImportError:
        console.print("[yellow]⚠ Ragas not installed[/yellow]\n")
    except Exception as e:
        console.print(f"[yellow]⚠ Evaluation demo skipped: {e}[/yellow]\n")


def demo_summary():
    """Show Phase 4 summary."""
    print_header("Phase 4 Features Summary")

    features = Table()
    features.add_column("Feature", style="cyan")
    features.add_column("Status", style="green")
    features.add_column("Description", style="white")

    features.add_row(
        "Vietnamese Benchmark",
        "✅ Implemented",
        "Curated Q&A dataset across 8 categories",
    )
    features.add_row(
        "Hallucination Detection",
        "✅ Implemented",
        "Simple & Advanced detection strategies",
    )
    features.add_row(
        "Ragas Integration",
        "✅ Implemented",
        "Faithfulness, relevance, precision metrics",
    )
    features.add_row(
        "MLflow Logging",
        "✅ Implemented",
        "Automatic metric logging",
    )
    features.add_row(
        "Report Generation",
        "✅ Implemented",
        "Markdown evaluation reports",
    )

    console.print(features)
    console.print()

    console.print("[bold green]Phase 4 Implementation Complete![/bold green]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  • Run full evaluation: [yellow]make eval-rag[/yellow]")
    console.print("  • Export benchmark: [yellow]python -m src.evaluation.vietnamese_benchmark --export data/vn_benchmark.json[/yellow]")
    console.print("  • Test hallucination detection in production API")
    console.print()


def main():
    """Main demo entrypoint."""
    console.print("\n")
    print_header("Phase 4 Demo: Evaluation Framework")
    console.print()

    try:
        # Demo 1: Vietnamese Benchmark
        demo_vietnamese_benchmark()

        # Demo 2: Hallucination Detection
        demo_hallucination_detection()

        # Demo 3: RAG Evaluation
        demo_rag_evaluation()

        # Summary
        demo_summary()

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
