#!/usr/bin/env python3
"""
Phase 1 Demo: Vietnamese Data Pipeline
======================================

This script demonstrates the core data processing capabilities:
1. Vietnamese text detection
2. Text cleaning and normalization
3. Quality scoring
4. Text chunking for RAG

Run: python scripts/demo_phase1.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Initialize rich console for pretty output
console = Console()


def demo_text_processor():
    """Demonstrate Vietnamese text processing capabilities."""
    from src.data_pipeline.text_processor import TextProcessor

    console.rule("[bold blue]1. Text Processor Demo")

    processor = TextProcessor(min_words=10, min_chars=50)

    # Sample Vietnamese texts
    samples = [
        {
            "name": "Clean Vietnamese",
            "text": "Việt Nam là một quốc gia nằm ở Đông Nam Á. Thủ đô của Việt Nam là Hà Nội. Đây là một đất nước có lịch sử văn hóa lâu đời và phong phú.",
        },
        {
            "name": "Text with URLs/Emails",
            "text": "Truy cập https://example.com hoặc liên hệ contact@vietnam.vn để biết thêm thông tin về du lịch Việt Nam và các địa điểm nổi tiếng.",
        },
        {
            "name": "Noisy Text",
            "text": "   Xin   chào!!!   Đây là    văn bản   có nhiều    khoảng trắng   và ký tự đặc biệt @#$%   cần được làm sạch...   ",
        },
        {
            "name": "Short Text",
            "text": "Xin chào",
        },
    ]

    for sample in samples:
        console.print(f"\n[yellow]Input ({sample['name']}):[/yellow]")
        console.print(f"  {sample['text'][:80]}...")

        result = processor.process(sample["text"])

        console.print(f"[green]Output:[/green]")
        console.print(f"  Text: {result.text[:80]}...")
        console.print(f"  Words: {result.word_count} | Chars: {result.char_count}")
        console.print(f"  Quality: {result.quality_score:.2f} | Valid: {result.is_valid}")


def demo_text_chunking():
    """Demonstrate text chunking for RAG."""
    from src.data_pipeline.text_processor import TextProcessor

    console.rule("[bold blue]2. Text Chunking for RAG")

    processor = TextProcessor()

    long_text = """
    Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính tập trung vào việc
    tạo ra các hệ thống có khả năng thực hiện các nhiệm vụ thường đòi hỏi trí thông minh
    của con người. Machine learning là một nhánh quan trọng của AI, cho phép máy tính
    học từ dữ liệu mà không cần được lập trình cụ thể. Deep learning, một kỹ thuật của
    machine learning, sử dụng mạng neural nhiều lớp để xử lý thông tin phức tạp.

    Ứng dụng của AI rất đa dạng, từ nhận dạng giọng nói, xử lý ngôn ngữ tự nhiên,
    đến xe tự lái và chẩn đoán y tế. Tại Việt Nam, AI đang được ứng dụng trong nhiều
    lĩnh vực như tài chính, y tế, giáo dục và nông nghiệp thông minh.
    """

    chunks = processor.chunk_text(long_text, chunk_size=200, overlap=30)

    console.print(f"\n[yellow]Original text length:[/yellow] {len(long_text)} chars")
    console.print(f"[yellow]Number of chunks:[/yellow] {len(chunks)}")

    table = Table(title="Text Chunks")
    table.add_column("Chunk #", style="cyan")
    table.add_column("Length", style="green")
    table.add_column("Preview", style="white")

    for i, chunk in enumerate(chunks):
        preview = chunk[:60].replace("\n", " ") + "..."
        table.add_row(str(i + 1), str(len(chunk)), preview)

    console.print(table)


def demo_vietnamese_detection():
    """Demonstrate Vietnamese language detection."""
    console.rule("[bold blue]3. Vietnamese Character Detection")

    from src.data_pipeline.vietnamese_detector import VietnameseDetector

    detector = VietnameseDetector()

    test_texts = [
        ("Xin chào, tôi là trợ lý AI", "Vietnamese"),
        ("Hello, I am an AI assistant", "English"),
        ("こんにちは、私はAIアシスタントです", "Japanese"),
        ("Mixed: Hello và xin chào bạn", "Mixed"),
    ]

    table = Table(title="Vietnamese Character Detection")
    table.add_column("Text", style="white")
    table.add_column("Expected", style="cyan")
    table.add_column("Has VN Chars", style="green")

    for text, expected in test_texts:
        has_vn = detector.has_vietnamese_chars(text)
        table.add_row(text[:40], expected, "✓" if has_vn else "✗")

    console.print(table)

    console.print("\n[dim]Note: Full language detection requires FastText model (lid.176.bin)[/dim]")


def demo_quality_scoring():
    """Demonstrate text quality scoring."""
    from src.data_pipeline.text_processor import TextProcessor

    console.rule("[bold blue]4. Text Quality Scoring")

    processor = TextProcessor()

    texts = [
        ("High quality article text with good vocabulary and structure about Vietnamese culture and history.", "Good"),
        ("spam spam spam spam spam spam spam spam spam spam", "Repetitive"),
        ("12345 67890 11111 22222 33333 44444 55555", "Too many numbers"),
        ("Short", "Too short"),
    ]

    table = Table(title="Quality Scores")
    table.add_column("Text Preview", style="white")
    table.add_column("Type", style="cyan")
    table.add_column("Score", style="green")

    for text, text_type in texts:
        score = processor.calculate_quality_score(text)
        table.add_row(text[:40] + "...", text_type, f"{score:.2f}")

    console.print(table)


def demo_sample_data():
    """Create and display sample data."""
    console.rule("[bold blue]5. Sample Data Generation")

    import json
    import subprocess
    from pathlib import Path

    # Run the sample data script as subprocess
    result = subprocess.run(
        [sys.executable, "scripts/download_sample_data.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        console.print(f"[red]Error generating data:[/red] {result.stderr}")
        return

    data_dir = Path("data")
    train_file = data_dir / "train.jsonl"
    eval_file = data_dir / "eval_rag.json"

    if train_file.exists():
        with open(train_file) as f:
            lines = f.readlines()
        console.print(f"\n[green]✓[/green] Training data: {len(lines)} samples")

        # Show first sample
        sample = json.loads(lines[0])
        console.print(Panel(
            f"[cyan]Topic:[/cyan] {sample.get('topic', 'N/A')}\n"
            f"[cyan]Text:[/cyan] {sample['text'][:100]}...",
            title="Sample Training Data"
        ))

    if eval_file.exists():
        with open(eval_file) as f:
            eval_data = json.load(f)
        console.print(f"[green]✓[/green] Evaluation data: {len(eval_data['questions'])} QA pairs")


def main():
    """Run all Phase 1 demos."""
    console.print(Panel.fit(
        "[bold green]VinaSmol RAG MLOps - Phase 1 Demo[/bold green]\n"
        "Vietnamese Data Pipeline Features",
        border_style="green"
    ))

    try:
        demo_text_processor()
        demo_text_chunking()
        demo_vietnamese_detection()
        demo_quality_scoring()
        demo_sample_data()

        console.print("\n")
        console.rule("[bold green]Phase 1 Demo Complete!")

        console.print(Panel(
            "[bold]Next Steps:[/bold]\n"
            "• Run tests: [cyan]make test[/cyan]\n"
            "• Start services: [cyan]make services-up[/cyan]\n"
            "• Launch API: [cyan]make api[/cyan]",
            title="What's Next?",
            border_style="blue"
        ))

    except ImportError as e:
        console.print(f"[red]Missing dependency:[/red] {e}")
        console.print("Run: [cyan]pip3 install --user rich[/cyan]")


if __name__ == "__main__":
    main()
