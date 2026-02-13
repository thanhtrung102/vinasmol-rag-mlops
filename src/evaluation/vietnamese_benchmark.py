"""Vietnamese Q&A benchmark dataset for RAG evaluation.

Provides curated Vietnamese question-answer pairs across multiple domains
for comprehensive RAG system evaluation.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuestion:
    """A benchmark question with metadata."""

    id: str
    category: str
    question: str
    ground_truth: str
    context: list[str]
    difficulty: str  # easy, medium, hard
    metadata: dict[str, Any]


class VietnameseBenchmark:
    """Vietnamese Q&A benchmark for RAG evaluation."""

    BENCHMARK_QUESTIONS = [
        {
            "id": "vn_geo_001",
            "category": "geography",
            "question": "Thủ đô của Việt Nam là gì?",
            "ground_truth": "Hà Nội là thủ đô của Việt Nam.",
            "context": [
                "Hà Nội là thủ đô của Việt Nam, nằm ở miền Bắc Việt Nam.",
                "Thủ đô Hà Nội có dân số khoảng 8 triệu người.",
            ],
            "difficulty": "easy",
        },
        {
            "id": "vn_culture_001",
            "category": "culture",
            "question": "Phở là món ăn gì?",
            "ground_truth": "Phở là món ăn truyền thống của Việt Nam, gồm bánh phở, nước dùng và thịt bò hoặc gà.",
            "context": [
                "Phở là món ăn truyền thống nổi tiếng nhất của Việt Nam.",
                "Phở gồm bánh phở, nước dùng được ninh từ xương, thịt bò hoặc gà, và rau thơm.",
                "Phở được UNESCO công nhận là di sản văn hóa phi vật thể.",
            ],
            "difficulty": "easy",
        },
        {
            "id": "vn_history_001",
            "category": "history",
            "question": "Việt Nam thống nhất năm nào?",
            "ground_truth": "Việt Nam thống nhất vào ngày 30 tháng 4 năm 1975.",
            "context": [
                "Ngày 30/4/1975 đánh dấu mốc lịch sử thống nhất đất nước Việt Nam.",
                "Sau nhiều năm chiến tranh, miền Nam và miền Bắc Việt Nam được thống nhất.",
            ],
            "difficulty": "medium",
        },
        {
            "id": "vn_tech_001",
            "category": "technology",
            "question": "AI là gì?",
            "ground_truth": "AI (Artificial Intelligence) hay trí tuệ nhân tạo là khả năng của máy tính thực hiện các tác vụ thông minh như con người.",
            "context": [
                "Trí tuệ nhân tạo (AI) là lĩnh vực khoa học máy tính nghiên cứu cách tạo ra các hệ thống thông minh.",
                "AI có thể học hỏi, suy luận và giải quyết vấn đề tương tự con người.",
                "Các ứng dụng AI bao gồm nhận dạng giọng nói, thị giác máy tính và xử lý ngôn ngữ tự nhiên.",
            ],
            "difficulty": "medium",
        },
        {
            "id": "vn_nature_001",
            "category": "nature",
            "question": "Vịnh Hạ Long nằm ở đâu?",
            "ground_truth": "Vịnh Hạ Long nằm ở tỉnh Quảng Ninh, miền Bắc Việt Nam.",
            "context": [
                "Vịnh Hạ Long là di sản thiên nhiên thế giới tại Quảng Ninh.",
                "Vịnh Hạ Long có hơn 1600 đảo đá vôi với cảnh quan tuyệt đẹp.",
                "UNESCO đã công nhận Vịnh Hạ Long là di sản thiên nhiên thế giới năm 1994.",
            ],
            "difficulty": "easy",
        },
        {
            "id": "vn_science_001",
            "category": "science",
            "question": "Machine Learning hoạt động như thế nào?",
            "ground_truth": "Machine Learning là quá trình máy tính học từ dữ liệu để cải thiện hiệu suất mà không cần lập trình tường minh.",
            "context": [
                "Machine Learning cho phép máy tính học từ kinh nghiệm (dữ liệu) và cải thiện theo thời gian.",
                "Có ba loại ML chính: học có giám sát, học không giám sát và học tăng cường.",
                "ML sử dụng các thuật toán toán học để tìm ra các mẫu trong dữ liệu.",
            ],
            "difficulty": "hard",
        },
        {
            "id": "vn_economy_001",
            "category": "economy",
            "question": "GDP của Việt Nam là bao nhiêu?",
            "ground_truth": "GDP của Việt Nam năm 2023 đạt khoảng 430 tỷ USD.",
            "context": [
                "Tổng sản phẩm quốc nội (GDP) của Việt Nam liên tục tăng trưởng qua các năm.",
                "Việt Nam là một trong những nền kinh tế tăng trưởng nhanh nhất Đông Nam Á.",
                "Năm 2023, GDP Việt Nam ước đạt khoảng 430 tỷ USD.",
            ],
            "difficulty": "medium",
        },
        {
            "id": "vn_sports_001",
            "category": "sports",
            "question": "Môn thể thao phổ biến nhất Việt Nam là gì?",
            "ground_truth": "Bóng đá là môn thể thao phổ biến nhất tại Việt Nam.",
            "context": [
                "Bóng đá là môn thể thao vua tại Việt Nam với hàng triệu người hâm mộ.",
                "Đội tuyển bóng đá Việt Nam đã đạt nhiều thành tích trong khu vực.",
                "Ngoài bóng đá, cầu lông và vovinam cũng rất phổ biến.",
            ],
            "difficulty": "easy",
        },
    ]

    def __init__(self):
        """Initialize the benchmark."""
        self.questions = [
            BenchmarkQuestion(
                id=q["id"],
                category=q["category"],
                question=q["question"],
                ground_truth=q["ground_truth"],
                context=q["context"],
                difficulty=q["difficulty"],
                metadata={},
            )
            for q in self.BENCHMARK_QUESTIONS
        ]

    def get_all(self) -> list[BenchmarkQuestion]:
        """Get all benchmark questions.

        Returns:
            List of all benchmark questions.
        """
        return self.questions

    def get_by_category(self, category: str) -> list[BenchmarkQuestion]:
        """Get questions by category.

        Args:
            category: Category name (geography, culture, history, etc.).

        Returns:
            List of questions in the category.
        """
        return [q for q in self.questions if q.category == category]

    def get_by_difficulty(self, difficulty: str) -> list[BenchmarkQuestion]:
        """Get questions by difficulty.

        Args:
            difficulty: Difficulty level (easy, medium, hard).

        Returns:
            List of questions at the difficulty level.
        """
        return [q for q in self.questions if q.difficulty == difficulty]

    def export_for_ragas(self) -> dict[str, list]:
        """Export benchmark in Ragas-compatible format.

        Returns:
            Dictionary with questions, ground_truths, and contexts.
        """
        return {
            "questions": [q.question for q in self.questions],
            "ground_truths": [q.ground_truth for q in self.questions],
            "contexts": [q.context for q in self.questions],
        }

    def save_to_file(self, filepath: str | Path) -> None:
        """Save benchmark to JSON file.

        Args:
            filepath: Path to save the benchmark.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "id": q.id,
                "category": q.category,
                "question": q.question,
                "ground_truth": q.ground_truth,
                "context": q.context,
                "difficulty": q.difficulty,
            }
            for q in self.questions
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Benchmark saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> "VietnameseBenchmark":
        """Load benchmark from JSON file.

        Args:
            filepath: Path to the benchmark file.

        Returns:
            VietnameseBenchmark instance.
        """
        filepath = Path(filepath)

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        benchmark = cls()
        benchmark.questions = [
            BenchmarkQuestion(
                id=q["id"],
                category=q["category"],
                question=q["question"],
                ground_truth=q["ground_truth"],
                context=q["context"],
                difficulty=q["difficulty"],
                metadata=q.get("metadata", {}),
            )
            for q in data
        ]

        return benchmark

    def get_statistics(self) -> dict[str, Any]:
        """Get benchmark statistics.

        Returns:
            Dictionary with statistics.
        """
        categories = {}
        difficulties = {}

        for q in self.questions:
            categories[q.category] = categories.get(q.category, 0) + 1
            difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1

        return {
            "total_questions": len(self.questions),
            "categories": categories,
            "difficulties": difficulties,
        }


def main():
    """CLI for Vietnamese benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Vietnamese RAG Benchmark")
    parser.add_argument(
        "--export",
        type=str,
        help="Export benchmark to JSON file",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show benchmark statistics",
    )
    args = parser.parse_args()

    benchmark = VietnameseBenchmark()

    if args.stats:
        stats = benchmark.get_statistics()
        print("\n=== Vietnamese Benchmark Statistics ===")
        print(f"Total Questions: {stats['total_questions']}")
        print("\nBy Category:")
        for cat, count in stats['categories'].items():
            print(f"  {cat}: {count}")
        print("\nBy Difficulty:")
        for diff, count in stats['difficulties'].items():
            print(f"  {diff}: {count}")

    if args.export:
        benchmark.save_to_file(args.export)
        print(f"\nBenchmark exported to {args.export}")


if __name__ == "__main__":
    main()
