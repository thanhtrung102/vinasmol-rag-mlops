#!/usr/bin/env python3
"""Download sample Vietnamese data for development and testing."""

import json
from pathlib import Path

# Sample Vietnamese texts for development
SAMPLE_DATA = [
    {
        "text": "Việt Nam là một quốc gia nằm ở Đông Nam Á. Thủ đô của Việt Nam là Hà Nội. "
        "Việt Nam có diện tích khoảng 331,212 km² và dân số hơn 98 triệu người.",
        "source": "sample",
        "topic": "geography",
    },
    {
        "text": "Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính nhằm tạo ra các "
        "hệ thống có khả năng thực hiện các nhiệm vụ đòi hỏi trí thông minh của con người.",
        "source": "sample",
        "topic": "technology",
    },
    {
        "text": "Phở là một món ăn truyền thống của Việt Nam, gồm bánh phở, nước dùng và thịt. "
        "Phở được coi là một trong những món ăn đặc trưng nhất của ẩm thực Việt Nam.",
        "source": "sample",
        "topic": "culture",
    },
    {
        "text": "Machine learning là một nhánh của trí tuệ nhân tạo, cho phép máy tính học từ dữ liệu "
        "mà không cần được lập trình cụ thể. Deep learning là một kỹ thuật của machine learning.",
        "source": "sample",
        "topic": "technology",
    },
    {
        "text": "Hồ Chí Minh, tên khai sinh là Nguyễn Sinh Cung, là nhà cách mạng và chính trị gia "
        "Việt Nam. Ông là người sáng lập Đảng Cộng sản Việt Nam và nước Việt Nam Dân chủ Cộng hòa.",
        "source": "sample",
        "topic": "history",
    },
]

SAMPLE_QA = [
    {
        "question": "Thủ đô của Việt Nam là gì?",
        "answer": "Thủ đô của Việt Nam là Hà Nội.",
        "context": [
            "Việt Nam là một quốc gia nằm ở Đông Nam Á. Thủ đô của Việt Nam là Hà Nội."
        ],
    },
    {
        "question": "Phở là món ăn như thế nào?",
        "answer": "Phở là một món ăn truyền thống của Việt Nam, gồm bánh phở, nước dùng và thịt.",
        "context": [
            "Phở là một món ăn truyền thống của Việt Nam, gồm bánh phở, nước dùng và thịt."
        ],
    },
    {
        "question": "Machine learning là gì?",
        "answer": "Machine learning là một nhánh của trí tuệ nhân tạo, cho phép máy tính học từ dữ liệu.",
        "context": [
            "Machine learning là một nhánh của trí tuệ nhân tạo, cho phép máy tính học từ dữ liệu "
            "mà không cần được lập trình cụ thể."
        ],
    },
]


def main():
    """Create sample data files."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Save training data
    train_file = data_dir / "train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for item in SAMPLE_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Created {train_file}")

    # Save evaluation data
    eval_file = data_dir / "eval_rag.json"
    eval_data = {
        "questions": [q["question"] for q in SAMPLE_QA],
        "answers": [q["answer"] for q in SAMPLE_QA],
        "contexts": [q["context"] for q in SAMPLE_QA],
    }
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"Created {eval_file}")

    print("\nSample data created successfully!")
    print(f"  - Training samples: {len(SAMPLE_DATA)}")
    print(f"  - QA pairs: {len(SAMPLE_QA)}")


if __name__ == "__main__":
    main()
