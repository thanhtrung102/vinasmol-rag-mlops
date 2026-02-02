"""LLM-based generation for RAG responses."""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationResult:
    """Result from the RAG generator."""

    answer: str
    context_used: list[str]
    metadata: dict[str, Any]


class RAGGenerator:
    """Generate answers using retrieved context and LLM."""

    DEFAULT_PROMPT_TEMPLATE = """Bạn là trợ lý AI hữu ích. Sử dụng ngữ cảnh được cung cấp để trả lời câu hỏi.
Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói rằng bạn không biết.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

    def __init__(
        self,
        model_name: str = "vinai/PhoGPT-4B-Chat",
        device: str | None = None,
        load_in_8bit: bool = True,
        prompt_template: str | None = None,
    ):
        """Initialize the generator.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to use (cuda, cpu, or auto).
            load_in_8bit: Whether to load model in 8-bit quantization.
            prompt_template: Custom prompt template with {context} and {question}.
        """
        self.model_name = model_name
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with appropriate settings for Codespaces (16GB RAM)
        load_kwargs = {"device_map": "auto" if device == "cuda" else None}
        if load_in_8bit and device == "cuda":
            load_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            **load_kwargs,
        )

        if device == "cpu":
            self.model = self.model.to(device)

    def generate(
        self,
        question: str,
        context_docs: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """Generate an answer using the provided context.

        Args:
            question: User's question.
            context_docs: List of relevant context documents.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.

        Returns:
            Generated answer with metadata.
        """
        # Format context
        context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(context_docs))

        # Build prompt
        prompt = self.prompt_template.format(context=context, question=question)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return GenerationResult(
            answer=response.strip(),
            context_used=context_docs,
            metadata={
                "model": self.model_name,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        )
