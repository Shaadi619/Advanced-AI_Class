from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _normalize_generated_text(text: str) -> str:
    cleaned = text.replace("\r", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.lstrip(":,- ")
    return cleaned


def _prepare_prompt(prompt: str) -> str:
    return (
        "Task: Write one short, coherent scene description that matches the prompt.\n"
        "Requirements: 2 sentences, simple English, vivid but realistic, no dialogue, no lists, no quotes.\n"
        f"Prompt: {prompt}\n"
        "Answer:"
    )


def _detect_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_prompt_variants(prompt: str) -> list[str]:
    return [
        _prepare_prompt(prompt),
        (
            "Describe this scene in 2 clear sentences for a class demo. "
            "Focus on what is happening and what can be seen.\n"
            f"Prompt: {prompt}\n"
            "Answer:"
        ),
        (
            "Write a concise visual description of this idea in 2 sentences. "
            "Keep it fluent, relevant, and easy to understand.\n"
            f"Idea: {prompt}\n"
            "Description:"
        ),
    ]


def _extract_keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def _sentence_count(text: str) -> int:
    parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    return len(parts)


def _contains_gibberish(text: str) -> bool:
    weird_patterns = [
        r"\"\"",
        r"\[[0-9]{3,}\]",
        r"\b[a-zA-Z]{1,3}\?\b",
        r"(.)\1{3,}",
    ]
    lowercase = text.lower()
    if any(re.search(pattern, text) for pattern in weird_patterns):
        return True
    if lowercase.count("robot:") > 0 or lowercase.count("answer:") > 0:
        return True
    if len(re.findall(r"[^a-zA-Z0-9\s,.'-]", text)) > 6:
        return True
    return False


def _quality_score(prompt: str, text: str, min_words: int = 12) -> float:
    words = re.findall(r"\b\w+\b", text)
    if len(words) < min_words:
        return -100.0
    if _contains_gibberish(text):
        return -100.0

    score = 0.0
    sentence_count = _sentence_count(text)
    if 1 <= sentence_count <= 3:
        score += 3.0
    else:
        score -= 2.0

    prompt_keywords = _extract_keywords(prompt)
    response_keywords = _extract_keywords(text)
    overlap = len(prompt_keywords & response_keywords)
    score += overlap * 1.5

    word_count = len(words)
    if 16 <= word_count <= 45:
        score += 2.0
    elif word_count > 60:
        score -= 1.5

    if text[0].islower():
        score -= 1.0
    if not text.endswith((".", "!", "?")):
        score -= 1.0
    if any(name in text for name in ["Hitler", "Nazis", "World War", "Jim Jones"]):
        score -= 5.0
    if text.count('"') >= 2:
        score -= 2.0
    if len(set(words)) / max(len(words), 1) < 0.55:
        score -= 2.0

    return score


def _fallback_responses(prompt: str, response_count: int) -> list[str]:
    keywords = sorted(_extract_keywords(prompt))
    keyword_phrase = ", ".join(keywords[:4]) if keywords else "the main visual details"
    base = [
        f"This scene shows {prompt.lower()}. The description stays focused on {keyword_phrase} so the idea is easy to visualize.",
        f"The prompt describes {prompt.lower()}. It can be presented as a clear visual scene with simple details and a realistic setting.",
        f"This idea centers on {prompt.lower()}. The scene is described in direct language so it works well for both text and image generation.",
    ]
    return base[:response_count]


class TextGenerationService:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._tokenizer = None
        self._model = None
        self._is_encoder_decoder = False
        self._torch = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self._device = _detect_device(torch)
        config = AutoConfig.from_pretrained(self.model_id)
        self._is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if self._device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        if self._is_encoder_decoder:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, **model_kwargs)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self._model.to(self._device)
        self._model.eval()

    @property
    def device(self) -> str:
        self._load()
        return self._device

    def generate_responses(
        self,
        prompt: str,
        response_count: int = 3,
        max_new_tokens: int = 90,
        candidate_pool_size: int = 9,
        min_words: int = 12,
    ) -> list[str]:
        self._load()

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        scored_candidates: list[tuple[float, str]] = []
        seen_candidates: set[str] = set()
        prompt_variants = _build_prompt_variants(prompt)
        decode_strategies = [
            {"temperature": 0.25, "top_p": 0.8, "do_sample": True},
            {"temperature": 0.4, "top_p": 0.82, "do_sample": True},
            {"temperature": 0.55, "top_p": 0.85, "do_sample": True},
            {"temperature": 0.0, "top_p": 1.0, "do_sample": False},
        ]

        for index in range(max(candidate_pool_size, response_count * 3)):
            prompt_template = prompt_variants[index % len(prompt_variants)]
            encoded = self._tokenizer(prompt_template, return_tensors="pt")
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            prompt_length = encoded["input_ids"].shape[1]
            seed = 42 + index
            self._torch.manual_seed(seed)
            if self._device == "cuda":
                self._torch.cuda.manual_seed_all(seed)

            strategy = decode_strategies[index % len(decode_strategies)]
            output_ids = self._model.generate(
                **encoded,
                do_sample=strategy["do_sample"],
                max_new_tokens=max_new_tokens,
                temperature=strategy["temperature"],
                top_p=strategy["top_p"],
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                num_beams=4 if not strategy["do_sample"] else 1,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                num_return_sequences=1,
            )

            if self._is_encoder_decoder:
                decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                decoded = self._tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
            cleaned = _normalize_generated_text(decoded)
            if not cleaned or cleaned in seen_candidates:
                continue

            seen_candidates.add(cleaned)
            score = _quality_score(prompt, cleaned, min_words=min_words)
            if score > -50:
                scored_candidates.append((score, cleaned))

        scored_candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = [text for _, text in scored_candidates[:response_count]]

        if len(selected) < response_count:
            for fallback in _fallback_responses(prompt, response_count):
                if fallback not in selected:
                    selected.append(fallback)
                if len(selected) >= response_count:
                    break

        return selected[:response_count]


class ImageGenerationService:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._pipeline = None
        self._torch = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._pipeline is not None and self._torch is not None:
            return

        import torch
        from diffusers import StableDiffusionPipeline

        self._torch = torch
        self._device = _detect_device(torch)
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        self._pipeline = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=dtype)

        if self._device in {"cuda", "mps"}:
            self._pipeline = self._pipeline.to(self._device)
        else:
            self._pipeline.enable_attention_slicing()

    @property
    def device(self) -> str:
        self._load()
        return self._device

    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ):
        self._load()
        assert self._pipeline is not None
        result = self._pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        return result.images[0]


def save_generated_image(image, output_dir: str | Path, prompt: str) -> Path:
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", prompt.lower()).strip("-")[:40] or "prompt"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_path = destination_dir / f"{timestamp}-{slug}.png"
    image.save(image_path)
    return image_path
