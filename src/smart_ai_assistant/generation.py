from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _normalize_generated_text(text: str) -> str:
    cleaned = text.replace("\r", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.lstrip(":,- ")
    return cleaned


def _prepare_prompt(prompt: str) -> str:
    return f"Prompt: {prompt}\nAssistant:"


def _detect_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


class TextGenerationService:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = _detect_device(torch)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if self._device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self._model.to(self._device)
        self._model.eval()

    @property
    def device(self) -> str:
        self._load()
        return self._device

    def generate_responses(self, prompt: str, response_count: int = 3, max_new_tokens: int = 90) -> list[str]:
        self._load()

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        prompt_template = _prepare_prompt(prompt)
        encoded = self._tokenizer(prompt_template, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        candidate_responses: list[str] = []
        sampling_temperatures = [0.7, 0.9, 1.1, 0.8, 1.0, 1.2]

        for index in range(max(response_count * 2, 6)):
            seed = 42 + index
            self._torch.manual_seed(seed)
            if self._device == "cuda":
                self._torch.cuda.manual_seed_all(seed)

            output_ids = self._model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=sampling_temperatures[index % len(sampling_temperatures)],
                top_p=0.92,
                repetition_penalty=1.15,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                num_return_sequences=1,
            )
            decoded = self._tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)
            cleaned = _normalize_generated_text(decoded)
            if cleaned and cleaned not in candidate_responses:
                candidate_responses.append(cleaned)
            if len(candidate_responses) >= response_count:
                break

        return candidate_responses[:response_count]


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
