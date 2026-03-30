from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    text_model_id: str = os.getenv("TEXT_MODEL_ID", "gpt2")
    image_model_id: str = os.getenv("IMAGE_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    feedback_db_path: Path = Path(os.getenv("FEEDBACK_DB_PATH", "data/feedback.db"))
    image_output_dir: Path = Path(os.getenv("IMAGE_OUTPUT_DIR", "outputs/generated_images"))
    text_response_count: int = int(os.getenv("TEXT_RESPONSE_COUNT", "3"))
    text_max_new_tokens: int = int(os.getenv("TEXT_MAX_NEW_TOKENS", "90"))
    image_num_inference_steps: int = int(os.getenv("IMAGE_NUM_INFERENCE_STEPS", "30"))
    image_guidance_scale: float = float(os.getenv("IMAGE_GUIDANCE_SCALE", "7.5"))
