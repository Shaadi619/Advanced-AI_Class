from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_ai_assistant import AppConfig, FeedbackStore, ImageGenerationService, TextGenerationService, save_generated_image


st.set_page_config(
    page_title="Smart AI Assistant with Human Feedback",
    page_icon="AI",
    layout="wide",
)

config = AppConfig()
store = FeedbackStore(config.feedback_db_path)


@st.cache_resource(show_spinner=False)
def get_text_service(model_id: str) -> TextGenerationService:
    return TextGenerationService(model_id=model_id)


@st.cache_resource(show_spinner=False)
def get_image_service(model_id: str) -> ImageGenerationService:
    return ImageGenerationService(model_id=model_id)


def _initialize_session_state() -> None:
    defaults = {
        "current_prompt": "",
        "generated_responses": [],
        "generated_image_path": None,
        "feedback_submitted": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_sidebar() -> tuple[str, str, int, int, int, int, float, int]:
    with st.sidebar:
        st.header("Configuration")
        text_model_id = st.text_input("Text model", value=config.text_model_id)
        image_model_id = st.text_input("Image model", value=config.image_model_id)
        response_count = st.slider("Number of text responses", min_value=2, max_value=3, value=config.text_response_count)
        max_new_tokens = st.slider("Max new tokens per response", min_value=30, max_value=120, value=config.text_max_new_tokens, step=5)
        candidate_pool_size = st.slider("Candidate pool size", min_value=6, max_value=18, value=config.text_candidate_pool_size, step=3)
        min_words = st.slider("Minimum response words", min_value=8, max_value=20, value=config.text_min_words)
        inference_steps = st.slider(
            "Diffusion steps",
            min_value=10,
            max_value=50,
            value=config.image_num_inference_steps,
            step=5,
        )
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=1.0,
            max_value=12.0,
            value=float(config.image_guidance_scale),
            step=0.5,
        )

        metrics = store.get_dashboard_metrics()
        st.divider()
        st.subheader("Project Metrics")
        st.metric("Prompts stored", metrics["prompt_count"])
        st.metric("Responses stored", metrics["response_count"])
        st.metric("Feedback entries", metrics["feedback_count"])
        st.caption("The ranked feedback table acts as the RLHF reward-model simulation.")
        with st.expander("Prompt Tips"):
            st.write("Use concrete visual prompts such as 'A robot helping humans in daily life' or 'An astronaut walking on Mars at sunset'.")
            st.write("Avoid abstract prompts like 'Explain machine learning' because they are weak for image generation and small text models.")

    return text_model_id, image_model_id, response_count, max_new_tokens, candidate_pool_size, min_words, guidance_scale, inference_steps


def _handle_generation(
    prompt: str,
    text_model_id: str,
    image_model_id: str,
    response_count: int,
    max_new_tokens: int,
    candidate_pool_size: int,
    min_words: int,
    guidance_scale: float,
    inference_steps: int,
) -> None:
    text_service = get_text_service(text_model_id)
    image_service = get_image_service(image_model_id)

    with st.spinner("Generating text candidates with the LLM..."):
        text_responses = text_service.generate_responses(
            prompt=prompt,
            response_count=response_count,
            max_new_tokens=max_new_tokens,
            candidate_pool_size=candidate_pool_size,
            min_words=min_words,
        )

    saved_responses = store.save_generated_responses(prompt, text_responses)

    with st.spinner("Generating an image with Stable Diffusion..."):
        image = image_service.generate_image(
            prompt=prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
        )
        image_path = save_generated_image(image, config.image_output_dir, prompt)

    st.session_state["current_prompt"] = prompt
    st.session_state["generated_responses"] = saved_responses
    st.session_state["generated_image_path"] = image_path
    st.session_state["feedback_submitted"] = False


def main() -> None:
    _initialize_session_state()

    st.title("Smart AI Assistant with Human Feedback")
    st.write(
        "Enter one multimodal prompt to generate multiple LLM responses, collect human feedback, "
        "rank the responses like a simplified RLHF loop, and generate a matching Stable Diffusion image."
    )

    text_model_id, image_model_id, response_count, max_new_tokens, candidate_pool_size, min_words, guidance_scale, inference_steps = _render_sidebar()

    prompt = st.text_input(
        "Shared prompt for text and image generation",
        value=st.session_state["current_prompt"],
        placeholder="Example: A futuristic smart city with flying cars",
    )

    if st.button("Generate Text and Image", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt before generating.")
        else:
            try:
                _handle_generation(
                    prompt=prompt.strip(),
                    text_model_id=text_model_id,
                    image_model_id=image_model_id,
                    response_count=response_count,
                    max_new_tokens=max_new_tokens,
                    candidate_pool_size=candidate_pool_size,
                    min_words=min_words,
                    guidance_scale=guidance_scale,
                    inference_steps=inference_steps,
                )
                st.success("Generation completed. You can now provide feedback and view the ranking table.")
            except Exception as error:
                st.exception(error)

    if st.session_state["generated_responses"]:
        st.subheader("Generated Text Responses")
        for index, response in enumerate(st.session_state["generated_responses"], start=1):
            st.markdown(f"**Response {index}**")
            st.write(response.response_text)

        st.subheader("Human Feedback")
        response_options = {response.id: f"Response {index + 1}" for index, response in enumerate(st.session_state["generated_responses"])}
        best_response_id = st.radio(
            "Select the best response",
            options=list(response_options.keys()),
            format_func=lambda response_id: response_options[response_id],
            horizontal=True,
        )

        ratings_by_response_id: dict[int, int] = {}
        for index, response in enumerate(st.session_state["generated_responses"], start=1):
            ratings_by_response_id[response.id] = st.slider(
                f"Rate Response {index}",
                min_value=1,
                max_value=5,
                value=3,
                key=f"rating-{response.id}",
            )

        if st.button("Submit Feedback", use_container_width=True):
            try:
                store.record_feedback(ratings_by_response_id, best_response_id)
                st.session_state["feedback_submitted"] = True
                st.success("Feedback saved. The ranking table below has been updated.")
            except Exception as error:
                st.exception(error)

    if st.session_state["generated_image_path"]:
        st.subheader("Generated Image")
        image_path = Path(st.session_state["generated_image_path"])
        st.image(str(image_path), caption=f"Saved to {image_path}")

    if st.session_state["current_prompt"]:
        st.subheader("RLHF Ranking Table")
        ranked_responses = store.get_ranked_responses(st.session_state["current_prompt"])
        if ranked_responses:
            st.table([response.as_table_row() for response in ranked_responses])
            st.caption(
                "Ranking priority: best-response votes, then average rating, then number of ratings. "
                "This serves as the simplified reward model for the project."
            )
        else:
            st.info("Generate responses first to populate the ranking table.")


if __name__ == "__main__":
    main()
