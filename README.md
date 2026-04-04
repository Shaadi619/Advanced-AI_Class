# Smart AI Assistant with Human Feedback

This project implements the full assignment workflow from the rubric:

- Generate `2-3` text responses from a pretrained instruction-following LLM (`google/flan-t5-base` by default).
- Collect human feedback through ratings and a best-response choice.
- Rank responses using stored feedback to simulate an RLHF reward model.
- Generate an image from the same prompt using a Stable Diffusion model.
- Provide presentation and demo assets so the project is ready to submit.
- Include both a Streamlit app and a Colab-ready notebook version.

## Rubric Coverage

| Rubric Item | How this project covers it |
| --- | --- |
| Text Generation (LLM) | `TextGenerationService` loads a pretrained Hugging Face LLM and produces three sampled responses for one prompt. |
| RLHF Simulation (Text Feedback) | The app stores ratings and best-response votes in SQLite, then ranks responses with a reward-style proxy. |
| Image Generation (Diffusion) | `ImageGenerationService` uses Stable Diffusion to create one image from the same prompt. |
| Functionality and Code Quality | The project is separated into config, generation, storage, app UI, tests, and documentation. |
| Presentation and Questions | A ready-made slide deck outline, demo script, and Colab notebook are included in the repo. |

## Project Structure

```text
.
|-- app.py
|-- presentation/
|   `-- Smart_AI_Assistant_Slides.md
|-- notebooks/
|   `-- Smart_AI_Assistant_Colab.ipynb
|-- requirements.txt
|-- src/
|   `-- smart_ai_assistant/
|       |-- __init__.py
|       |-- config.py
|       |-- generation.py
|       `-- storage.py
|-- tests/
|   `-- test_storage.py
`-- README.md
```

## Models Used

- Text model: `google/flan-t5-base`
- Image model: `runwayml/stable-diffusion-v1-5`

You can change either model in the Streamlit sidebar if your environment needs a different or larger alternative.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Launch the Streamlit app.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Colab Notebook Version

If you want a notebook workflow for demo day, use:

- [Smart_AI_Assistant_Colab.ipynb](/Users/shaads/Desktop/advanced-AI/notebooks/Smart_AI_Assistant_Colab.ipynb)

The notebook is self-contained and follows the same project logic:

1. Install dependencies in Colab
2. Set one shared prompt
3. Generate 3 instruction-following LLM responses
4. Generate 1 Stable Diffusion image
5. Enter ratings and a best-response choice
6. Display the RLHF-style ranking table

## Recommended Runtime

Stable Diffusion is GPU-heavy. For the smoothest demo:

- Use Google Colab with GPU enabled, or
- Use a local machine with CUDA/MPS support and enough VRAM.

The code supports `cuda`, `mps`, and `cpu`, but image generation on CPU will be slow.

## Text Quality Improvements

The text pipeline now uses several quality controls to produce cleaner responses:

- An instruction-following model is used by default.
- The prompt is rewritten into short scene-description tasks.
- Multiple prompt variants are generated for diversity.
- A larger candidate pool is sampled, then ranked automatically.
- Low-quality outputs are filtered using simple heuristics for length, overlap with the prompt, coherence, and repeated junk text.
- Safe fallback responses are used only if the model still fails to produce enough usable answers.

## How the RLHF Simulation Works

1. The user enters a single prompt that works for both text and image generation.
2. The text model generates multiple candidate responses.
3. The user rates each response from `1-5` and selects the best one.
4. Feedback is stored in `data/feedback.db`.
5. The app ranks responses using:
   - best-response vote count
   - average rating
   - number of ratings

This ranking table acts as a simplified reward model similar to RLHF.

## Suggested Demo Prompt

- `A futuristic smart city with flying cars`
- `A robot helping humans in daily life`
- `An astronaut walking on Mars at sunset`

## Run the Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Deliverables Included

- Source code: Streamlit app plus modular Python package
- Colab notebook: [Smart_AI_Assistant_Colab.ipynb](/Users/shaads/Desktop/advanced-AI/notebooks/Smart_AI_Assistant_Colab.ipynb)
- Presentation: [presentation/Smart_AI_Assistant_Slides.md](/Users/shaads/Desktop/advanced-AI/presentation/Smart_AI_Assistant_Slides.md)
- Presentation-ready script: [presentation/Final_Presentation_Script.md](/Users/shaads/Desktop/advanced-AI/presentation/Final_Presentation_Script.md)
- Demo guidance: see the final section of the slide deck for a short live demo flow

## Possible Questions During Presentation

- Why use a pretrained instruction-following model instead of training from scratch?
- Why is this called an RLHF simulation and not full RLHF?
- How is the reward signal represented in the system?
- What are the limitations of using average ratings and votes?
- Why must the same prompt work for both text and image generation?
