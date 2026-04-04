# Smart AI Assistant with Human Feedback

This file is a presentation-ready version of the project that you can paste directly into Google Slides, PowerPoint, or Canva.

## Slide 1 - Title

Title:
Smart AI Assistant with Human Feedback (Multimodal)

Subtitle:
Advanced AI Project

On-slide points:
- Text generation with an instruction-following LLM
- Human feedback and RLHF simulation
- Image generation with Stable Diffusion

Speaker notes:
This project demonstrates a simplified multimodal AI assistant. The system generates text and images from the same prompt, collects human feedback on text responses, and ranks the responses to simulate the main idea behind RLHF.

Suggested visual:
- A title slide with icons for text, feedback, and image generation

## Slide 2 - Objective

Title:
Project Objective

On-slide points:
- Build a multimodal AI assistant using pretrained models
- Generate both text and image outputs from one prompt
- Collect human feedback on text quality
- Simulate RLHF with stored preferences and rankings

Speaker notes:
The goal was not to train models from scratch. Instead, the project uses existing Hugging Face models and focuses on combining text generation, image generation, and human feedback into one system.

Suggested visual:
- A simple flow from prompt to text, feedback, and image

## Slide 3 - Why This Problem Matters

Title:
Problem Statement

On-slide points:
- AI assistants often work across multiple modalities
- Raw model outputs can vary in quality
- Human feedback helps identify better responses
- Multimodal prompts must work for both language and vision

Speaker notes:
A strong AI assistant should not only generate outputs, but also improve selection quality. Human feedback is useful because some responses are better aligned, clearer, or more helpful than others.

Suggested visual:
- Side-by-side example of weak and strong generated responses

## Slide 4 - System Architecture

Title:
System Architecture

On-slide points:
- User enters one shared prompt
- LLM generates 3 text responses
- Stable Diffusion generates 1 image
- User rates responses and selects the best one
- Feedback is stored and responses are ranked

Speaker notes:
The same prompt is sent to both branches of the system. One branch handles language generation, and the other handles image generation. After that, the user gives feedback only on the text side, and the ranking table acts as the simplified reward model.

Suggested diagram:

```text
User Prompt
   |----> Text Model ----> 3 Responses ----> Human Feedback ----> Ranking Table
   |
   |----> Diffusion Model ----> 1 Generated Image
```

## Slide 5 - Models Used

Title:
Models and Tools

On-slide points:
- Text model: `google/flan-t5-base`
- Image model: `runwayml/stable-diffusion-v1-5`
- Libraries: Transformers, Diffusers, Streamlit
- Storage: SQLite

Speaker notes:
I used `flan-t5-base` because it is instruction-following and produces cleaner responses than smaller baseline autocomplete-style models. Stable Diffusion v1.5 was used for image generation. SQLite was used to store the feedback and ranking data.

Suggested visual:
- Table with model names and their roles

## Slide 6 - Text Generation

Title:
Part 1 - Text Generation

On-slide points:
- Input: one user prompt
- Output: 2 to 3 candidate responses
- Prompt rewritten into short scene-description instructions
- Candidate filtering improves quality

Speaker notes:
The text system does more than one simple generation call. It creates a pool of candidates, applies prompt variants, and filters weak outputs so the final displayed responses are cleaner and more relevant.

Suggested visual:
- Screenshot of generated responses from the app or notebook

## Slide 7 - RLHF Simulation

Title:
Part 1 - RLHF Simulation

On-slide points:
- User rates each response from 1 to 5
- User selects the best response
- Feedback is stored in SQLite
- Ranking updates using votes and average ratings

Speaker notes:
This is called an RLHF simulation because the project includes the preference-collection part, but it does not retrain the language model. Instead, it uses the collected human signals to rank responses and simulate a reward model.

Suggested visual:
- Screenshot of the feedback interface and ranking table

## Slide 8 - Image Generation

Title:
Part 2 - Image Generation

On-slide points:
- Same prompt used for image generation
- Stable Diffusion creates one output image
- No feedback required on the image side
- Prompt must be visual and descriptive

Speaker notes:
The assignment required one prompt that works for both text and image generation. That is why the prompt should describe a scene that can be imagined visually, such as a robot helping humans or an astronaut on Mars.

Suggested visual:
- One generated image from your final run

## Slide 9 - Example Workflow

Title:
Example Run

On-slide points:
- Prompt: `A robot helping humans in daily life`
- Generate 3 text responses
- Generate 1 matching image
- Rate the text responses
- Select the best response
- Show updated ranking table

Speaker notes:
This slide is where you walk through one full example from input to final ranking. It helps connect all parts of the system in one clear story.

Suggested visual:
- 3-step montage: prompt, outputs, ranking table

## Slide 10 - Implementation Details

Title:
Implementation

On-slide points:
- Streamlit app for live demo
- Colab notebook for GPU-friendly execution
- Modular code structure
- Config, generation, storage, testing, and presentation assets

Speaker notes:
The project was organized into separate modules so each part is easier to explain and maintain. The Streamlit app supports demonstration, and the Colab notebook is useful when a GPU environment is needed for Stable Diffusion.

Suggested visual:
- Small repo structure snapshot

## Slide 11 - Challenges and Observations

Title:
Challenges and Observations

On-slide points:
- Small text models can produce weak or noisy outputs
- Better prompts and filtering improve quality
- Stable Diffusion is compute-heavy without a GPU
- Human ratings are subjective but still useful

Speaker notes:
One important observation is that text quality depends a lot on the model and prompt design. I improved the system by using an instruction-following model, prompt variants, candidate filtering, and fallback handling.

Suggested visual:
- Before/after comparison of weak vs improved text responses

## Slide 12 - Conclusion

Title:
Conclusion

On-slide points:
- The project meets all main assignment requirements
- It combines text, feedback, and image generation in one workflow
- It demonstrates the core idea of RLHF in simplified form
- It can be extended with stronger models or true fine-tuning

Speaker notes:
In conclusion, the project successfully integrates multimodal generation with human feedback. While it is not full RLHF, it clearly demonstrates how preference signals can guide response selection in an AI system.

Suggested visual:
- Summary diagram or checklist of completed rubric items

## Slide 13 - Demo Plan

Title:
Live Demo

On-slide points:
- Enter one multimodal prompt
- Generate text responses and image
- Rate each response
- Select the best one
- Show ranking table update

Speaker notes:
Keep the demo short and smooth. Choose one prompt that is concrete and visual, then show the system end to end in under five minutes.

Suggested visual:
- Demo checklist

## Slide 14 - Questions

Title:
Questions

On-slide points:
- Why is this only a simulation of RLHF?
- Why use pretrained models?
- How could the reward model be improved?
- How could image feedback be added later?

Speaker notes:
This final slide helps you transition into discussion. Be ready to explain the difference between collecting feedback and actually retraining a policy model.
