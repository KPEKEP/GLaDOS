# GLaDOS (ГЛаДОС)
[View in Russian](README.ru.md)

![image](https://github.com/KPEKEP/GLaDOS/assets/2512552/5cbb9d30-0e8d-4ede-9f13-47f9f93851dc)

This script emulates GLaDOS from Portal 1 and 2 using AI. At specified intervals, the AI analyzes the screen and provides a sarcastic comment in monologue mode, using GLaDOS's voice.

Example of a response:

[description.webm](https://github.com/KPEKEP/GLaDOS/assets/2512552/5fed804e-b928-45a3-acd8-0d1c63805afa)

## Installation

To install, follow these steps:

1. Clone the repository to your device.
2. Install the necessary dependencies by running `pip install -r requirements.txt`.
3. Create a `config.yaml` file based on `config_template.yaml` and fill in the required parameters such as `api_key`, `temperature`, `top_p`, `top_k`, and others.
4. Launch the application by running `python glados.py`.

## Usage

Once launched, GLaDOS will periodically describe the content of the screen and vocalize the obtained descriptions using synthesized speech. 
You can adjust the frequency of comments and other parameters in the `config.yaml` file.

## Configuration

The `config.yaml` file allows you to customize various aspects of the application:

- Choose between Ollama and Gemini as the inference engine
- Set screenshot capture settings
- Adjust sleep intervals between comments
- Configure model-specific parameters for both Ollama and Gemini
- Enable or disable translation of responses to Russian
- Customize TTS settings

## Features

- Screen capture and analysis
- AI-powered image description using either Ollama or Gemini
- Character role-playing as GLaDOS
- Text-to-speech synthesis using GLaDOS's voice
- Optional translation of responses from English to Russian
- Configurable comment frequency and sleep intervals

## File Descriptions

- `glados.py`: The main executable file of the application.
- `config.yaml`: Configuration file containing application parameters.
- `config_template.yaml`: Template for creating your own configuration file.
- `requirements.txt`: List of Python dependencies for the project.

## Dependencies

- yaml
- pillow
- TeraTTS
- ruaccent
- google-generativeai
- transliterate
- ollama
- transformers

## Acknowledgments

* [ruaccent](https://github.com/Den4ikAI/ruaccent) for Russian text accentuation
* [TeraTTS](https://github.com/Tera2Space/TeraTTS) for text-to-speech synthesis
* [Gemini](https://deepmind.google/technologies/gemini/) by Google DeepMind for AI-powered image analysis and text generation
* [Ollama](https://ollama.ai/) for local AI model inference
