import os
import base64
import yaml
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- Configuration and Constants ---
AUDIO_DIR = "audio_files"
ANALYSIS_DIR = "analysis_results"


def setup_directories():
    """Creates necessary directories if they don't exist."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def encode_audio(audio_path):
    """Encodes the audio file to base64."""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')


def analyze_call(audio_path, config):
    """
    Analyzes the call center audio using the Gemini API with a streaming response.

    Args:
        audio_path (str): The path to the audio file.
        config (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing the parsed analysis report or an error message.
    """
    try:
        # Configure the Gemini API client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)

        # Create the generative model instance with the system instruction
        model = genai.GenerativeModel(
            model_name=config['model_name'],
            system_instruction=config['system_instruction']
        )

        # Determine the correct mime type based on file extension
        file_extension = os.path.splitext(audio_path)[1].lower()
        mime_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/m4a",
            ".ogg": "audio/ogg",
        }
        mime_type = mime_types.get(file_extension)
        if not mime_type:
            raise ValueError(f"Unsupported audio file format: {file_extension}")

        # Prepare the request contents, including the prompt and audio file
        contents = [
            config['prompt'],
            {
                "mime_type": mime_type,
                "data": encode_audio(audio_path)
            }
        ]

        # Generate the content using a streaming request
        # FIX: Removed the unnecessary 'tools' parameter to prevent tool-use errors.
        response_stream = model.generate_content(
            contents,
            stream=True
        )

        # Aggregate the response text from all chunks in the stream
        full_response_text = ""
        for chunk in response_stream:
            # Add a check to ensure chunk.text exists before appending
            if chunk.text:
                full_response_text += chunk.text

        if not full_response_text.strip():
            # Check for a more descriptive finish reason if the response is empty
            try:
                finish_reason = response_stream.candidates[0].finish_reason
                if finish_reason != 1:  # 1 is STOP
                    return {"error": f"API call finished prematurely. Reason: {finish_reason.name}"}
            except (IndexError, AttributeError):
                pass  # Fallback to generic error
            return {"error": "Received an empty response from the API."}

        # Save the analysis result
        audio_filename = os.path.basename(audio_path)
        analysis_filename = os.path.splitext(audio_filename)[0] + ".json"
        analysis_path = os.path.join(ANALYSIS_DIR, analysis_filename)

        # The response from Gemini is a YAML-formatted string, so we parse it.
        try:
            # Clean the response text to ensure it's valid YAML/JSON
            cleaned_text = full_response_text.strip().replace("```yaml", "").replace("```", "")
            analysis_data = yaml.safe_load(cleaned_text)
            with open(analysis_path, "w") as f:
                json.dump(analysis_data, f, indent=4)
        except (yaml.YAMLError, json.JSONDecodeError) as parse_error:
            print(f"Warning: Could not parse model output as YAML/JSON. Saving raw text. Error: {parse_error}")
            # Save raw text if parsing fails
            with open(analysis_path.replace('.json', '.txt'), "w") as f:
                f.write(full_response_text)
            return {"error": "Failed to parse model output", "raw_output": full_response_text}

        return analysis_data

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return {"error": str(e)}


# Initial setup when the module is loaded
setup_directories()
