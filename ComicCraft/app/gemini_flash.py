from google import genai
from google.genai import types
from .config import GEMINI_API_KEY, GEMINI_OUTLINE_MODEL, USE_MOCK_AI
from .schemas import ComicOutline


def create_mock_outline(prompt, character, setting, tone, art_style, panels):
    return ComicOutline(panels=[
        {
            "panel_number": i,
            "title": f"Panel {i}",
            "scene_description": f"{character} continues the {tone.lower()} adventure in {setting}.",
            "image_prompt": f"{character} in {setting}, {art_style} comic art, panel {i}"
        }
        for i in range(1, panels + 1)
    ])


def generate_outline(prompt, character, setting, tone, art_style, panels):
    if USE_MOCK_AI or not GEMINI_API_KEY:
        return create_mock_outline(prompt, character, setting, tone, art_style, panels)

    client = genai.Client(api_key=GEMINI_API_KEY)
    schema = {
        "type": "object",
        "properties": {
            "panels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "panel_number": {"type": "integer"},
                        "title": {"type": "string"},
                        "scene_description": {"type": "string"},
                        "image_prompt": {"type": "string"}
                    },
                    "required": ["panel_number", "title", "scene_description", "image_prompt"]
                }
            }
        },
        "required": ["panels"]
    }

    instruction = f"""
Create a {panels}-panel comic outline.
Story prompt: {prompt}
Main character: {character}
Setting: {setting}
Tone: {tone}
Art style: {art_style}

For every panel provide panel_number, title, scene_description and image_prompt.
Keep the character and story consistent across panels.
Return ONLY JSON.
"""

    models_to_try = [GEMINI_OUTLINE_MODEL, "gemini-3.5-flash", "gemini-3.5-flash-lite"]
    last_error = None
    for model_name in models_to_try:
        try:
            print(f"[ComicCraft] Trying Gemini model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=instruction,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.9
                )
            )
            print(f"[ComicCraft] Gemini success: {model_name}")
            return ComicOutline.model_validate_json(response.text)
        except Exception as error:
            last_error = error
            print(f"[ComicCraft] Gemini failed: {model_name}")
            print(error)
            continue
    raise RuntimeError(f"Gemini outline generation failed: {last_error}")
