"""The text chunks generate.py writes into the PNG: `parameters` (the A1111 / Forge layout that
Civitai and image viewers read) and the app's own JSON, to load the entry again later.

Runs in the training venv next to generate.py and imports nothing of the app, which tests it
through anima_trainer.runtime_scripts.png_text and builds the texts in preview/image_metadata.
"""

PARAMETERS_KEY = "parameters"
SETTINGS_KEY = "anima_lora_studio"


def png_texts(parameters: str, settings: str) -> dict[str, str]:
    """The chunks to write, by key; empty texts are left out."""
    texts = {PARAMETERS_KEY: parameters, SETTINGS_KEY: settings}
    return {key: text for key, text in texts.items() if text}
