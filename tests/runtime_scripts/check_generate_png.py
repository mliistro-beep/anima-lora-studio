"""Run by the training venv's Python (see test_generate_png.py): saves a tiny image with
generate._save and prints the PNG's text chunks as JSON; no GPU or model."""

import json
import sys
from pathlib import Path

import generate
import torch
from PIL import Image

output = Path(sys.argv[1])
generate._save(torch.zeros(3, 8, 8), output, {"parameters": "a cat", "anima_lora_studio": "{}"})
with Image.open(output) as image:
    print(json.dumps(dict(image.text)))
