# Third-party notices

This app is licensed under the GNU General Public License, version 3 or later (see `LICENSE`).
It builds on the projects below, which keep their own licenses. The same list is shown by the
Credits button of the app (`src/anima_trainer/about/credits.py`).

## Citron Anima LoRA Trainer UI

<https://github.com/citronlegacy/citron-anima-lora-trainer-ui> - the Gradio trainer this app is
based on: its features, settings, defaults and TOML configs come from Citron's work.

```text
MIT License

Copyright (c) 2026 Citron Legacy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Downloaded by the app (not part of this repository)

| Project | Used for | License |
| --- | --- | --- |
| [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) | training backend, installed at a fixed commit | Apache-2.0 |
| [Anima](https://huggingface.co/circlestone-labs/Anima), by CircleStone Labs and Comfy Org | base DiT models | CircleStone Labs Non-Commercial License, and the NVIDIA Open Model License (derivative of NVIDIA Cosmos-Predict2-2B-Text2Image) |
| [Qwen3 0.6B Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), by the Qwen team | text encoder (file from the Anima repository) | Apache-2.0 |
| [Qwen-Image VAE](https://huggingface.co/Qwen/Qwen-Image), by the Qwen team | VAE (file from the Anima repository) | Apache-2.0 |

The Anima models and their derivatives may be used only for non-commercial purposes; check the
model license before using a LoRA trained on them.

## Python dependencies

| Package | License |
| --- | --- |
| PySide6 (Qt for Python) | LGPL-3.0 |
| pydantic | MIT |
| Textual | MIT |
| toml | MIT |
