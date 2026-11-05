"""Who made this app and what it builds on: Citron's trainer, sd-scripts and the models it
downloads. Licenses as published by each project (checked on 2026-10-27)."""

from dataclasses import dataclass

from anima_trainer.models.catalog import BASE_MODEL_URLS, QWEN3_FILENAME, VAE_FILENAME

_ANIMA_URL = "https://huggingface.co/circlestone-labs/Anima"


@dataclass(frozen=True)
class Credit:
    name: str
    role: str
    url: str  # empty when there is no page to link
    license: str
    license_urls: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    model_keys: tuple[str, ...] = ()  # catalog keys of the downloads this credit covers


@dataclass(frozen=True)
class CreditSection:
    title: str
    credits: tuple[Credit, ...]


AUTHOR = Credit("Mauro Liistro", "Author of this desktop app", "", "GPL-3.0-or-later")

CITRON = Credit(
    "Citron Anima LoRA Trainer UI, by Citron Legacy",
    "The Gradio trainer this app is based on: its features, settings, defaults and TOML "
    "configs come from Citron's work",
    "https://github.com/citronlegacy/citron-anima-lora-trainer-ui",
    "MIT",
    notes=("Copyright (c) 2026 Citron Legacy. Released under the MIT License.",),
)

SD_SCRIPTS = Credit(
    "sd-scripts, by kohya-ss",
    "Training backend, installed by the app at a fixed commit",
    "https://github.com/kohya-ss/sd-scripts",
    "Apache-2.0",
)

QT = Credit(
    "Qt for Python (PySide6), by The Qt Company",
    "User interface toolkit",
    "https://doc.qt.io/qtforpython-6/",
    "LGPL-3.0",
)

TEXTUAL = Credit(
    "Textual, by Textualize",
    "Terminal interface toolkit",
    "https://textual.textualize.io/",
    "MIT",
)

MODELS = (
    Credit(
        "Anima, by CircleStone Labs and Comfy Org",
        f"Base DiT models ({', '.join(BASE_MODEL_URLS)})",
        _ANIMA_URL,
        "CircleStone Labs Non-Commercial License; NVIDIA Open Model License (derivative of "
        "NVIDIA Cosmos-Predict2-2B-Text2Image)",
        license_urls=(
            f"{_ANIMA_URL}/blob/main/LICENSE.md",
            "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
        ),
        notes=(
            "The model and its derivatives may be used only for non-commercial purposes; "
            "check the model license before using a LoRA trained on it.",
        ),
        model_keys=tuple(BASE_MODEL_URLS),
    ),
    Credit(
        "Qwen3 0.6B Base, by the Qwen team (Alibaba Cloud)",
        "Text encoder, downloaded from the Anima repository",
        "https://huggingface.co/Qwen/Qwen3-0.6B-Base",
        "Apache-2.0",
        model_keys=(QWEN3_FILENAME,),
    ),
    Credit(
        "Qwen-Image VAE, by the Qwen team (Alibaba Cloud)",
        "VAE, downloaded from the Anima repository",
        "https://huggingface.co/Qwen/Qwen-Image",
        "Apache-2.0",
        model_keys=(VAE_FILENAME,),
    ),
)

CREDIT_SECTIONS = (
    CreditSection("Author", (AUTHOR,)),
    CreditSection("Based on", (CITRON,)),
    CreditSection("Models downloaded by the app", MODELS),
    CreditSection("Software", (SD_SCRIPTS, QT, TEXTUAL)),
)
