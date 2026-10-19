# Anima LoRA Trainer (desktop)

Desktop (PySide6) version of Citron's Anima LoRA Trainer UI. It reuses an existing
Citron installation (venv, sd-scripts, models) and adds epoch previews and a simple
image generator.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```
