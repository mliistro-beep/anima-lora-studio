# Changelog

All notable changes to Anima LoRA Studio are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The milestones (M0-M7) are
described in [DOCS/milestones.md](DOCS/milestones.md).

<!-- When v0.2.0 is tagged: replace "Unreleased" with the date and point the [0.2.0] link
     at compare/v0.1.0...v0.2.0. -->
## [0.2.0] - Unreleased

The release name is now Anima LoRA Studio. The package (`anima_trainer`), the command
(`anima-trainer`), the data folder (`~/.local/share/anima-trainer`) and the settings
(`~/.config/anima-trainer`) keep their names, so existing installs and version 0.1.0 keep working.

### Added

- Epoch previews during the training (M5), off by default: prompts, size, sampler, steps, CFG
  scale and seed in the new Previews tab, with a gallery of epochs x prompts that updates by
  itself. With previews off the TOML configs stay identical to Citron's and to 0.1.0.
- Generator tab (M5): generate with an epoch checkpoint, the final LoRA or no LoRA, and Compare
  Epochs (one image per checkpoint with the same prompt and seed); Stop.
- ER-SDE sampler at 25 steps (the default, ported from ComfyUI) besides Euler, for previews and
  the generator.
- Adaptive VRAM use for previews and the generator: profiles from the fastest to the lightest
  (`standard`, `block_swap`, `low_vram`), chosen from the free VRAM and retried with the next
  profile after an out-of-memory; a preview that runs out of memory is skipped without stopping
  the training. A system RAM check before each generation.
- A warning in the Training status and the Settings tab when the swap is under 8 GB or no
  out-of-memory daemon (earlyoom or systemd-oomd) runs, with the setup steps in
  `DOCS/SYSTEM_SETUP.md` (M6).
- `run_linux.sh` (M6): starts the app from the checkout, creating `.venv` and reinstalling only
  when `pyproject.toml` or the checkout path changed; `--install-desktop` / `--uninstall-desktop`
  add or remove the applications menu entry. Started from the menu, its output also goes to
  `~/.local/state/anima-trainer/launcher.log` and errors become notifications.
- App icon for the window and the menu entry.
- Preview and generator settings are kept in `<data folder>/preview.json` and
  `generator.json`; `config.json` keeps Citron's format.
- `CHANGELOG.md`.

### Changed

- Display name "Anima LoRA Studio" in the window title, the dialogs and the menu entry.

### Fixed

- Downloaded models are kept only after a size and SHA-256 match with the catalog, like the
  model files the user links.
- Booleans are rejected in decimal settings fields, as they already were in whole-number fields.

## [0.1.0] - 2026-10-27

First version: a PySide6 desktop app with the features, settings and TOML configs of Citron's
Anima LoRA Trainer UI, independent from Citron.

### Added

- Training, Advanced and Settings tabs with Citron's settings; Configure writes TOML configs
  byte-identical to Citron's, Start runs sd-scripts through `accelerate` with live progress, loss,
  log and Stop (M1-M3).
- Own training runtime in the app's data folder (M4): Install / Repair from the Settings tab
  creates the training venv, fetches sd-scripts at a pinned commit with its requirements and
  checks the GPU, resuming where it stopped; downloads of the Anima DiT, Qwen3 and the VAE with
  resume and cancel.
- Model files already on disk can be used in place after a size and SHA-256 check (M4b).
- GPL-3.0-or-later license, License and Credits dialogs, `THIRD_PARTY_NOTICES.md` and the
  development launcher `scripts/dev-run.sh` (M4c).

[0.2.0]: https://github.com/mliistro-beep/anima-lora-studio/compare/v0.1.0...feat/desktop-app
[0.1.0]: https://github.com/mliistro-beep/anima-lora-studio/releases/tag/v0.1.0
