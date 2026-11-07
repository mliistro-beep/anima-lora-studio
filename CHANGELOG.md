# Changelog

All notable changes to Anima LoRA Studio are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The Generator works without a training project: a base model of the catalog or a model file of
  yours, up to 8 LoRAs with their strength and an on/off switch (kohya, or PEFT renamed in memory),
  and its own output folder, with Use project folder and Open folder. A LoRA that cannot be applied
  (LoHa, LoKr, DoRA, a larger model...) stops the generation with a clear error.
- A Pre-prompt field and a Trigger for every LoRA: the prompt used is the pre-prompt, then the
  triggers of the LoRAs that are on (in their order), then the prompt; the log shows it. An added
  LoRA gets the trigger saved in its metadata (`modelspec.trigger_phrase`), if any.
- Compare Epochs shows a thumbnail for every checkpoint: a click shows it large, a double-click
  enlarges it (before, only the last image was left on screen).

### Changed

- Generated images go to `<data folder>/generated` unless a folder is chosen (before: the
  project's `<output>/generated`, still one click away with Use project folder).
- In the Generator, the project checkpoint's strength is called Checkpoint Strength.

- `anima-trainer train` checks that the GPU is free before asking for confirmation, and the terminal
  interface before showing the summary to approve, so a busy GPU no longer refuses the start only
  after the answer.

## [0.3.1] - 2026-11-06

### Added

- `./run_linux.sh --install-desktop` also adds the menu entry Anima LoRA Studio (terminal), which
  opens the terminal interface, and links `~/.local/bin/anima-trainer`, so `anima-trainer tui` and
  `anima-trainer train` work from any folder. A file there that is not such a link is kept.
  `--uninstall-desktop` removes both.

## [0.3.0] - 2026-11-05

### Added

- Training from the terminal, also over SSH with no display (M8): `anima-trainer tui`, a terminal
  interface built with Textual, and `anima-trainer train`, with `--yes` for scripts. Both show the
  samples and total steps to approve before starting.
- Presets: Style, Character, Concept and Person built in, your own as TOML files in
  `<data folder>/presets/`. Used from the terminal for now.
- A lock between processes on the GPU of a data folder: a training from the terminal refuses Start,
  Generate and Install in the window, and the other way round.

### Changed

- The `anima-trainer` command starts from `cli/entry.py`; without a command it opens the app as before.

### Fixed

- Group titles such as "Project & Paths" show the `&` instead of an underscore.
- Closing the terminal the app was started from no longer leaves the training running without a
  parent: SIGHUP now stops it. A signal already ignored stays ignored, so `nohup` keeps working.

## [0.2.0] - 2026-11-04

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
- A `RESOURCES` line in the training log at the start and every minute (M6): available RAM,
  used swap and, for each GPU, VRAM, temperature, power and load, synced to disk so that the
  log tells what the computer was doing even after a hard reset.
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

[Unreleased]: https://github.com/mliistro-beep/anima-lora-studio/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/mliistro-beep/anima-lora-studio/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/mliistro-beep/anima-lora-studio/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mliistro-beep/anima-lora-studio/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mliistro-beep/anima-lora-studio/releases/tag/v0.1.0
