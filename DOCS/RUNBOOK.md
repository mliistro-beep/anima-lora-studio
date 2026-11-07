# Runbook

Operating the Anima LoRA Studio desktop app on a workstation. It is a local desktop
app, not a service: there are no servers, health endpoints or alerting.

Current state: M0-M5 are done. M0-M4b were each verified with a real training run: the
training UI with Citron parity, the app's own training runtime installed from the Settings
tab, and model files already on disk used after a SHA-256 check. M4c added the License and
Credits buttons. M5 added previews during training (Previews tab) and the image generator
(Generator tab), verified from the real UI with a training running. M6 (release 0.2.0) adds
the `run_linux.sh` launcher with the applications menu entry, the swap / out-of-memory daemon
warning and the new display name. Later versions added training from the terminal
(`anima-trainer tui` and `train`) and, in M9, a generator independent from the training, the
same in the window and in the terminal (next sections). See [CHANGELOG.md](../CHANGELOG.md).

## Install and launch

0. Check swap and the out-of-memory daemon once per machine
   ([SYSTEM_SETUP.md](SYSTEM_SETUP.md)): 8-16 GB of swap and earlyoom (or systemd-oomd).
   Without them, running out of RAM freezes the whole computer instead of stopping the
   training.
1. Get the app and start it (Python 3.12 or newer with `venv`; on Debian/Ubuntu
   `sudo apt install python3 python3-venv git`):

   ```bash
   git clone https://github.com/mliistro-beep/anima-lora-studio.git
   cd anima-lora-studio
   ./run_linux.sh                    # first start: creates .venv and installs the app (network)
   ./run_linux.sh --install-desktop  # optional: menu entries and the anima-trainer command
   ```

   Later starts reinstall only when `pyproject.toml` changed (for example after `git pull`)
   or the checkout was moved; otherwise they start the app right away. Run the first start
   from a terminal, or use `--install-desktop` first: started from the menu, the install
   has no window, only a notification.

   `--install-desktop` adds two menu entries: Anima LoRA Studio (the window) and Anima LoRA
   Studio (terminal), which opens `anima-trainer tui` in your terminal emulator (closing that
   window stops a training, as quitting the interface does). It also links
   `~/.local/bin/anima-trainer` to the app in `.venv`, so `anima-trainer tui` and
   `anima-trainer train` work from any folder; a link from another checkout is replaced, any
   other file there is kept with a `WARN:`. If `~/.local/bin` was not in `PATH` yet, an
   `INFO:` line says so: most distributions add it at the next login once the folder exists.
   The link starts the app directly, without the launcher: after a `git pull`, start
   `./run_linux.sh` or one of the menu entries once, so new dependencies are installed.
   `./run_linux.sh --uninstall-desktop` removes the menu entries, the icon and the link (only
   a link to this checkout).

   For development use `scripts/dev-run.sh` (dev tools, `--scratch`). Trying a development
   build with `./run_linux.sh` uses your real data folder and settings: point
   `XDG_DATA_HOME` and `XDG_CONFIG_HOME` at scratch folders first (or use
   `scripts/dev-run.sh --scratch`), so a stable install such as v0.1.0 is not touched.
2. On the first start the Training tab may show a `WARN:` about the swap or the
   out-of-memory daemon (the Settings tab shows both checks): see step 0.
3. In the Settings tab, choose the data folder (or keep the default) and press Install.
   It needs Python 3.10+ with `venv` (on Debian/Ubuntu `python3-venv`), git and an
   internet connection; torch and the sd-scripts requirements are several GB. The steps
   and their output appear in the install log, also saved as
   `<data>/logs/install_<timestamp>.log`. Then download Qwen3, the VAE and the DiT with
   the Download buttons (the DiT is also downloaded on the first Start), or use files you
   already have (next section). Citron is not needed and the app never writes into a
   Citron folder.

## Use model files you already have

In the Settings tab, press **Use file...** on a model row and choose the file (for example
the copy of a Citron or ComfyUI install). The app checks its size and SHA-256 against the
official file on Hugging Face (a DiT takes a few seconds) and, if they match, uses the file
where it is, without copying it: the row shows "OK (your file)". A different file is
rejected with the reason. **Unlink** goes back to the copy in the data folder.

The choice is saved in `~/.config/anima-trainer/anima-trainer.conf` (group
`linked_models`) and is kept when the data folder changes. "Use file..." is disabled for a
model already present in the data folder.

## Train from the terminal

`anima-trainer tui` and `anima-trainer train` need the runtime and the models set up in the graphical
app (they never download); after that they use only QtCore, so no display is needed. Both train
on `anima-base-v1.0` only for now. They do not write `config.json` or `preview.json`: the window
finds its settings as you left them. Previews follow what is saved in `preview.json`, and the
summary says whether they are on. GPU, CPU threads and cache settings come from `config.json`.

### Remote use: tmux

Quitting the interface stops the training, and so does losing the connection, unless it runs
inside tmux (`nohup` also works for `train --yes`):

```bash
tmux new -s lora                  # after connecting with SSH
.venv/bin/anima-trainer tui       # inside tmux
# Ctrl+B, then D: detach; the training goes on. Connection lost: the same.
tmux attach -t lora               # back to it later
```

### Presets

Built in: Style, Character, Concept, Person. Your own are TOML files in `<data folder>/presets/`,
and only the values that differ from the defaults are needed:

```toml
name = "My preset"
description = "One line, shown when choosing"

[settings]
network_dim = 24
network_alpha = 24
learning_rate = 0.0001
max_train_epochs = 12   # only a suggestion
```

A preset always starts from the defaults, not from the last training. It cannot hold what is chosen
for each training (project, folders, repeats, resolution) or what belongs to one machine (GPU, CPU
threads, cache settings). A file with such a value, an unknown name, a multi-line description or
the name of another preset is ignored with a `WARN:` line, and the others still load.

If the VRAM is not enough, use a preset with `network_dim = 8` or lower the resolution.

### Exit codes of `train` and `tui`

| Code | Meaning |
|------|---------|
| 0 | Training completed (or nothing to do) |
| 1 | Training failed, or the interface crashed |
| 2 | Not started: wrong options, a problem in the data folder, declined, or the GPU is in use |
| 130 | Stopped: Stop, Ctrl+C, SIGTERM, or a lost terminal |

### One process at a time

`<data folder>/gpu.lock` lets one process use the GPU of a data folder: a training from the
terminal refuses Start, Generate and Install in the window, and the other way round, naming the
process that holds it; `train` checks it before asking for confirmation and `tui` before the
summary. A lock left by a
process that died does not block. If that process was
killed with SIGKILL its training may still be running: check `nvidia-smi` before starting another.

## Generate images

The generator works without a project or a training: the Generator tab in the window, or
Generate images in the start menu of `anima-trainer tui` (the training form has a Menu button
to go back there). Both show the same settings, saved in `<data>/generator.json`: what you set
in one is there in the other at the next start. It needs the runtime and the models installed,
as the training does.

- **Model**: a base model of the catalog, or **Model file** with any Anima DiT of yours
  (`.safetensors`); its header is checked first, so a LoRA, a text encoder, a VAE or a model of
  another family is refused with an `ERROR:` line.
- **LoRAs**: up to 8, each with on/off, a trigger and a strength from -2 to 2 (Add... in the
  window, Add LoRA in the terminal). kohya and PEFT files are accepted (a PEFT file is renamed
  while loading, with an `INFO:` line); LoHa, LoKr, DoRA, LoCon, OFT and LoRAs of other models
  are refused. The trigger is filled from the file (`modelspec.trigger_phrase`) when it has one.
  A file already in the list is skipped with an `INFO:` line.
- **Prompt**: the final prompt is the pre-prompt (for example `score_9, masterpiece`), then the
  triggers of the LoRAs that are on, in the order of the list, then the prompt. The log shows it
  (`INFO: Prompt: ...`).
- **Image settings**: size, sampler, steps, CFG (it follows the sampler: 6.5 with ER-SDE, 7 with
  Euler, unless you changed it), flow shift, seed. **Random seed** picks a new seed for every
  image, and **Images** makes 1 to 10 of them in a row with the same prompt. **Reset to
  defaults** puts these settings and the negative prompt back (random seed off, one image);
  pre-prompt, prompt, LoRAs, model and folder stay.
- **Queue**: **Add current** stores a copy of the whole form (prompts, LoRAs with trigger and
  strength, image settings, random seed and image count), up to 10 entries; **Remove**,
  **Clear**, then **Generate queue** runs them one after the other. Each entry makes its own
  number of images: 4 entries with 5 images each are 20 images.
- **Output**: the images go to `<data>/generated` unless you choose another folder (in the
  window also **Use project folder**, `<output>/generated`, and **Open folder**).

While it runs, a line shows where it is, `img 03/10 | batch 2/4 | step 12/25` (batch = the
queue entry; Generate alone is batch 1/1), with the log and the path of each image. Stop (or
`s` in the terminal) ends it; quitting asks first and leaves no process behind. A generation
holds the GPU lock like a training (see [One process at a time](#one-process-at-a-time)); in the
window it may join a training already running, with a lighter memory profile.

Each image is loaded and generated by its own process, so it takes about 1-1.5 minutes
(1024x1024, 25-30 steps), most of it loading the model. In a real test 6 LoRAs together needed
only about 300 MiB of VRAM more than none.

File names: `<prefix>_<label>_<time>_<seed>.png`, and with more than one image
`<prefix>_<label>_<time>_<NN>_<seed>.png`, numbered across the whole queue as in the progress
line; files are never overwritten. In the window **Compare Epochs** makes one image per
checkpoint of the open project, with the same seed, shown as thumbnails.

Every PNG holds two text chunks: `parameters`, in the A1111 / Forge layout that Civitai and
image viewers read (final prompt, negative, steps, sampler, CFG, seed, size, `<lora:name:strength>`),
and `anima_lora_studio`, a JSON copy of the entry that made that image (seed fixed, one image,
only the LoRAs used) with the app's name and version. Only file names are written, never
folders.

## Where things are

All paths are inside the app's data folder (`<data>`, default `~/.local/share/anima-trainer`,
or `$XDG_DATA_HOME/anima-trainer`; it can be changed in the Settings tab). Its layout is the
same as a Citron installation, but it belongs to the app only. The chosen folder is saved
in `~/.config/anima-trainer/anima-trainer.conf`.

| What | Path |
|------|------|
| Saved settings | `<data>/config.json` (same format as Citron) |
| Generated TOML configs | `<data>/configs/<project>_{training,dataset}_<timestamp>.toml` |
| Training logs | `<data>/logs/<project>_<timestamp>.log` |
| DiT base model | `<data>/models/anima/dit/<base-model>.safetensors`, or your linked file |
| Qwen3, VAE | `<data>/models/anima/text_encoder/`, `<data>/models/anima/vae/`, or your linked files |
| Partial downloads | same path with a `.part` suffix |
| Install state | `<data>/.install/*.commit` (one marker per install step) |
| Install logs | `<data>/logs/install_<timestamp>.log` |
| Launcher log (started without a terminal) | `~/.local/state/anima-trainer/launcher.log` |
| App venv (`run_linux.sh`) | `<checkout>/.venv`, install stamp `.venv/.anima-pyproject.sha256` |
| Menu entry and icon | `~/.local/share/applications/anima-lora-studio.desktop`, `~/.local/share/icons/hicolor/scalable/apps/anima-lora-studio.svg` |
| Generator settings and queue | `<data>/generator.json` (window and terminal) |
| Generated images | `<data>/generated/`, or the folder chosen in the generator |
| Checkpoints | the output directory set in the settings |

Relative image and output folders are resolved against the data folder, because
sd-scripts runs with that as its working directory.

## Common issues

**Nothing happens when Anima LoRA Studio is started from the menu.** The first start (or
the first after a `git pull` that changed `pyproject.toml`) installs the app before the
window opens; a notification says so. Errors also arrive as a notification when
`notify-send` is installed. Either way the launcher's output is in
`~/.local/state/anima-trainer/launcher.log`; or run `./run_linux.sh` from a terminal.

**"Python 3.12 or newer is needed".** `run_linux.sh` looks for `python3`, `python3.14`,
`python3.13` and `python3.12`. Install a newer Python with the package manager.

**"Could not create .venv with pip" or ".venv has no working Python or pip".** The first
needs `python3-venv` (Debian/Ubuntu); the second usually means the system Python was
upgraded under the venv. Remove `<checkout>/.venv` and start `./run_linux.sh` again.

**"WARN: Updating the app failed ... starting the installed version".** The reinstall after
a `pyproject.toml` change failed (often no network); the previous install is started and
the update is retried at the next start.

**The computer froze or became unreachable during a training.** Open the training log
(`<data folder>/logs/<project>_<date>.log`) and search for `RESOURCES`: one such line is
written at the start and every minute, and synced to disk, with the available RAM, the used
swap and each GPU's VRAM, temperature, power and load. The last lines show whether the RAM or
the swap ran out, or whether the GPU was at its power or temperature limit, before the end.
Epoch checkpoints already saved stay in the output folder.

**"WARN: Swap is ..." or "WARN: No out-of-memory daemon ...".** Shown at start and after
Apply in the data folder; the app still works, but running out of RAM can freeze the
computer. Set up swap and earlyoom as in [SYSTEM_SETUP.md](SYSTEM_SETUP.md).

**"Missing models: Qwen3, VAE" when configuring.** The text encoder or VAE is not in
`<data>/models/anima/`. Download it with its button in the Settings tab (only the DiT is
downloaded automatically on Start).

**"Training runtime incomplete or out of date: press Repair in Settings".** An install
step is missing or was done for an older version (for example after updating the app, or
an interrupted install). Press Repair: only the missing steps run. Installs made before the
M4 review fixes ask for it once (venv, pip, requirements, GPU check; the packages are
already there, so it is quick).

**"... is not a compatible ... file" after Use file....** The file is not the official
one: "wrong size" means a different model (checked before reading the file), "SHA-256
does not match" a different version or a damaged file. Pick the right file or download it.

**"Missing (your file was moved or changed)".** A linked file was moved, replaced or
edited, so it is no longer trusted. Press Use file... again on its new location, or
Download (the download goes to the data folder, never into your folder).

**Install failed or was cancelled.** Read the install log (Settings tab, or
`<data>/logs/install_<timestamp>.log`) and press Install again: steps already done are
skipped. "CUDA is not available" means torch cannot see the GPU: check the NVIDIA driver
(`nvidia-smi`). "Python ... not found" or "git not found" appear before anything is written.

**DiT download interrupted or cancelled.** The data stays in `<file>.part` and the next
download resumes from it. If the remote file changed and the `.part` is larger, it is
discarded and downloaded again. To force a clean download, delete the `.part`.

**"Downloaded file SHA-256 does not match" or "wrong size".** Every download is checked
against the catalog before it is kept; a damaged or different file is deleted, so press
Download again. If it keeps failing, Hugging Face may have published a new version of the
file: the catalog (`models/catalog.py`, `EXPECTED_FILES`) must be updated.

**Training fails with an out-of-memory hint.**
- "GPU out of memory": lower `network_dim` (for example 8) and/or `resolution` (512),
  keep batch size 1.
- "killed, probably by the system OOM killer": system RAM ran out; use the same
  settings and close other programs. If earlyoom stopped it, `journalctl -u earlyoom`
  shows it.

**Generator: "Not enough free system RAM: about N MiB needed".** The GPU memory left only
the profiles that keep the DiT in system RAM (about 6-7 GB, more margin while a training
runs), and that much RAM is not available. The generator refuses instead of pushing the
system into swap. Close other programs, or generate after the training ends, when the
`standard` profile fits the GPU and needs about 4 GB of RAM.

**The whole computer freezes (no mouse, no SSH) during a training or a generation.**
System RAM ran out with little swap and no out-of-memory daemon: the kernel keeps
swapping instead of stopping a program. After the reset, confirm it with
`journalctl -b -1 | grep -i "memory pressure"` and with `sar -r` if sysstat is installed
(memory available near 0, commit above 100%), then set up swap and earlyoom as in
[SYSTEM_SETUP.md](SYSTEM_SETUP.md). Seen on 2026-10-27 on a 16 GB machine with a 512 MB
swap file, during a 1024x1024 test run with previews.

The hint uses the training log and, when allowed, `dmesg`. On many desktops `dmesg`
needs root, so the hint can be missing even after a real kernel OOM; check
`journalctl -k` by hand.

**A run does not stop right away.** Stop sends SIGTERM to the whole process group
(`accelerate` and the training child), then SIGKILL after 10 seconds. The runner's
`shutdown()` does the same and waits; it runs on window close, on Ctrl+C and on SIGTERM.

**GPU still busy after a crash of the app itself.** If the app was killed hard (for
example `kill -9`), the training child keeps running in its own session. Find and stop
it:

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv
```

then `kill <pid>`.

**"Log file ... could not be written".** Usually a full disk. Training keeps running
and the output stays visible in the app, but the log on disk stops at the error.

**GPU list shows only "0" and "1".** Neither the training venv's torch nor `nvidia-smi`
answered (runtime not installed, driver problem). Training can still be started on GPU 0.

## Rollback

- To go back to an earlier version of the app: `git checkout <tag or commit>` and start
  `./run_linux.sh` (it reinstalls when `pyproject.toml` differs). Versions before 0.2.0 have
  no `run_linux.sh`: there, run `.venv/bin/pip install -e .` and `.venv/bin/anima-trainer`.
  The data folder and the settings are shared, and 0.1.0 reads them as they are.
- To remove the app: `./run_linux.sh --uninstall-desktop` (menu entries, icon and the
  `anima-trainer` link), then delete the checkout. The
  data folder (`~/.local/share/anima-trainer`, several GB) and
  `~/.config/anima-trainer/` are kept until you delete them.
- Citron's own web UI is independent of this app and keeps working. To reuse settings
  there, copy `<data>/config.json` into the Citron folder: the format is the same.
