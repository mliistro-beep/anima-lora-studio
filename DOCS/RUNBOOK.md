# Runbook

Operating the Anima LoRA Studio desktop app on a workstation. It is a local desktop
app, not a service: there are no servers, health endpoints or alerting.

Current state: M0-M5 are done. M0-M4b were each verified with a real training run: the
training UI with Citron parity, the app's own training runtime installed from the Settings
tab, and model files already on disk used after a SHA-256 check. M4c added the License and
Credits buttons. M5 added previews during training (Previews tab) and the image generator
(Generator tab), verified from the real UI with a training running. M6 (release 0.2.0) adds
the `run_linux.sh` launcher with the applications menu entry, the swap / out-of-memory daemon
warning and the new display name. See [milestones.md](milestones.md) and
[CHANGELOG.md](../CHANGELOG.md).

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
   ./run_linux.sh --install-desktop  # optional: add Anima LoRA Studio to the applications menu
   ```

   Later starts reinstall only when `pyproject.toml` changed (for example after `git pull`)
   or the checkout was moved; otherwise they start the app right away. Run the first start
   from a terminal, or use `--install-desktop` first: started from the menu, the install
   has no window, only a notification. `./run_linux.sh --uninstall-desktop` removes the menu
   entry and its icon.

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
- To remove the app: `./run_linux.sh --uninstall-desktop`, then delete the checkout. The
  data folder (`~/.local/share/anima-trainer`, several GB) and
  `~/.config/anima-trainer/` are kept until you delete them.
- Citron's own web UI is independent of this app and keeps working. To reuse settings
  there, copy `<data>/config.json` into the Citron folder: the format is the same.
