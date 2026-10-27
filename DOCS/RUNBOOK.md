# Runbook

Operating the Anima LoRA Trainer desktop app on a workstation. It is a local desktop
app, not a service: there are no servers, health endpoints or alerting.

Current state: M0-M4c are done. M0-M4b were each verified with a real training run: the
training UI with Citron parity, the app's own training runtime installed from the Settings
tab, and model files already on disk used after a SHA-256 check. M4c added the License and
Credits buttons. Previews and the image generator (M5) are not built yet. See [milestones.md](milestones.md).

## Install and launch

1. Install the app:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e .
   ```

2. Start it with `.venv/bin/anima-trainer` (from a checkout, for development:
   `scripts/dev-run.sh`, or `scripts/dev-run.sh --scratch` to keep your own settings untouched).
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
| Checkpoints | the output directory set in the settings |

Relative image and output folders are resolved against the data folder, because
sd-scripts runs with that as its working directory.

## Common issues

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

**Training fails with an out-of-memory hint.**
- "GPU out of memory": lower `network_dim` (for example 8) and/or `resolution` (512),
  keep batch size 1.
- "killed, probably by the system OOM killer": system RAM ran out; use the same
  settings and close other programs.

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

- To go back to an earlier version of the app: `git checkout <commit>` and reinstall with
  `.venv/bin/pip install -e .`.
- Citron's own web UI is independent of this app and keeps working. To reuse settings
  there, copy `<data>/config.json` into the Citron folder: the format is the same.
