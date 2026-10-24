# Runbook

Operating the Anima LoRA Trainer desktop app on a workstation. It is a local desktop
app, not a service: there are no servers, health endpoints or alerting.

Current state: the training UI (M3) is done and was verified with a real run. M4 (the
app's own training runtime, independent of Citron) is in progress: the app already uses
its own data folder and the installer that runs the install steps exists, but the Settings
tab does not start it yet (Install / Repair button), so until then the app reports
"Training runtime not installed" and cannot train. See [milestones.md](milestones.md).

## Install and launch

1. Install the app:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e .
   ```

2. Start it with `.venv/bin/anima-trainer`.
3. The training runtime (venv, sd-scripts, models) goes into the data folder shown in the
   Settings tab. Installing it from the app (Settings, Install) is part of M4 and not
   built yet. Citron is not needed and the app never writes into a Citron folder.

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
| DiT base model | `<data>/models/anima/dit/<base-model>.safetensors` |
| Partial downloads | same path with a `.part` suffix |
| Checkpoints | the output directory set in the settings |

Relative image and output folders are resolved against the data folder, because
sd-scripts runs with that as its working directory.

## Common issues

**"Missing models: Qwen3, VAE" when configuring.** The text encoder or VAE is not in
`<data>/models/anima/`. Only the DiT is downloaded automatically for now; downloading
Qwen3 and the VAE from the app is part of M4.

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
