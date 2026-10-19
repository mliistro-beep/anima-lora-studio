# Runbook

Operating the Anima LoRA Trainer desktop app on a workstation. It is a local desktop
app, not a service: there are no servers, health endpoints or alerting.

Current state: the core and services (M0-M2) are done; the training UI (M3) is not
built yet, so `anima-trainer` opens an empty window. See [milestones.md](milestones.md).

## Install and launch

1. Make sure Citron is installed and working (its own `setup_for_linux.sh`), with
   `.venv`, `sd-scripts` and `models/anima/text_encoder` and `models/anima/vae`.
2. Install the app:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e .
   ```

3. Start it with `.venv/bin/anima-trainer`.

## Where things are

All paths are inside the Citron root, so the app and Citron's web UI share them:

| What | Path |
|------|------|
| Saved settings | `<citron>/config.json` (same format as Citron) |
| Generated TOML configs | `<citron>/configs/<project>_{training,dataset}_<timestamp>.toml` |
| Training logs | `<citron>/logs/<project>_<timestamp>.log` |
| DiT base model | `<citron>/models/anima/dit/<base-model>.safetensors` |
| Partial downloads | same path with a `.part` suffix |
| Checkpoints | the output directory set in the settings |

Relative image and output folders are resolved against the Citron root, because
sd-scripts runs with that as its working directory.

## Common issues

**"Missing models: Qwen3, VAE" when configuring.** The text encoder or VAE is not in
`<citron>/models/anima/`. Run Citron's setup script; only the DiT is downloaded
automatically.

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
`shutdown()` does the same and waits; it will be called on window close once the UI
(M3) exists.

**GPU still busy after a crash of the app itself.** If the app was killed hard (for
example `kill -9`), the training child keeps running in its own session. Find and stop
it:

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv
```

then `kill <pid>`.

**"Log file ... could not be written".** Usually a full disk. Training keeps running
and the output stays visible in the app, but the log on disk stops at the error.

**GPU list shows only "0" and "1".** Neither the Citron venv's torch nor `nvidia-smi`
answered (wrong Citron root, driver problem). Training can still be started on GPU 0.

## Rollback

- To go back to an earlier version of the app: `git checkout <commit>` and reinstall with
  `.venv/bin/pip install -e .`.
- To go back to Citron's own web UI: it reads the same `config.json` and `configs/`, so
  it can be used at any time without converting anything.
