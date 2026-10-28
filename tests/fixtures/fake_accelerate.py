"""Stand-in for `accelerate launch` used by the runner tests.

Modes:
  replay <file> <exit_code>   write the file to stdout unchanged, then exit
  text <text> <exit_code>     write text (with \\r and \\n escapes expanded), then exit
  env <name>                  print the value of an environment variable
  sleep <pid_file>            start a sleeping child (like accelerate does), write its pid, wait
  ignore-term                 ignore SIGTERM, print "started", sleep (forces a SIGKILL)
  crash                       kill itself with SIGKILL
  train <exit_code> <args...> act like a short sd-scripts run on the app's real arguments: read
                              the training TOML (the last --config_file), print sd-scripts'
                              epoch, step and checkpoint lines, write the checkpoints and, when
                              the TOML has sample_prompts, one PNG per prompt per sampled epoch
"""

import json
import os
import signal
import struct
import subprocess
import sys
import time
import tomllib
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

STEPS_PER_EPOCH = 2
SAMPLE_TIME_FORMAT = "%Y%m%d%H%M%S"  # sd-scripts' sample file names


def _png(path: Path) -> None:
    """A valid 1x1 PNG, so the gallery can show it."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\x00\xc0\x40\x40")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
    )


def _say(line: str) -> None:
    print(line, flush=True)


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"lora")
    _say(f"saving checkpoint: {path}")


def _steps_line(done: int, total: int) -> str:
    """A tqdm line like sd-scripts prints (the parser reads steps, times and avr_loss)."""
    percent = 100 * done // total
    times = f"00:{done:02d}<00:{total - done:02d}"
    return f"steps: {percent}%| | {done}/{total} [{times}, 1.00s/it, avr_loss=0.12]"


def _training_toml(app_args: list[str]) -> Path:
    """The first --config_file is accelerate's own YAML, the last one the training TOML."""
    flags = [index for index, arg in enumerate(app_args) if arg == "--config_file"]
    if not flags or flags[-1] + 1 >= len(app_args):
        sys.exit("fake_accelerate train: no --config_file <training.toml> in the arguments")
    return Path(app_args[flags[-1] + 1])


def _samples(config: dict[str, Any], out: Path, name: str, epoch: int, step: int) -> None:
    prompts_file = config.get("sample_prompts")
    if not prompts_file or epoch % int(config.get("sample_every_n_epochs", 1)):
        return
    _say(f"Generating sample images at step {step}")
    stamp = datetime.now().strftime(SAMPLE_TIME_FORMAT)
    prompts = json.loads(Path(prompts_file).read_text(encoding="utf-8"))
    for index, prompt in enumerate(prompts):
        _png(out / "sample" / f"{name}_e{epoch:06d}_{index:02d}_{stamp}_{prompt['seed']}.png")


def _train(exit_code: int, app_args: list[str]) -> int:
    config = tomllib.loads(_training_toml(app_args).read_text(encoding="utf-8"))
    out, name = Path(config["output_dir"]), config["output_name"]
    epochs, every = int(config["max_train_epochs"]), int(config.get("save_every_n_epochs", 0))
    total = epochs * STEPS_PER_EPOCH
    for epoch in range(1, epochs + 1):
        _say(f"epoch {epoch}/{epochs}")
        for step_in_epoch in range(1, STEPS_PER_EPOCH + 1):
            _say(_steps_line((epoch - 1) * STEPS_PER_EPOCH + step_in_epoch, total))
        if epoch < epochs and every and epoch % every == 0:
            _save(out / f"{name}-{epoch:06d}.safetensors")
        _samples(config, out, name, epoch, epoch * STEPS_PER_EPOCH)
    _save(out / f"{name}.safetensors")
    _say("2026-10-20 09:00:00 INFO     model saved.")
    return exit_code


def main(argv: list[str]) -> int:
    mode = argv[1]
    if mode == "train":
        return _train(int(argv[2]), argv[3:])
    if mode == "replay":
        with open(argv[2], "rb") as source:
            sys.stdout.buffer.write(source.read())
        return int(argv[3])
    if mode == "text":
        sys.stdout.write(argv[2].replace("\\r", "\r").replace("\\n", "\n"))
        return int(argv[3])
    if mode == "env":
        print(os.environ.get(argv[2], "<unset>"))
        return 0
    if mode == "sleep":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with open(argv[2], "w") as handle:
            handle.write(str(child.pid))
        print("started", flush=True)
        child.wait()
        return child.returncode
    if mode == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("started", flush=True)
        time.sleep(60)
        return 0
    if mode == "crash":
        os.kill(os.getpid(), signal.SIGKILL)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
