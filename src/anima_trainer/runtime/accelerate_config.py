"""accelerate's single-GPU config, written by the app instead of copied from Citron.

The text is Citron's app_configs/accelerate_gpu.yaml byte for byte, so training runs with
the same accelerate settings (bf16, one process, no distribution).
"""

from pathlib import Path

from anima_trainer.runtime.paths import RuntimePaths

ACCELERATE_GPU_YAML = """\
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16
use_cpu: false
num_processes: 1
num_machines: 1
main_training_function: main
"""


def write_accelerate_config(paths: RuntimePaths) -> Path:
    """Write (or rewrite) the config in the data folder; it is small and always regenerable."""
    target = paths.accelerate_config_file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ACCELERATE_GPU_YAML, encoding="utf-8")
    return target
