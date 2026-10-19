"""GPU dropdown labels ("0: NVIDIA ...") to numeric indices, as in Citron."""

DEFAULT_GPU_INDEX = "0"


def gpu_index_from_choice(choice: str | None) -> str:
    prefix = str(choice).split(":")[0].strip() if choice else ""
    return prefix if prefix.isdigit() else DEFAULT_GPU_INDEX
