"""GPU dropdown labels ("0: NVIDIA ...") to numeric indices, as in Citron."""

DEFAULT_GPU_INDEX = "0"


def gpu_index_from_choice(choice: str | None) -> str:
    if not choice:
        return DEFAULT_GPU_INDEX
    return str(choice).split(":")[0].strip()
