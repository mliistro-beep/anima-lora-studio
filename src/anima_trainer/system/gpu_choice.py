"""GPU dropdown labels ("0: NVIDIA ...") to numeric indices, as in Citron."""

DEFAULT_GPU_INDEX = "0"


def gpu_index_from_choice(choice: str | None) -> str:
    prefix = str(choice).split(":")[0].strip() if choice else ""
    return prefix if prefix.isdigit() else DEFAULT_GPU_INDEX


def choice_for_index(choices: tuple[str, ...], saved_index: str) -> str:
    """The "<index>: name" (or bare "<index>") entry, else the first choice, as in Citron."""
    match = next((c for c in choices if c == saved_index or c.startswith(f"{saved_index}:")), None)
    if match is not None:
        return match
    return choices[0] if choices else saved_index
