"""The preview fields of the TUI's training form, over the shared preview.json settings.

The form shows one prompt: the first one. The window can keep more, one per line; they stay as
they are, and so do the image settings (sampler, steps, size, CFG...), which only the window shows.
"""

from dataclasses import dataclass

from anima_trainer.preview.settings import PreviewSettings

PROMPT_REQUIRED = "ERROR: Previews are on: write a preview prompt."


@dataclass(frozen=True)
class PreviewFields:
    enabled: bool
    pre_prompt: str
    trigger: str
    prompt: str
    negative_prompt: str


def fields_of(settings: PreviewSettings) -> PreviewFields:
    first = settings.prompts[0] if settings.prompts else ""
    return PreviewFields(
        settings.enabled, settings.pre_prompt, settings.trigger, first, settings.negative_prompt
    )


def kept_prompts_note(settings: PreviewSettings) -> str:
    """The prompts after the first one, which the form does not show but keeps."""
    others = len(settings.prompts) - 1
    if others < 1:
        return ""
    noun = "prompt" if others == 1 else "prompts"
    return f"+{others} more {noun} from the window, kept."


def previews_from_fields(
    saved: PreviewSettings, fields: PreviewFields
) -> PreviewSettings | tuple[str, ...]:
    """New settings with the form's values. The prompt field replaces the first saved prompt;
    cleared, it removes it (an error while previews are on)."""
    prompt = fields.prompt.strip()
    if fields.enabled and not prompt:
        return (PROMPT_REQUIRED,)
    prompts = (*((prompt,) if prompt else ()), *saved.prompts[1:])
    # Text fields only, over valid saved settings: nothing here can fail validation.
    return saved.with_changes(
        enabled=fields.enabled,
        pre_prompt=fields.pre_prompt,
        trigger=fields.trigger,
        prompts=prompts,
        negative_prompt=fields.negative_prompt.strip(),
    )
