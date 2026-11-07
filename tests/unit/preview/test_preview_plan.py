from anima_trainer.preview.preview_plan import plan_previews
from anima_trainer.preview.sample_prompts import SampleRequest, SampleSize
from anima_trainer.preview.settings import PreviewSettings

ON = PreviewSettings(enabled=True, prompts=("a girl",))


def test_previews_off_add_nothing():
    plan = plan_previews(PreviewSettings(prompts=("a",)), 12288)

    assert (plan.request, plan.messages, plan.error) == (None, (), None)


def test_automatic_size_follows_the_gpu_memory():
    plan = plan_previews(ON, 8192)

    assert plan.request == SampleRequest(ON, SampleSize(768, 768))
    assert plan.error is None
    assert plan.messages == (
        "",
        "Previews:",
        "  INFO: Preview size 768x768 chosen for 8192 MiB of GPU memory.",
        "  INFO: 1 prompt, every 1 epoch, ER-SDE, 25 steps, CFG 6.5.",
    )


def test_previews_without_prompts_are_an_error():
    plan = plan_previews(PreviewSettings(enabled=True), 12288)

    assert plan.request is None
    assert plan.error == "Previews are on but there is no preview prompt."


def test_too_little_gpu_memory_turns_previews_off_with_a_warning():
    plan = plan_previews(ON, 4096)

    assert plan.request is None and plan.error is None
    assert any(line.startswith("  WARN: Previews turned off") for line in plan.messages)


def test_fixed_size_is_used_as_chosen():
    settings = ON.with_changes(auto_size=False, width=832, height=1216, sampler="euler")

    plan = plan_previews(settings, 12288)

    assert plan.request == SampleRequest(settings, SampleSize(832, 1216))
    assert plan.messages[-1] == "  INFO: 1 prompt, every 1 epoch, Euler, 25 steps, CFG 6.5."
