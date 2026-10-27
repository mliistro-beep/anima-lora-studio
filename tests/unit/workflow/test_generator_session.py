from datetime import datetime
from pathlib import Path

import pytest

from anima_trainer.preview.inference_command import GenerationRequest
from anima_trainer.preview.sample_prompts import SampleSize
from anima_trainer.runtime.paths import RuntimePaths
from anima_trainer.training.runner import RunOutcome
from anima_trainer.workflow.generator_session import GeneratorSession
from tests.fakes import FakeRunner

NOW = datetime(2026, 10, 27, 12, 0, 0)


@pytest.fixture
def paths(tmp_path):
    runtime = RuntimePaths(tmp_path / "data")
    for file in (runtime.qwen3_model, runtime.vae_model, runtime.dit_model("anima-base-v1.0")):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"")
    return runtime


@pytest.fixture
def parts(qtbot):
    runner = FakeRunner()
    session = GeneratorSession(runner, clock=lambda: NOW)
    events = {"log": [], "steps": [], "images": [], "finished": [], "running": []}
    session.log_line.connect(events["log"].append)
    session.step_progress.connect(lambda done, total: events["steps"].append((done, total)))
    session.image_ready.connect(events["images"].append)
    session.finished.connect(lambda ok, message: events["finished"].append((ok, message)))
    session.running_changed.connect(events["running"].append)
    return session, runner, events


def _request(tmp_path, name="a.png", base_model="anima-base-v1.0"):
    return GenerationRequest(
        base_model=base_model, lora=None, lora_multiplier=1.0, prompt="a girl", negative_prompt="",
        size=SampleSize(1024, 1024), steps=25, cfg_scale=4.0, flow_shift=3.0, seed=1,
        sampler="er_sde", output=tmp_path / "out" / name,
    )  # fmt: skip


def _ok(runner, image):
    runner.line_received.emit(f"IMAGE: {image}")
    runner.finished.emit(RunOutcome(0, crashed=False, stopped=False, log_path=None))


def test_generates_the_queue_one_image_at_a_time(parts, paths, tmp_path):
    session, runner, events = parts
    first, second = _request(tmp_path, "1.png"), _request(tmp_path, "2.png")

    assert session.generate(paths, (first, second), "0", training=False) is None
    assert session.is_running() and len(runner.started) == 1
    runner.line_received.emit("PROFILE: standard")
    runner.line_received.emit("STEP: 3/25")
    _ok(runner, first.output)
    assert len(runner.started) == 2
    _ok(runner, second.output)

    assert events["steps"] == [(3, 25)]
    assert events["images"] == [first.output, second.output]
    assert events["finished"] == [(True, "OK: 2 images generated.")]
    assert events["running"] == [True, False]
    assert "INFO: Memory profile: standard" in events["log"]
    assert runner.started[0][1] == paths.logs_dir / "generate_2026-10-27_120000.log"
    assert not session.is_running()


def test_every_profile_is_passed_with_the_training_margin(parts, paths, tmp_path):
    from anima_trainer.preview.vram_profile import PROFILES, TRAINING_MARGIN_MIB

    session, runner, _ = parts

    session.generate(paths, (_request(tmp_path),), "0", training=True)

    arguments = runner.started[0][0].arguments
    assert arguments[arguments.index("--margin_mib") + 1] == str(TRAINING_MARGIN_MIB)
    profiles = arguments[arguments.index("--profiles") + 1]
    assert all(f'"name": "{profile.name}"' in profiles for profile in PROFILES)


def test_a_checkpoint_deleted_after_the_refresh_is_refused(parts, paths, tmp_path):
    session, runner, _ = parts
    request = _request(tmp_path)
    gone = GenerationRequest(**{**request.__dict__, "lora": tmp_path / "gone.safetensors"})

    error = session.generate(paths, (gone,), "0", training=False)

    assert error == f"Checkpoint not found: {tmp_path / 'gone.safetensors'} (press Refresh)"
    assert runner.started == []


def test_missing_models_are_refused_before_starting(parts, paths, tmp_path):
    session, runner, _ = parts
    paths.vae_model.unlink()

    error = session.generate(paths, (_request(tmp_path),), "0", training=False)

    assert error == "Missing models: VAE"
    assert runner.started == [] and not session.is_running()


def test_missing_dit_of_the_chosen_base_model_is_refused(parts, paths, tmp_path):
    session, runner, _ = parts

    error = session.generate(paths, (_request(tmp_path, base_model="anima-preview"),), "0", False)

    assert error == f"Missing model: DiT anima-preview ({paths.dit_model('anima-preview')})"


def test_second_request_and_empty_queue_are_refused(parts, paths, tmp_path):
    session, _, _ = parts

    assert session.generate(paths, (), "0", training=False) == "Nothing to generate."
    session.generate(paths, (_request(tmp_path),), "0", training=False)
    assert session.generate(paths, (_request(tmp_path),), "0", False) == (
        "A generation is already running."
    )


@pytest.mark.parametrize(
    "outcome, image, expected",
    [(RunOutcome(3, False, False, None), False,
      "ERROR: Not enough GPU memory or system RAM for any profile, see the log above."),
     (RunOutcome(1, False, False, Path("/l.log")), False,
      "ERROR: Generation failed (exit code 1), see /l.log"),
     (RunOutcome(0, False, False, None), False,
      "ERROR: Generation failed (exit code 0), see the log above"),
     (RunOutcome(-9, True, True, None), True, "INFO: Generation stopped.")],
)  # fmt: skip
def test_failures_stop_the_queue_with_a_message(parts, paths, tmp_path, outcome, image, expected):
    session, runner, events = parts
    session.generate(paths, (_request(tmp_path, "1.png"), _request(tmp_path, "2.png")), "0", False)
    if image:
        runner.line_received.emit(f"IMAGE: {tmp_path / 'x.png'}")

    runner.finished.emit(outcome)

    assert events["finished"] == [(False, expected)]
    assert len(runner.started) == 1 and not session.is_running()


def test_stop_clears_the_queue_and_stops_the_runner(parts, paths, tmp_path):
    session, runner, events = parts
    session.generate(paths, (_request(tmp_path, "1.png"), _request(tmp_path, "2.png")), "0", False)

    session.stop()
    runner.finished.emit(RunOutcome(-15, crashed=False, stopped=True, log_path=None))

    assert runner.stopped == 1
    assert events["finished"] == [(False, "INFO: Generation stopped.")]
    assert len(runner.started) == 1


def test_failed_start_is_reported(parts, paths, tmp_path):
    session, runner, events = parts
    session.generate(paths, (_request(tmp_path),), "0", training=False)

    runner.failed_to_start.emit("python not found")

    assert events["finished"] == [(False, "ERROR: python not found")]
    assert not session.is_running()


def test_other_lines_go_to_the_log_and_shutdown_reaches_the_runner(parts, paths, tmp_path):
    session, runner, events = parts
    session.generate(paths, (_request(tmp_path),), "0", training=False)

    runner.line_received.emit("INFO loading DiT")
    runner.line_received.emit("WARN: Out of GPU memory with standard, trying block_swap.")
    session.shutdown()

    assert events["log"][-2:] == [
        "INFO loading DiT",
        "WARN: Out of GPU memory with standard, trying block_swap.",
    ]
    assert runner.shut_down == 1
