"""Run by the training venv's Python (see test_train_with_previews.py); prints a JSON report.

Uses a fake anima_train_utils module, so no GPU and no sd-scripts are needed.
"""

import json
import sys
import types

import torch
import train_with_previews as hooks

calls = []
reports = []
tiling_seen = []
disable_calls = []


class FakeVae:
    def __init__(self):
        self.device = torch.device("cpu")
        self.moves = []
        self.use_tiling = False

    def enable_tiling(self):
        self.use_tiling = True

    def disable_tiling(self):
        disable_calls.append(True)
        self.use_tiling = False

    def to(self, device):
        self.moves.append(str(device))
        self.device = torch.device(device)
        return self


class FakeVae2d(FakeVae):
    """sd-scripts' image-only VAE: no use_tiling, enable_tiling raises."""

    def __init__(self):
        super().__init__()
        del self.use_tiling

    def enable_tiling(self):
        raise NotImplementedError


class TiledVae(FakeVae):
    """A VAE the training already set to tiles: it must stay so."""

    def __init__(self):
        super().__init__()
        self.use_tiling = True


def euler_do_sample(*args, **kwargs):
    calls.append("euler")
    return "euler-latents"


def inference(accelerator, args, dit, text_encoder, vae, tokenize_strategy,
              text_encoding_strategy, save_dir, prompt_dict, epoch, steps,
              sample_prompts_te_outputs, prompt_replacement):  # fmt: skip
    tiling_seen.append(getattr(vae, "use_tiling", None))
    if prompt_dict.get("oom"):
        vae.to("meta")  # moved to the GPU before decoding, then the decode runs out of memory
        raise torch.OutOfMemoryError("CUDA out of memory")
    if prompt_dict.get("fail"):
        raise RuntimeError("not a memory error")
    return utils.do_sample(64, 64, 1, None, None, 2, torch.float32, torch.device("cpu"))


utils = types.ModuleType("anima_train_utils")
utils.do_sample = euler_do_sample
utils._sample_image_inference = inference
hooks.er_sde_do_sample = lambda *a, **k: calls.append("er_sde") or "er-latents"
hooks.install_hooks(utils, reports.append)


def run(prompt, vae_class=FakeVae):
    """One preview through the hooks: what it returned and what it did to the VAE."""
    vae = vae_class()
    calls.clear(), tiling_seen.clear(), disable_calls.clear()
    error = None
    try:
        result = utils._sample_image_inference(*([None] * 4), vae, None, None, "dir", prompt,
                                               1, 10, None, None)  # fmt: skip
    except RuntimeError as exc:
        result, error = None, type(exc).__name__
    return {
        "result": result,
        "error": error,
        "samplers": list(calls),
        "tiling_during": tiling_seen[0],
        "tiling_after": getattr(vae, "use_tiling", None),
        "tiling_disabled": bool(disable_calls),
        "vae_device": str(vae.device),
    }


cases = {
    "er_sde": run({"sample_sampler": "er_sde", "enum": 0}),
    "euler": run({"sample_sampler": "euler", "enum": 1}),
    "default": run({"enum": 2}),
    "oom": run({"sample_sampler": "er_sde", "enum": 3, "oom": True}),
    "after_oom": run({"enum": 4}),
    "vae_2d": run({"enum": 5}, FakeVae2d),
    "already_tiled": run({"enum": 6}, TiledVae),
    "other_error": run({"enum": 7, "fail": True}),
}

# The real ER-SDE do_sample on the CPU with a tiny fake DiT (velocity = 0.1 * input).
import importlib  # noqa: E402

real = importlib.reload(hooks)


def fake_dit(x, t, emb, padding_mask=None):
    return 0.1 * x


first = real.er_sde_do_sample(64, 64, 5, fake_dit, None, 4, torch.float32, torch.device("cpu"))
again = real.er_sde_do_sample(64, 64, 5, fake_dit, None, 4, torch.float32, torch.device("cpu"))
other = real.er_sde_do_sample(64, 64, 6, fake_dit, None, 4, torch.float32, torch.device("cpu"))

print(
    json.dumps(
        {
            "cases": cases,
            "reports": reports,
            "shape": list(first.shape),
            "same_seed_equal": bool(torch.equal(first, again)),
            "other_seed_differs": not torch.equal(first, other),
        }
    )
)
sys.exit(0)
