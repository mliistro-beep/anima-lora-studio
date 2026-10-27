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


class FakeVae:
    def __init__(self):
        self.device = torch.device("cpu")
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        self.device = torch.device(device)
        return self


def euler_do_sample(*args, **kwargs):
    calls.append("euler")
    return "euler-latents"


def inference(accelerator, args, dit, text_encoder, vae, tokenize_strategy,
              text_encoding_strategy, save_dir, prompt_dict, epoch, steps,
              sample_prompts_te_outputs, prompt_replacement):  # fmt: skip
    if prompt_dict.get("oom"):
        vae.to("meta")  # moved to the GPU before decoding, then the decode runs out of memory
        raise torch.OutOfMemoryError("CUDA out of memory")
    return utils.do_sample(64, 64, 1, None, None, 2, torch.float32, torch.device("cpu"))


utils = types.ModuleType("anima_train_utils")
utils.do_sample = euler_do_sample
utils._sample_image_inference = inference
hooks.er_sde_do_sample = lambda *a, **k: calls.append("er_sde") or "er-latents"
hooks.install_hooks(utils, reports.append)


def run(prompt):
    vae = FakeVae()
    result = utils._sample_image_inference(*([None] * 4), vae, None, None, "dir", prompt,
                                           1, 10, None, None)  # fmt: skip
    return result, vae


er_result, _ = run({"sample_sampler": "er_sde", "enum": 0})
euler_result, _ = run({"sample_sampler": "euler", "enum": 1})
default_result, _ = run({"enum": 2})
oom_result, oom_vae = run({"sample_sampler": "er_sde", "enum": 3, "oom": True})
after_oom, _ = run({"enum": 4})

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
            "results": [er_result, euler_result, default_result, oom_result, after_oom],
            "calls": calls,
            "reports": reports,
            "oom_vae_device": str(oom_vae.device),
            "shape": list(first.shape),
            "same_seed_equal": bool(torch.equal(first, again)),
            "other_seed_differs": not torch.equal(first, other),
        }
    )
)
sys.exit(0)
