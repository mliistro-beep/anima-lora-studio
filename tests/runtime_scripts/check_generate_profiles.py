"""Run by the training venv's Python (see test_generate_profiles.py); prints a JSON report.

Reads the --profiles JSON built by the app (argv[1]) with generate.py's own parser and loads
the VAE of each profile through a fake anima_train_utils module, so no GPU and no
sd-scripts are needed.
"""

import argparse
import json
import sys
import types

import generate


class FakeVae:
    def __init__(self):
        self.tiling = False
        self.dtype = None

    def to(self, dtype):
        self.dtype = str(dtype)
        return self

    def eval(self):
        return self

    def enable_tiling(self):
        self.tiling = True


def load_qwen_image_vae(args, device="cpu", disable_mmap=True):
    return FakeVae()


library = types.ModuleType("library")
library.anima_train_utils = types.SimpleNamespace(load_qwen_image_vae=load_qwen_image_vae)
sys.modules["library"] = library

profiles = generate.parse_profiles(sys.argv[1])
vaes = [generate._load_vae(argparse.Namespace(), profile) for profile in profiles]
print(
    json.dumps(
        {
            "names": [profile.name for profile in profiles],
            "tiling": [vae.tiling for vae in vaes],
            "dtype": vaes[0].dtype,
        }
    )
)
