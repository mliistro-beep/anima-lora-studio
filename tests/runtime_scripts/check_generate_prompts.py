"""Run by the training venv's Python (see test_generate_prompts.py); prints a JSON report.

Passes the arguments built by the app (argv[1], JSON; argv[2] is sd-scripts) through
generate.py's parser and then sd-scripts' own anima_minimal_inference parser, as a real
generation does; no GPU or model.
"""

import json
import sys

import generate

sys.path.insert(0, sys.argv[2])
import anima_minimal_inference as ami  # noqa: E402

options = generate.parse_options(json.loads(sys.argv[1]))
profile = generate.parse_profiles(options.profiles)[0]
args = generate._inference_args(ami, options, profile)
print(json.dumps({"prompt": args.prompt, "negative_prompt": args.negative_prompt}))
