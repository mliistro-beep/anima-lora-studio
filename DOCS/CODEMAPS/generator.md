<!-- Generated: 2026-11-07 | Scope: image generator (M9) | Token estimate: ~900 -->
# Generator

The image generator (Generator tab in the window, Generate images in `anima-trainer tui`) and
the code Anima Merger Studio copies for its preview step ("copied and adapted, not imported",
tests included). The generation code has no ties to the training or to the UI: a request in,
one `generate.py` process per image, an image with its PNG metadata out.

## Flow
form (window: ui/tabs/generator_tab.py, TUI: cli/tui/generator_screens.py + cli/generator_values.py)
  -> GeneratorSettings (<data>/generator.json, shared by window and TUI)
  -> plan_generation / plan_queue (preview/generation_plan.py): checks model and LoRA headers,
     final prompt, seeds, file names, metadata -> GenerationRequest... or ERROR: lines
  -> GeneratorSession (workflow/generator_session.py): GPU lock, one request after the other, Stop
  -> build_generation_launch (preview/inference_command.py) -> <data>/.venv/bin/python generate.py
  -> generate.py: memory profile, LoRAs (PEFT renamed to kohya), sampler, PNG with text chunks
  -> INFO/WARN/ERROR lines, step progress, IMAGE: <path>

## Copy as they are (no import outside this list)
preview/safetensors_header.py  read_tensor_names(path), read_metadata(path); SafetensorsHeaderError
                               (header only, 100 MB limit, no torch)
preview/lora_files.py          read_lora(path) -> LoraFile(path, format kohya|peft, blocks, has_text_encoder, unused_keys),
                               lora_trigger(path) (modelspec.trigger_phrase), read_checkpoint_blocks(path),
                               check_lora(lora, blocks) -> ERROR:/WARN: lines; only Anima modules count
                               (blocks.N attention/mlp/adaln, llm_adapter); LoHa, LoKr, DoRA, LoCon, OFT
                               and .diff rejected
preview/progress_text.py       ImageProgress(image, images, batch, batches), progress_line(progress, done, total)
                               -> "img 03/10 | batch 2/4 | step 12/25"
runtime_scripts/generate.py    the inference script (training venv: torch, safetensors, sd-scripts'
                               anima_minimal_inference); options --dit --qwen3 --vae --lora/--lora_multiplier
                               (repeated) --prompt --negative_prompt --steps --cfg --flow_shift --seed
                               --sampler --profiles --margin_mib --ram_margin_mib --output --png_parameters
                               --png_settings
runtime_scripts/lora_inputs.py pair_loras(loras, strengths) -> ((file, strength), ...); ValueError when counts differ
runtime_scripts/lora_keys.py   kohya_key(peft_key), kohya_names(keys), lora_notes(...) -> INFO: renamed,
                               WARN: text encoder part ignored, WARN: unused keys
runtime_scripts/png_text.py    png_texts(parameters, settings) -> {"parameters", "anima_lora_studio"};
                               generate.py writes them as iTXt chunks (Pillow PngInfo)
runtime_scripts/profile_choice.py usable_profiles(needs, budget), not_enough_memory, next_profile_warning,
                               available_ram_mib(meminfo) (free VRAM and MemAvailable read once)
runtime_scripts/samplers.py    flow_sigmas, denoiser_from_velocity, euler, er_sde

The scripts in runtime_scripts/ import each other by plain module name: keep them in one folder,
started as files (`python <folder>/generate.py ...`), with sd-scripts' folder in
ANIMA_TRAINER_SD_SCRIPTS (repeated in generate.py on purpose, never imported from the app).

## Copy and adapt (app ties to replace)
preview/vram_profile.py        GenerationProfile, estimate_mib, fallback_chain, margin_mib, ram_margin_mib;
                               tie: SampleSize from preview/sample_prompts.py (a frozen width/height pair)
preview/generator_settings.py  LoraEntry(file, strength -2..2, enabled, trigger), GenerationEntry
                               (prompt, pre_prompt, negative, size, steps, cfg, shift, sampler, seed,
                               random_seed, count 1-10, loras max 8), GeneratorSettings (+ base_model,
                               model_file, output_folder, queue max 10), final_prompt();
                               ties: settings.schema (WholeNumber, DecimalNumber), settings.json_store,
                               settings.choices.BaseModelName, preview.settings (Sampler, DEFAULT_CFG, sides)
preview/image_metadata.py      image_metadata(entry, seed, checkpoint, version) -> ImageMetadata(parameters: A1111 text, settings: JSON
                               of that very image, seed fixed, count 1); file names only, never folders
preview/inference_command.py   LoraUse, GenerationRequest, output_image_path ({prefix}_{label}_{time}
                               [_{NN}]_{seed}.png), build_generation_launch -> LaunchSpec;
                               ties: RuntimePaths (venv python, qwen3, vae), training.command_builder
                               (LaunchSpec, SD_SCRIPTS_ENV), runtime_scripts.GENERATE
preview/generation_plan.py     GenerationTarget(project, checkpoints), random_seed() (secrets),
                               plan_generation(...), plan_queue(...) ("ERROR: Queue N: ...");
                               ties: RuntimePaths, __version__, preview/checkpoints.py (training project
                               checkpoints, Compare Epochs only), field_specs.INT_MAX
workflow/generator_session.py  GeneratorSession(QObject): generate(requests), stop, shutdown; signals
                               running_changed, log_line, step_progress(done, total), image_started
                               (ImageProgress), image_ready(Path), finished(ok, message);
                               ties: TrainingRunner (QProcess, process group), system/gpu_lock.py

## Tests to copy with them
tests/unit/preview/  test_safetensors_header, test_lora_files, test_lora_keys, test_lora_inputs,
                     test_png_text, test_profile_choice, test_progress_text, test_vram_profile,
                     test_generator_settings, test_image_metadata, test_inference_command,
                     test_generation_plan
tests/safetensors_helpers.py  writes small synthetic .safetensors files (kohya, PEFT, other models)
tests/runtime_scripts/        test_samplers, test_generate_prompts, test_generate_profiles,
                     test_generate_png (+ their check_*.py; run in an installed training venv with
                     ANIMA_TRAINER_TEST_RUNTIME=<data folder>, otherwise skipped)

## Not for the merger (app only)
ui/tabs/generator_tab.py, ui/widgets/lora_table.py, ui/widgets/queue_list.py, ui/preview_controller.py,
cli/generator_values.py, cli/terminal_generation.py, cli/tui/generator_screens.py,
cli/tui/generator_rows.py, preview/form_fields.py (form labels), preview/queue_text.py (queue row text)
