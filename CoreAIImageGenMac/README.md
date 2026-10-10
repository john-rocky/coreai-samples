# CoreAIImageGenMac

Type a prompt on a Mac and get a 1024×1024 image: FLUX.2 klein 4B runs on Apple's official Core AI diffusion
pipeline and draws one in 10.9 s on an M4 Max Mac Studio (4 steps, 2026-10-10).

The seed decides the starting noise: keep it to get the same image again, change it to get another one. Save… writes
the result as a PNG.

The model is [mlboydaisuke/FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI), a Core AI
export of Black Forest Labs' [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0)
made with apple/coreai-models' own exporter at the commit project.yml pins. The app offers two of its folders, pinned
to revision `039db98c`:

- **fp16** (`macos-fp16/`, the default): fp16 weights, a 14.1 GB download, 10.9 s per image.
- **int8** (`macos-int8/`): int8 weights on the text encoder and the transformer, a 7.5 GB download, 11.6 s per image.

Both match fp32 PyTorch component by component (see the gate below). The app downloads the one you pick with
swift-transformers' `HubApi` and runs it with `FlowTransformerPipeline` from apple/coreai-models: no engine, no runtime
patch. The first version of this app used [dushandz/FLUX.2-klein-4B-CoreAI](https://huggingface.co/dushandz/FLUX.2-klein-4B-CoreAI),
a community export made with the same exporter, while the repo above was re-exported; thanks to its author.

## What the app does

1. Pick fp16 or int8. **Download & Load** fetches the folder's 16 files into `~/Library/Caches/CoreAIImageGenMac/`.
   Core AI then specializes the three models for this Mac and caches the result. Later launches skip both steps.
   Picking the other model lets go of the loaded one.
2. Type a prompt and press **Generate**. Steps default to 4. Guidance 1.0 runs one pass per step, because
   the model is guidance-distilled; above 1.0 the pipeline adds a second, unconditional pass per step.
3. **Stop** ends the run after the step in progress and drops the unfinished image.
4. **Local…** opens a bundle folder you exported yourself.

## Measured on an M4 Max Mac Studio

macOS 27.0 (26A428), Release build, Core AI GPU preferred, 2026-10-10. Loads and images were timed with no other GPU
work running. The downloads ran over Wi-Fi; another download shared the link during part of the int8 one.

| What | fp16 | int8 |
|---|---|---|
| Download, 16 files | 14,090,986,305 bytes, 2,900 s | 7,540,032,209 bytes, 2,686 s |
| First load right after the download: Core AI specializes the three models for this Mac | 34.1 s | 8.7 s ¹ |
| The cache that leaves in `~/Library/Caches/coreai-cache/` | 14.1 GB | 7.5 GB |
| Load after a relaunch | 17.7–24.4 s ² | 1.5 s |
| One 1024×1024 image, 4 steps, guidance 1.0, after a relaunch | 10.9 s | 11.6 s |
| Peak memory (physical footprint) | 8.3 GiB (8,505 MiB) | 9.7 GiB (9,885 MiB) |

¹ Before the upload, a first load of the same int8 files took 21.9 s; what makes the difference was not found.
² Six relaunches. Earlier the same day, three relaunches made right after a first load took 2.1 s; the cause of the
difference was not found. The int8 relaunch took 1.5 s every time.

An image is the median of four runs: two prompts, each in two launches. The same prompt and seed gave the same image
in every launch: the PNGs have the same SHA-256.

Gate against fp32 PyTorch (diffusers' `Flux2KleinPipeline` through apple/coreai-models' own export wrappers), fp16 /
int8: text encoder cosine 0.99966–0.99995 / 0.9928–0.9948, transformer 0.999987 / 0.99981, VAE decoder 65.5 dB, the
seed-42 image 31.0 / 23.3 dB from the fp32 image. Both pass their bar (0.999 unquantized, 0.99 quantized). The
exporter's default for this model, int4, fails it: its image is 15.9 dB from fp32 and the layout changes.

## Build & run

```bash
brew install xcodegen
cd CoreAIImageGenMac
xcodegen generate
open CoreAIImageGenMac.xcodeproj   # set your team, then Run (the scheme uses Release)
```

`project.yml` pins apple/coreai-models to commit `97a14be` (2026-10-09). Its diffusion API changes between commits: the
June version of this app called `Flux2Pipeline` and `PipelineDescriptor`, which no longer exist.

## Your own export

The two folders come from apple/coreai-models' exporter at the pinned commit:

```bash
git clone https://github.com/apple/coreai-models && cd coreai-models
git checkout 97a14be40bb1c5b95badb08b8532530443d62d26
```

fp16 (the app's `macos-fp16/` is this folder without `Transformer_img2img_full.aimodel`):

```bash
uv run coreai.diffusion.export flux2-klein-4b --platform macOS \
  --compression none --single-function --resolution 1024 --output-dir exports/fp16
```

int8, the exporter's own 4bit preset changed to dtype int8:

```bash
cat > int8-block32.json <<'EOF'
{"execution_mode": "eager",
 "global_config": {"op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric_with_clipping", "granularity": {"type": "per_block", "block_size": 32}}}, "op_input_spec": null, "op_output_spec": null},
 "module_type_configs": {"diffusers.models.normalization.RMSNorm": null, "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm": null, "transformers.models.gemma2.modeling_gemma2.Gemma2RMSNorm": null, "transformers.models.umt5.modeling_umt5.UMT5LayerNorm": null}}
EOF
uv run coreai.diffusion.export flux2-klein-4b --platform macOS \
  --compression "$(cat int8-block32.json)" --output-dir exports/int8
```

Then in the app: Local… → `exports/fp16/FLUX.2-klein-4B` (or `exports/int8/FLUX.2-klein-4B`). `--compression` takes a
preset name or a JSON string, not a file path. Give each export its own `--output-dir`: the exporter skips any asset
that already exists in its folder.

## Notes

- Mac only. The June version also had an iPhone target; it is gone because it was never built or run against this
  pipeline.
- There is no negative-prompt field: `FlowTransformerPipeline` does not read `negativePrompt` for FLUX.2.
- Download, load and generation run as user-initiated activities, so macOS does not throttle them while the window is
  hidden. Without that, the download dropped to 0.06 MB/s with the screen locked.
- App Sandbox is off, so Local… can read any folder.
- Hands-off runs, for checks and screenshots:
  `CoreAIImageGenMac.app/Contents/MacOS/CoreAIImageGenMac -autoplay 1 -model int8 -prompt "…" -steps 4 -seed 42 -out ~/Desktop/a.png -log 1`
  presses the same buttons a person would; `Sources/Autoplay.swift` lists the arguments. `-log 1` writes the status
  lines and a JSON of the measured numbers to `~/Library/Application Support/CoreAIImageGenMac/`.
