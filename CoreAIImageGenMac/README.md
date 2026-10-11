# CoreAIImageGenMac

Type a prompt on a Mac and get an image from one of two models on Core AI: FLUX.2 klein 4B draws a 1024×1024 image in
10.9 s, and RGBA-Image-2.1-Turbo ([mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI](https://huggingface.co/mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI))
draws a 1024×1024 RGBA image with a real alpha channel in 39.8 s, on an M4 Max Mac Studio (2026-10-10 and 2026-10-11).

Ask RGBA-Image-2.1-Turbo for a transparent background and its fourth channel is the cut-out: a sticker comes out
without a background to remove. The app shows it on a checkerboard, and Save… writes a PNG with the alpha channel.
The seed decides the starting noise of either model: keep it to get the same image again, change it to get another one.

## The two models

**FLUX.2 klein 4B** is [mlboydaisuke/FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI),
a Core AI export of Black Forest Labs' [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(Apache-2.0) made with apple/coreai-models' own exporter at the commit project.yml pins. The app offers two of its
folders, pinned to revision `039db98c`:

- **fp16** (`macos-fp16/`, the default): fp16 weights, a 14.1 GB download, 10.9 s per image.
- **int8** (`macos-int8/`): int8 weights on the text encoder and the transformer, a 7.5 GB download, 11.6 s per image.

Both match fp32 PyTorch component by component (see the gate below). The app downloads the one you pick with
swift-transformers' `HubApi` and runs it with `FlowTransformerPipeline` from apple/coreai-models: no engine, no runtime
patch. The first version of this app used [dushandz/FLUX.2-klein-4B-CoreAI](https://huggingface.co/dushandz/FLUX.2-klein-4B-CoreAI),
a community export made with the same exporter, while the repo above was re-exported; thanks to its author.

**RGBA-Image-2.1-Turbo** is the community zoo's Core AI port of Qwen's
[Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
([model card](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/rgba-image-2.1-turbo)). Its weights are
under the Qwen Research License: research and evaluation only, no commercial use. The app offers the repo's two DiTs,
pinned to revision `c3bd985c`:

- **bf16 DiT**: a 32.4 GB download (text encoder, DiT, three VAE decoders).
- **int8 DiT**: int8 Linear weights, a 25.8 GB download.

No Core AI pipeline covers this model, so the app runs its graphs one by one with the runtime's own `AIModel` and
`InferenceFunction`, following the bundle's `host/host_contract.md`: tokenizer (swift-transformers), text encoder,
8 DiT steps, VAE decoder. Core AI compiles each graph for this Mac on the first load (just in time), so running the
app needs no Xcode tools. No engine, no runtime patch.

## What the app does

1. Pick a model. **Download & Load** first checks that the disk has room for the files and for the cache Core AI
   writes on the first load (79.1 GB for RGBA-Image-2.1-Turbo with the bf16 DiT). Then it fetches the files into
   `~/Library/Caches/CoreAIImageGenMac/`, and Core AI compiles the models for this Mac and caches the result. Later
   launches skip both steps. Picking another model lets go of the loaded one.
2. Type a prompt and press **Generate**.
   - FLUX.2 klein 4B: Steps default to 4. Guidance 1.0 runs one pass per step, because the model is
     guidance-distilled; above 1.0 the pipeline adds a second, unconditional pass per step.
   - RGBA-Image-2.1-Turbo: 8 steps, fixed by the model. Size picks 256, 512 or 1024 px, square. For a transparent
     background, write the prompt in the model card's form, shown under the field: "This is an RGBA image with
     transparency. {subject}. The image has alpha channel and the background is transparent."
3. **Stop** ends the run after the step in progress and drops the unfinished image.
4. **Save…** writes the image as a PNG.
5. **Local…** opens a bundle folder you exported or downloaded yourself.

## Measured on an M4 Max Mac Studio

### FLUX.2 klein 4B

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

### RGBA-Image-2.1-Turbo

macOS 27.0 (26A428), Release build, 2026-10-11. Core AI compiled every graph just in time (GPU preferred,
`expectFrequentReshapes`). Loads and images were timed with no other GPU work running. The timed launches opened a
local copy of the repo, the same files, because the download was still running.

| What | bf16 DiT | int8 DiT |
|---|---|---|
| Download, 31 files, over Wi-Fi (802.11ax) | not finished: 3.05 of 32.42 GB in 1,360 s (2.2 MB/s) ¹ | not measured ² |
| First load: Core AI compiles the five graphs for this Mac | 82.5 s | 23.3 s ³ |
| The cache that leaves in `~/Library/Caches/coreai-cache/` | 46.7 GB | 25.8 GB ⁴ |
| Load after a relaunch | 6.0–23.1 s ⁵ | 5.0 s |
| One 1024×1024 image | 39.8–39.9 s | 42.1–42.2 s |
| One 512×512 image | 9.6–9.8 s | not measured |
| Peak memory (physical footprint), 1024×1024 | 18.9 GiB (19,397 MiB) | 20.1 GiB (20,546 MiB) |
| Peak memory (physical footprint), 512×512 | 13.9 GiB (14,275 MiB) | not measured |

¹ The download was still running when this was written; the full figure follows. It was paused for the timed launches,
then lost its connection 798 MB into the 15.1 GB text encoder file, and `HubApi` starts an interrupted file over.
curl, run alongside the app, got 1.3 MB/s from the same host.
² The int8 entry downloads the int8 DiT (7.56 GB) in place of the bf16 one; that download was not timed.
³ After the bf16 entry's first load, so the text encoder and the VAE decoders came from the cache. Compiling the int8
DiT took 18.9 s of it.
⁴ 18.2 GB of it, the text encoder and the VAE decoders, is the same cache the bf16 entry uses.
⁵ Three relaunches: 23.1, 14.3 and 6.0 s. The text encoder took 20.5, 11.6 and 3.3 s of them; the cause of the
difference was not found.

An image's time runs from pressing Generate to the finished image: tokenizer, text encoder, 8 DiT steps, VAE
decoder. A 1024×1024 DiT step took 4.72 s with the bf16 DiT and 5.02 s with the int8 one. The first image after the
first load took 39.9 s, as long as the later ones. Each time is from two launches with two images each (int8
1024×1024: three images in two launches). The same prompt and seed gave the same image in every launch: the PNGs have
the same SHA-256.

Gate against the zoo's Python engine: the app's final latent and PNG equal those of
[`pipeline_engine.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/qwenimage21/pipeline_engine.py)
bit for bit, given the same prompt and the same starting noise (the reference's `torch.randn` draw, read with
`-noiseFile`). That holds at 256², 512² and 1024² with the bf16 DiT, and at 256² with the int8 DiT compiled ahead of
time. With the bf16 DiT, the just-in-time graphs give the same bytes as the ahead-of-time ones. The model card gates the
Python engine against the fp32 diffusers pipeline (bf16 DiT, white-composited PSNR: 43.79 / 42.45 / 42.50 dB at 256² /
512² / 1024²).

The int8 DiT compiled just in time computes slightly differently from its ahead-of-time compile. Against the fp32
reference at 256² (the card's apple prompt, seed 1234), the app's int8 image is 42.25 dB and its final latent has corr
0.999924. The bar was 40 dB and 0.9998; the card's ahead-of-time int8 scores 41.76 dB and 0.999944.

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

### FLUX.2 klein 4B

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

### RGBA-Image-2.1-Turbo

The conversion scripts and every gate are in the zoo: the
[model card](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/rgba-image-2.1-turbo) and
[`conversion/qwenimage21/`](https://github.com/john-rocky/coreai-model-zoo/tree/main/conversion/qwenimage21). Local…
opens a folder laid out like the Hugging Face repo: `host/`, `tokenizer/` and the `.aimodel` folders.

The app also runs graphs compiled ahead of time, the form the zoo's Python engine runs. Compile them inside that folder
as the card's step 2 does, with `xcrun coreai-build compile <name>.aimodel --output aot/<name> --platform macOS
--architecture h16c --preferred-compute gpu --expect-frequent-reshapes` (h16c is the M4 Max; the app reads this Mac's
name from `AIModel.deviceArchitectureName`). The app then opens each `aot/<name>/<name>.<arch>.aimodelc` it finds with
the default options, and compiles the other graphs just in time.

## Notes

- Mac only. The June version also had an iPhone target; it is gone because it was never built or run against this
  pipeline.
- There is no negative-prompt field: `FlowTransformerPipeline` does not read `negativePrompt` for FLUX.2, and
  RGBA-Image-2.1-Turbo runs without classifier-free guidance.
- RGBA-Image-2.1-Turbo's starting noise comes from the seed through apple/coreai-models' `TorchRandomSource`, a port of
  `torch.randn`. The same seed gives almost the same noise as the reference pipeline (max|Δ| ≈ 1e-6), not the same image.
- The app opens RGBA-Image-2.1-Turbo's `.aimodel` graphs with the GPU preferred and `expectFrequentReshapes` on. With
  the default options, or with the GPU preferred alone, the DiT aborts the whole process at its first step
  (`ANERegion.mm:414 … Code=-19`, macOS 27.0 26A428).
- Download, load and generation run as user-initiated activities, so macOS does not throttle them while the window is
  hidden. Without that, the download dropped to 0.06 MB/s with the screen locked.
- App Sandbox is off, so Local… can read any folder.
- Hands-off runs, for checks and screenshots:
  `CoreAIImageGenMac.app/Contents/MacOS/CoreAIImageGenMac -autoplay 1 -model rgba-turbo-bf16 -size 1024 -prompt "…" -seed 42 -out ~/Desktop/a.png -log 1`
  presses the same buttons a person would; `Sources/Autoplay.swift` lists the arguments (`-local`, `-aot`,
  `-noiseFile`, `-stopAfterStep`, …). `-log 1` writes the status lines and a JSON of the measured numbers to
  `~/Library/Application Support/CoreAIImageGenMac/`.
