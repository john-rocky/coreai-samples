# CoreAIImageGenMac

Type a prompt on a Mac and get a 1024×1024 image: FLUX.2 klein 4B runs on Apple's official Core AI diffusion
pipeline and draws one in 10.9 s on an M4 Max Mac Studio (4 steps, 2026-10-10).

The seed decides the starting noise: keep it to get the same image again, change it to get another one. Save… writes
the result as a PNG.

The model is [dushandz/FLUX.2-klein-4B-CoreAI](https://huggingface.co/dushandz/FLUX.2-klein-4B-CoreAI), a community
Core AI export of Black Forest Labs' [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(Apache-2.0) with 4-bit weights, pinned to revision `a7f65cdd`. Its model card says it was made with apple/coreai-models'
own recipe, `coreai.diffusion.export flux2-klein-4b --platform macOS`. The app downloads it on first use with
swift-transformers' `HubApi` and runs it with `FlowTransformerPipeline` from apple/coreai-models: no engine, no runtime
patch.

[mlboydaisuke/FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI) (exported 2026-07-20)
does not load with the pipeline pinned here. Its metadata.json has `version` instead of `metadata_version` and no
`name`, so loading stops with "unsupported metadata_version '0.1' (known: 0.2)". Its transformer also takes
`rotary_emb_cos` and `rotary_emb_sin`, where the pipeline now passes `img_ids` and `txt_ids`. That repo needs a
re-export with the current exporter.

## What the app does

1. **Download & Load** fetches 16 files (4,045,300,128 bytes) into `~/Library/Caches/CoreAIImageGenMac/`. Core AI then
   specializes the three models for this Mac and caches the result. Later launches skip both steps.
2. Type a prompt and press **Generate**. Steps default to 4. Guidance 1.0 runs one pass per step, because
   the model is guidance-distilled; above 1.0 the pipeline adds a second, unconditional pass per step.
3. **Stop** ends the run after the step in progress and drops the unfinished image.
4. **Local…** opens a bundle folder you exported yourself.

## Measured on an M4 Max Mac Studio

macOS 27.0 (26A428), Release build, Core AI GPU preferred, 2026-10-10. Loads and images were timed with no other GPU
work running; the download ran over Wi-Fi.

| What | Measured |
|---|---|
| Download, 16 files (4,045,300,128 bytes) | 307.7 s |
| First load: Core AI specializes the three models for this Mac (4.04 GB cache in `~/Library/Caches/coreai-cache/`) | 6.5 s |
| Load after a relaunch | 1.1 s |
| One 1024×1024 image, 4 steps, guidance 1.0, after a relaunch | 10.9 s (10.89 s and 10.88 s for two prompts) |
| The same, first image right after the first load | 13.2 s |
| Peak memory (physical footprint) | 9.5 GiB (9,776 MiB) |

After a relaunch, step 1 ends at 2.8 s, which includes reading the prompt. Each further step takes 2.5 s, and decoding
the image 0.7 s. The same prompt and seed gave the same image in two launches: both PNGs have the same SHA-256.

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

apple/coreai-models documents the export in `models/flux2/README.md`:

```bash
git clone https://github.com/apple/coreai-models && cd coreai-models
git checkout 97a14be40bb1c5b95badb08b8532530443d62d26
uv run coreai.diffusion.export flux2-klein-4b --platform macOS
# then in the app: Local… → exports/FLUX.2-klein-4B
```

## Notes

- Mac only. The June version also had an iPhone target; it is gone because it was never built or run against this
  pipeline.
- There is no negative-prompt field: `FlowTransformerPipeline` does not read `negativePrompt` for FLUX.2.
- Download, load and generation run as user-initiated activities, so macOS does not throttle them while the window is
  hidden. Without that, the download dropped to 0.06 MB/s with the screen locked.
- App Sandbox is off, so Local… can read any folder.
- Hands-off runs, for checks and screenshots:
  `CoreAIImageGenMac.app/Contents/MacOS/CoreAIImageGenMac -autoplay 1 -prompt "…" -steps 4 -seed 42 -out ~/Desktop/a.png -log 1`
  presses the same buttons a person would; `Sources/Autoplay.swift` lists the arguments. `-log 1` writes the status
  lines and a JSON of the measured numbers to `~/Library/Application Support/CoreAIImageGenMac/`.
