# Core AI Samples

**Sample apps for Apple's [Core AI](https://github.com/apple/coreai-models) framework**
(the Core ML successor, WWDC 2026, macOS/iOS 27) — the runnable examples the official
repo doesn't ship.

Every app builds against the **unmodified official runtime** out of the box. Models
come from Apple's official export recipes — or from the
[community model zoo](https://github.com/john-rocky/coreai-model-zoo) (most zoo
bundles run as-is; see [Zoo mode](#zoo-mode-community-models) for the rest).

| Sample | Platform | What it shows |
|---|---|---|
| [CoreAIChatMac](CoreAIChatMac/) | macOS | Chat with any exported LLM bundle — streaming, live load/TTFT/tok-s stats, gpt-oss "thinking" parsing |
| CLIPPhotoSearch | iOS | *(design stage — [memo](CLIPPhotoSearch-DESIGN.md))* on-device photo semantic search on the ANE |

Measured performance for everything here:
[apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench) ·
recipe-level knowledge: [zoo knowledge base](https://github.com/john-rocky/coreai-model-zoo/tree/main/knowledge).

## Quick start (CoreAIChatMac + gpt-oss-20b)

```bash
# 1. Export a model with Apple's official recipe
git clone https://github.com/apple/coreai-models && cd coreai-models
uv run coreai.llm.export openai/gpt-oss-20b        # ~13 GB, ~10 min

# 2. Build & run the app
brew install xcodegen
git clone https://github.com/john-rocky/coreai-samples && cd coreai-samples/CoreAIChatMac
xcodegen generate
open CoreAIChatMac.xcodeproj                        # Run (scheme is Release)

# 3. In the app: Choose Models Folder… → the exports/ dir → click the model
```

M4 Max reference numbers: gpt-oss-20b decodes ~78 tok/s, loads in ~2 s warm.

## Zoo mode (community models)

The [zoo](https://github.com/john-rocky/coreai-model-zoo)'s LLM bundles (Qwen3.5,
LFM2.5, Granite) are SSM/GDN hybrids carrying extra recurrent state — the stock
runtime stops at "Expected 2 states (KV cache), got 4". The zoo ships engine
patches for exactly this (Apple's repo takes no PRs, so they're patch files).
One-time setup:

```bash
./zoo/setup-zoo.sh                 # clone apple/coreai-models locally + apply zoo patches
# then build with the zoo project: open the regenerated CoreAIChatMac.xcodeproj
```

After that, **use the in-app "Get Models" button** to download zoo bundles straight
from Hugging Face (same atomic downloader as the zoo's iOS apps) — or point the app
at any bundle folder.

Multi-part zoo models (e.g. Gemma 4 PLE/table variants) need model-specific app
logic and live in the zoo's own apps instead.

## License

BSD-3-Clause (same as apple/coreai-models). Model weights keep their upstream licenses.
