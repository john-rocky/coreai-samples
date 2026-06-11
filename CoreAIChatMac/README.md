# CoreAIChatMac

A minimal macOS chat app for **Core AI language bundles on Apple's official runtime**
([apple/coreai-models](https://github.com/apple/coreai-models)). Point it at a folder
of exported bundles, click a model, chat — with live performance stats (load time,
TTFT, tok/s, memory footprint) in the footer.

Runs anything `coreai.llm.export` produces: gpt-oss-20b, qwen3, gemma3, mistral —
and zoo bundles. gpt-oss's harmony output is parsed into a collapsible "Thinking"
section + the final answer.

## Build & run

```bash
brew install xcodegen
cd apps/CoreAIChatMac
xcodegen generate
open CoreAIChatMac.xcodeproj   # Run (scheme uses Release — debug engine is ~3x slower)
```

Export a model first, e.g.:

```bash
git clone https://github.com/apple/coreai-models && cd coreai-models
uv run coreai.llm.export openai/gpt-oss-20b
```

Then in the app: **Choose Models Folder…** → select the `exports/` directory.

## Notes

- The model list shows any subdirectory containing a `metadata.json`.
- Multi-turn chat re-prefills the full history each turn via the bundle's own
  chat template (`tokenizer.applyChatTemplate`).
- Generation: temperature 0.7, max 2048 tokens per reply, official
  `VanillaDecodingStrategy` streaming.
- Measured on M4 Max 128GB: gpt-oss-20b decodes at ~78 tok/s
  (see `knowledge/apple-models-bench.md` for the full benchmark table).
