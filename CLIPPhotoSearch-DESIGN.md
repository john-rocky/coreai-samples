# CLIPPhotoSearch — design memo (pre-implementation)

Find "red bike at the beach" in your own photo library, fully on-device. iOS app on
Apple's official Core AI runtime + the official CLIP recipe. Pairs with the measured
numbers in `knowledge/apple-models-bench.md`: CLIP fp16 on ANE = **3.68 ms/inference,
289 MB** — indexing 10,000 photos ≈ 40 s of pure ANE compute, battery-friendly.

## Why this app

- Most-requested CV demo genre; instantly understandable in a 15-second video.
- Shows the zoo's unique angle: official recipe + `--dtype float16` + ANE = numbers
  nobody else publishes.
- Foundation for the embeddings pillar (same index/search machinery reused for text
  RAG later).

## Model plumbing (the one real design decision)

The official `models/clip/export.py` exports a **joint** function:
`(pixel_values, input_ids, attention_mask) → (image_embeds, text_embeds, ...)` —
both towers run on every call.

- **v1 (recipe untouched):** call the joint function with a fixed dummy text during
  photo indexing, and a fixed dummy image during text queries. Wasted tower cost is
  acceptable: text tower at seq-77 is small, and queries are one call each. Keep the
  official artifact byte-identical to the bench table.
- **v2 (tiny recipe fork, optional):** export image/text towers as separate functions
  via a function map (runtime supports multi-function assets). ~2× indexing speedup;
  do it only if v1 indexing UX feels slow on-device.

## Architecture

- `PhotoIndexer` (actor): PhotoKit enumeration → `CGImage` → 224×224 pixel buffer →
  Core AI fp16 ANE inference → L2-normalized 512-dim Float16 vector. Store
  `(localIdentifier, vector)` in a flat memory-mapped file (10k photos × 512 × 2 B ≈
  10 MB — no vector DB needed). Incremental: PHChange observer re-indexes only new
  assets. Background-friendly batches (e.g. 64/burst) to stay cool.
- `SearchEngine`: query → tokenizer (bundled CLIP BPE via swift-transformers) →
  text embedding → brute-force cosine over the mmap (10k × 512 dot products ≈
  instant with Accelerate/vDSP; no ANN index until ~1M photos).
- `UI`: single search field + results grid (PHCachingImageManager thumbnails),
  latency chip showing query-to-results ms (the demo stat), index progress ring on
  first launch.
- Compute units: `SpecializationOptions.from_preferred_compute_unit_kind(.neural_engine)`,
  fp16 artifact embedded in the app (289 MB) or downloaded from HF on first run
  (zoo in-app-download pattern from CoreAIChat).

## Privacy story (for the post)

Photos never leave the device; no network after model download; index is a local file.

## Open items before implementation

- Verify the bundled tokenizer files in the CLIP export cover Swift-side encoding
  (export saves the HF processor? if not, ship tokenizer.json alongside).
- Measure end-to-end per-photo cost incl. decode/resize (CPU side may dominate the
  3.7 ms ANE inference; batch the resize on a utility queue).
- iPhone CV harness numbers first (same .aimodel on A19 ANE) — blocked on device
  session availability; Mac numbers above are the planning basis.
