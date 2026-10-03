# CoreAIDecisionFormMac

Copy a customer email, press **Paste**, and a support ticket form fills at once: team, priority, customer
mood, next step, needed-by date and three yes/no flags. Each field is one question with fixed options. The
model reads the email once and returns a probability for every option of every question; nothing is
generated.

[Watch the 13 s clip](https://github.com/john-rocky/coreai-assets/blob/main/demos/clef-flash-mac.mp4): the sample email fills the form on an M4 Max Mac Studio ([how it was recorded](https://github.com/john-rocky/coreai-assets/tree/main/demos#clef-flash-macmp4-13-s)).

The model is [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Qwen3.5-9B with a joint
schema head, Apache-2.0), converted to Core AI:
[mlboydaisuke/clef-flash-CoreAI](https://huggingface.co/mlboydaisuke/clef-flash-CoreAI). The host code is the
zoo's `ClefFlash` package, vendored unmodified in [`ClefFlash/`](ClefFlash/) (from
[coreai-model-zoo `apps/ClefFlash`](https://github.com/john-rocky/coreai-model-zoo/tree/main/apps/ClefFlash) at
5ef2247). It drives Apple's official runtime directly (`AIModel` + `loadFunction`): no engine, no runtime patch.

## The request

One request per email, in the SystemOne-compatible shape: the email as the state, the form's fields as
typed questions (`choice`, `score`, `noul` = yes/no).

```swift
import ClefFlash
import CoreAI

let root = URL(filePath: "clef-flash-CoreAI")                        // a download of the HF repo
var decoderOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)
decoderOptions.expectFrequentReshapes = true
let gpu = SpecializationOptions(preferredComputeUnitKind: .gpu)
let decider = try await ClefDecider(
    assets: .init(decoderBundle: root.appending(path: "gpu-pipelined/clef_flash_decode_fp16_pf64"),
                  head: root.appending(path: "gpu-pipelined/clef_flash_head_bucket_fp16w32"),
                  table: root.appending(path: "host/lm_head_fp16.bin")),
    decoderOptions: decoderOptions, headOptions: gpu, towerOptions: gpu)

let request = try SystemOneRequest(json: .obj([
    ("model", .string("clef-flash")),
    ("state", .string(email)),
    ("questions", .obj([
        ("team", .obj([("type", .string("choice")), ("instructions", .string("Which team should handle this email?")),
                       ("criteria", .obj([("billing", .string("Payments, charges and refunds")),
                                          ("technical", .string("A product is faulty or does not work")),
                                          ("shipping", .string("Delivery, tracking and lost parcels")),
                                          ("account", .string("Sign-in and profile"))]))])),
        ("mood", .obj([("type", .string("score")), ("instructions", .string("How does the customer feel?")),
                       ("criteria", .array([.string("Calm"), .string("Annoyed"), .string("Frustrated"), .string("Angry")]))])),
        ("refund", .obj([("type", .string("noul")), ("instructions", .string("The customer asks for money back."))])),
    ])),
]))
let response = try await decider.decide(request: request)
// {"model":"clef-flash","answers":{"team":{"type":"choice","choice":"technical","confidence":…,"probabilities":{…}},
//  "mood":{"type":"score","score":…,"confidence":…,"legend":{…},"probabilities":{…}},"refund":{"type":"noul","noul":…}},
//  "usage":{"input_tokens":…,"output_tokens":0}}
```

The app's eight questions are `fields` in [`Sources/DecisionFormModel.swift`](Sources/DecisionFormModel.swift).
Change a question or its options there and the form follows.

## Get the model

The text form needs three folders of the repo, about 18.2 GB (the fp16 decoder 15.9 GB, the `lm_head` table
2.0 GB, the head 0.24 GB):

```bash
hf download mlboydaisuke/clef-flash-CoreAI \
    --include "gpu-pipelined/clef_flash_decode_fp16_pf64/*" "gpu-pipelined/clef_flash_head_bucket_fp16w32/*" "host/*" \
    --local-dir ~/Downloads/clef-flash-CoreAI
```

## Build & run

```bash
brew install xcodegen
cd CoreAIDecisionFormMac
xcodegen generate
open CoreAIDecisionFormMac.xcodeproj   # Run (the scheme uses Release)
```

The app loads `~/Downloads/clef-flash-CoreAI` when that folder exists; otherwise **Choose Models Folder…** →
the downloaded folder (the one holding `gpu-pipelined/` and `host/`).

## Notes

- Mac only. The fp16 decoder alone is 15.9 GB; an iPhone app gets about 6.4 GB.
- The first load specializes the `.aimodel` files for this Mac and writes about 30 GB to
  `~/Library/Caches/coreai-cache/`. The model zoo's CLI measured that first specialization on an idle M4 Max
  Mac Studio: 56.1 s, then 4.5 s from the cache (the model card's JIT row); the app's own first load was not timed.
  Later loads read that cache. The app then runs one decision on a short text, because the first decision of a
  process is slow.
- The sample email with the eight questions is 934 tokens, 809 of them the questions and their options; the
  decoder reads 64 tokens per call, so that is 15 calls. Fewer questions or options mean fewer calls.
- What a field shows: `choice` → the most probable option; `score` → the most probable level, and the bar is
  the expected level; `noul` → Yes when p(yes) ≥ 0.5. The p next to it is the probability of what is shown.
- Hands-off run, for recordings: `CoreAIDecisionFormMac.app/Contents/MacOS/CoreAIDecisionFormMac -autoplay form
  -modelsFolder <dir> [-trigger <file>] [-delay <s>] [-log 1]` presses Sample email once the model is ready
  (`Sources/Autoplay.swift`).
