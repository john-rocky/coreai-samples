# CoreAISearchiOS

Search your own notes by meaning on an iPhone: Granite Embedding 97M, converted with Apple's coreai-torch, runs on the
unmodified Core AI runtime and answers a query in 16.5 ms on an iPhone 18 Pro (2026-10-09).

Paste notes or import a `.txt` file, then type what you remember in other words: "where did I put the extra key for
my bicycle" finds "The spare key to the bike lock is taped under the second drawer of my desk." Two public-domain
Sherlock Holmes books (9,859 sentences) come with the app, so there is something to search before you add your own
text.

The model is [mlboydaisuke/Granite-Embedding-97M-Multilingual-R2-CoreAI](https://huggingface.co/mlboydaisuke/Granite-Embedding-97M-Multilingual-R2-CoreAI)
(IBM's [granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2),
Apache-2.0): the JIT bundle in `ios/fp32-s128`, pinned to revision `fdb16fc6`. The app downloads it on the first launch
with swift-transformers' `HubApi` and runs it through `AIModel` + `loadFunction`, GPU preferred: no engine, no runtime
patch.

## What the app does

1. First launch: **Download the model (415 MB)**. Core AI then compiles the model for this iPhone (the first load), and
   the app embeds the 9,859 example sentences once; later launches read both from disk.
2. **Add** opens a sheet: paste text or **Import .txt** (UTF-8). The text is split into sentences, up to 2,000 at a
   time; each sentence is embedded and saved with the app.
3. Type in the search field. The query is embedded and compared with every sentence (cosine similarity), and the 8
   best show with their source. Once you have added text, the search opens in **My notes**, which searches only your
   text; **All** adds the two books. Tap a result to read the sentences around it.

My notes is the default because the books can outrank your own sentence in All. With five test notes added, three of
five reworded queries put the right note first in All. For "internet login for the holiday house", a Holmes sentence
scored 0.0006 above the Wi-Fi note. My notes put the right note first for all five.

## Measured on an iPhone 18 Pro

iOS 27.2 (24B5099f), Release build, Core AI GPU preferred, the JIT `.aimodel`, one run, 2026-10-09.

| What | Measured |
|---|---|
| Download, 6 files (415,304,549 bytes) | 94.5 s over Wi-Fi |
| First load: Core AI compiles the model for this iPhone | 0.44 s, then 612 ms for the first embedding |
| Load after a relaunch | 0.054 s (the 25 MB tokenizer file takes 0.46 s on every launch) |
| Example books: 9,859 sentences, embedded once | 48.4 s = 4.91 ms per sentence (4.61 ms in the first 20 s) |
| Your text: 100 sentences | 4.8 ms per sentence |
| One query: embed + rank 9,864 sentences | 16.5 ms (median of 10, after one warm-up query) |
| Peak memory (physical footprint) | 216 MB |

The 9,859 book vectors made on this iPhone 18 Pro are bit-identical to the ones made on an iPhone 17 Pro with the
model compiled ahead of time for it (same SHA-256, 2026-09-20), so the five example queries return the same top 5 with
the same scores.

## Build & run

```bash
brew install xcodegen
cd CoreAISearchiOS
xcodegen generate
open CoreAISearchiOS.xcodeproj   # set your team, then Run on an iPhone with iOS 27 (the scheme uses Release)
```

## Notes

- The model files live in the app's Caches folder: they are not backed up, and if iOS removes them to free space the
  app offers the download again. A failed download keeps the files that finished; **Retry** downloads the rest.
- Your sentences and their vectors are saved in the app's Documents folder. **⋯ → Clear my notes** removes them.
- The tokenizer, `Sources/GraniteTokenizer.swift`, is part of the model's contract: no prefix, no stripping, the text
  cut at 126 tokens, then CLS and SEP. It reproduces the ids of the 35 fixtures in the model repo's `reference.json`.
- The example books are *The Adventures of Sherlock Holmes* and *The Hound of the Baskervilles* (Project Gutenberg
  #1661 and #2852, public domain), with the Gutenberg header and licence text removed.
- Hands-off runs, for checks and recordings, use launch arguments after `--`:
  `xcrun devicectl device process launch --device <id> com.coreai.searchios -- -autoplay 1 -notes notes.txt -query "…" -log 1`.
  They press the same buttons a person would; `Sources/Autoplay.swift` lists them. `-log 1` writes the status lines
  and a JSON of the measured numbers to the app's Documents folder.
