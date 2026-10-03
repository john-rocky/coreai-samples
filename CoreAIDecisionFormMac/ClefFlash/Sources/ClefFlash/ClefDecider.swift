// Vendored from github.com/john-rocky/coreai-model-zoo apps/ClefFlash/Sources/ClefFlash/ClefDecider.swift at 5ef2247, unmodified below this line.
// ClefDecider — a SystemOne request (+ one image at a fixed grid) -> the SystemOne response, the order decide.py runs:
//
//   let decider = try await ClefDecider(assets: .init(decoderBundle: bundleDir, decoder: aimodelcURL, head: headDir,
//                                                     headAsset: headAimodelcURL, table: tableURL,
//                                                     towers: [.g448: towerURL]))
//   let response = try await decider.decide(request: try SystemOneRequest(data: json), image: cgImage, grid: .g448)
//
//   request ──PromptBuilder──> ids, question / option spans, the decoder's static inputs
//   image ──ImagePreprocess (Pillow's bicubic)──> patches [4 G^2, 1536] ──tower──> image_embeds [G^2, 4096]
//   decoder: fresh zero states, ceil(T / 64) calls of `main` -> hidden [T, 4096] fp16
//   head arrays (span means, last token, lexical table rows, membership, types, masks) padded to the bucket
//   head `t<bucket>` -> one logit per option -> per question a float32 softmax -> the response (4 decimals)
//
// Contract checks at load: the bundle metadata (prefill chunk, context, vocabulary, image rows, prompt lengths), the
// tokenizer's special ids and the prompt's fixed pieces, the decoder's / head buckets' / towers' input, output and
// state names, shapes and types, the table's byte count. An asset that differs fails there, not in a probability.

import CoreAI
import CoreGraphics
import Foundation

@available(macOS 27, *)
public final class ClefDecider: @unchecked Sendable {
    public enum Grid: Int, Sendable, CaseIterable, CustomStringConvertible {
        case g256 = 8
        case g448 = 14

        public var side: Int { rawValue }
        public var tile: Int { ImagePreprocess.tileSide(grid: rawValue) }
        public var merged: MergedGrid { MergedGrid(h: rawValue, w: rawValue) }
        public var description: String { "g\(tile)" }

        public init?(tile: Int) {
            guard let g = Grid.allCases.first(where: { $0.tile == tile }) else { return nil }
            self = g
        }
    }

    /// Where the assets are. `decoder` / `headAsset` = the asset to load (`.aimodelc` AOT or `.aimodel`); nil = the
    /// bundle's `.aimodel`.
    public struct Assets: Sendable {
        public var decoderBundle: URL
        public var decoder: URL?
        public var head: URL
        public var headAsset: URL?
        public var table: URL
        public var towers: [Grid: URL]

        public init(decoderBundle: URL, decoder: URL? = nil, head: URL, headAsset: URL? = nil, table: URL,
                    towers: [Grid: URL] = [:]) {
            self.decoderBundle = decoderBundle
            self.decoder = decoder
            self.head = head
            self.headAsset = headAsset
            self.table = table
            self.towers = towers
        }
    }

    /// What the bundles' metadata.json files say.
    public struct Metadata: Sendable {
        public let name: String
        public let asset: String
        public let vocab: Int
        public let maxContext: Int
        public let chunk: Int
        public let nImageMax: Int
        public let prefixTokens: Int
        public let suffixTokens: Int
        public let headName: String
        public let headAsset: String
        public let buckets: [ClefHead.Bucket]

        public init(bundle: URL, head: URL) throws {
            let url = bundle.appendingPathComponent("metadata.json")
            let j = try JSONParser.parse(Data(contentsOf: url))
            guard let lang = j["language"], let asset = j["assets"]?["main"]?.string,
                  let vocab = lang["vocab_size"]?.intValue, let ctx = lang["max_context_length"]?.intValue,
                  let chunk = lang["prefill_chunk"]?.intValue, let nmax = j["vision"]?["n_image_max"]?.intValue,
                  let prompt = j["decision"]?["prompt"], let pre = prompt["prefix_tokens"]?.intValue,
                  let suf = prompt["suffix_tokens"]?.intValue
            else { throw ClefFlashError.bundle("\(url.path): no language / assets / vision / decision.prompt block") }
            name = j["name"]?.string ?? bundle.lastPathComponent
            self.asset = asset
            self.vocab = vocab
            maxContext = ctx
            self.chunk = chunk
            nImageMax = nmax
            prefixTokens = pre
            suffixTokens = suf
            let hurl = head.appendingPathComponent("metadata.json")
            let h = try JSONParser.parse(Data(contentsOf: hurl))
            guard let hm = h["head"], hm["shape"]?.string == "bucket", let b = hm["buckets"]?.array,
                  let hasset = h["assets"]?["main"]?.string
            else { throw ClefFlashError.bundle("\(hurl.path): not a bucket head (head.shape / head.buckets / assets.main)") }
            headName = h["name"]?.string ?? head.lastPathComponent
            headAsset = hasset
            buckets = try b.map { e in
                guard let a = e.array, a.count == 2, let f = a[0].string, let t = a[1].intValue else {
                    throw ClefFlashError.bundle("\(hurl.path): bucket \(PythonJSON.dumps(e))")
                }
                return ClefHead.Bucket(function: f, tokens: t)
            }
        }
    }

    /// Everything one decision did, for gates and timing.
    public struct Trace: Sendable {
        public let row: PromptRow
        public let prepared: ImagePreprocess.Prepared?
        /// The tower output, or the image rows given to `trace(embeds:)`.
        public let imageRows: [Float]?
        public let pass: ClefDecoder.Pass
        public let bucket: ClefHead.Bucket
        /// One logit per real option, in the head's option order.
        public let logits: [Float]
        public let probabilities: [[Float]]
        public let response: JSONValue
        public let seconds: [String: Double]
    }

    public let metadata: Metadata
    public let prompt: PromptBuilder
    public let decoder: ClefDecoder
    public let head: ClefHead
    public let table: LexicalTable
    public private(set) var towers: [Grid: VisionTower]
    public let tokenizerLoadSeconds: Double

    public init(assets: Assets, decoderOptions: SpecializationOptions = .default,
                headOptions: SpecializationOptions = .default, towerOptions: SpecializationOptions = .default) async throws
    {
        let meta = try Metadata(bundle: assets.decoderBundle, head: assets.head)
        guard meta.vocab == PromptBuilder.vocab, meta.nImageMax == ClefDecoder.imageRows else {
            throw ClefFlashError.contract("bundle vocab \(meta.vocab) / n_image_max \(meta.nImageMax) (want \(PromptBuilder.vocab) / \(ClefDecoder.imageRows))")
        }
        let t0 = ContinuousClock.now
        prompt = try await PromptBuilder.load(tokenizerFolder: assets.decoderBundle.appendingPathComponent("tokenizer"),
                                              nImageMax: meta.nImageMax, prefixTokens: meta.prefixTokens,
                                              suffixTokens: meta.suffixTokens)
        tokenizerLoadSeconds = secondsSince(t0)
        table = try LexicalTable(contentsOf: assets.table)
        var loaded: [Grid: VisionTower] = [:]
        for (g, url) in assets.towers.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            loaded[g] = try await VisionTower(contentsOf: url, grid: g.side, options: towerOptions)
        }
        towers = loaded
        head = try await ClefHead(contentsOf: assets.headAsset ?? assets.head.appendingPathComponent(meta.headAsset),
                                  buckets: meta.buckets, options: headOptions)
        decoder = try await ClefDecoder(contentsOf: assets.decoder ?? assets.decoderBundle.appendingPathComponent(meta.asset),
                                        chunk: meta.chunk, maxContext: meta.maxContext, options: decoderOptions)
        metadata = meta
    }

    /// The response for one request (`image == nil` = a text-only row).
    public func decide(request: SystemOneRequest, image: CGImage? = nil, grid: Grid = .g448) async throws -> JSONValue {
        try await trace(request: request, image: image, grid: grid).response
    }

    /// The whole decision with its intermediate values and times. `embeds` + `embedsGrid` replace the image and
    /// the tower with given image rows (float32 [N * 4096], N = embedsGrid.count): the gate's way to feed the
    /// oracle's rows, e.g. at a native grid no tower graph has.
    public func trace(request: SystemOneRequest, image: CGImage? = nil, grid: Grid = .g448, embeds: [Float]? = nil,
                      embedsGrid: MergedGrid? = nil) async throws -> Trace
    {
        let t0 = ContinuousClock.now
        var seconds: [String: Double] = [:]
        var prepared: ImagePreprocess.Prepared? = nil
        var rows: [Float]? = nil
        var merged: MergedGrid? = nil
        if let embeds {
            guard let g = embedsGrid else { throw ClefFlashError.contract("embeds without a grid") }
            rows = embeds
            merged = g
        } else if let image {
            guard let tower = towers[grid] else { throw ClefFlashError.bundle("no \(grid) tower loaded") }
            let p = try ImagePreprocess.prepare(image, grid: grid.side)
            seconds["decode_rgb"] = p.decodeSeconds
            seconds["resize"] = p.resizeSeconds
            seconds["patches"] = p.patchSeconds
            let t = ContinuousClock.now
            rows = try await tower.encode(patches: p.patches)
            seconds["tower"] = secondsSince(t)
            prepared = p
            merged = grid.merged
        }
        let t1 = ContinuousClock.now
        let row = try prompt.build(request, grid: merged)
        seconds["tokenize"] = secondsSince(t1)
        let t2 = ContinuousClock.now
        let inputs = try decoder.staticInputs(embeds: rows, row: row)
        seconds["static_inputs"] = secondsSince(t2)
        let pass = try await decoder.run(ids: row.decoderIDs, inputs: inputs)
        seconds["decoder"] = pass.seconds
        seconds["state_reset"] = pass.resetSeconds
        let t3 = ContinuousClock.now
        let arrays = try head.arrays(hidden16: pass.hidden, row: row, table: table)
        seconds["head_arrays"] = secondsSince(t3)
        let t4 = ContinuousClock.now
        let logits = try await head.run(arrays)
        seconds["head"] = secondsSince(t4)
        let t5 = ContinuousClock.now
        let probs = ClefHead.questionProbs(logits, layout: arrays.layout)
        seconds["softmax"] = secondsSince(t5)
        let t6 = ContinuousClock.now
        let perQuestion = zip(row.questions, probs).map { q, p in Array(zip(q.optionIDs, p)).map { (id: $0.0, p: $0.1) } }
        let response = try SystemOneResponse.make(request, probabilities: perQuestion, tokens: row.ids.count)
        seconds["response"] = secondsSince(t6)
        seconds["wall"] = secondsSince(t0)
        return Trace(row: row, prepared: prepared, imageRows: rows, pass: pass, bucket: arrays.bucket,
                     logits: Array(logits[0..<arrays.options]), probabilities: probs, response: response, seconds: seconds)
    }
}
