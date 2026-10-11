// RGBAImageTurboEngine — RGBA-Image-2.1-Turbo (mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI, a Core AI port
// of Qwen-Image-2.1-Turbo) as three Core AI graphs and the host code around them:
// prompt → tokenizer → text encoder → 8 DiT steps → VAE decoder → an RGBA image with a real alpha
// channel. The host steps follow the bundle's host/host_contract.md; "§n" below is its section n.

import CoreAI
import CoreAIDiffusionPipeline
import CoreGraphics
import Foundation
import Tokenizers

final class RGBAImageTurboEngine: ImageModelEngine, @unchecked Sendable {
    static let encoderName = "qi21_encoder_dynL_w16a32_ids_iofp32"
    static let sizes = [256, 512, 1024]
    static func decoderName(_ size: Int) -> String { "qi21_vae_\(size)_fp32" }

    /// §2: the text-to-image chat template. The system part's tokens are dropped after the encoder.
    static let systemPart = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    static func template(_ prompt: String) -> String {
        // An empty prompt becomes one space, as the reference pipeline does.
        systemPart + "<|im_start|>user\n\(prompt.isEmpty ? " " : prompt)<|im_end|>\n<|im_start|>assistant\n"
    }
    static let ditInputs = ["img_tokens", "txt_feats", "timestep", "txt_cos", "txt_sin", "img_cos", "img_sin"]
    static let channels = 64

    /// The DiT bundle this entry runs (bf16 or int8 weights; one interface).
    let ditName: String

    private var tokenizer: (any Tokenizer)?
    private var systemTokens: [Int] = []
    private var rope: RopeTables?
    private var schedule: TurboSchedule?
    private var encoder: CoreAIGraph?
    private var dit: CoreAIGraph?
    private var decoders: [Int: CoreAIGraph] = [:]
    private let running = RunningGeneration()

    init(dit: String) {
        ditName = dit
    }

    // MARK: - Load

    @concurrent func load(files: URL, aot: URL?, progress: @escaping @Sendable (String) -> Void) async throws -> RunReport.Load {
        let manager = FileManager.default
        let host = files.appending(path: "host"), tokenizerFolder = files.appending(path: "tokenizer")
        guard manager.fileExists(atPath: host.appending(path: "scheduler.json").path),
              manager.fileExists(atPath: tokenizerFolder.appending(path: "tokenizer.json").path)
        else {
            throw WrongFolder(
                report: "no host/scheduler.json or tokenizer/tokenizer.json in \(files.path)",
                message: "“\(files.lastPathComponent)” is not an RGBA-Image-2.1-Turbo bundle. Choose the folder that holds host/, tokenizer/ and the .aimodel folders.")
        }
        // A graph comes compiled for this Mac from `aot` (hands-off checks, which compile any missing one
        // into it) or from the bundle's own aot/ folder when it has one there (the model card's step 2).
        // Every other graph opens from its .aimodel, and Core AI compiles it just in time.
        let names = [Self.encoderName, ditName] + Self.sizes.map(Self.decoderName)
        let compiledFolder = aot ?? files.appending(path: "aot")
        let missing = names.filter {
            !manager.fileExists(atPath: files.appending(path: "\($0).aimodel").path)
                && AheadOfTimeCompiler.compiled($0, in: compiledFolder) == nil
        }
        guard missing.isEmpty else {
            throw WrongFolder(
                report: "missing \(missing.joined(separator: ",")) in \(files.path)",
                message: "“\(files.lastPathComponent)” is missing \(missing.map { "\($0).aimodel" }.joined(separator: ", ")). Choose a folder that holds all five .aimodel folders, or press Download & Load to get them.")
        }
        let start = ContinuousClock.now
        // §2: swift-transformers reads the bundle's tokenizer.json (Qwen2 byte-level BPE).
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
        let systemTokens = tokenizer.encode(text: Self.systemPart, addSpecialTokens: false)
        let rope = try RopeTables(folder: host)
        let schedule = try TurboSchedule(file: host.appending(path: "scheduler.json"))
        let initSeconds = start.duration(to: .now).seconds
        progress("tokenizer drop_idx=\(systemTokens.count) sigmas=\(schedule.sigmas) seconds=\(initSeconds)")

        // The text encoder, the DiT and the three VAE decoders.
        var graphs: [String: CoreAIGraph] = [:]
        var bundles: [RunReport.BundleLoad] = []
        for (index, name) in names.enumerated() {
            let source = files.appending(path: "\(name).aimodel")
            var url = source
            var compileSeconds: Double?
            if let found = AheadOfTimeCompiler.compiled(name, in: compiledFolder) {
                url = found
            } else if let aot {
                progress("compile \(index + 1)/\(names.count) name=\(name)")
                let result = try await AheadOfTimeCompiler.compile(source, name: name, into: aot)
                url = result.url
                compileSeconds = result.seconds
                progress("compiled name=\(name) seconds=\(result.seconds)")
            }
            progress("open \(index + 1)/\(names.count) name=\(name) file=\(url.path)")
            let graph = try await CoreAIGraph.load(url, name: name, log: progress)
            graphs[name] = graph
            bundles.append(.init(name: name, file: url.path, options: graph.options, load_seconds: graph.seconds,
                                 compile_seconds: compileSeconds))
            progress("opened name=\(name) options=\(graph.options) seconds=\(graph.seconds)")
        }
        let encoder = graphs[Self.encoderName]!, dit = graphs[ditName]!
        try encoder.expect(inputs: ["input_ids"], output: "hidden")
        try dit.expect(inputs: Self.ditInputs, output: "vel")
        var decoders: [Int: CoreAIGraph] = [:]
        for size in Self.sizes {
            let decoder = graphs[Self.decoderName(size)]!
            try decoder.expect(inputs: ["latents_packed"], output: "image")
            decoders[size] = decoder
        }
        self.tokenizer = tokenizer
        self.systemTokens = systemTokens
        self.rope = rope
        self.schedule = schedule
        self.encoder = encoder
        self.dit = dit
        self.decoders = decoders
        let seconds = start.duration(to: .now).seconds
        let mode = Set(bundles.map(\.options)).sorted().joined(separator: ",")
        return RunReport.Load(folder: files.path, init_seconds: initSeconds, load_resources_seconds: seconds - initSeconds,
                              total_seconds: seconds, mode: "options \(mode)",
                              image_size: Self.sizes.map { "\($0)x\($0)" }.joined(separator: " "),
                              after_download: false, bundles: bundles, drop_idx: systemTokens.count)
    }

    // MARK: - Generate

    @concurrent func generate(_ request: ImageRequest, progress: GenerationProgress) async throws -> GeneratedImage {
        guard let tokenizer, let rope, let schedule, let encoder, let dit else { throw ImageGenError("No model is loaded.") }
        guard let decoder = decoders[request.size] else {
            throw ImageGenError("RGBA-Image-2.1-Turbo makes 256, 512 or 1024 px images, not \(request.size).")
        }
        running.begin(progress)
        defer { running.end() }
        var details = RunReport.GenerationDetails(size: request.size)

        // §2 Tokenize: the whole template once, no padding.
        var clock = ContinuousClock.now
        let ids = tokenizer.encode(text: Self.template(request.prompt), addSpecialTokens: false)
        let drop = systemTokens.count
        guard ids.starts(with: systemTokens) else {
            throw ImageGenError("The prompt's tokens do not start with the template's system part.")
        }
        let fullCount = ids.count, textCount = fullCount - drop
        guard (16...512).contains(fullCount), (8...512).contains(textCount) else {
            throw ImageGenError("The prompt is too long: \(fullCount) tokens with the template, at most 512.")
        }
        details.token_ids = ids
        details.drop_idx = drop
        details.text_tokens = textCount
        details.tokenize_seconds = clock.duration(to: .now).seconds
        Telemetry.line("TOKENS Lfull=\(fullCount) drop_idx=\(drop) L=\(textCount) ids=\(ids)")

        // §3 Encode, then keep the prompt's rows: hidden[:, drop_idx:].
        clock = .now
        let hidden = try await encoder.run(["input_ids": NDArray(scalars: ids.map { Int32($0) }, shape: [1, fullCount])],
                                           output: "hidden").floats()
        let width = hidden.count / fullCount
        let promptRows = Array(hidden[(drop * width)...])
        var nanCount = promptRows.reduce(0) { $0 + ($1.isNaN ? 1 : 0) }
        details.encode_seconds = clock.duration(to: .now).seconds

        // §4 Sequence and RoPE: built once per image, the same for all steps.
        let h = request.size / 16, w = h, n = h * w, c = Self.channels
        var fixed = rope.inputs(textCount: textCount, height: h, width: w)
        fixed["txt_feats"] = NDArray(scalars: promptRows, shape: [1, textCount, width])

        // §5 Noise, channel-major [1, 1, 64, h, w], packed into tokens: x[y·w + x, c] = z[c, y, x].
        let noise: [Float]
        if let override = request.noise {
            guard override.count == c * n else {
                throw ImageGenError("The noise file holds \(override.count) values; a \(request.size) px image needs \(c * n).")
            }
            noise = override
            details.noise = request.noiseSource ?? "file"
        } else {
            // apple/coreai-models' port of torch.randn on a CPU generator seeded with the seed.
            var random = TorchRandomSource(seed: request.seed)
            noise = random.normalArray([1, 1, c, h, w])
            details.noise = "seed \(request.seed) (TorchRandomSource)"
        }
        var x = [Float](repeating: 0, count: n * c)
        for token in 0..<n {
            for channel in 0..<c { x[token * c + channel] = noise[channel * n + token] }
        }

        // §6 Sampler: FlowMatch Euler on the checkpoint's own sigmas, no CFG.
        let steps = schedule.timesteps.count
        var stepSeconds: [Double] = []
        for i in 0..<steps {
            clock = .now
            var inputs = fixed
            inputs["img_tokens"] = NDArray(scalars: x, shape: [1, n, c])
            inputs["timestep"] = NDArray(scalars: [schedule.timesteps[i] / 1000], shape: [1])
            let velocity = try await dit.run(inputs, output: "vel").floats()
            guard velocity.count == x.count else {
                throw ImageGenError("The DiT returned \(velocity.count) values, expected \(x.count).")
            }
            let dt = schedule.sigmas[i + 1] - schedule.sigmas[i]
            for k in x.indices {
                if velocity[k].isNaN { nanCount += 1 }
                x[k] = x[k] + dt * velocity[k]
            }
            stepSeconds.append(clock.duration(to: .now).seconds)
            progress.stepFinished(i + 1, of: steps)
            if progress.isCancelled, i + 1 < steps { throw GenerationStopped(afterStep: i + 1) }
        }
        details.dit_step_seconds = stepSeconds

        // §7 Decode: the sampler's latent as it is; the graph unpacks it and undoes the latent scaling.
        clock = .now
        let pixels = try await decoder.run(["latents_packed": NDArray(scalars: x, shape: [1, n, c])],
                                           output: "image").floats()
        let side = request.size
        guard pixels.count == 4 * side * side else {
            throw ImageGenError("The decoder returned \(pixels.count) values, expected \(4 * side * side).")
        }
        nanCount += pixels.reduce(0) { $0 + ($1.isNaN ? 1 : 0) }
        let image = try RGBA8.image(fromPlanes: pixels, side: side)
        details.decode_seconds = clock.duration(to: .now).seconds
        details.nan_values = nanCount
        details.graphs = [encoder.url.lastPathComponent, dit.url.lastPathComponent, decoder.url.lastPathComponent]
        return GeneratedImage(image: image, hasAlpha: true, stepSeconds: progress.steps, seed: request.seed,
                              finalLatents: x, details: details)
    }

    func cancel() { running.cancel() }
}

/// §4: host/rope_axis{0,1,2}_{cos,sin}.f32, one row per position −1024..8191, 8 + 28 + 28 values wide.
struct RopeTables: Sendable {
    static let widths = [8, 28, 28]
    static let firstPosition = -1024
    static let rows = 9216
    let cos: [[Float]]
    let sin: [[Float]]

    init(folder: URL) throws {
        cos = try (0..<3).map { try Float32File.read(folder.appending(path: "rope_axis\($0)_cos.f32"), count: Self.rows * Self.widths[$0]) }
        sin = try (0..<3).map { try Float32File.read(folder.appending(path: "rope_axis\($0)_sin.f32"), count: Self.rows * Self.widths[$0]) }
    }

    /// txt_cos / txt_sin [1, L, 64] and img_cos / img_sin [1, N, 64]. Text token i sits at (i, i, i);
    /// image token (y, x) at (L, y − (h − h/2), x − (w − w/2)), the grid centred on zero.
    func inputs(textCount: Int, height h: Int, width w: Int) -> [String: NDArray] {
        var positions: [(Int, Int, Int)] = (0..<textCount).map { ($0, $0, $0) }
        for y in 0..<h {
            for x in 0..<w { positions.append((textCount, y - (h - h / 2), x - (w - w / 2))) }
        }
        func rows(_ table: [[Float]], _ range: Range<Int>) -> [Float] {
            var out: [Float] = []
            out.reserveCapacity(range.count * 64)
            for (f, hp, wp) in positions[range] {
                for (axis, p) in [f, hp, wp].enumerated() {
                    let width = Self.widths[axis], start = (p - Self.firstPosition) * width
                    out.append(contentsOf: table[axis][start ..< start + width])
                }
            }
            return out
        }
        let text = 0..<textCount, image = textCount..<positions.count
        return [
            "txt_cos": NDArray(scalars: rows(cos, text), shape: [1, textCount, 64]),
            "txt_sin": NDArray(scalars: rows(sin, text), shape: [1, textCount, 64]),
            "img_cos": NDArray(scalars: rows(cos, image), shape: [1, image.count, 64]),
            "img_sin": NDArray(scalars: rows(sin, image), shape: [1, image.count, 64]),
        ]
    }
}

/// A file of raw little-endian float32 values (the host tables, a noise file).
enum Float32File {
    /// The values of `url`, which must hold exactly `count` of them.
    static func read(_ url: URL, count: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count == count * MemoryLayout<Float>.size else {
            throw ImageGenError("\(url.lastPathComponent) has \(data.count) bytes, expected \(count * MemoryLayout<Float>.size).")
        }
        return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            initialized = data.copyBytes(to: buffer) / MemoryLayout<Float>.size
        }
    }
}

/// §6: host/scheduler.json — the checkpoint's sigma grid, shifted (shift 1.0 leaves it unchanged),
/// then a final 0. All values fp32, as the reference scheduler computes them.
struct TurboSchedule: Sendable {
    /// One more than the steps, ending in 0.
    let sigmas: [Float]
    /// sigma · num_train_timesteps; the DiT's `timestep` input is timesteps[i] / 1000.
    let timesteps: [Float]

    init(file: URL) throws {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any],
              let grid = json["sample_sigmas"] as? [Double], !grid.isEmpty
        else { throw ImageGenError("\(file.lastPathComponent) has no sample_sigmas.") }
        guard (json["use_dynamic_shifting"] as? Bool) != true else {
            throw ImageGenError("\(file.lastPathComponent) asks for dynamic shifting, which this engine does not do.")
        }
        let shift = Float(json["shift"] as? Double ?? 1)
        var s = grid.map { Float($0) }
        s = s.map { shift * $0 / (1 + (shift - 1) * $0) }
        if let terminal = json["shift_terminal"] as? Double {
            let scale = (1 - s[s.count - 1]) / (1 - Float(terminal))
            s = s.map { 1 - (1 - $0) / scale }
        }
        let trainSteps = Float(json["num_train_timesteps"] as? Double ?? 1000)
        timesteps = s.map { $0 * trainSteps }
        sigmas = s + [0]
    }
}

/// §7: [1, 4, S, S] in [-1, 1] → 8-bit RGBA, round(clip(x · 0.5 + 0.5, 0, 1) · 255) with ties to even
/// (numpy's round, what the reference saves), straight (not premultiplied) alpha.
enum RGBA8 {
    static func bytes(fromPlanes planes: [Float], side: Int) -> [UInt8] {
        let count = side * side
        var out = [UInt8](repeating: 0, count: count * 4)
        for p in 0..<count {
            for channel in 0..<4 {
                var v = planes[channel * count + p] * 0.5 + 0.5
                v = v.isNaN ? 0 : min(max(v, 0), 1)
                out[p * 4 + channel] = UInt8((v * 255).rounded(.toNearestOrEven))
            }
        }
        return out
    }

    static func image(fromPlanes planes: [Float], side: Int) throws -> CGImage {
        try image(bytes(fromPlanes: planes, side: side), side: side, alpha: true)
    }

    static func image(_ bytes: [UInt8], side: Int, alpha: Bool) throws -> CGImage {
        let info = CGBitmapInfo(rawValue: (alpha ? CGImageAlphaInfo.last : CGImageAlphaInfo.noneSkipLast).rawValue)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: side * 4,
                                  space: space, bitmapInfo: info, provider: provider, decode: nil,
                                  shouldInterpolate: true, intent: .defaultIntent)
        else { throw ImageGenError("Could not make an image from the decoder's output.") }
        return image
    }

    /// The image over white, the way PIL's `paste(img, mask=alpha)` blends 8-bit values
    /// (what the reference writes as `_rgb.png`).
    static func overWhite(_ image: CGImage) throws -> CGImage {
        guard image.bitsPerPixel == 32, let data = image.dataProvider?.data as Data? else {
            throw ImageGenError("The image is not 8-bit RGBA.")
        }
        let side = image.width, rowBytes = image.bytesPerRow
        var out = [UInt8](repeating: 255, count: side * image.height * 4)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for y in 0..<image.height {
                for x in 0..<side {
                    let source = y * rowBytes + x * 4, target = (y * side + x) * 4
                    let a = Int(raw[source + 3])
                    for channel in 0..<3 {
                        let t = 255 * (255 - a) + Int(raw[source + channel]) * a + 128
                        out[target + channel] = UInt8(((t >> 8) + t) >> 8)
                    }
                }
            }
        }
        return try self.image(out, side: side, alpha: false)
    }
}
