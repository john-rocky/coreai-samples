// ImageModelEngine — what DiffusionEngine needs from a model family: load a bundle folder, turn a
// request into an image, stop early. FluxEngine (below) runs FLUX.2 klein on apple/coreai-models'
// own pipeline; RGBAImageTurboEngine runs RGBA-Image-2.1-Turbo graph by graph.

import CoreAIDiffusionPipeline
import CoreGraphics
import Foundation

/// What one press of Generate asks for.
struct ImageRequest: Sendable {
    var prompt: String
    var seed: UInt32
    /// Denoising steps: the Steps field, or the model's fixed count.
    var steps: Int
    /// FLUX only. 1.0 = one pass per step; above 1.0 = classifier-free guidance.
    var guidance: Float
    /// Square output size in pixels, for models with a Size picker.
    var size: Int
    /// Hands-off checks only: the starting noise read from a file, in place of the seed's.
    var noise: [Float]? = nil
    /// Where `noise` came from, for the report.
    var noiseSource: String? = nil
}

/// One finished image.
struct GeneratedImage: @unchecked Sendable {
    let image: CGImage
    /// True when the alpha channel is real (shown over a checkerboard, saved with the PNG).
    let hasAlpha: Bool
    /// Seconds from Generate to the end of each denoising step.
    let stepSeconds: [Double]
    let seed: UInt32
    /// The latent after the last step, exactly as the decoder read it (RGBA-Image-2.1-Turbo: [1, N, 64] fp32).
    var finalLatents: [Float]? = nil
    /// The engine's own numbers for the hands-off report.
    var details: RunReport.GenerationDetails? = nil
}

/// Stop was pressed before the last step; the unfinished image is dropped.
struct GenerationStopped: Error {
    let afterStep: Int
}

/// The folder is not a bundle of this model. `message` is shown as it is; `report` goes to the JSON.
struct WrongFolder: Error {
    let report: String
    let message: String
}

/// One family of image models. A new model is one ModelFiles entry plus one type conforming to this.
protocol ImageModelEngine: AnyObject, Sendable {
    /// Loads everything text-to-image needs from `files`, the bundle folder. `aot` (hands-off checks) is a
    /// folder of graphs compiled ahead of time for this Mac to use instead; nil = the engine's own choice.
    /// `progress` gets a short line for the status and the log while the load runs.
    @concurrent func load(files: URL, aot: URL?, progress: @escaping @Sendable (String) -> Void) async throws -> RunReport.Load
    /// Generates one image. Calls `progress.stepFinished` after each step and stops early once
    /// `cancel()` was called (throwing GenerationStopped when the image is not finished).
    @concurrent func generate(_ request: ImageRequest, progress: GenerationProgress) async throws -> GeneratedImage
    /// Asks the running `generate` to stop after the step in progress.
    func cancel()
}

/// The run in progress, so `cancel()` from the main actor reaches the engine's loop.
final class RunningGeneration: @unchecked Sendable {
    private let lock = NSLock()
    private var progress: GenerationProgress?

    func begin(_ next: GenerationProgress) { lock.withLock { progress = next } }
    func end() { lock.withLock { progress = nil } }
    func cancel() { lock.withLock { progress }?.cancel() }
}

/// FLUX.2 klein 4B on apple/coreai-models' official diffusion pipeline (`FlowTransformerPipeline`).
final class FluxEngine: ImageModelEngine, @unchecked Sendable {
    private var pipeline: FlowTransformerPipeline?
    private let running = RunningGeneration()

    @concurrent func load(files url: URL, aot: URL?, progress: @escaping @Sendable (String) -> Void) async throws -> RunReport.Load {
        pipeline = nil
        // The most common wrong pick gets a sentence of its own instead of the runtime's error.
        guard FileManager.default.fileExists(atPath: url.appending(path: "metadata.json").path) else {
            throw WrongFolder(
                report: "no metadata.json in \(url.path)",
                message: "“\(url.lastPathComponent)” is not a Core AI diffusion bundle: it has no metadata.json. Choose the folder that holds metadata.json, tokenizer/ and the .aimodel folders.")
        }
        let start = ContinuousClock.now
        // Reads metadata.json and the tokenizer and picks the components for the best
        // resolution the folder has (1024×1024 when Transformer and VAEDecoder are there).
        let built = try await FlowTransformerPipeline(from: url, mode: .auto)
        let initSeconds = start.duration(to: .now).seconds
        // Loads every model text-to-image runs. On the first launch Core AI also specializes
        // each one for this Mac and caches the result; later launches read the cache.
        try await built.loadResources()
        let seconds = start.duration(to: .now).seconds
        let size = built.defaultImageSize
        pipeline = built
        return RunReport.Load(folder: url.path, init_seconds: initSeconds, load_resources_seconds: seconds - initSeconds,
                              total_seconds: seconds, mode: built.mode.description,
                              image_size: "\(size.width)x\(size.height)", after_download: false)
    }

    @concurrent func generate(_ request: ImageRequest, progress: GenerationProgress) async throws -> GeneratedImage {
        guard let pipeline else { throw ImageGenError("No model is loaded.") }
        running.begin(progress)
        defer { running.end() }
        // FLUX.2 klein is guidance-distilled: one pass per step, and the model ignores the
        // guidance value. Above 1.0 the pipeline adds a second, unconditional pass per step and
        // mixes the two (classifier-free guidance).
        let configuration = PipelineConfiguration(
            prompt: request.prompt,
            seed: request.seed,
            stepCount: request.steps,
            guidanceScale: request.guidance,
            guidanceMode: request.guidance > 1 ? .manual : .distilled,
            // Keep the models loaded between images.
            lazyModelLoading: false)
        let result = try await pipeline.generateImages(configuration: configuration) { @Sendable update in
            progress.stepFinished(update.step, of: update.totalSteps)
            return !progress.isCancelled
        }
        if progress.isCancelled, progress.steps.count < request.steps {
            // The pipeline stops after the step it is on and still decodes; that
            // half-finished image is dropped. (Stop during the decode keeps the finished image.)
            throw GenerationStopped(afterStep: progress.steps.count)
        }
        guard let cgImage = result.images.first else { throw ImageGenError("The pipeline returned no image.") }
        return GeneratedImage(image: cgImage, hasAlpha: false, stepSeconds: progress.steps, seed: request.seed)
    }

    func cancel() { running.cancel() }
}
