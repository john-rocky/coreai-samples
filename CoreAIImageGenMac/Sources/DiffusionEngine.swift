// DiffusionEngine — downloads the FLUX.2 klein 4B Core AI bundle from the Hugging Face Hub, loads
// it with Apple's official CoreAIDiffusionPipeline (`FlowTransformerPipeline`) and turns a prompt
// into an image. Every button in ContentView calls a method here; the hands-off runs in
// Autoplay.swift call the same methods.

import CoreAIDiffusionPipeline
import CoreGraphics
import Foundation
import Hub
import ImageIO
import UniformTypeIdentifiers

/// The bundle this app downloads, pinned to one revision of the Hugging Face repo.
///
/// A community export made with apple/coreai-models' own recipe
/// (`coreai.diffusion.export flux2-klein-4b --platform macOS`): its transformer takes RoPE
/// position ids, which is what `FlowTransformerPipeline` passes at the coreai-models commit
/// pinned in project.yml.
enum ModelFiles {
    static let repo = "dushandz/FLUX.2-klein-4B-CoreAI"
    static let revision = "a7f65cdd6b2c8e2616dab3887efd0ae6ce983e44"
    static let title = "FLUX.2 klein 4B"

    /// What text-to-image at 1024×1024 needs, with each file's size at `revision`, in download
    /// order. The repo's VAE encoders (image-to-image) and half-size VAEs are left out.
    static let files: [(path: String, bytes: Int64)] = [
        ("metadata.json", 1_010),
        ("tokenizer/chat_template.jinja", 4_168),
        ("tokenizer/config.json", 375),
        ("tokenizer/tokenizer.json", 11_422_650),
        ("tokenizer/tokenizer_config.json", 375),
        ("vae_bn_mean.npy", 640),
        ("vae_bn_var.npy", 640),
        ("VAEDecoder.aimodel/main.hash", 32),
        ("VAEDecoder.aimodel/metadata.json", 395),
        ("VAEDecoder.aimodel/main.mlirb", 99_294_661),
        ("TextEncoder.aimodel/main.hash", 32),
        ("TextEncoder.aimodel/metadata.json", 396),
        ("TextEncoder.aimodel/main.mlirb", 1_752_686_429),
        ("Transformer.aimodel/main.hash", 32),
        ("Transformer.aimodel/metadata.json", 396),
        ("Transformer.aimodel/main.mlirb", 2_181_887_897),
    ]
    static let totalBytes = files.reduce(0) { $0 + $1.bytes }

    /// The files can be downloaded again, so they live in Caches.
    static let downloadBase = URL.cachesDirectory.appending(path: "CoreAIImageGenMac/huggingface")
    static var root: URL { HubApi(downloadBase: downloadBase).localRepoLocation(HubApi.Repo(id: repo)) }

    static var isOnDisk: Bool { files.allSatisfy { size(of: $0.path) == $0.bytes } }
    static var bytesOnDisk: Int64 { files.reduce(0) { $0 + min(size(of: $1.path) ?? 0, $1.bytes) } }

    private static func size(of path: String) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: root.appending(path: path).path)[.size] as? NSNumber)?.int64Value
    }

    /// Downloads the files that are not on disk yet with swift-transformers' `HubApi`, one file at
    /// a time, so a failed download keeps the files that finished and the next try fetches only
    /// the rest. `endpoint` replaces https://huggingface.co (nil = the default). `progress` gets
    /// the bytes done so far.
    static func download(endpoint: String?, progress: @escaping @Sendable (Int64) -> Void) async throws {
        // cache: nil = one copy of each file, in downloadBase (no second copy in the Hub cache).
        // useOfflineMode: false = no network gives the network's own error, not a missing-file one.
        let hub = HubApi(downloadBase: downloadBase, cache: nil, endpoint: endpoint, useOfflineMode: false)
        var done: Int64 = 0
        for file in files {
            if size(of: file.path) != file.bytes {
                let before = done
                try await hub.snapshot(from: HubApi.Repo(id: repo), revision: revision, matching: [file.path]) { @Sendable fraction in
                    progress(before + Int64(fraction.fractionCompleted * Double(file.bytes)))
                }
                try Task.checkCancellation()
                guard size(of: file.path) == file.bytes else {
                    throw ImageGenError("\(file.path) did not download completely.")
                }
            }
            done += file.bytes
            progress(done)
        }
    }
}

struct ImageGenError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

enum PNG {
    static func write(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw ImageGenError("Could not create \(url.lastPathComponent).") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw ImageGenError("Could not write \(url.lastPathComponent).") }
    }
}

/// Thread-safe flag and step clock, written from the pipeline's progress callback (off the main actor).
final class GenerationProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var stepSeconds: [Double] = []
    private let start = ContinuousClock.now

    var isCancelled: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }

    /// Seconds from Generate to the end of each denoising step.
    var steps: [Double] { lock.withLock { stepSeconds } }
    func stepFinished() { lock.withLock { stepSeconds.append(start.duration(to: .now).seconds) } }
    var elapsed: Double { start.duration(to: .now).seconds }
}

extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}

@MainActor
final class DiffusionEngine: ObservableObject {
    enum Status: Equatable {
        case idle
        case downloading(done: Int64, total: Int64)
        case loading
        case ready
        case generating(step: Int, total: Int)
        case stopping
        case failed(String)

        var isBusy: Bool {
            switch self {
            case .downloading, .loading, .generating, .stopping: true
            default: false
            }
        }

        var logName: String {
            switch self {
            case .idle: "idle"
            case .downloading: "downloading"
            case .loading: "loading"
            case .ready: "ready"
            case .generating(let step, let total): "generating step=\(step)/\(total)"
            case .stopping: "stopping"
            case .failed(let message): "failed message=\(message)"
            }
        }
    }

    @Published private(set) var status: Status = .idle
    /// What is loaded: the model's title, or the folder name for Local….
    @Published private(set) var modelName: String?
    /// The bundle folder that is loaded.
    @Published private(set) var modelFolder: URL?
    @Published private(set) var image: CGImage?
    @Published private(set) var imageCaption = ""
    /// A sentence about the last press of Generate that did not give an image.
    @Published private(set) var notice: String?

    // What the window edits.
    @Published var prompt = "" { didSet { notice = nil } }
    @Published var steps = 4
    @Published var guidance: Float = 1.0
    @Published var seedText = "42" { didSet { notice = nil } }

    /// Every measured number of this launch; Autoplay writes it out as JSON.
    var report = RunReport()
    private(set) var lastSeed: UInt32 = 0

    private var pipeline: FlowTransformerPipeline?
    private var work: Task<Void, Never>?
    private var progress = GenerationProgress()

    // MARK: - Download and load

    /// The Download & Load button. Skips the network when every file is already on disk.
    /// `loadAfterDownload: false` stops after the download (hands-off runs only).
    func downloadAndLoad(loadAfterDownload: Bool = true) {
        guard !status.isBusy else { return }
        clearResult()
        if ModelFiles.isOnDisk {
            Telemetry.line("DOWNLOAD skipped: all \(ModelFiles.files.count) files on disk (\(ModelFiles.totalBytes) bytes)")
            if loadAfterDownload { startLoad(ModelFiles.root, name: ModelFiles.title) }
            return
        }
        let endpoint = LaunchOptions.hubEndpoint
        let before = ModelFiles.bytesOnDisk
        let start = ContinuousClock.now
        setStatus(.downloading(done: before, total: ModelFiles.totalBytes))
        Telemetry.line("DOWNLOAD start repo=\(ModelFiles.repo) revision=\(ModelFiles.revision) endpoint=\(endpoint ?? "default") on_disk=\(before) total=\(ModelFiles.totalBytes)")
        work = Task {
            let activity = Self.beginWork("Downloading and loading the model")
            defer { ProcessInfo.processInfo.endActivity(activity) }
            do {
                try await ModelFiles.download(endpoint: endpoint) { bytes in
                    Task { @MainActor in self.showDownload(bytes) }
                }
                let seconds = start.duration(to: .now).seconds
                let onDisk = ModelFiles.bytesOnDisk
                report.download = .init(seconds: seconds, bytes_downloaded: onDisk - before, bytes_on_disk: onDisk,
                                        expected_bytes: ModelFiles.totalBytes, endpoint: endpoint ?? "default")
                Telemetry.line("DOWNLOAD done bytes_on_disk=\(onDisk) downloaded=\(onDisk - before) seconds=\(seconds)")
                if loadAfterDownload {
                    await loadPipeline(at: ModelFiles.root, name: ModelFiles.title, afterDownload: true)
                } else {
                    setStatus(.idle)
                }
            } catch {
                report.download_errors.append(error.localizedDescription)
                // swift-transformers' own message may already start with "Download failed: ".
                var detail = error.localizedDescription
                if detail.hasPrefix("Download failed: ") { detail.removeFirst("Download failed: ".count) }
                setStatus(.failed("Download failed: \(detail) Press Download & Load to try again."))
            }
        }
    }

    /// The Local… button: a bundle folder exported with `coreai.diffusion.export`.
    func loadLocal(_ url: URL) {
        guard !status.isBusy else { return }
        clearResult()
        startLoad(url, name: url.lastPathComponent)
    }

    /// Shows the busy status before the work starts, so whoever waits for "not busy" sees it at once.
    private func startLoad(_ url: URL, name: String) {
        setStatus(.loading)
        work = Task {
            let activity = Self.beginWork("Loading the model")
            defer { ProcessInfo.processInfo.endActivity(activity) }
            await loadPipeline(at: url, name: name, afterDownload: false)
        }
    }

    private func loadPipeline(at url: URL, name: String, afterDownload: Bool) async {
        pipeline = nil
        modelName = nil
        modelFolder = nil
        if status != .loading { setStatus(.loading) }
        Telemetry.line("LOAD start folder=\(url.path)")
        // The most common wrong pick gets a sentence of its own instead of the runtime's error.
        guard FileManager.default.fileExists(atPath: url.appending(path: "metadata.json").path) else {
            report.load_errors.append("no metadata.json in \(url.path)")
            setStatus(.failed("“\(url.lastPathComponent)” is not a Core AI diffusion bundle: it has no metadata.json. Choose the folder that holds metadata.json, tokenizer/ and the .aimodel folders."))
            return
        }
        let start = ContinuousClock.now
        do {
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
            modelName = name
            modelFolder = url
            report.loads.append(.init(folder: url.path, init_seconds: initSeconds, load_resources_seconds: seconds - initSeconds,
                                      total_seconds: seconds, mode: built.mode.description,
                                      image_size: "\(size.width)x\(size.height)", after_download: afterDownload))
            Telemetry.line("LOAD done seconds=\(seconds) init_s=\(initSeconds) resources_s=\(seconds - initSeconds) mode=\(built.mode) size=\(size.width)x\(size.height) after_download=\(afterDownload)")
            setStatus(.ready)
        } catch {
            report.load_errors.append(String(describing: error))
            setStatus(.failed("Could not load “\(url.lastPathComponent)”: \(error.localizedDescription)"))
        }
    }

    // MARK: - Generate

    /// The Generate button.
    func generate() {
        guard case .ready = status, let pipeline else { return }
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return show(notice: "Type a prompt first, then press Generate.") }
        guard let seed = UInt32(seedText.trimmingCharacters(in: .whitespaces)) else {
            return show(notice: "The seed must be a whole number from 0 to 4294967295.")
        }
        let steps = self.steps, guidance = self.guidance
        notice = nil
        lastSeed = seed
        let progress = GenerationProgress()
        self.progress = progress
        setStatus(.generating(step: 0, total: steps))
        Telemetry.line("GENERATE start steps=\(steps) guidance=\(guidance) seed=\(seed) prompt=\(text)")

        // FLUX.2 klein is guidance-distilled: one pass per step, and the model ignores the
        // guidance value. Above 1.0 the pipeline adds a second, unconditional pass per step and
        // mixes the two (classifier-free guidance).
        let configuration = PipelineConfiguration(
            prompt: text,
            seed: seed,
            stepCount: steps,
            guidanceScale: guidance,
            guidanceMode: guidance > 1 ? .manual : .distilled,
            // Keep the models loaded between images.
            lazyModelLoading: false)

        work = Task {
            let activity = Self.beginWork("Generating an image")
            defer { ProcessInfo.processInfo.endActivity(activity) }
            var record = RunReport.Generation(prompt: text, steps: steps, guidance: guidance, seed: seed)
            do {
                let result = try await pipeline.generateImages(configuration: configuration) { @Sendable update in
                    progress.stepFinished()
                    let step = update.step, total = update.totalSteps
                    Task { @MainActor in self.showStep(step, of: total) }
                    return !progress.isCancelled
                }
                record.seconds = progress.elapsed
                record.step_seconds = progress.steps
                if progress.isCancelled, progress.steps.count < steps {
                    // The pipeline stops after the step it is on and still decodes; that
                    // half-finished image is dropped. (Stop during the decode keeps the finished image.)
                    record.stopped_after_step = progress.steps.count
                    report.generations.append(record)
                    show(notice: "Stopped after step \(progress.steps.count) of \(steps). The unfinished image was not kept.")
                    setStatus(.ready)
                    return
                }
                guard let cgImage = result.images.first else { throw ImageGenError("The pipeline returned no image.") }
                record.width = cgImage.width
                record.height = cgImage.height
                report.generations.append(record)
                image = cgImage
                imageCaption = "\(cgImage.width)×\(cgImage.height) · \(steps) steps · seed \(seed) · \(String(format: "%.1f", record.seconds)) s"
                Telemetry.line("GENERATE done seconds=\(record.seconds) steps_s=\(record.step_seconds) size=\(cgImage.width)x\(cgImage.height)")
                setStatus(.ready)
            } catch {
                record.error = String(describing: error)
                report.generations.append(record)
                show(notice: "Generation failed: \(error.localizedDescription)")
                setStatus(.ready)
            }
        }
    }

    /// The Stop button. The pipeline finishes the step it is on, then returns.
    func stop() {
        guard case .generating = status else { return }
        progress.cancel()
        setStatus(.stopping)
    }

    // MARK: - Helpers

    /// Marks long work as user-initiated. Without it macOS throttles the app (App Nap) once its window
    /// is hidden: the download dropped to about 60 kB/s while curl got 6 MB/s on the same link.
    private static func beginWork(_ reason: String) -> NSObjectProtocol {
        ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: reason)
    }

    func show(notice text: String) {
        notice = text
        report.notices.append(text)
        Telemetry.line("NOTICE \(text)")
    }

    private func clearResult() {
        image = nil
        imageCaption = ""
        notice = nil
    }

    private func showDownload(_ bytes: Int64) {
        guard case .downloading(let done, let total) = status, bytes > done else { return }
        if bytes * 20 / max(total, 1) != done * 20 / max(total, 1) {   // a log line every 5 %
            Telemetry.line("DOWNLOAD progress bytes=\(bytes) percent=\(bytes * 100 / max(total, 1))")
        }
        status = .downloading(done: bytes, total: total)
    }

    private func showStep(_ step: Int, of total: Int) {
        guard case .generating = status else { return }
        setStatus(.generating(step: step, total: total))
    }

    private func setStatus(_ next: Status) {
        status = next
        Telemetry.line("PHASE \(next.logName)")
    }
}
