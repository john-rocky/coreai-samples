// DiffusionEngine — the app's state: which model is picked, its download and load, and Generate /
// Stop. The work itself goes to the picked model's ImageModelEngine (FluxEngine or
// RGBAImageTurboEngine). Every button in ContentView calls a method here; the hands-off runs in
// Autoplay.swift call the same methods.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

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

/// Thread-safe flag and step clock, written by the engine during a run (off the main actor).
final class GenerationProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var stepSeconds: [Double] = []
    private let start = ContinuousClock.now
    private let onStep: @Sendable (Int, Int) -> Void

    /// `onStep` gets (step, total) after each step.
    init(onStep: @escaping @Sendable (Int, Int) -> Void) {
        self.onStep = onStep
    }

    var isCancelled: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }

    /// Seconds from Generate to the end of each denoising step.
    var steps: [Double] { lock.withLock { stepSeconds } }
    func stepFinished(_ step: Int, of total: Int) {
        lock.withLock { stepSeconds.append(start.duration(to: .now).seconds) }
        onStep(step, total)
    }
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
    /// The bundle Download & Load fetches. Picking another one drops the loaded model.
    @Published var selectedID = ModelFiles.catalog[0].id {
        didSet {
            guard selectedID != oldValue else { return }
            unload()
            if let size = selected.defaultSize { self.size = size }
        }
    }
    var selected: ModelFiles { ModelFiles.catalog.first { $0.id == selectedID } ?? ModelFiles.catalog[0] }
    /// What is loaded: the model's title, or the folder name for Local….
    @Published private(set) var modelName: String?
    /// The bundle folder that is loaded.
    @Published private(set) var modelFolder: URL?
    /// The step of a load in progress that takes long (compiling for this Mac, opening a graph).
    @Published private(set) var loadDetail: String?
    @Published private(set) var image: CGImage?
    /// True when `image` has a real alpha channel: it is shown over a checkerboard and saved with it.
    @Published private(set) var imageHasAlpha = false
    @Published private(set) var imageCaption = ""
    /// A sentence about the last press of Generate that did not give an image.
    @Published private(set) var notice: String?

    // What the window edits.
    @Published var prompt = "" { didSet { notice = nil } }
    @Published var steps = 4
    @Published var guidance: Float = 1.0
    /// The square size of the next image, for models with a Size picker.
    @Published var size = 1024
    @Published var seedText = "42" { didSet { notice = nil } }

    /// Every measured number of this launch; Autoplay writes it out as JSON.
    var report = RunReport()
    private(set) var lastSeed: UInt32 = 0
    /// Hands-off checks only: the starting noise of the next images, read from a file (`-noiseFile`).
    var noiseOverride: (values: [Float], source: String)?
    /// The last image's final latent (hands-off runs write it next to the PNG).
    private(set) var lastLatents: [Float]?

    private var engine: (any ImageModelEngine)?
    private var work: Task<Void, Never>?
    /// Presses of Download & Load in this launch that went to the network.
    private var downloadTries = 0

    // MARK: - Download and load

    /// The Download & Load button. Skips the network when every file is already on disk.
    /// `loadAfterDownload: false` stops after the download (hands-off runs only).
    func downloadAndLoad(loadAfterDownload: Bool = true) {
        guard !status.isBusy else { return }
        clearResult()
        let model = selected
        report.hub = .init(id: model.id, repo: model.repo, revision: model.revision, folder: model.folder)
        if model.isOnDisk {
            Telemetry.line("DOWNLOAD skipped: all \(model.files.count) files of \(model.id) on disk (\(model.totalBytes) bytes)")
            if loadAfterDownload { startLoad(model.root, name: model.title) }
            return
        }
        downloadTries += 1
        let endpoint = (downloadTries == 1 ? LaunchOptions.hubEndpointFirstTry : nil) ?? LaunchOptions.hubEndpoint
        let before = model.bytesOnDisk
        // The files and the cache Core AI writes on the first load both need room: say so before the
        // download starts instead of failing partway through it.
        let files = model.totalBytes - before, needed = files + model.firstLoadCacheBytes
        let free = LaunchOptions.freeBytes ?? Self.freeDiskBytes()
        report.disk = .init(free_bytes: free, needed_bytes: needed, files_bytes: files, cache_bytes: model.firstLoadCacheBytes)
        Telemetry.line("DISK free=\(free.map(String.init) ?? "unknown") needed=\(needed) files=\(files) cache=\(model.firstLoadCacheBytes)")
        if let free, free < needed {
            report.download_errors.append("free disk \(free) bytes < needed \(needed) bytes")
            setStatus(.failed("Not enough free disk space. Download & Load needs \(Self.gigabytes(needed)) GB: \(Self.gigabytes(files)) GB of files, and \(Self.gigabytes(model.firstLoadCacheBytes)) GB that Core AI writes to its cache on the first load. \(Self.gigabytes(free)) GB is free."))
            return
        }
        let start = ContinuousClock.now
        setStatus(.downloading(done: before, total: model.totalBytes))
        Telemetry.line("DOWNLOAD start model=\(model.id) repo=\(model.repo) revision=\(model.revision) folder=\(model.folder) endpoint=\(endpoint ?? "default") on_disk=\(before) total=\(model.totalBytes)")
        work = Task {
            let activity = Self.beginWork("Downloading and loading the model")
            defer { ProcessInfo.processInfo.endActivity(activity) }
            do {
                try await model.download(endpoint: endpoint) { bytes in
                    Task { @MainActor in self.showDownload(bytes) }
                }
                let seconds = start.duration(to: .now).seconds
                let onDisk = model.bytesOnDisk
                report.download = .init(seconds: seconds, bytes_downloaded: onDisk - before, bytes_on_disk: onDisk,
                                        expected_bytes: model.totalBytes, endpoint: endpoint ?? "default")
                Telemetry.line("DOWNLOAD done bytes_on_disk=\(onDisk) downloaded=\(onDisk - before) seconds=\(seconds)")
                if loadAfterDownload {
                    await loadModel(at: model.root, name: model.title, afterDownload: true)
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

    /// The Local… button: a bundle folder of the picked model (for FLUX, one exported with
    /// `coreai.diffusion.export`).
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
            await loadModel(at: url, name: name, afterDownload: false)
        }
    }

    private func loadModel(at url: URL, name: String, afterDownload: Bool) async {
        engine = nil
        modelName = nil
        modelFolder = nil
        loadDetail = nil
        if status != .loading { setStatus(.loading) }
        Telemetry.line("LOAD start folder=\(url.path)")
        let model = selected
        let next = model.makeEngine()
        let aot = LaunchOptions.aot.map { URL(filePath: ($0 as NSString).expandingTildeInPath) }
        do {
            var load = try await next.load(files: url, aot: aot) { line in
                Telemetry.line("LOAD \(line)")
                Task { @MainActor in self.showLoad(line) }
            }
            load.after_download = afterDownload
            engine = next
            modelName = name
            modelFolder = url
            loadDetail = nil
            report.loads.append(load)
            Telemetry.line("LOAD done seconds=\(load.total_seconds) init_s=\(load.init_seconds) resources_s=\(load.load_resources_seconds) mode=\(load.mode) size=\(load.image_size) after_download=\(afterDownload)")
            setStatus(.ready)
        } catch let wrong as WrongFolder {
            loadDetail = nil
            report.load_errors.append(wrong.report)
            setStatus(.failed(wrong.message))
        } catch {
            loadDetail = nil
            report.load_errors.append(String(describing: error))
            setStatus(.failed("Could not load “\(url.lastPathComponent)”: \(error.localizedDescription)"))
        }
    }

    // MARK: - Generate

    /// The Generate button.
    func generate() {
        guard case .ready = status, let engine else { return }
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return show(notice: "Type a prompt first, then press Generate.") }
        guard let seed = UInt32(seedText.trimmingCharacters(in: .whitespaces)) else {
            return show(notice: "The seed must be a whole number from 0 to 4294967295.")
        }
        let model = selected
        let steps = model.fixedSteps ?? self.steps
        let guidance = model.usesGuidance ? self.guidance : 1
        let size = self.size
        notice = nil
        lastSeed = seed
        let progress = GenerationProgress { step, total in
            Task { @MainActor in self.showStep(step, of: total) }
        }
        setStatus(.generating(step: 0, total: steps))
        if model.sizes.isEmpty {
            Telemetry.line("GENERATE start steps=\(steps) guidance=\(guidance) seed=\(seed) prompt=\(text)")
        } else {
            Telemetry.line("GENERATE start steps=\(steps) size=\(size) seed=\(seed) noise=\(noiseOverride?.source ?? "seed") prompt=\(text)")
        }
        let request = ImageRequest(prompt: text, seed: seed, steps: steps, guidance: guidance, size: size,
                                   noise: noiseOverride?.values, noiseSource: noiseOverride?.source)

        work = Task {
            let activity = Self.beginWork("Generating an image")
            defer { ProcessInfo.processInfo.endActivity(activity) }
            var record = RunReport.Generation(prompt: text, steps: steps, guidance: guidance, seed: seed)
            do {
                let result = try await engine.generate(request, progress: progress)
                record.seconds = progress.elapsed
                record.step_seconds = progress.steps
                record.width = result.image.width
                record.height = result.image.height
                record.details = result.details
                report.generations.append(record)
                image = result.image
                imageHasAlpha = result.hasAlpha
                lastLatents = result.finalLatents
                imageCaption = "\(result.image.width)×\(result.image.height) · \(steps) steps · seed \(seed) · \(String(format: "%.1f", record.seconds)) s"
                Telemetry.line("GENERATE done seconds=\(record.seconds) steps_s=\(record.step_seconds) size=\(result.image.width)x\(result.image.height)")
                setStatus(.ready)
            } catch let stopped as GenerationStopped {
                record.seconds = progress.elapsed
                record.step_seconds = progress.steps
                record.stopped_after_step = stopped.afterStep
                report.generations.append(record)
                show(notice: "Stopped after step \(stopped.afterStep) of \(steps). The unfinished image was not kept.")
                setStatus(.ready)
            } catch {
                record.error = String(describing: error)
                report.generations.append(record)
                show(notice: "Generation failed: \(error.localizedDescription)")
                setStatus(.ready)
            }
        }
    }

    /// The Stop button. The engine finishes the step it is on, then returns.
    func stop() {
        guard case .generating = status else { return }
        engine?.cancel()
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
        imageHasAlpha = false
        imageCaption = ""
        notice = nil
    }

    /// Lets go of the loaded model, so Generate waits for the next Download & Load or Local….
    private func unload() {
        guard !status.isBusy else { return }
        engine = nil
        modelName = nil
        modelFolder = nil
        notice = nil
        setStatus(.idle)
    }

    private func showDownload(_ bytes: Int64) {
        guard case .downloading(let done, let total) = status, bytes > done else { return }
        if bytes * 20 / max(total, 1) != done * 20 / max(total, 1) {   // a log line every 5 %
            Telemetry.line("DOWNLOAD progress bytes=\(bytes) percent=\(bytes * 100 / max(total, 1))")
        }
        status = .downloading(done: bytes, total: total)
    }

    /// The engine's load lines that matter to someone waiting: compiling and opening each graph.
    private func showLoad(_ line: String) {
        guard status == .loading else { return }
        let words = line.split(separator: " ")
        guard let verb = words.first, let count = words.dropFirst().first, count.contains("/") else { return }
        let ofTotal = count.replacingOccurrences(of: "/", with: " of ")
        switch verb {
        case "compile": loadDetail = "Compiling graph \(ofTotal) with Xcode…"
        case "open": loadDetail = "Opening graph \(ofTotal)…"
        default: break
        }
    }

    /// Free space on the disk that holds ~/Library/Caches, counting what macOS can purge.
    private static func freeDiskBytes() -> Int64? {
        try? URL.cachesDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage
    }

    static func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.1f", Double(bytes) / 1e9)
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
