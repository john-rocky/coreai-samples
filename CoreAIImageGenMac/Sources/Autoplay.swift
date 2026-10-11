import AppKit
import CoreAI
import CryptoKit
import Darwin
import Foundation

/// Launch arguments for hands-off runs (checks and screenshots). Each one presses the same buttons
/// a person would; nothing here is a separate code path. Run the binary inside the app:
/// `CoreAIImageGenMac.app/Contents/MacOS/CoreAIImageGenMac -autoplay 1 -prompt "a red bicycle" -out ~/a.png -log 1`.
///
/// - `-autoplay 1`: press Download & Load (once more if the download fails), wait until the model is
///   ready, then generate.
/// - `-model fp16|int8|rgba-turbo-bf16|rgba-turbo-int8`: pick that model before Download & Load
///   (default: the first one).
/// - `-hubEndpoint <url>`: download from this host instead of https://huggingface.co.
/// - `-hubEndpointFirstTry <url>`: the first press of Download & Load goes to this host instead, the next one
///   to `-hubEndpoint` or Hugging Face (a closed port such as http://127.0.0.1:9 shows the failure and the retry).
/// - `-freeBytes <n>`: Download & Load acts as if the disk had n bytes free (shows the not-enough-space message).
/// - `-local <folder>`: press Local… with this folder instead of Download & Load.
/// - `-aot <folder>`: RGBA-Image-2.1-Turbo opens its graphs compiled ahead of time (`<name>.<arch>.aimodelc`)
///   from this folder, and compiles any missing one into it with `xcrun coreai-build` first (checks against
///   the Python engine, which runs that form). Without it, a graph comes from the bundle's own aot/ folder
///   when it is there, and otherwise Core AI compiles its `.aimodel` just in time.
/// - `-downloadOnly 1`: stop after the download, before the load.
/// - `-prompt "<text>"` or `-prompts <file, one prompt per line>`: type each prompt, press Generate.
/// - `-type 1` (recording): type each prompt into the Prompt field one character at a time, `-typeMs <ms>`
///   apart (default 35), instead of filling it in at once.
/// - `-trigger <file>` (recording): once the model is ready and the fields are set, wait until this file
///   exists before typing the first prompt (a recorder creates it once the capture is rolling).
/// - `-steps <n>`, `-guidance <x>`, `-seed <text>`, `-size 256|512|1024`: set the fields before Generate.
/// - `-noiseFile <file>`: RGBA-Image-2.1-Turbo starts from this noise instead of the seed's: float32,
///   channel-major [1, 1, 64, size/16, size/16], e.g. `torch.randn(...).numpy().tofile(file)`.
/// - `-stopAfterStep <n>`: press Stop once step n has finished.
/// - `-out <file.png>`: write each image there (`-2`, `-3`… before the extension for later prompts).
///   An RGBA image also gets `<name>_rgb.png` (over white) and `<name>_final_latents.f32` (the last
///   latent, [1, N, 64] float32).
/// - `-hold <s>`: wait this long before each press and after each image (default 2).
/// - `-holdAfter <s>`: wait this long after each image instead (default: the `-hold` value).
/// - `-log 1`: mirror the status lines to ~/Library/Application Support/CoreAIImageGenMac/autoplay.log;
///   at the end write result-<launch epoch>.json there with every measured number of the run.
/// - `-quit 1`: quit when the run is done.
///
/// A launch from a script can come up with no window (seen with the screen locked, 2026-10-10); adding
/// `-ApplePersistenceIgnoreState YES` brought the window back.
enum LaunchOptions {
    private static var defaults: UserDefaults { .standard }
    static var autoplay: Bool { defaults.bool(forKey: "autoplay") }
    static var log: Bool { defaults.bool(forKey: "log") }
    static var model: String? { defaults.string(forKey: "model") }
    static var hubEndpoint: String? { defaults.string(forKey: "hubEndpoint") }
    static var hubEndpointFirstTry: String? { defaults.string(forKey: "hubEndpointFirstTry") }
    static var freeBytes: Int64? { defaults.string(forKey: "freeBytes").flatMap { Int64($0) } }
    static var local: String? { defaults.string(forKey: "local") }
    static var aot: String? { defaults.string(forKey: "aot") }
    static var downloadOnly: Bool { defaults.bool(forKey: "downloadOnly") }
    static var prompt: String? { defaults.string(forKey: "prompt") }
    static var prompts: String? { defaults.string(forKey: "prompts") }
    static var type: Bool { defaults.bool(forKey: "type") }
    static var typeMs: Double { defaults.object(forKey: "typeMs") == nil ? 35 : defaults.double(forKey: "typeMs") }
    static var trigger: String? { defaults.string(forKey: "trigger") }
    static var steps: Int? { defaults.object(forKey: "steps") == nil ? nil : defaults.integer(forKey: "steps") }
    static var guidance: Float? { defaults.object(forKey: "guidance") == nil ? nil : defaults.float(forKey: "guidance") }
    static var seed: String? { defaults.string(forKey: "seed") }
    static var size: Int? { defaults.object(forKey: "size") == nil ? nil : defaults.integer(forKey: "size") }
    static var noiseFile: String? { defaults.string(forKey: "noiseFile") }
    static var stopAfterStep: Int? { defaults.object(forKey: "stopAfterStep") == nil ? nil : defaults.integer(forKey: "stopAfterStep") }
    static var out: String? { defaults.string(forKey: "out") }
    static var hold: Double { defaults.object(forKey: "hold") == nil ? 2 : defaults.double(forKey: "hold") }
    static var holdAfter: Double { defaults.object(forKey: "holdAfter") == nil ? hold : defaults.double(forKey: "holdAfter") }
    static var quit: Bool { defaults.bool(forKey: "quit") }
}

enum Telemetry {
    static let launchEpoch = Int(Date().timeIntervalSince1970)
    static let folder = URL.applicationSupportDirectory.appending(path: "CoreAIImageGenMac")

    /// With `-log 1`, stdout and stderr (this app's lines and Core AI's own messages) go to autoplay.log.
    static func begin() {
        if LaunchOptions.log {
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let url = folder.appending(path: "autoplay.log")
            let descriptor = url.path.withCString { Darwin.open($0, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR) }
            if descriptor >= 0 {
                dup2(descriptor, STDOUT_FILENO)
                dup2(descriptor, STDERR_FILENO)
                Darwin.close(descriptor)
            }
        }
        line("LAUNCH epoch=\(launchEpoch) pid=\(getpid()) model=\(RunReport.sysctl("hw.model")) os_build=\(RunReport.sysctl("kern.osversion")) arch=\(AIModel.deviceArchitectureName) thermal=\(RunReport.thermal()) args=\(ProcessInfo.processInfo.arguments.dropFirst().joined(separator: " "))")
    }

    static func line(_ text: String) {
        let time = Date().formatted(.iso8601.year().month().day().time(includingFractionalSeconds: true))
        print("IMAGEGEN \(time) \(text)")
        fflush(nil)
    }
}

/// Every measured number of one launch; written as JSON when a hands-off run ends.
struct RunReport: Codable {
    /// The Hugging Face bundle of the last Download & Load.
    struct Hub: Codable {
        var id: String
        var repo: String
        var revision: String
        var folder: String
    }
    /// The free-space check before a download.
    struct Disk: Codable {
        var free_bytes: Int64?
        var needed_bytes: Int64
        var files_bytes: Int64
        var cache_bytes: Int64
    }
    struct Download: Codable {
        var seconds: Double
        var bytes_downloaded: Int64
        var bytes_on_disk: Int64
        var expected_bytes: Int64
        var endpoint: String
    }
    struct Load: Codable {
        var folder: String
        var init_seconds: Double
        var load_resources_seconds: Double
        var total_seconds: Double
        var mode: String
        var image_size: String
        var after_download: Bool
        /// RGBA-Image-2.1-Turbo: each graph's file, options and seconds.
        var bundles: [BundleLoad]? = nil
        var drop_idx: Int? = nil
    }
    struct BundleLoad: Codable {
        var name: String
        /// The `.aimodel` (compiled just in time) or `.aimodelc` (compiled ahead of time) that was opened.
        var file: String
        /// "gpu-efr" (an `.aimodel`: GPU preferred and expectFrequentReshapes), or for an `.aimodelc`
        /// "default" (`SpecializationOptions.default`) or "gpu" (preferring the GPU, tried when default failed).
        var options: String
        var load_seconds: Double
        /// Set when the app compiled it during this load.
        var compile_seconds: Double?
    }
    struct Pixels: Codable {
        var sha256: String
        var mean_rgb: [Double]
        var min: Int
        var max: Int
        var fraction_0: Double
        var fraction_255: Double
        /// RGBA images: the alpha channel's statistics.
        var alpha: Alpha? = nil
    }
    struct Alpha: Codable {
        var min: Int
        var max: Int
        var mean: Double
        var fraction_0: Double
        var fraction_255: Double
    }
    /// What an engine that runs the graphs itself (RGBA-Image-2.1-Turbo) reports about one image.
    struct GenerationDetails: Codable {
        var size: Int?
        /// "seed N (TorchRandomSource)" or the noise file.
        var noise: String?
        var token_ids: [Int]?
        var drop_idx: Int?
        /// L, the prompt tokens the DiT reads.
        var text_tokens: Int?
        var tokenize_seconds: Double?
        var encode_seconds: Double?
        /// Each DiT step on its own (the call and the Euler update).
        var dit_step_seconds: [Double]?
        var decode_seconds: Double?
        /// NaN values in the encoder rows, the DiT outputs and the decoder output together.
        var nan_values: Int?
        /// The compiled graphs that ran.
        var graphs: [String]?
    }
    struct Generation: Codable {
        var prompt: String
        var steps: Int
        var guidance: Float
        var seed: UInt32
        /// Generate pressed → image returned (text encoding, every step, VAE decode).
        var seconds: Double = 0
        /// Seconds from Generate to the end of each denoising step.
        var step_seconds: [Double] = []
        var stopped_after_step: Int?
        var error: String?
        var width: Int?
        var height: Int?
        var png_path: String?
        var png_bytes: Int?
        var png_sha256: String?
        var pixels: Pixels?
        /// RGBA images: the copy over white and the final latent written next to the PNG.
        var rgb_png_path: String?
        var latents_path: String?
        var latents_sha256: String?
        var details: GenerationDetails?
    }

    var status = "RUNNING"
    var launch_epoch = Telemetry.launchEpoch
    var model = RunReport.sysctl("hw.model")
    var os_version = ProcessInfo.processInfo.operatingSystemVersionString
    var os_build = RunReport.sysctl("kern.osversion")
    var architecture = AIModel.deviceArchitectureName
    var physical_memory_gb = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
    var hub: Hub?
    var disk: Disk?
    var compute = "Core AI: FLUX through the pipeline's SpecializationOptions (GPU preferred); RGBA-Image-2.1-Turbo with the options in loads[].bundles; per-operation placement not observed"
    var arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
    var thermal_at_launch = RunReport.thermal()
    var thermal_at_end: String?
    var footprint_mb_at_end: Double?
    var peak_footprint_mb: Double?
    var download_errors: [String] = []
    var download: Download?
    var load_errors: [String] = []
    var loads: [Load] = []
    var notices: [String] = []
    var generations: [Generation] = []

    static func sysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "unavailable" }
        var bytes = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return "unavailable" }
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    /// The process's physical footprint now and its peak, in MB (GPU buffers count: memory is unified).
    static func footprint() -> (now: Double, peak: Double)? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return (Double(info.phys_footprint) / 1_048_576, Double(info.ledger_phys_footprint_peak) / 1_048_576)
    }

    /// SHA-256 and simple statistics of the image's own 8-bit bytes (a blank image shows as min == max).
    static func pixels(of image: CGImage) -> Pixels? {
        guard image.bitsPerComponent == 8, let data = image.dataProvider?.data as Data? else { return nil }
        let channels = image.bitsPerPixel / 8, width = image.width, height = image.height, rowBytes = image.bytesPerRow
        var sums = [Double](repeating: 0, count: 3)
        var low = 255, high = 0, zeros = 0, full = 0
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for y in 0..<height {
                for x in 0..<width {
                    let pixel = y * rowBytes + x * channels
                    for c in 0..<3 {
                        let value = Int(raw[pixel + c])
                        sums[c] += Double(value)
                        low = Swift.min(low, value)
                        high = Swift.max(high, value)
                        if value == 0 { zeros += 1 }
                        if value == 255 { full += 1 }
                    }
                }
            }
        }
        let samples = Double(width * height * 3)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        var alpha: Alpha?
        if channels == 4, image.alphaInfo == .last {
            var aLow = 255, aHigh = 0, aZeros = 0, aFull = 0, aSum = 0.0
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                for y in 0..<height {
                    for x in 0..<width {
                        let value = Int(raw[y * rowBytes + x * channels + 3])
                        aSum += Double(value)
                        aLow = Swift.min(aLow, value)
                        aHigh = Swift.max(aHigh, value)
                        if value == 0 { aZeros += 1 }
                        if value == 255 { aFull += 1 }
                    }
                }
            }
            let count = Double(width * height)
            alpha = Alpha(min: aLow, max: aHigh, mean: aSum / count, fraction_0: Double(aZeros) / count,
                          fraction_255: Double(aFull) / count)
        }
        return Pixels(sha256: digest, mean_rgb: sums.map { $0 / Double(width * height) }, min: low, max: high,
                      fraction_0: Double(zeros) / samples, fraction_255: Double(full) / samples, alpha: alpha)
    }
}

@MainActor
enum Autoplay {
    private static var started = false

    /// Runs the hands-off sequence once per launch, when `-autoplay 1` is given (a second window does not start it again).
    static func runOnce(_ engine: DiffusionEngine) async {
        guard LaunchOptions.autoplay, !started else { return }
        started = true
        await run(engine)
    }

    private static func run(_ engine: DiffusionEngine) async {
        var status = "DONE"
        do {
            try await pause()
            if let id = LaunchOptions.model {
                guard ModelFiles.catalog.contains(where: { $0.id == id }) else {
                    throw ImageGenError("-model \(id) is not one of: \(ModelFiles.catalog.map(\.id).joined(separator: ", "))")
                }
                Telemetry.line("PICK model=\(id)")
                engine.selectedID = id
                try await pause()
            }
            if let path = LaunchOptions.local {
                Telemetry.line("TAP Local… folder=\(path)")
                engine.loadLocal(URL(filePath: (path as NSString).expandingTildeInPath))
                try await until("the load", timeout: 1800) { !engine.status.isBusy }
            } else {
                Telemetry.line("TAP Download & Load")
                engine.downloadAndLoad(loadAfterDownload: !LaunchOptions.downloadOnly)
                try await until("the download and the load", timeout: 3600) { !engine.status.isBusy }
                if case .failed(let message) = engine.status, message.hasPrefix("Download failed") {
                    try await pause()
                    Telemetry.line("TAP Download & Load (again)")
                    engine.downloadAndLoad(loadAfterDownload: !LaunchOptions.downloadOnly)
                    try await until("the second try", timeout: 3600) { !engine.status.isBusy }
                }
            }
            if case .failed = engine.status {
                try await pause()
                return finish(engine, status: "FAILED")
            }
            if LaunchOptions.downloadOnly { return finish(engine, status: "DONE") }

            for (index, text) in try promptList().enumerated() {
                try await pause()
                if let steps = LaunchOptions.steps { engine.steps = steps }
                if let guidance = LaunchOptions.guidance { engine.guidance = guidance }
                if let seed = LaunchOptions.seed { engine.seedText = seed }
                if let size = LaunchOptions.size { engine.size = size }
                if let path = LaunchOptions.noiseFile {
                    let url = URL(filePath: (path as NSString).expandingTildeInPath)
                    engine.noiseOverride = (try Float32File.read(url, count: 64 * (engine.size / 16) * (engine.size / 16)), url.path)
                }
                if index == 0, let trigger = LaunchOptions.trigger {
                    let path = (trigger as NSString).expandingTildeInPath
                    Telemetry.line("WAIT trigger=\(path)")
                    try await until("the trigger file", timeout: 3600) { FileManager.default.fileExists(atPath: path) }
                }
                if LaunchOptions.type {
                    Telemetry.line("TYPING characters=\(text.count) ms_each=\(LaunchOptions.typeMs)")
                    try await typeCharacters(text, into: engine)
                } else {
                    engine.prompt = text
                }
                Telemetry.line("TYPED prompt=\(text) steps=\(engine.steps) guidance=\(engine.guidance) seed=\(engine.seedText) size=\(engine.size)")
                try await pause()
                Telemetry.line("TAP Generate")
                let before = engine.report.generations.count
                engine.generate()
                guard engine.status != .ready else {
                    Telemetry.line("NOT STARTED notice=\(engine.notice ?? "none")")
                    try await pause()
                    continue
                }
                if let stopStep = LaunchOptions.stopAfterStep {
                    try await until("step \(stopStep)", timeout: 600) {
                        if case .generating(let step, _) = engine.status { return step >= stopStep }
                        return !engine.status.isBusy
                    }
                    Telemetry.line("TAP Stop")
                    engine.stop()
                }
                try await until("the image", timeout: 600) { !engine.status.isBusy }
                guard engine.report.generations.count > before else { continue }
                let last = engine.report.generations.count - 1
                if engine.report.generations[last].width != nil, let image = engine.image {
                    record(image, hasAlpha: engine.imageHasAlpha, latents: engine.lastLatents, index: index,
                           into: &engine.report.generations[last])
                }
                try await Task.sleep(for: .seconds(LaunchOptions.holdAfter))
            }
        } catch {
            Telemetry.line("ERROR \(error)")
            status = "ERROR"
        }
        finish(engine, status: status)
    }

    private static func promptList() throws -> [String] {
        if let text = LaunchOptions.prompt { return [text] }
        guard let path = LaunchOptions.prompts else { return [] }
        return try String(contentsOf: URL(filePath: (path as NSString).expandingTildeInPath), encoding: .utf8)
            .split(whereSeparator: \.isNewline).map(String.init).filter { !$0.isEmpty }
    }

    /// Writes the image to `-out` and adds its size, hashes and pixel statistics to the record.
    private static func record(_ image: CGImage, hasAlpha: Bool, latents: [Float]?, index: Int,
                               into generation: inout RunReport.Generation) {
        generation.pixels = RunReport.pixels(of: image)
        if let out = LaunchOptions.out {
            var url = URL(filePath: (out as NSString).expandingTildeInPath)
            if index > 0 {
                let stem = url.deletingPathExtension().lastPathComponent
                url = url.deletingLastPathComponent().appending(path: "\(stem)-\(index + 1).\(url.pathExtension)")
            }
            do {
                try PNG.write(image, to: url)
                let data = try Data(contentsOf: url)
                generation.png_path = url.path
                generation.png_bytes = data.count
                generation.png_sha256 = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                if hasAlpha {
                    let stem = url.deletingPathExtension().lastPathComponent, folder = url.deletingLastPathComponent()
                    let rgb = folder.appending(path: "\(stem)_rgb.png")
                    try PNG.write(try RGBA8.overWhite(image), to: rgb)
                    generation.rgb_png_path = rgb.path
                    if let latents {
                        let file = folder.appending(path: "\(stem)_final_latents.f32")
                        let bytes = latents.withUnsafeBufferPointer { Data(buffer: $0) }
                        try bytes.write(to: file)
                        generation.latents_path = file.path
                        generation.latents_sha256 = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                    }
                }
            } catch {
                Telemetry.line("ERROR writing \(url.path): \(error)")
            }
        }
        Telemetry.line("IMAGE \(image.width)x\(image.height) seconds=\(generation.seconds) png=\(generation.png_path ?? "not written") png_sha256=\(generation.png_sha256 ?? "-") pixels_sha256=\(generation.pixels?.sha256 ?? "-") min=\(generation.pixels?.min ?? -1) max=\(generation.pixels?.max ?? -1) alpha_min=\(generation.pixels?.alpha?.min ?? -1) alpha_mean=\(generation.pixels?.alpha?.mean ?? -1)")
    }

    /// Types `text` into the Prompt field one character at a time, `-typeMs` apart.
    private static func typeCharacters(_ text: String, into engine: DiffusionEngine) async throws {
        engine.prompt = ""
        let start = ContinuousClock.now
        for (offset, character) in text.enumerated() {
            try await Task.sleep(until: start + .milliseconds(LaunchOptions.typeMs * Double(offset)), clock: .continuous)
            engine.prompt.append(character)
        }
    }

    private static func pause() async throws {
        try await Task.sleep(for: .seconds(LaunchOptions.hold))
    }

    private static func until(_ what: String, timeout: Double, _ condition: () -> Bool) async throws {
        let start = ContinuousClock.now
        while !condition() {
            if start.duration(to: .now).seconds > timeout {
                throw ImageGenError("Timed out after \(Int(timeout)) s waiting for \(what)")
            }
            try await Task.sleep(for: .milliseconds(100))
        }
    }

    private static func finish(_ engine: DiffusionEngine, status: String) {
        engine.report.status = status
        engine.report.thermal_at_end = RunReport.thermal()
        if let footprint = RunReport.footprint() {
            engine.report.footprint_mb_at_end = footprint.now
            engine.report.peak_footprint_mb = footprint.peak
        }
        let url = Telemetry.folder.appending(path: "result-\(Telemetry.launchEpoch).json")
        do {
            try FileManager.default.createDirectory(at: Telemetry.folder, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(engine.report).write(to: url, options: .atomic)
            Telemetry.line("DONE status=\(status) result=\(url.path) peak_footprint_mb=\(engine.report.peak_footprint_mb ?? -1)")
        } catch {
            Telemetry.line("DONE status=\(status) result=not-written error=\(error)")
        }
        if LaunchOptions.quit { NSApplication.shared.terminate(nil) }
    }
}
