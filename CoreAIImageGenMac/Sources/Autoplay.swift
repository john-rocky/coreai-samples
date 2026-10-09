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
/// - `-hubEndpoint <url>`: download from this host instead of https://huggingface.co.
/// - `-local <folder>`: press Local… with this folder instead of Download & Load.
/// - `-downloadOnly 1`: stop after the download, before the load.
/// - `-prompt "<text>"` or `-prompts <file, one prompt per line>`: type each prompt, press Generate.
/// - `-steps <n>`, `-guidance <x>`, `-seed <text>`: set the fields before Generate.
/// - `-stopAfterStep <n>`: press Stop once step n has finished.
/// - `-out <file.png>`: write each image there (`-2`, `-3`… before the extension for later prompts).
/// - `-hold <s>`: wait this long before each press and after each image (default 2).
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
    static var hubEndpoint: String? { defaults.string(forKey: "hubEndpoint") }
    static var local: String? { defaults.string(forKey: "local") }
    static var downloadOnly: Bool { defaults.bool(forKey: "downloadOnly") }
    static var prompt: String? { defaults.string(forKey: "prompt") }
    static var prompts: String? { defaults.string(forKey: "prompts") }
    static var steps: Int? { defaults.object(forKey: "steps") == nil ? nil : defaults.integer(forKey: "steps") }
    static var guidance: Float? { defaults.object(forKey: "guidance") == nil ? nil : defaults.float(forKey: "guidance") }
    static var seed: String? { defaults.string(forKey: "seed") }
    static var stopAfterStep: Int? { defaults.object(forKey: "stopAfterStep") == nil ? nil : defaults.integer(forKey: "stopAfterStep") }
    static var out: String? { defaults.string(forKey: "out") }
    static var hold: Double { defaults.object(forKey: "hold") == nil ? 2 : defaults.double(forKey: "hold") }
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
    }
    struct Pixels: Codable {
        var sha256: String
        var mean_rgb: [Double]
        var min: Int
        var max: Int
        var fraction_0: Double
        var fraction_255: Double
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
    }

    var status = "RUNNING"
    var launch_epoch = Telemetry.launchEpoch
    var model = RunReport.sysctl("hw.model")
    var os_version = ProcessInfo.processInfo.operatingSystemVersionString
    var os_build = RunReport.sysctl("kern.osversion")
    var architecture = AIModel.deviceArchitectureName
    var physical_memory_gb = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
    var repo = ModelFiles.repo
    var revision = ModelFiles.revision
    var compute = "Core AI, GPU preferred (the pipeline's SpecializationOptions); per-operation placement not observed"
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
        return Pixels(sha256: digest, mean_rgb: sums.map { $0 / Double(width * height) }, min: low, max: high,
                      fraction_0: Double(zeros) / samples, fraction_255: Double(full) / samples)
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
                engine.prompt = text
                if let steps = LaunchOptions.steps { engine.steps = steps }
                if let guidance = LaunchOptions.guidance { engine.guidance = guidance }
                if let seed = LaunchOptions.seed { engine.seedText = seed }
                Telemetry.line("TYPED prompt=\(text) steps=\(engine.steps) guidance=\(engine.guidance) seed=\(engine.seedText)")
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
                    record(image, index: index, into: &engine.report.generations[last])
                }
                try await pause()
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
    private static func record(_ image: CGImage, index: Int, into generation: inout RunReport.Generation) {
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
            } catch {
                Telemetry.line("ERROR writing \(url.path): \(error)")
            }
        }
        Telemetry.line("IMAGE \(image.width)x\(image.height) seconds=\(generation.seconds) png=\(generation.png_path ?? "not written") png_sha256=\(generation.png_sha256 ?? "-") pixels_sha256=\(generation.pixels?.sha256 ?? "-") min=\(generation.pixels?.min ?? -1) max=\(generation.pixels?.max ?? -1)")
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
