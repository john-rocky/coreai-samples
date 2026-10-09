import CoreAI
import Darwin
import Foundation
import os

/// Launch arguments for hands-off runs (recordings and on-device checks). Each one presses the same
/// buttons a person would; nothing here is a separate code path. With devicectl, put them after `--`:
/// `xcrun devicectl device process launch --device <id> com.coreai.searchios -- -autoplay 1 -log 1`.
///
/// - `-autoplay 1`: tap Download (and Retry once if it fails), wait until the search is ready, then run the rest.
/// - `-hubEndpoint <url>`: download from this host instead of https://huggingface.co.
/// - `-notes <file in Documents>`: Import .txt with that file, then tap Add. `-showAddSheet 1` opens the sheet
///   first and keeps it on screen for `-hold` seconds (default 3) before Add.
/// - `-filter notes` / `-filter all`: tap My notes or All. `-clearNotes 1`: tap Clear my notes.
/// - `-query "<text>"`: type the text into the search field, one character every 70 ms.
/// - `-queries <file in Documents>`: run each line as a query at once, one after another.
/// - `-log 1`: mirror the status lines to Documents/search-autoplay.log; at the end write
///   Documents/search-result-<launch epoch>.json with every measured number of the run.
enum LaunchOptions {
    private static var defaults: UserDefaults { .standard }
    static var autoplay: Bool { defaults.bool(forKey: "autoplay") }
    static var log: Bool { defaults.bool(forKey: "log") }
    static var hubEndpoint: String? { defaults.string(forKey: "hubEndpoint") }
    static var notes: String? { defaults.string(forKey: "notes") }
    static var showAddSheet: Bool { defaults.bool(forKey: "showAddSheet") }
    @MainActor static var filter: SearchModel.Filter? {
        switch defaults.string(forKey: "filter") {
        case "notes": .notes
        case "all": .all
        default: nil
        }
    }
    static var clearNotes: Bool { defaults.bool(forKey: "clearNotes") }
    static var query: String? { defaults.string(forKey: "query") }
    static var queries: String? { defaults.string(forKey: "queries") }
    static var hold: Double { defaults.object(forKey: "hold") == nil ? 3 : defaults.double(forKey: "hold") }
}

enum Telemetry {
    static let launchEpoch = Int(Date().timeIntervalSince1970)

    /// With `-log 1`, stdout and stderr (this app's lines and Core AI's own messages) go to Documents/search-autoplay.log.
    static func begin() {
        if LaunchOptions.log {
            let url = URL.documentsDirectory.appending(path: "search-autoplay.log")
            let descriptor = url.path.withCString { Darwin.open($0, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR) }
            if descriptor >= 0 {
                dup2(descriptor, STDOUT_FILENO)
                dup2(descriptor, STDERR_FILENO)
                Darwin.close(descriptor)
            }
        }
        line("LAUNCH epoch=\(launchEpoch) pid=\(getpid()) device=\(RunReport.sysctl("hw.machine")) os_build=\(RunReport.sysctl("kern.osversion")) arch=\(AIModel.deviceArchitectureName) thermal=\(RunReport.thermal()) available_mb=\(os_proc_available_memory() / 1_048_576) args=\(ProcessInfo.processInfo.arguments.dropFirst().joined(separator: " "))")
    }

    static func line(_ text: String) {
        print("SEARCH \(ISO8601DateFormatter().string(from: Date())) \(text)")
        fflush(nil)
    }
}

/// Every measured number of one launch; written as JSON when a hands-off run ends.
struct RunReport: Codable {
    struct Download: Codable { var seconds: Double; var bytes_on_disk: Int64; var expected_bytes: Int64; var endpoint: String }
    struct Load: Codable { var model_seconds: Double; var tokenizer_seconds: Double; var first_call_ms: Double; var after_download: Bool }
    struct BookIndex: Codable { var built: Bool; var metadata: BookIndexMetadata }
    struct NotesAdd: Codable { var source: String; var sentences: Int; var seconds: Double; var ms_per_sentence: Double; var truncated: Int }
    struct Row: Codable { var source: String; var id: String; var label: String; var text: String; var score: Float }
    struct Query: Codable {
        var query: String
        var scope: String
        var embed_and_rank_ms: Double
        var body_tokens: Int
        var shown: [Row]
        var book_top8: [Row]
        var note_top8: [Row]
    }

    var status = "RUNNING"
    var launch_epoch = Telemetry.launchEpoch
    var device = RunReport.sysctl("hw.machine")
    var os_version = ProcessInfo.processInfo.operatingSystemVersionString
    var os_build = RunReport.sysctl("kern.osversion")
    var architecture = AIModel.deviceArchitectureName
    var app_build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unavailable"
    var repo = ModelFiles.repo
    var revision = ModelFiles.revision
    var compute = "Core AI, GPU preferred (SpecializationOptions); per-operation placement not observed"
    var arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
    var available_memory_mb_at_launch = Double(os_proc_available_memory()) / 1_048_576
    var thermal_at_launch = RunReport.thermal()
    var thermal_at_end: String?
    var footprint_mb_at_end: Double?
    var peak_footprint_mb: Double?
    var download_errors: [String] = []
    var download: Download?
    var loads: [Load] = []
    var book_index: BookIndex?
    var notes_restored: Int?
    var notes_adds: [NotesAdd] = []
    var notes_total_at_end = 0
    var messages: [String] = []
    var queries: [Query] = []

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

    /// The process's physical footprint now and its peak (what jetsam counts), in MB.
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
}

@MainActor
enum Autoplay {
    static func run(_ model: SearchModel) async {
        var status = "DONE"
        do {
            try await until("the first screen") { model.phase != .checking }
            if model.phase == .needsDownload {
                try await Task.sleep(for: .seconds(LaunchOptions.hold))
                Telemetry.line("TAP Download")
                model.download()
                try await until("the download", timeout: 1200) { model.phase != .downloading }
            }
            if case .downloadFailed = model.phase {
                try await Task.sleep(for: .seconds(LaunchOptions.hold))
                Telemetry.line("TAP Retry")
                model.download()
                try await until("the retry", timeout: 1200) { model.phase != .downloading }
                if case .downloadFailed = model.phase { return finish(model, status: "DOWNLOAD_FAILED") }
            }
            try await until("the search screen", timeout: 1800) {
                if case .failed = model.phase { return true }
                return model.phase == .ready
            }
            if case .failed = model.phase { return finish(model, status: "FAILED") }
            Telemetry.line("FILTER \(model.filter.rawValue) at ready (notes=\(model.notes.count))")

            if LaunchOptions.clearNotes {
                Telemetry.line("TAP Clear my notes")
                await model.clearNotes()
            }
            if let name = LaunchOptions.notes {
                if LaunchOptions.showAddSheet {
                    Telemetry.line("TAP Add your text")
                    model.presentAddSheet()
                    try await Task.sleep(for: .milliseconds(700))
                }
                Telemetry.line("TAP Import .txt file=\(name)")
                model.importFile(URL.documentsDirectory.appending(path: name))
                if LaunchOptions.showAddSheet {
                    Telemetry.line("SHEET open source=\(model.draft.source) chars=\(model.draft.text.count) hold_s=\(LaunchOptions.hold)")
                    try await Task.sleep(for: .seconds(LaunchOptions.hold))
                }
                if model.draft.message == nil {
                    Telemetry.line("TAP Add")
                    model.addDraft()
                    await model.addTask?.value
                }
            }
            if let filter = LaunchOptions.filter {
                Telemetry.line("TAP \(filter.rawValue)")
                model.filter = filter
            }
            if let text = LaunchOptions.query {
                model.query = ""
                for character in text {
                    model.query.append(character)
                    try await Task.sleep(for: .milliseconds(70))
                }
                Telemetry.line("TYPED \(text)")
                try await until("the result of the typed query") { model.lastSearch?.query == text }
                if let result = model.lastSearch { record(result, model) }
            }
            if let name = LaunchOptions.queries {
                let lines = try String(contentsOf: URL.documentsDirectory.appending(path: name), encoding: .utf8)
                    .split(whereSeparator: \.isNewline).map(String.init).filter { !$0.isEmpty }
                for text in lines {
                    record(try await model.searchNow(text), model)
                    try await Task.sleep(for: .milliseconds(500))
                }
            }
        } catch {
            Telemetry.line("ERROR \(error)")
            status = "ERROR"
        }
        finish(model, status: status)
    }

    private static func until(_ what: String, timeout: Double = 120, _ condition: () -> Bool) async throws {
        let start = DispatchTime.now().uptimeNanoseconds
        while !condition() {
            if Clock.milliseconds(since: start) > timeout * 1000 {
                throw SearchError.invalid("Timed out after \(Int(timeout)) s waiting for \(what)")
            }
            try await Task.sleep(for: .milliseconds(100))
        }
    }

    private static func record(_ result: SearchResult, _ model: SearchModel) {
        func book(_ index: Int, _ score: Float) -> RunReport.Row {
            let row = model.library!.sentences[index]
            return .init(source: "book", id: row.id, label: "\(row.work) · \(row.title)", text: row.text, score: score)
        }
        func note(_ index: Int, _ score: Float) -> RunReport.Row {
            guard model.notes.indices.contains(index) else {
                return .init(source: "note", id: "missing-\(index)", label: "My notes", text: "", score: score)
            }
            let row = model.notes[index]
            return .init(source: "note", id: row.id, label: "My notes · \(row.source)", text: row.text, score: score)
        }
        let shown = result.hits.map { $0.source == .book ? book($0.index, $0.score) : note($0.index, $0.score) }
        model.report.queries.append(.init(query: result.query, scope: result.scope.rawValue,
            embed_and_rank_ms: result.embedAndRankMS, body_tokens: result.bodyTokens, shown: shown,
            book_top8: result.bookTop.map { book($0.index, $0.score) },
            note_top8: result.noteTop.map { note($0.index, $0.score) }))
        let top = shown.first.map { "\($0.source):\($0.id) \(String(format: "%.4f", $0.score)) \($0.text.prefix(60))" } ?? "none"
        Telemetry.line("RESULT ms=\(String(format: "%.3f", result.embedAndRankMS)) scope=\(result.scope.rawValue) query=\(result.query) top1=\(top)")
    }

    private static func finish(_ model: SearchModel, status: String) {
        model.report.status = status
        model.report.thermal_at_end = RunReport.thermal()
        model.report.notes_total_at_end = model.notes.count
        if let footprint = RunReport.footprint() {
            model.report.footprint_mb_at_end = footprint.now
            model.report.peak_footprint_mb = footprint.peak
        }
        let name = "search-result-\(Telemetry.launchEpoch).json"
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(model.report).write(to: URL.documentsDirectory.appending(path: name), options: .atomic)
            Telemetry.line("DONE status=\(status) result=\(name) peak_footprint_mb=\(model.report.peak_footprint_mb ?? -1)")
        } catch {
            Telemetry.line("DONE status=\(status) result=not-written error=\(error)")
        }
    }
}
