import Foundation
import SwiftUI

@MainActor
final class SearchModel: ObservableObject {
    enum Phase: Equatable {
        case checking
        case needsDownload
        case downloading
        case downloadFailed(String)
        case loading
        case indexing
        case ready
        case failed(String)

        var logName: String {
            switch self {
            case .checking: "checking"
            case .needsDownload: "needsDownload"
            case .downloading: "downloading"
            case .downloadFailed(let message): "downloadFailed message=\(message)"
            case .loading: "loading"
            case .indexing: "indexing"
            case .ready: "ready"
            case .failed(let message): "failed message=\(message)"
            }
        }
    }

    enum Filter: String, CaseIterable, Identifiable {
        case all = "All"
        case notes = "My notes"
        var id: Self { self }
    }

    /// The Add sheet's state: the text (pasted or imported), where it came from, and the
    /// sentences embedded so far while adding.
    struct Draft: Equatable {
        var text = ""
        var source = "Pasted text"
        var message: String?
        var added = 0
        var total = 0
        var isAdding: Bool { total > 0 }
    }

    static let maxSentencesPerAdd = 2_000

    @Published private(set) var phase: Phase = .checking
    @Published private(set) var downloadedBytes: Int64 = 0
    @Published private(set) var indexProgress = IndexProgress(done: 0, total: 0, elapsedSeconds: 0)
    @Published private(set) var hits: [Hit] = []
    @Published private(set) var lastSearch: SearchResult?
    @Published private(set) var notes: [NoteSentence] = []
    @Published private(set) var notice: String?
    @Published var query = "" { didSet { if query != oldValue && !settingQuery { scheduleSearch() } } }
    @Published var filter: Filter = .all { didSet { if filter != oldValue { scheduleSearch(debounce: false) } } }
    @Published var isAddSheetPresented = false
    @Published var draft = Draft()

    private(set) var library: BookLibrary?
    private(set) var presets: [String] = []
    private(set) var addTask: Task<Void, Never>?
    /// What happened in this process, for hands-off runs (`Autoplay.swift`).
    var report = RunReport()
    private let engine = EmbeddingEngine()
    private var started = false
    private var settingQuery = false
    private var work: Task<Void, Never>?
    private var searchTask: Task<Void, Never>?
    private var generation = 0
    private var loggedDownloadStep: Int64 = -1

    var bookCount: Int { library?.sentences.count ?? 0 }

    func start() {
        guard !started else { return }
        started = true
        Telemetry.begin()
        if LaunchOptions.autoplay { Task { await Autoplay.run(self) } }
        do {
            guard let booksURL = Bundle.main.url(forResource: "sentences", withExtension: "json"),
                  let presetsURL = Bundle.main.url(forResource: "presets", withExtension: "json") else {
                throw SearchError.invalid("Missing bundled example books")
            }
            library = try BookLibrary(data: Data(contentsOf: booksURL))
            presets = try JSONDecoder().decode([String].self, from: Data(contentsOf: presetsURL))
        } catch {
            setPhase(.failed(String(describing: error)))
            return
        }
        if ModelFiles.isOnDisk { prepare(afterDownload: false) } else { setPhase(.needsDownload) }
    }

    private func setPhase(_ next: Phase) {
        phase = next
        Telemetry.line("PHASE \(next.logName)")
    }

    // MARK: Download and load

    /// The Download button and the Retry button.
    func download() {
        switch phase {
        case .needsDownload, .downloadFailed: break
        default: return
        }
        let endpoint = LaunchOptions.hubEndpoint
        downloadedBytes = ModelFiles.bytesOnDisk
        loggedDownloadStep = -1
        setPhase(.downloading)
        Telemetry.line("DOWNLOAD start endpoint=\(endpoint ?? "default") on_disk=\(downloadedBytes) total=\(ModelFiles.totalBytes)")
        let start = DispatchTime.now().uptimeNanoseconds
        work = Task {
            do {
                try await ModelFiles.download(endpoint: endpoint) { bytes in
                    Task { @MainActor in self.showDownload(bytes) }
                }
                let seconds = Clock.milliseconds(since: start) / 1000
                report.download = .init(seconds: seconds, bytes_on_disk: ModelFiles.bytesOnDisk,
                                        expected_bytes: ModelFiles.totalBytes, endpoint: endpoint ?? "default")
                Telemetry.line("DOWNLOAD done bytes_on_disk=\(ModelFiles.bytesOnDisk) seconds=\(seconds)")
                prepare(afterDownload: true)
            } catch {
                report.download_errors.append(error.localizedDescription)
                setPhase(.downloadFailed(error.localizedDescription))
            }
        }
    }

    private func showDownload(_ bytes: Int64) {
        guard phase == .downloading else { return }
        downloadedBytes = max(downloadedBytes, bytes)
        let step = downloadedBytes * 20 / max(ModelFiles.totalBytes, 1)   // a log line every 5 %
        if step != loggedDownloadStep {
            loggedDownloadStep = step
            Telemetry.line("DOWNLOAD progress bytes=\(downloadedBytes) percent=\(step * 5)")
        }
    }

    /// The Try again button after a load error.
    func retryLoad() {
        guard case .failed = phase, library != nil else { return }
        if ModelFiles.isOnDisk { prepare(afterDownload: false) } else { setPhase(.needsDownload) }
    }

    private func prepare(afterDownload: Bool) {
        guard let library else { return }
        setPhase(.loading)
        work = Task {
            do {
                let load = try await engine.load(modelFolder: ModelFiles.modelURL, tokenizerFolder: ModelFiles.tokenizerURL)
                report.loads.append(.init(model_seconds: load.modelSeconds, tokenizer_seconds: load.tokenizerSeconds,
                                          first_call_ms: load.firstCallMS, after_download: afterDownload))
                Telemetry.line("LOAD model_s=\(load.modelSeconds) tokenizer_s=\(load.tokenizerSeconds) first_call_ms=\(load.firstCallMS) after_download=\(afterDownload)")
                notes = try await engine.restoreNotes(revision: ModelFiles.revision)
                filter = Self.defaultFilter(hasNotes: !notes.isEmpty)
                report.notes_restored = notes.count
                Telemetry.line("NOTES restored count=\(notes.count)")
                setPhase(.indexing)
                indexProgress = IndexProgress(done: 0, total: library.sentences.count, elapsedSeconds: 0)
                let (index, built) = try await engine.prepareBookIndex(library, revision: ModelFiles.revision) { [weak self] update in
                    await self?.showIndex(update)
                }
                report.book_index = .init(built: built, metadata: index)
                Telemetry.line("INDEX \(built ? "built" : "loaded") rows=\(index.rows) seconds=\(index.index_total_seconds) ms_per_sentence=\(index.index_ms_per_sentence) first20s_sentences=\(index.first20s_sentences) first20s_ms_per_sentence=\(index.first20s_ms_per_sentence) thermal=\(RunReport.thermal())")
                setPhase(.ready)
                scheduleSearch(debounce: false)
            } catch {
                setPhase(.failed(String(describing: error)))
            }
        }
    }

    private func showIndex(_ update: IndexProgress) {
        indexProgress = update
        if update.done % 500 == 0 || update.done == update.total {
            Telemetry.line("INDEX progress=\(update.done)/\(update.total) elapsed_s=\(update.elapsedSeconds)")
        }
    }

    // MARK: Search

    private func scheduleSearch(debounce: Bool = true) {
        guard phase == .ready else { return }
        generation += 1
        let current = generation
        searchTask?.cancel()
        lastSearch = nil
        let text = query
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { hits = []; return }
        let scope: SearchScope = filter == .all ? .all : .notes
        searchTask = Task { [weak self] in
            do {
                if debounce { try await Task.sleep(for: .milliseconds(60)) }
                guard let self else { return }
                let result = try await self.engine.search(text, scope: scope)
                try Task.checkCancellation()
                guard self.generation == current else { return }
                self.present(result)
            } catch is CancellationError {
                // A newer edit owns the screen. Core AI calls are serialized by the engine.
            } catch {
                guard let self, self.generation == current else { return }
                self.notice = "Search failed: \(error.localizedDescription)"
            }
        }
    }

    private func present(_ result: SearchResult) {
        lastSearch = result
        hits = result.hits
    }

    /// Runs `text` at once (no typing, no debounce) and shows it like a typed query.
    func searchNow(_ text: String) async throws -> SearchResult {
        searchTask?.cancel()
        generation += 1
        settingQuery = true
        query = text
        settingQuery = false
        let result = try await engine.search(text, scope: filter == .all ? .all : .notes)
        present(result)
        return result
    }

    // MARK: Your text

    func presentAddSheet() {
        draft = Draft()
        isAddSheetPresented = true
    }

    func closeAddSheet() {
        guard !draft.isAdding else { return }
        isAddSheetPresented = false
    }

    /// The Import .txt button: the file's text goes into the sheet's editor. Only UTF-8 text is read.
    func importFile(_ url: URL) {
        draft.message = nil
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        guard let data = try? Data(contentsOf: url), let text = String(data: data, encoding: .utf8) else {
            show("Could not read this file as text")
            return
        }
        draft.text = text
        draft.source = url.lastPathComponent
    }

    /// The Add button: splits the text into sentences, embeds them, and adds them to the search.
    func addDraft() {
        guard addTask == nil else { return }
        draft.message = nil
        let sentences = TextSplitter.sentences(in: draft.text)
        guard !sentences.isEmpty else { show("Nothing to add"); return }
        guard sentences.count <= Self.maxSentencesPerAdd else {
            show("This text has \(sentences.count.formatted()) sentences. Add up to \(Self.maxSentencesPerAdd.formatted()) at a time.")
            return
        }
        let source = draft.source
        draft.added = 0
        draft.total = sentences.count
        Telemetry.line("NOTES adding count=\(sentences.count) source=\(source)")
        addTask = Task {
            do {
                let result = try await engine.addNotes(sentences, source: source, revision: ModelFiles.revision) { [weak self] done in
                    await self?.showAdded(done)
                }
                notes = result.notes
                report.notes_adds.append(.init(source: source, sentences: result.added, seconds: result.seconds,
                    ms_per_sentence: result.seconds * 1000 / Double(max(result.added, 1)), truncated: result.truncated))
                Telemetry.line("NOTES added count=\(result.added) total=\(notes.count) seconds=\(result.seconds) ms_per_sentence=\(result.seconds * 1000 / Double(max(result.added, 1)))")
                draft = Draft()
                isAddSheetPresented = false
                notice = "Added \(result.added.formatted()) sentences from \(source)"
                useDefaultFilter()
            } catch is CancellationError {
                draft.total = 0
                show("Cancelled. Nothing was added.")
            } catch {
                draft.total = 0
                show("Could not add this text: \(error.localizedDescription)")
            }
            addTask = nil
        }
    }

    private func showAdded(_ done: Int) {
        draft.added = done
    }

    func cancelAdd() {
        addTask?.cancel()
    }

    /// The Clear my notes button.
    func clearNotes() async {
        do {
            try await engine.clearNotes()
            notes = []
            notice = "Cleared your notes"
            Telemetry.line("NOTES cleared total=0")
            useDefaultFilter()
        } catch {
            notice = "Could not clear your notes: \(error.localizedDescription)"
        }
    }

    /// Your text is what the app is for: once you have added some, search opens in My notes.
    /// The example books stay one tap away in All.
    static func defaultFilter(hasNotes: Bool) -> Filter { hasNotes ? .notes : .all }

    private func useDefaultFilter() {
        let wanted = Self.defaultFilter(hasNotes: !notes.isEmpty)
        if filter != wanted { filter = wanted } else { scheduleSearch(debounce: false) }
        Telemetry.line("FILTER \(filter.rawValue) (notes=\(notes.count))")
    }

    private func show(_ message: String) {
        draft.message = message
        report.messages.append(message)
        Telemetry.line("MESSAGE \(message)")
    }
}
