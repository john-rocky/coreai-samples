import CoreAI
import Darwin
import Foundation

enum Clock {
    static func milliseconds(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
    }
}

struct IndexProgress: Sendable {
    let done: Int
    let total: Int
    let elapsedSeconds: Double
    var millisecondsPerSentence: Double { done > 0 ? elapsedSeconds * 1000 / Double(done) : 0 }
}

struct LoadReport: Sendable {
    let tokenizerSeconds: Double
    /// `AIModel(contentsOf:)` + `loadFunction`: Core AI specializes the `.aimodel` here on the first load.
    let modelSeconds: Double
    /// One embedding right after the load; the first call of a process is the slow one.
    let firstCallMS: Double
}

struct NotesAdded: Sendable {
    let notes: [NoteSentence]
    let added: Int
    let seconds: Double
    let truncated: Int
}

enum SearchScope: String, Sendable { case all, notes }

struct Hit: Sendable, Equatable, Identifiable {
    enum Source: String, Sendable { case book, note }
    let source: Source
    let index: Int
    let score: Float
    var id: String { "\(source.rawValue)-\(index)" }
}

struct SearchResult: Sendable {
    let query: String
    let scope: SearchScope
    /// What the screen shows: the best 8 of the books and your text together.
    let hits: [Hit]
    /// The best 8 book sentences alone (empty for `.notes`) and the best 8 of your sentences alone.
    let bookTop: [RankedSentence]
    let noteTop: [RankedSentence]
    let embedAndRankMS: Double
    let bodyTokens: Int
}

/// The tokenizer, the Core AI function and both sets of vectors live off the main actor.
/// A permit also serializes across actor reentrancy at function.run's suspension point.
actor EmbeddingEngine {
    private var tokenizer: GraniteTokenizer?
    private var model: AIModel?
    private var function: InferenceFunction?
    private var idsDescriptor: NDArrayDescriptor?
    private var maskDescriptor: NDArrayDescriptor?
    private var bookMatrix: [Float] = []
    private var notes: [NoteSentence] = []
    private var noteMatrix: [Float] = []
    private var occupied = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    private func acquire() async {
        if !occupied { occupied = true; return }
        await withCheckedContinuation { waiters.append($0) }
    }

    private func release() {
        if waiters.isEmpty { occupied = false }
        else { waiters.removeFirst().resume() }
    }

    func load(modelFolder: URL, tokenizerFolder: URL) async throws -> LoadReport {
        await acquire()
        defer { release() }
        let tokenizerStart = DispatchTime.now().uptimeNanoseconds
        let tokenizer = try GraniteTokenizer(tokenizerURL: tokenizerFolder.appending(path: "tokenizer.json"),
                                             tokenizerConfigURL: tokenizerFolder.appending(path: "tokenizer_config.json"))
        let tokenizerSeconds = Clock.milliseconds(since: tokenizerStart) / 1000
        let modelStart = DispatchTime.now().uptimeNanoseconds
        let loaded = try await AIModel(contentsOf: modelFolder,
                                       options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        // The graph's contract: token ids and mask [1,128] int32 in, one [1,384] fp32 unit vector out.
        guard let descriptor = loaded.functionDescriptor(for: "main"), descriptor.stateNames.isEmpty,
              Set(descriptor.inputNames) == Set(["input_ids", "attention_mask"]), descriptor.outputNames == ["embedding"],
              case .ndArray(let ids) = descriptor.inputDescriptor(of: "input_ids"),
              case .ndArray(let mask) = descriptor.inputDescriptor(of: "attention_mask"),
              case .ndArray(let output) = descriptor.outputDescriptor(of: "embedding"),
              ids.shape == [1, 128], mask.shape == [1, 128], ids.scalarType == .int32, mask.scalarType == .int32,
              output.shape == [1, 384], output.scalarType == .float32,
              let main = try loaded.loadFunction(named: "main") else {
            throw SearchError.invalid("Stateless main function contract mismatch")
        }
        let modelSeconds = Clock.milliseconds(since: modelStart) / 1000
        self.tokenizer = tokenizer
        model = loaded // Retain the owning model through all asynchronous function uses.
        function = main
        idsDescriptor = ids
        maskDescriptor = mask
        let callStart = DispatchTime.now().uptimeNanoseconds
        _ = try await embed("Warm up.")
        return LoadReport(tokenizerSeconds: tokenizerSeconds, modelSeconds: modelSeconds,
                          firstCallMS: Clock.milliseconds(since: callStart))
    }

    private func makeInput(_ values: [Int32], descriptor: NDArrayDescriptor) throws -> NDArray {
        guard descriptor.shape == [1, 128], descriptor.scalarType == .int32, values.count == 128 else {
            throw SearchError.invalid("Input contract mismatch")
        }
        var result = NDArray(descriptor: descriptor)
        var view = result.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: values)
        return result
    }

    private func embed(_ text: String) async throws -> (vector: [Float], bodyTokens: Int) {
        try Task.checkCancellation()
        guard let tokenizer, let function, let idsDescriptor, let maskDescriptor else {
            throw SearchError.invalid("Model is not ready")
        }
        // No prefix, stripping or normalization. Tokenizer truncates the body at 126,
        // wraps CLS/SEP, and pads with 179935. The second tokenization uses its cache.
        let bodyTokens = try tokenizer.encodeBody(text: text).count
        let encoded = try tokenizer.encode(text: text, sequenceLength: 128)
        let ids = try makeInput(encoded.inputIDs, descriptor: idsDescriptor)
        let mask = try makeInput(encoded.attentionMask, descriptor: maskDescriptor)
        var outputs = try await function.run(inputs: ["input_ids": ids, "attention_mask": mask])
        guard let array = outputs.remove("embedding")?.ndArray, array.shape == [1, 384], array.scalarType == .float32 else {
            throw SearchError.invalid("Missing [1,384] fp32 embedding")
        }
        let view = array.view(as: Float.self)
        guard view.interleaveLayout == nil else { throw SearchError.invalid("Interleaved output is unsupported") }
        let values = view.withUnsafePointer { pointer, _, strides in
            (0..<384).map { pointer[$0 * strides[1]] }
        }
        guard values.allSatisfy(\.isFinite) else { throw SearchError.invalid("Nonfinite embedding") }
        let norm = sqrt(values.reduce(0.0) { $0 + Double($1) * Double($1) })
        guard abs(norm - 1) <= 0.002 else { throw SearchError.invalid("Embedding is not a unit vector: \(norm)") }
        try Task.checkCancellation()
        return (values, bodyTokens)
    }

    // MARK: The example books

    /// Reads the saved book vectors, or embeds all sentences once and saves them.
    func prepareBookIndex(_ library: BookLibrary, revision: String,
                          progress: @Sendable (IndexProgress) async -> Void) async throws -> (BookIndexMetadata, built: Bool) {
        await acquire()
        defer { release() }
        let rows = library.sentences.count
        let key = "\(library.sha256)@\(revision)"
        if let saved = BookIndexFile.load(rows: rows, key: key) {
            bookMatrix = saved.0
            await progress(IndexProgress(done: rows, total: rows, elapsedSeconds: saved.1.index_total_seconds))
            return (saved.1, false)
        }
        var matrix: [Float] = []
        matrix.reserveCapacity(rows * SentenceRanker.dimension)
        var truncated = 0, first20s = 0
        var first20sSeconds = 0.0
        let start = DispatchTime.now().uptimeNanoseconds
        await progress(IndexProgress(done: 0, total: rows, elapsedSeconds: 0))
        for (i, sentence) in library.sentences.enumerated() {
            let embedded = try await embed(sentence.text)
            matrix.append(contentsOf: embedded.vector)
            if embedded.bodyTokens > 126 { truncated += 1 }
            let elapsed = Clock.milliseconds(since: start) / 1000
            if elapsed <= 20 { first20s = i + 1; first20sSeconds = elapsed }
            await progress(IndexProgress(done: i + 1, total: rows, elapsedSeconds: elapsed))
        }
        let seconds = Clock.milliseconds(since: start) / 1000
        let metadata = BookIndexMetadata(schema: 1, rows: rows, dimension: SentenceRanker.dimension, key: key,
            matrix_sha256: IndexStore.digest(IndexStore.encode(matrix)), index_total_seconds: seconds,
            index_ms_per_sentence: seconds * 1000 / Double(rows), first20s_sentences: first20s,
            first20s_ms_per_sentence: first20s > 0 ? first20sSeconds * 1000 / Double(first20s) : 0,
            truncated_at_126_body_tokens: truncated)
        try BookIndexFile.save(matrix, metadata: metadata)
        bookMatrix = matrix
        return (metadata, true)
    }

    // MARK: Your text

    /// Loads the saved sentences; if their vectors are missing or came from another model revision,
    /// embeds the saved text again.
    func restoreNotes(revision: String) async throws -> [NoteSentence] {
        await acquire()
        defer { release() }
        guard let file = try NotesStore.load() else { return [] }
        if file.revision == revision, let matrix = NotesStore.loadMatrix(rows: file.sentences.count) {
            notes = file.sentences
            noteMatrix = matrix
            return notes
        }
        var matrix: [Float] = []
        for sentence in file.sentences { matrix.append(contentsOf: try await embed(sentence.text).vector) }
        try NotesStore.save(file.sentences, matrix: matrix, revision: revision)
        notes = file.sentences
        noteMatrix = matrix
        return notes
    }

    /// Embeds `texts`, then adds them and saves. Cancelling (or any error) adds nothing.
    func addNotes(_ texts: [String], source: String, revision: String,
                  progress: @Sendable (Int) async -> Void) async throws -> NotesAdded {
        await acquire()
        defer { release() }
        let document = UUID().uuidString
        var vectors: [Float] = []
        vectors.reserveCapacity(texts.count * SentenceRanker.dimension)
        var truncated = 0
        let start = DispatchTime.now().uptimeNanoseconds
        for (i, text) in texts.enumerated() {
            let embedded = try await embed(text)
            vectors.append(contentsOf: embedded.vector)
            if embedded.bodyTokens > 126 { truncated += 1 }
            await progress(i + 1)
        }
        let seconds = Clock.milliseconds(since: start) / 1000
        let added = texts.enumerated().map {
            NoteSentence(id: "\(document)-\($0.offset)", document: document, source: source, text: $0.element)
        }
        try NotesStore.save(notes + added, matrix: noteMatrix + vectors, revision: revision)
        notes += added
        noteMatrix += vectors
        return NotesAdded(notes: notes, added: added.count, seconds: seconds, truncated: truncated)
    }

    func clearNotes() async throws {
        await acquire()
        defer { release() }
        try NotesStore.clear()
        notes = []
        noteMatrix = []
    }

    // MARK: Search

    func search(_ query: String, scope: SearchScope) async throws -> SearchResult {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        let start = DispatchTime.now().uptimeNanoseconds
        let embedded = try await embed(query)
        let bookTop = scope == .all && !bookMatrix.isEmpty
            ? try SentenceRanker.rank(matrix: bookMatrix, query: embedded.vector) : []
        let noteTop = noteMatrix.isEmpty ? [] : try SentenceRanker.rank(matrix: noteMatrix, query: embedded.vector)
        let merged = bookTop.map { Hit(source: .book, index: $0.index, score: $0.score) }
            + noteTop.map { Hit(source: .note, index: $0.index, score: $0.score) }
        // Highest score first; on a tie your text comes before the books, then row order.
        let hits = merged.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            if $0.source != $1.source { return $0.source == .note }
            return $0.index < $1.index
        }
        return SearchResult(query: query, scope: scope, hits: Array(hits.prefix(8)), bookTop: bookTop, noteTop: noteTop,
                            embedAndRankMS: Clock.milliseconds(since: start), bodyTokens: embedded.bodyTokens)
    }
}
