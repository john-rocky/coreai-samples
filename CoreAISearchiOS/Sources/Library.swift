import CryptoKit
import Foundation
import NaturalLanguage

// MARK: - The example shelf

/// One sentence of the bundled example books, linked to its neighbours for "Read in context".
struct BookSentence: Codable, Sendable, Identifiable, Equatable {
    let id: String
    let work: String
    let title: String
    let text: String
    let previous_id: String?
    let next_id: String?
}

/// Two public-domain Sherlock Holmes books (Project Gutenberg #1661 and #2852, the Gutenberg
/// header and licence text removed), one row per sentence: something to search before you add your own text.
struct BookLibrary: Sendable {
    let sentences: [BookSentence]
    let sha256: String
    private let positions: [String: Int]

    init(data: Data) throws {
        let rows = try JSONDecoder().decode([BookSentence].self, from: data)
        guard !rows.isEmpty, Set(rows.map(\.id)).count == rows.count,
              rows.allSatisfy({ !$0.title.isEmpty && !$0.text.isEmpty && !$0.text.localizedCaseInsensitiveContains("Gutenberg") }) else {
            throw SearchError.invalid("Bundled library validation failed")
        }
        let positions = Dictionary(uniqueKeysWithValues: rows.enumerated().map { ($0.element.id, $0.offset) })
        for row in rows {
            for (neighbour, inverse) in [(row.previous_id, true), (row.next_id, false)] {
                if let neighbour {
                    guard let i = positions[neighbour], rows[i].work == row.work, rows[i].title == row.title,
                          (inverse ? rows[i].next_id : rows[i].previous_id) == row.id else {
                        throw SearchError.invalid("Library context links are inconsistent")
                    }
                }
            }
        }
        sentences = rows
        sha256 = IndexStore.digest(data)
        self.positions = positions
    }

    var works: [String] {
        var seen = Set<String>()
        return sentences.map(\.work).filter { seen.insert($0).inserted }
    }

    func context(at index: Int, radius: Int = 2) -> [BookSentence] {
        guard sentences.indices.contains(index) else { return [] }
        var before: [BookSentence] = [], after: [BookSentence] = []
        var cursor = sentences[index]
        for _ in 0..<radius {
            guard let id = cursor.previous_id, let i = positions[id] else { break }
            cursor = sentences[i]
            before.insert(cursor, at: 0)
        }
        cursor = sentences[index]
        for _ in 0..<radius {
            guard let id = cursor.next_id, let i = positions[id] else { break }
            cursor = sentences[i]
            after.append(cursor)
        }
        return before + [sentences[index]] + after
    }
}

// MARK: - Your text

/// One sentence of text you added. `document` groups the sentences of one paste or one file, in order.
struct NoteSentence: Codable, Sendable, Identifiable, Equatable {
    let id: String
    let document: String
    let source: String
    let text: String
}

enum TextSplitter {
    /// The sentences of `text` in order: NaturalLanguage's sentence boundaries, each trimmed of
    /// surrounding whitespace; whitespace-only pieces are dropped.
    static func sentences(in text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        var result: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty { result.append(sentence) }
            return true
        }
        return result
    }
}

extension [NoteSentence] {
    /// Up to `radius` sentences either side of `index`, from the same paste or file.
    func context(at index: Int, radius: Int = 2) -> [NoteSentence] {
        guard indices.contains(index) else { return [] }
        let document = self[index].document
        var low = index, high = index
        while low > startIndex, index - low < radius, self[low - 1].document == document { low -= 1 }
        while high < endIndex - 1, high - index < radius, self[high + 1].document == document { high += 1 }
        return Array(self[low...high])
    }
}

// MARK: - Saved vectors

enum IndexStore {
    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func encode(_ matrix: [Float]) -> Data {
        // All supported targets are little-endian arm64. The file has no header:
        // exactly N * 384 contiguous IEEE754 float32 values in row order.
        matrix.withUnsafeBytes { Data($0) }
    }

    static func decode(_ data: Data, rows: Int) -> [Float]? {
        guard data.count == rows * SentenceRanker.dimension * MemoryLayout<Float>.size else { return nil }
        var matrix = [Float](repeating: 0, count: rows * SentenceRanker.dimension)
        _ = matrix.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return matrix.allSatisfy(\.isFinite) ? matrix : nil
    }
}

struct BookIndexMetadata: Codable, Sendable {
    let schema: Int
    let rows: Int
    let dimension: Int
    /// What the vectors were made from: the library's SHA-256 and the model revision.
    let key: String
    let matrix_sha256: String
    let index_total_seconds: Double
    let index_ms_per_sentence: Double
    let first20s_sentences: Int
    let first20s_ms_per_sentence: Double
    let truncated_at_126_body_tokens: Int
}

/// The example books' vectors, made once on this iPhone. They can be made again, so they live in Caches.
enum BookIndexFile {
    static let matrixURL = URL.cachesDirectory.appending(path: "book-index.f32")
    static let metadataURL = URL.cachesDirectory.appending(path: "book-index.json")

    static func load(rows: Int, key: String) -> ([Float], BookIndexMetadata)? {
        guard let meta = try? JSONDecoder().decode(BookIndexMetadata.self, from: Data(contentsOf: metadataURL)),
              meta.schema == 1, meta.rows == rows, meta.dimension == SentenceRanker.dimension, meta.key == key,
              let data = try? Data(contentsOf: matrixURL), IndexStore.digest(data) == meta.matrix_sha256,
              let matrix = IndexStore.decode(data, rows: rows) else { return nil }
        return (matrix, meta)
    }

    static func save(_ matrix: [Float], metadata: BookIndexMetadata) throws {
        try IndexStore.encode(matrix).write(to: matrixURL, options: .atomic)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: metadataURL, options: .atomic)
    }
}

/// Your sentences and their vectors, in Documents so they survive relaunches and are backed up with the app.
enum NotesStore {
    struct File: Codable {
        var schema = 1
        /// The model revision that made the vectors; another revision means the text is embedded again.
        let revision: String
        let sentences: [NoteSentence]
    }

    static let textURL = URL.documentsDirectory.appending(path: "my-notes.json")
    static let matrixURL = URL.documentsDirectory.appending(path: "my-notes.f32")

    static func load() throws -> File? {
        guard FileManager.default.fileExists(atPath: textURL.path) else { return nil }
        return try JSONDecoder().decode(File.self, from: Data(contentsOf: textURL))
    }

    static func loadMatrix(rows: Int) -> [Float]? {
        guard let data = try? Data(contentsOf: matrixURL) else { return nil }
        return IndexStore.decode(data, rows: rows)
    }

    static func save(_ sentences: [NoteSentence], matrix: [Float], revision: String) throws {
        try IndexStore.encode(matrix).write(to: matrixURL, options: .atomic)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(File(revision: revision, sentences: sentences)).write(to: textURL, options: .atomic)
    }

    static func clear() throws {
        for url in [textURL, matrixURL] where FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }
}
