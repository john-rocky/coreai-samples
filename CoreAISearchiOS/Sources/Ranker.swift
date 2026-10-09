import Accelerate
import Foundation

struct RankedSentence: Codable, Sendable, Equatable {
    let index: Int
    let score: Float
}

enum SearchError: Error, CustomStringConvertible, LocalizedError, Sendable {
    case invalid(String)
    var description: String {
        switch self { case .invalid(let message): return message }
    }
    var errorDescription: String? { description }
}

/// Rows and query are L2-normalized embeddings, so the dot product is the cosine similarity.
enum SentenceRanker {
    static let dimension = 384

    static func scores(matrix: [Float], query: [Float]) throws -> [Float] {
        guard query.count == dimension, !matrix.isEmpty,
              matrix.count.isMultiple(of: dimension), query.allSatisfy(\.isFinite) else {
            throw SearchError.invalid("Embedding matrix/query shape or finite-value mismatch")
        }
        let count = matrix.count / dimension
        var scores = [Float](repeating: 0, count: count)
        vDSP_mmul(matrix, 1, query, 1, &scores, 1,
                  vDSP_Length(count), 1, vDSP_Length(dimension))
        guard scores.allSatisfy(\.isFinite) else {
            throw SearchError.invalid("Ranking produced a nonfinite score")
        }
        return scores
    }

    static func top(_ scores: [Float], count: Int = 8) -> [RankedSentence] {
        // Stable tie policy is row order. Select only k entries without sorting all N.
        var best: [RankedSentence] = []
        for (index, score) in scores.enumerated() {
            let item = RankedSentence(index: index, score: score)
            let position = best.firstIndex { score > $0.score || (score == $0.score && index < $0.index) } ?? best.count
            if position < count {
                best.insert(item, at: position)
                if best.count > count { best.removeLast() }
            }
        }
        return best
    }

    static func rank(matrix: [Float], query: [Float], count: Int = 8) throws -> [RankedSentence] {
        top(try scores(matrix: matrix, query: query), count: count)
    }
}
