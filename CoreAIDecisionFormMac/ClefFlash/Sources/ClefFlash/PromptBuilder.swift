// Vendored from github.com/john-rocky/coreai-model-zoo apps/ClefFlash/Sources/ClefFlash/PromptBuilder.swift at 5ef2247, unmodified below this line.
// PromptBuilder — a SystemOne request -> the decoder row, the head's spans and the decoder's static inputs, the
// author's `encode_record()` (conversion/clef_flash/host.py `build_ids` / `static_inputs`). Every piece is tokenized on
// its own without special tokens, and the pieces are concatenated:
//
//   prefix   "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"         36 ids
//   [image]  <|vision_start|> (248053), N x <|image_pad|> (248056), <|vision_end|> (248054), "\n" (198)   N + 3 ids
//   state    render(state)            (cut so the row fits the author's max_length 16384)
//   schema   "\n\nSCHEMA FIELDS:\n", then per question i (1-based)
//            "\nFIELD {i}\nID: {id}\nTYPE: {type}\nINSTRUCTION: " + [render(instructions or id)]
//            + "\nALLOWED OPTIONS:\n" + per option j "OPTION {j}: " + [render({"option_id", "description"})] + "\n"
//            + "END FIELD\n"                                             [..] = the head's question / option spans
//   suffix   "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"   18 ids
//
// Static inputs: the N pads go to the graph as ids V + k (V = 248320, k row-major over the merged grid),
// image_rc[k] = (k / W, k % W), rope_shift_start = 37 + N (the <|vision_end|> index), rope_shift_amount =
// N - max(H, W); a text-only row gets start 1 << 30, amount 0, image_rc zero.

import Foundation
import Tokenizers

@available(macOS 27, *)
public struct MergedGrid: Sendable, Equatable, CustomStringConvertible {
    public let h: Int
    public let w: Int

    public init(h: Int, w: Int) {
        self.h = h
        self.w = w
    }

    public var count: Int { h * w }
    public var description: String { "\(h)x\(w)" }
}

/// One question's place in the row: spans are [start, end) over the processor-form ids.
@available(macOS 27, *)
public struct QuestionLayout: Sendable, Equatable {
    public let questionID: String
    public let type: String
    public let typeID: Int
    public let questionSpan: [Int]
    public let optionSpans: [[Int]]
    public let optionIDs: [String]
}

@available(macOS 27, *)
public struct PromptRow: Sendable {
    /// The processor-form ids (N x <|image_pad|>): what the spans index and the lexical gather reads.
    public let ids: [Int]
    /// The decoder's input_ids: the image pads mapped to V + k.
    public let decoderIDs: [Int32]
    public let questions: [QuestionLayout]
    /// Index of <|vision_start|> (36) for an image row.
    public let tokenOffset: Int?
    public let grid: MergedGrid?
    /// [n_image_max * 2] row-major (k / W, k % W) for k < N, zero after.
    public let imageRC: [Int32]
    public let ropeShiftStart: Int32
    public let ropeShiftAmount: Int32
    public let stateTokens: Int
    public let stateTokensKept: Int
}

@available(macOS 27, *)
public struct PromptBuilder: Sendable {
    public static let vocab = 248_320
    public static let visionStart = 248_053
    public static let visionEnd = 248_054
    public static let imagePad = 248_056
    public static let padID = 248_044           // <|endoftext|>: fills the last decoder chunk
    public static let imEnd = 248_046
    public static let newline = 198
    public static let noShift: Int32 = 1 << 30
    public static let maxLength = 16_384         // the author's encode_record / systemone default

    public static let systemPrompt = "Read the complete state and schema. Decide every field jointly. Each answer "
        + "must be exactly one of that field's allowed options."
    public static let prefix = "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\nSTATE:\n"
    public static let suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    public static let schemaHeader = "\n\nSCHEMA FIELDS:\n"

    public let tokenizer: any Tokenizer
    public let nImageMax: Int
    public let prefixIDs: [Int]
    public let suffixIDs: [Int]
    public let newlineIDs: [Int]

    /// Loads the tokenizer and checks the ids the contract names (`metadata.json` decision.prompt and the special
    /// tokens).
    public init(tokenizer: any Tokenizer, nImageMax: Int, prefixTokens: Int = 36, suffixTokens: Int = 18) throws {
        self.tokenizer = tokenizer
        self.nImageMax = nImageMax
        prefixIDs = tokenizer.encode(text: Self.prefix, addSpecialTokens: false)
        suffixIDs = tokenizer.encode(text: Self.suffix, addSpecialTokens: false)
        newlineIDs = tokenizer.encode(text: "\n", addSpecialTokens: false)
        var bad: [String] = []
        if prefixIDs.count != prefixTokens { bad.append("prefix encodes to \(prefixIDs.count) ids, the contract says \(prefixTokens)") }
        if suffixIDs.count != suffixTokens { bad.append("suffix encodes to \(suffixIDs.count) ids, the contract says \(suffixTokens)") }
        if newlineIDs != [Self.newline] { bad.append("\"\\n\" encodes to \(newlineIDs), not [198]") }
        for (text, id) in [("<|endoftext|>", Self.padID), ("<|vision_start|>", Self.visionStart),
                           ("<|vision_end|>", Self.visionEnd), ("<|image_pad|>", Self.imagePad), ("<|im_end|>", Self.imEnd)] {
            let got = tokenizer.encode(text: text, addSpecialTokens: false)
            if got != [id] { bad.append("\(text) encodes to \(got), not [\(id)]") }
        }
        if prefixIDs.last.map({ $0 == Self.newline }) != true || suffixIDs.last != 25 {
            bad.append("prefix / suffix ends \(prefixIDs.suffix(2)) / \(suffixIDs.suffix(2)) (want ...198 / ...25)")
        }
        if !bad.isEmpty { throw ClefFlashError.contract("tokenizer: \(bad.joined(separator: "; "))") }
    }

    public static func load(tokenizerFolder: URL, nImageMax: Int, prefixTokens: Int = 36, suffixTokens: Int = 18)
        async throws -> PromptBuilder
    {
        try PromptBuilder(tokenizer: try await AutoTokenizer.from(modelFolder: tokenizerFolder), nImageMax: nImageMax,
                          prefixTokens: prefixTokens, suffixTokens: suffixTokens)
    }

    func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// `host.build_ids`. `grid` = the merged grid of the tower that makes the image rows, nil for a text-only row.
    public func build(_ request: SystemOneRequest, grid: MergedGrid?) throws -> PromptRow {
        var schema = encode(Self.schemaHeader)
        var raw: [(QuestionLayout, Int)] = []
        for (qi, q) in request.questions.enumerated() {
            schema += encode("\nFIELD \(qi + 1)\nID: \(q.id)\nTYPE: \(q.type)\nINSTRUCTION: ")
            let q0 = schema.count
            var instructions = q.instructions ?? .null
            if instructions.isNull || instructions == .string("") { instructions = .string(q.id) }
            schema += encode(PythonJSON.render(instructions))
            let q1 = schema.count
            schema += encode("\nALLOWED OPTIONS:\n")
            var spans: [[Int]] = []
            var oids: [String] = []
            for (oi, o) in try q.options().enumerated() {
                schema += encode("OPTION \(oi + 1): ")
                let o0 = schema.count
                var sem = [JSONMember("option_id", .string(o.id))]
                if let d = o.description { sem.append(JSONMember("description", d)) }
                schema += encode(PythonJSON.render(.object(sem)))
                spans.append([o0, schema.count])
                oids.append(o.id)
                schema += newlineIDs
            }
            schema += encode("END FIELD\n")
            raw.append((QuestionLayout(questionID: q.id, type: q.type, typeID: SystemOneRequest.questionTypes[q.type]!,
                                       questionSpan: [q0, q1], optionSpans: spans, optionIDs: oids), 0))
        }
        var prefix = prefixIDs
        var tokenOffset: Int? = nil
        if let g = grid {
            guard g.count <= nImageMax else {
                throw ClefFlashError.prompt("\(g) = \(g.count) image tokens > n_image_max \(nImageMax)")
            }
            tokenOffset = prefix.count
            prefix += [Self.visionStart] + Array(repeating: Self.imagePad, count: g.count) + [Self.visionEnd] + newlineIDs
        }
        let stateAll = encode(PythonJSON.render(request.state))
        let fixed = prefix.count + schema.count + suffixIDs.count
        guard fixed <= Self.maxLength else {
            throw ClefFlashError.prompt("schema requires \(fixed) tokens before state; maximum is \(Self.maxLength)")
        }
        let state = Array(stateAll.prefix(Self.maxLength - fixed))
        let shift = prefix.count + state.count
        let questions = raw.map { q, _ in
            QuestionLayout(questionID: q.questionID, type: q.type, typeID: q.typeID,
                           questionSpan: q.questionSpan.map { $0 + shift },
                           optionSpans: q.optionSpans.map { $0.map { $0 + shift } }, optionIDs: q.optionIDs)
        }
        let ids = prefix + state + schema + suffixIDs
        var mapped = ids.map { Int32($0) }
        var rc = [Int32](repeating: 0, count: nImageMax * 2)
        var start = Self.noShift
        var amount: Int32 = 0
        if let g = grid, let i0 = tokenOffset {
            for k in 0..<g.count {
                mapped[i0 + 1 + k] = Int32(Self.vocab + k)
                rc[2 * k] = Int32(k / g.w)
                rc[2 * k + 1] = Int32(k % g.w)
            }
            start = Int32(i0 + 1 + g.count)
            amount = Int32(g.count - max(g.h, g.w))
        }
        return PromptRow(ids: ids, decoderIDs: mapped, questions: questions, tokenOffset: tokenOffset, grid: grid,
                         imageRC: rc, ropeShiftStart: start, ropeShiftAmount: amount, stateTokens: stateAll.count,
                         stateTokensKept: state.count)
    }
}
