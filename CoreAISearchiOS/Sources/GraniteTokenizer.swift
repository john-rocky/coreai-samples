import Foundation

/// Independent tokenizer for the pinned Granite 97M R2 raw-text contract.
/// No prompts, trimming, Unicode normalization, HF package, or model execution.
final class GraniteTokenizer {
    enum TokenizerError: Error, CustomStringConvertible {
        case unsupported(String)
        case missingToken(String)
        var description: String {
            switch self {
            case .unsupported(let message): return "Unsupported tokenizer: \(message)"
            case .missingToken(let token): return "Missing BPE token: \(token)"
            }
        }
    }

    private struct AddedToken: Decodable {
        let id: Int32
        let content: String
        let single_word: Bool
        let lstrip: Bool
        let rstrip: Bool
        let normalized: Bool
    }
    private struct Model: Decodable {
        let type: String
        let ignore_merges: Bool
        let byte_fallback: Bool
        let dropout: Double?
        let unk_token: String?
        let continuing_subword_prefix: String?
        let end_of_word_suffix: String?
        let vocab: [String: Int32]
        let merges: [[String]]
    }
    private struct TokenizerData: Decodable {
        let model: Model
        let added_tokens: [AddedToken]
    }
    private let vocab: [String: Int32]
    private let ranks: [String: Int]
    private let pattern: NSRegularExpression
    private let addedPattern: NSRegularExpression
    private let wordStart: NSRegularExpression
    private let wordEnd: NSRegularExpression
    private let added: [String: AddedToken]
    private let byteAlphabet: [String]
    private var cache: [String: [Int32]] = [:]
    private let clsID: Int32 = 179934
    private let sepID: Int32 = 179938
    private let padID: Int32 = 179935

    init(tokenizerURL: URL, tokenizerConfigURL: URL) throws {
        let bytes = try Data(contentsOf: tokenizerURL)
        let typed = try JSONDecoder().decode(TokenizerData.self, from: bytes)
        guard let root = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              root["normalizer"] is NSNull,
              let pre = root["pre_tokenizer"] as? [String: Any],
              pre["type"] as? String == "Sequence",
              let steps = pre["pretokenizers"] as? [[String: Any]], steps.count == 2,
              steps[0]["type"] as? String == "Split",
              steps[0]["behavior"] as? String == "Isolated",
              steps[0]["invert"] as? Bool == false,
              let regexPattern = steps[0]["pattern"] as? [String: String],
              regexPattern.count == 1, let expression = regexPattern["Regex"],
              steps[1]["type"] as? String == "ByteLevel",
              steps[1]["add_prefix_space"] as? Bool == false,
              steps[1]["use_regex"] as? Bool == false
        else { throw TokenizerError.unsupported("normalizer or pretokenizer") }
        let model = typed.model
        guard model.type == "BPE", model.ignore_merges, !model.byte_fallback,
              model.dropout == nil, model.unk_token == nil,
              model.continuing_subword_prefix == nil, model.end_of_word_suffix == nil
        else { throw TokenizerError.unsupported("BPE configuration") }
        guard let post = root["post_processor"] as? [String: Any],
              post["type"] as? String == "TemplateProcessing",
              let single = post["single"] as? [[String: Any]], single.count == 3,
              (single[0]["SpecialToken"] as? [String: Any])?["id"] as? String == "<|startoftext|>",
              (single[1]["Sequence"] as? [String: Any])?["id"] as? String == "A",
              (single[2]["SpecialToken"] as? [String: Any])?["id"] as? String == "<|return|>"
        else { throw TokenizerError.unsupported("single-text template") }
        let configBytes = try Data(contentsOf: tokenizerConfigURL)
        guard let config = try JSONSerialization.jsonObject(with: configBytes) as? [String: Any],
              (config["padding_side"] as? String ?? "right") == "right",
              (config["truncation_side"] as? String ?? "right") == "right",
              config["cls_token"] as? String == "<|startoftext|>",
              config["sep_token"] as? String == "<|return|>",
              config["pad_token"] as? String == "<|endoftext|>"
        else { throw TokenizerError.unsupported("special tokens/padding/truncation") }

        var added: [String: AddedToken] = [:]
        for token in typed.added_tokens {
            guard !token.lstrip, !token.rstrip, !token.normalized, !token.content.isEmpty
            else { throw TokenizerError.unsupported("added-token flags") }
            added[token.content] = token
        }
        guard added["<|startoftext|>"]?.id == 179934,
              added["<|return|>"]?.id == 179938,
              added["<|endoftext|>"]?.id == 179935
        else { throw TokenizerError.unsupported("special IDs") }
        self.added = added
        self.vocab = model.vocab
        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(model.merges.count)
        for (rank, pair) in model.merges.enumerated() {
            guard pair.count == 2, !pair[0].contains("\0"), !pair[1].contains("\0")
            else { throw TokenizerError.unsupported("merge pair") }
            ranks[pair[0] + "\0" + pair[1]] = rank
        }
        self.ranks = ranks
        self.pattern = try NSRegularExpression(pattern: expression)
        let addedExpression = added.keys.sorted {
            $0.utf8.count == $1.utf8.count ? $0 < $1 : $0.utf8.count > $1.utf8.count
        }.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|")
        self.addedPattern = try NSRegularExpression(pattern: addedExpression)
        self.wordStart = try NSRegularExpression(pattern: #"^\w"#)
        self.wordEnd = try NSRegularExpression(pattern: #"\w$"#)

        let visible = Set(Array(33...126) + Array(161...172) + Array(174...255))
        var extra = 0
        self.byteAlphabet = (0...255).map { byte in
            if visible.contains(byte) { return String(UnicodeScalar(byte)!) }
            defer { extra += 1 }
            return String(UnicodeScalar(256 + extra)!)
        }
    }

    private func bpe(_ token: String) throws -> [Int32] {
        if let existing = cache[token] { return existing }
        // ignore_merges gives a vocabulary hit precedence over applying merges.
        if let id = vocab[token] { return [id] }
        // HF BPE omits absent initial symbols if UNK and byte fallback are both
        // unset. This vocabulary lacks some controls, including mapped NUL.
        var symbols = token.unicodeScalars.map(String.init).filter { vocab[$0] != nil }
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex: Int?
            for index in 0..<(symbols.count - 1) {
                if let rank = ranks[symbols[index] + "\0" + symbols[index + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            guard let index = bestIndex else { break }
            symbols[index] += symbols[index + 1]
            symbols.remove(at: index + 1)
        }
        let ids = try symbols.map { symbol -> Int32 in
            guard let id = vocab[symbol] else { throw TokenizerError.missingToken(symbol) }
            return id
        }
        if cache.count < 32768 { cache[token] = ids }
        return ids
    }

    private func ordinary(_ text: String) throws -> [Int32] {
        let nsText = text as NSString
        let full = NSRange(location: 0, length: nsText.length)
        var ids: [Int32] = []
        var cursor = 0
        func appendPiece(_ range: NSRange) throws {
            guard range.length > 0 else { return }
            let raw = nsText.substring(with: range)
            let encoded = raw.utf8.map { byteAlphabet[Int($0)] }.joined()
            ids.append(contentsOf: try bpe(encoded))
        }
        for match in pattern.matches(in: text, range: full) {
            if match.range.location > cursor {
                try appendPiece(NSRange(location: cursor, length: match.range.location - cursor))
            }
            try appendPiece(match.range)
            cursor = NSMaxRange(match.range)
        }
        if cursor < nsText.length {
            try appendPiece(NSRange(location: cursor, length: nsText.length - cursor))
        }
        return ids
    }

    func encodeBody(text: String) throws -> [Int32] {
        let nsText = text as NSString
        let full = NSRange(location: 0, length: nsText.length)
        var ids: [Int32] = []
        var cursor = 0
        for match in addedPattern.matches(in: text, range: full) {
            let content = nsText.substring(with: match.range)
            guard let token = added[content] else { throw TokenizerError.missingToken(content) }
            if token.single_word {
                let before = nsText.substring(to: match.range.location)
                let after = nsText.substring(from: NSMaxRange(match.range))
                let leftWord = wordEnd.firstMatch(in: before, range: NSRange(location: 0, length: (before as NSString).length)) != nil
                let rightWord = wordStart.firstMatch(in: after, range: NSRange(location: 0, length: (after as NSString).length)) != nil
                if leftWord || rightWord { continue }
            }
            ids.append(contentsOf: try ordinary(nsText.substring(with: NSRange(location: cursor, length: match.range.location - cursor))))
            ids.append(token.id)
            cursor = NSMaxRange(match.range)
        }
        ids.append(contentsOf: try ordinary(nsText.substring(from: cursor)))
        return ids
    }

    func encode(text: String, sequenceLength: Int) throws -> (inputIDs: [Int32], attentionMask: [Int32]) {
        guard sequenceLength >= 2 else { throw TokenizerError.unsupported("sequenceLength must be >=2") }
        let body = try encodeBody(text: text)
        var ids = [clsID] + Array(body.prefix(sequenceLength - 2)) + [sepID]
        var mask = Array(repeating: Int32(1), count: ids.count)
        let padding = sequenceLength - ids.count
        ids.append(contentsOf: repeatElement(padID, count: padding))
        mask.append(contentsOf: repeatElement(Int32(0), count: padding))
        return (ids, mask)
    }
}

#if GRANITE_TOKENIZER_CLI
@main
enum GraniteTokenizerCLI {
    struct Request: Decodable { let id: String; let text: String; let sequence_length: Int }
    struct Response: Encodable { let id: String; let input_ids: [Int32]; let attention_mask: [Int32] }
    static func main() throws {
        guard CommandLine.arguments.count == 4 else {
            throw GraniteTokenizer.TokenizerError.unsupported("usage: tokenizer tokenizer.json tokenizer_config.json requests.json")
        }
        let tokenizer = try GraniteTokenizer(
            tokenizerURL: URL(fileURLWithPath: CommandLine.arguments[1]),
            tokenizerConfigURL: URL(fileURLWithPath: CommandLine.arguments[2]))
        let requests = try JSONDecoder().decode([Request].self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[3])))
        let responses = try requests.map { request -> Response in
            let result = try tokenizer.encode(text: request.text, sequenceLength: request.sequence_length)
            return Response(id: request.id, input_ids: result.inputIDs, attention_mask: result.attentionMask)
        }
        FileHandle.standardOutput.write(try JSONEncoder().encode(responses))
    }
}
#endif
