// DecisionFormModel — copy a customer email, and every field of a support ticket fills at once. Each field is one
// typed question (choice / score / yes-no); the eight questions and the email go to clef-flash as one SystemOne-shaped
// request, and one read of the email answers all of them with a probability for every option. Nothing is generated.

import AppKit
import ClefFlash
import CoreAI
import Foundation
import Observation

@MainActor
@Observable
final class DecisionFormModel {
    /// One form field = one typed question of the request.
    struct Field: Identifiable {
        let id: String              // the question id
        let label: String
        let question: JSONValue     // {"type", "instructions", "criteria"}
        /// What the form shows for each option id (choice) or level index (score).
        let words: [String: String]

        var type: String { question["type"]?.string ?? "" }
    }

    /// A filled field.
    struct Value {
        let text: String            // the answer as the form shows it
        let probability: Double     // the model's probability of that answer
        let level: Double?          // score: the expected level, 0 ... levels - 1
        let levels: Int
    }

    enum Phase: Equatable {
        case noModel
        case loading(since: Date)
        case warming
        case ready
        case failed(String)
    }

    // MARK: - The form: eight questions, asked together

    nonisolated static let fields: [Field] = [
        choice("team", "Team", "Which team should handle this email?", [
            ("billing", "Payments, charges and refunds", "Billing"),
            ("technical", "A product is faulty or does not work", "Technical"),
            ("shipping", "Delivery, tracking and lost parcels", "Shipping"),
            ("account", "Sign-in and profile", "Account"),
        ]),
        score("priority", "Priority", "How urgent is it?", ["Can wait", "This week", "Today", "Right now"]),
        score("mood", "Customer mood", "How does the customer feel?", ["Calm", "Annoyed", "Frustrated", "Angry"]),
        choice("next_step", "Next step", "What should support do next?", [
            ("replace", "Ship a replacement", "Send a replacement"),
            ("refund", "Refund the payment", "Refund"),
            ("track", "Chase the delivery", "Chase the delivery"),
            ("help", "Reply with a fix or an answer", "Reply with a fix"),
            ("escalate", "Pass it to a manager", "Pass to a manager"),
        ]),
        choice("needed_by", "Needed by", "By when does the customer need it?", [
            ("today", "Today", "Today"),
            ("week", "Within a week", "Within a week"),
            ("month", "Within a month", "Within a month"),
            ("none", "No date given", "No date given"),
        ]),
        yesNo("refund", "Wants money back", "The customer asks for money back."),
        yesNo("repeat", "Had this problem before", "The customer has had this problem before."),
        yesNo("cancel", "Threatens to cancel", "The customer threatens to cancel or leave."),
    ]

    nonisolated static let sampleEmail = """
        Subject: Espresso machine leaking again

        Hi, the espresso machine from order WH-58207 started leaking from the base yesterday. It's the second one that has done this.

        I'm hosting a brunch on Saturday, so please send a replacement before then. I don't want a refund, just a machine that works.

        Priya
        """

    /// Read once after loading: the process's first decision pays the runtime's warm-up.
    nonisolated static let warmUpText = "Hello, a quick question about my order."

    // MARK: - State

    var phase: Phase = .noModel
    var folder: URL?
    var loadSeconds: Double?
    var email = ""
    var values: [String: Value] = [:]
    /// Bumped once per fill, so every field flashes together.
    var fillVersion = 0
    var working = false
    var status = "Copy a customer email, then press Paste."
    var seconds: Double?
    var tokens: Int?
    private var decider: ClefDecider?

    var isReady: Bool { phase == .ready }

    // MARK: - Loading

    /// The folder is a download of the Hugging Face repo mlboydaisuke/clef-flash-CoreAI (its layout:
    /// gpu-pipelined/<bundle>/, host/lm_head_fp16.bin). A text-only form needs the fp16 decoder, the head and the table.
    func load(folder: URL) async {
        let gp = folder.appending(path: "gpu-pipelined")
        let assets = ClefDecider.Assets(
            decoderBundle: gp.appending(path: "clef_flash_decode_fp16_pf64"),
            head: gp.appending(path: "clef_flash_head_bucket_fp16w32"),
            table: folder.appending(path: "host/lm_head_fp16.bin"))
        let needed = [assets.decoderBundle.appending(path: "metadata.json"), assets.head.appending(path: "metadata.json"), assets.table]
        if let missing = needed.first(where: { !FileManager.default.fileExists(atPath: $0.path) }) {
            phase = .failed("Not a clef-flash-CoreAI folder: no \(missing.path(percentEncoded: false))")
            return
        }
        self.folder = folder
        decider = nil
        values = [:]
        let start = Date()
        phase = .loading(since: start)
        do {
            // The `.aimodel` files are specialized for this Mac by the runtime (JIT), with the flags the zoo gates used:
            // the decoder GPU-preferred with frequent reshapes, the head GPU-preferred.
            var decoderOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)
            decoderOptions.expectFrequentReshapes = true
            let gpu = SpecializationOptions(preferredComputeUnitKind: .gpu)
            let loaded = try await ClefDecider(assets: assets, decoderOptions: decoderOptions, headOptions: gpu, towerOptions: gpu)
            phase = .warming
            _ = try await loaded.decide(request: try Self.request(for: Self.warmUpText))
            decider = loaded
            loadSeconds = Date().timeIntervalSince(start)
            phase = .ready
        } catch {
            phase = .failed("Could not load clef-flash: \(error.localizedDescription)")
        }
    }

    // MARK: - The fill

    func paste() {
        guard let text = NSPasteboard.general.string(forType: .string)?.trimmingCharacters(in: .whitespacesAndNewlines),
            !text.isEmpty
        else {
            status = "The clipboard holds no text."
            return
        }
        Task { await fill(from: text) }
    }

    func useSample() {
        Task { await fill(from: Self.sampleEmail) }
    }

    func clear() {
        email = ""
        values = [:]
        seconds = nil
        tokens = nil
        fillVersion += 1
        status = "Copy a customer email, then press Paste."
    }

    func fill(from text: String) async {
        guard let decider, !working else { return }
        working = true
        defer { working = false }
        email = text
        values = [:]
        seconds = nil
        fillVersion += 1
        status = "Reading the email…"
        do {
            let start = ContinuousClock.now
            let response = try await decider.decide(request: try Self.request(for: text))
            let elapsed = start.duration(to: .now)
            let s = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
            values = Self.values(from: response)
            seconds = s
            tokens = response["usage"]?["input_tokens"]?.intValue
            fillVersion += 1
            status = "\(values.count) of \(Self.fields.count) fields · \(Self.fields.count) decisions · \(String(format: "%.2f", s)) s"
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Request and response

    /// {"model", "state": the email, "questions": {id: question}} — the SystemOne-compatible request shape.
    nonisolated static func request(for text: String) throws -> SystemOneRequest {
        try SystemOneRequest(json: .obj([
            ("model", .string("clef-flash")),
            ("state", .string(text)),
            ("questions", .obj(fields.map { ($0.id, $0.question) })),
        ]))
    }

    /// The response's answers as the form shows them.
    nonisolated static func values(from response: JSONValue) -> [String: Value] {
        var out: [String: Value] = [:]
        for field in fields {
            guard let a = response["answers"]?[field.id] else { continue }
            switch field.type {
            case "choice":
                guard let id = a["choice"]?.string, let p = a["confidence"]?.double else { continue }
                out[field.id] = Value(text: field.words[id] ?? id, probability: p, level: nil, levels: 0)
            case "score":
                guard let probs = a["probabilities"]?.members, let best = probs.max(by: { ($0.value.double ?? 0) < ($1.value.double ?? 0) })
                else { continue }
                out[field.id] = Value(text: field.words[best.key] ?? best.key, probability: best.value.double ?? 0,
                                      level: a["score"]?.double, levels: probs.count)
            default:
                guard let yes = a["noul"]?.double else { continue }
                out[field.id] = Value(text: yes >= 0.5 ? "Yes" : "No", probability: yes >= 0.5 ? yes : 1 - yes,
                                      level: nil, levels: 0)
            }
        }
        return out
    }

    // MARK: - Building the questions

    /// A choice: (option id, the description the model reads, the word the form shows).
    nonisolated private static func choice(_ id: String, _ label: String, _ instructions: String,
                                           _ options: [(String, String, String)]) -> Field {
        Field(id: id, label: label,
              question: .obj([("type", .string("choice")), ("instructions", .string(instructions)),
                              ("criteria", .obj(options.map { ($0.0, .string($0.1)) }))]),
              words: Dictionary(uniqueKeysWithValues: options.map { ($0.0, $0.2) }))
    }

    /// A score: ordered levels, read back as the expected level and the most probable one.
    nonisolated private static func score(_ id: String, _ label: String, _ instructions: String, _ levels: [String]) -> Field {
        Field(id: id, label: label,
              question: .obj([("type", .string("score")), ("instructions", .string(instructions)),
                              ("criteria", .array(levels.map { .string($0) }))]),
              words: Dictionary(uniqueKeysWithValues: levels.enumerated().map { (String($0.offset), $0.element) }))
    }

    /// A yes-no question (noul): the response carries p(true).
    nonisolated private static func yesNo(_ id: String, _ label: String, _ statement: String) -> Field {
        Field(id: id, label: label, question: .obj([("type", .string("noul")), ("instructions", .string(statement))]), words: [:])
    }
}
