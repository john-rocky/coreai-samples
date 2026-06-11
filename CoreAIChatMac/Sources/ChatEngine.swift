// ChatEngine — loads a Core AI language bundle with Apple's official runtime
// (CoreAILanguageModels) and streams chat completions with live performance
// stats. Works with any bundle exported by `coreai.llm.export` (gpt-oss,
// qwen3, gemma3, mistral, zoo models, ...).

import CoreAILanguageModels
import Darwin
import Foundation
import Tokenizers

struct ModelEntry: Identifiable, Hashable {
    let url: URL
    let sizeBytes: Int64

    var id: URL { url }
    var name: String { url.lastPathComponent }
    var sizeLabel: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}

struct ChatMessage: Identifiable, Equatable {
    enum Role { case user, assistant }
    let id = UUID()
    let role: Role
    var thinking: String = ""   // harmony "analysis" channel (gpt-oss)
    var content: String = ""
    var isStreaming = false
}

struct LiveStats: Equatable {
    var loadSeconds: Double?
    var promptTokens: Int = 0
    var ttftSeconds: Double?
    var generatedTokens: Int = 0
    var tokensPerSecond: Double?
    var footprintBytes: UInt64 = 0
}

@MainActor
final class ChatEngine: ObservableObject {
    @Published var models: [ModelEntry] = []
    @Published var selectedModel: ModelEntry?
    @Published var status: Status = .idle
    @Published var messages: [ChatMessage] = []
    @Published var stats = LiveStats()

    enum Status: Equatable {
        case idle, loading, ready, generating
        case error(String)

        var label: String {
            switch self {
            case .idle: return "No model"
            case .loading: return "Loading…"
            case .ready: return "Ready"
            case .generating: return "Generating…"
            case .error(let message): return "Error: \(message)"
            }
        }
    }

    private var engine: (any InferenceEngine)?
    private var tokenizer: (any Tokenizer)?
    private var generationTask: Task<Void, Never>?

    // MARK: - Model discovery

    func scanFolder(_ folder: URL) {
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
        models = entries
            .filter { fm.fileExists(atPath: $0.appendingPathComponent("metadata.json").path) }
            .map { ModelEntry(url: $0, sizeBytes: Self.directorySize($0.resolvingSymlinksInPath())) }
            .sorted { $0.sizeBytes < $1.sizeBytes }
    }

    private static func directorySize(_ url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    // MARK: - Load

    func load(_ entry: ModelEntry) {
        generationTask?.cancel()
        selectedModel = entry
        status = .loading
        messages = []
        stats = LiveStats()
        engine = nil
        tokenizer = nil

        Task {
            do {
                let bundle = try LanguageBundle(from: entry.url.path)
                let engineConfig = ModelConfig(
                    name: bundle.name,
                    tokenizer: bundle.tokenizer,
                    vocabSize: bundle.vocabSize,
                    maxContextLength: bundle.maxContextLength,
                    serializedModel: [bundle.modelAssetPath],
                    function: bundle.language.functionMap?.name(for: "main") ?? "main"
                )
                let configData = try JSONEncoder().encode(engineConfig)
                let modelURL = try bundle.requireModelURL(for: ModelBundle.ComponentKey.main)

                let start = SuspendingClock.now
                async let engineResult = EngineFactory.createEngine(
                    config: configData, modelURL: modelURL)
                async let tokenizerResult = bundle.loadTokenizer()
                let loadedEngine = try await engineResult
                let loadedTokenizer = try await tokenizerResult
                let elapsed = Self.seconds(from: start, to: .now)

                self.engine = loadedEngine
                self.tokenizer = loadedTokenizer
                self.stats.loadSeconds = elapsed
                self.stats.footprintBytes = Self.physFootprint()
                self.status = .ready
            } catch {
                self.status = .error("\(error)")
            }
        }
    }

    // MARK: - Chat

    func send(_ text: String) {
        guard let engine, let tokenizer, status == .ready else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        messages.append(ChatMessage(role: .user, content: trimmed))
        var reply = ChatMessage(role: .assistant, isStreaming: true)
        let replyID = reply.id
        messages.append(reply)
        status = .generating
        stats.ttftSeconds = nil
        stats.generatedTokens = 0
        stats.tokensPerSecond = nil

        // Full-history prompt via the bundle's own chat template (multi-turn).
        // Assistant turns feed back only the final answer (harmony convention).
        let history: [[String: any Sendable]] = messages.dropLast().map {
            ["role": $0.role == .user ? "user" : "assistant", "content": $0.content]
        }

        generationTask = Task {
            do {
                let promptTokens = try tokenizer.applyChatTemplate(messages: history)
                self.stats.promptTokens = promptTokens.count

                try await engine.reset()
                let stream = DecodingStrategyFactory.create(type: .vanilla).decode(
                    from: .tokens(promptTokens),
                    tokenizer: tokenizer,
                    inferenceEngine: engine,
                    samplingConfiguration: SamplingConfiguration(temperature: 0.7),
                    options: InferenceOptions(maxTokens: 2048, includeLogits: false),
                    stopSequences: StopSequences(for: tokenizer)
                )

                let requestStart = SuspendingClock.now
                var firstTokenAt: SuspendingClock.Instant?
                var raw = ""
                var tokenCount = 0
                // Live tok/s over a rolling window: the pipelined engine bursts
                // at decode start, so a cumulative average over-reads on short
                // replies. The window converges to steady-state.
                var window: [SuspendingClock.Instant] = []
                let windowSize = 32

                for try await chunk in stream {
                    if Task.isCancelled { break }
                    let now = SuspendingClock.now
                    if firstTokenAt == nil {
                        firstTokenAt = now
                        self.stats.ttftSeconds = Self.seconds(from: requestStart, to: now)
                    }
                    raw += chunk.text
                    tokenCount += 1
                    window.append(now)
                    if window.count > windowSize { window.removeFirst() }

                    let parsed = HarmonyParser.parse(raw)
                    reply.thinking = parsed.thinking
                    reply.content = parsed.answer
                    self.update(replyID, with: reply)

                    self.stats.generatedTokens = tokenCount
                    if window.count > 8 {
                        let span = Self.seconds(from: window.first!, to: now)
                        if span > 0 {
                            self.stats.tokensPerSecond = Double(window.count - 1) / span
                        }
                    }
                }

                reply.isStreaming = false
                self.update(replyID, with: reply)
                self.stats.footprintBytes = Self.physFootprint()
                if ProcessInfo.processInfo.environment["CHATMAC_DEBUG"] != nil {
                    print("RAW_OUTPUT_START\n\(raw)\nRAW_OUTPUT_END")
                    print("PARSED thinking=\(reply.thinking.count) answer=\(reply.content.count)")
                }
                self.status = .ready
            } catch {
                reply.isStreaming = false
                if reply.content.isEmpty { reply.content = "(generation failed: \(error))" }
                self.update(replyID, with: reply)
                self.status = .ready
            }
        }
    }

    func stopGeneration() {
        generationTask?.cancel()
    }

    // Fresh conversation on the loaded model: clears history (and with it the
    // per-turn full-history prefill depth) without reloading the engine.
    func newChat() {
        generationTask?.cancel()
        messages = []
        stats.ttftSeconds = nil
        stats.generatedTokens = 0
        stats.promptTokens = 0
        stats.tokensPerSecond = nil
        if status == .generating { status = .ready }
    }

    private func update(_ id: UUID, with message: ChatMessage) {
        if let index = messages.firstIndex(where: { $0.id == id }) {
            messages[index] = message
        }
    }

    // MARK: - Helpers

    static func seconds(from start: SuspendingClock.Instant, to end: SuspendingClock.Instant) -> Double {
        let d = end - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    static func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }
}

// Splits gpt-oss "harmony" output into the analysis channel (thinking) and the
// final channel (answer). Models without harmony markers pass through as-is.
enum HarmonyParser {
    static func parse(_ raw: String) -> (thinking: String, answer: String) {
        guard raw.contains("<|channel|>") else {
            return ("", strip(raw))
        }
        var thinking = ""
        var answer = ""
        if let analysisRange = raw.range(of: "<|channel|>analysis<|message|>") {
            let afterAnalysis = raw[analysisRange.upperBound...]
            if let end = afterAnalysis.range(of: "<|end|>") {
                thinking = String(afterAnalysis[..<end.lowerBound])
            } else {
                thinking = String(afterAnalysis)
            }
        }
        if let finalRange = raw.range(of: "<|channel|>final<|message|>") {
            answer = String(raw[finalRange.upperBound...])
        }
        return (strip(thinking), strip(answer))
    }

    private static func strip(_ text: String) -> String {
        var out = text
        for marker in ["<|return|>", "<|end|>", "<|endoftext|>"] {
            out = out.replacingOccurrences(of: marker, with: "")
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
