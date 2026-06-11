import AppKit
import SwiftUI

struct ChatView: View {
    @StateObject private var engine = ChatEngine()
    @AppStorage("modelsFolder") private var modelsFolderPath = ""
    @State private var draft = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            VStack(spacing: 0) {
                conversation
                Divider()
                inputBar
                statsBar
            }
        }
        .navigationTitle(engine.selectedModel?.name ?? "CoreAI Chat")
        .toolbar {
            ToolbarItem {
                Button {
                    engine.newChat()
                } label: {
                    Label("New Chat", systemImage: "square.and.pencil")
                }
                .disabled(engine.selectedModel == nil)
                .help("Clear the conversation (keeps the loaded model)")
            }
        }
        .onAppear {
            // Demo/testing hooks: CHATMAC_FOLDER / CHATMAC_MODEL / CHATMAC_PROMPT
            // auto-scan, auto-load, and auto-send for hands-free runs.
            if let folder = ProcessInfo.processInfo.environment["CHATMAC_FOLDER"] {
                modelsFolderPath = folder
            }
            if !modelsFolderPath.isEmpty {
                engine.scanFolder(URL(fileURLWithPath: modelsFolderPath))
            }
            if let autoModel = ProcessInfo.processInfo.environment["CHATMAC_MODEL"],
                let entry = engine.models.first(where: { $0.name == autoModel }) {
                engine.load(entry)
            }
        }
        .onChange(of: engine.status) {
            if engine.status == .ready, !autoPromptSent,
                let prompt = ProcessInfo.processInfo.environment["CHATMAC_PROMPT"] {
                autoPromptSent = true
                engine.send(prompt)
            } else if engine.status == .ready, autoPromptSent,
                let snapshotPath = ProcessInfo.processInfo.environment["CHATMAC_SNAPSHOT"] {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
                    snapshotCard(to: snapshotPath)
                }
            }
        }
    }

    @State private var autoPromptSent = false

    // Renders a clean share-card PNG of the conversation + stats via
    // ImageRenderer (no screen-recording permission needed). Demo/docs hook,
    // paired with CHATMAC_PROMPT.
    private func snapshotCard(to path: String) {
        let card = SnapshotCard(
            modelName: engine.selectedModel?.name ?? "model",
            modelSize: engine.selectedModel?.sizeLabel ?? "",
            messages: engine.messages,
            stats: engine.stats
        )
        let renderer = ImageRenderer(content: card.frame(width: 880))
        renderer.scale = 2.0
        guard let image = renderer.nsImage,
            let tiff = image.tiffRepresentation,
            let rep = NSBitmapImageRep(data: tiff),
            let data = rep.representation(using: .png, properties: [:]) else { return }
        try? data.write(to: URL(fileURLWithPath: path))
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        VStack(spacing: 0) {
            List(engine.models, selection: Binding(
                get: { engine.selectedModel },
                set: { entry in if let entry { engine.load(entry) } }
            )) { entry in
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.name).font(.system(.body, design: .monospaced)).lineLimit(1)
                    Text(entry.sizeLabel).font(.caption).foregroundStyle(.secondary)
                }
                .tag(entry)
            }
            .listStyle(.sidebar)

            Divider()
            Button {
                pickFolder()
            } label: {
                Label("Choose Models Folder…", systemImage: "folder")
                    .frame(maxWidth: .infinity)
            }
            .padding(8)
        }
        .navigationSplitViewColumnWidth(min: 220, ideal: 260)
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Scan"
        if panel.runModal() == .OK, let url = panel.url {
            modelsFolderPath = url.path
            engine.scanFolder(url)
        }
    }

    // MARK: - Conversation

    private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    if engine.messages.isEmpty {
                        emptyState
                    }
                    ForEach(engine.messages) { message in
                        MessageBubble(message: message)
                    }
                    Color.clear.frame(height: 1).id("end")
                }
                .padding()
            }
            .onChange(of: engine.messages) {
                proxy.scrollTo("end", anchor: .bottom)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "cpu").font(.system(size: 42)).foregroundStyle(.tertiary)
            Text(emptyStateText).foregroundStyle(.secondary)
        }
        .padding(.top, 120)
    }

    private var emptyStateText: String {
        switch engine.status {
        case .idle:
            return engine.models.isEmpty
                ? "Choose a folder containing exported Core AI model bundles."
                : "Select a model to load."
        case .loading: return "Loading \(engine.selectedModel?.name ?? "model")…"
        case .ready: return "Ask anything."
        case .generating: return ""
        case .error(let message): return message
        }
    }

    // MARK: - Input

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Message", text: $draft, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .padding(10)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                .focused($inputFocused)
                .onSubmit(submit)

            if engine.status == .generating {
                Button {
                    engine.stopGeneration()
                } label: {
                    Image(systemName: "stop.circle.fill").font(.title2)
                }
                .buttonStyle(.plain)
            } else {
                Button(action: submit) {
                    Image(systemName: "arrow.up.circle.fill").font(.title2)
                }
                .buttonStyle(.plain)
                .disabled(engine.status != .ready || draft.isEmpty)
            }
        }
        .padding(10)
    }

    private func submit() {
        guard engine.status == .ready else { return }
        engine.send(draft)
        draft = ""
        inputFocused = true
    }

    // MARK: - Stats footer

    private var statsBar: some View {
        HStack(spacing: 14) {
            statusChip
            if let load = engine.stats.loadSeconds {
                stat("clock", String(format: "load %.1fs", load))
            }
            if let ttft = engine.stats.ttftSeconds {
                stat("timer", String(format: "TTFT %.2fs", ttft))
            }
            if let tps = engine.stats.tokensPerSecond {
                stat("bolt.fill", String(format: "%.1f tok/s", tps))
                    .foregroundStyle(.primary)
                    .fontWeight(.semibold)
            }
            if engine.stats.promptTokens > 0 {
                stat("text.alignleft",
                     "\(engine.stats.promptTokens) in / \(engine.stats.generatedTokens) out")
            }
            if engine.stats.footprintBytes > 0 {
                stat("memorychip",
                     ByteCountFormatter.string(
                        fromByteCount: Int64(engine.stats.footprintBytes), countStyle: .memory))
            }
            Spacer()
            Text("Apple coreai-models · official runtime")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .font(.callout.monospacedDigit())
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private var statusChip: some View {
        HStack(spacing: 5) {
            Circle().frame(width: 7, height: 7).foregroundStyle(statusColor)
            Text(engine.status.label).font(.callout)
        }
    }

    private var statusColor: Color {
        switch engine.status {
        case .idle: return .gray
        case .loading: return .orange
        case .ready: return .green
        case .generating: return .blue
        case .error: return .red
        }
    }

    private func stat(_ symbol: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: symbol).font(.caption).foregroundStyle(.secondary)
            Text(value)
        }
    }
}

// MARK: - Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 80) }
            VStack(alignment: .leading, spacing: 6) {
                if !message.thinking.isEmpty {
                    DisclosureGroup {
                        Text(message.thinking)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } label: {
                        Label("Thinking", systemImage: "brain")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                if message.content.isEmpty && message.isStreaming && message.thinking.isEmpty {
                    ProgressView().controlSize(.small)
                } else {
                    Text(message.content.isEmpty && message.isStreaming ? "…" : message.content)
                        .textSelection(.enabled)
                }
            }
            .padding(12)
            .background(
                message.role == .user ? AnyShapeStyle(.tint.opacity(0.85)) : AnyShapeStyle(.quaternary.opacity(0.6)),
                in: RoundedRectangle(cornerRadius: 14)
            )
            .foregroundStyle(message.role == .user ? AnyShapeStyle(.white) : AnyShapeStyle(.primary))
            if message.role == .assistant { Spacer(minLength: 80) }
        }
    }
}
