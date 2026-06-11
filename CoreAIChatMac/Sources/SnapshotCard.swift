// Share-card rendering of a conversation + live stats, used by the
// CHATMAC_SNAPSHOT demo hook (ImageRenderer output, no window capture).

import SwiftUI

struct SnapshotCard: View {
    let modelName: String
    let modelSize: String
    let messages: [ChatMessage]
    let stats: LiveStats

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            ForEach(messages) { message in
                cardBubble(message)
            }
            statsRow
        }
        .padding(20)
        .background(Color(nsColor: .windowBackgroundColor))
        .environment(\.colorScheme, .dark)
    }

    // Card variant of the chat bubble: thinking is rendered expanded (the
    // in-app DisclosureGroup defaults to collapsed, which hides the best part
    // of a reasoning model in a static image).
    @ViewBuilder
    private func cardBubble(_ message: ChatMessage) -> some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 80)
                Text(message.content)
                    .padding(12)
                    .background(.tint.opacity(0.85), in: RoundedRectangle(cornerRadius: 14))
                    .foregroundStyle(.white)
            }
        } else {
            VStack(alignment: .leading, spacing: 8) {
                if !message.thinking.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("Thinking", systemImage: "brain")
                            .font(.caption).foregroundStyle(.secondary)
                        Text(message.thinking)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    .padding(10)
                    .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
                }
                Text(message.content)
                    .padding(12)
                    .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
            }
            .padding(.trailing, 80)
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "cpu").foregroundStyle(.tint)
            Text(modelName).font(.headline.monospaced())
            Text(modelSize).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text("Apple Core AI · official runtime · M4 Max")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var statsRow: some View {
        HStack(spacing: 16) {
            if let load = stats.loadSeconds {
                label("clock", String(format: "load %.1fs", load))
            }
            if let ttft = stats.ttftSeconds {
                label("timer", String(format: "TTFT %.2fs", ttft))
            }
            if let tps = stats.tokensPerSecond {
                label("bolt.fill", String(format: "%.1f tok/s", tps)).fontWeight(.bold)
            }
            label("text.alignleft", "\(stats.promptTokens) in / \(stats.generatedTokens) out")
            Spacer()
            Text("github.com/john-rocky/coreai-model-zoo")
                .font(.caption).foregroundStyle(.secondary)
        }
        .font(.callout.monospacedDigit())
        .padding(.top, 4)
    }

    private func label(_ symbol: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: symbol).font(.caption).foregroundStyle(.secondary)
            Text(value)
        }
    }
}
