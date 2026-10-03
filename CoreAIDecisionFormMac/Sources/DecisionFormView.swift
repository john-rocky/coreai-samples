import AppKit
import SwiftUI

struct DecisionFormView: View {
    @Environment(Autoplay.self) private var autoplay
    @State private var model = DecisionFormModel()
    @AppStorage("modelsFolder") private var modelsFolderPath = ""

    var body: some View {
        VStack(spacing: 16) {
            header
            controls
            emailBox
            form
            footer
        }
        .padding(22)
        .task {
            if let folder = startFolder { await model.load(folder: folder) }
            await autoplay.run(model: model) { model.useSample() }
        }
    }

    // MARK: - Header and controls

    private var header: some View {
        VStack(spacing: 4) {
            Text("Support ticket autofill").font(.system(size: 30, weight: .bold))
            Text("Copy a customer email — every field fills at once, on device")
                .font(.system(size: 17)).foregroundStyle(.secondary)
        }
    }

    private var controls: some View {
        HStack(spacing: 10) {
            Button("Paste") { model.paste() }
                .buttonStyle(.borderedProminent)
                .disabled(!model.isReady || model.working)
            Button("Sample email") { model.useSample() }
                .disabled(!model.isReady || model.working)
            Button("Clear") { model.clear() }
            Spacer()
            Text(model.status)
                .font(.system(size: 19, weight: model.seconds == nil ? .regular : .semibold).monospacedDigit())
                .foregroundStyle(model.seconds == nil ? .secondary : .primary)
        }
        .controlSize(.large)
    }

    // MARK: - The email

    private var emailBox: some View {
        ScrollView {
            Group {
                if model.email.isEmpty {
                    Text("The email you paste appears here.").foregroundStyle(.tertiary)
                } else {
                    Text(emailText).textSelection(.enabled)
                }
            }
            .font(.system(size: 18))
            .lineSpacing(2)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
        }
        .frame(height: 250)
        .background(RoundedRectangle(cornerRadius: 12).fill(.background))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary))
    }

    /// The email as pasted, its subject line in bold.
    private var emailText: AttributedString {
        var lines = model.email.components(separatedBy: "\n")
        guard let first = lines.first, first.hasPrefix("Subject:") else { return AttributedString(model.email) }
        lines.removeFirst()
        var subject = AttributedString(first)
        subject.font = .system(size: 21, weight: .bold)
        return subject + AttributedString("\n" + lines.joined(separator: "\n"))
    }

    // MARK: - The form

    @ViewBuilder
    private var form: some View {
        switch model.phase {
        case .ready:
            Grid(horizontalSpacing: 12, verticalSpacing: 12) {
                ForEach(0..<DecisionFormModel.fields.count / 2, id: \.self) { row in
                    GridRow {
                        ForEach(DecisionFormModel.fields[(row * 2)..<(row * 2 + 2)]) { field in
                            FieldCell(field: field, value: model.values[field.id], version: model.fillVersion)
                        }
                    }
                }
            }
        case .loading(let since):
            TimelineView(.periodic(from: since, by: 1)) { context in
                notice("Loading clef-flash… \(Int(context.date.timeIntervalSince(since))) s",
                       "The first load prepares the model for this Mac and writes about 30 GB of cache. Later loads read that cache.")
            }
        case .warming:
            notice("Warming up…", "One decision on a short text, so the first real one runs at full speed.")
        case .noModel:
            notice("No model loaded", "Choose the clef-flash-CoreAI folder (a download of the Hugging Face repo).")
        case .failed(let message):
            notice("Not loaded", message)
        }
    }

    private func notice(_ title: String, _ detail: String) -> some View {
        VStack(spacing: 8) {
            Text(title).font(.system(size: 22, weight: .semibold).monospacedDigit())
            Text(detail).font(.system(size: 15)).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, minHeight: 380)
    }

    // MARK: - Footer: the model and where it came from

    private var footer: some View {
        HStack(spacing: 10) {
            Text(footerText).font(.system(size: 13).monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
            Spacer()
            Button("Choose Models Folder…", action: pickFolder)
                .disabled(isLoading)
        }
    }

    private var footerText: String {
        var parts = ["clef-flash 9B · fp16 · Core AI, Mac GPU"]
        if let tokens = model.tokens { parts.append("\(tokens) tokens, one read") }
        return parts.joined(separator: " · ")
    }

    private var isLoading: Bool {
        switch model.phase {
        case .loading, .warming: return true
        default: return false
        }
    }

    /// The saved folder, else ~/Downloads/clef-flash-CoreAI when it exists.
    private var startFolder: URL? {
        if !modelsFolderPath.isEmpty { return URL(filePath: modelsFolderPath, directoryHint: .isDirectory) }
        let downloads = URL.downloadsDirectory.appending(path: "clef-flash-CoreAI", directoryHint: .isDirectory)
        return FileManager.default.fileExists(atPath: downloads.path) ? downloads : nil
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Load"
        panel.message = "Choose the clef-flash-CoreAI folder (it holds gpu-pipelined/ and host/)."
        if panel.runModal() == .OK, let url = panel.url {
            modelsFolderPath = url.path
            Task { await model.load(folder: url) }
        }
    }
}

/// One field: its label, the answer, and how sure the model was.
private struct FieldCell: View {
    let field: DecisionFormModel.Field
    let value: DecisionFormModel.Value?
    let version: Int
    @State private var flash = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(field.label).font(.system(size: 16)).foregroundStyle(.secondary)
            Text(value?.text ?? "—")
                .font(.system(size: 30, weight: .semibold))
                .foregroundStyle(value == nil ? .tertiary : .primary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            HStack(spacing: 8) {
                if let value, let level = value.level, value.levels > 1 {
                    LevelBar(level: level, levels: value.levels)
                }
                Text(value.map { "p \(String(format: "%.2f", $0.probability))" } ?? " ")
                    .font(.system(size: 14).monospacedDigit()).foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12).fill(flash ? Color.green.opacity(0.22) : Color.secondary.opacity(0.07)))
        .animation(.easeOut(duration: 1.2), value: flash)
        .onChange(of: version) {
            guard value != nil else { return }
            flash = true
            Task {
                try? await Task.sleep(for: .milliseconds(150))
                flash = false
            }
        }
    }
}

/// A score's expected level on its scale: segments up to the level, the last one partly filled.
private struct LevelBar: View {
    let level: Double
    let levels: Int

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<levels, id: \.self) { k in
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2).fill(.quaternary)
                        RoundedRectangle(cornerRadius: 2).fill(Color.accentColor)
                            .frame(width: geo.size.width * min(1, max(0, level + 1 - Double(k))))
                    }
                }
                .frame(width: 26, height: 8)
            }
        }
    }
}
