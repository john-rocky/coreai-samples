import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject var engine: DiffusionEngine
    @State private var choosingFolder = false

    var body: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    group("Model") { modelControls }
                    group("Prompt") { promptField }
                    group("Settings") { settingsControls }
                    generateButton
                    if let notice = engine.notice {
                        Text(notice)
                            .font(.callout).foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(18)
            }
            .frame(minWidth: 320, idealWidth: 350, maxWidth: 440)
            canvas.frame(minWidth: 460)
        }
        .fileImporter(isPresented: $choosingFolder, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { engine.loadLocal(url) }
        }
    }

    // MARK: - Controls

    @ViewBuilder private var modelControls: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("Model", selection: $engine.selectedID) {
                ForEach(ModelFiles.catalog) { model in
                    Text(model.title).tag(model.id)
                }
            }
            .pickerStyle(.radioGroup)
            .labelsHidden()
            .disabled(engine.status.isBusy)
            Text(modelSource)
                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
        }
        HStack {
            Button { engine.downloadAndLoad() } label: {
                Label("Download & Load", systemImage: "arrow.down.circle")
            }
            Button { choosingFolder = true } label: {
                Label("Local…", systemImage: "folder")
            }
        }
        .disabled(engine.status.isBusy)
        statusLine
    }

    /// Where the model comes from: the Hugging Face folder, or the folder picked with Local….
    private var modelSource: String {
        let model = engine.selected
        if let folder = engine.modelFolder, folder.standardizedFileURL != model.root.standardizedFileURL {
            return "Folder: \(folder.path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))"
        }
        return "Hugging Face: \(model.repo)/\(model.folder), \(gigabytes(model.totalBytes)) GB"
    }

    private var promptField: some View {
        TextField("Describe the image you want", text: $engine.prompt, axis: .vertical)
            .lineLimit(3...8)
            .onSubmit { engine.generate() }
    }

    @ViewBuilder private var settingsControls: some View {
        Stepper("Steps: \(engine.steps)", value: $engine.steps, in: 1...50)
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Guidance")
                Spacer()
                Text(String(format: "%.1f", engine.guidance)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: $engine.guidance, in: 1...10, step: 0.5)
            Text("1.0: one pass per step (the model is guidance-distilled). Above 1.0: two passes per step.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        HStack {
            Text("Seed")
            TextField("seed", text: $engine.seedText)
                .multilineTextAlignment(.trailing)
                .monospacedDigit()
            Button { engine.seedText = String(UInt32.random(in: 0 ... .max)) } label: {
                Image(systemName: "die.face.5")
            }
            .buttonStyle(.borderless)
            .help("Random seed")
        }
    }

    private var statusLine: some View {
        VStack(alignment: .leading, spacing: 6) {
            if case .downloading(let done, let total) = engine.status {
                ProgressView(value: Double(done), total: Double(max(total, 1)))
            }
            HStack(alignment: .top, spacing: 8) {
                if engine.status.isBusy { ProgressView().controlSize(.small) }
                Text(statusText)
                    .font(.caption).foregroundStyle(statusColor)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
        }
    }

    private var statusText: String {
        switch engine.status {
        case .idle: "No model loaded"
        case .downloading(let done, let total):
            "Downloading… \(gigabytes(done)) of \(gigabytes(total)) GB (\(done * 100 / max(total, 1)) %)"
        case .loading: "Loading the model…"
        case .ready: "Ready"
        case .generating(let step, let total): step == 0 ? "Reading the prompt…" : "Step \(step) of \(total)"
        case .stopping: "Stopping after this step…"
        case .failed(let message): message
        }
    }

    private var statusColor: Color {
        switch engine.status {
        case .failed: .red
        case .ready: .green
        default: .secondary
        }
    }

    private var generateButton: some View {
        Group {
            switch engine.status {
            case .generating, .stopping:
                Button(role: .destructive) { engine.stop() } label: {
                    Label("Stop", systemImage: "stop.fill").frame(maxWidth: .infinity)
                }
                .disabled(engine.status == .stopping)
            default:
                Button { engine.generate() } label: {
                    Label("Generate", systemImage: "sparkles").frame(maxWidth: .infinity)
                }
                .disabled(engine.status != .ready)
            }
        }
        .controlSize(.large)
        .buttonStyle(.borderedProminent)
    }

    // MARK: - Canvas

    private var canvas: some View {
        ZStack {
            Color(white: 0.09)
            if let image = engine.image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .padding(12)
            } else {
                placeholder
            }
            if case .generating(let step, let total) = engine.status {
                VStack {
                    Spacer()
                    ProgressView(value: Double(step), total: Double(max(total, 1))) {
                        Text(statusText).font(.caption)
                    }
                    .padding(12)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
                    .padding(20)
                }
            }
        }
        .overlay(alignment: .topTrailing) {
            if engine.image != nil {
                HStack(spacing: 10) {
                    Text(engine.imageCaption).font(.caption).foregroundStyle(.white.opacity(0.7))
                    Button("Save…", action: save)
                }
                .padding(10)
            }
        }
    }

    private var placeholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "photo.artframe")
                .font(.system(size: 46)).foregroundStyle(.tertiary)
            Text(placeholderText)
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
    }

    private var placeholderText: String {
        switch engine.status {
        case .idle:
            "Press Download & Load to get the model (\(gigabytes(engine.selected.totalBytes)) GB, once), or Local… to open a bundle you exported."
        case .downloading:
            "Downloading the model from Hugging Face. It stays on this Mac; later launches skip the download."
        case .loading:
            "Loading the model into Core AI. The first load on this Mac also specializes each model for it."
        case .ready: "Type a prompt and press Generate."
        case .failed: "Press Download & Load to try again, or Local… to choose another folder."
        case .generating, .stopping: ""
        }
    }

    @ViewBuilder
    private func group<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            content()
        }
    }

    private func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.2f", Double(bytes) / 1e9)
    }

    private func save() {
        guard let image = engine.image else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "flux2-klein-seed\(engine.lastSeed).png"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try PNG.write(image, to: url)
        } catch {
            engine.show(notice: "Could not save the image: \(error.localizedDescription)")
        }
    }
}
