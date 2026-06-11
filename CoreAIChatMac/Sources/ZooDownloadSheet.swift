// In-app download of community zoo bundles from Hugging Face (same
// ModelDownloader as the zoo's iOS apps). Official-recipe models aren't
// hosted as bundles — export those locally with `coreai.llm.export`.

import SwiftUI

struct ZooModel: Identifiable {
    let name: String
    let detail: String
    let repo: String
    let remote: String
    var id: String { repo + remote }
    var localName: String { String(remote.split(separator: "/").last ?? "model") }
}

enum ZooCatalog {
    static let models: [ZooModel] = [
        ZooModel(
            name: "Qwen3.5 0.8B (int8, fast head)",
            detail: "~0.9 GB · ~210 tok/s on M4 Max",
            repo: "mlboydaisuke/qwen3.5-0.8B-CoreAI",
            remote: "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_perchan_sym"),
        ZooModel(
            name: "LFM2.5 1.2B (int8, fast head)",
            detail: "~1.3 GB · ~275 tok/s on M4 Max",
            repo: "mlboydaisuke/LFM2.5-1.2B-CoreAI",
            remote: "gpu-pipelined/lfm2_5_1_2b_instruct_decode_int8hu_block32_sym"),
        ZooModel(
            name: "Granite 4.0-h 1B (int8, fast head)",
            detail: "~1.2 GB · Mamba2/SSM hybrid",
            repo: "mlboydaisuke/granite-4.0-h-CoreAI",
            remote: "gpu-pipelined/granite_4_0_h_1b_decode_int8hu_block32_sym"),
        ZooModel(
            name: "Granite 4.0-h 350M (fp16)",
            detail: "~0.8 GB · smallest, quick test",
            repo: "mlboydaisuke/granite-4.0-h-CoreAI",
            remote: "gpu-pipelined/granite_4_0_h_350m_decode_fp16"),
        ZooModel(
            name: "Qwen3.5 2B (int8, fast head)",
            detail: "~2 GB · ~160 tok/s on M4 Max",
            repo: "mlboydaisuke/qwen3.5-2B-CoreAI",
            remote: "gpu-pipelined/qwen3_5_2b_decode_int8hu_perchan_sym"),
    ]
}

struct ZooDownloadSheet: View {
    let modelsFolder: URL
    let onDownloaded: () -> Void
    @Environment(\.dismiss) private var dismiss
    @StateObject private var downloader = ModelDownloader()
    @State private var activeID: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Community zoo models").font(.headline)
            Text("Downloads from Hugging Face into \(modelsFolder.path). Official-recipe models aren't hosted — export those with `coreai.llm.export`.")
                .font(.caption).foregroundStyle(.secondary)

            #if !ZOO_PATCHED
            Label(
                "These models use extended state (SSM/GDN hybrids) and need the zoo engine patches: run ./zoo/setup-zoo.sh once, then rebuild with project-zoo.yml.",
                systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
            #endif

            List(ZooCatalog.models) { model in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.name)
                        Text(model.detail).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if FileManager.default.fileExists(
                        atPath: modelsFolder.appendingPathComponent(model.localName).path) {
                        Label("Installed", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green).labelStyle(.iconOnly)
                    } else if activeID == model.id, downloader.busy {
                        ProgressView(value: downloader.fraction).frame(width: 90)
                    } else {
                        Button("Get") { download(model) }
                            .disabled(downloader.busy)
                    }
                }
            }
            .frame(minHeight: 220)

            if case .failed(let message) = downloader.phase {
                Text(message).font(.caption).foregroundStyle(.red)
            } else if downloader.busy {
                Text(downloader.detail).font(.caption).foregroundStyle(.secondary)
            }

            HStack {
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(width: 480)
    }

    private func download(_ model: ZooModel) {
        activeID = model.id
        Task {
            await downloader.fetch(
                repo: model.repo,
                items: [.init(remote: model.remote, local: model.localName)],
                into: modelsFolder)
            onDownloaded()
        }
    }
}
