// ModelFiles — the models the Model picker offers. Each one is a folder of a Hugging Face repo pinned
// to one revision, the files text-to-image needs, and the engine that runs them. Adding a model is one
// entry in `catalog` plus one type that conforms to ImageModelEngine.

import Foundation
import Hub

/// One bundle this app can download: a folder of a Hugging Face repo, pinned to one revision.
struct ModelFiles: Identifiable, Sendable {
    /// Whether the model's images carry an alpha channel.
    enum ImageKind: Sendable { case rgb, rgba }

    /// Short name, also the value of `-model` in hands-off runs.
    let id: String
    let title: String
    let repo: String
    let revision: String
    /// The folder of the repo that holds the bundle ("" = the top of the repo).
    let folder: String
    /// What text-to-image needs, with each file's size at `revision`, in download order.
    let files: [(path: String, bytes: Int64)]
    let kind: ImageKind
    /// Denoising steps when the model fixes them; nil = the Steps field decides.
    let fixedSteps: Int?
    /// The square sizes the Size picker offers (empty = the bundle decides), and the one it starts on.
    let sizes: [Int]
    let defaultSize: Int?
    /// Whether the Guidance slider applies.
    let usesGuidance: Bool
    /// What Core AI's cache (~/Library/Caches/coreai-cache/) gained on the first load of these files on an
    /// M4 Max; Download & Load wants this much free disk space on top of the files.
    let firstLoadCacheBytes: Int64
    /// A line under the Prompt field: the prompt form the model wants.
    let promptNote: String?
    /// The weights' licence, when it limits what the images may be used for.
    let licenceNote: String?
    /// Makes the engine that loads and runs this bundle.
    let makeEngine: @Sendable () -> any ImageModelEngine

    init(id: String, title: String, repo: String, revision: String, folder: String,
         files: [(path: String, bytes: Int64)], kind: ImageKind = .rgb, fixedSteps: Int? = nil,
         sizes: [Int] = [], defaultSize: Int? = nil, usesGuidance: Bool = true, firstLoadCacheBytes: Int64,
         promptNote: String? = nil, licenceNote: String? = nil,
         makeEngine: @escaping @Sendable () -> any ImageModelEngine = { FluxEngine() }) {
        self.id = id
        self.title = title
        self.repo = repo
        self.revision = revision
        self.folder = folder
        self.files = files
        self.kind = kind
        self.fixedSteps = fixedSteps
        self.sizes = sizes
        self.defaultSize = defaultSize
        self.usesGuidance = usesGuidance
        self.firstLoadCacheBytes = firstLoadCacheBytes
        self.promptNote = promptNote
        self.licenceNote = licenceNote
        self.makeEngine = makeEngine
    }

    var totalBytes: Int64 { files.reduce(0) { $0 + $1.bytes } }

    /// The files can be downloaded again, so they live in Caches.
    static let downloadBase = URL.cachesDirectory.appending(path: "CoreAIImageGenMac/huggingface")
    /// The bundle folder on disk, the one the engine opens.
    var root: URL {
        HubApi(downloadBase: Self.downloadBase).localRepoLocation(HubApi.Repo(id: repo)).appending(path: folder)
    }

    var isOnDisk: Bool { files.allSatisfy { size(of: $0.path) == $0.bytes } }
    var bytesOnDisk: Int64 { files.reduce(0) { $0 + min(size(of: $1.path) ?? 0, $1.bytes) } }

    private func size(of path: String) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: root.appending(path: path).path)[.size] as? NSNumber)?.int64Value
    }

    /// Downloads the files that are not on disk yet with swift-transformers' `HubApi`, one file at
    /// a time, so a failed download keeps the files that finished and the next try fetches only
    /// the rest. `endpoint` replaces https://huggingface.co (nil = the default). `progress` gets
    /// the bytes done so far.
    func download(endpoint: String?, progress: @escaping @Sendable (Int64) -> Void) async throws {
        // cache: nil = one copy of each file, in downloadBase (no second copy in the Hub cache).
        // useOfflineMode: false = no network gives the network's own error, not a missing-file one.
        let hub = HubApi(downloadBase: Self.downloadBase, cache: nil, endpoint: endpoint, useOfflineMode: false)
        var done: Int64 = 0
        for (index, file) in files.enumerated() {
            if size(of: file.path) != file.bytes {
                let before = done
                let start = ContinuousClock.now
                Telemetry.line("DOWNLOAD file \(index + 1)/\(files.count) start path=\(file.path) bytes=\(file.bytes)")
                let pattern = folder.isEmpty ? file.path : "\(folder)/\(file.path)"
                try await hub.snapshot(from: HubApi.Repo(id: repo), revision: revision, matching: [pattern]) { @Sendable fraction in
                    progress(before + Int64(fraction.fractionCompleted * Double(file.bytes)))
                }
                try Task.checkCancellation()
                guard size(of: file.path) == file.bytes else {
                    throw ImageGenError("\(file.path) did not download completely.")
                }
                Telemetry.line("DOWNLOAD file \(index + 1)/\(files.count) done path=\(file.path) seconds=\(start.duration(to: .now).seconds)")
            }
            done += file.bytes
            progress(done)
        }
    }
}

extension ModelFiles {
    /// The two folders of mlboydaisuke/FLUX.2-klein-4B-CoreAI (exported with the apple/coreai-models
    /// commit that project.yml pins; the first is the default), then the two choices of DiT in
    /// mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI.
    static let catalog: [ModelFiles] = [
        ModelFiles(
            id: "fp16", title: "FLUX.2 klein 4B (fp16, 14.1 GB)",
            repo: "mlboydaisuke/FLUX.2-klein-4B-CoreAI",
            revision: "039db98cbf1247680ea5e76048f87b3ce061e6d6", folder: "macos-fp16",
            files: [
                ("metadata.json", 974),
                ("tokenizer/chat_template.jinja", 4_168),
                ("tokenizer/config.json", 375),
                ("tokenizer/tokenizer.json", 11_422_650),
                ("tokenizer/tokenizer_config.json", 375),
                ("vae_bn_mean.npy", 640),
                ("vae_bn_var.npy", 640),
                ("VAEDecoder.aimodel/main.hash", 32),
                ("VAEDecoder.aimodel/metadata.json", 395),
                ("VAEDecoder.aimodel/main.mlirb", 99_294_680),
                ("TextEncoder.aimodel/main.hash", 32),
                ("TextEncoder.aimodel/metadata.json", 396),
                ("TextEncoder.aimodel/main.mlirb", 6_228_947_553),
                ("Transformer.aimodel/main.hash", 32),
                ("Transformer.aimodel/metadata.json", 396),
                ("Transformer.aimodel/main.mlirb", 7_751_312_967),
            ],
            firstLoadCacheBytes: 14_079_913_984),
        ModelFiles(
            id: "int8", title: "FLUX.2 klein 4B (int8, 7.5 GB)",
            repo: "mlboydaisuke/FLUX.2-klein-4B-CoreAI",
            revision: "039db98cbf1247680ea5e76048f87b3ce061e6d6", folder: "macos-int8",
            files: [
                ("metadata.json", 1_549),
                ("tokenizer/chat_template.jinja", 4_168),
                ("tokenizer/config.json", 375),
                ("tokenizer/tokenizer.json", 11_422_650),
                ("tokenizer/tokenizer_config.json", 375),
                ("vae_bn_mean.npy", 640),
                ("vae_bn_var.npy", 640),
                ("VAEDecoder.aimodel/main.hash", 32),
                ("VAEDecoder.aimodel/metadata.json", 395),
                ("VAEDecoder.aimodel/main.mlirb", 99_294_687),
                ("TextEncoder.aimodel/main.hash", 32),
                ("TextEncoder.aimodel/metadata.json", 396),
                ("TextEncoder.aimodel/main.mlirb", 3_309_658_432),
                ("Transformer.aimodel/main.hash", 32),
                ("Transformer.aimodel/metadata.json", 396),
                ("Transformer.aimodel/main.mlirb", 4_119_647_410),
            ],
            firstLoadCacheBytes: 7_531_368_448),
        ModelFiles(
            id: "rgba-turbo-bf16", title: "RGBA-Image-2.1-Turbo (bf16 DiT, 32.4 GB)",
            repo: "mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI",
            revision: "c3bd985ce0d1d76fb899d4dc1bfcacf0f153cc1e", folder: "",
            files: rgbaTurboShared + [
                ("qi21t_dit_full_bf16_dyn_iofp32.aimodel/main.hash", 32),
                ("qi21t_dit_full_bf16_dyn_iofp32.aimodel/metadata.json", 1_008),
                ("qi21t_dit_full_bf16_dyn_iofp32.aimodel/main.mlirb", 14_230_567_115),
            ],
            kind: .rgba, fixedSteps: 8, sizes: [256, 512, 1024], defaultSize: 1024, usesGuidance: false,
            firstLoadCacheBytes: 46_671_654_912, promptNote: rgbaTurboForm, licenceNote: rgbaTurboLicence,
            makeEngine: { RGBAImageTurboEngine(dit: "qi21t_dit_full_bf16_dyn_iofp32") }),
        ModelFiles(
            id: "rgba-turbo-int8", title: "RGBA-Image-2.1-Turbo (int8 DiT, 25.8 GB)",
            repo: "mlboydaisuke/RGBA-Image-2.1-Turbo-CoreAI",
            revision: "c3bd985ce0d1d76fb899d4dc1bfcacf0f153cc1e", folder: "",
            files: rgbaTurboShared + [
                ("qi21t_dit_full_int8lin_dyn_iofp32.aimodel/main.hash", 32),
                ("qi21t_dit_full_int8lin_dyn_iofp32.aimodel/metadata.json", 1_044),
                ("qi21t_dit_full_int8lin_dyn_iofp32.aimodel/main.mlirb", 7_560_188_326),
            ],
            kind: .rgba, fixedSteps: 8, sizes: [256, 512, 1024], defaultSize: 1024, usesGuidance: false,
            firstLoadCacheBytes: 25_759_629_312, promptNote: rgbaTurboForm, licenceNote: rgbaTurboLicence,
            makeEngine: { RGBAImageTurboEngine(dit: "qi21t_dit_full_int8lin_dyn_iofp32") }),
    ]

    /// The model card's prompt form for a transparent background.
    private static let rgbaTurboForm = "For a transparent background: This is an RGBA image with transparency. {subject}. The image has alpha channel and the background is transparent."
    private static let rgbaTurboLicence = "Built with Qwen. Qwen Research License: non-commercial use only."

    /// Everything of RGBA-Image-2.1-Turbo but its two DiTs: the card, the licence, the host tables,
    /// the tokenizer, the three VAE decoders (256, 512, 1024 px) and the text encoder.
    private static let rgbaTurboShared: [(path: String, bytes: Int64)] = [
        ("config.json", 370),
        ("LICENSE", 7_831),
        ("NOTICE", 4_101),
        ("README.md", 20_208),
        ("host/host_contract.md", 10_437),
        ("host/scheduler.json", 253),
        ("host/rope_axis0_cos.f32", 294_912),
        ("host/rope_axis0_sin.f32", 294_912),
        ("host/rope_axis1_cos.f32", 1_032_192),
        ("host/rope_axis1_sin.f32", 1_032_192),
        ("host/rope_axis2_cos.f32", 1_032_192),
        ("host/rope_axis2_sin.f32", 1_032_192),
        ("tokenizer/chat_template.jinja", 5_292),
        ("tokenizer/processor_config.json", 1_300),
        ("tokenizer/tokenizer_config.json", 705),
        ("tokenizer/tokenizer.json", 11_422_650),
        ("qi21_vae_256_fp32.aimodel/main.hash", 32),
        ("qi21_vae_256_fp32.aimodel/metadata.json", 1_097),
        ("qi21_vae_256_fp32.aimodel/main.mlirb", 1_012_364_804),
        ("qi21_vae_512_fp32.aimodel/main.hash", 32),
        ("qi21_vae_512_fp32.aimodel/metadata.json", 1_098),
        ("qi21_vae_512_fp32.aimodel/main.mlirb", 1_012_366_781),
        ("qi21_vae_1024_fp32.aimodel/main.hash", 32),
        ("qi21_vae_1024_fp32.aimodel/metadata.json", 1_102),
        ("qi21_vae_1024_fp32.aimodel/main.mlirb", 1_012_370_701),
        ("qi21_encoder_dynL_w16a32_ids_iofp32.aimodel/main.hash", 32),
        ("qi21_encoder_dynL_w16a32_ids_iofp32.aimodel/metadata.json", 1_100),
        ("qi21_encoder_dynL_w16a32_ids_iofp32.aimodel/main.mlirb", 15_137_683_256),
    ]
}
