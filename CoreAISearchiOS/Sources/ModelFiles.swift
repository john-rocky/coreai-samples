import Foundation
import Hub

/// The model files this app downloads, pinned to one revision of the Hugging Face repo.
/// `ios/fp32-s128` holds the JIT `.aimodel`: Core AI specializes it for the iPhone it runs on
/// at the first load and caches the result, so one download serves every iPhone with iOS 27.
enum ModelFiles {
    static let repo = "mlboydaisuke/Granite-Embedding-97M-Multilingual-R2-CoreAI"
    static let revision = "fdb16fc6f4574a081ee04a597778d4b7f3b8a238"
    static let modelFolder = "ios/fp32-s128/granite97m_fp32_s128_bound.aimodel"
    static let tokenizerFolder = "ios/fp32-s128/tokenizer"

    /// Every file and its size at `revision`, in download order (the weights are 94 % of the bytes).
    static let files: [(path: String, bytes: Int64)] = [
        ("\(modelFolder)/main.mlirb", 389_988_764),
        ("\(modelFolder)/main.hash", 32),
        ("\(modelFolder)/metadata.json", 350),
        ("\(tokenizerFolder)/tokenizer.json", 25_301_672),
        ("\(tokenizerFolder)/tokenizer_config.json", 12_860),
        ("\(tokenizerFolder)/special_tokens_map.json", 871),
    ]
    static let totalBytes = files.reduce(0) { $0 + $1.bytes }

    /// The files can be downloaded again, so they live in Caches: not backed up, and if iOS
    /// purges them to free space the app shows the download screen again.
    static let downloadBase = URL.cachesDirectory.appending(path: "huggingface")
    static var root: URL { HubApi(downloadBase: downloadBase).localRepoLocation(HubApi.Repo(id: repo)) }
    static var modelURL: URL { root.appending(path: modelFolder) }
    static var tokenizerURL: URL { root.appending(path: tokenizerFolder) }

    static var isOnDisk: Bool { files.allSatisfy { size(of: $0.path) == $0.bytes } }
    static var bytesOnDisk: Int64 { files.reduce(0) { $0 + (size(of: $1.path) ?? 0) } }

    private static func size(of path: String) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: root.appending(path: path).path)[.size] as? NSNumber)?.int64Value
    }

    /// Downloads the files that are not on disk yet with swift-transformers' `HubApi`, one file at
    /// a time, so a failed download keeps the files that finished and Retry fetches only the rest.
    /// `endpoint` replaces https://huggingface.co (nil = the default). `progress` gets the bytes done so far.
    static func download(endpoint: String?, progress: @escaping @Sendable (Int64) -> Void) async throws {
        // cache: nil = one copy of each file, in downloadBase (no second copy in the Hub cache).
        // useOfflineMode: false = no network gives the network's own error instead of a missing-file one.
        let hub = HubApi(downloadBase: downloadBase, cache: nil, endpoint: endpoint, useOfflineMode: false)
        var done: Int64 = 0
        for file in files {
            if size(of: file.path) != file.bytes {
                let before = done
                try await hub.snapshot(from: HubApi.Repo(id: repo), revision: revision, matching: [file.path]) { @Sendable fraction in
                    progress(before + Int64(fraction.fractionCompleted * Double(file.bytes)))
                }
                try Task.checkCancellation()
                guard size(of: file.path) == file.bytes else {
                    throw SearchError.invalid("\(file.path) did not download completely")
                }
            }
            done += file.bytes
            progress(done)
        }
    }
}
