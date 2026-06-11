// ModelDownloader.swift — in-app model delivery for the Core AI sample apps.
//
// Downloads `.aimodel` bundles (directory trees) from a Hugging Face model repo into a
// destination directory. Every file streams into a hidden staging directory and a bundle is
// renamed into place only when ALL of its files are complete: AIModel must never see a partial
// bundle — a partially-present bundle poisons the content-keyed coreai-cache (later loads fail
// ENOENT until the cache is wiped).
//
// Usage:
//   let items = [ModelDownloader.Item(remote: "ios-gpu/foo.aimodel", local: "foo.aimodel")]
//   await downloader.fetch(repo: "https://huggingface.co/org/repo", items: items, into: docs)
//
// `remote` is a directory (or file) path inside the repo; `local` is the name created under the
// destination. Items already present at the destination are skipped (they were placed
// atomically, so presence == complete). Downloaded bundles are excluded from iCloud backup.

import Foundation

@MainActor
final class ModelDownloader: ObservableObject {
    struct Item {
        let remote: String   // path inside the HF repo
        let local: String    // name under the destination directory
    }

    enum Phase: Equatable {
        case idle, listing, downloading(file: String), failed(String), done
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var fraction: Double = 0
    @Published private(set) var detail = ""

    var busy: Bool {
        switch phase { case .listing, .downloading: return true; default: return false }
    }

    private var totalBytes: Int64 = 0
    private var shownFraction = -1.0

    // Accepts "https://huggingface.co/<org>/<name>[/...]" or a bare "<org>/<name>".
    nonisolated static func repoId(from s: String) -> String? {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if let u = URL(string: t), let host = u.host, host.hasSuffix("huggingface.co") {
            let parts = u.path.split(separator: "/").map(String.init)
            return parts.count >= 2 ? "\(parts[0])/\(parts[1])" : nil
        }
        let parts = t.split(separator: "/").map(String.init)
        return parts.count == 2 ? t : nil
    }

    func fetch(repo: String, items: [Item], into dest: URL) async {
        guard let repoId = Self.repoId(from: repo) else {
            phase = .failed("not a Hugging Face repo URL"); return
        }
        let fm = FileManager.default
        let pending = items.filter { !fm.fileExists(atPath: dest.appendingPathComponent($0.local).path) }
        guard !pending.isEmpty else { phase = .done; return }
        phase = .listing; fraction = 0; shownFraction = -1; detail = "listing files…"

        let delegate = DownloadDelegate()
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        do {
            var plan: [(item: Item, files: [PlannedFile])] = []
            for item in pending { plan.append((item, try await Self.list(repoId: repoId, item: item))) }
            totalBytes = plan.flatMap(\.files).reduce(0) { $0 + $1.size }
            var doneBytes: Int64 = 0

            for (item, files) in plan {
                // Stage the whole bundle, then one atomic rename into place (same volume).
                let staging = dest.appendingPathComponent(".staging-\(item.local)")
                try? fm.removeItem(at: staging)
                try fm.createDirectory(at: staging, withIntermediateDirectories: true)
                for f in files {
                    phase = .downloading(file: f.rel)
                    let target = staging.appendingPathComponent(f.rel)
                    try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                    let base = doneBytes
                    let got = try await Self.download(f.url, via: session, delegate: delegate) { [weak self] written in
                        Task { @MainActor in self?.tick(base + written) }
                    }
                    try fm.moveItem(at: got, to: target)
                    doneBytes += f.size
                }
                let final = dest.appendingPathComponent(item.local)
                try? fm.removeItem(at: final)
                try fm.moveItem(at: staging, to: final)
                var noBackup = URLResourceValues(); noBackup.isExcludedFromBackup = true
                var u = final; try? u.setResourceValues(noBackup)
            }
            fraction = 1; detail = ""
            phase = .done
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func tick(_ done: Int64) {
        guard busy else { return }
        let f = totalBytes > 0 ? Double(done) / Double(totalBytes) : 0
        guard f - shownFraction >= 0.002 else { return }   // ~50 UI updates per download
        shownFraction = f; fraction = min(f, 1)
        detail = "\(Self.fmt(done)) / \(Self.fmt(totalBytes))"
    }

    // MARK: repo listing

    private struct PlannedFile { let url: URL; let rel: String; let size: Int64 }
    private struct TreeEntry: Decodable {
        let type: String, path: String, size: Int64?
        let lfs: LFS?
        struct LFS: Decodable { let size: Int64? }
    }

    // Enumerate the files under `item.remote` via the HF tree API (paths here hold a handful of
    // files — pagination is not handled).
    private nonisolated static func list(repoId: String, item: Item) async throws -> [PlannedFile] {
        guard let api = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main/\(item.remote)?recursive=true") else {
            throw err("bad path \(item.remote)")
        }
        let (data, resp) = try await URLSession.shared.data(from: api)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            throw err("\(item.remote): not found in \(repoId)")
        }
        let entries = try JSONDecoder().decode([TreeEntry].self, from: data)
        let prefix = item.remote.hasSuffix("/") ? item.remote : item.remote + "/"
        return try entries.filter { $0.type == "file" }.map { e in
            let rel = e.path == item.remote ? (e.path as NSString).lastPathComponent
                                            : String(e.path.dropFirst(prefix.count))
            guard let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(e.path)") else {
                throw err("bad file path \(e.path)")
            }
            return PlannedFile(url: url, rel: rel, size: e.lfs?.size ?? e.size ?? 0)
        }
    }

    // MARK: single-file download (progress via delegate; the temp file is claimed synchronously
    // inside the delegate callback, then returned for the caller to move into staging)

    private nonisolated static func download(
        _ url: URL, via session: URLSession, delegate: DownloadDelegate,
        onBytes: @escaping @Sendable (Int64) -> Void
    ) async throws -> URL {
        try await withCheckedThrowingContinuation { cont in
            delegate.onBytes = onBytes
            delegate.onFinish = { cont.resume(with: $0) }
            session.downloadTask(with: url).resume()
        }
    }

    private nonisolated static func err(_ msg: String) -> Error {
        NSError(domain: "ModelDownloader", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }

    private nonisolated static func fmt(_ b: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: b, countStyle: .file)
    }
}

// Delegate callbacks arrive serialized on the session's queue; `onFinish` is cleared after the
// first resume so the didComplete(error:) that follows a success cannot double-resume.
private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    var onBytes: (@Sendable (Int64) -> Void)?
    var onFinish: (@Sendable (Result<URL, Error>) -> Void)?

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        onBytes?(totalBytesWritten)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        let code = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? -1
        guard code == 200 else {
            finish(.failure(NSError(domain: "ModelDownloader", code: code, userInfo: [
                NSLocalizedDescriptionKey: "HTTP \(code) for \(downloadTask.originalRequest?.url?.lastPathComponent ?? "?")"])))
            return
        }
        let keep = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: keep)
            finish(.success(keep))
        } catch { finish(.failure(error)) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { finish(.failure(error)) }
    }

    private func finish(_ r: Result<URL, Error>) {
        onFinish?(r)
        onFinish = nil
    }
}
