// CoreAIGraph — one Core AI graph opened with the runtime's own API (`AIModel`, `InferenceFunction`,
// `NDArray`), plus Xcode's ahead-of-time compile for a folder of compiled graphs.

import CoreAI
import Foundation

/// One graph (its `main` function) loaded for this Mac.
final class CoreAIGraph: @unchecked Sendable {
    let name: String
    let url: URL
    /// The specialization options it opened with: "gpu-efr" (an .aimodel, compiled just in time), or
    /// "default" or "gpu" (a compiled .aimodelc).
    let options: String
    /// Seconds `AIModel(contentsOf:options:)` and `loadFunction` took.
    let seconds: Double
    /// Kept alive while the function runs.
    private let model: AIModel
    private let function: InferenceFunction

    var inputNames: [String] { function.descriptor.inputNames }
    var outputNames: [String] { function.descriptor.outputNames }

    private init(name: String, url: URL, options: String, seconds: Double, model: AIModel, function: InferenceFunction) {
        self.name = name
        self.url = url
        self.options = options
        self.seconds = seconds
        self.model = model
        self.function = function
    }

    /// Opens `url`. An `.aimodel` is compiled by Core AI on its first load on this Mac and read from Core
    /// AI's cache after that (just in time), with the GPU preferred and expectFrequentReshapes: with the
    /// default options, or with the GPU alone, RGBA-Image-2.1-Turbo's DiT aborts the whole process at its
    /// first step (`ANERegion.mm:414 … Code=-19`, macOS 27.0 26A428), and Swift cannot catch that. A
    /// compiled `.aimodelc` opens with the runtime's default options, the ones the Python reference engine
    /// uses; if that fails, it tries once more preferring the GPU. `log` gets one line per failed try.
    static func load(_ url: URL, name: String, log: (String) -> Void) async throws -> CoreAIGraph {
        var failures: [String] = []
        let attempts: [(String, SpecializationOptions)]
        if url.pathExtension == "aimodelc" {
            attempts = [("default", .default), ("gpu", SpecializationOptions(preferredComputeUnitKind: .gpu))]
        } else {
            var justInTime = SpecializationOptions(preferredComputeUnitKind: .gpu)
            justInTime.expectFrequentReshapes = true
            attempts = [("gpu-efr", justInTime)]
        }
        for (label, options) in attempts {
            let start = ContinuousClock.now
            do {
                let model = try await AIModel(contentsOf: url, options: options)
                guard let function = try model.loadFunction(named: "main") else {
                    throw ImageGenError("\(url.lastPathComponent) has no main function.")
                }
                return CoreAIGraph(name: name, url: url, options: label, seconds: start.duration(to: .now).seconds,
                                   model: model, function: function)
            } catch {
                failures.append("\(label): \(error)")
                log("open \(name) options=\(label) failed: \(error)")
            }
        }
        throw ImageGenError("Core AI could not open \(url.lastPathComponent) (\(failures.joined(separator: "; "))).")
    }

    /// Throws unless the graph has these inputs and this output, so a wrong file fails at load
    /// with a sentence instead of at the first image.
    func expect(inputs: [String], output: String) throws {
        let missing = inputs.filter { !inputNames.contains($0) }
        guard missing.isEmpty, outputNames.contains(output) else {
            throw ImageGenError("\(url.lastPathComponent) does not have the expected interface: inputs \(inputNames), outputs \(outputNames).")
        }
    }

    /// Runs `main` and returns the output called `output`.
    func run(_ inputs: [String: NDArray], output: String) async throws -> NDArray {
        var outputs = try await function.run(inputs: inputs)
        guard let array = outputs.remove(output)?.ndArray else {
            throw ImageGenError("\(name) returned no \(output).")
        }
        return array
    }
}

extension NDArray {
    /// The values of an fp32 array in row-major order.
    func floats() throws -> [Float] {
        guard scalarType == .float32 else { throw ImageGenError("Expected a float32 output, got \(scalarType).") }
        let count = shape.reduce(1, *)
        var result = [Float](repeating: 0, count: count)
        result.withUnsafeMutableBufferPointer { out in
            view(as: Float.self).withUnsafePointer { pointer, shape, strides in
                let rank = shape.count
                var contiguous = true
                var expected = 1
                for d in stride(from: rank - 1, through: 0, by: -1) {
                    if shape[d] > 1, strides[d] != expected { contiguous = false }
                    expected *= shape[d]
                }
                if contiguous {
                    out.baseAddress!.update(from: pointer, count: count)
                    return
                }
                var index = [Int](repeating: 0, count: rank)
                for i in 0..<count {
                    var offset = 0
                    for d in 0..<rank { offset += index[d] * strides[d] }
                    out[i] = pointer[offset]
                    var d = rank - 1
                    while d >= 0 {
                        index[d] += 1
                        if index[d] < shape[d] { break }
                        index[d] = 0
                        d -= 1
                    }
                }
            }
        }
        return result
    }
}

/// Xcode's `coreai-build compile`, run once per graph, for a folder of graphs compiled ahead of time for
/// this Mac (the form the Python reference engine runs, host/host_contract.md section 8). The app does not
/// need it: by default Core AI compiles each `.aimodel` just in time.
enum AheadOfTimeCompiler {
    /// This Mac's GPU architecture, as Core AI names it (h16c on an M4 Max).
    static var architecture: String { AIModel.deviceArchitectureName }

    /// The graph compiled for this Mac in `folder`, when it is there: `<name>.<arch>.aimodelc`, or
    /// `<name>/<name>.<arch>.aimodelc` (what the model card's `--output aot/<name>` writes). A symlink is
    /// followed.
    static func compiled(_ name: String, in folder: URL) -> URL? {
        let file = "\(name).\(architecture).aimodelc"
        for url in [folder.appending(path: file), folder.appending(path: name).appending(path: file)]
        where FileManager.default.fileExists(atPath: url.path) {
            return url.resolvingSymlinksInPath()
        }
        return nil
    }

    /// Compiles `bundle` for this Mac's GPU into `<folder>/<name>.<arch>.aimodelc` and returns it with
    /// the seconds it took. The tool's output goes to `<folder>/<name>.compile.log`.
    static func compile(_ bundle: URL, name: String, into folder: URL) async throws -> (url: URL, seconds: Double) {
        let manager = FileManager.default
        try manager.createDirectory(at: folder, withIntermediateDirectories: true)
        let work = folder.appending(path: "\(name).compiling")
        try? manager.removeItem(at: work)
        let logURL = folder.appending(path: "\(name).compile.log")
        manager.createFile(atPath: logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: logURL)
        defer { try? log.close() }

        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/xcrun")
        process.arguments = ["coreai-build", "compile", bundle.path, "--output", work.path,
                             "--platform", "macOS", "--architecture", architecture,
                             "--preferred-compute", "gpu", "--expect-frequent-reshapes"]
        process.standardOutput = log
        process.standardError = log
        let start = ContinuousClock.now
        let status: Int32 = try await withCheckedThrowingContinuation { continuation in
            process.terminationHandler = { continuation.resume(returning: $0.terminationStatus) }
            do {
                try process.run()
            } catch {
                process.terminationHandler = nil
                continuation.resume(throwing: ImageGenError("Could not run xcrun: \(error.localizedDescription) Compiling the model for this Mac needs Xcode 27."))
            }
        }
        let seconds = start.duration(to: .now).seconds
        let product = work.appending(path: "\(name).\(architecture).aimodelc")
        guard status == 0, manager.fileExists(atPath: product.path) else {
            let output = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
            let tail = output.split(whereSeparator: \.isNewline).suffix(3).joined(separator: " ")
            throw ImageGenError("xcrun coreai-build could not compile \(bundle.lastPathComponent) (exit \(status)): \(tail) Compiling the model for this Mac needs Xcode 27.")
        }
        let destination = folder.appending(path: "\(name).\(architecture).aimodelc")
        try? manager.removeItem(at: destination)
        try manager.moveItem(at: product, to: destination)
        try? manager.removeItem(at: work)
        return (destination, seconds)
    }
}
