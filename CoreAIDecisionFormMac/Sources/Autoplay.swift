// Autoplay — drives the screen hands-off, for a recording or a smoke run from a script:
//
//   CoreAIDecisionFormMac.app/Contents/MacOS/CoreAIDecisionFormMac -autoplay form -modelsFolder <dir> -delay 1.5 \
//       [-trigger <file>] [-log 1]
//
// loads the model from <dir>, waits until it is ready, then (with -trigger) until that file exists — a recorder
// creates it once the capture is rolling — then `delay` seconds, and presses Sample email. With -log 1 the status line
// is mirrored into Documents/autoplay-form.log as it changes, with the filled fields as the form shows them, then a
// DONE line. Nothing else changes: the screen is the same code with the same buttons; this only presses one.

import Foundation
import Observation

@MainActor
@Observable
final class Autoplay {
    let enabled: Bool
    let delay: Double
    let trigger: String?
    let log: Bool
    private var fired = false

    init() {
        let defaults = UserDefaults.standard  // -key value command-line pairs land here
        enabled = defaults.string(forKey: "autoplay") == "form"
        let d = defaults.double(forKey: "delay")
        delay = d > 0 ? d : 1.5
        trigger = defaults.string(forKey: "trigger").map {
            $0.hasPrefix("/") ? $0 : URL.documentsDirectory.appending(path: $0).path
        }
        log = defaults.bool(forKey: "log")
    }

    /// Runs `action` once the model is ready (and the trigger file exists), then mirrors the status for 30 seconds.
    func run(model: DecisionFormModel, action: () -> Void) async {
        guard enabled, !fired else { return }
        fired = true
        while !model.isReady {
            if case .failed(let message) = model.phase {
                write("failed: \(message)")
                return
            }
            try? await Task.sleep(for: .milliseconds(200))
        }
        write("ready · loaded and warmed up in \(String(format: "%.1f", model.loadSeconds ?? 0)) s")
        if let trigger {
            while !FileManager.default.fileExists(atPath: trigger) {
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        try? await Task.sleep(for: .seconds(delay))
        action()
        guard log else { return }
        var last = ""
        var filled = false
        let end = Date().addingTimeInterval(30)
        while Date() < end {
            if model.status != last {
                last = model.status
                write(last)
            }
            if !filled, model.seconds != nil {
                filled = true
                for field in DecisionFormModel.fields {
                    guard let v = model.values[field.id] else { continue }
                    write("\(field.label): \(v.text) p \(String(format: "%.4f", v.probability))"
                        + (v.level.map { " level \(String(format: "%.4f", $0))" } ?? ""))
                }
                if let tokens = model.tokens { write("\(tokens) tokens") }
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
        write("DONE")
    }

    /// Appends one timestamped line to Documents/autoplay-form.log (only with -log 1).
    private func write(_ line: String) {
        guard log else { return }
        let url = URL.documentsDirectory.appending(path: "autoplay-form.log")
        let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .current,
                                                formatOptions: [.withInternetDateTime, .withFractionalSeconds])
        let text = "\(stamp) \(line)\n"
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
