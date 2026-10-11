// CoreAIImageGenMac — type a prompt, get an image on this Mac: FLUX.2 klein 4B on Apple's official
// CoreAIDiffusionPipeline (apple/coreai-models), or RGBA-Image-2.1-Turbo, whose images have a real
// alpha channel, on the Core AI runtime.

import SwiftUI

@main
struct CoreAIImageGenApp: App {
    @StateObject private var engine = DiffusionEngine()

    init() {
        Telemetry.begin()
    }

    var body: some Scene {
        WindowGroup {
            ContentView(engine: engine)
                .frame(minWidth: 820, minHeight: 640)
                .task { await Autoplay.runOnce(engine) }
        }
        .windowResizability(.contentMinSize)
    }
}
