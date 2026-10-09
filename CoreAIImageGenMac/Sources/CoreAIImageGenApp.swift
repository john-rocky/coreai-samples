// CoreAIImageGenMac — type a prompt, get a 1024×1024 image from FLUX.2 klein 4B running on
// Apple's official CoreAIDiffusionPipeline (apple/coreai-models).

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
