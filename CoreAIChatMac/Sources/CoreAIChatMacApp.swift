// CoreAIChatMac — a minimal macOS chat app for Core AI language bundles on
// Apple's official runtime. Pick a folder of exported bundles, click a model,
// chat — with live load / TTFT / tok/s / memory stats in the footer.

import SwiftUI

@main
struct CoreAIChatMacApp: App {
    var body: some Scene {
        WindowGroup {
            ChatView()
                .frame(minWidth: 980, minHeight: 640)
        }
        .windowResizability(.contentSize)
    }
}
