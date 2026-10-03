// CoreAIDecisionFormMac — copy a customer email, and a support ticket form fills at once: eight typed decisions
// (choice / score / yes-no) answered by clef-flash in one read of the email, on Apple's official Core AI runtime.

import SwiftUI

@main
struct DecisionFormApp: App {
    @State private var autoplay = Autoplay()

    var body: some Scene {
        WindowGroup {
            DecisionFormView()
                .environment(autoplay)
                .frame(minWidth: 680, minHeight: 920)
        }
        .defaultSize(width: 700, height: 960)
        .windowResizability(.contentMinSize)
    }
}
