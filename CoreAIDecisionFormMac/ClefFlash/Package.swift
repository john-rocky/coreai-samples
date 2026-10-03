// swift-tools-version: 6.1
// ClefFlash, vendored from github.com/john-rocky/coreai-model-zoo apps/ClefFlash at 5ef2247: the library only (the zoo
// package also builds the `clef-flash` CLI). The Swift host of clef-flash on Core AI: a SystemOne-shaped request
// (+ one image) -> the typed decisions, on the system CoreAI framework and swift-transformers' tokenizer only.
import PackageDescription

let package = Package(
    name: "ClefFlash",
    platforms: [.macOS("27.0")],
    products: [
        .library(name: "ClefFlash", targets: ["ClefFlash"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        .target(
            name: "ClefFlash",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
    ]
)
