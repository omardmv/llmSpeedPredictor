// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "InferenceLab",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "InferenceLab", path: "Sources/InferenceLab"),
    ]
)
