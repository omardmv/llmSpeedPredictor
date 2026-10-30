import AppKit
import SwiftUI

/// Self-check mode, for looking at the UI from a terminal:
///
///     INFERENCE_LAB_SNAPSHOT=request .build/release/InferenceLab
///
/// opens that screen, presses its main button, saves the window to
/// /tmp/inferencelab-<screen>.png and quits. Draws the window itself, so it needs no
/// Screen Recording permission.
/// Add INFERENCE_LAB_SNAPSHOT_RUN=1 with `load` to start a tiny real load test first.
enum Snapshot {
    static let screen: Screen? = {
        guard let name = ProcessInfo.processInfo.environment["INFERENCE_LAB_SNAPSHOT"] else { return nil }
        return Screen.allCases.first { "\($0)" == name }
    }()

    static func isActive(_ s: Screen) -> Bool { screen == s }

    /// Seconds to wait for the screen's work before taking the picture.
    static var delay: Double {
        Double(ProcessInfo.processInfo.environment["INFERENCE_LAB_SNAPSHOT_DELAY"] ?? "") ?? 12
    }

    @MainActor
    static func scheduleIfNeeded() {
        guard let screen else { return }
        // INFERENCE_LAB_SNAPSHOT_WIDTH=1000 checks the layout at the narrowest window.
        if let w = Double(ProcessInfo.processInfo.environment["INFERENCE_LAB_SNAPSHOT_WIDTH"] ?? "") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                NSApp.windows.first { $0.frame.width > 600 }?.setContentSize(NSSize(width: w, height: 1100))
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
            guard let window = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil && $0.frame.width > 600 }),
                  let view = window.contentView,
                  let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { NSApp.terminate(nil); return }
            view.cacheDisplay(in: view.bounds, to: rep)
            let url = URL(fileURLWithPath: "/tmp/inferencelab-\(screen).png")
            try? rep.representation(using: .png, properties: [:])?.write(to: url)
            print(url.path)
            NSApp.terminate(nil)  // a normal quit, so child processes get stopped
        }
    }
}
