import AppKit
import SwiftUI

/// Chart colors: a validated colorblind-safe categorical palette, with its own dark-mode steps.
enum Palette {
    static let blue = dynamic(light: 0x2A78D6, dark: 0x3987E5)
    static let orange = dynamic(light: 0xEB6834, dark: 0xD95926)
    static let aqua = dynamic(light: 0x1BAF7A, dark: 0x199E70)
    static let yellow = dynamic(light: 0xEDA100, dark: 0xC98500)
    static let gray = dynamic(light: 0xC3C2B7, dark: 0x5A5953)
    static let magenta = dynamic(light: 0xE87BA4, dark: 0xD55181)
    static let green = dynamic(light: 0x008300, dark: 0x008300)

    /// Fixed order for comparing runs: run 1 is always blue, run 2 orange, ... never cycled.
    static let series = [blue, orange, aqua, yellow, magenta, green]

    // Phases of one request. Same colors everywhere in the app.
    static let load = aqua
    static let prefill = orange
    static let decode = blue
    static let other = gray

    private static func dynamic(light: Int, dark: Int) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let hex = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
                           green: CGFloat((hex >> 8) & 0xFF) / 255,
                           blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
        })
    }
}

enum Fmt {
    static func gb(_ bytes: Double) -> String { String(format: "%.2f GB", bytes / 1_073_741_824) }
    static func mb(_ bytes: Double) -> String { String(format: "%.0f MB", bytes / 1_048_576) }
    static func bytes(_ b: Double) -> String { b >= 1_073_741_824 ? gb(b) : mb(b) }
    static func secs(_ s: Double) -> String { s < 1 ? String(format: "%.0f ms", s * 1000) : String(format: "%.2f s", s) }
}

/// A big number with a label, the basic tile of every screen.
struct StatTile: View {
    let title: String
    let value: String
    var detail: String? = nil
    var color: Color? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                if let color { Circle().fill(color).frame(width: 8, height: 8) }
                Text(title).font(.caption).foregroundStyle(.secondary)
            }
            Text(value).font(.system(size: 26, weight: .semibold, design: .rounded)).monospacedDigit()
            if let detail { Text(detail).font(.caption2).foregroundStyle(.secondary).lineLimit(2) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(12)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
    }
}

/// A titled box around a chart or a group of controls.
struct Card<Content: View>: View {
    let title: String
    var subtitle: String? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                if let subtitle { Text(subtitle).font(.caption).foregroundStyle(.secondary) }
            }
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }
}

/// Small colored square + label, for chart legends.
struct LegendDot: View {
    let color: Color
    let label: String
    var body: some View {
        HStack(spacing: 5) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 10, height: 10)
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// Same as SwiftUI's `@State`. Without Xcode installed, the Command Line Tools lack the
/// plugin behind the `@State` macro, so this wraps the plain `State` struct instead.
@propertyWrapper
struct Local<Value>: DynamicProperty {
    private let state: State<Value>
    init(wrappedValue: Value) { state = State(initialValue: wrappedValue) }
    var wrappedValue: Value {
        get { state.wrappedValue }
        nonmutating set { state.wrappedValue = newValue }
    }
    var projectedValue: Binding<Value> { state.projectedValue }
}
