import AppKit
import SwiftUI

@main
struct InferenceLabApp: App {
    @Local private var monitor = Monitor()

    init() {
        // Needed when started with `swift run`: show a Dock icon and a real window.
        NSApplication.shared.setActivationPolicy(.regular)
        DispatchQueue.main.async { NSApp.activate(ignoringOtherApps: true) }
    }

    var body: some Scene {
        WindowGroup("Inference Lab") {
            ContentView()
                .environment(monitor)
                .onAppear { monitor.start(); Snapshot.scheduleIfNeeded() }
                .frame(minWidth: 1000, minHeight: 680)
        }
        .defaultSize(width: 1280, height: Snapshot.screen == nil ? 860 : 1100)
        .windowToolbarStyle(.unified)
    }
}

enum Screen: String, CaseIterable, Identifiable {
    case live = "Live"
    case model = "Model"
    case tokens = "Tokens"
    case request = "One request"
    case load = "Load test"

    var id: String { rawValue }
    var icon: String {
        switch self {
        case .live: "waveform.path.ecg"
        case .model: "cpu"
        case .tokens: "textformat.abc"
        case .request: "stopwatch"
        case .load: "person.3.fill"
        }
    }
    var blurb: String {
        switch self {
        case .live: "GPU, memory and loaded models, right now"
        case .model: "What the model is made of and what it costs in memory"
        case .tokens: "How text is cut into tokens"
        case .request: "Load, prefill and decode of a single request"
        case .load: "Many users at once: latency vs throughput"
        }
    }
}

struct ContentView: View {
    @Local private var screen: Screen = Snapshot.screen ?? .live

    var body: some View {
        NavigationSplitView {
            List(Screen.allCases, selection: $screen) { s in
                Label(s.rawValue, systemImage: s.icon).tag(s)
            }
            .navigationSplitViewColumnWidth(min: 170, ideal: 190)
            StatusFooter()
        } detail: {
            // Every screen stays alive when you switch away, so a running test keeps going.
            ZStack {
                LiveView().shown(screen == .live)
                ModelView().shown(screen == .model)
                TokensView(active: screen == .tokens).shown(screen == .tokens)
                RequestView().shown(screen == .request)
                LoadTestView().shown(screen == .load)
            }
            .navigationTitle(screen.rawValue)
            .navigationSubtitle(screen.blurb)
        }
    }
}

/// Tiny always-visible status under the sidebar.
struct StatusFooter: View {
    @Environment(Monitor.self) private var monitor

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Circle().fill(monitor.ollamaUp ? Palette.aqua : Palette.orange).frame(width: 8, height: 8)
                Text(monitor.ollamaUp ? "Ollama \(monitor.ollamaVersion ?? "")" : "Ollama not running")
            }
            Text("GPU \(monitor.latest?.gpuUtil ?? 0)%  ·  \(monitor.loaded.count) model(s) loaded")
                .foregroundStyle(.secondary)
        }
        .font(.caption)
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Model picker shared by several screens: the two project models first, then everything installed.
struct ModelPicker: View {
    @Binding var model: String
    @Local private var installed: [String] = []

    var body: some View {
        Picker("Model", selection: $model) {
            Text("fast · qwen2.5:0.5b").tag("fast")
            Text("quality · qwen2.5:1.5b").tag("quality")
            let others = installed.filter { !Ollama.aliases.values.contains($0) }
            if !others.isEmpty {
                Divider()
                ForEach(others, id: \.self) { Text($0).tag($0) }
            }
        }
        .frame(maxWidth: 260)
        .task { installed = (try? await Ollama.installed()) ?? [] }
    }
}

func resolveModel(_ m: String) -> String { Ollama.aliases[m] ?? m }

/// Red box for errors coming back from Python or Ollama.
struct ErrorBanner: View {
    let message: String
    var body: some View {
        Label(message, systemImage: "exclamationmark.triangle.fill")
            .font(.callout)
            .foregroundStyle(.white)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.red.opacity(0.8), in: RoundedRectangle(cornerRadius: 8))
            .textSelection(.enabled)
    }
}

extension View {
    /// Hide a screen without destroying it (its state and running tests survive).
    func shown(_ visible: Bool) -> some View {
        opacity(visible ? 1 : 0).allowsHitTesting(visible).accessibilityHidden(!visible)
    }
}
