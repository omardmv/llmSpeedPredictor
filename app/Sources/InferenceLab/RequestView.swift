import Charts
import SwiftUI

/// One line of `python -m bench run --json`.
struct RunResult: Decodable, Identifiable {
    struct GPU: Decodable { let gpuUtilAvg: Double; let gpuUtilMax: Double; let gpuMemMax: Double }
    var id = UUID()
    var number = 0
    var cold = false
    let loadS: Double
    let prefillS: Double
    let decodeS: Double
    let wallS: Double
    let ttftS: Double
    let promptTokens: Int
    let outputTokens: Int
    let stop: String
    let gapP50Ms: Double
    let gapP95Ms: Double
    let gapMaxMs: Double
    let gpu: GPU
    let gpuSamples: [[Double]]   // [seconds, busy %, memory bytes]
    let tTokens: [Double]        // when each streamed piece arrived
    let text: String
    let promptChars: Int

    enum CodingKeys: String, CodingKey {
        case loadS, prefillS, decodeS, wallS, ttftS, promptTokens, outputTokens, stop, gapP50Ms, gapP95Ms, gapMaxMs,
             gpu, gpuSamples, tTokens, text, promptChars
    }

    var otherS: Double { max(wallS - loadS - prefillS - decodeS, 0) }
    var prefillSpeed: Double { prefillS > 0 ? Double(promptTokens) / prefillS : 0 }
    var decodeSpeed: Double { decodeS > 0 ? Double(outputTokens) / decodeS : 0 }
    /// Ollama counts reused prompt tokens too, so a cached prompt shows an impossible speed.
    var promptWasCached: Bool { promptTokens > 50 && prefillSpeed > 20_000 }
}

struct RequestView: View {
    @Local private var model = "quality"
    @Local private var prompt = "Explain how a car engine works."
    @Local private var pad = 0
    @Local private var maxTokens = 256.0
    @Local private var cold = true
    @Local private var repeatCount = 3
    @Local private var runs: [RunResult] = []
    @Local private var selected: UUID?
    @Local private var task: Task<Void, Never>?
    @Local private var error: String?
    @AppStorage(ServerChoice.backendKey) private var backend = "ollama"
    @AppStorage(ServerChoice.urlKey) private var url = "http://localhost:8080"

    private var current: RunResult? { runs.first { $0.id == selected } ?? runs.last }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                controls
                if let error { ErrorBanner(message: error) }
                if !runs.isEmpty { runPicker }
                if let r = current {
                    phaseCard(r)
                    tiles(r)
                    HStack(alignment: .top, spacing: 14) {
                        arrivalCard(r)
                        gpuCard(r)
                    }
                    if runs.count > 1 { historyCard }
                    Card(title: "Answer", subtitle: "\(r.outputTokens) tokens · stop reason: \(r.stop == "length" ? "hit max_tokens" : "model chose to stop")") {
                        Text(r.text).textSelection(.enabled).font(.callout)
                    }
                }
            }
            .padding(18)
        }
        .onAppear { if Snapshot.isActive(.request) { send() } }
    }

    private var controls: some View {
        Card(title: "Request") {
            ServerPicker()
            HStack(spacing: 12) {
                ModelPicker(model: $model)
                Toggle("Cold start (unload first)", isOn: $cold).fixedSize()
                    .disabled(backend == "llama")
                    .help(backend == "llama" ? "llama-server keeps its model loaded; restart it for a cold start" : "")
                Stepper("Send it \(repeatCount)×", value: $repeatCount, in: 1...6)
                    .fixedSize()
                    .help("Same request several times in a row: watch load and prefill disappear")
                Spacer(minLength: 0)
                if task != nil {
                    ProgressView().controlSize(.small)
                    Button("Stop") { task?.cancel() }.fixedSize()
                } else {
                    Button("Send") { send() }.buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command).fixedSize()
                }
            }
            TextField("Prompt", text: $prompt, axis: .vertical).lineLimit(1...4).textFieldStyle(.roundedBorder)
            HStack(spacing: 18) {
                Stepper("Add \(pad) paragraphs before it", value: $pad, in: 0...20)
                    .fixedSize()
                    .help("Real paragraphs from the Dolly dataset: a longer prompt means a longer prefill")
                HStack {
                    Text("Answer limit (max_tokens)").fixedSize()
                    Slider(value: Binding(get: { maxTokens }, set: { maxTokens = ($0 / 10).rounded() * 10 }), in: 10...1020).frame(minWidth: 120, maxWidth: 240)
                    Text("\(Int(maxTokens)) tokens").monospacedDigit().fixedSize()
                }
                Spacer(minLength: 0)
            }
        }
    }

    private func send() {
        error = nil
        runs = []
        var args = ["run", model, "--prompt", prompt, "--pad", "\(pad)", "--max-tokens", "\(Int(maxTokens))",
                    "--repeat", "\(repeatCount)"]
        if cold && backend != "llama" { args.append("--cold") }
        args += ServerChoice.args(backend: backend, url: url)
        task = Task {
            do {
                for try await line in Bench.lines(args) {
                    var r = try Bench.decode(RunResult.self, line)
                    r.number = runs.count + 1
                    r.cold = cold && r.number == 1
                    runs.append(r)
                    selected = r.id
                }
            } catch is CancellationError {
            } catch {
                self.error = error.localizedDescription
            }
            task = nil
        }
    }

    private var runPicker: some View {
        Picker("Request", selection: Binding(get: { current?.id }, set: { selected = $0 })) {
            ForEach(runs) { r in Text("#\(r.number)\(r.cold ? " (cold)" : "")").tag(Optional(r.id)) }
        }
        .pickerStyle(.segmented)
        .frame(maxWidth: 520)
    }

    private func phaseCard(_ r: RunResult) -> some View {
        let parts: [(String, Double, Color)] = [("load", r.loadS, Palette.load), ("prefill", r.prefillS, Palette.prefill),
                                                ("decode", r.decodeS, Palette.decode), ("other", r.otherS, Palette.other)]
        return Card(title: "Where the \(Fmt.secs(r.wallS)) went",
                    subtitle: "load = weights into memory · prefill = read the prompt · decode = write the answer, one token at a time") {
            Chart(parts, id: \.0) { name, secs, _ in
                BarMark(x: .value("seconds", secs), y: .value("", "request"))
                    .foregroundStyle(by: .value("phase", name))
            }
            .chartForegroundStyleScale(domain: parts.map(\.0), range: parts.map(\.2))
            .chartYAxis(.hidden)
            .chartLegend(.hidden)
            .chartXAxisLabel("seconds")
            .frame(height: 56)
            HStack(spacing: 18) {
                ForEach(parts, id: \.0) { name, secs, color in LegendDot(color: color, label: "\(name) \(Fmt.secs(secs))") }
            }
        }
    }

    private func tiles(_ r: RunResult) -> some View {
        HStack(spacing: 12) {
            StatTile(title: "Time to first token", value: Fmt.secs(r.ttftS), detail: "what the user waits before text appears")
            StatTile(title: "Prefill", value: r.promptWasCached ? "cached" : String(format: "%.0f tok/s", r.prefillSpeed),
                     detail: r.promptWasCached ? "\(r.promptTokens) tokens found in the KV cache, not recomputed"
                                               : "\(r.promptTokens) prompt tokens in \(Fmt.secs(r.prefillS))",
                     color: Palette.prefill)
            StatTile(title: "Decode", value: String(format: "%.0f tok/s", r.decodeSpeed),
                     detail: "\(r.outputTokens) tokens in \(Fmt.secs(r.decodeS))", color: Palette.decode)
            StatTile(title: "Gap between tokens", value: String(format: "%.1f ms", r.gapP50Ms),
                     detail: String(format: "p95 %.1f ms · max %.1f ms", r.gapP95Ms, r.gapMaxMs))
            StatTile(title: "GPU busy", value: String(format: "%.0f%%", r.gpu.gpuUtilAvg),
                     detail: String(format: "max %.0f%% · peak memory %@", r.gpu.gpuUtilMax, Fmt.gb(r.gpu.gpuMemMax)))
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private func arrivalCard(_ r: RunResult) -> some View {
        Card(title: "Tokens arriving", subtitle: "flat = waiting · steep straight line = decode at a steady speed") {
            Chart {
                if r.loadS > 0.005 {
                    RectangleMark(xStart: .value("s", 0), xEnd: .value("s", r.loadS))
                        .foregroundStyle(Palette.load.opacity(0.18))
                }
                RectangleMark(xStart: .value("s", r.loadS), xEnd: .value("s", r.loadS + r.prefillS))
                    .foregroundStyle(Palette.prefill.opacity(0.22))
                ForEach(Array(r.tTokens.enumerated()), id: \.offset) { i, t in
                    LineMark(x: .value("seconds", t), y: .value("token", i + 1))
                        .foregroundStyle(Palette.decode)
                        .lineStyle(StrokeStyle(lineWidth: 2))
                }
                RuleMark(x: .value("TTFT", r.ttftS))
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .annotation(position: .top, alignment: .leading) {
                        Text("first token \(Fmt.secs(r.ttftS))").font(.caption2).foregroundStyle(.secondary)
                    }
            }
            .chartXScale(domain: 0...max(r.wallS, 0.01))
            .chartXAxisLabel("seconds after sending")
            .chartYAxisLabel("tokens received")
            .frame(height: 230)
        }
    }

    private func gpuCard(_ r: RunResult) -> some View {
        Card(title: "GPU during this request", subtitle: "sampled every 0.1 s from the GPU driver") {
            Chart {
                ForEach(Array(r.gpuSamples.enumerated()), id: \.offset) { _, s in
                    AreaMark(x: .value("seconds", s[0]), y: .value("busy", s[1]))
                        .foregroundStyle(Palette.blue.opacity(0.25))
                    LineMark(x: .value("seconds", s[0]), y: .value("busy", s[1]))
                        .foregroundStyle(Palette.blue)
                }
                RuleMark(x: .value("TTFT", r.ttftS))
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }
            .chartXScale(domain: 0...max(r.wallS, 0.01))
            .chartYScale(domain: 0...100)
            .chartXAxisLabel("seconds after sending")
            .chartYAxisLabel("GPU busy %")
            .frame(height: 230)
        }
    }

    private var historyCard: some View {
        Card(title: "All \(runs.count) requests", subtitle: "same request each time: what changes after the first one?") {
            Table(runs) {
                TableColumn("#") { r in Text("\(r.number)\(r.cold ? " cold" : "")") }.width(60)
                TableColumn("load") { r in Text(Fmt.secs(r.loadS)).monospacedDigit() }
                TableColumn("prompt tokens") { r in Text("\(r.promptTokens)").monospacedDigit() }
                TableColumn("prefill") { r in Text(r.promptWasCached ? "cached" : Fmt.secs(r.prefillS)).monospacedDigit() }
                TableColumn("first token") { r in Text(Fmt.secs(r.ttftS)).monospacedDigit() }
                TableColumn("decode tok/s") { r in Text(String(format: "%.1f", r.decodeSpeed)).monospacedDigit() }
                TableColumn("total") { r in Text(Fmt.secs(r.wallS)).monospacedDigit() }
            }
            .frame(height: CGFloat(runs.count) * 26 + 34)
        }
    }
}
