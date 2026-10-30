import Charts
import SwiftUI

struct ModelReport: Decodable {
    struct Loaded: Decodable {
        let ollamaTotal: Double
        let ollamaOnGpu: Double
        let gpuMemDelta: Double
        let contextLength: Int
    }
    let name: String
    let architecture: String
    let quantization: String
    let parameters: Double
    let fileBytes: Double
    let bitsPerParameter: Double
    let layers: Int
    let hiddenSize: Int
    let ffnSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let maxContext: Int
    let kvBytesPerToken: Double
    let numCtx: Int
    let loaded: Loaded?
}

struct ModelView: View {
    @Environment(Monitor.self) private var monitor
    @Local private var model = "quality"
    @Local private var ctx = 4096
    @Local private var report: ModelReport?
    @Local private var running = false
    @Local private var error: String?
    @Local private var users = 8.0
    @Local private var calcCtx = 4096.0

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                controls
                if let error { ErrorBanner(message: error) }
                if let r = report {
                    tiles(r)
                    HStack(alignment: .top, spacing: 14) {
                        architecture(r)
                        VStack(spacing: 14) {
                            memory(r)
                            kvCalculator(r)
                        }
                    }
                } else if !running {
                    Text("Pick a model and press Load & measure. It loads the model into memory and asks Ollama what it took.")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(18)
        }
        .onAppear { if Snapshot.isActive(.model) { Task { await measure() } } }
    }

    private var controls: some View {
        HStack(spacing: 12) {
            ModelPicker(model: $model)
            Picker("Context", selection: $ctx) {
                ForEach([2048, 4096, 8192, 16384, 32768], id: \.self) { Text("\($0) tokens").tag($0) }
            }
            .frame(maxWidth: 200)
            .help("num_ctx: how many tokens of KV cache Ollama reserves")
            Button(running ? "Measuring…" : "Load & measure") { Task { await measure() } }
                .buttonStyle(.borderedProminent)
                .disabled(running)
            if running { ProgressView().controlSize(.small) }
            Spacer()
        }
    }

    private func measure() async {
        running = true
        error = nil
        defer { running = false }
        do {
            for try await line in Bench.lines(["model", model, "--ctx", "\(ctx)"]) {
                report = try Bench.decode(ModelReport.self, line)
                calcCtx = Double(ctx)
            }
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func tiles(_ r: ModelReport) -> some View {
        HStack(spacing: 12) {
            StatTile(title: "Parameters", value: String(format: "%.2f B", r.parameters / 1e9),
                     detail: "\(r.architecture) · \(r.quantization)")
            StatTile(title: "Bits per parameter", value: String(format: "%.2f", r.bitsPerParameter),
                     detail: "16 = full fp16 · ~4.5–5 = Q4")
            StatTile(title: "Weights on disk", value: Fmt.bytes(r.fileBytes),
                     detail: "read from memory once for every generated token")
            StatTile(title: "KV cache per token", value: String(format: "%.1f KB", r.kvBytesPerToken / 1024),
                     detail: "grows with every token in the conversation")
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private func architecture(_ r: ModelReport) -> some View {
        Card(title: "Architecture", subtitle: r.name) {
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 10) {
                fact("Layers", "\(r.layers)", "every new token passes through all of them, one after another")
                fact("Hidden size", "\(r.hiddenSize)", "length of the vector that stands for one token")
                fact("Feed-forward size", "\(r.ffnSize)", "the wider layer inside each block, where most weights are")
                fact("Query heads", "\(r.attentionHeads)", "\"what am I looking for?\" — one per head")
                fact("Key/value heads", "\(r.kvHeads)",
                     r.kvHeads < r.attentionHeads ? "fewer than query heads: grouped-query attention, \(r.attentionHeads / r.kvHeads) query heads share one K/V"
                                                  : "one K/V per query head")
                fact("Head size", "\(r.headDim)", "hidden size ÷ query heads")
                fact("Trained context", "\(r.maxContext)", "Ollama only reserves num_ctx of it")
                Divider().gridCellColumns(3)
                fact("KV per token", "\(Int(r.kvBytesPerToken)) B",
                     "2 (K and V) × \(r.layers) layers × \(r.kvHeads) KV heads × \(r.headDim) × 2 bytes (fp16)")
            }
        }
        .frame(maxWidth: 560)
    }

    private func fact(_ name: String, _ value: String, _ why: String) -> some View {
        GridRow {
            Text(name).foregroundStyle(.secondary)
            Text(value).font(.body.monospacedDigit().weight(.semibold))
            Text(why).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func memory(_ r: ModelReport) -> some View {
        Card(title: "Memory once loaded",
             subtitle: r.loaded.map { "num_ctx \($0.contextLength) · Ollama reports \(Fmt.bytes($0.ollamaTotal))" }) {
            if let l = r.loaded {
                let kv = r.kvBytesPerToken * Double(l.contextLength)
                let rest = max(l.ollamaTotal - r.fileBytes - kv, 0)
                let parts: [(String, Double, Color)] = [("weights", r.fileBytes, Palette.blue),
                                                        ("KV cache, 1 slot", kv, Palette.orange),
                                                        ("everything else", rest, Palette.gray)]
                Chart(parts, id: \.0) { name, bytes, _ in
                    BarMark(x: .value("MB", bytes / 1_048_576), y: .value("", "memory"))
                        .foregroundStyle(by: .value("part", name))
                }
                .chartForegroundStyleScale(domain: parts.map(\.0), range: parts.map(\.2))
                .chartYAxis(.hidden)
                .chartXAxisLabel("MB")
                .chartLegend(.hidden)
                .frame(height: 54)
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(parts, id: \.0) { name, bytes, color in
                        HStack { LegendDot(color: color, label: name); Spacer(); Text(Fmt.bytes(bytes)).monospacedDigit() }
                    }
                }
                .font(.callout)
                Text("\"Everything else\" = Ollama's total − weights − one KV slot: scratch buffers for the math, plus KV cache for any extra slots. How many slots did it reserve?")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Not loaded.").foregroundStyle(.secondary)
            }
        }
    }

    private func kvCalculator(_ r: ModelReport) -> some View {
        let kv = r.kvBytesPerToken * calcCtx * users.rounded()
        // macOS lets the GPU use roughly 3/4 of unified memory by default.
        let budget = monitor.machine.memoryBytes * 0.75 - r.fileBytes
        return Card(title: "KV cache calculator", subtitle: "how much memory N users with full contexts would need") {
            HStack {
                Text("Users").frame(width: 70, alignment: .leading)
                Slider(value: $users, in: 1...64)
                Text("\(Int(users.rounded()))").monospacedDigit().frame(width: 50, alignment: .trailing)
            }
            HStack {
                Text("Context").frame(width: 70, alignment: .leading)
                Slider(value: Binding(get: { log2(calcCtx) }, set: { calcCtx = pow(2, $0.rounded()) }), in: 9...15, step: 1)
                Text("\(Int(calcCtx))").monospacedDigit().frame(width: 50, alignment: .trailing)
            }
            Gauge(value: min(kv / max(budget, 1), 1)) {
                EmptyView()
            } currentValueLabel: {
                EmptyView()
            }
            .tint(kv > budget ? Color.red : Palette.orange)
            HStack {
                Text("\(Fmt.bytes(kv)) of KV cache").font(.title3.weight(.semibold)).monospacedDigit()
                Spacer()
                Text(kv > budget ? "does not fit" : "fits")
                    .font(.callout.weight(.semibold)).foregroundStyle(kv > budget ? .red : .secondary)
            }
            Text("Budget ≈ 75% of \(Fmt.gb(monitor.machine.memoryBytes)) unified memory − \(Fmt.bytes(r.fileBytes)) weights = \(Fmt.gb(budget))")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
