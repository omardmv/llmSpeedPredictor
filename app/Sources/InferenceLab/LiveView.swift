import Charts
import SwiftUI

struct LiveView: View {
    @Environment(Monitor.self) private var monitor

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                machineRow
                nowRow
                HStack(alignment: .top, spacing: 14) {
                    gpuChart
                    memoryChart
                }
                modelsCard
            }
            .padding(18)
        }
    }

    private var machineRow: some View {
        let m = monitor.machine
        return HStack(spacing: 12) {
            StatTile(title: "Chip", value: m.chip.replacingOccurrences(of: "Apple ", with: ""),
                     detail: "macOS \(m.macOS)")
            StatTile(title: "CPU cores", value: "\(m.performanceCores) + \(m.efficiencyCores)",
                     detail: "performance + efficiency")
            StatTile(title: "GPU cores", value: "\(m.gpuCores)", detail: "this is where the model runs")
            StatTile(title: "Unified memory", value: Fmt.gb(m.memoryBytes),
                     detail: "shared by CPU and GPU, there is no separate VRAM")
            StatTile(title: "Memory bandwidth", value: m.bandwidthGBs.map { "\($0) GB/s" } ?? "?",
                     detail: "spec sheet: bytes per second between memory and chip")
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private var nowRow: some View {
        let s = monitor.latest
        return HStack(spacing: 12) {
            StatTile(title: "GPU busy", value: "\(s?.gpuUtil ?? 0)%", color: Palette.blue)
            StatTile(title: "GPU memory in use", value: Fmt.gb(s?.gpuMemory ?? 0), color: Palette.orange)
            StatTile(title: "Memory free", value: "\(s?.memory.freePercent ?? 0)%",
                     detail: "what macOS counts as available")
            StatTile(title: "Models loaded", value: "\(monitor.loaded.count)",
                     detail: Fmt.bytes(monitor.loaded.reduce(0) { $0 + $1.size }) + " total")
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private var gpuChart: some View {
        Card(title: "GPU busy", subtitle: "last 2 minutes · send a request and watch it jump") {
            Chart(monitor.samples) { s in
                AreaMark(x: .value("time", s.time), y: .value("busy %", s.gpuUtil))
                    .foregroundStyle(Palette.blue.opacity(0.25))
                    .interpolationMethod(.monotone)
                LineMark(x: .value("time", s.time), y: .value("busy %", s.gpuUtil))
                    .foregroundStyle(Palette.blue)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                    .interpolationMethod(.monotone)
            }
            .chartYScale(domain: 0...100)
            .chartYAxis { AxisMarks(values: [0, 25, 50, 75, 100]) { v in
                AxisGridLine(); AxisValueLabel { Text("\(v.as(Int.self) ?? 0)%") }
            } }
            .chartXAxis { AxisMarks(values: .stride(by: .second, count: 30)) { _ in AxisGridLine() } }
            .frame(height: 200)
        }
    }

    private var memoryChart: some View {
        Card(title: "Unified memory", subtitle: "GPU buffers (model weights, KV cache) live in \"wired\"") {
            let total = monitor.machine.memoryBytes / 1_073_741_824
            Chart {
                ForEach(monitor.samples) { s in
                    ForEach([("apps", s.memory.app), ("wired", s.memory.wired), ("compressed", s.memory.compressed)],
                            id: \.0) { name, bytes in
                        AreaMark(x: .value("time", s.time), y: .value("GB", bytes / 1_073_741_824))
                            .foregroundStyle(by: .value("kind", name))
                    }
                }
                RuleMark(y: .value("total", total))
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .annotation(position: .top, alignment: .leading) {
                        Text("\(Int(total)) GB total").font(.caption2).foregroundStyle(.secondary)
                    }
            }
            .chartForegroundStyleScale(["apps": Palette.blue, "wired": Palette.orange, "compressed": Palette.aqua])
            .chartYScale(domain: 0...(total * 1.08))
            .chartXAxis { AxisMarks(values: .stride(by: .second, count: 30)) { _ in AxisGridLine() } }
            .chartLegend(position: .top, alignment: .leading)
            .frame(height: 200)
        }
    }

    private var modelsCard: some View {
        Card(title: "Loaded in Ollama", subtitle: "a model unloads after sitting idle; the next request is then a cold start") {
            if !monitor.ollamaUp {
                Text("Ollama is not running. Start the Ollama app or run `ollama serve`.").foregroundStyle(.secondary)
            } else if monitor.loaded.isEmpty {
                Text("Nothing loaded. The next request will have to load a model first.").foregroundStyle(.secondary)
            } else {
                TimelineView(.periodic(from: .now, by: 1)) { ctx in
                    VStack(spacing: 10) {
                        ForEach(monitor.loaded) { m in row(m, now: ctx.date) }
                    }
                }
            }
        }
    }

    private func row(_ m: Ollama.Loaded, now: Date) -> some View {
        let left = max(0, Int(m.expires?.timeIntervalSince(now) ?? 0))
        return HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text(m.name).font(.body.weight(.medium))
                Text("context \(m.contextLength ?? 0) tokens").font(.caption).foregroundStyle(.secondary)
            }
            .frame(width: 180, alignment: .leading)
            VStack(alignment: .leading, spacing: 3) {
                GeometryReader { geo in
                    let frac = m.size > 0 ? m.sizeVram / m.size : 0
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3).fill(Palette.gray.opacity(0.5))
                        RoundedRectangle(cornerRadius: 3).fill(Palette.orange).frame(width: geo.size.width * frac)
                    }
                }
                .frame(height: 8)
                Text("\(Fmt.bytes(m.size)) total · \(Fmt.bytes(m.sizeVram)) on the GPU")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text(left > 0 ? "unloads in \(left / 60)m \(left % 60)s" : "unloading…")
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                .frame(width: 120, alignment: .trailing)
            Button("Unload") { Task { await Ollama.unload(m.name); await monitor.refreshOllama() } }
                .help("Drop it from memory so the next request is a cold start")
        }
    }
}
