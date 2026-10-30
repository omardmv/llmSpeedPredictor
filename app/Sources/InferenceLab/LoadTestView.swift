import Charts
import SwiftUI

/// One request from a sweep (a JSON line from Python, or a row from a saved CSV).
struct SweepRequest: Identifiable {
    let id = UUID()
    let users: Int
    let user: Int
    let startS: Double
    let endS: Double
    let ttftS: Double
    let totalS: Double
    let promptTokens: Int
    let outputTokens: Int
    let tpotMs: Double

    init(_ d: [String: Double]) {
        users = Int(d["users"] ?? 0); user = Int(d["user"] ?? 0)
        startS = d["start_s"] ?? 0; endS = d["end_s"] ?? 0
        ttftS = d["ttft_s"] ?? 0; totalS = d["total_s"] ?? 0
        promptTokens = Int(d["prompt_tokens"] ?? 0); outputTokens = Int(d["output_tokens"] ?? 0)
        tpotMs = d["tpot_ms"] ?? 0
    }
}

/// Numbers for one concurrency level, computed from its requests.
struct SweepLevel: Identifiable {
    let users: Int
    let requests: Int
    let ttftP50: Double
    let ttftP95: Double
    let tpotP50: Double
    let perUserTokS: Double
    let totalTokS: Double
    var id: Int { users }

    init(users: Int, rows: [SweepRequest]) {
        self.users = users
        requests = rows.count
        let ttft = rows.map(\.ttftS)
        ttftP50 = percentile(ttft, 50)
        ttftP95 = percentile(ttft, 95)
        tpotP50 = percentile(rows.map(\.tpotMs), 50)
        perUserTokS = percentile(rows.map { Double($0.outputTokens) / max($0.totalS - $0.ttftS, 1e-9) }, 50)
        let wall = (rows.map(\.endS).max() ?? 0) - (rows.map(\.startS).min() ?? 0)
        totalTokS = wall > 0 ? Double(rows.reduce(0) { $0 + $1.outputTokens }) / wall : 0
    }
}

/// Same as numpy.percentile (linear interpolation).
func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let s = values.sorted()
    let pos = Double(s.count - 1) * p / 100
    let lo = Int(pos.rounded(.down)), hi = Int(pos.rounded(.up))
    return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
}

struct LoadTestView: View {
    @Local private var model = "quality"
    @Local private var chosenLevels: Set<Int> = [1, 2, 4, 8, 16]
    @Local private var perUser = 3
    @Local private var maxTokens = 200.0
    @Local private var rows: [SweepRequest] = []
    @Local private var gpuByLevel: [Int: Double] = [:]
    @Local private var running: Task<Void, Never>?
    @Local private var progress = ""
    @Local private var error: String?
    @Local private var ganttLevel = 0
    @Local private var slo = 2.0
    @Local private var savedFiles: [URL] = []
    @Local private var title = ""

    private static let options = [1, 2, 4, 8, 16, 32]

    private var levels: [SweepLevel] {
        Dictionary(grouping: rows, by: \.users).keys.sorted().map { u in SweepLevel(users: u, rows: rows.filter { $0.users == u }) }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                controls
                if let error { ErrorBanner(message: error) }
                if !rows.isEmpty {
                    if !title.isEmpty { Text(title).font(.title3.weight(.semibold)) }
                    tiles
                    HStack(alignment: .top, spacing: 14) { ttftCard; perUserCard }
                    HStack(alignment: .top, spacing: 14) { totalCard; ganttCard }
                    tableCard
                } else if running == nil {
                    Text("Each \"user\" sends requests back to back, each with a different real prompt. Start with the defaults; it takes a few minutes.")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(18)
        }
        .task {
            listSaved()
            if Snapshot.isActive(.load), let latest = savedFiles.first { open(latest) }
        }
    }

    // MARK: controls

    private var controls: some View {
        Card(title: "Load test") {
            HStack(spacing: 12) {
                ModelPicker(model: $model)
                Text("Users at once:")
                ForEach(Self.options, id: \.self) { n in
                    Toggle("\(n)", isOn: Binding(get: { chosenLevels.contains(n) },
                                                 set: { on in if on { chosenLevels.insert(n) } else { chosenLevels.remove(n) } }))
                        .toggleStyle(.button)
                }
                Spacer()
            }
            HStack(spacing: 18) {
                Stepper("\(perUser) requests per user", value: $perUser, in: 1...10)
                HStack {
                    Text("max_tokens")
                    Slider(value: $maxTokens, in: 32...512).frame(width: 180)
                    Text("\(Int(maxTokens))").monospacedDigit().frame(width: 36, alignment: .trailing)
                }
                Spacer()
                Menu("Open earlier run") {
                    ForEach(savedFiles, id: \.self) { url in Button(url.deletingPathExtension().lastPathComponent) { open(url) } }
                }
                .frame(width: 170)
                .disabled(savedFiles.isEmpty)
                if running != nil {
                    ProgressView().controlSize(.small)
                    Text(progress).font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                    Button("Stop") { running?.cancel() }
                } else {
                    Button("Run") { run() }.buttonStyle(.borderedProminent).disabled(chosenLevels.isEmpty)
                }
            }
        }
    }

    private func run() {
        error = nil
        rows = []
        gpuByLevel = [:]
        title = "\(resolveModel(model)) · \(perUser) requests per user · max_tokens \(Int(maxTokens))"
        let levelList = chosenLevels.sorted()
        let total = levelList.reduce(0) { $0 + $1 * perUser }
        progress = "warming up…"
        running = Task {
            do {
                let args = ["sweep", model, "--users", levelList.map(String.init).joined(separator: ","),
                            "--requests", "\(perUser)", "--max-tokens", "\(Int(maxTokens))"]
                for try await line in Bench.lines(args) {
                    guard let obj = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else { continue }
                    let nums = obj.compactMapValues { ($0 as? NSNumber)?.doubleValue }
                    switch obj["type"] as? String {
                    case "level_start":
                        progress = "\(Int(nums["users"] ?? 0)) users…  \(rows.count)/\(total) requests"
                    case "request":
                        rows.append(SweepRequest(nums))
                        ganttLevel = rows.last?.users ?? 0
                        progress = "\(ganttLevel) users…  \(rows.count)/\(total) requests"
                    case "level":
                        gpuByLevel[Int(nums["users"] ?? 0)] = nums["gpu_util_avg"]
                    case "done":
                        listSaved()
                    default: break
                    }
                }
            } catch is CancellationError {
            } catch {
                self.error = error.localizedDescription
            }
            running = nil
        }
    }

    private func listSaved() {
        let files = (try? FileManager.default.contentsOfDirectory(at: Bench.benchResults, includingPropertiesForKeys: nil)) ?? []
        savedFiles = files.filter { $0.pathExtension == "csv" && $0.lastPathComponent.hasPrefix("sweep_") }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    private func open(_ url: URL) {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return }
        let lines = text.split(whereSeparator: \.isNewline)
        guard let header = lines.first?.split(separator: ",").map(String.init) else { return }
        rows = lines.dropFirst().map { line in
            let vals = line.split(separator: ",").map { Double($0) ?? 0 }
            return SweepRequest(Dictionary(uniqueKeysWithValues: zip(header, vals)))
        }
        gpuByLevel = [:]
        ganttLevel = rows.map(\.users).max() ?? 0
        title = url.deletingPathExtension().lastPathComponent
        error = nil
    }

    // MARK: results

    private var tiles: some View {
        let lv = levels
        let best = lv.max { $0.totalTokS < $1.totalTokS }
        let meeting = lv.filter { $0.ttftP95 <= slo }.map(\.users).max()
        return HStack(spacing: 12) {
            StatTile(title: "Requests done", value: "\(rows.count)", detail: "\(lv.count) concurrency levels")
            StatTile(title: "Best total throughput", value: String(format: "%.0f tok/s", best?.totalTokS ?? 0),
                     detail: "at \(best?.users ?? 0) users")
            StatTile(title: "Throughput 1 user → most users",
                     value: String(format: "%.1f×", (lv.last?.totalTokS ?? 0) / max(lv.first?.totalTokS ?? 1, 1e-9)),
                     detail: "≈1× means requests are queued, not batched")
            VStack(alignment: .leading, spacing: 4) {
                Text("SLO: p95 time to first token ≤ \(String(format: "%.1f", slo)) s").font(.caption).foregroundStyle(.secondary)
                Slider(value: $slo, in: 0.1...10)
                Text(meeting.map { "met up to \($0) users" } ?? "not met at any level")
                    .font(.system(size: 20, weight: .semibold, design: .rounded))
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private var ttftCard: some View {
        Card(title: "Time to first token", subtitle: "how long a user stares at an empty reply") {
            Chart {
                ForEach(levels) { l in
                    LineMark(x: .value("users", l.users), y: .value("s", l.ttftP50), series: .value("", "p50"))
                        .foregroundStyle(by: .value("", "p50"))
                    PointMark(x: .value("users", l.users), y: .value("s", l.ttftP50)).foregroundStyle(by: .value("", "p50"))
                    LineMark(x: .value("users", l.users), y: .value("s", l.ttftP95), series: .value("", "p95"))
                        .foregroundStyle(by: .value("", "p95"))
                    PointMark(x: .value("users", l.users), y: .value("s", l.ttftP95)).foregroundStyle(by: .value("", "p95"))
                }
                RuleMark(y: .value("SLO", slo))
                    .foregroundStyle(.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .annotation(position: .top, alignment: .leading) { Text("SLO").font(.caption2).foregroundStyle(.secondary) }
            }
            .chartForegroundStyleScale(["p50": Palette.blue, "p95": Palette.orange])
            .chartXScale(type: .log)
            .chartXAxis { AxisMarks(values: levels.map(\.users)) }
            .chartXAxisLabel("users at once")
            .chartYAxisLabel("seconds")
            .chartLegend(position: .top, alignment: .leading)
            .frame(height: 220)
        }
    }

    private var perUserCard: some View {
        Card(title: "Speed each user sees", subtitle: "tokens per second while their answer streams") {
            levelChart(\.perUserTokS, unit: "tokens / second")
        }
    }

    private var totalCard: some View {
        Card(title: "Total server throughput", subtitle: "all output tokens ÷ wall time, across every user") {
            levelChart(\.totalTokS, unit: "tokens / second")
        }
    }

    private func levelChart(_ key: KeyPath<SweepLevel, Double>, unit: String) -> some View {
        Chart(levels) { l in
            LineMark(x: .value("users", l.users), y: .value(unit, l[keyPath: key])).foregroundStyle(Palette.blue)
            PointMark(x: .value("users", l.users), y: .value(unit, l[keyPath: key])).foregroundStyle(Palette.blue)
                .annotation(position: .top) { Text(String(format: "%.0f", l[keyPath: key])).font(.caption2).foregroundStyle(.secondary) }
        }
        .chartXScale(type: .log)
        .chartXAxis { AxisMarks(values: levels.map(\.users)) }
        .chartYScale(domain: 0...((levels.map { $0[keyPath: key] }.max() ?? 1) * 1.2))
        .chartXAxisLabel("users at once")
        .chartYAxisLabel(unit)
        .frame(height: 220)
    }

    private var ganttCard: some View {
        let level = levels.contains { $0.users == ganttLevel } ? ganttLevel : (levels.last?.users ?? 0)
        let reqs = rows.filter { $0.users == level }.sorted { $0.startS < $1.startS }
        let t0 = reqs.first?.startS ?? 0
        return Card(title: "Every request at \(level) users", subtitle: "grey = waiting for the first token · blue = streaming the answer") {
            Picker("Level", selection: $ganttLevel) {
                ForEach(levels) { l in Text("\(l.users)").tag(l.users) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            Chart(Array(reqs.enumerated()), id: \.offset) { i, r in
                BarMark(xStart: .value("s", r.startS - t0), xEnd: .value("s", r.startS - t0 + r.ttftS),
                        y: .value("request", "\(i)"), height: .ratio(0.75))
                    .foregroundStyle(Palette.gray)
                BarMark(xStart: .value("s", r.startS - t0 + r.ttftS), xEnd: .value("s", r.endS - t0),
                        y: .value("request", "\(i)"), height: .ratio(0.75))
                    .foregroundStyle(Palette.blue)
            }
            .chartYAxis(.hidden)
            .chartXAxisLabel("seconds")
            .frame(height: 190)
        }
    }

    private var tableCard: some View {
        Card(title: "By level") {
            Table(levels) {
                TableColumn("users") { l in Text("\(l.users)").monospacedDigit() }
                TableColumn("requests") { l in Text("\(l.requests)").monospacedDigit() }
                TableColumn("TTFT p50") { l in Text(Fmt.secs(l.ttftP50)).monospacedDigit() }
                TableColumn("TTFT p95") { l in Text(Fmt.secs(l.ttftP95)).monospacedDigit() }
                TableColumn("ms / token") { l in Text(String(format: "%.1f", l.tpotP50)).monospacedDigit() }
                TableColumn("tok/s per user") { l in Text(String(format: "%.1f", l.perUserTokS)).monospacedDigit() }
                TableColumn("tok/s total") { l in Text(String(format: "%.1f", l.totalTokS)).monospacedDigit() }
                TableColumn("GPU busy") { l in Text(gpuByLevel[l.users].map { String(format: "%.0f%%", $0) } ?? "–") }
            }
            .frame(height: CGFloat(levels.count) * 26 + 34)
        }
    }
}
