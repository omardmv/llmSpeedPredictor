import Charts
import SwiftUI

/// One request from a sweep (a JSON line from Python, or a row from a saved CSV).
struct SweepRequest: Identifiable {
    let id = UUID()
    let round: Int
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
        round = Int(d["round"] ?? 1)  // older CSVs have no round column
        users = Int(d["users"] ?? 0); user = Int(d["user"] ?? 0)
        startS = d["start_s"] ?? 0; endS = d["end_s"] ?? 0
        ttftS = d["ttft_s"] ?? 0; totalS = d["total_s"] ?? 0
        promptTokens = Int(d["prompt_tokens"] ?? 0); outputTokens = Int(d["output_tokens"] ?? 0)
        tpotMs = d["tpot_ms"] ?? 0
    }
}

/// Numbers for one concurrency level. With repeated rounds, each round is measured on its
/// own and the rounds are averaged; `spread` says how much they disagree.
struct SweepLevel: Identifiable {
    let users: Int
    let requests: Int
    let rounds: Int
    let ttftP50: Double
    let ttftP95: Double
    let tpotP50: Double
    let perUserTokS: Double
    let totalTokS: Double
    /// (max - min) / mean across rounds, in %: TTFT p95, tok/s per user, tok/s total.
    let spread: (ttft: Double, perUser: Double, total: Double)
    let trusted: Bool
    var id: Int { users }

    static let spreadMax = 10.0

    private struct Round { let ttftP50, ttftP95, tpotP50, perUser, total: Double }

    private static func measure(_ rows: [SweepRequest]) -> Round {
        let ttft = rows.map(\.ttftS)
        let wall = (rows.map(\.endS).max() ?? 0) - (rows.map(\.startS).min() ?? 0)
        return Round(ttftP50: percentile(ttft, 50), ttftP95: percentile(ttft, 95),
                     tpotP50: percentile(rows.map(\.tpotMs), 50),
                     perUser: percentile(rows.map { Double($0.outputTokens) / max($0.totalS - $0.ttftS, 1e-9) }, 50),
                     total: wall > 0 ? Double(rows.reduce(0) { $0 + $1.outputTokens }) / wall : 0)
    }

    init(users: Int, rows: [SweepRequest]) {
        self.users = users
        requests = rows.count
        let perRound = Dictionary(grouping: rows, by: \.round).values.map(Self.measure)
        rounds = perRound.count
        func mean(_ k: KeyPath<Round, Double>) -> Double { perRound.map { $0[keyPath: k] }.reduce(0, +) / Double(max(perRound.count, 1)) }
        func pct(_ k: KeyPath<Round, Double>) -> Double {
            let v = perRound.map { $0[keyPath: k] }
            guard v.count > 1, let lo = v.min(), let hi = v.max() else { return 0 }
            return (hi - lo) / max(mean(k), 1e-9) * 100
        }
        ttftP50 = mean(\.ttftP50); ttftP95 = mean(\.ttftP95); tpotP50 = mean(\.tpotP50)
        perUserTokS = mean(\.perUser); totalTokS = mean(\.total)
        spread = (pct(\.ttftP95), pct(\.perUser), pct(\.total))
        // Same rule as bench/sweep.py: tiny TTFTs (< 0.1 s apart) don't count as disagreeing.
        let ttfts = perRound.map(\.ttftP95)
        let ttftOK = spread.ttft <= Self.spreadMax || ((ttfts.max() ?? 0) - (ttfts.min() ?? 0)) < 0.1
        trusted = ttftOK && spread.perUser <= Self.spreadMax && spread.total <= Self.spreadMax
    }
}

/// What `python -m bench check --json` returns.
struct HeatPoint: Identifiable {
    let id = UUID()
    let t: Double
    let mhz: Double
    let temp: Double
}

struct QuietCheck: Decodable {
    let quiet: Bool
    let problems: [String]
    let gpuBusyAvg: Double
    let cpuBusyMachine: Double
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
    @Local private var maxTokens = 400.0
    @Local private var rows: [SweepRequest] = []
    @Local private var gpuByLevel: [Int: Double] = [:]
    /// Per level, one dict per round: gpu_temp_avg, gpu_mhz_avg, tokens_per_joule, ... (from macmon)
    @Local private var heatByLevel: [Int: [[String: Double]]] = [:]
    /// GPU clock and temperature over the whole run, from the run's .json (t = seconds; negative = warm-up)
    @Local private var heatSeries: [HeatPoint] = []
    @Local private var running: Task<Void, Never>?
    @Local private var progress = ""
    @Local private var error: String?
    @Local private var ganttLevel = 0
    @Local private var slo = 2.0
    @Local private var savedFiles: [URL] = []
    @Local private var title = ""
    @AppStorage(ServerChoice.backendKey) private var backend = "ollama"
    @AppStorage(ServerChoice.urlKey) private var url = "http://localhost:8080"
    @Local private var repeatRounds = 2
    @Local private var checking = false
    @Local private var noisyCheck: QuietCheck?
    @Local private var lastCheck: QuietCheck?
    private static let perUserMin = 20.0  // second half of the ticket's SLO

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
                    if !heatSeries.isEmpty { heatCard }
                } else if running == nil {
                    Text("Each \"user\" sends requests back to back, each with a different real prompt. Start with the defaults; it takes a few minutes.")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(18)
        }
        .task {
            listSaved()
            if Snapshot.isActive(.load) {
                if ProcessInfo.processInfo.environment["INFERENCE_LAB_SNAPSHOT_RUN"] != nil {
                    // tiny real run, to check the quiet check + rounds path end to end
                    model = "fast"; chosenLevels = [1, 2]; perUser = 1; maxTokens = 30; repeatRounds = 2
                    await checkThenRun()
                } else if let latest = savedFiles.first {
                    open(latest)
                }
            }
        }
    }

    // MARK: controls

    private var controls: some View {
        Card(title: "Load test") {
            ServerPicker()
            // Row 1: what to test
            HStack(spacing: 12) {
                ModelPicker(model: $model)
                Text("Users at once:").fixedSize()
                ForEach(Self.options, id: \.self) { n in
                    Toggle("\(n)", isOn: Binding(get: { chosenLevels.contains(n) },
                                                 set: { on in if on { chosenLevels.insert(n) } else { chosenLevels.remove(n) } }))
                        .toggleStyle(.button)
                        .fixedSize()
                }
                Spacer(minLength: 0)
            }
            // Row 2: how much traffic
            HStack(spacing: 18) {
                Stepper("\(perUser) requests per user", value: $perUser, in: 1...10)
                    .fixedSize()
                    .help("How many requests each simulated user sends, one after another, at every level")
                HStack {
                    Text("Answer limit (max_tokens)").fixedSize()
                    Slider(value: Binding(get: { maxTokens }, set: { maxTokens = ($0 / 10).rounded() * 10 }), in: 30...800).frame(minWidth: 120, maxWidth: 240)
                    Text("\(Int(maxTokens)) tokens").monospacedDigit().fixedSize()
                }
                .help("The most tokens the model may write per answer. Longer answers = more decode time per request")
                Stepper("Run it \(repeatRounds)×", value: $repeatRounds, in: 1...3)
                    .fixedSize()
                    .help("Repeat the whole test and compare the rounds. If they differ by more than 10%, that level can't be trusted")
                Spacer(minLength: 0)
            }
            // Row 3: actions
            HStack(spacing: 12) {
                Menu("Open earlier run") {
                    ForEach(savedFiles, id: \.self) { url in Button(url.deletingPathExtension().lastPathComponent) { open(url) } }
                }
                .fixedSize()
                .disabled(savedFiles.isEmpty)
                Spacer(minLength: 0)
                if checking {
                    ProgressView().controlSize(.small)
                    Text("checking the machine is quiet…").font(.callout).foregroundStyle(.secondary)
                } else if running != nil {
                    ProgressView().controlSize(.small)
                    Text(progress).font(.callout.monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
                    Button("Stop") { running?.cancel() }.fixedSize()
                } else {
                    Button("Run load test") { Task { await checkThenRun() } }
                        .buttonStyle(.borderedProminent)
                        .fixedSize()
                        .disabled(chosenLevels.isEmpty)
                }
            }
            if let c = lastCheck {
                Label(c.quiet ? "Machine was quiet when this run started"
                              : "Started on a busy machine: " + c.problems.joined(separator: " · "),
                      systemImage: c.quiet ? "checkmark.seal" : "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(c.quiet ? Color.secondary : Color.orange)
            }
        }
        .alert("The machine isn't quiet", isPresented: Binding(get: { noisyCheck != nil }, set: { if !$0 { noisyCheck = nil } })) {
            Button("Run anyway", role: .destructive) { lastCheck = noisyCheck; noisyCheck = nil; run() }
            Button("Cancel", role: .cancel) { noisyCheck = nil }
        } message: {
            Text((noisyCheck?.problems ?? []).map { "• " + $0 }.joined(separator: "\n")
                 + "\n\nOther work shares the GPU and memory bandwidth with the model, so results may be off. Close these and try again.")
        }
    }

    /// Measure the machine for 5 s first; only ask the user if something is busy.
    private func checkThenRun() async {
        checking = true
        defer { checking = false }
        do {
            for try await line in Bench.lines(["check"]) {
                let c = try Bench.decode(QuietCheck.self, line)
                if c.quiet { lastCheck = c; run() } else { noisyCheck = c }
            }
        } catch {
            self.error = error.localizedDescription
        }
    }

    private func run() {
        error = nil
        rows = []
        gpuByLevel = [:]
        heatByLevel = [:]
        heatSeries = []
        title = "\(resolveModel(model)) · \(backend == "llama" ? "llama-server" : "Ollama") · \(perUser) requests per user · max_tokens \(Int(maxTokens)) · \(repeatRounds) round(s)"
        let levelList = chosenLevels.sorted()
        let total = levelList.reduce(0) { $0 + $1 * perUser } * repeatRounds
        progress = "warming up…"
        running = Task {
            do {
                let args = ["sweep", model, "--users", levelList.map(String.init).joined(separator: ","),
                            "--requests", "\(perUser)", "--max-tokens", "\(Int(maxTokens))",
                            "--repeat", "\(repeatRounds)", "--force"] + ServerChoice.args(backend: backend, url: url)
                for try await line in Bench.lines(args) {
                    guard let obj = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else { continue }
                    let nums = obj.compactMapValues { ($0 as? NSNumber)?.doubleValue }
                    switch obj["type"] as? String {
                    case "level_start":
                        let round = repeatRounds > 1 ? "round \(Int(nums["round"] ?? 1)) · " : ""
                        progress = "\(round)\(Int(nums["users"] ?? 0)) users…  \(rows.count)/\(total) requests"
                    case "request":
                        rows.append(SweepRequest(nums))
                        ganttLevel = rows.last?.users ?? 0
                        let round = repeatRounds > 1 ? "round \(rows.last?.round ?? 1) · " : ""
                        progress = "\(round)\(ganttLevel) users…  \(rows.count)/\(total) requests"
                    case "warmup":
                        progress = String(format: "warm-up %.0fs · GPU %.0f °C · %.0f MHz · +%.1f °C/min",
                                          nums["seconds"] ?? 0, nums["gpu_temp"] ?? 0, nums["gpu_mhz"] ?? 0, nums["rise"] ?? 0)
                    case "level":
                        let u = Int(nums["users"] ?? 0)
                        gpuByLevel[u] = nums["gpu_util_avg"]
                        heatByLevel[u, default: []].append(nums)
                    case "done":
                        listSaved()
                        if let meta = obj["meta"] as? String { loadMeta(URL(fileURLWithPath: meta)) }
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
        let files = (try? FileManager.default.contentsOfDirectory(at: Bench.benchResults,
                                                                  includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        func date(_ u: URL) -> Date { (try? u.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast }
        // Newest first, whatever the model name in the file is.
        savedFiles = files.filter { $0.pathExtension == "csv" && $0.lastPathComponent.hasPrefix("sweep_") }
            .sorted { date($0) > date($1) }
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
        heatByLevel = [:]
        heatSeries = []
        ganttLevel = rows.map(\.users).max() ?? 0
        title = url.deletingPathExtension().lastPathComponent
        lastCheck = nil
        error = nil
        loadMeta(url.deletingPathExtension().appendingPathExtension("json"))
    }

    /// The .json saved next to each CSV: per-round level numbers (GPU busy, heat) and the heat timeline.
    private func loadMeta(_ url: URL) {
        guard let data = try? Data(contentsOf: url),
              let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        let num = { (v: Any?) in (v as? NSNumber)?.doubleValue }
        if let rounds = meta["rounds"] as? [String: [[String: Any]]] {
            for (key, list) in rounds {
                guard let u = Int(key) else { continue }
                heatByLevel[u] = list.map { $0.compactMapValues(num) }
                gpuByLevel[u] = heatByLevel[u]?.compactMap { $0["gpu_util_avg"] }.average
            }
        }
        heatSeries = (meta["heat"] as? [[String: Any]] ?? []).compactMap { p in
            guard let t = num(p["t"]), let mhz = num(p["gpu_mhz"]), let temp = num(p["gpu_temp"]) else { return nil }
            return HeatPoint(t: t, mhz: mhz, temp: temp)
        }
    }

    private func heat(_ users: Int, _ key: String) -> Double? {
        heatByLevel[users]?.compactMap { $0[key] }.average
    }

    // MARK: results

    private var tiles: some View {
        let lv = levels
        let best = lv.max { $0.totalTokS < $1.totalTokS }
        let meeting = lv.filter { $0.ttftP95 <= slo && $0.perUserTokS >= Self.perUserMin }.map(\.users).max()
        let rounds = lv.map(\.rounds).max() ?? 1
        let untrusted = lv.filter { !$0.trusted }.map { "\($0.users)" }
        return HStack(spacing: 12) {
            StatTile(title: "Requests done", value: "\(rows.count)", detail: "\(lv.count) concurrency levels")
            StatTile(title: "Rounds agree?",
                     value: rounds < 2 ? "1 round" : (untrusted.isEmpty ? "yes" : "no"),
                     detail: rounds < 2 ? "repeat the test to see if the numbers hold"
                                        : (untrusted.isEmpty ? "all levels within \(Int(SweepLevel.spreadMax))%"
                                                             : "differ > \(Int(SweepLevel.spreadMax))% at \(untrusted.joined(separator: ", ")) users"),
                     color: rounds < 2 ? nil : (untrusted.isEmpty ? Palette.aqua : Color.red))
            StatTile(title: "Best total throughput", value: String(format: "%.0f tok/s", best?.totalTokS ?? 0),
                     detail: "at \(best?.users ?? 0) users")
            StatTile(title: "Throughput 1 user → most users",
                     value: String(format: "%.1f×", (lv.last?.totalTokS ?? 0) / max(lv.first?.totalTokS ?? 1, 1e-9)),
                     detail: "≈1× means requests are queued, not batched")
            VStack(alignment: .leading, spacing: 4) {
                Text("SLO: p95 time to first token ≤ \(String(format: "%.1f", slo)) s and ≥ \(Int(Self.perUserMin)) tok/s per user")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
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
        // With repeated rounds, show round 1 only, so the timeline isn't two runs on top of each other.
        let reqs = rows.filter { $0.users == level && $0.round == 1 }.sorted { $0.startS < $1.startS }
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

    /// Clock and temperature over the run, as two charts on the same time axis (never one chart with two scales).
    private var heatCard: some View {
        Card(title: "Heat during the run",
             subtitle: "grey area = warm-up · when the temperature climbs, watch whether the clock (and tok/s) drops") {
            let tMin = heatSeries.map(\.t).min() ?? 0, tMax = heatSeries.map(\.t).max() ?? 1
            let temps = heatSeries.map(\.temp)
            Text("GPU clock (MHz)").font(.caption).foregroundStyle(.secondary)
            Chart {
                if tMin < 0 { RectangleMark(xStart: .value("s", tMin), xEnd: .value("s", 0)).foregroundStyle(.gray.opacity(0.15)) }
                ForEach(heatSeries) { p in
                    LineMark(x: .value("seconds", p.t), y: .value("MHz", p.mhz)).foregroundStyle(Palette.yellow)
                }
                if let top = HeatReader.maxGPUMHz {
                    RuleMark(y: .value("max", top)).foregroundStyle(.secondary).lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                }
            }
            .chartXScale(domain: tMin...tMax)
            .chartYScale(domain: 0...((HeatReader.maxGPUMHz ?? 1600) * 1.1))
            .frame(height: 140)
            Text("GPU temperature (°C)").font(.caption).foregroundStyle(.secondary)
            Chart {
                if tMin < 0 { RectangleMark(xStart: .value("s", tMin), xEnd: .value("s", 0)).foregroundStyle(.gray.opacity(0.15)) }
                ForEach(heatSeries) { p in
                    LineMark(x: .value("seconds", p.t), y: .value("°C", p.temp)).foregroundStyle(Palette.aqua)
                }
            }
            .chartXScale(domain: tMin...tMax)
            .chartYScale(domain: ((temps.min() ?? 30) - 3)...((temps.max() ?? 60) + 3))
            .chartXAxisLabel("seconds (negative = warm-up)")
            .frame(height: 140)
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
                TableColumn("GPU busy · °C · clock") { l in
                    let busy = gpuByLevel[l.users].map { String(format: "%.0f%%", $0) } ?? "–"
                    if let temp = heat(l.users, "gpu_temp_avg"), let mhz = heat(l.users, "gpu_mhz_avg") {
                        Text("\(busy) · \(Int(temp)) °C · \(Int(mhz)) MHz").monospacedDigit()
                    } else {
                        Text(busy)
                    }
                }
                TableColumn("tok/J") { l in
                    Text(heat(l.users, "tokens_per_joule").map { String(format: "%.1f", $0) } ?? "–").monospacedDigit()
                        .help("Tokens per joule of chip energy: higher = cheaper to serve")
                }
                TableColumn("rounds agree?") { l in
                    if l.rounds < 2 {
                        Text("–").foregroundStyle(.secondary)
                    } else {
                        Text(l.trusted ? "yes" : String(format: "no · %.0f%%", max(l.spread.ttft, l.spread.perUser, l.spread.total)))
                            .foregroundStyle(l.trusted ? Color.secondary : Color.red)
                            .help(String(format: "TTFT p95 %.1f%% · tok/s per user %.1f%% · tok/s total %.1f%%",
                                         l.spread.ttft, l.spread.perUser, l.spread.total))
                    }
                }
            }
            .frame(height: CGFloat(levels.count) * 26 + 34)
        }
    }
}

extension Array where Element == Double {
    var average: Double? { isEmpty ? nil : reduce(0, +) / Double(count) }
}
