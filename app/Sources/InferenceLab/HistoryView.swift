import AppKit
import Charts
import SwiftUI

/// One saved load test: research/bench/sweep_*.csv plus the .json written next to it (newer runs).
struct RunRecord: Identifiable, Hashable {
    let id: String              // file name without extension
    let csv: URL
    let date: Date
    let model: String
    let server: String          // "Ollama", "llama-server"
    let slots: Int?
    let ctxPerSlot: Int?
    let levels: [SweepLevel]
    let rounds: Int
    let maxTokens: Int?
    let requestsPerUser: Int?
    let quiet: Bool?
    let warmTemp: Double?       // GPU °C when measuring started (after warm-up)
    let tokensPerJoule: [Int: Double]
    let clockByLevel: [Int: Double]
    let longestAnswer: Int      // from the data: shows the answer limit even for runs without a .json

    static func == (a: RunRecord, b: RunRecord) -> Bool { a.id == b.id }
    func hash(into h: inout Hasher) { h.combine(id) }

    var setup: String {
        server == "llama-server" ? "llama-server · \(slots.map { "\($0) slot\($0 == 1 ? "" : "s")" } ?? "?") · ctx \(ctxPerSlot ?? 0)"
                                 : "Ollama"
    }
    /// Short name for chart legends.
    var shortSetup: String {
        server == "llama-server" ? "llama ×\(slots ?? 0)" : "Ollama"
    }
    var userLevels: String {
        let u = levels.map(\.users)
        return u.count > 1 ? "\(u.first!)–\(u.last!) users" : "\(u.first ?? 0) users"
    }

    /// Largest level meeting the SLO: p95 time to first token and per-user speed.
    func usersServed(ttft: Double, perUser: Double) -> Int {
        levels.filter { $0.ttftP95 <= ttft && $0.perUserTokS >= perUser }.map(\.users).max() ?? 0
    }
    var bestTotal: Double { levels.map(\.totalTokS).max() ?? 0 }
}

enum RunFiles {
    /// Rows of a sweep CSV (header + numbers).
    static func readCSV(_ url: URL) -> [SweepRequest] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let lines = text.split(whereSeparator: \.isNewline)
        guard let header = lines.first?.split(separator: ",").map(String.init) else { return [] }
        return lines.dropFirst().map { line in
            let vals = line.split(separator: ",").map { Double($0) ?? 0 }
            return SweepRequest(Dictionary(uniqueKeysWithValues: zip(header, vals)))
        }
    }

    static func loadAll() -> [RunRecord] {
        let files = (try? FileManager.default.contentsOfDirectory(at: Bench.benchResults, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "csv" && $0.lastPathComponent.hasPrefix("sweep_") }
            .compactMap(load)
            .sorted { $0.date > $1.date }
    }

    static func load(_ csv: URL) -> RunRecord? {
        let rows = readCSV(csv)
        guard !rows.isEmpty else { return nil }
        let base = csv.deletingPathExtension().lastPathComponent
        let meta = (try? Data(contentsOf: csv.deletingPathExtension().appendingPathExtension("json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
        let server = meta["server"] as? [String: Any]
        let settings = meta["settings"] as? [String: Any]
        let before = meta["before"] as? [String: Any]
        let num = { (v: Any?) in (v as? NSNumber)?.doubleValue }

        // Name: sweep_<model>[_<backend tag>]_<yyyyMMdd-HHmmss>
        let parts = base.split(separator: "_").map(String.init)
        let stamp = parts.last ?? ""
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        let tag = parts.count > 3 ? parts[2] : ""
        let isLlama = (server?["backend"] as? String) == "llama-server" || tag.hasPrefix("llama")

        // Per-level energy and clock from the .json (averaged over rounds).
        var tpj: [Int: Double] = [:], clock: [Int: Double] = [:]
        for (key, list) in meta["rounds"] as? [String: [[String: Any]]] ?? [:] {
            guard let u = Int(key) else { continue }
            tpj[u] = list.compactMap { num($0["tokens_per_joule"]) }.average
            clock[u] = list.compactMap { num($0["gpu_mhz_avg"]) }.average
        }
        let heat = meta["heat"] as? [[String: Any]] ?? []
        let firstMeasured = heat.first { (num($0["t"]) ?? -1) >= 0 }

        return RunRecord(
            id: base, csv: csv, date: fmt.date(from: stamp) ?? .distantPast,
            model: settings?["model"] as? String ?? parts.dropFirst().first ?? "?",
            server: isLlama ? "llama-server" : "Ollama",
            slots: (server?["slots"] as? Int) ?? (isLlama ? Int(tag.split(separator: "-").first { $0.hasPrefix("np") }?.dropFirst(2) ?? "") : 1),
            ctxPerSlot: server?["ctx_per_slot"] as? Int,
            levels: Dictionary(grouping: rows, by: \.users).keys.sorted().map { u in SweepLevel(users: u, rows: rows.filter { $0.users == u }) },
            rounds: Set(rows.map(\.round)).count,
            maxTokens: settings?["max_tokens"] as? Int,
            requestsPerUser: settings?["requests_per_user"] as? Int,
            quiet: before?["quiet"] as? Bool,
            warmTemp: firstMeasured.flatMap { num($0["gpu_temp"]) },
            tokensPerJoule: tpj,
            clockByLevel: clock,
            longestAnswer: rows.map(\.outputTokens).max() ?? 0)
    }
}

/// Notes you write about a run ("OBS recording", "np4 c16384"), kept in the app's settings.
enum RunNotes {
    static func get(_ id: String) -> String { (UserDefaults.standard.dictionary(forKey: "runNotes") as? [String: String])?[id] ?? "" }
    static func set(_ id: String, _ note: String) {
        var all = UserDefaults.standard.dictionary(forKey: "runNotes") as? [String: String] ?? [:]
        all[id] = note.isEmpty ? nil : note
        UserDefaults.standard.set(all, forKey: "runNotes")
    }
}

struct HistoryView: View {
    let active: Bool
    @Local private var runs: [RunRecord] = []
    @Local private var selection: Set<String> = []
    @Local private var noteEditing: RunRecord?
    @Local private var noteText = ""
    @Local private var notesVersion = 0
    @AppStorage("baselineRun") private var baselineID = ""

    // The ticket's SLO, same as the Load test screen.
    private let sloTTFT = 2.0
    private let sloPerUser = 20.0
    private static let maxCompared = 6

    private var compared: [RunRecord] {
        runs.filter { selection.contains($0.id) }.sorted { $0.date < $1.date }.prefix(Self.maxCompared).map { $0 }
    }
    private var baseline: RunRecord? { runs.first { $0.id == baselineID } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                header
                listCard
                if compared.isEmpty {
                    Text("Select runs in the list (⌘-click for several) to compare them. Right-click a run to set it as the baseline or add a note.")
                        .foregroundStyle(.secondary)
                } else {
                    compareTiles
                    LazyVGrid(columns: [GridItem(.flexible(), spacing: 14), GridItem(.flexible(), spacing: 14)], spacing: 14) {
                        compareChart("Time to first token, p95", "seconds", \.ttftP95, slo: sloTTFT)
                        compareChart("Total server throughput", "tokens / second", \.totalTokS)
                        compareChart("Speed each user sees", "tokens / second", \.perUserTokS, slo: sloPerUser)
                        energyChart
                    }
                }
            }
            .padding(18)
        }
        .task(id: active) {
            if active || Snapshot.isActive(.history) { reload() }
            if Snapshot.isActive(.history), selection.isEmpty {
                selection = Set(runs.prefix(3).map(\.id))
                if baselineID.isEmpty { baselineID = runs.last?.id ?? "" }
            }
        }
        .alert("Note for this run", isPresented: Binding(get: { noteEditing != nil }, set: { if !$0 { noteEditing = nil } })) {
            TextField("e.g. OBS recording, np4 c16384", text: $noteText)
            Button("Save") {
                if let r = noteEditing { RunNotes.set(r.id, noteText); notesVersion += 1 }
                noteEditing = nil
            }
            Button("Cancel", role: .cancel) { noteEditing = nil }
        }
    }

    private func reload() {
        runs = RunFiles.loadAll()
        selection = selection.filter { id in runs.contains { $0.id == id } }
    }

    private var header: some View {
        HStack {
            Text("\(runs.count) saved load tests in research/bench").foregroundStyle(.secondary)
            Spacer()
            Text("SLO: p95 first token ≤ \(String(format: "%.1f", sloTTFT)) s and ≥ \(Int(sloPerUser)) tok/s per user")
                .font(.caption).foregroundStyle(.secondary)
            Button { reload() } label: { Label("Reload", systemImage: "arrow.clockwise") }
        }
    }

    // MARK: list

    private func label(_ r: RunRecord) -> String {
        _ = notesVersion
        let note = RunNotes.get(r.id)
        return note.isEmpty ? r.setup : "\(r.setup) · \(note)"
    }

    /// Label + time, so two runs with the same setup stay separate lines.
    private func chartLabel(_ r: RunRecord) -> String {
        _ = notesVersion
        let note = RunNotes.get(r.id)
        return [r.shortSetup, note.isEmpty ? nil : note,
                r.date.formatted(.dateTime.month(.abbreviated).day().hour().minute())].compactMap { $0 }.joined(separator: " · ")
    }

    private func color(_ r: RunRecord) -> Color? {
        guard let i = compared.firstIndex(of: r) else { return nil }
        return Palette.series[i]
    }

    private var listCard: some View {
        Card(title: "Runs", subtitle: "newest first · ★ = baseline every other run is compared against") {
            Table(runs, selection: $selection) {
                TableColumn("") { r in
                    HStack(spacing: 4) {
                        if let c = color(r) { Circle().fill(c).frame(width: 9, height: 9) }
                        if r.id == baselineID { Image(systemName: "star.fill").foregroundStyle(Palette.yellow).font(.caption) }
                    }
                }
                .width(34)
                TableColumn("when") { r in Text(r.date.formatted(date: .abbreviated, time: .shortened)).monospacedDigit() }
                    .width(min: 110, ideal: 130)
                TableColumn("server · note") { r in Text(label(r)).lineLimit(1).help(label(r)) }
                    .width(min: 160, ideal: 260)
                TableColumn("load") { r in
                    Text([r.userLevels, r.requestsPerUser.map { "\($0) req" }, r.maxTokens.map { "\($0) tok" } ?? "≤\(r.longestAnswer) tok",
                          r.rounds > 1 ? "\(r.rounds) rounds" : nil].compactMap { $0 }.joined(separator: " · "))
                        .foregroundStyle(.secondary).lineLimit(1)
                }
                .width(min: 140, ideal: 220)
                TableColumn("conditions") { r in conditions(r) }.width(min: 90, ideal: 130)
                TableColumn("users served") { r in
                    Text("\(r.usersServed(ttft: sloTTFT, perUser: sloPerUser))").monospacedDigit().fontWeight(.semibold)
                        .foregroundStyle(r.longestAnswer < 300 ? Color.secondary : Color.primary)
                        .help(r.longestAnswer < 300 ? "short answers (≤\(r.longestAnswer) tokens): not comparable with 400-token runs" : "")
                }
                .width(min: 70, ideal: 90)
                TableColumn("best tok/s") { r in Text(String(format: "%.0f", r.bestTotal)).monospacedDigit() }
                    .width(min: 60, ideal: 80)
                TableColumn("vs ★") { r in ratio(r) }.width(min: 60, ideal: 90)
                TableColumn("tok/J") { r in
                    Text(Array(r.tokensPerJoule.values).average.map { String(format: "%.1f", $0) } ?? "–").monospacedDigit()
                }
                .width(min: 40, ideal: 55)
            }
            .contextMenu(forSelectionType: String.self) { ids in
                if let id = ids.first, let r = runs.first(where: { $0.id == id }) {
                    Button("Use as baseline ★") { baselineID = r.id }
                    Button("Edit note…") { noteText = RunNotes.get(r.id); noteEditing = r }
                    Divider()
                    Button("Open charts") { openCharts(r) }
                    Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([r.csv]) }
                }
            }
            .frame(height: min(CGFloat(runs.count) * 26 + 34, 300))
        }
    }

    private func conditions(_ r: RunRecord) -> some View {
        HStack(spacing: 6) {
            if let q = r.quiet {
                Image(systemName: q ? "checkmark.seal" : "exclamationmark.triangle")
                    .foregroundStyle(q ? Color.secondary : Color.orange)
                    .help(q ? "machine was quiet" : "machine was busy when it started")
            }
            if let t = r.warmTemp {
                Text(String(format: "%.0f °C", t)).foregroundStyle(.secondary).help("GPU temperature when measuring started")
            }
            if r.quiet == nil && r.warmTemp == nil {
                Text("no record").foregroundStyle(.tertiary).help("older run, before conditions were saved")
            }
        }
        .font(.caption)
    }

    @ViewBuilder private func ratio(_ r: RunRecord) -> some View {
        if let b = baseline, b.id != r.id {
            let users = Double(r.usersServed(ttft: sloTTFT, perUser: sloPerUser)) / Double(max(b.usersServed(ttft: sloTTFT, perUser: sloPerUser), 1))
            let tput = r.bestTotal / max(b.bestTotal, 1e-9)
            Text(String(format: "%.1f× users", users))
                .monospacedDigit()
                .foregroundStyle(users >= 4 ? Palette.aqua : (users > 1 ? Color.primary : Color.secondary))
                .help(String(format: "throughput %.1f× the baseline", tput))
        } else {
            Text(r.id == baselineID ? "baseline" : "–").foregroundStyle(.secondary)
        }
    }

    private func openCharts(_ r: RunRecord) {
        let base = r.csv.deletingPathExtension()
        for suffix in ["", "_heat"] {
            let png = URL(fileURLWithPath: base.path + suffix + ".png")
            if FileManager.default.fileExists(atPath: png.path) { NSWorkspace.shared.open(png) }
        }
    }

    // MARK: compare

    private var compareTiles: some View {
        HStack(spacing: 12) {
            ForEach(compared) { r in
                let served = r.usersServed(ttft: sloTTFT, perUser: sloPerUser)
                let vsBase = baseline.flatMap { b in b.id == r.id ? nil : Double(served) / Double(max(b.usersServed(ttft: sloTTFT, perUser: sloPerUser), 1)) }
                StatTile(title: chartLabel(r), value: "\(served) user\(served == 1 ? "" : "s")",
                         detail: [String(format: "best %.0f tok/s", r.bestTotal),
                                  vsBase.map { String(format: "%.1f× the baseline", $0) } ?? (r.id == baselineID ? "baseline" : nil)]
                            .compactMap { $0 }.joined(separator: " · "),
                         color: color(r))
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private struct Point: Identifiable {
        let id = UUID()
        let run: String
        let users: Int
        let value: Double
    }

    private func points(_ key: KeyPath<SweepLevel, Double>) -> [Point] {
        compared.flatMap { r in r.levels.map { Point(run: chartLabel(r), users: $0.users, value: $0[keyPath: key]) } }
    }

    private var styleScale: (domain: [String], range: [Color]) {
        (compared.map(chartLabel), compared.indices.map { Palette.series[$0] })
    }

    private var allUsers: [Int] { Array(Set(compared.flatMap { $0.levels.map(\.users) })).sorted() }

    private func compareChart(_ title: String, _ unit: String, _ key: KeyPath<SweepLevel, Double>, slo: Double? = nil) -> some View {
        Card(title: title, subtitle: unit) {
            Chart {
                ForEach(points(key)) { p in
                    LineMark(x: .value("users", p.users), y: .value(unit, p.value), series: .value("run", p.run))
                        .foregroundStyle(by: .value("run", p.run))
                        .lineStyle(StrokeStyle(lineWidth: 2))
                    PointMark(x: .value("users", p.users), y: .value(unit, p.value))
                        .foregroundStyle(by: .value("run", p.run))
                        .symbolSize(40)
                }
                if let slo {
                    RuleMark(y: .value("SLO", slo))
                        .foregroundStyle(.secondary)
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                        .annotation(position: .top, alignment: .leading) { Text("SLO").font(.caption2).foregroundStyle(.secondary) }
                }
            }
            .chartForegroundStyleScale(domain: styleScale.domain, range: styleScale.range)
            .chartXScale(type: .log)
            .chartXAxis { AxisMarks(values: allUsers) }
            .chartXAxisLabel("users at once")
            .chartLegend(position: .top, alignment: .leading)
            .frame(height: 230)
        }
    }

    private var energyChart: some View {
        let pts = compared.flatMap { r in r.tokensPerJoule.map { Point(run: chartLabel(r), users: $0.key, value: $0.value) } }
            .sorted { $0.users < $1.users }
        return Card(title: "Tokens per joule", subtitle: "energy efficiency: higher = cheaper to serve (runs with macmon only)") {
            if pts.isEmpty {
                Text("None of the selected runs recorded energy. Runs made after installing macmon have it.")
                    .foregroundStyle(.secondary).frame(height: 230)
            } else {
                Chart(pts) { p in
                    LineMark(x: .value("users", p.users), y: .value("tok/J", p.value), series: .value("run", p.run))
                        .foregroundStyle(by: .value("run", p.run))
                        .lineStyle(StrokeStyle(lineWidth: 2))
                    PointMark(x: .value("users", p.users), y: .value("tok/J", p.value))
                        .foregroundStyle(by: .value("run", p.run))
                        .symbolSize(40)
                }
                .chartForegroundStyleScale(domain: styleScale.domain, range: styleScale.range)
                .chartXScale(type: .log)
                .chartXAxis { AxisMarks(values: allUsers) }
                .chartXAxisLabel("users at once")
                .chartLegend(position: .top, alignment: .leading)
                .frame(height: 230)
            }
        }
    }
}
