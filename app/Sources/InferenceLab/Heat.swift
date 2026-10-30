import Foundation

/// GPU temperature, clock and power from `macmon pipe` (brew install macmon; no sudo).
/// A hot chip lowers its clock, and a lower clock means fewer tokens per second.
struct HeatReading {
    let gpuTemp: Double     // °C
    let cpuTemp: Double
    let gpuMHz: Double
    let gpuPower: Double    // watts
    let chipPower: Double   // CPU + GPU + neural engine, watts
    let fanRPM: Double
}

final class HeatReader {
    static let path: String? = ["/opt/homebrew/bin/macmon", "/usr/local/bin/macmon"]
        .first { FileManager.default.isExecutableFile(atPath: $0) }

    /// Highest GPU clock this chip can run at, read once.
    static let maxGPUMHz: Double? = {
        guard let path, let line = run(path, ["pipe", "-s", "1", "--soc-info"]).split(separator: "\n").first,
              let obj = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              let soc = obj["soc"] as? [String: Any], let freqs = soc["gpu_freqs"] as? [Double] else { return nil }
        return freqs.max()
    }()

    private var process: Process?
    var onReading: ((HeatReading) -> Void)?

    func start(intervalMs: Int = 1000) {
        guard process == nil, let path = Self.path else { return }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = ["pipe", "-i", "\(intervalMs)"]
        let out = Pipe()
        p.standardOutput = out
        p.standardError = FileHandle.nullDevice
        var buffer = Data()
        out.fileHandleForReading.readabilityHandler = { [weak self] h in
            buffer.append(h.availableData)
            while let nl = buffer.firstIndex(of: 0x0A) {
                let line = buffer[..<nl]
                buffer.removeSubrange(...nl)
                if let r = Self.parse(Data(line)) { self?.onReading?(r) }
            }
        }
        try? p.run()
        process = p
    }

    func stop() {
        process?.terminate()
        process = nil
    }

    static func parse(_ data: Data) -> HeatReading? {
        guard let d = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let temp = d["temp"] as? [String: Any] else { return nil }
        let num = { (v: Any?) in (v as? NSNumber)?.doubleValue ?? 0 }
        let fans = (d["fans"] as? [[String: Any]] ?? []).map { num($0["rpm"]) }
        return HeatReading(gpuTemp: num(temp["gpu_temp_avg"]), cpuTemp: num(temp["cpu_temp_avg"]),
                           gpuMHz: num(d["gpu_freq_mhz"]), gpuPower: num(d["gpu_power"]),
                           chipPower: num(d["all_power"]), fanRPM: fans.max() ?? 0)
    }

    private static func run(_ path: String, _ args: [String]) -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        let out = Pipe()
        p.standardOutput = out
        p.standardError = FileHandle.nullDevice
        try? p.run()
        p.waitUntilExit()
        return String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    }
}
