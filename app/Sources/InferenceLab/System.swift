import Foundation
import IOKit
import Observation

/// Reads the machine directly from the kernel and the GPU driver. No sudo, no subprocesses.
enum System {
    struct Machine {
        let chip: String
        let performanceCores: Int
        let efficiencyCores: Int
        let gpuCores: Int
        let memoryBytes: Double
        let bandwidthGBs: Int?
        let macOS: String
    }

    struct GPU {
        let utilization: Int      // % of time the GPU was busy
        let memoryInUse: Double   // bytes the GPU is holding right now
        let memoryAllocated: Double
    }

    struct Memory {
        let app: Double
        let wired: Double
        let compressed: Double
        let fileCache: Double
        let freePercent: Int
    }

    // Spec sheet numbers; there is no API for memory bandwidth.
    static let bandwidth: [String: Int] = [
        "Apple M1": 68, "Apple M1 Pro": 200, "Apple M1 Max": 400, "Apple M1 Ultra": 800,
        "Apple M2": 100, "Apple M2 Pro": 200, "Apple M2 Max": 400, "Apple M2 Ultra": 800,
        "Apple M3": 100, "Apple M3 Pro": 150, "Apple M3 Max": 400,
        "Apple M4": 120, "Apple M4 Pro": 273, "Apple M4 Max": 546,
    ]

    static func sysctlString(_ name: String) -> String {
        var size = 0
        sysctlbyname(name, nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname(name, &buf, &size, nil, 0)
        return String(cString: buf)
    }

    static func sysctlInt(_ name: String) -> Int {
        var value: Int64 = 0
        var size = MemoryLayout<Int64>.size
        if sysctlbyname(name, &value, &size, nil, 0) != 0 { return 0 }
        // Some values are 32-bit; reading 4 bytes into an Int64 still gives the right number.
        return Int(value)
    }

    /// Properties of the GPU's IOAccelerator entry in the I/O registry (what `ioreg` prints).
    private static func acceleratorProperties() -> [String: Any]? {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) == KERN_SUCCESS
        else { return nil }
        defer { IOObjectRelease(iterator) }
        var entry = IOIteratorNext(iterator)
        while entry != 0 {
            var props: Unmanaged<CFMutableDictionary>?
            let ok = IORegistryEntryCreateCFProperties(entry, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS
            IOObjectRelease(entry)
            if ok, let dict = props?.takeRetainedValue() as? [String: Any], dict["PerformanceStatistics"] != nil {
                return dict
            }
            entry = IOIteratorNext(iterator)
        }
        return nil
    }

    static func machine() -> Machine {
        let chip = sysctlString("machdep.cpu.brand_string")
        let os = ProcessInfo.processInfo.operatingSystemVersion
        return Machine(
            chip: chip,
            performanceCores: sysctlInt("hw.perflevel0.physicalcpu"),
            efficiencyCores: sysctlInt("hw.perflevel1.physicalcpu"),
            gpuCores: acceleratorProperties()?["gpu-core-count"] as? Int ?? 0,
            memoryBytes: Double(ProcessInfo.processInfo.physicalMemory),
            bandwidthGBs: bandwidth[chip],
            macOS: "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
        )
    }

    static func gpu() -> GPU {
        let stats = acceleratorProperties()?["PerformanceStatistics"] as? [String: Any] ?? [:]
        func num(_ key: String) -> Double { (stats[key] as? NSNumber)?.doubleValue ?? 0 }
        return GPU(utilization: Int(num("Device Utilization %")),
                   memoryInUse: num("In use system memory"),
                   memoryAllocated: num("Alloc system memory"))
    }

    /// Unified memory split the way Activity Monitor shows it.
    static func memory() -> Memory {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        _ = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        let page = Double(vm_kernel_page_size)
        return Memory(
            app: Double(Int64(stats.internal_page_count) - Int64(stats.purgeable_count)) * page,
            wired: Double(stats.wire_count) * page,
            compressed: Double(stats.compressor_page_count) * page,
            fileCache: Double(stats.external_page_count) * page,
            freePercent: sysctlInt("kern.memorystatus_level")
        )
    }
}

/// Samples the machine a few times per second and keeps the last two minutes.
@MainActor @Observable
final class Monitor {
    struct Sample: Identifiable {
        let id = UUID()
        let time: Date
        let gpuUtil: Int
        let gpuMemory: Double
        let memory: System.Memory
        let heat: HeatReading?   // nil without macmon
    }

    let machine = System.machine()
    var samples: [Sample] = []
    var loaded: [Ollama.Loaded] = []
    var ollamaVersion: String?
    var ollamaUp = false
    var latest: Sample? { samples.last }

    private var timer: Timer?
    private var tick = 0
    private let keep = 240  // 2 minutes at 2 samples per second

    var latestHeat: HeatReading?
    let heatAvailable = HeatReader.path != nil
    private let heatReader = HeatReader()

    func start() {
        guard timer == nil else { return }
        heatReader.onReading = { [weak self] r in Task { @MainActor in self?.latestHeat = r } }
        heatReader.start()
        sample()
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.sample() }
        }
    }

    func stop() { heatReader.stop() }

    private func sample() {
        let gpu = System.gpu()
        samples.append(Sample(time: .now, gpuUtil: gpu.utilization, gpuMemory: gpu.memoryInUse, memory: System.memory(),
                              heat: latestHeat))
        if samples.count > keep { samples.removeFirst(samples.count - keep) }
        tick += 1
        if tick % 2 == 1 { Task { await refreshOllama() } }
    }

    func refreshOllama() async {
        do {
            loaded = try await Ollama.loaded()
            if ollamaVersion == nil { ollamaVersion = try await Ollama.version() }
            ollamaUp = true
        } catch {
            ollamaUp = false
            loaded = []
        }
    }
}
