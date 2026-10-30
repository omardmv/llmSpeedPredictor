import Foundation

/// Runs the Python measuring kit (`python -m bench ... --json`) and streams its output lines.
/// The Python code stays the single source of truth for how things are measured.
enum Bench {
    /// The project root: this file lives in <root>/app/Sources/InferenceLab/.
    static let root: URL = {
        if let env = ProcessInfo.processInfo.environment["INFERENCE_LAB_ROOT"] {
            return URL(fileURLWithPath: env)
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }()

    static var python: URL { root.appendingPathComponent(".venv/bin/python") }
    static var benchResults: URL { root.appendingPathComponent("research/bench") }

    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    /// One JSON line at a time. Cancelling the task kills the Python process.
    static func lines(_ args: [String]) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let process = Process()
            process.executableURL = python
            process.arguments = ["-m", "bench"] + args + ["--json"]
            process.currentDirectoryURL = root
            let out = Pipe(), err = Pipe()
            process.standardOutput = out
            process.standardError = err

            var buffer = Data()
            out.fileHandleForReading.readabilityHandler = { handle in
                buffer.append(handle.availableData)
                while let nl = buffer.firstIndex(of: 0x0A) {
                    let line = String(decoding: buffer[..<nl], as: UTF8.self)
                    buffer.removeSubrange(...nl)
                    if !line.isEmpty { continuation.yield(line) }
                }
            }
            process.terminationHandler = { p in
                out.fileHandleForReading.readabilityHandler = nil
                if p.terminationStatus == 0 || p.terminationReason == .uncaughtSignal {
                    continuation.finish()
                } else {
                    let msg = String(decoding: err.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
                    let last = msg.split(separator: "\n").suffix(3).joined(separator: "\n")
                    continuation.finish(throwing: Failure(message: last.isEmpty ? "bench exited with \(p.terminationStatus)" : last))
                }
            }
            continuation.onTermination = { _ in if process.isRunning { process.terminate() } }
            do {
                try process.run()
            } catch {
                continuation.finish(throwing: Failure(message: "Could not start \(python.path): \(error.localizedDescription)"))
            }
        }
    }

    static func decode<T: Decodable>(_ type: T.Type, _ line: String) throws -> T {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(T.self, from: Data(line.utf8))
    }
}
