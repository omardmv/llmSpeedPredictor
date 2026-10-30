import Foundation

/// The few Ollama endpoints the app talks to directly.
enum Ollama {
    static let base = URL(string: "http://localhost:11434")!
    static let aliases = ["fast": "qwen2.5:0.5b", "quality": "qwen2.5:1.5b"]

    struct Loaded: Decodable, Identifiable, Hashable {
        let name: String
        let size: Double
        let sizeVram: Double
        let expiresAt: String
        let contextLength: Int?
        var id: String { name }

        var expires: Date? {
            let f = ISO8601DateFormatter()
            f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let d = f.date(from: expiresAt) { return d }
            f.formatOptions = [.withInternetDateTime]
            // Ollama sends nanoseconds; trim them to something ISO8601DateFormatter accepts.
            let trimmed = expiresAt.replacingOccurrences(of: #"\.\d+"#, with: "", options: .regularExpression)
            return f.date(from: trimmed)
        }
    }

    private static func get(_ path: String) async throws -> Data {
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.timeoutInterval = 3
        return try await URLSession.shared.data(for: req).0
    }

    private static func post(_ path: String, _ body: [String: Any]) async throws -> Data {
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.httpMethod = "POST"
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        req.timeoutInterval = 60
        return try await URLSession.shared.data(for: req).0
    }

    private static var decoder: JSONDecoder {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }

    static func version() async throws -> String {
        struct V: Decodable { let version: String }
        return try decoder.decode(V.self, from: try await get("api/version")).version
    }

    static func loaded() async throws -> [Loaded] {
        struct PS: Decodable { let models: [Loaded]? }
        return try decoder.decode(PS.self, from: try await get("api/ps")).models ?? []
    }

    /// Every model downloaded on this machine.
    static func installed() async throws -> [String] {
        struct Tags: Decodable { struct M: Decodable { let name: String }; let models: [M] }
        return try decoder.decode(Tags.self, from: try await get("api/tags")).models.map(\.name).sorted()
    }

    /// The GGUF weights file on disk (llama-server can open the same file).
    static func ggufPath(_ model: String) async throws -> String {
        let data = try await post("api/show", ["model": model])
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let modelfile = obj?["modelfile"] as? String ?? ""
        for line in modelfile.split(separator: "\n") where line.hasPrefix("FROM /") {
            return String(line.dropFirst(5))
        }
        throw Bench.Failure(message: "No GGUF path for \(model)")
    }

    static func unload(_ model: String) async {
        _ = try? await post("api/generate", ["model": model, "keep_alive": 0])
    }
}
