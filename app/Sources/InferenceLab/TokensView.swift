import Charts
import SwiftUI

/// Runs llama-server (llama.cpp) on the same GGUF file Ollama uses, only for its tokenizer.
/// Same file, same tokenizer, same token IDs as Ollama.
@MainActor @Observable
final class Tokenizer {
    struct Token: Identifiable, Hashable {
        let id: Int        // position in the text
        let tokenID: Int
        let piece: String
        var isSpecial: Bool { piece.hasPrefix("<|") && piece.hasSuffix("|>") }
    }

    enum Status: Equatable { case stopped, starting(String), ready(String), failed(String) }

    var status: Status = .stopped
    var vocabSize = 0
    private var process: Process?
    private var model = ""
    private let port = 8092
    private var base: URL { URL(string: "http://127.0.0.1:\(port)")! }

    func start(_ model: String) async {
        if self.model == model, case .ready = status { return }
        stop()
        self.model = model
        status = .starting(model)
        do {
            let path = try await Ollama.ggufPath(model)
            Self.killLeftovers(port: port)
            let p = Process()
            p.executableURL = URL(fileURLWithPath: Self.llamaServer())
            // -ngl 0: stay on the CPU. Only the tokenizer is needed, not the GPU.
            p.arguments = ["-m", path, "--port", "\(port)", "-c", "512", "-ngl", "0"]
            p.standardOutput = FileHandle.nullDevice
            p.standardError = FileHandle.nullDevice
            try p.run()
            process = p
            for _ in 0..<100 {
                if let (data, _) = try? await URLSession.shared.data(from: base.appendingPathComponent("health")),
                   String(decoding: data, as: UTF8.self).contains("ok") {
                    vocabSize = await fetchVocab()
                    status = .ready(model)
                    return
                }
                try await Task.sleep(for: .milliseconds(200))
            }
            throw Bench.Failure(message: "llama-server did not start")
        } catch {
            status = .failed(error.localizedDescription)
            stop()
        }
    }

    func stop() {
        process?.terminate()
        process = nil
        if case .ready = status { status = .stopped }
    }

    /// A tokenizer left running by an earlier crash would answer for the wrong model.
    private static func killLeftovers(port: Int) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        p.arguments = ["-f", "llama-server .*--port \(port) "]
        try? p.run()
        p.waitUntilExit()
    }

    private static func llamaServer() -> String {
        for p in ["/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"]
        where FileManager.default.isExecutableFile(atPath: p) { return p }
        return "/opt/homebrew/bin/llama-server"
    }

    private func post(_ path: String, _ body: [String: Any]) async throws -> Any {
        var req = URLRequest(url: base.appendingPathComponent(path))
        req.httpMethod = "POST"
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, _) = try await URLSession.shared.data(for: req)
        return try JSONSerialization.jsonObject(with: data)
    }

    private func fetchVocab() async -> Int {
        guard let (data, _) = try? await URLSession.shared.data(from: base.appendingPathComponent("v1/models")),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let first = (obj["data"] as? [[String: Any]])?.first,
              let meta = first["meta"] as? [String: Any] else { return 0 }
        return meta["n_vocab"] as? Int ?? 0
    }

    func tokenize(_ text: String) async -> [Token] {
        guard case .ready = status, !text.isEmpty,
              let obj = try? await post("tokenize", ["content": text, "with_pieces": true,
                                                     "add_special": false, "parse_special": true]) as? [String: Any],
              let list = obj["tokens"] as? [[String: Any]] else { return [] }
        return list.enumerated().map { i, t in
            // A piece is a string, or raw bytes when it is only part of a multi-byte character.
            let piece: String
            if let s = t["piece"] as? String { piece = s }
            else if let bytes = t["piece"] as? [Int] { piece = bytes.map { String(format: "<%02X>", $0) }.joined() }
            else { piece = "?" }
            return Token(id: i, tokenID: t["id"] as? Int ?? -1, piece: piece)
        }
    }

    func chatTemplate(_ text: String) async -> String {
        let obj = try? await post("apply-template", ["messages": [["role": "user", "content": text]]]) as? [String: Any]
        return obj?["prompt"] as? String ?? ""
    }
}

struct TokensView: View {
    let active: Bool
    @Local private var tokenizer = Tokenizer()
    @Local private var model = "quality"
    @Local private var text = "Large language models split text into tokens and generate one token at a time."
    @Local private var tokens: [Tokenizer.Token] = []
    @Local private var showTemplate = false
    @Local private var templateTokens = 0
    @Local private var comparison: [(name: String, chars: Int, tokens: Int)] = []

    static let samples: [(String, String)] = [
        ("English", "Large language models split text into tokens and generate one token at a time."),
        ("Azerbaijani", "Böyük dil modelləri mətni tokenlərə bölür və hər dəfə bir token yaradır."),
        ("Uzbek", "Katta til modellari matnni tokenlarga boʻladi va har safar bitta token yaratadi."),
        ("Russian", "Большие языковые модели делят текст на токены и генерируют по одному токену за раз."),
        ("Python", "def mean(xs):\n    return sum(xs) / len(xs) if xs else 0.0\n"),
        ("JSON", #"{"model": "qwen2.5:1.5b", "max_tokens": 256, "temperature": 0.7}"#),
        ("Numbers", "3.14159 2653589 1000000 2026-10-30 192.168.0.1"),
        ("Emoji", "🚀🔥😂👍 ok"),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                controls
                Card(title: "Your text", subtitle: "type anything; it is re-tokenized as you type") {
                    TextEditor(text: $text)
                        .font(.body)
                        .frame(minHeight: 70, maxHeight: 140)
                        .scrollContentBackground(.hidden)
                        .padding(6)
                        .background(.background, in: RoundedRectangle(cornerRadius: 8))
                    HStack {
                        ForEach(Self.samples, id: \.0) { name, sample in
                            Button(name) { text = sample }.controlSize(.small)
                        }
                    }
                }
                tokensCard
                comparisonCard
            }
            .padding(18)
        }
        .task(id: active ? model : "") {
            if active { await tokenizer.start(resolveModel(model)); await refresh(); await compare() }
        }
        .task(id: "\(text)|\(showTemplate)|\(tokenizer.status == .ready(resolveModel(model)))") {
            try? await Task.sleep(for: .milliseconds(120))  // wait until typing pauses
            await refresh()
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
            tokenizer.stop()
        }
    }

    private var controls: some View {
        HStack(spacing: 12) {
            ModelPicker(model: $model)
            Toggle("Show chat template", isOn: $showTemplate)
                .help("What the model really reads: your text wrapped in the chat template")
            Spacer()
            switch tokenizer.status {
            case .starting: ProgressView().controlSize(.small); Text("starting tokenizer…").foregroundStyle(.secondary)
            case .ready: Text("\(tokenizer.vocabSize.formatted()) tokens in the vocabulary").foregroundStyle(.secondary)
            case .failed(let msg): Text(msg).foregroundStyle(.red)
            case .stopped: EmptyView()
            }
        }
    }

    private func refresh() async {
        if showTemplate {
            let wrapped = await tokenizer.chatTemplate(text)
            tokens = await tokenizer.tokenize(wrapped)
            templateTokens = tokens.count - (await tokenizer.tokenize(text)).count
        } else {
            tokens = await tokenizer.tokenize(text)
        }
    }

    private func compare() async {
        var rows: [(String, Int, Int)] = []
        for (name, sample) in Self.samples {
            rows.append((name, sample.count, await tokenizer.tokenize(sample).count))
        }
        comparison = rows.map { (name: $0.0, chars: $0.1, tokens: $0.2) }
    }

    private var tokensCard: some View {
        let chars = text.count
        return Card(title: "\(tokens.count) tokens",
                    subtitle: showTemplate
                        ? "the chat template added \(templateTokens) tokens around your text"
                        : "\(chars) characters · \(tokens.isEmpty ? "–" : String(format: "%.2f", Double(chars) / Double(tokens.count))) characters per token · hover a token for its ID") {
            FlowLayout(spacing: 3) {
                ForEach(tokens) { t in TokenChip(token: t) }
            }
            .padding(.vertical, 4)
        }
    }

    private var comparisonCard: some View {
        Card(title: "Characters per token", subtitle: "more characters per token = cheaper and faster for the same text") {
            Chart(comparison, id: \.name) { row in
                BarMark(x: .value("chars per token", Double(row.chars) / Double(max(row.tokens, 1))),
                        y: .value("text", row.name))
                    .foregroundStyle(Palette.blue)
                    .cornerRadius(4)
                    .annotation(position: .trailing) {
                        Text(String(format: "%.2f  (%d chars → %d tokens)", Double(row.chars) / Double(max(row.tokens, 1)),
                                    row.chars, row.tokens))
                            .font(.caption).foregroundStyle(.secondary)
                    }
            }
            .chartXAxisLabel("characters per token")
            .frame(height: 230)
        }
    }
}

/// One token as a colored chip. Spaces show as ␣ and newlines as ↵.
struct TokenChip: View {
    let token: Tokenizer.Token
    private static let colors = [Palette.blue, Palette.orange, Palette.aqua, Palette.yellow]

    var body: some View {
        let shown = token.piece.replacingOccurrences(of: " ", with: "␣").replacingOccurrences(of: "\n", with: "↵")
        Text(shown)
            .font(.system(.body, design: .monospaced))
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .background(token.isSpecial ? Color.gray.opacity(0.35) : Self.colors[token.id % 4].opacity(0.28),
                        in: RoundedRectangle(cornerRadius: 4))
            .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(token.isSpecial ? Color.gray : .clear))
            .help("token #\(token.id + 1) · ID \(token.tokenID)\(token.isSpecial ? " · special token" : "")")
    }
}

/// Lays children out left to right and wraps to the next line, like words in a paragraph.
struct FlowLayout: Layout {
    var spacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 600
        var x: CGFloat = 0, y: CGFloat = 0, row: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x + s.width > width, x > 0 { x = 0; y += row + spacing; row = 0 }
            x += s.width + spacing
            row = max(row, s.height)
        }
        return CGSize(width: width, height: y + row)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, row: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x + s.width > bounds.maxX, x > bounds.minX { x = bounds.minX; y += row + spacing; row = 0 }
            v.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(s))
            x += s.width + spacing
            row = max(row, s.height)
        }
    }
}
