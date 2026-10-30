import SwiftUI

/// Which server the One request and Load test screens measure. Shared through UserDefaults,
/// so both screens always agree.
enum ServerChoice {
    static let backendKey = "backend"   // "ollama" or "llama"
    static let urlKey = "llamaURL"

    static func args(backend: String, url: String) -> [String] {
        backend == "llama" ? ["--backend", "llama", "--url", url] : []
    }
}

/// "Server: Ollama | llama-server", plus the live setup of the llama-server (slots, context).
struct ServerPicker: View {
    @AppStorage(ServerChoice.backendKey) private var backend = "ollama"
    @AppStorage(ServerChoice.urlKey) private var url = "http://localhost:8080"
    @Local private var status = ""
    @Local private var up = false

    var body: some View {
        HStack(spacing: 12) {
            Picker("Server", selection: $backend) {
                Text("Ollama").tag("ollama")
                Text("llama-server").tag("llama")
            }
            .pickerStyle(.segmented)
            .fixedSize()
            if backend == "llama" {
                TextField("URL", text: $url)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 190)
                Label(status, systemImage: up ? "checkmark.circle.fill" : "xmark.circle")
                    .font(.callout)
                    .foregroundStyle(up ? Color.secondary : Color.orange)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
        }
        // Re-check every 3 s: you start and stop llama-server yourself in a terminal.
        .task(id: "\(backend)|\(url)") {
            while backend == "llama", !Task.isCancelled {
                await refresh()
                try? await Task.sleep(for: .seconds(3))
            }
        }
    }

    private func refresh() async {
        let props: [String: Any]? = await fetch("/props")
        let slots: [[String: Any]]? = await fetch("/slots")
        guard let props, let slots else {
            up = false
            status = "not running: start llama-server in a terminal"
            return
        }
        let n = props["total_slots"] as? Int ?? slots.count
        let ctx = slots.first?["n_ctx"] as? Int ?? 0
        let busy = slots.filter { $0["is_processing"] as? Bool == true }.count
        up = true
        status = "\(n) slots × \(ctx) tokens context · \(busy) busy now"
    }

    private func fetch<T>(_ path: String) async -> T? {
        guard let u = URL(string: url + path) else { return nil }
        var req = URLRequest(url: u)
        req.timeoutInterval = 2
        guard let (data, _) = try? await URLSession.shared.data(for: req) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? T
    }
}
