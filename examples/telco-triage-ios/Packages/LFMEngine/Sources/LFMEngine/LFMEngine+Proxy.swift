import Foundation

/// Default timeout for proxy health checks (seconds).
private let proxyHealthTimeout: TimeInterval = 5
/// Default timeout for proxy inference requests (seconds).
private let proxyInferenceTimeout: TimeInterval = 10

/// Proxy mode implementation for HTTP inference backends.
extension LFMEngine {

    // MARK: - Connection Validation

    func validateProxyConnection() async throws {
        guard let baseURL = proxyBaseURL else {
            throw LFMEngineError.modelLoadFailed("proxyBaseURL is required for proxy mode")
        }

        let healthURL = baseURL.appendingPathComponent("health")
        let request = URLRequest(url: healthURL, timeoutInterval: proxyHealthTimeout)

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                throw LFMEngineError.modelLoadFailed("Backend health check failed")
            }
        } catch let error as LFMEngineError {
            throw error
        } catch {
            throw LFMEngineError.modelLoadFailed("Cannot reach backend: \(error.localizedDescription)")
        }
    }

    // MARK: - Inference

    func generateViaProxy(_ params: GenerationParams) async throws -> InferenceResult {
        guard let baseURL = proxyBaseURL else {
            throw LFMEngineError.modelNotLoaded
        }

        let capability = activeAdapter?.capability ?? "mobile-intent-router"
        let url = baseURL
            .appendingPathComponent("api/v1/inference")
            .appendingPathComponent(capability)

        var request = URLRequest(url: url, timeoutInterval: proxyInferenceTimeout)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "text": params.prompt,
            "max_tokens": params.maxTokens,
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let start = CFAbsoluteTimeGetCurrent()
        let (data, response) = try await URLSession.shared.data(for: request)
        let networkLatency = (CFAbsoluteTimeGetCurrent() - start) * 1000

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw LFMEngineError.inferenceFailed("Backend returned error")
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let resultText = json["text"] as? String ?? ""
        let lfmLatency = json["latency_ms"] as? Double ?? networkLatency
        let tokensGenerated = json["response_tokens"] as? Int ?? resultText.split(separator: " ").count

        return InferenceResult(
            text: resultText,
            latencyMs: lfmLatency,
            tokensGenerated: tokensGenerated,
            model: modelConfig.name,
            adapter: activeAdapter?.name,
            ranOn: .cloud
        )
    }

    // MARK: - Streaming

    func streamViaProxy(
        _ params: GenerationParams,
        continuation: AsyncThrowingStream<StreamToken, Error>.Continuation
    ) async throws {
        // Fetch full result and emit as stream. True SSE streaming deferred.
        let result = try await generateViaProxy(params)
        let words = result.text.components(separatedBy: " ")

        for (index, word) in words.enumerated() {
            let token = StreamToken(
                text: index == 0 ? word : " " + word,
                index: index,
                isFinal: index == words.count - 1
            )
            continuation.yield(token)
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}
