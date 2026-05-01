import CryptoKit
import Foundation
import os.log

/// Manages model and adapter file lifecycle: download, cache, verify, delete.
///
/// Models are stored in the app's Documents directory under `models/`.
/// Adapters are stored under `models/adapters/`.
public actor ModelManager {

    // MARK: - Configuration

    public struct Config: Sendable {
        public let modelsDirectory: URL
        public let adaptersDirectory: URL
        public let baseModelURL: URL?
        public let adapterBaseURL: URL?

        public init(
            modelsDirectory: URL? = nil,
            adaptersDirectory: URL? = nil,
            baseModelURL: URL? = nil,
            adapterBaseURL: URL? = nil
        ) {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            self.modelsDirectory = modelsDirectory ?? docs.appendingPathComponent("models")
            self.adaptersDirectory = adaptersDirectory ?? docs.appendingPathComponent("models/adapters")
            self.baseModelURL = baseModelURL
            self.adapterBaseURL = adapterBaseURL
        }
    }

    // MARK: - State

    private let config: Config
    let fileManager = FileManager.default
    private let logger = Logger(subsystem: "ai.liquid.banking", category: "ModelManager")

    // MARK: - Init

    public init(config: Config = Config()) {
        self.config = config
    }

    // MARK: - Directory Management

    /// Ensure model directories exist.
    public func ensureDirectories() throws {
        try fileManager.createDirectory(at: config.modelsDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: config.adaptersDirectory, withIntermediateDirectories: true)
    }

    // MARK: - Model Queries

    /// Check if the base model file exists locally.
    public func isBaseModelAvailable(_ modelConfig: ModelConfig) -> Bool {
        let path = config.modelsDirectory.appendingPathComponent(modelConfig.fileName)
        return fileManager.fileExists(atPath: path.path)
    }

    /// Check if an adapter file exists locally.
    public func isAdapterAvailable(_ adapter: AdapterConfig) -> Bool {
        let path = config.adaptersDirectory.appendingPathComponent(adapter.fileName)
        return fileManager.fileExists(atPath: path.path)
    }

    /// Get the local file path for a model.
    public func modelPath(_ modelConfig: ModelConfig) -> URL {
        config.modelsDirectory.appendingPathComponent(modelConfig.fileName)
    }

    /// Get the local file path for an adapter.
    public func adapterPath(_ adapter: AdapterConfig) -> URL {
        config.adaptersDirectory.appendingPathComponent(adapter.fileName)
    }

    /// List all downloaded model files.
    public func listDownloadedModels() throws -> [URL] {
        guard fileManager.fileExists(atPath: config.modelsDirectory.path) else {
            return []
        }
        let contents = try fileManager.contentsOfDirectory(
            at: config.modelsDirectory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        )
        return contents.filter { $0.pathExtension == "gguf" }
    }

    /// List all downloaded adapter files.
    public func listDownloadedAdapters() throws -> [URL] {
        guard fileManager.fileExists(atPath: config.adaptersDirectory.path) else {
            return []
        }
        let contents = try fileManager.contentsOfDirectory(
            at: config.adaptersDirectory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        )
        return contents.filter { $0.pathExtension == "gguf" }
    }

    // MARK: - Download

    /// Download the base model with progress reporting.
    public func downloadBaseModel(
        _ modelConfig: ModelConfig,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws -> URL {
        guard let baseURL = config.baseModelURL else {
            throw LFMEngineError.noDownloadURL
        }

        let remoteURL = baseURL.appendingPathComponent(modelConfig.fileName)
        let localPath = self.modelPath(modelConfig)

        try await downloadFile(from: remoteURL, to: localPath, onProgress: onProgress)
        logger.info("Base model downloaded: \(modelConfig.fileName)")
        return localPath
    }

    /// Download a model from an explicit URL with progress reporting.
    ///
    /// Used by `ModelRegistry` to download manifest-specified models where each
    /// model has its own `downloadURL` rather than relying on a configured base URL.
    public func downloadModel(
        from remoteURL: URL,
        fileName: String,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws -> URL {
        guard remoteURL.scheme == "https" else {
            throw LFMEngineError.modelLoadFailed(
                "Invalid download URL scheme '\(remoteURL.scheme ?? "nil")' for \(fileName). Only https is allowed."
            )
        }

        let localPath = config.modelsDirectory.appendingPathComponent(fileName)
        try ensureDirectories()
        try await downloadFile(from: remoteURL, to: localPath, onProgress: onProgress)
        logger.info("Model downloaded: \(fileName)")
        return localPath
    }

    /// Download an adapter with progress reporting.
    public func downloadAdapter(
        _ adapter: AdapterConfig,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws -> URL {
        guard let baseURL = config.adapterBaseURL else {
            throw LFMEngineError.adapterLoadFailed(
                adapter: adapter.name,
                reason: "No adapter download URL configured"
            )
        }

        let remoteURL = baseURL.appendingPathComponent(adapter.fileName)
        let localPath = self.adapterPath(adapter)

        try await downloadFile(from: remoteURL, to: localPath, onProgress: onProgress)
        logger.info("Adapter downloaded: \(adapter.fileName)")
        return localPath
    }

    // MARK: - Cleanup

    /// Delete a specific model file.
    public func deleteModel(_ modelConfig: ModelConfig) throws {
        let path = self.modelPath(modelConfig)
        if fileManager.fileExists(atPath: path.path) {
            try fileManager.removeItem(at: path)
            logger.info("Deleted model: \(modelConfig.fileName)")
        }
    }

    /// Delete a specific adapter file.
    public func deleteAdapter(_ adapter: AdapterConfig) throws {
        let path = self.adapterPath(adapter)
        if fileManager.fileExists(atPath: path.path) {
            try fileManager.removeItem(at: path)
            logger.info("Deleted adapter: \(adapter.fileName)")
        }
    }

    /// Delete all downloaded models and adapters.
    public func deleteAll() throws {
        if fileManager.fileExists(atPath: config.modelsDirectory.path) {
            try fileManager.removeItem(at: config.modelsDirectory)
            logger.info("Deleted all models and adapters")
        }
        try ensureDirectories()
    }

    /// Total disk space used by models and adapters in bytes.
    public func totalDiskUsage() throws -> Int64 {
        var total: Int64 = 0
        let models = try listDownloadedModels()
        let adapters = try listDownloadedAdapters()

        for url in models + adapters {
            let attrs = try fileManager.attributesOfItem(atPath: url.path)
            total += (attrs[.size] as? Int64) ?? 0
        }
        return total
    }

    // MARK: - Checksum Verification

    /// Verify the SHA256 checksum of a downloaded file against an expected hash.
    ///
    /// Uses streaming reads (64KB chunks) to avoid loading the entire file into memory.
    /// Critical for 700MB+ GGUF files that would OOM on 4GB devices.
    ///
    /// Returns `true` if the file's computed hash matches the expected value.
    /// Returns `false` if the file doesn't exist or hashes differ.
    public func verifyChecksum(at filePath: URL, expected: String) throws -> Bool {
        guard fileManager.fileExists(atPath: filePath.path) else {
            return false
        }
        let handle = try FileHandle(forReadingFrom: filePath)
        defer { try? handle.close() }

        var hasher = SHA256()
        let chunkSize = 65_536
        while autoreleasepool(invoking: {
            let chunk = handle.readData(ofLength: chunkSize)
            guard !chunk.isEmpty else { return false }
            hasher.update(data: chunk)
            return true
        }) {}

        let digest = hasher.finalize()
        // Skip verification when no checksum is provided (development builds)
        guard !expected.isEmpty else { return true }

        let actual = digest.map { String(format: "%02x", $0) }.joined()
        logger.debug("Checksum for \(filePath.lastPathComponent): \(actual)")
        return actual == expected.lowercased()
    }

}
