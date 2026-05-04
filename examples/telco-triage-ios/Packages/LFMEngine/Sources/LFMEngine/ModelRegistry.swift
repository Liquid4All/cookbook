import Foundation
import os.log

// MARK: - Pack Status

/// Current state of a model pack on this device.
public enum PackStatus: Sendable, Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case downloaded(sizeOnDisk: Int64)
    case updateAvailable(currentVersion: Int, latestVersion: Int)
    case incompatible(reason: String)
    case comingSoon
}

// MARK: - Model Registry

/// Bridges the model manifest with the file-level ModelManager.
///
/// Single source of truth for what packs are available, downloaded, and recommended.
/// Uses the existing `ModelManager` actor for all file operations.
public actor ModelRegistry {

    private let manager: ModelManager
    private var manifest: ModelManifest
    private var activeDownloads: [String: Double] = [:]
    private let logger = Logger(subsystem: "ai.liquid.banking", category: "ModelRegistry")

    /// Base URL for resolving relative download paths in the manifest.
    private let downloadBaseURL: String

    public init(manager: ModelManager, manifest: ModelManifest, downloadBaseURL: String = "") {
        self.manager = manager
        self.manifest = manifest
        self.downloadBaseURL = downloadBaseURL
    }

    // MARK: - Queries

    /// All packs defined in the manifest.
    public func availablePacks() -> [ModelPack] {
        manifest.packs
    }

    /// Packs whose models are all present on disk.
    public func downloadedPacks() async -> [ModelPack] {
        var result: [ModelPack] = []
        for pack in manifest.packs {
            if await allModelsDownloaded(for: pack) {
                result.append(pack)
            }
        }
        return result
    }

    /// Packs the device can run, filtered by RAM and storage.
    public func recommendedPacks(for profile: DeviceCapabilities) -> [ModelPack] {
        manifest.packs.filter { pack in
            pack.minRAMMB <= profile.inferenceRAMMB &&
            pack.totalSizeBytes <= profile.availableStorageMB * 1_048_576
        }
    }

    /// Returns the first downloaded pack that provides the given capability, or nil.
    ///
    /// Used by the engine proxy to determine if a specialist is available locally.
    public func downloadedPackForCapability(_ capability: String) async -> ModelPack? {
        for pack in manifest.packs where pack.capabilities.contains(capability) {
            if await allModelsDownloaded(for: pack) {
                return pack
            }
        }
        return nil
    }

    /// Returns the model file path for the primary model in a pack.
    ///
    /// Phase 3 uses this to get the GGUF path for loading into LlamaBackend.
    public func modelPathForPack(_ packID: String) async -> URL? {
        guard let pack = manifest.packs.first(where: { $0.id == packID }),
              let primaryModel = pack.models.first else {
            return nil
        }
        let modelConfig = ModelConfig(
            name: primaryModel.fileName,
            fileName: primaryModel.fileName,
            contextLength: primaryModel.contextLength,
            gpuLayers: primaryModel.gpuLayers
        )
        let path = await manager.modelPath(modelConfig)
        return await manager.isBaseModelAvailable(modelConfig) ? path : nil
    }

    /// Current status of a specific pack.
    public func packStatus(_ packID: String) async -> PackStatus {
        guard let pack = manifest.packs.first(where: { $0.id == packID }) else {
            return .incompatible(reason: "Pack not found in manifest")
        }

        if let progress = activeDownloads[packID] {
            return .downloading(progress: progress)
        }

        if await allModelsDownloaded(for: pack) {
            let size = await totalSizeOnDisk(for: pack)
            return .downloaded(sizeOnDisk: size)
        }

        return .notDownloaded
    }

    // MARK: - Download Lifecycle

    /// Download all models in a pack with progress reporting.
    public func downloadPack(
        _ packID: String,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard let pack = manifest.packs.first(where: { $0.id == packID }) else {
            throw LFMEngineError.modelLoadFailed("Pack '\(packID)' not found in manifest")
        }

        activeDownloads[packID] = 0.0
        defer { activeDownloads.removeValue(forKey: packID) }

        let totalBytes = pack.models.reduce(Int64(0)) { $0 + $1.sizeBytes }
        var downloadedBytes: Int64 = 0

        for model in pack.models {
            guard let downloadURL = model.resolvedDownloadURL(base: downloadBaseURL) else {
                throw LFMEngineError.noDownloadURL
            }
            let localPath = try await manager.downloadModel(
                from: downloadURL,
                fileName: model.fileName
            ) { [weak self] dlProgress in
                let currentTotal = downloadedBytes + dlProgress.bytesDownloaded
                let fraction = totalBytes > 0 ? Double(currentTotal) / Double(totalBytes) : 0
                Task { await self?.updateProgress(packID: packID, fraction: fraction) }
                progress(fraction)
            }

            downloadedBytes += model.sizeBytes

            // Verify checksum after download
            let valid = try await verifyChecksum(at: localPath, expected: model.sha256)
            if !valid {
                let modelConfig = ModelConfig(
                    name: model.fileName,
                    fileName: model.fileName,
                    contextLength: model.contextLength,
                    gpuLayers: model.gpuLayers
                )
                try await manager.deleteModel(modelConfig)
                throw LFMEngineError.checksumMismatch(fileName: model.fileName)
            }
        }

        logger.info("Pack '\(packID)' downloaded and verified")
    }

    /// Delete all models in a pack from disk.
    public func deletePack(_ packID: String) async throws {
        guard let pack = manifest.packs.first(where: { $0.id == packID }) else {
            throw LFMEngineError.modelLoadFailed("Pack '\(packID)' not found in manifest")
        }

        for model in pack.models {
            let modelConfig = ModelConfig(
                name: model.fileName,
                fileName: model.fileName,
                contextLength: model.contextLength,
                gpuLayers: model.gpuLayers
            )
            try await manager.deleteModel(modelConfig)
        }

        logger.info("Pack '\(packID)' deleted")
    }

    /// Verify integrity of all models in a downloaded pack.
    public func verifyPack(_ packID: String) async throws -> Bool {
        guard let pack = manifest.packs.first(where: { $0.id == packID }) else {
            return false
        }

        for model in pack.models {
            let modelConfig = ModelConfig(
                name: model.fileName,
                fileName: model.fileName,
                contextLength: model.contextLength,
                gpuLayers: model.gpuLayers
            )
            let path = await manager.modelPath(modelConfig)
            if try await !verifyChecksum(at: path, expected: model.sha256) {
                return false
            }
        }
        return true
    }

    // MARK: - Manifest Refresh

    /// Fetch an updated manifest from a remote URL.
    public func refreshManifest(from url: URL) async throws {
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw LFMEngineError.modelLoadFailed("Failed to fetch manifest")
        }
        let newManifest = try ModelManifest.from(data: data)
        if newManifest.version > manifest.version {
            manifest = newManifest
            logger.info("Manifest updated to version \(newManifest.version)")
        }
    }

    // MARK: - Private

    private func updateProgress(packID: String, fraction: Double) {
        activeDownloads[packID] = fraction
    }

    private func allModelsDownloaded(for pack: ModelPack) async -> Bool {
        for model in pack.models {
            let modelConfig = ModelConfig(
                name: model.fileName,
                fileName: model.fileName,
                contextLength: model.contextLength,
                gpuLayers: model.gpuLayers
            )
            if await !manager.isBaseModelAvailable(modelConfig) {
                return false
            }
        }
        return true
    }

    private func totalSizeOnDisk(for pack: ModelPack) async -> Int64 {
        var total: Int64 = 0
        for model in pack.models {
            let modelConfig = ModelConfig(
                name: model.fileName,
                fileName: model.fileName,
                contextLength: model.contextLength,
                gpuLayers: model.gpuLayers
            )
            let path = await manager.modelPath(modelConfig)
            let attrs = try? FileManager.default.attributesOfItem(atPath: path.path)
            total += (attrs?[.size] as? Int64) ?? 0
        }
        return total
    }

    /// Verify SHA256 checksum of a downloaded file.
    /// Delegates to ModelManager's streaming implementation to avoid OOM on large files.
    func verifyChecksum(at url: URL, expected: String) async throws -> Bool {
        try await manager.verifyChecksum(at: url, expected: expected)
    }
}

// MARK: - Device Capabilities Protocol

/// Abstraction for device profiling used by ModelRegistry recommendations.
/// Implemented by DeviceProfile in the app target.
public protocol DeviceCapabilities: Sendable {
    var inferenceRAMMB: Int { get }
    var availableStorageMB: Int64 { get }
}
