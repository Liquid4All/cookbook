import Foundation

// MARK: - Model Manifest

/// Top-level manifest describing all available model packs.
///
/// Ships bundled for offline use; refreshed from CDN when connectivity allows.
public struct ModelManifest: Codable, Sendable {
    public let version: Int
    public let lastUpdated: Date
    public let packs: [ModelPack]

    public init(version: Int, lastUpdated: Date, packs: [ModelPack]) {
        self.version = version
        self.lastUpdated = lastUpdated
        self.packs = packs
    }
}

// MARK: - Model Pack

/// A downloadable group of models that enables specific capabilities.
///
/// Each pack contains one or more GGUF model files, a minimum RAM requirement,
/// and the list of capability IDs it unlocks.
public struct ModelPack: Codable, Sendable, Identifiable {
    public let id: String
    public let displayName: String
    public let description: String
    public let icon: String
    public let models: [ManifestModel]
    public let totalSizeBytes: Int64
    public let minRAMMB: Int
    public let capabilities: [String]
    /// Optional status override (e.g., "coming-soon"). Nil means available.
    public let status: String?

    /// Well-known status values for manifest packs.
    public static let statusComingSoon = "coming-soon"

    /// Whether this pack is marked as coming soon in the manifest.
    public var isComingSoon: Bool { status == Self.statusComingSoon }

    public init(
        id: String,
        displayName: String,
        description: String,
        icon: String,
        models: [ManifestModel],
        totalSizeBytes: Int64,
        minRAMMB: Int,
        capabilities: [String],
        status: String? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.description = description
        self.icon = icon
        self.models = models
        self.totalSizeBytes = totalSizeBytes
        self.minRAMMB = minRAMMB
        self.capabilities = capabilities
        self.status = status
    }
}

// MARK: - Manifest Model

/// A single model file within a pack, with download path and integrity checksum.
public struct ManifestModel: Codable, Sendable {
    public let fileName: String
    public let sizeBytes: Int64
    public let sha256: String
    /// Relative download path (e.g., "/models/banking-poc/model.gguf").
    /// Resolved against a base URL at download time via `resolvedDownloadURL(base:)`.
    public let downloadPath: String
    public let quantization: String
    public let contextLength: Int
    public let gpuLayers: Int

    public init(
        fileName: String,
        sizeBytes: Int64,
        sha256: String,
        downloadPath: String,
        quantization: String,
        contextLength: Int,
        gpuLayers: Int
    ) {
        self.fileName = fileName
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
        self.downloadPath = downloadPath
        self.quantization = quantization
        self.contextLength = contextLength
        self.gpuLayers = gpuLayers
    }

    /// Resolve the download path against a base URL.
    public func resolvedDownloadURL(base: String) -> URL? {
        URL(string: base + downloadPath)
    }
}

// MARK: - Manifest Decoding

extension ModelManifest {

    /// Decode a manifest from bundled or downloaded JSON data.
    public static func from(data: Data) throws -> ModelManifest {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(ModelManifest.self, from: data)
    }

    /// Encode to JSON for caching.
    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }
}
