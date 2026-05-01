import Foundation

/// Download operations for ModelManager.
///
/// Extracted to keep ModelManager.swift under 300 lines.
/// Contains the streaming download logic with atomic writes.
extension ModelManager {

    /// Internal download implementation with atomic write-to-temp-then-rename.
    ///
    /// Writes to a `.tmp` file during download, cleans up on error,
    /// and atomically renames to the final path on success.
    /// This prevents partial/corrupt files from being treated as valid downloads.
    func downloadFile(
        from remoteURL: URL,
        to localPath: URL,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws {
        var request = URLRequest(url: remoteURL)
        // Skip ngrok interstitial for tunnel-based model serving
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw LFMEngineError.downloadHTTPError(
                statusCode: (response as? HTTPURLResponse)?.statusCode ?? 0
            )
        }

        let totalBytes = httpResponse.expectedContentLength
        let fileName = localPath.lastPathComponent

        // Write to a temporary file, then atomically move on success.
        let tmpPath = localPath.appendingPathExtension("tmp")

        if fileManager.fileExists(atPath: tmpPath.path) {
            try fileManager.removeItem(at: tmpPath)
        }
        fileManager.createFile(atPath: tmpPath.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tmpPath)

        var downloaded: Int64 = 0
        var lastReportedPercent: Int = -1
        let chunkSize = 65_536

        do {
            var buffer = Data(capacity: chunkSize)
            for try await byte in asyncBytes {
                buffer.append(byte)
                if buffer.count >= chunkSize {
                    handle.write(buffer)
                    downloaded += Int64(buffer.count)
                    buffer.removeAll(keepingCapacity: true)

                    // Report progress at ~1% increments
                    let percent = totalBytes > 0 ? Int(downloaded * 100 / totalBytes) : 0
                    if percent > lastReportedPercent {
                        lastReportedPercent = percent
                        onProgress(DownloadProgress(
                            bytesDownloaded: downloaded,
                            totalBytes: totalBytes,
                            fileName: fileName
                        ))
                    }
                }
            }

            // Write remaining bytes
            if !buffer.isEmpty {
                handle.write(buffer)
                downloaded += Int64(buffer.count)
            }

            try handle.close()
        } catch {
            try? handle.close()
            try? fileManager.removeItem(at: tmpPath)
            throw error
        }

        // Atomic move: replace existing file (if any) with the completed download
        if fileManager.fileExists(atPath: localPath.path) {
            try fileManager.removeItem(at: localPath)
        }
        try fileManager.moveItem(at: tmpPath, to: localPath)

        onProgress(DownloadProgress(
            bytesDownloaded: downloaded,
            totalBytes: downloaded,
            fileName: fileName
        ))
    }
}
