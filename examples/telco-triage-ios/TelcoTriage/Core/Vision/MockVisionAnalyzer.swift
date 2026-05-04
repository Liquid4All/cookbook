import Foundation
import UIKit

/// Alpha vision analyzer. Does rudimentary heuristic classification so the
/// demo is useful before the real LFM2.5-VL pack drops in — aspect ratio
/// and dominant color give us enough signal to route to one of three
/// canned responses (router / bill / error screen).
///
/// Implemented as an `actor` so concurrent `analyze` calls serialize and
/// the `@MainActor`-isolated `packManager` reference is only accessed from
/// inside `MainActor.run` blocks.
public actor MockVisionAnalyzer: VisionAnalyzer {
    private let packManager: SpecialistPackManager

    public init(packManager: SpecialistPackManager) {
        self.packManager = packManager
    }

    public func analyze(image: UIImage, prompt: String) async throws -> VisionResult {
        let start = Date()
        // Simulate inference time — the real VL-450M at Q4_0 runs around 300–600ms
        // on an A15, so target that band.
        try? await Task.sleep(nanoseconds: UInt64.random(in: 350_000_000...700_000_000))

        let installed = await MainActor.run { packManager.isInstalled(SpecialistPack.vision.id) }
        let category = inferCategory(image: image, prompt: prompt)

        let (headline, detail, hints): (String, String, [String])
        switch category {
        case .router:
            headline = "Router lights photograph"
            detail = installed
                ? "The solid white indicator on the front means your router has internet and 5 GHz Wi-Fi is broadcasting. No action needed."
                : "I detected a router but with the Visual Troubleshoot pack I can read the LED pattern exactly. Download it to get a precise diagnosis."
            hints = installed
                ? ["If the light turns yellow later, run a speed test", "If it turns red, the Restart Router tool usually fixes it"]
                : ["Install the Visual Troubleshoot pack for LED-pattern diagnosis"]
        case .bill:
            headline = "Bill screenshot"
            detail = installed
                ? "This bill has your base fiber plan plus one recurring equipment fee. The $15 'other charge' is your set-top-box rental."
                : "Looks like a bill. With the Visual Troubleshoot pack installed I can OCR line items and explain each charge."
            hints = installed
                ? ["Tap the set-top fee to see rental-vs-buy math", "Ask me to compare this to last month"]
                : ["Install the Visual Troubleshoot pack for line-item explanation"]
        case .errorScreen:
            headline = "Device error screen"
            detail = installed
                ? "Error shown: 'Cannot reach carrier servers'. This usually clears after a router restart."
                : "Looks like an error screen. Install the Visual Troubleshoot pack to OCR the exact message."
            hints = installed
                ? ["Try the Restart Router tool — I can run it for you"]
                : ["Install the Visual Troubleshoot pack to read the message"]
        case .other:
            headline = "Photo received"
            detail = installed
                ? "I don't see a router, bill, or error screen in this image. Try a clearer photo of the part you want help with."
                : "I can only guess from shape / color right now. Install the Visual Troubleshoot pack and I'll describe what's in the image on-device."
            hints = []
        }

        return VisionResult(
            category: category,
            headline: headline,
            detail: detail,
            actionableHints: hints,
            latencyMS: Int(Date().timeIntervalSince(start) * 1000),
            usedPack: installed
        )
    }

    // MARK: - Heuristic classification

    private func inferCategory(image: UIImage, prompt: String) -> VisionResult.Category {
        let lowered = prompt.lowercased()
        if lowered.contains("bill") || lowered.contains("charge") {
            return .bill
        }
        if lowered.contains("router") || lowered.contains("light") || lowered.contains("led") {
            return .router
        }
        if lowered.contains("error") || lowered.contains("screen") || lowered.contains("message") {
            return .errorScreen
        }

        // Aspect ratio / color fallbacks.
        let aspect = image.size.width / max(image.size.height, 1)
        if aspect < 0.7 { return .bill }              // tall → portrait screenshot
        if let color = image.dominantColorHex(), color.hasPrefix("white") { return .router }
        return .other
    }
}

private extension UIImage {
    /// Naive dominant-color check. Good enough for the heuristic fallback.
    func dominantColorHex() -> String? {
        guard let cg = cgImage else { return nil }
        let width = 1
        let height = 1
        var pixel = [UInt8](repeating: 0, count: 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixel,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        let (r, g, b) = (pixel[0], pixel[1], pixel[2])
        if r > 220 && g > 220 && b > 220 { return "white" }
        if r < 40 && g < 40 && b < 40 { return "black" }
        return nil
    }
}
