import XCTest
@testable import TelcoTriage

final class ClassifierHeadMetadataTests: XCTestCase {
    func test_labelsOnlyMetadataPopulatesClassifierMaps() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("classifier-head-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let metaURL = tempDir.appendingPathComponent("classifier_meta.json")
        let weightsURL = tempDir.appendingPathComponent("classifier_weights.bin")
        let biasURL = tempDir.appendingPathComponent("classifier_bias.bin")

        let meta = """
        {
          "num_classes": 2,
          "hidden_dim": 2,
          "task": "labels_only_metadata",
          "format": "float32_row_major",
          "type": "multi_label",
          "labels": ["local_answer", "cloud_assist"]
        }
        """
        try meta.write(to: metaURL, atomically: true, encoding: .utf8)

        try Self.writeFloats([-10, 0, 10, 0], to: weightsURL)
        try Self.writeFloats([0, 0], to: biasURL)

        let head = try ClassifierHead(weightsURL: weightsURL, biasURL: biasURL, metaURL: metaURL)
        XCTAssertEqual(head.id2label[0], "local_answer")
        XCTAssertEqual(head.id2label[1], "cloud_assist")
        XCTAssertEqual(head.label2id["cloud_assist"], 1)

        let prediction = head.classifyMultiLabel([1, 0], threshold: 0.5)
        XCTAssertEqual(prediction.activeLabels, ["cloud_assist"])
    }

    private static func writeFloats(_ values: [Float], to url: URL) throws {
        var mutable = values
        let data = mutable.withUnsafeMutableBytes { Data($0) }
        try data.write(to: url)
    }
}
