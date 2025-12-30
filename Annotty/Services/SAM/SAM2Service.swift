import Foundation
import CoreML
import CoreGraphics
import UIKit
import Combine
import Metal

/// SAM model type selection
enum SAMModelType: String, CaseIterable, Identifiable {
    case tiny = "Tiny"
    case small = "Small"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .tiny: return "SAM Tiny"
        case .small: return "SAM Small"
        }
    }

    var description: String {
        switch self {
        case .tiny: return "Faster, less accurate (~11M params)"
        case .small: return "Slower, more accurate (~38M params)"
        }
    }
}

/// SAM 2.1 Core ML Service for point prompt segmentation
/// Supports both Tiny and Small models with runtime switching
@MainActor
final class SAM2Service: ObservableObject {
    // MARK: - Published State

    @Published private(set) var isReady: Bool = false
    @Published private(set) var isProcessing: Bool = false
    @Published private(set) var lastError: String?
    @Published private(set) var currentModelType: SAMModelType?

    // MARK: - Models (stored as Any to support both Tiny and Small)

    private var imageEncoderTiny: SAM2_1TinyImageEncoderFLOAT16?
    private var promptEncoderTiny: SAM2_1TinyPromptEncoderFLOAT16?
    private var maskDecoderTiny: SAM2_1TinyMaskDecoderFLOAT16?

    private var imageEncoderSmall: SAM2_1SmallImageEncoderFLOAT16?
    private var promptEncoderSmall: SAM2_1SmallPromptEncoderFLOAT16?
    private var maskDecoderSmall: SAM2_1SmallMaskDecoderFLOAT16?

    // MARK: - Cached Embeddings

    /// Cached image embedding for repeated prompts on same image
    private var cachedImageEmbedding: MLMultiArray?
    private var cachedFeatsS0: MLMultiArray?
    private var cachedFeatsS1: MLMultiArray?
    private var cachedImageSize: CGSize?

    /// Input size expected by the model
    static let inputSize: Int = 1024

    /// Output mask size from decoder
    static let maskSize: Int = 256

    // MARK: - Initialization

    init() {
        print("[SAM2] Service initialized (models not loaded yet)")
    }

    /// Load all three Core ML models for the specified model type
    /// - Parameter modelType: The SAM model variant to load (Tiny or Small)
    func loadModels(modelType: SAMModelType = .tiny) async throws {
        // If already loaded with the same model type, skip
        if isReady && currentModelType == modelType {
            print("[SAM2] Models already loaded for \(modelType.displayName)")
            return
        }

        // If switching models, unload current models first
        if isReady && currentModelType != modelType {
            print("[SAM2] Switching from \(currentModelType?.displayName ?? "none") to \(modelType.displayName)")
            unloadModels()
        }

        print("[SAM2] Loading \(modelType.displayName) models...")

        do {
            // Auto-detect GPU capability based on Metal GPU Family
            let config = MLModelConfiguration()
            config.computeUnits = Self.detectOptimalComputeUnits()

            switch modelType {
            case .tiny:
                async let encoder = SAM2_1TinyImageEncoderFLOAT16(configuration: config)
                async let prompt = SAM2_1TinyPromptEncoderFLOAT16(configuration: config)
                async let decoder = SAM2_1TinyMaskDecoderFLOAT16(configuration: config)

                imageEncoderTiny = try await encoder
                promptEncoderTiny = try await prompt
                maskDecoderTiny = try await decoder

            case .small:
                async let encoder = SAM2_1SmallImageEncoderFLOAT16(configuration: config)
                async let prompt = SAM2_1SmallPromptEncoderFLOAT16(configuration: config)
                async let decoder = SAM2_1SmallMaskDecoderFLOAT16(configuration: config)

                imageEncoderSmall = try await encoder
                promptEncoderSmall = try await prompt
                maskDecoderSmall = try await decoder
            }

            currentModelType = modelType
            isReady = true
            lastError = nil
            print("[SAM2] \(modelType.displayName) models loaded successfully")
        } catch {
            lastError = error.localizedDescription
            print("[SAM2] Failed to load models: \(error)")
            throw SAMError.modelNotLoaded
        }
    }

    /// Unload current models to free memory
    func unloadModels() {
        imageEncoderTiny = nil
        promptEncoderTiny = nil
        maskDecoderTiny = nil
        imageEncoderSmall = nil
        promptEncoderSmall = nil
        maskDecoderSmall = nil
        clearCache()
        isReady = false
        currentModelType = nil
        print("[SAM2] Models unloaded")
    }

    /// Detect optimal compute units based on device GPU capability
    /// - Returns: `.cpuAndGPU` for A13/M1+ devices, `.cpuOnly` for older devices
    private static func detectOptimalComputeUnits() -> MLComputeUnits {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[SAM2] No Metal device found, using CPU only")
            return .cpuOnly
        }

        // Check for Apple GPU Family 6+ (A13, M1, M2, M3, M4, etc.)
        // Family 6+ supports Metal residency sets required by SAM2 Core ML ops
        if device.supportsFamily(.apple6) {
            print("[SAM2] GPU Family Apple6+ detected (\(device.name)) - using CPU + GPU")
            return .cpuAndGPU
        } else {
            print("[SAM2] GPU Family < Apple6 detected (\(device.name)) - using CPU only")
            return .cpuOnly
        }
    }

    // MARK: - Image Encoding

    /// Encode image and cache embeddings for repeated prompts
    func encodeImage(_ cgImage: CGImage) async throws {
        guard let modelType = currentModelType else {
            throw SAMError.modelNotLoaded
        }

        isProcessing = true
        defer { isProcessing = false }

        // Resize to 1024x1024
        guard let resizedBuffer = resizeImageToPixelBuffer(cgImage, size: Self.inputSize) else {
            throw SAMError.invalidInput
        }

        print("[SAM2] Encoding image (\(cgImage.width)x\(cgImage.height) -> 1024x1024) with \(modelType.displayName)...")

        switch modelType {
        case .tiny:
            guard let encoder = imageEncoderTiny else { throw SAMError.modelNotLoaded }
            let output = try encoder.prediction(image: resizedBuffer)
            cachedImageEmbedding = output.image_embedding
            cachedFeatsS0 = output.feats_s0
            cachedFeatsS1 = output.feats_s1

        case .small:
            guard let encoder = imageEncoderSmall else { throw SAMError.modelNotLoaded }
            let output = try encoder.prediction(image: resizedBuffer)
            cachedImageEmbedding = output.image_embedding
            cachedFeatsS0 = output.feats_s0
            cachedFeatsS1 = output.feats_s1
        }

        cachedImageSize = CGSize(width: cgImage.width, height: cgImage.height)
        print("[SAM2] Image encoded and cached")
    }

    /// Clear cached embeddings (call when image changes)
    func clearCache() {
        cachedImageEmbedding = nil
        cachedFeatsS0 = nil
        cachedFeatsS1 = nil
        cachedImageSize = nil
        print("[SAM2] Cache cleared")
    }

    // MARK: - Point Prompt Prediction

    /// Predict mask from a single point prompt
    /// - Parameters:
    ///   - point: Point in normalized coordinates (0-1)
    ///   - isForeground: true for foreground point, false for background
    /// - Returns: Binary mask array matching the original image size
    func predictFromPoint(
        point: CGPoint,
        isForeground: Bool = true
    ) async throws -> SAMMaskResult {
        guard let modelType = currentModelType,
              let imageEmbedding = cachedImageEmbedding,
              let featsS0 = cachedFeatsS0,
              let featsS1 = cachedFeatsS1,
              let originalSize = cachedImageSize else {
            throw SAMError.modelNotLoaded
        }

        isProcessing = true
        defer { isProcessing = false }

        // Convert normalized point to 1024x1024 coordinates
        let scaledPoint = CGPoint(
            x: point.x * CGFloat(Self.inputSize),
            y: point.y * CGFloat(Self.inputSize)
        )

        print("[SAM2] Predicting from point: (\(String(format: "%.1f", scaledPoint.x)), \(String(format: "%.1f", scaledPoint.y))) with \(modelType.displayName)")

        // Create point coordinates array [1, 1, 2]
        let pointsArray = try MLMultiArray(shape: [1, 1, 2], dataType: .float16)
        pointsArray[[0, 0, 0] as [NSNumber]] = NSNumber(value: Float(scaledPoint.x))
        pointsArray[[0, 0, 1] as [NSNumber]] = NSNumber(value: Float(scaledPoint.y))

        // Create labels array [1, 1] - 1 for foreground, 0 for background
        let labelsArray = try MLMultiArray(shape: [1, 1], dataType: .float16)
        labelsArray[[0, 0] as [NSNumber]] = NSNumber(value: isForeground ? 1 : 0)

        // Run prompt encoder and mask decoder based on model type
        let (lowResMasks, scores): (MLMultiArray, MLMultiArray)

        switch modelType {
        case .tiny:
            guard let promptEnc = promptEncoderTiny, let maskDec = maskDecoderTiny else {
                throw SAMError.modelNotLoaded
            }
            let promptOutput = try promptEnc.prediction(points: pointsArray, labels: labelsArray)
            let maskOutput = try maskDec.prediction(
                image_embedding: imageEmbedding,
                sparse_embedding: promptOutput.sparse_embeddings,
                dense_embedding: promptOutput.dense_embeddings,
                feats_s0: featsS0,
                feats_s1: featsS1
            )
            lowResMasks = maskOutput.low_res_masks
            scores = maskOutput.scores

        case .small:
            guard let promptEnc = promptEncoderSmall, let maskDec = maskDecoderSmall else {
                throw SAMError.modelNotLoaded
            }
            let promptOutput = try promptEnc.prediction(points: pointsArray, labels: labelsArray)
            let maskOutput = try maskDec.prediction(
                image_embedding: imageEmbedding,
                sparse_embedding: promptOutput.sparse_embeddings,
                dense_embedding: promptOutput.dense_embeddings,
                feats_s0: featsS0,
                feats_s1: featsS1
            )
            lowResMasks = maskOutput.low_res_masks
            scores = maskOutput.scores
        }

        // Get best mask (highest score)
        var bestIdx = 0
        var bestScore: Float = -Float.infinity
        for i in 0..<3 {
            let score = scores[[0, i] as [NSNumber]].floatValue
            if score > bestScore {
                bestScore = score
                bestIdx = i
            }
        }

        print("[SAM2] Best mask index: \(bestIdx), score: \(String(format: "%.3f", bestScore))")

        // Extract mask and upscale to original image size
        let mask = extractMask(from: lowResMasks, index: bestIdx)
        let upscaledMask = upscaleMask(mask, toSize: originalSize)

        return SAMMaskResult(
            mask: upscaledMask,
            score: bestScore,
            size: originalSize
        )
    }

    // MARK: - Multi-Point Prediction

    /// Predict mask from multiple points
    /// - Parameters:
    ///   - points: Array of (point, isForeground) tuples in normalized coordinates
    /// - Returns: Binary mask array matching the original image size
    func predictFromPoints(
        points: [(point: CGPoint, isForeground: Bool)]
    ) async throws -> SAMMaskResult {
        guard let modelType = currentModelType,
              let imageEmbedding = cachedImageEmbedding,
              let featsS0 = cachedFeatsS0,
              let featsS1 = cachedFeatsS1,
              let originalSize = cachedImageSize else {
            throw SAMError.modelNotLoaded
        }

        guard points.count >= 1 && points.count <= 16 else {
            throw SAMError.invalidInput
        }

        isProcessing = true
        defer { isProcessing = false }

        let numPoints = points.count

        // Create point coordinates array [1, numPoints, 2]
        let pointsArray = try MLMultiArray(shape: [1, numPoints as NSNumber, 2], dataType: .float16)
        let labelsArray = try MLMultiArray(shape: [1, numPoints as NSNumber], dataType: .float16)

        for (i, p) in points.enumerated() {
            let scaledX = Float(p.point.x * CGFloat(Self.inputSize))
            let scaledY = Float(p.point.y * CGFloat(Self.inputSize))
            pointsArray[[0, i, 0] as [NSNumber]] = NSNumber(value: scaledX)
            pointsArray[[0, i, 1] as [NSNumber]] = NSNumber(value: scaledY)
            labelsArray[[0, i] as [NSNumber]] = NSNumber(value: p.isForeground ? 1 : 0)
        }

        print("[SAM2] Predicting from \(numPoints) points with \(modelType.displayName)")

        // Run prompt encoder and mask decoder based on model type
        let (lowResMasks, scores): (MLMultiArray, MLMultiArray)

        switch modelType {
        case .tiny:
            guard let promptEnc = promptEncoderTiny, let maskDec = maskDecoderTiny else {
                throw SAMError.modelNotLoaded
            }
            let promptOutput = try promptEnc.prediction(points: pointsArray, labels: labelsArray)
            let maskOutput = try maskDec.prediction(
                image_embedding: imageEmbedding,
                sparse_embedding: promptOutput.sparse_embeddings,
                dense_embedding: promptOutput.dense_embeddings,
                feats_s0: featsS0,
                feats_s1: featsS1
            )
            lowResMasks = maskOutput.low_res_masks
            scores = maskOutput.scores

        case .small:
            guard let promptEnc = promptEncoderSmall, let maskDec = maskDecoderSmall else {
                throw SAMError.modelNotLoaded
            }
            let promptOutput = try promptEnc.prediction(points: pointsArray, labels: labelsArray)
            let maskOutput = try maskDec.prediction(
                image_embedding: imageEmbedding,
                sparse_embedding: promptOutput.sparse_embeddings,
                dense_embedding: promptOutput.dense_embeddings,
                feats_s0: featsS0,
                feats_s1: featsS1
            )
            lowResMasks = maskOutput.low_res_masks
            scores = maskOutput.scores
        }

        // Get best mask
        var bestIdx = 0
        var bestScore: Float = -Float.infinity
        for i in 0..<3 {
            let score = scores[[0, i] as [NSNumber]].floatValue
            if score > bestScore {
                bestScore = score
                bestIdx = i
            }
        }

        let mask = extractMask(from: lowResMasks, index: bestIdx)
        let upscaledMask = upscaleMask(mask, toSize: originalSize)

        return SAMMaskResult(
            mask: upscaledMask,
            score: bestScore,
            size: originalSize
        )
    }

    // MARK: - BBox Prediction

    /// Predict mask from a bounding box prompt
    /// - Parameters:
    ///   - topLeft: Top-left corner in normalized coordinates (0-1)
    ///   - bottomRight: Bottom-right corner in normalized coordinates (0-1)
    /// - Returns: Binary mask array matching the original image size
    func predictFromBBox(
        topLeft: CGPoint,
        bottomRight: CGPoint
    ) async throws -> SAMMaskResult {
        guard let modelType = currentModelType,
              let imageEmbedding = cachedImageEmbedding,
              let featsS0 = cachedFeatsS0,
              let featsS1 = cachedFeatsS1,
              let originalSize = cachedImageSize else {
            throw SAMError.modelNotLoaded
        }

        isProcessing = true
        defer { isProcessing = false }

        // Convert normalized points to 1024x1024 coordinates
        let scaledTopLeft = CGPoint(
            x: topLeft.x * CGFloat(Self.inputSize),
            y: topLeft.y * CGFloat(Self.inputSize)
        )
        let scaledBottomRight = CGPoint(
            x: bottomRight.x * CGFloat(Self.inputSize),
            y: bottomRight.y * CGFloat(Self.inputSize)
        )

        print("[SAM2] Predicting from BBox: (\(String(format: "%.1f", scaledTopLeft.x)), \(String(format: "%.1f", scaledTopLeft.y))) - (\(String(format: "%.1f", scaledBottomRight.x)), \(String(format: "%.1f", scaledBottomRight.y))) with \(modelType.displayName)")

        // Create point coordinates array [1, 2, 2] - two points for bbox
        let pointsArray = try MLMultiArray(shape: [1, 2, 2], dataType: .float16)
        pointsArray[[0, 0, 0] as [NSNumber]] = NSNumber(value: Float(scaledTopLeft.x))
        pointsArray[[0, 0, 1] as [NSNumber]] = NSNumber(value: Float(scaledTopLeft.y))
        pointsArray[[0, 1, 0] as [NSNumber]] = NSNumber(value: Float(scaledBottomRight.x))
        pointsArray[[0, 1, 1] as [NSNumber]] = NSNumber(value: Float(scaledBottomRight.y))

        // Create labels array [1, 2] - label 2 for top-left, label 3 for bottom-right
        let labelsArray = try MLMultiArray(shape: [1, 2], dataType: .float16)
        labelsArray[[0, 0] as [NSNumber]] = NSNumber(value: 2)  // top-left corner
        labelsArray[[0, 1] as [NSNumber]] = NSNumber(value: 3)  // bottom-right corner

        // Run prompt encoder and mask decoder based on model type
        let (lowResMasks, scores): (MLMultiArray, MLMultiArray)

        switch modelType {
        case .tiny:
            guard let promptEnc = promptEncoderTiny, let maskDec = maskDecoderTiny else {
                throw SAMError.modelNotLoaded
            }
            let promptOutput = try promptEnc.prediction(points: pointsArray, labels: labelsArray)
            let maskOutput = try maskDec.prediction(
                image_embedding: imageEmbedding,
                sparse_embedding: promptOutput.sparse_embeddings,
                dense_embedding: promptOutput.dense_embeddings,
                feats_s0: featsS0,
                feats_s1: featsS1
            )
            lowResMasks = maskOutput.low_res_masks
            scores = maskOutput.scores

        case .small:
            guard let promptEnc = promptEncoderSmall, let maskDec = maskDecoderSmall else {
                throw SAMError.modelNotLoaded
            }
            let promptOutput = try promptEnc.prediction(points: pointsArray, labels: labelsArray)
            let maskOutput = try maskDec.prediction(
                image_embedding: imageEmbedding,
                sparse_embedding: promptOutput.sparse_embeddings,
                dense_embedding: promptOutput.dense_embeddings,
                feats_s0: featsS0,
                feats_s1: featsS1
            )
            lowResMasks = maskOutput.low_res_masks
            scores = maskOutput.scores
        }

        // Get best mask (highest score)
        var bestIdx = 0
        var bestScore: Float = -Float.infinity
        for i in 0..<3 {
            let score = scores[[0, i] as [NSNumber]].floatValue
            if score > bestScore {
                bestScore = score
                bestIdx = i
            }
        }

        print("[SAM2] Best mask index: \(bestIdx), score: \(String(format: "%.3f", bestScore))")

        // Extract mask and upscale to original image size
        let mask = extractMask(from: lowResMasks, index: bestIdx)
        let upscaledMask = upscaleMask(mask, toSize: originalSize)

        return SAMMaskResult(
            mask: upscaledMask,
            score: bestScore,
            size: originalSize
        )
    }

    // MARK: - Private Helpers

    /// Resize CGImage to pixel buffer for model input
    private func resizeImageToPixelBuffer(_ image: CGImage, size: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            size, size,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            return nil
        }

        // Draw image centered and scaled to fit
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        return buffer
    }

    /// Extract single mask from multi-mask output
    private func extractMask(from masks: MLMultiArray, index: Int) -> [Float] {
        let maskSize = Self.maskSize
        var result = [Float](repeating: 0, count: maskSize * maskSize)

        for y in 0..<maskSize {
            for x in 0..<maskSize {
                let value = masks[[0, index as NSNumber, y as NSNumber, x as NSNumber] as [NSNumber]].floatValue
                result[y * maskSize + x] = value
            }
        }

        return result
    }

    /// Upscale mask to target size using bilinear interpolation
    private func upscaleMask(_ mask: [Float], toSize size: CGSize) -> [UInt8] {
        let srcSize = Self.maskSize
        let dstWidth = Int(size.width)
        let dstHeight = Int(size.height)

        var result = [UInt8](repeating: 0, count: dstWidth * dstHeight)

        let scaleX = Float(srcSize) / Float(dstWidth)
        let scaleY = Float(srcSize) / Float(dstHeight)

        for y in 0..<dstHeight {
            for x in 0..<dstWidth {
                // Map to source coordinates
                let srcX = Float(x) * scaleX
                let srcY = Float(y) * scaleY

                // Bilinear interpolation
                let x0 = Int(srcX)
                let y0 = Int(srcY)
                let x1 = min(x0 + 1, srcSize - 1)
                let y1 = min(y0 + 1, srcSize - 1)

                let fx = srcX - Float(x0)
                let fy = srcY - Float(y0)

                let v00 = mask[y0 * srcSize + x0]
                let v10 = mask[y0 * srcSize + x1]
                let v01 = mask[y1 * srcSize + x0]
                let v11 = mask[y1 * srcSize + x1]

                let value = v00 * (1 - fx) * (1 - fy) +
                           v10 * fx * (1 - fy) +
                           v01 * (1 - fx) * fy +
                           v11 * fx * fy

                // Threshold at 0 for binary mask
                result[y * dstWidth + x] = value > 0 ? 1 : 0
            }
        }

        return result
    }
}

// MARK: - Result Types

/// Result of SAM mask prediction
struct SAMMaskResult {
    /// Binary mask (0 or 1) at original image resolution
    let mask: [UInt8]
    /// Confidence score for this mask
    let score: Float
    /// Size of the mask (matches original image)
    let size: CGSize
}
