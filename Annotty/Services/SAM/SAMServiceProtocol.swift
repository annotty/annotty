import Foundation
import CoreGraphics

/// Protocol for SAM (Segment Anything Model) integration
/// MVP: Stub implementation only
/// Future: On-device inference with CoreML
protocol SAMServiceProtocol {
    /// Initialize the SAM model (load weights)
    func initialize() async throws

    /// Check if model is ready for inference
    var isReady: Bool { get }

    /// Refine a mask region using SAM
    /// - Parameters:
    ///   - image: Source image crop (will be resized to 1024 on longest edge)
    ///   - existingMask: Existing mask as prompt
    ///   - bbox: Bounding box of the region in image coordinates
    /// - Returns: Refined mask data
    func refine(
        image: CGImage,
        existingMask: [UInt8],
        bbox: CGRect
    ) async throws -> [UInt8]
}

/// Stub implementation of SAM service for MVP
/// Returns unchanged mask (no-op)
class SAMStubService: SAMServiceProtocol {
    var isReady: Bool { false }

    func initialize() async throws {
        // No-op for MVP
        print("SAM stub: initialize() called - no operation")
    }

    func refine(
        image: CGImage,
        existingMask: [UInt8],
        bbox: CGRect
    ) async throws -> [UInt8] {
        // Return unchanged mask for MVP
        print("SAM stub: refine() called - returning unchanged mask")
        return existingMask
    }
}

/// SAM service error types
enum SAMError: Error, LocalizedError {
    case modelNotLoaded
    case inferenceError(String)
    case invalidInput

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "SAM model is not loaded"
        case .inferenceError(let message):
            return "SAM inference error: \(message)"
        case .invalidInput:
            return "Invalid input for SAM"
        }
    }
}
