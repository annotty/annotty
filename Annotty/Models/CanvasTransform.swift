import Foundation
import CoreGraphics
import simd

/// Manages pan, zoom, and rotation transformations for the canvas
/// Provides conversion between screen coordinates and internal mask coordinates
struct CanvasTransform {
    /// Translation offset in screen points
    var translation: CGPoint = .zero

    /// Zoom scale factor (1.0 = original size)
    var scale: CGFloat = 1.0

    /// Rotation angle in radians (free rotation, no snap)
    var rotation: CGFloat = 0.0

    /// The mask scale factor (from image to internal mask, typically 2.0 with 4096 max clamp)
    var maskScaleFactor: Float = 2.0

    /// Minimum allowed zoom scale
    static let minScale: CGFloat = 0.1

    /// Maximum allowed zoom scale
    static let maxScale: CGFloat = 10.0

    /// Combined transformation matrix
    var matrix: CGAffineTransform {
        CGAffineTransform.identity
            .translatedBy(x: translation.x, y: translation.y)
            .rotated(by: rotation)
            .scaledBy(x: scale, y: scale)
    }

    /// Inverse of the combined transformation matrix
    var inverseMatrix: CGAffineTransform {
        matrix.inverted()
    }

    /// Convert screen point to image coordinates
    func screenToImage(_ screenPoint: CGPoint) -> CGPoint {
        screenPoint.applying(inverseMatrix)
    }

    /// Convert image coordinates to screen point
    func imageToScreen(_ imagePoint: CGPoint) -> CGPoint {
        imagePoint.applying(matrix)
    }

    /// Convert screen point to internal mask coordinates
    /// Uses the dynamic mask scale factor (2x with 4096px max clamp)
    func screenToMask(_ screenPoint: CGPoint) -> CGPoint {
        let imagePoint = screenToImage(screenPoint)
        return CGPoint(
            x: imagePoint.x * CGFloat(maskScaleFactor),
            y: imagePoint.y * CGFloat(maskScaleFactor)
        )
    }

    /// Convert internal mask coordinates to screen point
    func maskToScreen(_ maskPoint: CGPoint) -> CGPoint {
        let imagePoint = CGPoint(
            x: maskPoint.x / CGFloat(maskScaleFactor),
            y: maskPoint.y / CGFloat(maskScaleFactor)
        )
        return imageToScreen(imagePoint)
    }

    /// Convert screen radius to mask radius
    /// Zoom-independent: radius in mask coordinates stays constant regardless of zoom
    func screenRadiusToMask(_ screenRadius: CGFloat) -> Float {
        // Radius is zoom-independent, so we only apply mask scale factor
        Float(screenRadius) * maskScaleFactor
    }

    /// Apply pan gesture delta
    mutating func applyPan(delta: CGPoint) {
        translation.x += delta.x
        translation.y += delta.y
    }

    /// Apply pinch gesture scale factor around a center point
    mutating func applyPinch(scaleFactor: CGFloat, center: CGPoint) {
        let newScale = (scale * scaleFactor).clamped(to: Self.minScale...Self.maxScale)
        let actualScaleFactor = newScale / scale

        // Adjust translation to keep center point fixed
        translation.x = center.x - (center.x - translation.x) * actualScaleFactor
        translation.y = center.y - (center.y - translation.y) * actualScaleFactor

        scale = newScale
    }

    /// Apply rotation gesture angle delta around a center point
    mutating func applyRotation(angleDelta: CGFloat, center: CGPoint) {
        // Free rotation, no snapping
        rotation += angleDelta

        // Adjust translation to rotate around center point
        let cosA = cos(angleDelta)
        let sinA = sin(angleDelta)
        let dx = translation.x - center.x
        let dy = translation.y - center.y

        translation.x = center.x + dx * cosA - dy * sinA
        translation.y = center.y + dx * sinA + dy * cosA
    }

    /// Reset to identity transform
    mutating func reset() {
        translation = .zero
        scale = 1.0
        rotation = 0.0
    }

    /// Convert to simd matrix for Metal shaders
    func toSimdMatrix() -> simd_float3x3 {
        let m = matrix
        return simd_float3x3(
            simd_float3(Float(m.a), Float(m.b), 0),
            simd_float3(Float(m.c), Float(m.d), 0),
            simd_float3(Float(m.tx), Float(m.ty), 1)
        )
    }

    /// Inverse simd matrix for Metal shaders
    func toInverseSimdMatrix() -> simd_float3x3 {
        let m = inverseMatrix
        return simd_float3x3(
            simd_float3(Float(m.a), Float(m.b), 0),
            simd_float3(Float(m.c), Float(m.d), 0),
            simd_float3(Float(m.tx), Float(m.ty), 1)
        )
    }
}

// MARK: - CGFloat Extension

extension CGFloat {
    func clamped(to range: ClosedRange<CGFloat>) -> CGFloat {
        if self < range.lowerBound { return range.lowerBound }
        if self > range.upperBound { return range.upperBound }
        return self
    }
}
