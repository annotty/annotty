import Foundation
import Metal
import MetalKit
import CoreGraphics
import UIKit

/// Manages Metal textures for images and masks
class TextureManager {
    private let device: MTLDevice
    private let textureLoader: MTKTextureLoader

    /// Currently loaded source image texture
    private(set) var imageTexture: MTLTexture?

    /// Mask textures by class ID
    private(set) var maskTextures: [Int: MTLTexture] = [:]

    /// Source image size
    private(set) var imageSize: CGSize = .zero

    /// Internal mask size (2x with 4096 clamp)
    private(set) var maskSize: CGSize = .zero

    /// Scale factor from image to mask
    private(set) var maskScaleFactor: Float = 2.0

    /// Maximum mask dimension
    static let maxMaskDimension = 4096

    /// Maximum number of classes
    static let maxClasses = MaskClass.maxClasses

    init(device: MTLDevice) {
        self.device = device
        self.textureLoader = MTKTextureLoader(device: device)
    }

    // MARK: - Image Loading

    /// Load source image from URL
    func loadImage(from url: URL) throws {
        guard let image = UIImage(contentsOfFile: url.path),
              let cgImage = image.cgImage else {
            throw TextureError.invalidImage
        }

        try loadImage(cgImage)
    }

    /// Load source image from CGImage
    func loadImage(_ cgImage: CGImage) throws {
        let width = cgImage.width
        let height = cgImage.height
        imageSize = CGSize(width: width, height: height)

        // Calculate mask dimensions with 4096 max clamp
        let maxEdge = max(width, height)
        maskScaleFactor = min(2.0, Float(Self.maxMaskDimension) / Float(maxEdge))

        let maskWidth = Int(Float(width) * maskScaleFactor)
        let maskHeight = Int(Float(height) * maskScaleFactor)
        maskSize = CGSize(width: maskWidth, height: maskHeight)

        // Create image texture
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw TextureError.textureCreationFailed
        }

        // Copy image data to texture
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )

        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        texture.replace(
            region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                              size: MTLSize(width: width, height: height, depth: 1)),
            mipmapLevel: 0,
            withBytes: pixelData,
            bytesPerRow: bytesPerRow
        )

        imageTexture = texture

        // Clear existing mask textures
        maskTextures.removeAll()
    }

    // MARK: - Mask Textures

    /// Create or get mask texture for a class
    func getMaskTexture(for classID: Int) throws -> MTLTexture {
        if let existing = maskTextures[classID] {
            return existing
        }

        // Ensure image is loaded before creating mask
        guard imageTexture != nil else {
            throw TextureError.noImageLoaded
        }

        // Check class limit
        guard maskTextures.count < Self.maxClasses else {
            throw TextureError.maxClassesReached
        }

        let texture = try createMaskTexture()
        maskTextures[classID] = texture
        return texture
    }

    /// Create a new empty mask texture
    private func createMaskTexture() throws -> MTLTexture {
        // Guard against zero-size textures (no image loaded yet)
        guard maskSize.width > 0 && maskSize.height > 0 else {
            throw TextureError.noImageLoaded
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Uint,
            width: Int(maskSize.width),
            height: Int(maskSize.height),
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw TextureError.textureCreationFailed
        }

        // Initialize to zero
        let byteCount = Int(maskSize.width) * Int(maskSize.height)
        var zeros = [UInt8](repeating: 0, count: byteCount)
        texture.replace(
            region: MTLRegion(
                origin: MTLOrigin(x: 0, y: 0, z: 0),
                size: MTLSize(width: Int(maskSize.width), height: Int(maskSize.height), depth: 1)
            ),
            mipmapLevel: 0,
            withBytes: &zeros,
            bytesPerRow: Int(maskSize.width)
        )

        return texture
    }

    /// Upload mask data to texture
    func uploadMask(_ data: [UInt8], to classID: Int) throws {
        let texture = try getMaskTexture(for: classID)

        guard data.count == Int(maskSize.width) * Int(maskSize.height) else {
            throw TextureError.invalidMaskData
        }

        var mutableData = data
        texture.replace(
            region: MTLRegion(
                origin: MTLOrigin(x: 0, y: 0, z: 0),
                size: MTLSize(width: Int(maskSize.width), height: Int(maskSize.height), depth: 1)
            ),
            mipmapLevel: 0,
            withBytes: &mutableData,
            bytesPerRow: Int(maskSize.width)
        )
    }

    /// Read mask data from texture
    func readMask(from classID: Int) -> [UInt8]? {
        guard let texture = maskTextures[classID] else { return nil }

        let width = texture.width
        let height = texture.height
        var data = [UInt8](repeating: 0, count: width * height)

        texture.getBytes(
            &data,
            bytesPerRow: width,
            from: MTLRegion(
                origin: MTLOrigin(x: 0, y: 0, z: 0),
                size: MTLSize(width: width, height: height, depth: 1)
            ),
            mipmapLevel: 0
        )

        return data
    }

    /// Read a region from mask texture (for undo patches)
    func readMaskRegion(from classID: Int, bbox: CGRect) -> Data? {
        guard let texture = maskTextures[classID] else { return nil }

        // Use floor for min and ceil for max to ensure we capture all affected pixels
        let minX = max(0, Int(floor(bbox.minX)))
        let minY = max(0, Int(floor(bbox.minY)))
        let maxX = min(texture.width, Int(ceil(bbox.maxX)))
        let maxY = min(texture.height, Int(ceil(bbox.maxY)))

        let regionWidth = maxX - minX
        let regionHeight = maxY - minY

        guard regionWidth > 0 && regionHeight > 0 else { return nil }

        var data = [UInt8](repeating: 0, count: regionWidth * regionHeight)

        texture.getBytes(
            &data,
            bytesPerRow: regionWidth,
            from: MTLRegion(
                origin: MTLOrigin(x: minX, y: minY, z: 0),
                size: MTLSize(width: regionWidth, height: regionHeight, depth: 1)
            ),
            mipmapLevel: 0
        )

        return Data(data)
    }

    /// Write a region to mask texture (for undo restore)
    func writeMaskRegion(to classID: Int, bbox: CGRect, data: Data) {
        guard let texture = maskTextures[classID] else { return }

        // Use floor for min and ceil for max to match readMaskRegion
        let minX = max(0, Int(floor(bbox.minX)))
        let minY = max(0, Int(floor(bbox.minY)))
        let maxX = min(texture.width, Int(ceil(bbox.maxX)))
        let maxY = min(texture.height, Int(ceil(bbox.maxY)))

        let regionWidth = maxX - minX
        let regionHeight = maxY - minY

        guard regionWidth > 0 && regionHeight > 0 else { return }

        data.withUnsafeBytes { buffer in
            guard let ptr = buffer.baseAddress else { return }
            texture.replace(
                region: MTLRegion(
                    origin: MTLOrigin(x: minX, y: minY, z: 0),
                    size: MTLSize(width: regionWidth, height: regionHeight, depth: 1)
                ),
                mipmapLevel: 0,
                withBytes: ptr,
                bytesPerRow: regionWidth
            )
        }
    }

    /// Remove mask texture for a class
    func removeMaskTexture(for classID: Int) {
        maskTextures.removeValue(forKey: classID)
    }

    /// Clear all textures
    func clear() {
        imageTexture = nil
        maskTextures.removeAll()
        imageSize = .zero
        maskSize = .zero
    }
}

// MARK: - Errors

enum TextureError: Error, LocalizedError {
    case invalidImage
    case textureCreationFailed
    case maxClassesReached
    case invalidMaskData
    case noImageLoaded

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Failed to load image"
        case .textureCreationFailed:
            return "Failed to create Metal texture"
        case .maxClassesReached:
            return "Maximum number of classes reached (8)"
        case .invalidMaskData:
            return "Invalid mask data size"
        case .noImageLoaded:
            return "No image loaded - cannot create mask texture"
        }
    }
}
