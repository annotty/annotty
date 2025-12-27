import Foundation
import Metal
import MetalKit
import simd
import Combine

/// Main Metal rendering coordinator
/// Manages render pipelines, compute pipelines, and GPU operations
class MetalRenderer: NSObject, ObservableObject {
    // MARK: - Metal Objects

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let textureManager: TextureManager

    private var renderPipelineState: MTLRenderPipelineState?
    private var brushStampPipelineState: MTLComputePipelineState?
    private var clearMaskPipelineState: MTLComputePipelineState?

    // MARK: - Uniforms

    @Published var imageAlpha: Float = 1.0
    @Published var maskAlpha: Float = 0.5
    @Published var maskColor: simd_float4 = simd_float4(1, 0, 0, 0.5) // Red default

    // MARK: - State

    @Published var currentClassID: Int = 0
    var canvasTransform = CanvasTransform()
    private var viewportSize: CGSize = .zero
    private(set) var contentScaleFactor: CGFloat = 1.0

    // MARK: - State Flags

    private(set) var isPipelineReady: Bool = false

    // MARK: - Initialization

    override init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }

        self.device = device
        self.commandQueue = commandQueue
        self.textureManager = TextureManager(device: device)

        super.init()

        // Debug: Print struct sizes for Metal alignment verification
        print("📐 CanvasUniforms size: \(MemoryLayout<CanvasUniforms>.size) bytes")
        print("📐 CanvasUniforms stride: \(MemoryLayout<CanvasUniforms>.stride) bytes")
        print("📐 CanvasUniforms alignment: \(MemoryLayout<CanvasUniforms>.alignment) bytes")
        print("📐 simd_float3x3 size: \(MemoryLayout<simd_float3x3>.size) bytes")
        print("📐 simd_float3x3 stride: \(MemoryLayout<simd_float3x3>.stride) bytes")

        do {
            try setupPipelines()
            isPipelineReady = true
        } catch {
            print("⚠️ Metal pipeline setup failed: \(error)")
            print("⚠️ Make sure Shaders.metal is added to Xcode's Compile Sources")
            isPipelineReady = false
        }
    }

    // MARK: - Pipeline Setup

    private func setupPipelines() throws {
        guard let library = device.makeDefaultLibrary() else {
            print("❌ device.makeDefaultLibrary() returned nil")
            print("❌ Shaders.metal must be added to the Xcode project's Compile Sources")
            throw RendererError.libraryCreationFailed
        }

        // Render pipeline for canvas
        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.vertexFunction = library.makeFunction(name: "canvasVertex")
        renderDescriptor.fragmentFunction = library.makeFunction(name: "canvasFragment")
        renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        // Enable blending
        renderDescriptor.colorAttachments[0].isBlendingEnabled = true
        renderDescriptor.colorAttachments[0].rgbBlendOperation = .add
        renderDescriptor.colorAttachments[0].alphaBlendOperation = .add
        renderDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        renderDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
        renderDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        renderDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        renderPipelineState = try device.makeRenderPipelineState(descriptor: renderDescriptor)

        // Compute pipeline for brush stamp
        guard let brushStampFunction = library.makeFunction(name: "brushStamp") else {
            throw RendererError.functionNotFound("brushStamp")
        }
        brushStampPipelineState = try device.makeComputePipelineState(function: brushStampFunction)

        // Compute pipeline for clear mask
        guard let clearMaskFunction = library.makeFunction(name: "clearMask") else {
            throw RendererError.functionNotFound("clearMask")
        }
        clearMaskPipelineState = try device.makeComputePipelineState(function: clearMaskFunction)
    }

    // MARK: - Rendering

    /// Main render function called by MTKView
    func render(to drawable: CAMetalDrawable, with renderPassDescriptor: MTLRenderPassDescriptor) {
        guard let renderPipelineState = renderPipelineState,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        // Skip full rendering if no image is loaded - just clear to background
        guard textureManager.imageTexture != nil else {
            renderEncoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
            return
        }

        renderEncoder.setRenderPipelineState(renderPipelineState)

        // Set uniforms - use stride for proper Metal alignment
        var uniforms = createUniforms()
        let uniformsSize = MemoryLayout<CanvasUniforms>.stride
        renderEncoder.setVertexBytes(&uniforms, length: uniformsSize, index: 0)
        renderEncoder.setFragmentBytes(&uniforms, length: uniformsSize, index: 0)

        // Set textures
        if let imageTexture = textureManager.imageTexture {
            renderEncoder.setFragmentTexture(imageTexture, index: 0)
        }

        if let maskTexture = textureManager.maskTextures[currentClassID] {
            renderEncoder.setFragmentTexture(maskTexture, index: 1)
        }

        // Draw full-screen triangle
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        renderEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    /// Create uniforms structure for shaders
    private func createUniforms() -> CanvasUniforms {
        return CanvasUniforms(
            transform: canvasTransform.toSimdMatrix(),
            inverseTransform: canvasTransform.toInverseSimdMatrix(),
            imageAlpha: imageAlpha,
            maskAlpha: maskAlpha,
            _padding1: .zero,
            maskColor: maskColor,
            canvasSize: simd_float2(Float(viewportSize.width), Float(viewportSize.height)),
            imageSize: simd_float2(Float(textureManager.imageSize.width), Float(textureManager.imageSize.height)),
            maskSize: simd_float2(Float(textureManager.maskSize.width), Float(textureManager.maskSize.height)),
            maskScaleFactor: textureManager.maskScaleFactor,
            _padding2: 0,
            _padding3: .zero
        )
    }

    // MARK: - Brush Operations

    /// Apply a brush stamp at the given position (point is in UIKit points)
    func applyStamp(at point: CGPoint, radius: Float, isPainting: Bool) {
        guard let computePipeline = brushStampPipelineState,
              let maskTexture = try? textureManager.getMaskTexture(for: currentClassID),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        // Convert touch point (in UIKit points) to screen coordinates (in pixels)
        let screenPoint = convertTouchToScreen(point)
        // Convert screen point to mask coordinates
        let maskPoint = canvasTransform.screenToMask(screenPoint)

        // Brush radius needs to account for:
        // - maskScaleFactor: convert from image pixels to mask pixels
        // - contentScaleFactor: convert from UI points to screen pixels
        let adjustedRadius = radius * canvasTransform.maskScaleFactor * Float(contentScaleFactor)

        var params = BrushParams(
            center: simd_float2(Float(maskPoint.x), Float(maskPoint.y)),
            radius: adjustedRadius,
            paintValue: isPainting ? 1 : 0,
            _padding: (0, 0, 0)
        )

        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setTexture(maskTexture, index: 0)
        computeEncoder.setBytes(&params, length: MemoryLayout<BrushParams>.size, index: 0)

        // Calculate thread groups
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (maskTexture.width + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (maskTexture.height + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()

        // Don't wait - let Metal handle synchronization implicitly
        // The render pass will wait for compute to finish on shared texture
        commandBuffer.commit()
    }

    /// Apply multiple stamps in a single command buffer (optimized for strokes)
    func applyStamps(at points: [CGPoint], radius: Float, isPainting: Bool) {
        guard points.count > 0,
              let computePipeline = brushStampPipelineState,
              let maskTexture = try? textureManager.getMaskTexture(for: currentClassID),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setTexture(maskTexture, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (maskTexture.width + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (maskTexture.height + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )

        // Brush radius adjusted for mask and screen scale
        let adjustedRadius = radius * canvasTransform.maskScaleFactor * Float(contentScaleFactor)

        // Encode all stamps in a single command buffer
        for point in points {
            let screenPoint = convertTouchToScreen(point)
            let maskPoint = canvasTransform.screenToMask(screenPoint)

            var params = BrushParams(
                center: simd_float2(Float(maskPoint.x), Float(maskPoint.y)),
                radius: adjustedRadius,
                paintValue: isPainting ? 1 : 0,
                _padding: (0, 0, 0)
            )

            computeEncoder.setBytes(&params, length: MemoryLayout<BrushParams>.size, index: 0)
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        }

        computeEncoder.endEncoding()
        commandBuffer.commit()
    }

    /// Apply multiple stamps along a stroke path (legacy, calls individual stamps)
    func applyStroke(points: [CGPoint], radius: Float, isPainting: Bool) {
        applyStamps(at: points, radius: radius, isPainting: isPainting)
    }

    /// Clear current mask
    func clearCurrentMask() {
        guard let computePipeline = clearMaskPipelineState,
              let maskTexture = textureManager.maskTextures[currentClassID],
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        var clearValue: UInt32 = 0

        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setTexture(maskTexture, index: 0)
        computeEncoder.setBytes(&clearValue, length: MemoryLayout<UInt32>.size, index: 0)

        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (maskTexture.width + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (maskTexture.height + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )

        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    // MARK: - Image Loading

    /// Load image from URL
    func loadImage(from url: URL) throws {
        try textureManager.loadImage(from: url)
        canvasTransform.maskScaleFactor = textureManager.maskScaleFactor
    }

    /// Load image from CGImage
    func loadImage(_ cgImage: CGImage) throws {
        try textureManager.loadImage(cgImage)
        canvasTransform.maskScaleFactor = textureManager.maskScaleFactor
    }

    // MARK: - Viewport

    /// Update viewport size and scale factor
    func updateViewportSize(_ size: CGSize, scaleFactor: CGFloat = 1.0) {
        viewportSize = size
        contentScaleFactor = scaleFactor
    }

    /// Convert touch point (in points) to screen coordinates (in pixels for shader)
    func convertTouchToScreen(_ point: CGPoint) -> CGPoint {
        return CGPoint(
            x: point.x * contentScaleFactor,
            y: point.y * contentScaleFactor
        )
    }

    // MARK: - Class Management

    /// Set current class for editing
    func setCurrentClass(_ classID: Int) {
        currentClassID = classID
    }

    /// Check if we can add more classes
    var canAddClass: Bool {
        textureManager.maskTextures.count < TextureManager.maxClasses
    }
}

// MARK: - CanvasUniforms Bridge

/// Bridge structure matching ShaderTypes.h
/// Must match memory layout exactly with Metal shader
struct CanvasUniforms {
    var transform: simd_float3x3
    var inverseTransform: simd_float3x3
    var imageAlpha: Float
    var maskAlpha: Float
    var _padding1: simd_float2  // Padding to align maskColor to 16 bytes
    var maskColor: simd_float4
    var canvasSize: simd_float2
    var imageSize: simd_float2
    var maskSize: simd_float2
    var maskScaleFactor: Float
    var _padding2: Float  // Padding for 16-byte struct alignment
    var _padding3: simd_float2
}

/// Bridge structure matching ShaderTypes.h
struct BrushParams {
    var center: simd_float2
    var radius: Float
    var paintValue: UInt8
    var _padding: (UInt8, UInt8, UInt8)
}

// MARK: - Errors

enum RendererError: Error, LocalizedError {
    case libraryCreationFailed
    case functionNotFound(String)
    case pipelineCreationFailed

    var errorDescription: String? {
        switch self {
        case .libraryCreationFailed:
            return "Failed to create Metal library"
        case .functionNotFound(let name):
            return "Metal function not found: \(name)"
        case .pipelineCreationFailed:
            return "Failed to create pipeline state"
        }
    }
}

