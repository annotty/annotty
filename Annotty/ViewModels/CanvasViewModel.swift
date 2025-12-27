import SwiftUI
import Combine
import simd

/// Main state coordinator for the canvas
/// Manages drawing state, image navigation, and coordinates with Metal renderer
class CanvasViewModel: ObservableObject {
    // MARK: - Renderer

    @Published private(set) var renderer: MetalRenderer?

    // MARK: - Gesture Coordinator

    let gestureCoordinator = GestureCoordinator()

    // MARK: - Drawing State

    @Published var brushRadius: Float = 20
    @Published var isPainting: Bool = true
    @Published var isDrawing: Bool = false
    @Published var lastDrawPoint: CGPoint = .zero
    @Published var currentScale: CGFloat = 1.0  // Current zoom level for UI updates

    // MARK: - Display Settings

    @Published var annotationColor: Color = .red {
        didSet {
            updateRendererColor()
        }
    }

    @Published var imageTransparency: Float = 1.0 {
        didSet {
            renderer?.imageAlpha = imageTransparency
        }
    }

    // MARK: - Image Navigation

    @Published private(set) var currentImageIndex: Int = 0
    @Published private(set) var totalImageCount: Int = 0

    // MARK: - Alerts

    @Published var showClassLimitAlert: Bool = false

    // MARK: - Undo Manager

    private let undoManager = AnnotationUndoManager()

    // MARK: - Image Manager

    private let imageManager = ImageItemManager()

    // MARK: - View State

    private var viewSize: CGSize = .zero
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Stroke Tracking

    private var strokePoints: [CGPoint] = []
    private var strokeBbox: CGRect = .null
    private var strokeStartPatch: Data?
    private var currentClassID: Int = 0

    /// Original bbox at stroke start (for proper patch expansion)
    private var originalStrokeBbox: CGRect = .null
    /// Original patch at stroke start (for proper patch expansion)
    private var originalStrokePatch: Data?

    // MARK: - Initialization

    init() {
        setupRenderer()
        setupBindings()
        setupGestureCallbacks()
    }

    private func setupRenderer() {
        renderer = MetalRenderer()
        updateRendererColor()
    }

    private func setupGestureCallbacks() {
        // Drawing callbacks
        gestureCoordinator.onStrokeBegin = { [weak self] point in
            self?.beginStroke(at: point)
        }

        gestureCoordinator.onStrokeContinue = { [weak self] point in
            self?.continueStroke(to: point)
        }

        gestureCoordinator.onStrokeEnd = { [weak self] in
            self?.endStroke()
        }

        gestureCoordinator.onStrokeCancel = { [weak self] in
            self?.cancelStroke()
        }

        // Navigation callbacks
        gestureCoordinator.onPan = { [weak self] translation in
            self?.handlePan(translation: translation)
        }

        gestureCoordinator.onPinch = { [weak self] scale, center in
            self?.handlePinchDelta(scale: scale, at: center)
        }

        gestureCoordinator.onRotation = { [weak self] rotation, center in
            self?.handleRotationDelta(angle: rotation, at: center)
        }

        // Undo/Redo callbacks
        gestureCoordinator.onUndo = { [weak self] in
            self?.undo()
        }

        gestureCoordinator.onRedo = { [weak self] in
            self?.redo()
        }
    }

    private func setupBindings() {
        // Sync image manager with published properties
        imageManager.$currentIndex
            .receive(on: DispatchQueue.main)
            .sink { [weak self] index in
                self?.currentImageIndex = index
            }
            .store(in: &cancellables)

        imageManager.$items
            .receive(on: DispatchQueue.main)
            .sink { [weak self] items in
                self?.totalImageCount = items.count
            }
            .store(in: &cancellables)
    }

    // MARK: - View Size

    func updateViewSize(_ size: CGSize) {
        viewSize = size
        renderer?.updateViewportSize(size)
    }

    // MARK: - Color

    private func updateRendererColor() {
        guard let renderer = renderer else { return }

        // Convert SwiftUI Color to simd_float4
        // Use sRGB color space for reliable conversion
        let uiColor = UIColor(annotationColor)
        var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0

        // Convert to sRGB color space first for reliable component extraction
        if let srgbColor = uiColor.cgColor.converted(
            to: CGColorSpaceCreateDeviceRGB(),
            intent: .defaultIntent,
            options: nil
        ) {
            let components = srgbColor.components ?? [0, 0, 0, 1]
            red = components.count > 0 ? components[0] : 0
            green = components.count > 1 ? components[1] : 0
            blue = components.count > 2 ? components[2] : 0
            alpha = components.count > 3 ? components[3] : 1
        } else {
            // Fallback to direct extraction
            uiColor.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        }

        // Set mask color with semi-transparency for overlay effect
        renderer.maskColor = simd_float4(Float(red), Float(green), Float(blue), 0.5)
        print("🎨 Updated mask color: R=\(red), G=\(green), B=\(blue)")
    }

    // MARK: - Image Loading

    func loadImage(from url: URL) {
        do {
            try renderer?.loadImage(from: url)
        } catch {
            print("Failed to load image: \(error)")
        }
    }

    func loadImagesFromFolder(_ folderURL: URL) {
        let imagesURL = folderURL.appendingPathComponent("images")
        let annotationsURL = folderURL.appendingPathComponent("annotations")

        imageManager.loadImages(from: imagesURL)
        imageManager.checkAnnotations(in: annotationsURL)

        if let firstImage = imageManager.currentItem {
            loadImage(from: firstImage.url)
        }
    }

    // MARK: - Image Navigation

    func previousImage() {
        // Auto-save before switching
        saveCurrentAnnotation()
        imageManager.previous()
        loadCurrentImage()
    }

    func nextImage() {
        // Auto-save before switching
        saveCurrentAnnotation()
        imageManager.next()
        loadCurrentImage()
    }

    private func loadCurrentImage() {
        guard let item = imageManager.currentItem else { return }
        loadImage(from: item.url)

        // Load annotation if exists
        if let annotationURL = item.annotationURL {
            loadAnnotation(from: annotationURL)
        }
    }

    // MARK: - Drawing

    /// Brush preview size for UI (in screen coordinates, reflects actual drawn size)
    var brushPreviewSize: CGFloat {
        CGFloat(brushRadius) * 2 * currentScale
    }

    func beginStroke(at point: CGPoint) {
        isDrawing = true
        lastDrawPoint = point
        strokePoints = [point]

        // Convert touch point to screen pixels, then to mask coordinates
        let screenPoint = renderer?.convertTouchToScreen(point) ?? point
        let maskPoint = renderer?.canvasTransform.screenToMask(screenPoint) ?? point
        // Radius in mask coordinates (must match applyStamp calculation)
        // Add 1 pixel margin to ensure we capture all affected pixels
        let maskScaleFactor = renderer?.canvasTransform.maskScaleFactor ?? 2.0
        let scaleFactor = renderer?.contentScaleFactor ?? 1.0
        let radius = CGFloat(brushRadius * maskScaleFactor * Float(scaleFactor)) + 1.0
        strokeBbox = CGRect(
            x: maskPoint.x - radius,
            y: maskPoint.y - radius,
            width: radius * 2,
            height: radius * 2
        )

        // Ensure mask texture exists before capturing patch
        // (texture is created lazily, so first stroke needs this)
        _ = try? renderer?.textureManager.getMaskTexture(for: currentClassID)

        // Capture patch before stroke
        strokeStartPatch = renderer?.textureManager.readMaskRegion(
            from: currentClassID,
            bbox: strokeBbox
        )

        // Store original bbox and patch for proper expansion
        originalStrokeBbox = strokeBbox
        originalStrokePatch = strokeStartPatch

        // Apply first stamp
        renderer?.applyStamp(at: point, radius: brushRadius, isPainting: isPainting)
    }

    func continueStroke(to point: CGPoint) {
        guard isDrawing else { return }

        lastDrawPoint = point

        // Interpolate points for smooth continuous strokes
        if let lastPoint = strokePoints.last {
            let distance = hypot(point.x - lastPoint.x, point.y - lastPoint.y)
            // Use smaller step interval (30% of brush radius) for overlapping stamps
            // This ensures continuous lines even when pen moves quickly
            let stepInterval = max(1.0, CGFloat(brushRadius) * 0.3)
            let steps = max(1, Int(ceil(distance / stepInterval)))

            // Collect all interpolated points for batch processing
            var interpolatedPoints: [CGPoint] = []

            for i in 1...steps {
                let t = CGFloat(i) / CGFloat(steps)
                let interpolatedPoint = CGPoint(
                    x: lastPoint.x + (point.x - lastPoint.x) * t,
                    y: lastPoint.y + (point.y - lastPoint.y) * t
                )

                // Expand bbox (convert to screen pixels first)
                let screenPoint = renderer?.convertTouchToScreen(interpolatedPoint) ?? interpolatedPoint
                let maskPoint = renderer?.canvasTransform.screenToMask(screenPoint) ?? interpolatedPoint
                // Radius in mask coordinates (must match applyStamp calculation)
                // Add 1 pixel margin to ensure we capture all affected pixels
                let maskScaleFactor = renderer?.canvasTransform.maskScaleFactor ?? 2.0
                let scaleFactor = renderer?.contentScaleFactor ?? 1.0
                let radius = CGFloat(brushRadius * maskScaleFactor * Float(scaleFactor)) + 1.0
                let stampRect = CGRect(
                    x: maskPoint.x - radius,
                    y: maskPoint.y - radius,
                    width: radius * 2,
                    height: radius * 2
                )

                if strokeBbox.isNull {
                    strokeBbox = stampRect
                } else {
                    let newBbox = strokeBbox.union(stampRect)
                    if newBbox != strokeBbox {
                        // Expand captured patch
                        expandStrokePatch(to: newBbox)
                        strokeBbox = newBbox
                    }
                }

                interpolatedPoints.append(interpolatedPoint)
            }

            // Apply all stamps in a single GPU batch
            renderer?.applyStamps(at: interpolatedPoints, radius: brushRadius, isPainting: isPainting)

            // Only store the final point (not all interpolated points)
            strokePoints.append(point)
        }
    }

    func endStroke() {
        guard isDrawing else { return }

        isDrawing = false

        // Create undo action
        if let patch = strokeStartPatch, !strokeBbox.isNull {
            let action = UndoAction(
                classID: currentClassID,
                bbox: strokeBbox,
                previousPatch: patch
            )
            undoManager.pushUndo(action)
        }

        // Reset stroke state
        strokePoints.removeAll()
        strokeBbox = .null
        strokeStartPatch = nil
        originalStrokeBbox = .null
        originalStrokePatch = nil

        // Schedule auto-save
        scheduleAutoSave()
    }

    private func expandStrokePatch(to newBbox: CGRect) {
        // We need to properly composite: the previous correct patch + new regions
        // The texture now has painted data in the previous region, so we can't just re-read
        guard let previousPatch = originalStrokePatch,
              !originalStrokeBbox.isNull,
              let textureManager = renderer?.textureManager,
              let texture = textureManager.maskTextures[currentClassID] else { return }

        // Calculate integer bounds for new bbox
        // Use floor for min and ceil for max to ensure we capture all affected pixels
        let newMinX = max(0, Int(floor(newBbox.minX)))
        let newMinY = max(0, Int(floor(newBbox.minY)))
        let newMaxX = min(texture.width, Int(ceil(newBbox.maxX)))
        let newMaxY = min(texture.height, Int(ceil(newBbox.maxY)))
        let newWidth = newMaxX - newMinX
        let newHeight = newMaxY - newMinY

        guard newWidth > 0 && newHeight > 0 else { return }

        // Read the new bbox from texture (has painted data in previous region, original data in new regions)
        var newPatchData = [UInt8](repeating: 0, count: newWidth * newHeight)
        texture.getBytes(
            &newPatchData,
            bytesPerRow: newWidth,
            from: MTLRegion(
                origin: MTLOrigin(x: newMinX, y: newMinY, z: 0),
                size: MTLSize(width: newWidth, height: newHeight, depth: 1)
            ),
            mipmapLevel: 0
        )

        // Calculate integer bounds for previous bbox
        let prevMinX = max(0, Int(originalStrokeBbox.minX))
        let prevMinY = max(0, Int(originalStrokeBbox.minY))
        let prevMaxX = min(texture.width, Int(originalStrokeBbox.maxX))
        let prevMaxY = min(texture.height, Int(originalStrokeBbox.maxY))
        let prevWidth = prevMaxX - prevMinX
        let prevHeight = prevMaxY - prevMinY

        // Calculate offset of previous region within new region
        let offsetX = prevMinX - newMinX
        let offsetY = prevMinY - newMinY

        // Copy previous patch data into the new patch at the correct offset
        // This restores the "before stroke" state for the previously covered region
        previousPatch.withUnsafeBytes { prevBuffer in
            guard let prevPtr = prevBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }

            for y in 0..<prevHeight {
                for x in 0..<prevWidth {
                    let prevIndex = y * prevWidth + x
                    let newIndex = (y + offsetY) * newWidth + (x + offsetX)
                    if prevIndex < previousPatch.count && newIndex < newPatchData.count {
                        newPatchData[newIndex] = prevPtr[prevIndex]
                    }
                }
            }
        }

        let compositedPatch = Data(newPatchData)
        strokeStartPatch = compositedPatch

        // Update original tracking to the composited result
        // So next expansion uses the correct "before" data for the entire area so far
        originalStrokeBbox = CGRect(x: newMinX, y: newMinY, width: newWidth, height: newHeight)
        originalStrokePatch = compositedPatch
    }

    // MARK: - Stroke Cancellation

    /// Cancel current stroke and restore previous state (called when 2+ fingers detected)
    func cancelStroke() {
        guard isDrawing else { return }

        isDrawing = false

        // Restore the mask to pre-stroke state
        if let patch = strokeStartPatch, !strokeBbox.isNull {
            renderer?.textureManager.writeMaskRegion(
                to: currentClassID,
                bbox: strokeBbox,
                data: patch
            )
        }

        // Reset stroke state
        strokePoints.removeAll()
        strokeBbox = .null
        strokeStartPatch = nil
        originalStrokeBbox = .null
        originalStrokePatch = nil

        print("🚫 Stroke cancelled - restored previous state")
    }

    // MARK: - Navigation Gesture Handling (UIKit)

    /// Handle pan gesture (2-finger drag)
    func handlePan(translation: CGPoint) {
        // Convert translation from points to pixels (same as touch coordinates)
        let scaledTranslation = renderer?.convertTouchToScreen(translation) ?? translation
        renderer?.canvasTransform.applyPan(delta: scaledTranslation)
    }

    /// Handle pinch gesture with delta scale and center point
    func handlePinchDelta(scale: CGFloat, at center: CGPoint) {
        let screenCenter = renderer?.convertTouchToScreen(center) ?? center
        renderer?.canvasTransform.applyPinch(scaleFactor: scale, center: screenCenter)
        // Update published scale for UI
        currentScale = renderer?.canvasTransform.scale ?? 1.0
    }

    /// Handle rotation gesture with delta angle and center point
    func handleRotationDelta(angle: CGFloat, at center: CGPoint) {
        let screenCenter = renderer?.convertTouchToScreen(center) ?? center
        renderer?.canvasTransform.applyRotation(angleDelta: angle, center: screenCenter)
    }

    // MARK: - Undo/Redo

    func undo() {
        guard let action = undoManager.undo() else { return }

        // Restore previous patch
        renderer?.textureManager.writeMaskRegion(
            to: action.classID,
            bbox: action.bbox,
            data: action.previousPatch
        )
    }

    func redo() {
        guard let action = undoManager.redo() else { return }

        // For redo, we need to re-apply the stroke
        // This is simplified - full implementation would store the new patch too
        renderer?.textureManager.writeMaskRegion(
            to: action.classID,
            bbox: action.bbox,
            data: action.previousPatch
        )
    }

    // MARK: - Auto Save

    private var autoSaveWorkItem: DispatchWorkItem?

    private func scheduleAutoSave() {
        autoSaveWorkItem?.cancel()

        autoSaveWorkItem = DispatchWorkItem { [weak self] in
            self?.saveCurrentAnnotation()
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5, execute: autoSaveWorkItem!)
    }

    private func saveCurrentAnnotation() {
        // Implementation will be added in Phase 4
        print("Auto-save triggered")
    }

    private func loadAnnotation(from url: URL) {
        // Implementation will be added in Phase 4
        print("Loading annotation from: \(url)")
    }

    // MARK: - Class Management

    func addNewClass() -> Bool {
        guard renderer?.canAddClass == true else {
            showClassLimitAlert = true
            return false
        }

        // Implementation for adding new class
        return true
    }
}

