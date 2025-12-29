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
    @Published var isFillMode: Bool = false {
        didSet {
            gestureCoordinator.isFillMode = isFillMode
        }
    }
    @Published var lastDrawPoint: CGPoint = .zero
    @Published var currentScale: CGFloat = 1.0  // Current zoom level for UI updates

    // MARK: - Class Management

    /// Preset colors mapped to class IDs (index+1 = classID)
    /// These must match the colors in MetalRenderer.classColors exactly
    /// Class 1=red, 2=orange, 3=yellow, 4=green, 5=cyan, 6=blue, 7=purple, 8=pink
    static let classColors: [Color] = [
        Color(red: 1, green: 0, blue: 0),        // 1: red
        Color(red: 1, green: 0.5, blue: 0),      // 2: orange
        Color(red: 1, green: 1, blue: 0),        // 3: yellow
        Color(red: 0, green: 1, blue: 0),        // 4: green
        Color(red: 0, green: 1, blue: 1),        // 5: cyan
        Color(red: 0, green: 0, blue: 1),        // 6: blue
        Color(red: 0.5, green: 0, blue: 1),      // 7: purple
        Color(red: 1, green: 0.4, blue: 0.7)     // 8: pink
    ]

    /// Current active class ID (1-8, 0 = eraser/background)
    @Published private(set) var currentClassID: Int = 1

    /// Custom class names (index 0-7 = class 1-8)
    /// Empty string means unnamed class
    @Published var classNames: [String] = Array(repeating: "", count: 8) {
        didSet {
            saveClassNames()
        }
    }

    /// UserDefaults key for class names
    private static let classNamesKey = "annotty.classNames"

    // MARK: - Display Settings

    @Published var annotationColor: Color = Color(red: 1, green: 0, blue: 0) {
        didSet {
            // Update currentClassID based on selected color (index + 1)
            if let index = Self.classColors.firstIndex(of: annotationColor) {
                currentClassID = index + 1
                renderer?.setCurrentClass(currentClassID)
            }
        }
    }

    /// Image contrast (0.0 - 2.0, 1.0 = normal, 0% - 200%)
    @Published var imageContrast: Float = 1.0 {
        didSet {
            renderer?.imageContrast = imageContrast
        }
    }

    /// Image brightness (-1.0 to 1.0, 0.0 = normal)
    @Published var imageBrightness: Float = 0.0 {
        didSet {
            renderer?.imageBrightness = imageBrightness
        }
    }

    /// Mask fill opacity (0.0 - 1.0, affects fill only, edges stay opaque)
    @Published var maskFillAlpha: Float = 0.5 {
        didSet {
            renderer?.maskFillAlpha = maskFillAlpha
        }
    }

    // MARK: - Image Navigation

    @Published private(set) var currentImageIndex: Int = 0
    @Published private(set) var totalImageCount: Int = 0

    // MARK: - Saving State

    @Published private(set) var isSaving: Bool = false

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

    /// Original bbox at stroke start (for proper patch expansion)
    private var originalStrokeBbox: CGRect = .null
    /// Original patch at stroke start (for proper patch expansion)
    private var originalStrokePatch: Data?

    // MARK: - Initialization

    init() {
        print("🚀 CanvasViewModel init")
        setupRenderer()
        setupBindings()
        setupGestureCallbacks()
        loadClassNames()
        initializeProjectFolder()
    }

    /// Initialize the project folder structure and load existing images
    private func initializeProjectFolder() {
        guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("[Project] Failed to get Documents directory")
            return
        }

        // Use Documents directly (no subfolder)
        print("[Project] Root: \(documentsURL.path)")

        do {
            try ProjectFileService.shared.initializeProject(at: documentsURL)
            print("[Project] Folder structure: images/, annotations/, labels/")
            reloadImagesFromProject()
        } catch {
            print("[Project] Failed to initialize: \(error)")
        }
    }

    /// Open a different project folder
    func openProject(at folderURL: URL) {
        let didStartAccessing = folderURL.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }

        do {
            try ProjectFileService.shared.initializeProject(at: folderURL)
            print("[Project] Opened project: \(folderURL.lastPathComponent)")
            reloadImagesFromProject()
        } catch {
            print("[Project] Failed to open: \(error)")
        }
    }

    /// Reload images from the project's images folder
    func reloadImagesFromProject() {
        let imageURLs = ProjectFileService.shared.getImageURLs()
        print("[Project] Found \(imageURLs.count) images")

        imageManager.setImages(imageURLs)

        // Load first image with its annotation (if exists)
        if imageManager.currentItem != nil {
            loadCurrentImage()
        }
    }

    /// Import a single image to the project
    func importImage(from sourceURL: URL) {
        do {
            let destinationURL = try ProjectFileService.shared.copyImageToProject(sourceURL)
            print("[Import] Copied: \(destinationURL.lastPathComponent)")
            reloadImagesFromProject()

            if let index = imageManager.items.firstIndex(where: { $0.url == destinationURL }) {
                imageManager.goTo(index: index)
                loadCurrentImage()
            }
        } catch {
            print("[Import] Failed: \(error)")
        }
    }

    /// Import all images from a folder
    func importImagesFromFolder(_ folderURL: URL) {
        let fileManager = FileManager.default
        let supportedExtensions = ["png", "jpg", "jpeg"]

        let didStartAccessing = folderURL.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }

        do {
            let contents = try fileManager.contentsOfDirectory(
                at: folderURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )

            let imageFiles = contents.filter { supportedExtensions.contains($0.pathExtension.lowercased()) }
            print("[Import] Found \(imageFiles.count) images in folder")

            for imageURL in imageFiles {
                _ = try? ProjectFileService.shared.copyImageToProject(imageURL)
            }

            reloadImagesFromProject()
        } catch {
            print("[Import] Folder read failed: \(error)")
        }
    }

    private func setupRenderer() {
        renderer = MetalRenderer()
        // Set initial class ID (1 = red)
        renderer?.setCurrentClass(currentClassID)
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

        // Fill callback
        gestureCoordinator.onFillTap = { [weak self] point in
            self?.floodFill(at: point)
            // Auto-disable fill mode after fill
            self?.isFillMode = false
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
        guard !isSaving else { return }
        saveAndNavigate { [weak self] in
            self?.imageManager.previous()
        }
    }

    func nextImage() {
        guard !isSaving else { return }
        saveAndNavigate { [weak self] in
            self?.imageManager.next()
        }
    }

    func goToImage(index: Int) {
        guard !isSaving else { return }
        guard index != currentImageIndex else { return }
        saveAndNavigate { [weak self] in
            self?.imageManager.goTo(index: index)
        }
    }

    /// Save current annotation asynchronously, then navigate
    private func saveAndNavigate(navigation: @escaping () -> Void) {
        guard let imageItem = imageManager.currentItem,
              let textureManager = renderer?.textureManager else {
            navigation()
            loadCurrentImage()
            return
        }

        // Read mask data from GPU (must be on main thread)
        guard let maskData = textureManager.readMask() else {
            navigation()
            loadCurrentImage()
            return
        }

        // Check if mask has any data
        let hasData = maskData.contains { $0 != 0 }
        guard hasData else {
            navigation()
            loadCurrentImage()
            return
        }

        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)
        let color = annotationColor
        let imageURL = imageItem.url

        // Show saving indicator
        isSaving = true

        // PNG generation and file write on background thread
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let pngData = self?.createColoredPNG(
                from: maskData,
                width: maskWidth,
                height: maskHeight,
                color: color
            ) else {
                DispatchQueue.main.async {
                    self?.isSaving = false
                    navigation()
                    self?.loadCurrentImage()
                }
                return
            }

            // Save to file
            do {
                try ProjectFileService.shared.saveAnnotation(pngData, for: imageURL)
                print("[Save] Saved annotation (async)")
            } catch {
                print("[Save] Failed: \(error)")
            }

            // Back to main thread for navigation
            DispatchQueue.main.async {
                self?.isSaving = false
                navigation()
                self?.loadCurrentImage()
            }
        }
    }

    private func loadCurrentImage() {
        guard let item = imageManager.currentItem else { return }
        loadImage(from: item.url)

        // Always check for annotation file dynamically (not just cached URL)
        // This ensures newly saved annotations are detected
        if let annotationURL = ProjectFileService.shared.getAnnotationURL(for: item.url),
           FileManager.default.fileExists(atPath: annotationURL.path) {
            loadAnnotation(from: annotationURL)
        }
    }

    // MARK: - Drawing

    /// Brush preview size for UI (in screen coordinates, reflects actual drawn size)
    var brushPreviewSize: CGFloat {
        CGFloat(brushRadius) * 2 * currentScale
    }

    /// Counter for throttling bbox expansion checks
    private var strokePointCounter: Int = 0

    func beginStroke(at point: CGPoint) {
        isDrawing = true
        lastDrawPoint = point
        strokePoints = [point]
        strokePointCounter = 0

        // Convert touch point to screen pixels, then to mask coordinates
        let screenPoint = renderer?.convertTouchToScreen(point) ?? point
        let maskPoint = renderer?.canvasTransform.screenToMask(screenPoint) ?? point

        // Guard against invalid coordinates (NaN or infinite from degenerate transforms)
        guard maskPoint.x.isFinite && maskPoint.y.isFinite else {
            print("[Stroke] Invalid maskPoint at begin: \(maskPoint), skipping stroke setup")
            isDrawing = false
            return
        }

        // Ensure mask texture exists before capturing patch
        _ = try? renderer?.textureManager.getMaskTexture()

        // Use very large initial bbox to minimize expensive expandStrokePatch calls
        // For most strokes, this single upfront allocation is much faster than repeated expansions
        if let textureManager = renderer?.textureManager {
            let maxWidth = textureManager.maskSize.width
            let maxHeight = textureManager.maskSize.height

            // Use minimum 2000x2000 or full texture (whichever is smaller)
            // This covers most strokes without needing expansion
            let initialSize: CGFloat = min(2000, min(maxWidth, maxHeight))
            let halfSize = initialSize / 2

            strokeBbox = CGRect(
                x: maskPoint.x - halfSize,
                y: maskPoint.y - halfSize,
                width: initialSize,
                height: initialSize
            )

            // Clamp to texture bounds
            strokeBbox = strokeBbox.intersection(CGRect(x: 0, y: 0, width: maxWidth, height: maxHeight))
        }

        // Capture patch before stroke (larger area upfront)
        strokeStartPatch = renderer?.textureManager.readMaskRegion(bbox: strokeBbox)

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

            // Track if we need to expand bbox (throttled check)
            var needsBboxExpansion = false
            var expandedBbox = strokeBbox

            for i in 1...steps {
                let t = CGFloat(i) / CGFloat(steps)
                let interpolatedPoint = CGPoint(
                    x: lastPoint.x + (point.x - lastPoint.x) * t,
                    y: lastPoint.y + (point.y - lastPoint.y) * t
                )

                interpolatedPoints.append(interpolatedPoint)
                strokePointCounter += 1

                // Only check bbox expansion every 20 points to reduce overhead
                // The large initial bbox should cover most cases
                if strokePointCounter % 20 == 0 {
                    let screenPoint = renderer?.convertTouchToScreen(interpolatedPoint) ?? interpolatedPoint
                    let maskPoint = renderer?.canvasTransform.screenToMask(screenPoint) ?? interpolatedPoint

                    // Skip if coordinates are invalid (NaN or infinite)
                    guard maskPoint.x.isFinite && maskPoint.y.isFinite else { continue }

                    let maskScaleFactor = renderer?.canvasTransform.maskScaleFactor ?? 2.0
                    // Radius in mask coordinates (UI "1" = 1 original image pixel)
                    let radius = CGFloat(brushRadius * maskScaleFactor) + 1.0
                    let stampRect = CGRect(
                        x: maskPoint.x - radius,
                        y: maskPoint.y - radius,
                        width: radius * 2,
                        height: radius * 2
                    )

                    if !expandedBbox.isNull {
                        let newBbox = expandedBbox.union(stampRect)
                        if newBbox != expandedBbox {
                            expandedBbox = newBbox
                            needsBboxExpansion = true
                        }
                    }
                }
            }

            // Apply all stamps in a single GPU batch (this is fast)
            renderer?.applyStamps(at: interpolatedPoints, radius: brushRadius, isPainting: isPainting)

            // Expand bbox only once per continueStroke call if needed (expensive operation)
            if needsBboxExpansion && expandedBbox != strokeBbox {
                expandStrokePatch(to: expandedBbox)
                strokeBbox = expandedBbox
            }

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

        // Note: Auto-save removed for performance
        // Saving happens on image navigation or app background
    }

    private func expandStrokePatch(to newBbox: CGRect) {
        // We need to properly composite: the previous correct patch + new regions
        // The texture now has painted data in the previous region, so we can't just re-read
        guard let previousPatch = originalStrokePatch,
              !originalStrokeBbox.isNull,
              let textureManager = renderer?.textureManager,
              let texture = textureManager.maskTexture else { return }

        // Guard against invalid bbox values (NaN or infinite)
        guard newBbox.minX.isFinite && newBbox.minY.isFinite &&
              newBbox.maxX.isFinite && newBbox.maxY.isFinite else {
            print("[Stroke] Invalid bbox values, skipping expansion")
            return
        }

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
        // Guard against invalid values
        guard originalStrokeBbox.minX.isFinite && originalStrokeBbox.minY.isFinite &&
              originalStrokeBbox.maxX.isFinite && originalStrokeBbox.maxY.isFinite else {
            print("[Stroke] Invalid originalStrokeBbox values, skipping expansion")
            return
        }
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
        // Use row-by-row memcpy for performance (much faster than pixel-by-pixel)
        previousPatch.withUnsafeBytes { prevBuffer in
            guard let prevPtr = prevBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }

            for y in 0..<prevHeight {
                let prevRowStart = y * prevWidth
                let newRowStart = (y + offsetY) * newWidth + offsetX

                // Bounds check for safety
                guard prevRowStart + prevWidth <= previousPatch.count,
                      newRowStart + prevWidth <= newPatchData.count else { continue }

                // Copy entire row at once using memcpy
                memcpy(&newPatchData[newRowStart], prevPtr.advanced(by: prevRowStart), prevWidth)
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
            renderer?.textureManager.writeMaskRegion(bbox: strokeBbox, data: patch)
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

    /// Reset view to default (pan/zoom/rotation)
    func resetView() {
        renderer?.canvasTransform.reset()
        currentScale = 1.0
        print("[View] Reset to default")
    }

    // MARK: - Undo/Redo

    func undo() {
        guard let action = undoManager.undo() else { return }

        // Restore previous patch
        renderer?.textureManager.writeMaskRegion(bbox: action.bbox, data: action.previousPatch)
    }

    func redo() {
        guard let action = undoManager.redo() else { return }

        // For redo, we need to re-apply the stroke
        // This is simplified - full implementation would store the new patch too
        renderer?.textureManager.writeMaskRegion(bbox: action.bbox, data: action.previousPatch)
    }

    /// Clear all annotations for current image (undoable with 2-finger tap)
    func clearAllAnnotations() {
        guard let textureManager = renderer?.textureManager else {
            print("[Clear] No texture manager")
            return
        }

        // Read current mask data for undo
        guard let currentMaskData = textureManager.readMask() else {
            print("[Clear] No mask data to backup")
            return
        }

        // Check if there's anything to clear
        let hasData = currentMaskData.contains { $0 != 0 }
        guard hasData else {
            print("[Clear] Mask is already empty")
            return
        }

        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)

        // Create undo action with full mask
        let bbox = CGRect(x: 0, y: 0, width: CGFloat(maskWidth), height: CGFloat(maskHeight))
        let previousPatch = Data(currentMaskData)

        let action = UndoAction(
            classID: 0,  // 0 indicates clear action
            bbox: bbox,
            previousPatch: previousPatch
        )
        undoManager.pushUndo(action)

        // Clear the mask
        renderer?.clearMask()
        print("[Clear] Cleared all annotations (undoable)")
    }

    // MARK: - Save

    /// Save current annotation (called on image navigation and app background)
    func saveBeforeBackground() {
        saveCurrentAnnotation()
        print("[Save] Background save completed")
    }

    private func saveCurrentAnnotation() {
        guard let imageItem = imageManager.currentItem,
              let textureManager = renderer?.textureManager else {
            print("[Save] No image or texture manager")
            return
        }

        // Read mask data from GPU
        guard let maskData = textureManager.readMask() else {
            print("[Save] No mask data to save")
            return
        }

        // Check if mask has any data (skip save if empty)
        let hasData = maskData.contains { $0 != 0 }
        guard hasData else {
            print("[Save] Mask is empty, skipping save")
            return
        }

        // Convert binary mask to color PNG
        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)

        guard let pngData = createColoredPNG(
            from: maskData,
            width: maskWidth,
            height: maskHeight,
            color: annotationColor
        ) else {
            print("[Save] Failed to create PNG")
            return
        }

        // Save to annotations folder
        do {
            try ProjectFileService.shared.saveAnnotation(pngData, for: imageItem.url)
            print("[Save] Saved annotation for \(imageItem.baseName)")
        } catch {
            print("[Save] Failed: \(error)")
        }
    }

    private func loadAnnotation(from url: URL) {
        guard let textureManager = renderer?.textureManager,
              let data = try? Data(contentsOf: url),
              let image = UIImage(data: data),
              let cgImage = image.cgImage else {
            print("[Load] Failed to load annotation image")
            return
        }

        let width = cgImage.width
        let height = cgImage.height

        // Read pixel data from PNG
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            print("[Load] Failed to create context")
            return
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert colored PNG to class ID mask
        // Each pixel color is snapped to nearest class color
        var maskData = [UInt8](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            let offset = i * bytesPerPixel
            let r = pixelData[offset]
            let g = pixelData[offset + 1]
            let b = pixelData[offset + 2]
            let a = pixelData[offset + 3]

            // If transparent or white, it's background (classID = 0)
            let isBackground = a < 128 || (r > 250 && g > 250 && b > 250)
            if isBackground {
                maskData[i] = 0
            } else {
                // Find nearest class color
                maskData[i] = UInt8(findNearestClassID(r: r, g: g, b: b))
            }
        }

        // Check if we need to resize (annotation size might differ from mask size)
        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)

        if width == maskWidth && height == maskHeight {
            // Same size, upload directly
            do {
                try textureManager.uploadMask(maskData)
                print("[Load] Loaded annotation (\(width)x\(height))")
            } catch {
                print("[Load] Upload failed: \(error)")
            }
        } else {
            // Need to resize - use nearest neighbor for binary mask
            let resizedMask = resizeMask(
                maskData,
                fromWidth: width, fromHeight: height,
                toWidth: maskWidth, toHeight: maskHeight
            )
            do {
                try textureManager.uploadMask(resizedMask)
                print("[Load] Loaded and resized annotation (\(width)x\(height) → \(maskWidth)x\(maskHeight))")
            } catch {
                print("[Load] Upload failed: \(error)")
            }
        }
    }

    /// Class colors as RGB tuples (index+1 = classID)
    private static let classRGBColors: [(UInt8, UInt8, UInt8)] = [
        (255, 0, 0),       // 1: red
        (255, 128, 0),     // 2: orange
        (255, 255, 0),     // 3: yellow
        (0, 255, 0),       // 4: green
        (0, 255, 255),     // 5: cyan
        (0, 0, 255),       // 6: blue
        (128, 0, 255),     // 7: purple
        (255, 102, 178)    // 8: pink
    ]

    /// Create a multi-class colored PNG from mask data
    /// Mask values: 0=background(white), 1-8=class colors
    private func createColoredPNG(from maskData: [UInt8], width: Int, height: Int, color: Color) -> Data? {
        // Create RGBA pixel data
        let bytesPerPixel = 4
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

        for i in 0..<(width * height) {
            let offset = i * bytesPerPixel
            let classID = Int(maskData[i])

            if classID > 0 && classID <= Self.classRGBColors.count {
                // Masked pixel: use corresponding class color
                let (r, g, b) = Self.classRGBColors[classID - 1]
                pixelData[offset] = r
                pixelData[offset + 1] = g
                pixelData[offset + 2] = b
                pixelData[offset + 3] = 255
            } else {
                // Background: white with full alpha
                pixelData[offset] = 255
                pixelData[offset + 1] = 255
                pixelData[offset + 2] = 255
                pixelData[offset + 3] = 255
            }
        }

        // Create CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * bytesPerPixel,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
              let cgImage = context.makeImage() else {
            return nil
        }

        // Convert to PNG data
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.pngData()
    }

    /// Find the nearest class ID for a given RGB color
    /// Returns 1-8 for matching class colors, or 1 as fallback
    private func findNearestClassID(r: UInt8, g: UInt8, b: UInt8) -> Int {
        var minDistance = Int.max
        var nearestClassID = 1

        for (index, (cr, cg, cb)) in Self.classRGBColors.enumerated() {
            let distance = abs(Int(r) - Int(cr)) + abs(Int(g) - Int(cg)) + abs(Int(b) - Int(cb))
            if distance < minDistance {
                minDistance = distance
                nearestClassID = index + 1  // classID is 1-indexed
            }
        }

        return nearestClassID
    }

    /// Resize binary mask using nearest neighbor interpolation
    private func resizeMask(_ source: [UInt8], fromWidth: Int, fromHeight: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: toWidth * toHeight)

        let scaleX = Float(fromWidth) / Float(toWidth)
        let scaleY = Float(fromHeight) / Float(toHeight)

        for y in 0..<toHeight {
            for x in 0..<toWidth {
                let srcX = Int(Float(x) * scaleX)
                let srcY = Int(Float(y) * scaleY)
                let srcIndex = srcY * fromWidth + srcX
                let dstIndex = y * toWidth + x

                if srcIndex < source.count {
                    result[dstIndex] = source[srcIndex]
                }
            }
        }

        return result
    }

    // MARK: - Flood Fill

    /// Perform flood fill at the given touch location
    func floodFill(at touchPoint: CGPoint) {
        guard let renderer = renderer else {
            print("[FloodFill] No renderer or texture manager")
            return
        }
        let textureManager = renderer.textureManager

        // Convert touch point to screen pixels, then to mask coordinates
        let screenPoint = renderer.convertTouchToScreen(touchPoint)
        let maskPoint = renderer.canvasTransform.screenToMask(screenPoint)

        // Guard against invalid coordinates (NaN or infinite from degenerate transforms)
        guard maskPoint.x.isFinite && maskPoint.y.isFinite else {
            print("[FloodFill] Invalid maskPoint: \(maskPoint), skipping fill")
            return
        }

        let maskWidth = Int(textureManager.maskSize.width)
        let maskHeight = Int(textureManager.maskSize.height)

        // Ensure the point is within bounds
        let startX = Int(maskPoint.x)
        let startY = Int(maskPoint.y)

        guard startX >= 0 && startX < maskWidth && startY >= 0 && startY < maskHeight else {
            print("[FloodFill] Point out of bounds: (\(startX), \(startY))")
            return
        }

        // Get current mask data
        guard var maskData = textureManager.readMask() else {
            // Create new mask if it doesn't exist
            _ = try? textureManager.getMaskTexture()
            guard var newMaskData = textureManager.readMask() else {
                print("[FloodFill] Failed to get mask data")
                return
            }
            performFloodFill(on: &newMaskData, width: maskWidth, height: maskHeight, startX: startX, startY: startY)
            return
        }

        // Check what's at the starting point
        let startIndex = startY * maskWidth + startX
        let targetValue = maskData[startIndex]

        // If tapping on the same class as currently selected, do nothing
        if targetValue == UInt8(currentClassID) {
            print("[FloodFill] Same class selected, skipping")
            return
        }

        // Capture undo patch before flood fill
        let bbox = CGRect(x: 0, y: 0, width: CGFloat(maskWidth), height: CGFloat(maskHeight))
        let previousPatch = Data(maskData)

        // Perform flood fill
        performFloodFill(on: &maskData, width: maskWidth, height: maskHeight, startX: startX, startY: startY)

        // Upload the modified mask
        do {
            try textureManager.uploadMask(maskData)
            print("[FloodFill] Fill completed")

            // Create undo action
            let action = UndoAction(
                classID: currentClassID,
                bbox: bbox,
                previousPatch: previousPatch
            )
            undoManager.pushUndo(action)
        } catch {
            print("[FloodFill] Failed to upload mask: \(error)")
        }
    }

    /// Flood fill algorithm using BFS (Breadth-First Search)
    /// Replaces contiguous region of targetValue with currentClassID (1-8)
    /// Works for both empty regions (0) and existing class regions (1-8)
    private func performFloodFill(on maskData: inout [UInt8], width: Int, height: Int, startX: Int, startY: Int) {
        let startIndex = startY * width + startX
        let targetValue = maskData[startIndex]
        let fillValue = UInt8(currentClassID)

        // If target is same as fill value, nothing to do
        if targetValue == fillValue {
            return
        }

        // BFS queue
        var queue: [(Int, Int)] = [(startX, startY)]
        var visited = Set<Int>()
        visited.insert(startIndex)

        // Direction offsets: up, down, left, right
        let dx = [0, 0, -1, 1]
        let dy = [-1, 1, 0, 0]

        var fillCount = 0

        while !queue.isEmpty {
            let (x, y) = queue.removeFirst()
            let index = y * width + x

            // Fill this pixel with current class ID
            maskData[index] = fillValue
            fillCount += 1

            // Check all 4 neighbors
            for i in 0..<4 {
                let nx = x + dx[i]
                let ny = y + dy[i]

                // Bounds check
                guard nx >= 0 && nx < width && ny >= 0 && ny < height else { continue }

                let neighborIndex = ny * width + nx

                // Skip if already visited
                guard !visited.contains(neighborIndex) else { continue }

                // Fill if same value as target (contiguous region)
                if maskData[neighborIndex] == targetValue {
                    visited.insert(neighborIndex)
                    queue.append((nx, ny))
                }
            }
        }

        let action = targetValue == 0 ? "Filled" : "Replaced class \(targetValue) with"
        print("[FloodFill] \(action) \(fillCount) pixels → class \(currentClassID)")
    }

    // MARK: - Class Names Persistence

    /// Save class names to UserDefaults
    private func saveClassNames() {
        UserDefaults.standard.set(classNames, forKey: Self.classNamesKey)
        print("[ClassNames] Saved: \(classNames.filter { !$0.isEmpty })")
    }

    /// Load class names from UserDefaults
    private func loadClassNames() {
        if let saved = UserDefaults.standard.stringArray(forKey: Self.classNamesKey) {
            // Ensure we always have exactly 8 elements
            if saved.count == 8 {
                classNames = saved
            } else {
                // Pad or truncate to 8 elements
                var adjusted = saved
                while adjusted.count < 8 { adjusted.append("") }
                classNames = Array(adjusted.prefix(8))
            }
            print("[ClassNames] Loaded: \(classNames.filter { !$0.isEmpty })")
        }
    }

    /// Clear all class names
    func clearClassNames() {
        classNames = Array(repeating: "", count: 8)
        print("[ClassNames] Cleared")
    }

}

