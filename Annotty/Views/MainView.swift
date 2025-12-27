import SwiftUI

/// Main app view with the complete UI layout
/// Layout: Left panel | Center canvas | Right panel
/// Top bar spans the full width
struct MainView: View {
    @StateObject private var viewModel = CanvasViewModel()
    @State private var showingExportSheet = false
    @State private var showingImagePicker = false
    @State private var showingClassLimitAlert = false

    var body: some View {
        VStack(spacing: 0) {
            // Top bar
            TopBarView(
                currentIndex: viewModel.currentImageIndex,
                totalCount: viewModel.totalImageCount,
                isSaving: viewModel.isSaving,
                onPrevious: { viewModel.previousImage() },
                onNext: { viewModel.nextImage() },
                onExport: { showingExportSheet = true },
                onLoad: { showingImagePicker = true },
                onReload: { viewModel.reloadImagesFromProject() }
            )

            // Main content area
            HStack(spacing: 0) {
                // Left panel - Thickness slider
                LeftPanelView(
                    brushRadius: $viewModel.brushRadius,
                    isPainting: $viewModel.isPainting,
                    actualBrushSize: viewModel.brushPreviewSize
                )
                .frame(width: 80)

                // Center - Canvas
                CanvasContainerView(viewModel: viewModel)

                // Right panel - Color, transparency, SAM
                RightPanelView(
                    annotationColor: $viewModel.annotationColor,
                    imageTransparency: $viewModel.imageTransparency,
                    onSAMTapped: {
                        // SAM stub - no-op for MVP
                        print("SAM button tapped (stub)")
                    }
                )
                .frame(width: 100)
            }
        }
        .background(Color(white: 0.15))
        .ignoresSafeArea(.keyboard)
        .alert("Class Limit Reached", isPresented: $showingClassLimitAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("Maximum number of classes reached (8)")
        }
        .sheet(isPresented: $showingImagePicker) {
            ImagePickerView(
                onImageSelected: { url in
                    viewModel.importImage(from: url)
                },
                onFolderSelected: { url in
                    viewModel.importImagesFromFolder(url)
                },
                onProjectSelected: { url in
                    viewModel.openProject(at: url)
                }
            )
        }
        .sheet(isPresented: $showingExportSheet) {
            ExportSheetView(viewModel: viewModel)
        }
        .onReceive(viewModel.$showClassLimitAlert) { show in
            if show {
                showingClassLimitAlert = true
                viewModel.showClassLimitAlert = false
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in
            // Save when app goes to background
            viewModel.saveBeforeBackground()
        }
    }
}

// MARK: - Left Panel

struct LeftPanelView: View {
    @Binding var brushRadius: Float
    @Binding var isPainting: Bool
    var actualBrushSize: CGFloat  // Actual size considering zoom level

    /// Maximum display size for brush preview (in points)
    private let maxPreviewSize: CGFloat = 60

    /// Preview size (shows actual brush size, clamped to fit)
    private var previewSize: CGFloat {
        min(actualBrushSize, maxPreviewSize)
    }

    /// Scale factor shown when brush is larger than preview area
    private var displayScale: String? {
        if actualBrushSize > maxPreviewSize {
            let ratio = actualBrushSize / maxPreviewSize
            return String(format: "×%.1f", ratio)
        }
        return nil
    }

    var body: some View {
        VStack(spacing: 16) {
            // Paint/Erase toggle
            VStack(spacing: 8) {
                Button(action: { isPainting = true }) {
                    Image(systemName: "pencil.tip")
                        .font(.title2)
                        .foregroundColor(isPainting ? .green : .gray)
                }
                .buttonStyle(.plain)

                Button(action: { isPainting = false }) {
                    Image(systemName: "eraser.fill")
                        .font(.title2)
                        .foregroundColor(!isPainting ? .red : .gray)
                }
                .buttonStyle(.plain)
            }
            .padding(.top, 20)

            Spacer()

            // Brush size preview circle (reflects actual drawn size)
            ZStack {
                // Background circle (shows max preview area)
                Circle()
                    .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                    .frame(width: maxPreviewSize, height: maxPreviewSize)

                // Actual brush size circle
                Circle()
                    .fill(isPainting ? Color.green.opacity(0.5) : Color.red.opacity(0.5))
                    .frame(width: previewSize, height: previewSize)

                Circle()
                    .stroke(isPainting ? Color.green : Color.red, lineWidth: 2)
                    .frame(width: previewSize, height: previewSize)

                // Show scale indicator if brush is larger than preview
                if let scale = displayScale {
                    Text(scale)
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                }
            }
            .frame(width: maxPreviewSize, height: maxPreviewSize)
            .animation(.easeOut(duration: 0.1), value: previewSize)

            // Brush size number
            Text("\(Int(brushRadius))")
                .font(.caption)
                .foregroundColor(.white)

            // Thickness slider (vertical)
            ThicknessSliderView(radius: $brushRadius)
                .frame(height: 250)

            Spacer()
        }
        .frame(maxHeight: .infinity)
        .background(Color(white: 0.1))
    }
}

// MARK: - Preview

#Preview {
    MainView()
}
