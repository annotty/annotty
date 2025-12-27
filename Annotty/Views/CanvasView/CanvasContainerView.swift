import SwiftUI

/// Container view that combines MetalCanvasView with UIKit gesture handling
struct CanvasContainerView: View {
    @ObservedObject var viewModel: CanvasViewModel

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Metal canvas with UIKit touch handling
                if let renderer = viewModel.renderer {
                    MetalCanvasView(
                        renderer: renderer,
                        gestureCoordinator: viewModel.gestureCoordinator
                    )
                } else {
                    // Placeholder when no renderer
                    Rectangle()
                        .fill(Color(white: 0.2))
                        .overlay(
                            Text("No image loaded")
                                .foregroundColor(.gray)
                        )
                }

                // Brush preview circle (when drawing)
                if viewModel.isDrawing {
                    Circle()
                        .stroke(viewModel.isPainting ? Color.green : Color.red, lineWidth: 2)
                        .frame(width: viewModel.brushPreviewSize, height: viewModel.brushPreviewSize)
                        .position(viewModel.lastDrawPoint)
                        .allowsHitTesting(false)
                }
            }
            .onAppear {
                viewModel.updateViewSize(geometry.size)
            }
            .onChange(of: geometry.size) { _, newSize in
                viewModel.updateViewSize(newSize)
            }
        }
    }
}

// MARK: - Preview

#Preview {
    CanvasContainerView(viewModel: CanvasViewModel())
}
