import SwiftUI

/// Overlay view that displays a bounding box rectangle during SAM bbox drag
struct SAMBBoxOverlay: View {
    let start: CGPoint
    let end: CGPoint

    /// Computed rectangle from start and end points
    private var rect: CGRect {
        let minX = min(start.x, end.x)
        let minY = min(start.y, end.y)
        let maxX = max(start.x, end.x)
        let maxY = max(start.y, end.y)
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    var body: some View {
        ZStack {
            // Semi-transparent fill
            Rectangle()
                .fill(Color.cyan.opacity(0.15))
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)

            // Dashed border
            Rectangle()
                .stroke(
                    Color.cyan,
                    style: StrokeStyle(
                        lineWidth: 2,
                        lineCap: .round,
                        lineJoin: .round,
                        dash: [8, 4]
                    )
                )
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)

            // Corner markers
            ForEach(cornerPositions, id: \.self) { position in
                Circle()
                    .fill(Color.white)
                    .frame(width: 8, height: 8)
                    .overlay(
                        Circle()
                            .stroke(Color.cyan, lineWidth: 2)
                    )
                    .position(position)
            }
        }
    }

    /// Corner positions for visual markers
    private var cornerPositions: [CGPoint] {
        [
            CGPoint(x: rect.minX, y: rect.minY),  // Top-left
            CGPoint(x: rect.maxX, y: rect.minY),  // Top-right
            CGPoint(x: rect.minX, y: rect.maxY),  // Bottom-left
            CGPoint(x: rect.maxX, y: rect.maxY)   // Bottom-right
        ]
    }
}

// MARK: - Preview

#Preview {
    ZStack {
        Color(white: 0.2)
        SAMBBoxOverlay(
            start: CGPoint(x: 100, y: 100),
            end: CGPoint(x: 300, y: 250)
        )
    }
}
