import SwiftUI

/// Top bar with image navigation and export button
struct TopBarView: View {
    let currentIndex: Int
    let totalCount: Int
    let onPrevious: () -> Void
    let onNext: () -> Void
    let onExport: () -> Void
    let onLoad: () -> Void

    var body: some View {
        HStack {
            // Load button
            Button(action: onLoad) {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                    Text("Load")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.blue.opacity(0.8))
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)

            Spacer()

            // Image navigation
            ImageNavigatorView(
                currentIndex: currentIndex,
                totalCount: totalCount,
                onPrevious: onPrevious,
                onNext: onNext
            )

            Spacer()

            // Export button
            Button(action: onExport) {
                HStack(spacing: 4) {
                    Image(systemName: "square.and.arrow.up")
                    Text("Export annotation")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.green.opacity(0.8))
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(totalCount == 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color(white: 0.1))
    }
}

// MARK: - Preview

#Preview {
    TopBarView(
        currentIndex: 12,
        totalCount: 128,
        onPrevious: {},
        onNext: {},
        onExport: {},
        onLoad: {}
    )
}
