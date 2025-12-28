import SwiftUI

/// Top bar with image navigation and export button
struct TopBarView: View {
    let currentIndex: Int
    let totalCount: Int
    let isSaving: Bool
    let onPrevious: () -> Void
    let onNext: () -> Void
    let onGoTo: (Int) -> Void
    let onExport: () -> Void
    let onLoad: () -> Void
    let onReload: () -> Void

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

            // Reload button
            Button(action: onReload) {
                Image(systemName: "arrow.clockwise")
                    .padding(8)
                    .background(Color.gray.opacity(0.5))
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
                onNext: onNext,
                onGoTo: onGoTo
            )

            // Saving indicator
            if isSaving {
                HStack(spacing: 6) {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(0.8)
                    Text("Saving...")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .padding(.leading, 12)
            }

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
        isSaving: false,
        onPrevious: {},
        onNext: {},
        onGoTo: { _ in },
        onExport: {},
        onLoad: {},
        onReload: {}
    )
}
