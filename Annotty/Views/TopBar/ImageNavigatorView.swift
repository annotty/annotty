import SwiftUI

/// Image navigation controls (◀ 12/128 ▶)
struct ImageNavigatorView: View {
    let currentIndex: Int
    let totalCount: Int
    let onPrevious: () -> Void
    let onNext: () -> Void

    private var displayIndex: Int {
        totalCount > 0 ? currentIndex + 1 : 0
    }

    var body: some View {
        HStack(spacing: 16) {
            // Previous button
            Button(action: onPrevious) {
                Image(systemName: "chevron.left")
                    .font(.title2)
                    .foregroundColor(currentIndex > 0 ? .white : .gray)
            }
            .buttonStyle(.plain)
            .disabled(currentIndex <= 0)

            // Position indicator
            Text("\(displayIndex)/\(totalCount)")
                .font(.system(size: 18, weight: .medium, design: .monospaced))
                .foregroundColor(.white)
                .frame(minWidth: 80)

            // Next button
            Button(action: onNext) {
                Image(systemName: "chevron.right")
                    .font(.title2)
                    .foregroundColor(currentIndex < totalCount - 1 ? .white : .gray)
            }
            .buttonStyle(.plain)
            .disabled(currentIndex >= totalCount - 1)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color(white: 0.2))
        .cornerRadius(8)
    }
}

// MARK: - Preview

#Preview {
    ImageNavigatorView(
        currentIndex: 11,
        totalCount: 128,
        onPrevious: {},
        onNext: {}
    )
    .background(Color(white: 0.1))
}
