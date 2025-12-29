import SwiftUI

/// Right panel with annotation color, settings button, and SAM button
struct RightPanelView: View {
    @Binding var annotationColor: Color
    @Binding var isFillMode: Bool
    let classNames: [String]
    let onSettingsTapped: () -> Void
    let onSAMTapped: () -> Void

    /// Preset colors for annotation (index+1 = classID)
    /// Class 1=red, 2=orange, 3=yellow, 4=green, 5=cyan, 6=blue, 7=purple, 8=pink
    private let presetColors: [Color] = [
        .red, .orange, .yellow,
        .green, .cyan, .blue,
        .purple, .pink
    ]

    /// Display name for a class (shows class number if unnamed)
    private func displayName(for index: Int) -> String {
        let name = classNames[index]
        return name.isEmpty ? "\(index + 1)" : name
    }

    var body: some View {
        VStack(spacing: 24) {
            // Annotation color section with class names
            VStack(spacing: 6) {
                Text("Classes")
                    .font(.caption)
                    .foregroundColor(.gray)

                // Color + name list (vertical)
                VStack(spacing: 3) {
                    ForEach(Array(presetColors.enumerated()), id: \.offset) { index, color in
                        HStack(spacing: 6) {
                            // Color circle
                            Circle()
                                .fill(color)
                                .frame(width: 18, height: 18)
                                .overlay(
                                    Circle()
                                        .stroke(annotationColor == color ? Color.white : Color.clear, lineWidth: 2)
                                )

                            // Class name (truncated)
                            Text(displayName(for: index))
                                .font(.caption2)
                                .foregroundColor(annotationColor == color ? .white : .gray)
                                .lineLimit(1)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(annotationColor == color ? Color.white.opacity(0.15) : Color.clear)
                        )
                        .contentShape(Rectangle())
                        .onTapGesture {
                            annotationColor = color
                        }
                    }
                }

                // Fill mode toggle button
                Button(action: {
                    isFillMode.toggle()
                }) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(isFillMode ? annotationColor : Color(white: 0.2))
                        .frame(width: 50, height: 50)
                        .overlay(
                            Image(systemName: "drop.fill")
                                .font(.title2)
                                .foregroundColor(isFillMode ? .white : .gray)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(isFillMode ? Color.white : Color.gray.opacity(0.5), lineWidth: isFillMode ? 2 : 1)
                        )
                }
                .buttonStyle(.plain)

                Text(isFillMode ? "Tap to fill" : "Fill")
                    .font(.caption2)
                    .foregroundColor(isFillMode ? .cyan : .gray)
            }
            .padding(.top, 20)

            Divider()
                .background(Color.gray)

            // Image settings button
            VStack(spacing: 8) {
                Text("Image")
                    .font(.caption)
                    .foregroundColor(.gray)

                Button(action: onSettingsTapped) {
                    VStack(spacing: 4) {
                        Image(systemName: "slider.horizontal.3")
                            .font(.title2)
                        Text("Settings")
                            .font(.caption2)
                    }
                    .foregroundColor(.white)
                    .padding(12)
                    .background(Color(white: 0.25))
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }

            Spacer()

            // SAM button (stub for MVP)
            Button(action: onSAMTapped) {
                VStack(spacing: 4) {
                    Image(systemName: "wand.and.stars")
                        .font(.title2)
                    Text("SAM")
                        .font(.caption)
                }
                .foregroundColor(.gray)
                .padding(12)
                .background(Color(white: 0.2))
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(true) // Disabled for MVP
            .opacity(0.5)

            Spacer()
                .frame(height: 20)
        }
        .frame(maxHeight: .infinity)
        .background(Color(white: 0.1))
    }
}

// MARK: - Preview

#Preview {
    RightPanelView(
        annotationColor: .constant(.red),
        isFillMode: .constant(false),
        classNames: ["iris", "eyelid", "sclera", "pupil", "", "", "", ""],
        onSettingsTapped: {},
        onSAMTapped: {}
    )
    .frame(width: 120, height: 600)
}
