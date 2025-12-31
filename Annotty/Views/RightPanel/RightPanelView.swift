import SwiftUI

/// Right panel with annotation color, settings button, and SAM button
struct RightPanelView: View {
    @Binding var annotationColor: Color
    @Binding var isFillMode: Bool
    @Binding var isSmoothMode: Bool
    @Binding var isSAMMode: Bool
    let isSAMLoading: Bool
    let isSAMProcessing: Bool
    let classNames: [String]
    let onSettingsTapped: () -> Void
    let onSAMTapped: () -> Void

    /// Preset colors for annotation (index+1 = classID)
    /// These must match MetalRenderer.classColors exactly
    private let presetColors: [Color] = [
        Color(red: 1, green: 0, blue: 0),        // 1: red
        Color(red: 1, green: 0.5, blue: 0),      // 2: orange
        Color(red: 1, green: 1, blue: 0),        // 3: yellow
        Color(red: 0, green: 1, blue: 0),        // 4: green
        Color(red: 0, green: 1, blue: 1),        // 5: cyan
        Color(red: 0, green: 0, blue: 1),        // 6: blue
        Color(red: 0.5, green: 0, blue: 1),      // 7: purple
        Color(red: 1, green: 0.4, blue: 0.7)     // 8: pink
    ]

    /// Display name for a class (shows class number if unnamed)
    private func displayName(for index: Int) -> String {
        let name = classNames[index]
        return name.isEmpty ? "\(index + 1)" : name
    }

    /// SAM button label based on current state
    private var samButtonLabel: String {
        if isSAMLoading {
            return "Loading..."
        } else if isSAMProcessing {
            return "Processing"
        } else if isSAMMode {
            return "Tap object"
        } else {
            return "SAM"
        }
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

                // Smooth mode toggle button
                Button(action: {
                    isSmoothMode.toggle()
                }) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(isSmoothMode ? annotationColor : Color(white: 0.2))
                        .frame(width: 50, height: 50)
                        .overlay(
                            Image(systemName: "waveform.path")
                                .font(.title2)
                                .foregroundColor(isSmoothMode ? .white : .gray)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(isSmoothMode ? Color.white : Color.gray.opacity(0.5), lineWidth: isSmoothMode ? 2 : 1)
                        )
                }
                .buttonStyle(.plain)

                Text(isSmoothMode ? "Trace edge" : "Smooth")
                    .font(.caption2)
                    .foregroundColor(isSmoothMode ? .cyan : .gray)
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

            // SAM button
            Button(action: onSAMTapped) {
                VStack(spacing: 4) {
                    if isSAMLoading || isSAMProcessing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.8)
                            .frame(width: 24, height: 24)
                    } else {
                        Image(systemName: "wand.and.stars")
                            .font(.title2)
                    }
                    Text(samButtonLabel)
                        .font(.caption)
                }
                .foregroundColor(isSAMMode ? .cyan : .white)
                .padding(12)
                .background(isSAMMode ? Color.cyan.opacity(0.3) : Color(white: 0.2))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isSAMMode ? Color.cyan : Color.clear, lineWidth: 2)
                )
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(isSAMLoading || isSAMProcessing)

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
        annotationColor: .constant(Color(red: 1, green: 0, blue: 0)),
        isFillMode: .constant(false),
        isSmoothMode: .constant(false),
        isSAMMode: .constant(false),
        isSAMLoading: false,
        isSAMProcessing: false,
        classNames: ["iris", "eyelid", "sclera", "pupil", "", "", "", ""],
        onSettingsTapped: {},
        onSAMTapped: {}
    )
    .frame(width: 120, height: 600)
}
