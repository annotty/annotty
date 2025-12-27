import SwiftUI

/// Right panel with annotation color, transparency slider, and SAM button
struct RightPanelView: View {
    @Binding var annotationColor: Color
    @Binding var imageTransparency: Float
    let onSAMTapped: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            // Annotation color picker
            VStack(spacing: 8) {
                Text("Color")
                    .font(.caption)
                    .foregroundColor(.gray)

                ColorPicker("", selection: $annotationColor)
                    .labelsHidden()
                    .frame(width: 44, height: 44)
            }
            .padding(.top, 20)

            Divider()
                .background(Color.gray)

            // Image transparency slider
            VStack(spacing: 8) {
                Text("Opacity")
                    .font(.caption)
                    .foregroundColor(.gray)

                TransparencySliderView(value: $imageTransparency)
                    .frame(height: 200)

                Text("\(Int(imageTransparency * 100))%")
                    .font(.caption)
                    .foregroundColor(.white)
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
        imageTransparency: .constant(1.0),
        onSAMTapped: {}
    )
    .frame(width: 100, height: 600)
}
