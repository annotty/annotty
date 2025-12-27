import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// Image picker for selecting images or folders
struct ImagePickerView: View {
    @Environment(\.dismiss) private var dismiss
    let onImageSelected: (URL) -> Void

    @State private var selectedItem: PhotosPickerItem?
    @State private var showingFilePicker = false

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Photos Library option
                PhotosPicker(
                    selection: $selectedItem,
                    matching: .images
                ) {
                    HStack {
                        Image(systemName: "photo.on.rectangle")
                            .font(.title2)
                        Text("Select from Photos")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .onChange(of: selectedItem) { _, newItem in
                    Task {
                        await loadImage(from: newItem)
                    }
                }

                // Files option
                Button(action: { showingFilePicker = true }) {
                    HStack {
                        Image(systemName: "folder")
                            .font(.title2)
                        Text("Select from Files")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.orange)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }

                Spacer()

                Text("Select an image or folder containing images")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            .padding(24)
            .navigationTitle("Load Images")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
            .fileImporter(
                isPresented: $showingFilePicker,
                allowedContentTypes: [.image, .folder],
                allowsMultipleSelection: false
            ) { result in
                handleFileImport(result)
            }
        }
    }

    private func loadImage(from item: PhotosPickerItem?) async {
        guard let item = item else { return }

        do {
            if let data = try await item.loadTransferable(type: Data.self) {
                // Save to temporary file
                let tempURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                    .appendingPathExtension("png")

                try data.write(to: tempURL)
                await MainActor.run {
                    onImageSelected(tempURL)
                    dismiss()
                }
            }
        } catch {
            print("Failed to load image: \(error)")
        }
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            if let url = urls.first {
                onImageSelected(url)
                dismiss()
            }
        case .failure(let error):
            print("File import failed: \(error)")
        }
    }
}

// MARK: - Preview

#Preview {
    ImagePickerView { url in
        print("Selected: \(url)")
    }
}
