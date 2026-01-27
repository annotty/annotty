import SwiftUI

@main
struct AnnottyApp: App {
    @StateObject private var importCoordinator = ImportCoordinator()

    var body: some Scene {
        WindowGroup {
            MainView()
                .environmentObject(importCoordinator)
                .onOpenURL { url in
                    importCoordinator.enqueue(url)
                }
        }
    }
}
