# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Annotty is a professional iPad segmentation annotation app designed for machine learning data preparation. It provides an intuitive drawing experience using Apple Pencil for creating pixel-perfect segmentation masks.

**Repository:** https://github.com/annotty/annotty
**Target Platform:** iPadOS (Apple Pencil required)
**Tech Stack:** SwiftUI + Metal (no external dependencies)

**Tagline:** Professional iPad annotation tool with Apple Pencil — intuitive drawing meets smart segmentation.

## Current Status (2025-01)

### Implemented Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Paint/Erase** | ✅ Complete | Apple Pencil drawing with adjustable brush size |
| **Flood Fill** | ✅ Complete | One-tap fill for enclosed regions |
| **Boundary Smoothing** | ✅ Complete | Competition-based moving average algorithm |
| **SAM 2.1 Integration** | ✅ Complete | Point prompt and box prompt segmentation |
| **Multi-class Support** | ✅ Complete | Up to 8 classes with custom names |
| **Export** | ✅ Complete | PNG, COCO JSON, YOLO-seg formats |
| **Undo/Redo** | ✅ Complete | 2-finger tap / 3-finger tap |
| **Pan/Zoom/Rotate** | ✅ Complete | 2-finger gestures |

### Architecture

```
User Input (Apple Pencil / Gestures)
    ↓
GestureCoordinator (Gestures/) → filters pencil vs finger input
    ↓
CanvasViewModel (ViewModels/) → state orchestration & tool logic
    ↓
MetalRenderer + TextureManager (Metal/) → GPU pipelines
    ↓
Shaders.metal → brush stamp, mask compositing
    ↓
Display (SwiftUI overlay + Metal rendering)
```

### Key Components

| File | Purpose |
|------|---------|
| `CanvasViewModel.swift` | Main state coordinator: drawing, fill, smooth, SAM, undo/redo |
| `MetalRenderer.swift` | Metal pipeline setup & rendering |
| `TextureManager.swift` | GPU texture management for images & masks |
| `GestureCoordinator.swift` | Input classification & gesture routing |
| `CanvasTransform.swift` | Pan/zoom/rotation matrix with coordinate transforms |
| `SAM2Service.swift` | SAM 2.1 CoreML inference service |

## Build & Development

```bash
# Open in Xcode
open Annotty.xcodeproj

# Build from command line
xcodebuild -scheme Annotty -destination 'platform=iOS Simulator,name=iPad Pro 13-inch (M5)'

# Run tests
xcodebuild test -scheme Annotty -destination 'platform=iOS Simulator,name=iPad Pro 13-inch (M5)'
```

## Project Structure

```
Annotty/
├── App/                    # App entry point
├── Models/                 # Data models (CanvasTransform, UndoAction, etc.)
├── Views/
│   ├── MainView.swift
│   ├── CanvasView/         # MetalCanvasView, SmoothStrokeOverlay
│   ├── LeftPanel/          # Brush size slider
│   ├── RightPanel/         # Tools, colors, settings
│   └── TopBar/             # Navigation, export
├── ViewModels/
│   ├── CanvasViewModel.swift  # Central state management
│   └── UndoManager.swift
├── Metal/
│   ├── MetalRenderer.swift
│   ├── TextureManager.swift
│   └── Shaders/
├── Gestures/
│   └── GestureCoordinator.swift
├── Services/
│   ├── FileManager/        # Project I/O, auto-save
│   ├── Export/             # PNG, COCO, YOLO
│   ├── ImageProcessing/    # Color parsing, contours
│   └── SAM/                # SAM 2.1 integration
│       ├── SAM2Service.swift
│       └── Models/         # CoreML models (.mlpackage)
└── Utils/
```

## Runtime Folder Structure

```
YourProject/
├── images/          # Source images (PNG, JPG)
├── annotations/     # Editable color masks (auto-saved)
└── labels/          # Exported ML-ready labels
```

## Technical Specifications

### Internal Mask
- Resolution: **2x source image (max 4096px)**
- Format: `UInt8` buffer, `MTLPixelFormat.r8Uint`
- Max classes: **8**

### Smoothing Algorithm
- **Competition-based moving average**
- Kernel size: 7-31px (configurable in Settings)
- 2-pass application for smooth results
- Handles class-to-class and class-to-background boundaries

### SAM 2.1 Models
- **Tiny**: Faster, lower memory
- **Small**: More accurate
- Models included in `Services/SAM/Models/`

## Gesture Mapping

| Input | Action |
|-------|--------|
| Apple Pencil | Paint / Erase / Tool action |
| 2-finger drag | Pan |
| 2-finger pinch | Zoom |
| 2-finger rotate | Rotate |
| 2-finger tap | Undo |
| 3-finger tap | Redo |

## OSS Scope & Strategy

### In Scope (This Repository)
- All annotation tools (Paint, Fill, Smooth, SAM)
- Local file management
- Export formats (PNG, COCO, YOLO)
- On-device AI (SAM 2.1)

### Out of Scope (Separate Development)
- Cloud services
- User accounts
- Team collaboration
- Extended workflows

**Note:** This is intentional. OSS focuses on the core annotation experience.
Cloud and collaboration features are developed separately.

## Future OSS Roadmap

- [ ] Video annotation support
- [ ] Additional export formats
- [ ] Custom color palette

## Licensing

- **This project:** MIT License
- **SAM 2.1 models:** Apache 2.0 (Meta Platforms, Inc.)

See `THIRD_PARTY_NOTICES.md` for details.

## Auto-Save Triggers

- Stroke end (500ms debounce)
- Image navigation (immediate)
- App backgrounding (immediate)
