# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Annoty is an iPad segmentation annotation app designed for machine learning data preparation. It provides a Procreate-like drawing experience using Apple Pencil for creating segmentation masks.

**Target Platform:** iPadOS (offline, Apple Pencil required)
**Tech Stack:** SwiftUI + Metal + Swift Package Manager

**Core User Flow:** Open image → Paint/Erase → (Future: SAM refinement) → Undo if needed → Export annotation

## Build & Development

```bash
# Open in Xcode
open Annoty.xcodeproj

# Build from command line
xcodebuild -scheme Annoty -destination 'platform=iOS Simulator,name=iPad Pro (12.9-inch)'

# Run tests
xcodebuild test -scheme Annoty -destination 'platform=iOS Simulator,name=iPad Pro (12.9-inch)'
```

## Project Structure

```
Annoty/
├── App/                    # App entry point
│   └── AnnotyApp.swift
├── Models/                 # Data models
│   ├── AnnotationProject.swift
│   ├── ImageItem.swift
│   ├── MaskClass.swift
│   ├── InternalMask.swift
│   ├── UndoAction.swift
│   └── CanvasTransform.swift
├── Views/                  # SwiftUI views
│   ├── MainView.swift
│   ├── CanvasView/         # Metal canvas integration
│   ├── LeftPanel/          # Thickness slider
│   ├── RightPanel/         # Color, transparency, SAM
│   ├── TopBar/             # Navigation, export
│   └── Common/             # Shared components
├── ViewModels/             # State management
│   ├── CanvasViewModel.swift
│   └── UndoManager.swift
├── Metal/                  # GPU rendering
│   ├── MetalRenderer.swift
│   ├── TextureManager.swift
│   └── Shaders/
│       ├── Shaders.metal
│       └── ShaderTypes.h
├── Gestures/               # Input handling
│   ├── GestureCoordinator.swift
│   └── DrawingCanvasView.swift
├── Services/
│   ├── FileManager/        # Project files, auto-save
│   ├── Export/             # PNG, COCO, YOLO exporters
│   ├── ImageProcessing/    # Color parsing, contour extraction
│   └── SAM/                # SAM stub (future)
└── Utils/                  # Extensions
```

## App Folder Structure (Runtime)

```
AppRoot/
├── images/        # Source images
├── annotations/   # Editable color PNG masks (for visualization/editing)
└── labels/        # Export output (PNG/COCO/YOLO for ML training)
```

## Key Technical Specifications

### Internal Mask
- Resolution: **2x the source image with 4096px max clamp**
  - 1024px image → 2048px mask (×2)
  - 2048px image → 4096px mask (×2)
  - 4096px image → 4096px mask (×1, clamped)
- Type: `UInt8` buffer (values strictly 0 or 1), **Boolean arrays prohibited**
- GPU texture format: `MTLPixelFormat.r8Uint`
- Max classes: **8** (show alert when limit reached)

### Undo Patches
- **MVP: Uncompressed storage** (compression out of scope)
- Patch data: `Data` type holding raw UInt8 bytes

### Annotation Loading
- Auto-load `annotations/{basename}.png` when opening image
- Color interpretation:
  - White (#FFFFFF) = background (not masked)
  - Any other color = mask class
- **Color snapping:** Nearest unique RGB value (no tolerance clustering)
- **Anti-aliased pixels:** Treated as nearest solid color

### Export Formats (to `labels/` folder)
1. **PNG** - Color mask or class-separate binary masks
2. **COCO JSON** - Polygon segmentation with category_id
3. **YOLO-seg** - Normalized (0-1) polygon coordinates

## UI Layout

- **Left:** Pen thickness slider (vertical, 1-100px radius, logarithmic scale)
- **Center:** Metal canvas with image + mask overlay
- **Right:** Annotation color picker, image transparency slider, SAM button (stub)
- **Top bar:** Image navigation (◀ 12/128 ▶), Export annotation button

## Gesture Mapping

| Input | Action | Implementation |
|-------|--------|----------------|
| Pencil drag | Paint/Erase | `UITouch.type == .pencil` |
| 2-finger drag | Pan | `UIPanGestureRecognizer` |
| 2-finger pinch | Zoom | `UIPinchGestureRecognizer` |
| 2-finger twist | Free rotation | `UIRotationGestureRecognizer` |
| 2-finger tap | Undo | Custom tap gesture |
| 3-finger tap | Redo | Custom tap gesture |

## Drawing Implementation

- Brush: Circle stamp method via Metal compute shader
- Stamp interval: ≤ radius for continuous strokes
- Pencil coordinates inverse-transformed to mask coordinates via `CanvasTransform`

## Undo/Redo (bbox patch method)

1. At stroke start: capture bbox region from GPU
2. During stroke: expand bbox as stamps are applied
3. At stroke end: create `UndoAction` with (classID, bbox, previousPatch)
4. Undo: restore patch to GPU texture

## Auto-Save

Triggers:
- Stroke end (500ms debounce)
- Image navigation (immediate)
- App backgrounding (immediate)

## MVP Scope

- SAM integration: UI stub only (no inference)
- No cloud sync, no video support
