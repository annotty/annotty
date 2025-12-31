<div align="center">

# Annotty

### Professional Segmentation Annotation for iPad

**Draw pixel-perfect masks with Apple Pencil — the way annotation should feel.**

[![Platform](https://img.shields.io/badge/Platform-iPadOS-blue.svg)](https://www.apple.com/ipad/)
[![Swift](https://img.shields.io/badge/Swift-5.9-orange.svg)](https://swift.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Export Formats](#export-formats) • [Contributing](#contributing)

</div>

> **Scope:** Annotty is designed for research, education, and prototyping.
> Cloud services and extended workflows are developed separately.

---

## Why Annotty?

Creating segmentation masks shouldn't feel like a chore. Annotty brings the intuitive drawing experience of professional art apps to machine learning annotation.

| Traditional Tools | Annotty |
|-------------------|---------|
| Mouse clicking, slow and imprecise | Natural brush strokes with Apple Pencil |
| Complex desktop workflows | Simple, focused iPad experience |
| Manual edge cleanup | Smart tools: Fill, Smooth, AI-assist |

---

## Features

### Core Drawing Experience

| Feature | Description |
|---------|-------------|
| **Apple Pencil Native** | Pressure-sensitive drawing with low latency |
| **Paint / Erase** | Switch modes instantly, adjust brush size |
| **Flood Fill** | One-tap to fill enclosed regions |
| **Boundary Smoothing** | Trace edges to remove jagged boundaries |
| **Multi-class Support** | Up to 8 annotation classes with custom names |

### Smart Tools

| Tool | How to Use |
|------|------------|
| **Paint/Erase** | Draw directly with Apple Pencil |
| **Fill** | Tap to flood-fill enclosed areas |
| **Smooth** | Trace boundaries to refine edges |
| **AI Segment** | Tap or drag for automatic segmentation (optional) |

### Navigation & Controls

| Input | Action |
|-------|--------|
| Apple Pencil | Paint / Erase mask |
| 2-finger drag | Pan canvas |
| 2-finger pinch | Zoom in/out |
| 2-finger rotate | Rotate view |
| 2-finger tap | Undo |
| 3-finger tap | Redo |

### Export Formats

| Format | Use Case |
|--------|----------|
| **PNG Masks** | Direct use in training pipelines |
| **COCO JSON** | Standard format for instance segmentation |
| **YOLO-seg** | Ultralytics YOLO segmentation training |

---

## Installation

### Requirements

- iPad with Apple Pencil support
- iPadOS 17.0+
- Xcode 15+ (for building from source)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/ykitaguchi77/annotty.git
cd annotty

# Open in Xcode
open Annotty.xcodeproj

# Build and run on your iPad
# Select your iPad as the target device and press Cmd+R
```

---

## Usage

### Basic Workflow

```
1. Load Images    →  Import folder or open existing project
2. Annotate       →  Paint, Fill, Smooth, or use AI assist
3. Navigate       →  Swipe through images (auto-saves)
4. Export         →  Choose format (PNG/COCO/YOLO)
```

### Tools (Right Panel)

| Button | Function |
|--------|----------|
| **Color swatches** | Select annotation class (1-8) |
| **Fill** | Toggle flood-fill mode |
| **Smooth** | Toggle boundary smoothing mode |
| **AI** | Toggle AI segmentation mode |
| **Settings** | Open settings panel |

### Settings Panel

| Setting | Description |
|---------|-------------|
| **Contrast / Brightness** | Adjust image display |
| **Mask Opacity** | Control mask transparency |
| **Smooth Kernel Size** | Adjust smoothing strength (7-31px) |
| **Class Names** | Customize class labels |

### Smooth Tool

The Smooth tool uses a competition-based moving average algorithm:

1. Tap **Smooth** to enter smooth mode
2. Trace along jagged boundaries
3. Boundaries are automatically refined
4. Works for both class-to-class and class-to-background edges

---

## Project Structure

```
YourProject/
├── images/          # Source images (PNG, JPG)
├── annotations/     # Editable color masks (auto-saved)
└── labels/          # Exported ML-ready labels
```

---

## Technical Details

### Architecture

- **UI**: SwiftUI
- **Rendering**: Metal (GPU-accelerated)
- **AI**: Core ML (optional)
- **Storage**: Local filesystem

### Mask Specification

- Resolution: 2x source image (max 4096px)
- Format: 8-bit indexed color PNG
- Classes: Up to 8 per project

---

## Use Cases

| Industry | Application |
|----------|-------------|
| **Medical AI** | Annotate scans with precise boundaries |
| **Autonomous Vehicles** | Label road scenes with detail |
| **Manufacturing** | Mark defects for inspection models |
| **Agriculture** | Segment crops, weeds, and soil |
| **Research** | Quick dataset creation for experiments |

---

## Roadmap

- [x] Apple Pencil drawing with pressure sensitivity
- [x] Flood fill tool
- [x] Boundary smoothing tool
- [x] AI-assisted segmentation
- [ ] Video annotation support
- [ ] Additional export formats
- [ ] Custom color palette

---

## Contributing

We welcome contributions! Here's how you can help:

1. **Report bugs** — Open an issue with reproduction steps
2. **Suggest features** — Share your ideas in Discussions
3. **Submit PRs** — Fork, branch, code, and submit
4. **Spread the word** — Star the repo and share with others

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

This project includes third-party models and software. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

---

<div align="center">

**Built for the ML community**

[Report Bug](https://github.com/ykitaguchi77/annotty/issues) • [Request Feature](https://github.com/ykitaguchi77/annotty/issues) • [Discussions](https://github.com/ykitaguchi77/annotty/discussions)

</div>
