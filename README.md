<div align="center">

# Annotty

### The First iPad App with SAM 2.1 for ML Segmentation Annotation

**Draw pixel-perfect segmentation masks with Apple Pencil. Powered by AI.**

[![Platform](https://img.shields.io/badge/Platform-iPadOS-blue.svg)](https://www.apple.com/ipad/)
[![Swift](https://img.shields.io/badge/Swift-5.9-orange.svg)](https://swift.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SAM 2.1](https://img.shields.io/badge/SAM-2.1-purple.svg)](https://github.com/facebookresearch/sam2)

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Export Formats](#export-formats) • [Contributing](#contributing)

</div>

> **Note**: This repository provides an open-source, on-device annotation tool.
> Cloud-based services, medical-grade inference, and enterprise features
> are provided separately and are not part of this repository.

---

## The Problem

Creating segmentation masks for machine learning is **painful**:

| Frustration | Current Tools |
|-------------|---------------|
| **Desktop-only workflows** | CVAT, Label Studio, Labelbox — all require a computer |
| **No stylus optimization** | Mouse clicking is slow and imprecise for pixel-level work |
| **Cloud dependency** | Your sensitive data uploaded to third-party servers |
| **Expensive pricing** | Enterprise tools cost $500-2,000+/month per team |
| **No AI assistance on mobile** | SAM integration exists only on web/desktop |

**You shouldn't need a $2,000/month enterprise contract to annotate images efficiently.**

---

## The Solution

**Annotty** brings professional-grade segmentation annotation to iPad with AI superpowers:

### Features

| Feature | Description |
|---------|-------------|
| **Apple Pencil Native** | Draw masks like you're painting in Procreate — natural, fast, precise |
| **SAM 2.1 Integration** | Tap or draw a box, let Meta's AI complete the mask instantly |
| **Flood Fill** | One-tap to fill enclosed regions |
| **Boundary Smoothing** | Trace edges to smooth jagged boundaries with moving average filter |
| **100% Offline** | No internet required. Your data never leaves your device |
| **Multi-class Support** | Up to 8 annotation classes with customizable names |
| **Free & Open Source** | No subscriptions, no limits, forever free |

### Tool Modes

| Tool | How to Use |
|------|------------|
| **Paint/Erase** | Draw directly with Apple Pencil |
| **Fill** | Tap to flood-fill enclosed areas |
| **Smooth** | Trace boundaries to remove jagged edges |
| **SAM** | Tap (point prompt) or drag (box prompt) for AI segmentation |

### SAM 2.1 Modes

- **Point Prompt**: Tap anywhere on an object → AI segments it
- **Box Prompt**: Draw a bounding box → AI fills the precise mask
- **Model Options**: Choose Tiny (faster) or Small (more accurate)

### Export Formats

| Format | Use Case |
|--------|----------|
| **PNG Masks** | Direct use in training pipelines |
| **COCO JSON** | Standard format for instance segmentation |
| **YOLO-seg** | Ultralytics YOLO segmentation training |

---

## Why Annotty?

<table>
<tr>
<td width="50%">

### Before Annotty
- Open laptop, launch Docker, start CVAT
- Click... click... click... with mouse
- Upload images to cloud service
- Pay $99/month minimum
- No AI help on mobile

</td>
<td width="50%">

### With Annotty
- Open iPad, launch app
- Draw naturally with Apple Pencil
- Everything stays on your device
- Free forever
- SAM 2.1 does the heavy lifting

</td>
</tr>
</table>

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
2. Annotate       →  Paint with Apple Pencil, Fill, Smooth, or use SAM
3. Navigate       →  Swipe through images, auto-saves
4. Export         →  Choose format (PNG/COCO/YOLO)
```

### Controls

| Input | Action |
|-------|--------|
| Apple Pencil | Paint / Erase mask |
| 2-finger drag | Pan canvas |
| 2-finger pinch | Zoom in/out |
| 2-finger rotate | Rotate view |
| 2-finger tap | Undo |
| 3-finger tap | Redo |

### Tools (Right Panel)

| Button | Function |
|--------|----------|
| **Color swatches** | Select annotation class (1-8) |
| **Fill** | Toggle flood-fill mode (tap to fill) |
| **Smooth** | Toggle boundary smoothing mode |
| **SAM** | Toggle AI segmentation mode |
| **Settings (gear)** | Open settings panel |

### Settings Panel

| Setting | Description |
|---------|-------------|
| **Contrast / Brightness** | Adjust image display |
| **Mask Fill / Edge Alpha** | Control mask transparency |
| **Smooth Kernel Size** | Adjust smoothing strength (7-31px) |
| **SAM Model** | Choose Tiny or Small model |
| **Class Names** | Customize class labels |

### Smooth Mode

The Smooth tool uses a **competition-based moving average algorithm**:

1. Tap the **Smooth** button to enter smooth mode
2. Trace along jagged boundaries with Apple Pencil
3. Boundaries are automatically smoothed
4. Works for both class-to-class and class-to-background edges
5. Adjust **Kernel Size** in Settings for stronger/weaker effect

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
- **AI**: Core ML (SAM 2.1)
- **Storage**: Local filesystem

### Mask Specification

- Resolution: 2x source image (max 4096px)
- Format: 8-bit indexed color PNG
- Classes: Up to 8 per project

### Smoothing Algorithm

The boundary smoothing uses a competition-based approach:
- Each boundary pixel competes among all classes (including background)
- Winner is determined by moving average in kernel window
- Multiple passes ensure smooth results without gaps between classes

---

## Use Cases

| Industry | Application |
|----------|-------------|
| **Medical AI** | Annotate scans with HIPAA-compliant offline workflow |
| **Autonomous Vehicles** | Label road scenes with precision |
| **Manufacturing** | Mark defects for quality inspection models |
| **Agriculture** | Segment crops, weeds, and soil |
| **Research** | Quick dataset creation for experiments |

---

## Roadmap

- [x] Flood fill tool
- [x] Boundary smoothing tool
- [x] SAM 2.1 integration (Point & Box prompts)
- [ ] Video annotation support
- [ ] Cloud sync (optional)
- [ ] Team collaboration
- [ ] Custom model integration

---

## Contributing

We welcome contributions! Here's how you can help:

1. **Report bugs** — Open an issue with reproduction steps
2. **Suggest features** — Share your ideas in Discussions
3. **Submit PRs** — Fork, branch, code, and submit
4. **Spread the word** — Star the repo and share with others

---

## Acknowledgments

- [Meta AI](https://github.com/facebookresearch/sam2) — SAM 2.1 model (Apache 2.0)
- Apple — Metal framework and Core ML

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

SAM 2.1 models are licensed under Apache 2.0 by Meta AI.

---

<div align="center">

**Built with love for the ML community**

[Report Bug](https://github.com/ykitaguchi77/annotty/issues) • [Request Feature](https://github.com/ykitaguchi77/annotty/issues) • [Discussions](https://github.com/ykitaguchi77/annotty/discussions)

</div>
