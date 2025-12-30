# SAM 2.1 Core ML Models

This directory contains Core ML versions of Meta's Segment Anything Model 2.1 (SAM 2.1).

## Original Source

- **Model**: Segment Anything Model 2 (SAM 2)
- **Developer**: Meta AI (Facebook Research)
- **Repository**: https://github.com/facebookresearch/sam2
- **License**: Apache 2.0

## Models Included

| Model | Description |
|-------|-------------|
| SAM2_1TinyImageEncoderFLOAT16 | Image encoder (Tiny variant) |
| SAM2_1TinyPromptEncoderFLOAT16 | Prompt encoder (Tiny variant) |
| SAM2_1TinyMaskDecoderFLOAT16 | Mask decoder (Tiny variant) |
| SAM2_1SmallImageEncoderFLOAT16 | Image encoder (Small variant) |
| SAM2_1SmallPromptEncoderFLOAT16 | Prompt encoder (Small variant) |
| SAM2_1SmallMaskDecoderFLOAT16 | Mask decoder (Small variant) |

## Conversion

These models have been converted from PyTorch to Core ML format (`.mlpackage`) for use on iOS/iPadOS devices.

- Format: Core ML (FLOAT16)
- Input size: 1024x1024

## License

The original SAM 2 model weights and code are licensed under the Apache License 2.0.

```
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## References

- [SAM 2 Paper](https://ai.meta.com/sam2/)
- [SAM 2 GitHub](https://github.com/facebookresearch/sam2)
- [Meta AI Blog Post](https://ai.meta.com/blog/segment-anything-2/)
