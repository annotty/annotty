#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

/// Uniforms passed to canvas shaders
struct CanvasUniforms {
    /// Canvas transform matrix (pan/zoom/rotate)
    matrix_float3x3 transform;
    /// Inverse of transform matrix
    matrix_float3x3 inverseTransform;
    /// Source image transparency (0.0 - 1.0)
    float imageAlpha;
    /// Mask overlay transparency (0.0 - 1.0)
    float maskAlpha;
    /// Padding to align maskColor to 16 bytes
    simd_float2 _padding1;
    /// Current annotation display color (RGBA)
    simd_float4 maskColor;
    /// Viewport size in pixels
    simd_float2 canvasSize;
    /// Source image size in pixels
    simd_float2 imageSize;
    /// Internal mask size in pixels (2x image with 4096 max clamp)
    simd_float2 maskSize;
    /// Scale factor from image to mask coordinates
    float maskScaleFactor;
    /// Padding for 16-byte struct alignment
    float _padding2;
    simd_float2 _padding3;
};

/// Parameters for brush stamp compute shader
struct BrushParams {
    /// Stamp center in mask coordinates
    simd_float2 center;
    /// Brush radius in mask pixels
    float radius;
    /// Paint value: 1 for paint, 0 for erase
    uint8_t paintValue;
    /// Padding for alignment
    uint8_t _padding[3];
};

/// Vertex output for canvas rendering
struct VertexOut {
    simd_float4 position;
    simd_float2 texCoord;
};

/// Buffer indices for shader bindings
enum BufferIndex {
    BufferIndexUniforms = 0,
    BufferIndexVertices = 1,
    BufferIndexBrushParams = 0
};

/// Texture indices for shader bindings
enum TextureIndex {
    TextureIndexImage = 0,
    TextureIndexMask = 1,
    TextureIndexOutput = 0
};

#endif /* ShaderTypes_h */
