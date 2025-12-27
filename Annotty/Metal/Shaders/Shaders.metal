#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

// MARK: - Vertex Shader

struct VertexIn {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct FragmentIn {
    float4 position [[position]];
    float2 texCoord;
};

/// Full-screen quad vertex shader for canvas rendering
vertex FragmentIn canvasVertex(uint vertexID [[vertex_id]],
                                constant CanvasUniforms &uniforms [[buffer(BufferIndexUniforms)]]) {
    // Full-screen triangle vertices
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };

    float2 texCoords[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };

    FragmentIn out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.texCoord = texCoords[vertexID];

    return out;
}

// MARK: - Fragment Shader

/// Transform UV coordinates using the inverse canvas transform
float2 transformUV(float2 uv, float2 canvasSize, float2 imageSize, matrix_float3x3 inverseTransform) {
    // Convert UV to screen coordinates
    float2 screenPos = uv * canvasSize;

    // Apply inverse transform to get image coordinates
    float3 transformed = inverseTransform * float3(screenPos, 1.0);
    float2 imagePos = transformed.xy;

    // Convert to UV in image space
    return imagePos / imageSize;
}

/// Canvas compositing fragment shader
/// Blends source image with mask overlay
fragment float4 canvasFragment(FragmentIn in [[stage_in]],
                                texture2d<float, access::sample> imageTexture [[texture(TextureIndexImage)]],
                                texture2d<uint, access::read> maskTexture [[texture(TextureIndexMask)]],
                                constant CanvasUniforms &uniforms [[buffer(BufferIndexUniforms)]]) {

    constexpr sampler linearSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

    // Transform UV to image space
    float2 imageUV = transformUV(in.texCoord, uniforms.canvasSize, uniforms.imageSize, uniforms.inverseTransform);

    // Check if UV is within image bounds
    if (imageUV.x < 0.0 || imageUV.x > 1.0 || imageUV.y < 0.0 || imageUV.y > 1.0) {
        // Background color (dark gray)
        return float4(0.2, 0.2, 0.2, 1.0);
    }

    // Sample source image with bilinear filtering
    float4 imageColor = imageTexture.sample(linearSampler, imageUV);

    // Calculate mask UV (scaled by maskScaleFactor)
    float2 maskUV = imageUV * float2(uniforms.maskSize);
    uint2 maskCoord = uint2(maskUV);

    // Clamp to mask bounds
    maskCoord = min(maskCoord, uint2(uniforms.maskSize) - 1);

    // Sample mask with nearest neighbor (read directly)
    uint maskValue = maskTexture.read(maskCoord).r;

    // Apply wash-out effect: blend toward white as imageAlpha decreases
    // This makes the image appear faded/washed out rather than darkened
    float4 white = float4(1.0, 1.0, 1.0, 1.0);
    float4 result = mix(white, imageColor, uniforms.imageAlpha);
    result.a = 1.0; // Keep full opacity for the composited result

    if (maskValue > 0) {
        // Blend mask color over image (mask stays at full color, not washed out)
        float4 maskOverlay = float4(uniforms.maskColor.rgb, uniforms.maskAlpha);
        result = mix(result, maskOverlay, maskOverlay.a);
    }

    return result;
}

// MARK: - Compute Shaders

/// Brush stamp compute shader
/// Applies a circular stamp to the mask texture
kernel void brushStamp(texture2d<uint, access::read_write> mask [[texture(TextureIndexOutput)]],
                       constant BrushParams &params [[buffer(BufferIndexBrushParams)]],
                       uint2 gid [[thread_position_in_grid]]) {

    // Check bounds
    if (gid.x >= mask.get_width() || gid.y >= mask.get_height()) {
        return;
    }

    // Calculate distance from stamp center
    float2 pos = float2(gid);
    float dist = distance(pos, params.center);

    // Apply stamp if within radius
    if (dist <= params.radius) {
        mask.write(uint4(params.paintValue, 0, 0, 0), gid);
    }
}

/// Clear mask compute shader
/// Fills the entire mask with a value
kernel void clearMask(texture2d<uint, access::write> mask [[texture(TextureIndexOutput)]],
                      constant uint &clearValue [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]]) {

    if (gid.x >= mask.get_width() || gid.y >= mask.get_height()) {
        return;
    }

    mask.write(uint4(clearValue, 0, 0, 0), gid);
}
