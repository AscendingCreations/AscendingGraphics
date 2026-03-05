struct wrapped_f32 {
  @size(16) elem: f32
}

struct Global {
    views: array<mat4x4<f32>, 8>,
    scales: array<wrapped_f32, 8>,
    proj: mat4x4<f32>,
    inverse_proj: mat4x4<f32>,
    eye: vec3<f32>,
    @size(16)  size: vec2<f32>,
    @size(16)  seconds: f32,
};

@group(0)
@binding(0)
var<uniform> global: Global;

struct VertexInput {
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) position: vec3<f32>,
    @location(1) color: u32,
    @location(2) camera_view: u32,
};

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    } else {
        return pow((c + 0.055) / 1.055, 2.4);
    }
}

fn unpack_color(color: u32) -> vec4<f32> {
    return vec4<f32>(
        srgb_to_linear(f32((color & 0xff0000u) >> 16u) / 255.0),
        srgb_to_linear(f32((color & 0xff00u) >> 8u) / 255.0),
        srgb_to_linear(f32((color & 0xffu)) / 255.0),
        f32((color & 0xff000000u) >> 24u) / 255.0,
    );
}

@vertex
fn vertex(
    vertex: VertexInput,
) -> VertexOutput {
    var result: VertexOutput;
    var pos = vertex.position;
    let scale_mat = mat4x4<f32> (
                vec4<f32>(global.scales[vertex.camera_view].elem, 0.0, 0.0, 0.0),
                vec4<f32>(0.0, global.scales[vertex.camera_view].elem, 0.0, 0.0),
                vec4<f32>(0.0, 0.0, 1.0, 0.0),
                vec4<f32>(0.0, 0.0, 0.0, 1.0),
            );

    result.clip_position = (global.proj * global.views[vertex.camera_view] * scale_mat) * vec4<f32>(pos, 1.0);

    result.color = unpack_color(vertex.color);
    return result;
}

@fragment
fn fragment(vertex: VertexOutput,) -> @location(0) vec4<f32> {
    return vertex.color;
}