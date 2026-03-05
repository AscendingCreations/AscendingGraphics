struct wrapped_f32 {
  @size(16) elem: f32
}

struct Global {
    views: array<mat4x4<f32>, 8>, //16
    scales: array<wrapped_f32, 8>, //4
    proj: mat4x4<f32>, //16
    inverse_proj: mat4x4<f32>, //16
    eye: vec3<f32>, //16
    @size(16)  size: vec2<f32>, //8
    @size(16)  seconds: f32, //4
};

@group(0)
@binding(0)
var<uniform> global: Global;

struct VertexInput {
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) v_pos: vec2<f32>,
    @location(1) pos: vec3<f32>,
    @location(2) hw: vec2<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) layer: u32,
    @location(5) color: u32,
    @location(6) camera_view: u32,
    @location(7) is_color: u32,
};

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) layer: i32,
    @location(4) is_color: u32,
};

@group(1)
@binding(0)
var tex: texture_2d_array<f32>;
@group(1)
@binding(1)
var tex_sample: sampler;

@group(2)
@binding(0)
var emoji_tex: texture_2d_array<f32>;
@group(2)
@binding(1)
var emoji_tex_sample: sampler;

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
    var pos = vertex.pos;
    var size = vec2<u32>(0u);

    if vertex.is_color == 1u {
        size = textureDimensions(emoji_tex);
    } else {
        size = textureDimensions(tex);
    }

    let fsize = vec2<f32> (f32(size.x), f32(size.y));
    let v = vertex.vertex_idx % 4u;

    switch v {
        case 1u: {
            result.uv = vec2<f32>(vertex.uv.x + vertex.hw.x, vertex.uv.y + vertex.hw.y) / fsize;
            pos.x += vertex.hw.x;
        }
        case 2u: {
            result.uv = vec2<f32>(vertex.uv.x + vertex.hw.x, vertex.uv.y) / fsize;
            pos.x += vertex.hw.x;
            pos.y += vertex.hw.y;
        }
        case 3u: {
            result.uv = vec2<f32>(vertex.uv.x, vertex.uv.y) / fsize;
            pos.y += vertex.hw.y;
        }
        default: {
            result.uv = vec2<f32>(vertex.uv.x, vertex.uv.y + vertex.hw.y) / fsize;
        }
    }

    let scale_mat = mat4x4<f32> (
        vec4<f32>(global.scales[vertex.camera_view].elem, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, global.scales[vertex.camera_view].elem, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    result.clip_position = (global.proj * global.views[vertex.camera_view] * scale_mat) * vec4<f32>(pos.xy, pos.z, 1.0);
    result.layer = i32(vertex.layer);
    result.is_color = vertex.is_color;
    result.color = unpack_color(vertex.color);
    return result;
}

// Fragment shader
@fragment
fn fragment(vertex: VertexOutput,) -> @location(0) vec4<f32> {
     let object_color = select(
        textureSampleLevel(tex, tex_sample, vertex.uv.xy, vertex.layer, 1.0),
        textureSampleLevel(emoji_tex, emoji_tex_sample, vertex.uv.xy, vertex.layer, 1.0),
        vertex.is_color == 1u);
    
    if select(object_color.r, object_color.a, vertex.is_color == 1u) <= 0.0 {
        discard;
    }

    return select(vertex.color.rgba * object_color.r, object_color, vertex.is_color == 1u);
}