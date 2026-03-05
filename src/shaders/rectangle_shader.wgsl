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
    @location(1) position: vec3<f32>,
    @location(2) size: vec2<f32>,
    @location(3) uv: vec4<f32>,
    @location(4) color: u32,
    @location(5) border_width: f32,
    @location(6) border_color: u32,
    @location(7) layer: u32,
    @location(8) radius: f32,
    @location(9) camera_view: u32,
};

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(3) container_data: vec4<f32>,
    @location(4) color: vec4<f32>,
    @location(5) border_color: vec4<f32>,
    @location(6) size: vec2<f32>,
    @location(7) border_width: f32,
    @location(8) radius: f32,
    @location(9) layer: i32,
    @location(10) tex_size: vec2<f32>,
};

@group(1)
@binding(0)
var tex: texture_2d_array<f32>;
@group(1)
@binding(1)
var tex_sample: sampler;

fn unpack_tex_data(data: vec2<u32>) -> vec4<u32> {
    return vec4<u32>(
        u32(data[0] & 0xffffu), 
        u32((data[0] & 0xffff0000u) >> 16u),
        u32(data[1] & 0xffffu),
        u32((data[1] & 0xffff0000u) >> 16u)
    );
}

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
    let v = vertex.vertex_idx % 4u;
    let tex_data = vertex.uv;
    let size = textureDimensions(tex);
    let fsize = vec2<f32> (f32(size.x), f32(size.y));
    var pos = vertex.position;

     switch v {
        case 1u: {
            result.uv = vec2<f32>(tex_data[2], tex_data[3]);
            pos.x += vertex.size.x;
        }
        case 2u: {
            result.uv = vec2<f32>(tex_data[2], 0.0);
            pos.x += vertex.size.x;
            pos.y += vertex.size.y;
        }
        case 3u: {
            result.uv = vec2<f32>(0.0, 0.0);
            pos.y += vertex.size.y;
        }
        default: {
            result.uv = vec2<f32>(0.0, tex_data[3]);
        }
    }

    let scale_mat = mat4x4<f32> (
        vec4<f32>(global.scales[vertex.camera_view].elem, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, global.scales[vertex.camera_view].elem, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    result.clip_position = (global.proj * global.views[vertex.camera_view] * scale_mat) * vec4<f32>(pos, 1.0);
    result.size = vertex.size * global.scales[vertex.camera_view].elem;
    result.position = ((global.views[vertex.camera_view] * scale_mat) * vec4<f32>(vertex.position.xy, 1.0, 1.0)).xy;
    result.container_data = tex_data;
    result.border_width = vertex.border_width;
    result.radius = vertex.radius;
    result.tex_size = fsize;
    result.layer = i32(vertex.layer);
    result.color = unpack_color(vertex.color);
    result.border_color = unpack_color(vertex.border_color);
    return result;
}

fn distance_alg(
    frag_coord: vec2<f32>,
    position: vec2<f32>,
    size: vec2<f32>,
    radius: f32
) -> f32 {
    var inner_size: vec2<f32> = size - vec2<f32>(radius, radius) * 2.0;
    var top_left: vec2<f32> = position + vec2<f32>(radius, radius);
    var bottom_right: vec2<f32> = top_left + inner_size;

    var top_left_distance: vec2<f32> =  top_left - frag_coord;
    var bottom_right_distance: vec2<f32> = frag_coord - bottom_right;

    var dist: vec2<f32> = vec2<f32>(
        max(max(top_left_distance.x, bottom_right_distance.x), 0.0),
        max(max(top_left_distance.y, bottom_right_distance.y), 0.0)
    );

    return sqrt(dist.x * dist.x + dist.y * dist.y);
}

@fragment
fn fragment(vertex: VertexOutput,) -> @location(0) vec4<f32> {
    let coords = (vertex.container_data.xy + vertex.uv.xy) / vertex.tex_size;

    let c1 = select(
        vec4<f32>(0.0), 
        textureSampleLevel(tex, tex_sample, coords, vertex.layer, 1.0),
        vertex.container_data[2] > 0.0 && vertex.container_data[3] > 0.0
    );
    let container_color = select(vertex.color, c1  * vertex.color, vertex.container_data[2] > 0.0 && vertex.container_data[3] > 0.0);
    let radius = vertex.radius;
    let clippy = vec2<f32>(vertex.clip_position.x, global.size.y - vertex.clip_position.y);
    let border: f32 = max(radius - vertex.border_width, 0.0);
    let distance = distance_alg( 
            clippy, 
            vertex.position.xy + vec2<f32>(vertex.border_width), 
            vertex.size - vec2<f32>(vertex.border_width * 2.0), 
            border 
        );
    let border_mix: f32 = smoothstep(
            max(border - 0.5, 0.0),
            border + 0.5,
            distance
        );
    let mixed_color: vec4<f32> = select(
        container_color,
        mix(container_color, vertex.border_color, vec4<f32>(border_mix)), 
        vertex.border_width > 0.0
    );
    let dist: f32 = distance_alg(
        clippy,
        vertex.position.xy,
        vertex.size,
        radius
    );
    let radius_alpha: f32 = 1.0 - smoothstep(
        max(radius - 0.5, 0.0),
        radius + 0.5,
        dist);
    let alpha = mixed_color.a * radius_alpha;

    if (alpha <= 0.0) {
        discard;
    }

    return vec4<f32>(mixed_color.r, mixed_color.g, mixed_color.b, alpha);
}