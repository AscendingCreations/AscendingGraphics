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

struct Map {
    pos: vec2<f32>,
    tilesize: f32,
    camera_view: u32,
};

const c_maps: u32 = 500u;

@group(0)
@binding(0)
var<uniform> global: Global;

struct VertexInput {
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) v_pos: vec2<f32>,
    @location(1) position: vec3<f32>,
    @location(2) tile_id: u32,
    @location(3) texture_layer: u32,
    @location(4) color: u32,
    @location(5) map_layer: u32,
    @location(6) map_id: u32,
    @location(7) anim_time: u32,
};

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) uv_layer: i32,
    @location(2) color: vec4<f32>,
};

@group(1)
@binding(0)
var tex: texture_2d_array<f32>;
@group(1)
@binding(1)
var tex_sample: sampler;

@group(2)
@binding(0)
var<uniform> map: array<Map, c_maps>;

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
    let v = vertex.vertex_idx % 4u;
    let size = textureDimensions(tex);
    let fsize = vec2<f32> (f32(size.x), f32(size.y));
    let total_tiles = u32(size.x / u32(map[vertex.map_id].tilesize));
    let tileposx = f32(vertex.tile_id % total_tiles) * map[vertex.map_id].tilesize;
    let tileposy = f32(vertex.tile_id / total_tiles) * map[vertex.map_id].tilesize;

    pos.x += map[vertex.map_id].pos.x;
    pos.y += map[vertex.map_id].pos.y;

    switch v {
        case 1u: {
            result.uv = vec2<f32>(tileposx + map[vertex.map_id].tilesize, tileposy + map[vertex.map_id].tilesize) / fsize;
            pos.x += map[vertex.map_id].tilesize;
        }
        case 2u: {
            result.uv = vec2<f32>(tileposx + map[vertex.map_id].tilesize, tileposy) / fsize;
            pos.x += map[vertex.map_id].tilesize;
            pos.y += map[vertex.map_id].tilesize;
        }
        case 3u: {
            result.uv = vec2<f32>(tileposx, tileposy) / fsize;
            pos.y += map[vertex.map_id].tilesize;
        }
        default: {
            result.uv = vec2<f32>(tileposx, tileposy + map[vertex.map_id].tilesize) / fsize;
        }
    }

    let scale_mat = mat4x4<f32> (
        vec4<f32>(global.scales[map[vertex.map_id].camera_view].elem, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, global.scales[map[vertex.map_id].camera_view].elem, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    ); 

    result.clip_position = (global.proj * global.views[map[vertex.map_id].camera_view] * scale_mat) * vec4<f32>(pos, 1.0);
    result.color = unpack_color(vertex.color);

    let id = global.seconds / (f32(vertex.anim_time) / 1000.0);
    let frame = u32(floor(id % f32(4)));

    if vertex.map_layer == 3 && frame != 0u {
        result.uv = vec2<f32>(0.0, 0.0);
        result.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else if vertex.map_layer == 4 && frame != 1u {
        result.uv = vec2<f32>(0.0, 0.0);
        result.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else if vertex.map_layer == 5 && frame != 2u {
        result.uv = vec2<f32>(0.0, 0.0);
        result.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else if vertex.map_layer == 6 && frame != 3u {
        result.uv = vec2<f32>(0.0, 0.0);
        result.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    result.uv_layer = i32(vertex.texture_layer);
    return result;
}

// Fragment shader
@fragment
fn fragment(vertex: VertexOutput,) -> @location(0) vec4<f32> {
    let object_color = textureSampleLevel(tex, tex_sample, vertex.uv, vertex.uv_layer, 1.0);

    let color = object_color * vertex.color;

    if (color.a <= 0.0) {
        discard;
    }

    return color;
}

