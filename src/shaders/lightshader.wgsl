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

struct AreaLights {
    pos: vec2<f32>,
    color: u32,
    max_distance: f32,
    anim_speed: f32,
    dither: f32,
    animate: u32,
    camera_view: u32,
};

struct DirLights {
    pos: vec2<f32>,
    color: u32,
    max_distance: f32,
    max_width: f32,
    anim_speed: f32,
    angle: f32,
    dither: f32,
    fade_distance: f32,
    edge_fade_distance: f32,
    animate: u32,
    camera_view: u32,
};

@group(0)
@binding(0)
var<uniform> global: Global;

struct VertexInput {
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) v_pos: vec2<f32>,
    @location(1) world_color: vec4<f32>,
    @location(2) enable_lights: u32,
    @location(3) dir_count: u32,
    @location(4) area_count: u32,
    @location(5) pos: vec3<f32>,
    @location(6) size: vec2<f32>,
};

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec4<f32>,
    @location(1) col: vec4<f32>,
    @location(2) enable_lights: u32,
    @location(3) dir_count: u32,
    @location(4) area_count: u32,
};

const c_area_lights: u32 = 2000u;
const c_dir_lights: u32 = 1333u;

@group(1)
@binding(0)
var<uniform> u_areas: array<AreaLights, c_area_lights>;
@group(2)
@binding(0)
var<uniform> u_dirs: array<DirLights, c_dir_lights>;

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

    switch v {
        case 1u: {
            result.clip_position = global.proj * vec4<f32>(vertex.pos.x + vertex.size.x, vertex.pos.y, vertex.pos.z, 1.0);
        }
        case 2u: {
            result.clip_position = global.proj * vec4<f32>(vertex.pos.x + vertex.size.x, vertex.pos.y + vertex.size.y, vertex.pos.z, 1.0);
        }
        case 3u: {
            result.clip_position = global.proj * vec4<f32>(vertex.pos.x, vertex.pos.y + vertex.size.y, vertex.pos.z, 1.0);
        }
        default: {
            result.clip_position = global.proj * vec4<f32>(vertex.pos.x, vertex.pos.y, vertex.pos.z, 1.0);
        }
    }

    result.tex_coords = global.inverse_proj * result.clip_position;
    result.tex_coords = result.tex_coords / result.tex_coords.w;
    result.col = vec4<f32>(srgb_to_linear(vertex.world_color.r), srgb_to_linear(vertex.world_color.g), srgb_to_linear(vertex.world_color.b), vertex.world_color.a);
    result.enable_lights = vertex.enable_lights;
    result.dir_count = vertex.dir_count;
    result.area_count = vertex.area_count;
    return result;
}

const pi: f32 = 3.14159265;
const two_pi: f32 = 6.2831853;

fn flash_light(light_pos: vec2<f32>, pixel_pos: vec2<f32>, angle: f32, w_angle: f32, range: f32) -> f32 {
    let d = distance(light_pos, pixel_pos);
    let degree_radian = radians(angle + (w_angle / 2.0));
    let w_radian = clamp(radians(w_angle), 0.0, 2.0 * pi);
    // Calculate the start angle from the direction angle and the angle width of the "cone".
    let s_angle = degree_radian - (w_radian / 2.0);
    // Calculate the start vector.
    let s = vec2<f32>(cos(s_angle), sin(s_angle));
    // Calculate the direction between the pixel position and the light position and normalize the vector.
    let direction = normalize(pixel_pos - light_pos);
    let p_pos = vec3<f32>(pixel_pos.x, pixel_pos.y, 0.0);
    let l_pos = vec3<f32>(light_pos.x, light_pos.y, 0.0);
    let d_dir = vec3<f32>(direction.x, direction.y, 0.0);

    // Only emit light if the direction projected onto the start vector is within the angle width of our cone. 1.0 - (d / range)
    return select(1.0 - (d / range), 0.0, atan2(length(cross(vec3(direction, 0.0), vec3(s, 0.0))), dot(direction, s)) >= w_radian);
}

// Fragment shader
@fragment
fn fragment(vertex: VertexOutput,) -> @location(0) vec4<f32> {
    var col = vertex.col;

    if (vertex.enable_lights > 0u) {
        for(var i = 0u; i < min(vertex.area_count, c_area_lights); i += 1u) {
            let light = u_areas[i];
            let light_pos = vec3<f32>(light.pos.x, light.pos.y, 1.0);
            let scale_mat = mat4x4<f32> (
                vec4<f32>(global.scales[light.camera_view].elem, 0.0, 0.0, 0.0),
                vec4<f32>(0.0, global.scales[light.camera_view].elem, 0.0, 0.0),
                vec4<f32>(0.0, 0.0, 1.0, 0.0),
                vec4<f32>(0.0, 0.0, 0.0, 1.0),
            );
            let pos = (global.views[light.camera_view] * scale_mat) * vec4<f32>(light_pos, 1.0);
            var max_distance = light.max_distance * global.scales[light.camera_view].elem;

            let light_color = unpack_color(light.color);
            max_distance = max_distance - (f32(light.animate) *(1.0 * sin(global.seconds * light.anim_speed)));
            let dist = distance(pos.xy, vertex.tex_coords.xy);
            let cutoff = max(0.1, max_distance);
            let value = 1.0 - (dist / cutoff);
            var color2 = col; 
            let alpha = select(color2.a, mix(color2.a, light_color.a, value), dist <= cutoff);
            color2.a = alpha;
            col = select(color2, mix(color2, light_color, vec4<f32>(value)), dist <= cutoff);
        }

        for(var i = 0u; i < min(vertex.dir_count, c_dir_lights); i += 1u) {
            let light = u_dirs[i];
            let light_pos = vec3<f32>(light.pos.x, light.pos.y, 1.0);
            let scale_mat = mat4x4<f32> (
                vec4<f32>(global.scales[light.camera_view].elem, 0.0, 0.0, 0.0),
                vec4<f32>(0.0, global.scales[light.camera_view].elem, 0.0, 0.0),
                vec4<f32>(0.0, 0.0, 1.0, 0.0),
                vec4<f32>(0.0, 0.0, 0.0, 1.0),
            );
            let pos = (global.views[light.camera_view] * scale_mat) * vec4<f32>(light_pos, 1.0);
            var max_distance = light.max_distance * global.scales[light.camera_view].elem;
            var max_width = light.max_width * global.scales[light.camera_view].elem;
            let fade_distance = light.fade_distance * global.scales[light.camera_view].elem;
            let edge_fade_distance = light.edge_fade_distance * global.scales[light.camera_view].elem;

            let light_color = unpack_color(light.color);
            max_distance = max_distance - (f32(light.animate) *(1.0 * sin(global.seconds * light.anim_speed)));
            let dist_cutoff = max(0.1, max_distance);
            max_width = max_width - (f32(light.animate) *(1.0 * sin(global.seconds * light.anim_speed)));
            let width_cutoff = max(0.1, max_width);
            let value = flash_light(pos.xy, vertex.tex_coords.xy, light.angle, width_cutoff, dist_cutoff);
            var color2 = col; 
            let d = distance(pos.xy, vertex.tex_coords.xy);
            let alpha = select(color2.a, mix(color2.a, light_color.a, value),  d <= dist_cutoff);
            color2.a = alpha;
            col = select(color2, mix(color2, light_color, vec4<f32>(value)),  d <= dist_cutoff);
        }
    } 

    if (col.a <= 0.0) {
        discard;
    }

    return col;
}