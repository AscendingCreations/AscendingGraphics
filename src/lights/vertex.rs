use crate::{BufferData, BufferLayout};
#[cfg(feature = "rayon")]
use rayon::{iter::repeat_n, prelude::*};
#[cfg(not(feature = "rayon"))]
use std::iter::repeat_n;
/// Vertex Details for [`crate::Lights`] that matches the Shaders Vertex Layout.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightsVertex {
    pub world_color: [f32; 4],
    pub enable_lights: u32,
    pub dir_count: u32,
    pub area_count: u32,
    pub pos: [f32; 3],
    pub size: [f32; 2],
}

impl Default for LightsVertex {
    fn default() -> Self {
        Self {
            world_color: [0.0; 4],
            enable_lights: 0,
            dir_count: 0,
            area_count: 0,
            pos: [0.0; 3],
            size: [0.0; 2],
        }
    }
}

impl BufferLayout for LightsVertex {
    fn attributes() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![1 => Float32x4, 2 => Uint32, 3 => Uint32, 4 => Uint32, 5 => Float32x3, 6=>Float32x2 ].to_vec()
    }

    fn default_buffer() -> BufferData {
        Self::with_capacity(10_000, 0)
    }

    fn with_capacity(
        vertex_capacity: usize,
        _index_capacity: usize,
    ) -> BufferData {
        let instance_arr: Vec<LightsVertex> =
            repeat_n(LightsVertex::default(), vertex_capacity).collect();

        BufferData {
            vertexs: bytemuck::cast_slice(&instance_arr).to_vec(),
            ..Default::default()
        }
    }

    fn stride() -> usize {
        std::mem::size_of::<[f32; 12]>()
    }
}
