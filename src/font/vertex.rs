use crate::{BufferData, BufferLayout};

#[cfg(feature = "rayon")]
use rayon::{iter::repeat_n, prelude::*};
#[cfg(not(feature = "rayon"))]
use std::iter::repeat_n;

/// Vertex Details for [`crate::Text`] that matches the Shaders Vertex Layout.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextVertex {
    pub pos: [f32; 3],
    pub size: [f32; 2],
    pub tex_coord: [f32; 2],
    pub layer: u32,
    pub color: u32,
    pub camera_view: u32,
    pub is_color: u32,
}

impl Default for TextVertex {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0, 1.0],
            size: [0.0; 2],
            tex_coord: [0.0; 2],
            layer: 0,
            color: 0,
            camera_view: 0,
            is_color: 0,
        }
    }
}

impl BufferLayout for TextVertex {
    fn attributes() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![1 => Float32x3, 2 => Float32x2, 3 => Float32x2, 4 => Uint32, 5 => Uint32, 6 => Uint32, 7 => Uint32]
            .to_vec()
    }

    fn default_buffer() -> BufferData {
        Self::with_capacity(4096, 0)
    }

    fn with_capacity(
        vertex_capacity: usize,
        _index_capacity: usize,
    ) -> BufferData {
        let instance_arr: Vec<TextVertex> =
            repeat_n(TextVertex::default(), vertex_capacity).collect();

        BufferData {
            vertexs: bytemuck::cast_slice(&instance_arr).to_vec(),
            ..Default::default()
        }
    }

    fn stride() -> usize {
        std::mem::size_of::<[f32; 11]>()
    }
}
