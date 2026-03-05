use crate::{BufferData, BufferLayout};
#[cfg(feature = "rayon")]
use rayon::{iter::repeat_n, prelude::*};
#[cfg(not(feature = "rayon"))]
use std::iter::repeat_n;

/// Vertex Details for [`crate::Rect`] that matches the Shaders Vertex Layout.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RectVertex {
    /// Position on the Screen.
    pub pos: [f32; 3],
    /// Width and Height of the Rect.
    pub size: [f32; 2],
    /// Texture X, Y, W and H if any apply.
    pub uv: [f32; 4],
    /// Color of the Rect.
    pub color: u32,
    /// Width of the Rects Border.
    pub border_width: f32,
    /// Color of the Rects Border.
    pub border_color: u32,
    /// Texture Array Layer if one applies.
    pub layer: u32,
    /// Rectangle Radius.
    pub radius: f32,
    /// Camera Type numberical.
    pub camera_view: u32,
}

impl Default for RectVertex {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            size: [0.0; 2],
            uv: [0.0; 4],
            color: 0,
            border_width: 0.0,
            border_color: 0,
            layer: 0,
            radius: 1.0,
            camera_view: 0,
        }
    }
}

impl BufferLayout for RectVertex {
    fn attributes() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![1 => Float32x3, 2 => Float32x2, 3 => Float32x4, 4 => Uint32, 5 => Float32, 6 => Uint32, 7 => Uint32, 8 => Float32, 9 => Uint32]
            .to_vec()
    }

    // default set as large enough to contain 1_000 shapes.
    fn default_buffer() -> BufferData {
        Self::with_capacity(1_000, 0)
    }

    fn with_capacity(
        vertex_capacity: usize,
        _index_capacity: usize,
    ) -> BufferData {
        let instance_arr: Vec<RectVertex> =
            repeat_n(RectVertex::default(), vertex_capacity).collect();

        BufferData {
            vertexs: bytemuck::cast_slice(&instance_arr).to_vec(),
            ..Default::default()
        }
    }

    fn stride() -> usize {
        std::mem::size_of::<[f32; 15]>()
    }
}
