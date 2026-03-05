use crate::{BufferData, BufferLayout};
#[cfg(feature = "rayon")]
use rayon::{iter::repeat_n, prelude::*};
#[cfg(not(feature = "rayon"))]
use std::iter::repeat_n;
/// Vertex Details for [`crate::Map`] that matches the Shaders Vertex Layout.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileVertex {
    pub pos: [f32; 3],
    pub tile_id: u32,
    pub texture_layer: u32,
    pub color: u32,
    pub map_layer: u32,
    pub map_index: u32,
    pub anim_time: u32,
}

impl Default for TileVertex {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            tile_id: 0,
            texture_layer: 0,
            color: 0,
            map_layer: 0,
            map_index: 0,
            anim_time: 0,
        }
    }
}

impl BufferLayout for TileVertex {
    fn attributes() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![1 => Float32x3, 2 =>  Uint32, 3 => Uint32, 4 => Uint32, 5 => Uint32, 6 => Uint32, 7 => Uint32]
            .to_vec()
    }

    fn default_buffer() -> BufferData {
        Self::with_capacity(82_944, 0)
    }

    fn with_capacity(
        vertex_capacity: usize,
        _index_capacity: usize,
    ) -> BufferData {
        let instance_arr: Vec<TileVertex> =
            repeat_n(TileVertex::default(), vertex_capacity).collect();

        BufferData {
            vertexs: bytemuck::cast_slice(&instance_arr).to_vec(),
            ..Default::default()
        }
    }

    fn stride() -> usize {
        std::mem::size_of::<[f32; 9]>()
    }
}
