use crate::{BufferData, BufferLayout};
use cosmic_text::Color;
use lyon::{math::Point as LPoint, tessellation as tess};
#[cfg(feature = "rayon")]
use rayon::{iter::repeat_n, prelude::*};
#[cfg(not(feature = "rayon"))]
use std::iter::repeat_n;

/// Vertex Details for [`crate::Mesh2D`] that matches the Shaders Vertex Layout.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Mesh2DVertex {
    pub pos: [f32; 3],
    pub color: u32,
    pub camera: u32,
}

impl Default for Mesh2DVertex {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            color: 0,
            camera: 0,
        }
    }
}

impl BufferLayout for Mesh2DVertex {
    fn attributes() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32, 2 => Uint32]
            .to_vec()
    }

    fn default_buffer() -> BufferData {
        Self::with_capacity(1_000, 6_000)
    }

    fn with_capacity(
        vertex_capacity: usize,
        index_capacity: usize,
    ) -> BufferData {
        let vbo_arr: Vec<Mesh2DVertex> =
            repeat_n(Mesh2DVertex::default(), vertex_capacity).collect();

        let mut indices: Vec<u32> = Vec::with_capacity(index_capacity * 6);
        (0..index_capacity as u32).for_each(|_| {
            indices.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        });

        BufferData {
            vertexs: bytemuck::cast_slice(&vbo_arr).to_vec(),
            indexs: bytemuck::cast_slice(&indices).to_vec(),
        }
    }

    fn stride() -> usize {
        std::mem::size_of::<[f32; 5]>()
    }
}

/// Vertex Builder details [`crate::Mesh2DBuilder`].
///
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct VertexBuilder {
    pub z: f32,
    pub color: Color,
    pub camera: u32,
}

impl VertexBuilder {
    pub fn new_vertex(self, pos: LPoint) -> Mesh2DVertex {
        Mesh2DVertex {
            pos: [pos.x, pos.y, self.z],
            color: self.color.0,
            camera: self.camera,
        }
    }
}

impl tess::StrokeVertexConstructor<Mesh2DVertex> for VertexBuilder {
    fn new_vertex(&mut self, vertex: tess::StrokeVertex) -> Mesh2DVertex {
        let position = vertex.position();
        Mesh2DVertex {
            pos: [position.x, position.y, self.z],
            color: self.color.0,
            camera: self.camera,
        }
    }
}

impl tess::FillVertexConstructor<Mesh2DVertex> for VertexBuilder {
    fn new_vertex(&mut self, vertex: tess::FillVertex) -> Mesh2DVertex {
        let position = vertex.position();
        Mesh2DVertex {
            pos: [position.x, position.y, self.z],
            color: self.color.0,
            camera: self.camera,
        }
    }
}
