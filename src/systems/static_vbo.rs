use crate::{AsBufferPass, BufferPass, GpuDevice};
use bytemuck::{Pod, Zeroable};
use std::mem;
use wgpu::util::DeviceExt;

/// Preset Vertex layout for reuse.
#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct Vertex {
    _pos: [f32; 2],
}

const INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

const VERTS: [Vertex; 4] = [
    Vertex { _pos: [0.0, 0.0] },
    Vertex { _pos: [1.0, 0.0] },
    Vertex { _pos: [1.0, 1.0] },
    Vertex { _pos: [0.0, 1.0] },
];

/// StaticVertexBuffer holds premade vbo and ibo buffers that can be reused between
/// InstanceBuffered Objects. This Decreases GPU memory and boosts speed.
#[derive(Debug)]
pub struct StaticVertexBuffer {
    pub vbo: wgpu::Buffer,
    pub ibo: wgpu::Buffer,
}

impl<'a> AsBufferPass<'a> for StaticVertexBuffer {
    fn as_buffer_pass(&'a self) -> BufferPass<'a> {
        BufferPass {
            vertex_buffer: &self.vbo,
            index_buffer: &self.ibo,
        }
    }
}

impl StaticVertexBuffer {
    /// Used to create [`wgpu::BufferSlice`] from the static context.
    pub fn create_buffer(gpu_device: &GpuDevice) -> Self {
        Self {
            vbo: gpu_device.device().create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("static vertex buffer"),
                    contents: bytemuck::cast_slice(&VERTS),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ),
            ibo: gpu_device.device().create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("static index buffer"),
                    contents: bytemuck::cast_slice(&INDICES),
                    usage: wgpu::BufferUsages::INDEX,
                },
            ),
        }
    }

    /// Returns the index_count.
    pub fn index_count() -> u32 {
        INDICES.len() as u32
    }

    //VertexBufferLayout for StaticBufferObject VBO.
    pub fn vertex_attribute() -> wgpu::VertexAttribute {
        wgpu::VertexAttribute {
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x2,
            offset: 0,
        }
    }

    /// Returns the stride of the [`StaticVertexBuffer`]
    pub fn stride() -> u64 {
        mem::size_of::<Vertex>() as u64
    }

    /// Returns the [`wgpu::BufferSlice`] of indices.
    pub fn indices(&self) -> wgpu::BufferSlice<'_> {
        self.ibo.slice(..)
    }

    /// creates a new [`StaticVertexBuffer`]
    pub fn new(gpu_device: &GpuDevice) -> Self {
        Self::create_buffer(gpu_device)
    }

    /// Returns the [`wgpu::BufferSlice`] of vertices.
    pub fn vertices(&self) -> wgpu::BufferSlice<'_> {
        self.vbo.slice(..)
    }
}
