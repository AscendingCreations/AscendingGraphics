use crate::{GpuDevice, Layout};
use bytemuck::{Pod, Zeroable};

///Current Max uniform Array size in wgpu shader.
pub const MAX_MAPS: usize = 500;

/// Uniform Details for [crate::Map`] that matches the Shaders Uniform Layout.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MapRaw {
    pub pos: [f32; 2],
    pub tilesize: f32,
    pub camera_view: u32,
}

/// Uniform Layout for [crate::Map`] base shared Data.
///
#[repr(C)]
#[derive(Clone, Copy, Hash, Pod, Zeroable)]
pub struct MapLayout;

impl Layout for MapLayout {
    fn create_layout(
        &self,
        gpu_device: &mut GpuDevice,
    ) -> wgpu::BindGroupLayout {
        gpu_device.device().create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("map_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            },
        )
    }
}
