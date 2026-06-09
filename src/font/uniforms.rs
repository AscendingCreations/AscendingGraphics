use crate::{GpuDevice, Layout};
use bytemuck::{Pod, Zeroable};

///Current Max uniform Array size in wgpu shader.
pub const MAX_UNIFORMED_TEXT: usize = 4000;

/// Uniform Details for [crate::Text`] that matches the Shaders Uniform Layout.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextUniformRaw {
    pub pos: [f32; 2],
    pub padding: f32,
}

/// Uniform Layout for [crate::Text`] base shared Data.
///
#[repr(C)]
#[derive(Clone, Copy, Hash, Pod, Zeroable)]
pub struct TextUniformLayout;

impl Layout for TextUniformLayout {
    fn create_layout(
        &self,
        gpu_device: &mut GpuDevice,
    ) -> wgpu::BindGroupLayout {
        gpu_device.device().create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("text_uniform_bind_group_layout"),
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
