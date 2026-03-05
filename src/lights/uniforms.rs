use crate::{GpuDevice, Layout};
use bytemuck::{Pod, Zeroable};

/// Uniform Details for [crate::AreaLight`] that matches the Shaders Uniform Layout.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AreaLightRaw {
    pub pos: [f32; 2],
    pub color: u32,
    pub max_distance: f32,
    pub anim_speed: f32,
    pub dither: f32,
    pub animate: u32,
    pub camera_view: u32,
}

/// Uniform Details for [crate::DirectionalLight`] that matches the Shaders Uniform Layout.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectionalLightRaw {
    pub pos: [f32; 2],
    pub color: u32,
    pub max_distance: f32,
    pub max_width: f32,
    pub anim_speed: f32,
    pub angle: f32,
    pub dither: f32,
    pub fade_distance: f32,
    pub edge_fade_distance: f32,
    pub animate: u32,
    pub camera_view: u32,
}

/// Uniform Layout for [crate::AreaLight`].
///
#[repr(C)]
#[derive(Clone, Copy, Hash, Pod, Zeroable)]
pub struct AreaLightLayout;

impl Layout for AreaLightLayout {
    fn create_layout(
        &self,
        gpu_device: &mut GpuDevice,
    ) -> wgpu::BindGroupLayout {
        gpu_device.device().create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("area_light_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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

/// Uniform Layout for [crate::DirectionalLight`].
///
#[repr(C)]
#[derive(Clone, Copy, Hash, Pod, Zeroable)]
pub struct DirLightLayout;

impl Layout for DirLightLayout {
    fn create_layout(
        &self,
        gpu_device: &mut GpuDevice,
    ) -> wgpu::BindGroupLayout {
        gpu_device.device().create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("dir_light_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
