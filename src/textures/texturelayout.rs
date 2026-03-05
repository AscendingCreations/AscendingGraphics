use crate::{GpuDevice, Layout};
use bytemuck::{Pod, Zeroable};

/// [`crate::AtlasSet`] rendering Layout
///
#[repr(C)]
#[derive(Clone, Copy, Hash, Pod, Zeroable, Debug)]
pub struct TextureLayout;

impl Layout for TextureLayout {
    fn create_layout(
        &self,
        gpu_device: &mut GpuDevice,
    ) -> wgpu::BindGroupLayout {
        let entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    sample_type: wgpu::TextureSampleType::Float {
                        filterable: true,
                    },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(
                    wgpu::SamplerBindingType::Filtering,
                ),
                count: None,
            },
        ];

        gpu_device.device().create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("texture_bind_group_layout"),
                entries: &entries,
            },
        )
    }
}
