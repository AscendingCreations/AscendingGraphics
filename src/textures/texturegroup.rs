use wgpu::BindGroupLayout;

use crate::GpuRenderer;

/// [`crate::AtlasSet`] rendering TextureGroup
///
#[derive(Debug)]
pub struct TextureGroup {
    /// Texture's [`wgpu::TextureView`] for WGPU.
    pub texture_view: wgpu::TextureView,
    /// Textures WGPU [`wgpu::BindGroup`].
    pub bind_group: wgpu::BindGroup,
}

impl TextureGroup {
    pub fn from_view(
        renderer: &GpuRenderer,
        texture_view: wgpu::TextureView,
        layout: &BindGroupLayout,
    ) -> Self {
        let diffuse_sampler =
            renderer.device().create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Texture_sampler"),
                lod_max_clamp: 0.0,
                ..Default::default()
            });

        Self {
            bind_group: renderer.device().create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Texture Bind Group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &texture_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &diffuse_sampler,
                            ),
                        },
                    ],
                },
            ),
            texture_view,
        }
    }
}
