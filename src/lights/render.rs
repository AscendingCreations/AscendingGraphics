use crate::{
    AreaLightLayout, AreaLightRaw, DirLightLayout, DirectionalLightRaw,
    GpuRenderer, GraphicsError, InstanceBuffer, LightRenderPipeline, Lights,
    LightsVertex, MAX_AREA_LIGHTS, MAX_DIR_LIGHTS, OrderedIndex,
    StaticVertexBuffer,
};
use std::{iter, mem};
use wgpu::util::{DeviceExt, align_to};

/// Instance Buffer Setup for [`Lights`].
///
#[derive(Debug)]
pub struct LightRenderer {
    /// Instance Buffer holding all Rendering information for [`Lights`].
    pub buffer: InstanceBuffer<LightsVertex>,
    /// Uniform buffer for the array of [`crate::AreaLight`]'s.
    area_buffer: wgpu::Buffer,
    /// Uniform buffer for the array of [`crate::DirectionalLight`]'s.
    dir_buffer: wgpu::Buffer,
    /// Uniform buffer BindGroup for the array of [`crate::AreaLight`]'s.
    area_bind_group: wgpu::BindGroup,
    /// Uniform buffer BindGroup for the array of [`crate::DirectionalLight`]'s.
    dir_bind_group: wgpu::BindGroup,
}

impl LightRenderer {
    /// Creates a new [`LightRenderer`].
    ///
    pub fn new(renderer: &mut GpuRenderer) -> Result<Self, GraphicsError> {
        // The size + Padding == 32.
        let area_alignment: usize =
            align_to(mem::size_of::<AreaLightRaw>(), 32) as usize;
        // The size + Padding == 48.
        let dir_alignment: usize =
            align_to(mem::size_of::<DirectionalLightRaw>(), 48) as usize;

        let area: Vec<u8> =
            iter::repeat_n(0u8, MAX_AREA_LIGHTS * area_alignment).collect();

        let area_buffer = renderer.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Area Light buffer"),
                contents: &area, //2000
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let dirs: Vec<u8> =
            iter::repeat_n(0u8, MAX_DIR_LIGHTS * dir_alignment).collect();

        let dir_buffer = renderer.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Directional Light buffer"),
                contents: &dirs, //2000
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Create the bind group layout for the area lights.
        let layout = renderer.create_layout(AreaLightLayout);

        // Create the bind group.
        let area_bind_group =
            renderer
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: area_buffer.as_entire_binding(),
                    }],
                    label: Some("area_lights_bind_group"),
                });

        // Create the bind group layout for the directional lights.
        let layout = renderer.create_layout(DirLightLayout);

        // Create the bind group.
        let dir_bind_group =
            renderer
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dir_buffer.as_entire_binding(),
                    }],
                    label: Some("dir_lights_bind_group"),
                });

        Ok(Self {
            buffer: InstanceBuffer::new(renderer.gpu_device(), 32),
            dir_buffer,
            area_buffer,
            area_bind_group,
            dir_bind_group,
        })
    }

    /// Adds a Buffer [`OrderedIndex`] to the Rendering Store to get processed.
    /// This must be done before [`LightRenderer::finalize`] but after [`Lights::update`] in order for it to Render.
    ///
    /// # Arguments
    /// - index: The [`OrderedIndex`] of the Object we want to render.
    /// - buffer_layer: The Buffer Layer we want to add this Object too.
    ///
    pub fn add_buffer_store(
        &mut self,
        renderer: &GpuRenderer,
        index: OrderedIndex,
        buffer_layer: usize,
    ) {
        self.buffer.add_buffer_store(renderer, index, buffer_layer);
    }

    /// Finalizes the Buffer by processing staged [`OrderedIndex`]'s and uploading it to the GPU.
    /// Must be called after all the [`LightRenderer::add_buffer_store`]'s.
    ///
    pub fn finalize(&mut self, renderer: &mut GpuRenderer) {
        self.buffer.finalize(renderer)
    }

    /// Updates a [`Lights`] and adds its [`OrderedIndex`] to staging using [`LightRenderer::add_buffer_store`].
    /// This must be done before [`LightRenderer::finalize`] in order for it to Render.
    ///
    /// # Arguments
    /// - lights: [`Lights`] we want to update and prepare for rendering.
    /// - buffer_layer: The Buffer Layer we want to add this Object too.
    ///
    pub fn update(
        &mut self,
        lights: &mut Lights,
        renderer: &mut GpuRenderer,
        buffer_layer: usize,
    ) {
        let index = lights.update(
            renderer,
            &mut self.area_buffer,
            &mut self.dir_buffer,
        );

        self.add_buffer_store(renderer, index, buffer_layer);
    }
}

/// Trait used to Grant Direct [`Lights`] Rendering to [`wgpu::RenderPass`]
pub trait RenderLights<'a, 'b>
where
    'b: 'a,
{
    /// Renders the all [`Lights`]'s within the buffer layer to screen that have been processed and finalized.
    ///
    fn render_lights(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b LightRenderer,
        buffer_layer: usize,
    );
}

impl<'a, 'b> RenderLights<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn render_lights(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b LightRenderer,
        buffer_layer: usize,
    ) {
        if let Some(Some(details)) = buffer.buffer.buffers.get(buffer_layer)
            && buffer.buffer.count() > 0
        {
            self.set_bind_group(1, &buffer.area_bind_group, &[]);
            self.set_bind_group(2, &buffer.dir_bind_group, &[]);
            self.set_vertex_buffer(1, buffer.buffer.instances(None));
            self.set_pipeline(
                renderer.get_pipelines(LightRenderPipeline).unwrap(),
            );

            self.draw_indexed(
                0..StaticVertexBuffer::index_count(),
                0,
                details.start..details.end,
            );
        }
    }
}
