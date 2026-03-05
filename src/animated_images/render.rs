use crate::{
    AnimImage, AnimImageRenderPipeline, AtlasSet, GpuRenderer, GraphicsError,
    ImageVertex, InstanceBuffer, OrderedIndex, StaticVertexBuffer, System,
};

/// Instance Buffer Setup for [`AnimImage`].
///
#[derive(Debug)]
pub struct AnimImageRenderer {
    pub buffer: InstanceBuffer<ImageVertex>,
}

impl AnimImageRenderer {
    /// Creates a new [`AnimImageRenderer`].
    ///
    pub fn new(renderer: &GpuRenderer) -> Result<Self, GraphicsError> {
        Ok(Self {
            buffer: InstanceBuffer::new(renderer.gpu_device(), 512),
        })
    }

    /// Adds a Buffer [`OrderedIndex`] to the Rendering Store to get processed.
    /// This must be done before [`AnimImageRenderer::finalize`] but after [`AnimImage::update`] in order for it to Render.
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
    /// Must be called after all the [`AnimImageRenderer::add_buffer_store`]'s.
    ///
    pub fn finalize(&mut self, renderer: &mut GpuRenderer) {
        self.buffer.finalize(renderer)
    }

    /// Updates a [`AnimImage`] and adds its [`OrderedIndex`] to staging using [`AnimImageRenderer::add_buffer_store`].
    /// This must be done before [`AnimImageRenderer::finalize`] in order for it to Render.
    ///
    /// # Arguments
    /// - image: [`AnimImage`] we want to update and prepare for rendering.
    /// - atlas: [`AtlasSet`] the [`AnimImage`] needs to render with.
    /// - buffer_layer: The Buffer Layer we want to add this Object too.
    ///
    pub fn update(
        &mut self,
        image: &mut AnimImage,
        renderer: &mut GpuRenderer,
        atlas: &mut AtlasSet,
        buffer_layer: usize,
    ) {
        let index = image.update(renderer, atlas);

        self.add_buffer_store(renderer, index, buffer_layer);
    }

    /// Sets the Instance Buffer to enable Rendering With Scissor Clipping.
    /// This must be Set for the Optional Bounds to be used.
    ///
    pub fn use_clipping(&mut self) {
        self.buffer.set_as_clipped();
    }
}

/// Trait used to Grant Direct [`AnimImage`] Rendering to [`wgpu::RenderPass`]
pub trait RenderAnimImage<'a, 'b, Controls>
where
    'b: 'a,
    Controls: camera::controls::Controls,
{
    /// Renders the all [`AnimImage`]'s within the buffer layer to screen that have been processed and finalized.
    ///
    fn render_animated_image(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b AnimImageRenderer,
        atlas: &'b AtlasSet,
        system: &'b System<Controls>,
        buffer_layer: usize,
    );
}

impl<'a, 'b, Controls> RenderAnimImage<'a, 'b, Controls>
    for wgpu::RenderPass<'a>
where
    'b: 'a,
    Controls: camera::controls::Controls,
{
    fn render_animated_image(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b AnimImageRenderer,
        atlas: &'b AtlasSet,
        system: &'b System<Controls>,
        buffer_layer: usize,
    ) {
        if buffer.buffer.is_clipped() {
            if let Some(details) =
                buffer.buffer.clipped_buffers.get(buffer_layer)
            {
                let mut scissor_is_default = true;

                if buffer.buffer.count() > 0 {
                    self.set_bind_group(1, atlas.bind_group(), &[]);
                    self.set_vertex_buffer(1, buffer.buffer.instances(None));
                    self.set_pipeline(
                        renderer
                            .get_pipelines(AnimImageRenderPipeline)
                            .unwrap(),
                    );
                    for (details, bounds, camera_view) in details {
                        if let Some(bounds) = bounds {
                            let bounds =
                                system.world_to_screen(*camera_view, bounds);

                            self.set_scissor_rect(
                                bounds.x as u32,
                                bounds.y as u32,
                                bounds.z as u32,
                                bounds.w as u32,
                            );
                            scissor_is_default = false;
                        }

                        self.draw_indexed(
                            0..StaticVertexBuffer::index_count(),
                            0,
                            details.start..details.end,
                        );

                        if !scissor_is_default {
                            self.set_scissor_rect(
                                0,
                                0,
                                system.screen_size[0] as u32,
                                system.screen_size[1] as u32,
                            );
                            scissor_is_default = true;
                        };
                    }
                }
            }
        } else if let Some(Some(details)) =
            buffer.buffer.buffers.get(buffer_layer)
            && buffer.buffer.count() > 0
        {
            self.set_bind_group(1, atlas.bind_group(), &[]);
            self.set_vertex_buffer(1, buffer.buffer.instances(None));
            self.set_pipeline(
                renderer.get_pipelines(AnimImageRenderPipeline).unwrap(),
            );

            self.draw_indexed(
                0..StaticVertexBuffer::index_count(),
                0,
                details.start..details.end,
            );
        }
    }
}
