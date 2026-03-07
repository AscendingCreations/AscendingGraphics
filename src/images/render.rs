use crate::{
    AtlasSet, GpuRenderer, GraphicsError, Image, ImageRenderPipeline,
    ImageVertex, InstanceBuffer, OrderedIndex, StaticVertexBuffer,
};

/// Instance Buffer Setup for [`Image`].
///
#[derive(Debug)]
pub struct ImageRenderer {
    pub buffer: InstanceBuffer<ImageVertex>,
}

impl ImageRenderer {
    /// Creates a new [`ImageRenderer`].
    ///
    pub fn new(renderer: &GpuRenderer) -> Result<Self, GraphicsError> {
        Ok(Self {
            buffer: InstanceBuffer::with_capacity(renderer.gpu_device(), 512),
        })
    }

    /// Adds a Buffer [`OrderedIndex`] to the Rendering Store to get processed.
    /// This must be done before [`ImageRenderer::finalize`] but after [`Image::update`] in order for it to Render.
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
    /// Must be called after all the [`ImageRenderer::add_buffer_store`]'s.
    ///
    pub fn finalize(&mut self, renderer: &mut GpuRenderer) {
        self.buffer.finalize(renderer)
    }

    /// Updates a [`Image`] and adds its [`OrderedIndex`] to staging using [`ImageRenderer::add_buffer_store`].
    /// This must be done before [`ImageRenderer::finalize`] in order for it to Render.
    ///
    /// # Arguments
    /// - image: [`Image`] we want to update and prepare for rendering.
    /// - atlas: [`AtlasSet`] the [`Image`] needs to render with.
    /// - buffer_layer: The Buffer Layer we want to add this Object too.
    ///
    pub fn update(
        &mut self,
        image: &mut Image,
        renderer: &mut GpuRenderer,
        atlas: &mut AtlasSet,
        buffer_layer: usize,
    ) {
        let index = image.update(renderer, atlas);

        self.add_buffer_store(renderer, index, buffer_layer);
    }
}

/// Trait used to Grant Direct [`Image`] Rendering to [`wgpu::RenderPass`]
pub trait RenderImage<'a, 'b>
where
    'b: 'a,
{
    /// Renders the all [`Image`]'s within the buffer layer to screen that have been processed and finalized.
    ///
    fn render_image(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b ImageRenderer,
        atlas: &'b AtlasSet,
        buffer_layer: usize,
    );
}

impl<'a, 'b> RenderImage<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn render_image(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b ImageRenderer,
        atlas: &'b AtlasSet,
        buffer_layer: usize,
    ) {
        if let Some(Some(details)) = buffer.buffer.buffers.get(buffer_layer)
            && buffer.buffer.count() > 0
        {
            self.set_bind_group(1, atlas.bind_group(), &[]);
            self.set_vertex_buffer(1, buffer.buffer.instances(None));
            self.set_pipeline(
                renderer.get_pipelines(ImageRenderPipeline).unwrap(),
            );

            self.draw_indexed(
                0..StaticVertexBuffer::index_count(),
                0,
                details.start..details.end,
            );
        }
    }
}
