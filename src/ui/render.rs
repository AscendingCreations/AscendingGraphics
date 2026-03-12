use crate::{
    AtlasSet, GpuRenderer, GraphicsError, InstanceBuffer, Rect,
    RectRenderPipeline, RectVertex, StaticVertexBuffer,
    instance_buffer::OrderedIndex,
};

/// Instance Buffer Setup for [`Rect`]'s.
///
#[derive(Debug)]
pub struct RectRenderer {
    /// Instance Buffer holding all Rendering information for [`Rect`]'s.
    pub buffer: InstanceBuffer<RectVertex>,
}

impl RectRenderer {
    /// Creates a new [`RectRenderer`].
    ///
    pub fn new(renderer: &GpuRenderer) -> Result<Self, GraphicsError> {
        Ok(Self {
            buffer: InstanceBuffer::with_capacity(renderer.gpu_device(), 512),
        })
    }

    /// Adds a Buffer [`OrderedIndex`] to the Rendering Store to get processed.
    /// This must be done before [`RectRenderer::finalize`] but after [`Rect::update`] in order for it to Render.
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
    /// Must be called after all the [`RectRenderer::add_buffer_store`]'s.
    ///
    pub fn finalize(&mut self, renderer: &mut GpuRenderer) {
        self.buffer.finalize(renderer)
    }

    /// Updates a [`Rect`] and adds its [`OrderedIndex`] to staging using [`RectRenderer::add_buffer_store`].
    /// This must be done before [`RectRenderer::finalize`] in order for it to Render.
    ///
    /// # Arguments
    /// - rect: [`Rect`] we want to update and prepare for rendering.
    /// - atlas: [`AtlasSet`] the [`Rect`] needs to render with.
    /// - buffer_layer: The Buffer Layer we want to add this Object too.
    ///
    pub fn update(
        &mut self,
        rect: &mut Rect,
        renderer: &mut GpuRenderer,
        atlas: &mut AtlasSet,
        buffer_layer: usize,
    ) {
        let index = rect.update(renderer, atlas);

        self.add_buffer_store(renderer, index, buffer_layer);
    }
}

/// Trait used to Grant Direct [`Rect`] Rendering to [`wgpu::RenderPass`]
pub trait RenderRects<'a, 'b>
where
    'b: 'a,
{
    /// Renders the all [`Rect`]'s within the buffer layer to screen that have been processed and finalized.
    ///
    fn render_rects(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b RectRenderer,
        atlas: &'b AtlasSet,
        buffer_layer: usize,
    );
}

impl<'a, 'b> RenderRects<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn render_rects(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b RectRenderer,
        atlas: &'b AtlasSet,
        buffer_layer: usize,
    ) {
        if let Some(Some(details)) = buffer.buffer.buffers.get(buffer_layer)
            && buffer.buffer.count() > 0
        {
            self.set_bind_group(1, &atlas.texture_group.bind_group, &[]);
            self.set_vertex_buffer(1, buffer.buffer.instances(None));
            self.set_pipeline(
                renderer.get_pipelines(RectRenderPipeline).unwrap(),
            );

            self.draw_indexed(
                0..StaticVertexBuffer::index_count(),
                0,
                details.start..details.end,
            );
        }
    }
}
