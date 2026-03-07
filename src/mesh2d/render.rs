use crate::{
    AsBufferPass, GpuRenderer, GraphicsError, Mesh2D, Mesh2DRenderPipeline,
    Mesh2DVertex, OrderedIndex, SetBuffers, VertexBuffer,
};

#[derive(Debug)]
pub struct Mesh2DRenderer {
    pub vbos: VertexBuffer<Mesh2DVertex>,
}

impl Mesh2DRenderer {
    /// Creates a new [`Mesh2DRenderer`].
    ///
    pub fn new(renderer: &GpuRenderer) -> Result<Self, GraphicsError> {
        Ok(Self {
            vbos: VertexBuffer::new(renderer.gpu_device()),
        })
    }

    /// Adds a Buffer [`OrderedIndex`] to the Rendering Store to get processed.
    /// This must be done before [`Mesh2DRenderer::finalize`] but after [`Mesh2D::update`] in order for it to Render.
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
        self.vbos.add_buffer_store(renderer, index, buffer_layer);
    }

    /// Finalizes the Buffer by processing staged [`OrderedIndex`]'s and uploading it to the GPU.
    /// Must be called after all the [`Mesh2DRenderer::add_buffer_store`]'s.
    ///
    pub fn finalize(&mut self, renderer: &mut GpuRenderer) {
        self.vbos.finalize(renderer);
    }

    /// Updates a [`Mesh2D`] and adds its [`OrderedIndex`] to staging using [`Mesh2DRenderer::add_buffer_store`].
    /// This must be done before [`Mesh2DRenderer::finalize`] in order for it to Render.
    ///
    /// # Arguments
    /// - mesh: [`Mesh2D`] we want to update and prepare for rendering.
    /// - atlas: [`AtlasSet`] the [`Mesh2D`] needs to render with.
    /// - buffer_layer: The Buffer Layer we want to add this Object too.
    ///
    pub fn update(
        &mut self,
        mesh: &mut Mesh2D,
        renderer: &mut GpuRenderer,
        buffer_layer: usize,
    ) {
        let index = mesh.update(renderer);

        self.add_buffer_store(renderer, index, buffer_layer);
    }
}

/// Trait used to Grant Direct [`Mesh2D`] Rendering to [`wgpu::RenderPass`]
pub trait RenderMesh2D<'a, 'b>
where
    'b: 'a,
{
    /// Renders the all [`Mesh2D`]'s within the buffer layer to screen that have been processed and finalized.
    ///
    fn render_2dmeshs(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b Mesh2DRenderer,
        buffer_layer: usize,
    );
}

impl<'a, 'b> RenderMesh2D<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn render_2dmeshs(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b Mesh2DRenderer,
        buffer_layer: usize,
    ) {
        let vbos = buffer.vbos.get_layer(buffer_layer);

        if !vbos.is_empty() {
            self.set_buffers(buffer.vbos.as_buffer_pass());
            self.set_pipeline(
                renderer.get_pipelines(Mesh2DRenderPipeline).unwrap(),
            );

            for (_layer, details) in vbos {
                // Indexs can always start at 0 per mesh data.
                // Base vertex is the Addition to the Index
                self.draw_indexed(
                    details.indices_start..details.indices_end,
                    details.vertex_base, //i as i32 * details.max,
                    0..1,
                );
            }

            //we need to reset this back for anything else that might need it after mesh is drawn.
            self.set_vertex_buffer(0, renderer.buffer_object.vertices());
            self.set_index_buffer(
                renderer.buffer_object.indices(),
                wgpu::IndexFormat::Uint32,
            );
        }
    }
}
