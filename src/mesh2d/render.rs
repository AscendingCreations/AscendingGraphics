use crate::{
    AsBufferPass, GpuRenderer, GraphicsError, Mesh2D, Mesh2DRenderPipeline,
    Mesh2DVertex, OrderedIndex, SetBuffers, System, VertexBuffer,
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
            vbos: VertexBuffer::new(renderer.gpu_device(), 512),
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

    /// Sets the Instance Buffer to enable Rendering With Scissor Clipping.
    /// This must be Set for the Optional Bounds to be used.
    ///
    pub fn use_clipping(&mut self) {
        self.vbos.set_as_clipped();
    }
}

/// Trait used to Grant Direct [`Mesh2D`] Rendering to [`wgpu::RenderPass`]
pub trait RenderMesh2D<'a, 'b, Controls>
where
    'b: 'a,
    Controls: camera::controls::Controls,
{
    /// Renders the all [`Mesh2D`]'s within the buffer layer to screen that have been processed and finalized.
    ///
    fn render_2dmeshs(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b Mesh2DRenderer,
        system: &'b System<Controls>,
        buffer_layer: usize,
    );
}

impl<'a, 'b, Controls> RenderMesh2D<'a, 'b, Controls> for wgpu::RenderPass<'a>
where
    'b: 'a,
    Controls: camera::controls::Controls,
{
    fn render_2dmeshs(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b Mesh2DRenderer,
        system: &'b System<Controls>,
        buffer_layer: usize,
    ) {
        if let Some(vbos) = buffer.vbos.buffers.get(buffer_layer)
            && !vbos.is_empty()
        {
            self.set_buffers(buffer.vbos.as_buffer_pass());
            self.set_pipeline(
                renderer.get_pipelines(Mesh2DRenderPipeline).unwrap(),
            );

            if buffer.vbos.is_clipped() {
                let mut scissor_is_default = true;

                for (details, bounds, camera_view) in vbos {
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
                    // Indexs can always start at 0 per mesh data.
                    // Base vertex is the Addition to the Index
                    self.draw_indexed(
                        details.indices_start..details.indices_end,
                        details.vertex_base, //i as i32 * details.max,
                        0..1,
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
            } else {
                for (details, _bounds, _camer_type) in vbos {
                    // Indexs can always start at 0 per mesh data.
                    // Base vertex is the Addition to the Index
                    self.draw_indexed(
                        details.indices_start..details.indices_end,
                        details.vertex_base, //i as i32 * details.max,
                        0..1,
                    );
                }
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
