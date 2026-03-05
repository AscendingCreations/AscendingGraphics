use crate::{
    AsBufferPass, AtlasSet, GpuRenderer, GraphicsError, InstanceBuffer,
    MAX_MAPS, Map, MapLayout, MapRaw, MapRenderPipeline, OrderedIndex,
    SetBuffers, StaticVertexBuffer, TileVertex,
};
use std::{collections::VecDeque, iter, mem};
use wgpu::util::{DeviceExt, align_to};

/// Instance Buffer Setup for [`Map`]'s.
///
#[derive(Debug)]
pub struct MapRenderer {
    /// Instance Buffer holding all Rendering information for [`Map`]'s.
    pub buffer: InstanceBuffer<TileVertex>,
    /// Stores each unused buffer ID to be pulled into a map_index_buffer for the map ID.
    pub unused_indexs: VecDeque<usize>,
    /// Uniform buffer for the 500 count array of [`crate::Map`]'s base shared data.
    pub(crate) map_buffer: wgpu::Buffer,
    /// Uniform buffer BindGroup for the 500 count array of [`crate::Map`]'s base shared data.
    map_bind_group: wgpu::BindGroup,
}

impl MapRenderer {
    /// Creates a new [`MapRenderer`].
    ///
    /// # Arguments
    /// - map_count: The number of Maps to presize the instance buffer by.
    ///
    pub fn new(
        renderer: &mut GpuRenderer,
        map_count: u32,
    ) -> Result<Self, GraphicsError> {
        let map_alignment: usize =
            align_to(mem::size_of::<MapRaw>(), 16) as usize;

        let maps: Vec<u8> =
            iter::repeat_n(0u8, MAX_MAPS * map_alignment).collect();

        let map_buffer = renderer.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("map uniform buffer"),
                contents: &maps, //500
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Create the bind group layout for the map
        let layout = renderer.create_layout(MapLayout);

        // Create the bind group.
        let map_bind_group =
            renderer
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: map_buffer.as_entire_binding(),
                    }],
                    label: Some("map_bind_group"),
                });

        let mut unused_indexs = VecDeque::with_capacity(MAX_MAPS);

        for i in 0..MAX_MAPS {
            unused_indexs.push_back(i);
        }

        Ok(Self {
            buffer: InstanceBuffer::with_capacity(
                renderer.gpu_device(),
                9_216 * map_count as usize,
                144,
            ),
            map_buffer,
            map_bind_group,
            unused_indexs,
        })
    }

    /// Adds a Buffer [`OrderedIndex`] to the Rendering Store to get processed.
    /// This must be done before [`MapRenderer::finalize`] but after [`Map::update`] in order for it to Render.
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
    /// Must be called after all the [`MapRenderer::add_buffer_store`]'s.
    ///
    pub fn finalize(&mut self, renderer: &mut GpuRenderer) {
        self.buffer.finalize(renderer);
    }

    /// Updates a [`Map`] and adds its [`OrderedIndex`]'s to staging using [`MapRenderer::add_buffer_store`].
    /// This must be done before [`MapRenderer::finalize`] in order for it to Render.
    ///
    /// # Arguments
    /// - map: [`Map`] we want to update and prepare for rendering.
    /// - atlas: [`AtlasSet`] the [`Map`] needs to render with.
    /// - buffer_layers: The Buffer Layer's we want to add this Object too.
    ///
    pub fn update(
        &mut self,
        map: &mut Map,
        renderer: &mut GpuRenderer,
        atlas: &mut AtlasSet,
        buffer_layers: [usize; 2],
    ) {
        if let Some((bottom, upper)) = map.update(renderer, atlas, self) {
            self.add_buffer_store(renderer, bottom, buffer_layers[0]);
            self.add_buffer_store(renderer, upper, buffer_layers[1]);
        }
    }

    /// Returns a reference too [`wgpu::BindGroup`].
    ///
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.map_bind_group
    }
}

/// Trait used to Grant Direct [`Map`] Rendering to [`wgpu::RenderPass`]
pub trait RenderMap<'a, 'b>
where
    'b: 'a,
{
    /// Renders the all [`Map`]'s within the buffer layer to screen that have been processed and finalized.
    ///
    fn render_map(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b MapRenderer,
        atlas: &'b AtlasSet,
        buffer_layer: usize,
    );
}

impl<'a, 'b> RenderMap<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn render_map(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b MapRenderer,
        atlas: &'b AtlasSet,
        buffer_layer: usize,
    ) {
        if let Some(Some(details)) = buffer.buffer.buffers.get(buffer_layer)
            && buffer.buffer.count() > 0
        {
            self.set_buffers(renderer.buffer_object.as_buffer_pass());
            self.set_bind_group(1, atlas.bind_group(), &[]);
            self.set_bind_group(2, &buffer.map_bind_group, &[]);
            self.set_vertex_buffer(1, buffer.buffer.instances(None));
            self.set_pipeline(
                renderer.get_pipelines(MapRenderPipeline).unwrap(),
            );
            self.draw_indexed(
                0..StaticVertexBuffer::index_count(),
                0,
                details.start..details.end,
            );
        }
    }
}
