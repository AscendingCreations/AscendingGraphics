use std::{
    collections::VecDeque,
    iter, mem,
    sync::{LazyLock, Mutex},
};

use crate::{
    Allocation, AsBufferPass, AtlasSet, GpuRenderer, GraphicsError,
    InstanceBuffer, MAX_UNIFORMED_TEXT, SetBuffers, StaticVertexBuffer, Text,
    TextRenderPipeline, TextUniformLayout, TextUniformRaw, TextVertex, Vec2,
    instance_buffer::OrderedIndex,
};
use cosmic_text::{CacheKey, SwashCache, SwashImage};
#[cfg(feature = "logging")]
use log::{error, warn};
use wgpu::util::{DeviceExt as _, align_to};

/// [`Text`] text and Emoji AtlasSet holder.
///
pub struct TextAtlas {
    /// AtlasSet holding data from Text only.
    pub(crate) text: AtlasSet<Vec2>,
    /// AtlasSet holding data from Colored Emoji's only.
    pub(crate) emoji: AtlasSet<Vec2>,
}

impl TextAtlas {
    /// Creates a new [`TextAtlas`].
    ///
    /// # Arguments
    /// - size: Used for both Width and Height. Limited to max of limits.max_texture_dimension_2d and min of 256.
    /// - size: Used for both the Text Atlas and Emoji Atlas.
    ///
    pub fn new(
        renderer: &mut GpuRenderer,
        size: u32,
    ) -> Result<Self, GraphicsError> {
        Ok(Self {
            text: AtlasSet::new(
                renderer,
                wgpu::TextureFormat::R8Unorm,
                false,
                size,
            ),
            emoji: AtlasSet::new(
                renderer,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                false,
                size,
            ),
        })
    }

    /// Calles Trim on both internal [`AtlasSet`]'s
    ///
    pub fn trim(&mut self) {
        self.emoji.trim();
        self.text.trim();
    }

    pub fn get_by_key(
        &mut self,
        key: &CacheKey,
    ) -> Option<(Allocation<Vec2>, bool)> {
        if let Some(allocation) = self.text.get_by_key(key) {
            Some((allocation, false))
        } else {
            self.emoji
                .get_by_key(key)
                .map(|allocation| (allocation, true))
        }
    }

    pub fn upload_with_alloc(
        &mut self,
        renderer: &mut GpuRenderer,
        is_color: bool,
        key: CacheKey,
        image: &SwashImage,
    ) -> Result<(Allocation<Vec2>, bool), GraphicsError> {
        if is_color {
            let (_, allocation) = self
                .emoji
                .upload_with_alloc(
                    key,
                    &image.data,
                    image.placement.width,
                    image.placement.height,
                    Vec2::new(
                        image.placement.left as f32,
                        image.placement.top as f32,
                    ),
                    renderer,
                )
                .ok_or(GraphicsError::AtlasFull)?;
            Ok((allocation, is_color))
        } else {
            let (_, allocation) = self
                .text
                .upload_with_alloc(
                    key,
                    &image.data,
                    image.placement.width,
                    image.placement.height,
                    Vec2::new(
                        image.placement.left as f32,
                        image.placement.top as f32,
                    ),
                    renderer,
                )
                .ok_or(GraphicsError::AtlasFull)?;
            Ok((allocation, is_color))
        }
    }
}

/// Instance Buffer Setup for [`Text`].
///
#[derive(Debug)]
pub struct TextRenderer {
    pub buffer: InstanceBuffer<TextVertex>,
    pub swash_cache: SwashCache,
    /// Stores each unused buffer ID to be pulled into a map_index_buffer for the map ID.
    pub unused_indexs: VecDeque<usize>,
    /// Uniform buffer for the 2500 count array of [`crate::Text`]'s base shared data.
    pub(crate) text_buffer: wgpu::Buffer,
    /// Uniform buffer BindGroup for the 2500 count array of [`crate::Text`]'s base shared data.
    text_bind_group: wgpu::BindGroup,
}

impl TextRenderer {
    /// Creates a new [`TextRenderer`].
    ///
    pub fn new(renderer: &mut GpuRenderer) -> Result<Self, GraphicsError> {
        let text_alignment: usize =
            align_to(mem::size_of::<TextUniformRaw>(), 16) as usize;

        let text_uniforms: Vec<u8> =
            iter::repeat_n(0u8, MAX_UNIFORMED_TEXT * text_alignment).collect();

        let text_buffer = renderer.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("text uniform buffer"),
                contents: &text_uniforms, //500
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Create the bind group layout for the map
        let layout = renderer.create_layout(TextUniformLayout);

        // Create the bind group.
        let text_bind_group =
            renderer
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: text_buffer.as_entire_binding(),
                    }],
                    label: Some("map_bind_group"),
                });

        let mut unused_indexs = VecDeque::with_capacity(MAX_UNIFORMED_TEXT);

        for i in 1..MAX_UNIFORMED_TEXT {
            unused_indexs.push_back(i);
        }

        Ok(Self {
            buffer: InstanceBuffer::with_capacity(renderer.gpu_device(), 1024),
            swash_cache: SwashCache::new(),
            text_buffer,
            text_bind_group,
            unused_indexs,
        })
    }

    /// Adds a Buffer [`OrderedIndex`] to the Rendering Store to get processed.
    /// This must be done before [`TextRenderer::finalize`] but after [`Text::update`] in order for it to Render.
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
    /// Must be called after all the [`TextRenderer::add_buffer_store`]'s.
    ///
    pub fn finalize(&mut self, renderer: &mut GpuRenderer) {
        self.buffer.finalize(renderer);
    }

    /// Updates a [`Text`] and adds its [`TextOrderedIndex`] to staging using [`TextRenderer::add_buffer_store`].
    /// This must be done before [`TextRenderer::finalize`] in order for it to Render.
    ///
    /// # Arguments
    /// - text: [`Text`] we want to update and prepare for rendering.
    /// - atlas: [`TextAtlas`] the [`Text`] needs to render with.
    /// - buffer_layer: The Buffer Layer we want to add this Object too.
    ///
    pub fn update(
        &mut self,
        text: &mut Text,
        atlas: &mut TextAtlas,
        renderer: &mut GpuRenderer,
        buffer_layer: usize,
    ) -> Result<(), GraphicsError> {
        let index = text.update(self, atlas, renderer)?;

        self.add_buffer_store(renderer, index, buffer_layer);
        Ok(())
    }

    /// Returns a reference too [`wgpu::BindGroup`].
    ///
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.text_bind_group
    }
}

/// Trait used to Grant Direct [`Text`] Rendering to [`wgpu::RenderPass`]
pub trait RenderText<'a, 'b>
where
    'b: 'a,
{
    /// Renders the all [`Text`]'s within the buffer layer to screen that have been processed and finalized.
    ///
    fn render_text(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b TextRenderer,
        atlas: &'b TextAtlas,
        buffer_layer: usize,
    );
}

impl<'a, 'b> RenderText<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn render_text(
        &mut self,
        renderer: &'b GpuRenderer,
        buffer: &'b TextRenderer,
        atlas: &'b TextAtlas,
        buffer_layer: usize,
    ) {
        if let Some(Some(details)) = buffer.buffer.buffers.get(buffer_layer)
            && buffer.buffer.count() > 0
        {
            self.set_buffers(renderer.buffer_object.as_buffer_pass());
            self.set_bind_group(1, atlas.text.bind_group(), &[]);
            self.set_bind_group(2, atlas.emoji.bind_group(), &[]);
            self.set_bind_group(3, buffer.bind_group(), &[]);
            self.set_vertex_buffer(1, buffer.buffer.instances(None));
            self.set_pipeline(
                renderer.get_pipelines(TextRenderPipeline).unwrap(),
            );
            self.draw_indexed(
                0..StaticVertexBuffer::index_count(),
                0,
                details.start..details.end,
            );
        }
    }
}
