use crate::{
    BufferPass, BufferStore, GpuDevice, GpuWindow, GraphicsError, Index,
    Layout, LayoutStorage, OtherError, PipeLineLayout, PipelineStorage,
    StaticVertexBuffer,
};
use cosmic_text::FontSystem;
use slotmap::SlotMap;
use std::sync::Arc;

use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};

/// Handles the [`GpuWindow`], [`GpuDevice`] and [`BufferStore`]'s.
/// Also handles other information important to Rendering to the GPU.
///
#[derive(Debug)]
pub struct GpuRenderer {
    pub(crate) window: GpuWindow,
    pub(crate) device: GpuDevice,
    pub(crate) buffer_stores: SlotMap<Index, BufferStore>,
    pub(crate) layout_storage: LayoutStorage,
    pub(crate) pipeline_storage: PipelineStorage,
    pub(crate) depthbuffer: wgpu::TextureView,
    pub(crate) framebuffer: Option<wgpu::TextureView>,
    pub(crate) frame: Option<wgpu::SurfaceTexture>,
    pub font_sys: FontSystem,
    pub buffer_object: StaticVertexBuffer,
    pub backend: wgpu::Backend,
}

/// Used to deturmine which Pipelines to enable when using the Renderer.
///
#[derive(Debug, Default, Copy, Clone)]
pub struct EnabledPipelines {
    pub image_pipeline: bool,
    pub anim_image_pipeline: bool,
    pub map_pipeline: bool,
    pub text_pipeline: bool,
    pub mesh2d_pipeline: bool,
    pub light_pipeline: bool,
    pub rect_pipeline: bool,
}

impl EnabledPipelines {
    pub fn all() -> Self {
        Self {
            image_pipeline: true,
            anim_image_pipeline: true,
            map_pipeline: true,
            text_pipeline: true,
            mesh2d_pipeline: true,
            light_pipeline: true,
            rect_pipeline: true,
        }
    }
}

/// Trait to allow [`wgpu::RenderPass`] to Set the Vertex and Index buffers.
///
pub trait SetBuffers<'a, 'b>
where
    'b: 'a,
{
    /// Sets the Vertex and Index buffers from a [`BufferPass`]
    fn set_buffers(&mut self, buffer: BufferPass<'b>);
}

impl<'a, 'b> SetBuffers<'a, 'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn set_buffers(&mut self, buffer: BufferPass<'b>) {
        self.set_vertex_buffer(0, buffer.vertex_buffer.slice(..));
        self.set_index_buffer(
            buffer.index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
    }
}

impl GpuRenderer {
    /// Creates a New GpuRenderer.
    ///
    pub fn new(window: GpuWindow, device: GpuDevice) -> Self {
        let buffer_object = StaticVertexBuffer::create_buffer(&device);
        let depth_buffer = window.create_depth_texture(&device);
        let backend = window.adapter.get_info().backend;

        Self {
            window,
            device,
            buffer_stores: SlotMap::with_capacity_and_key(1024),
            layout_storage: LayoutStorage::new(),
            pipeline_storage: PipelineStorage::new(),
            depthbuffer: depth_buffer,
            framebuffer: None,
            frame: None,
            font_sys: FontSystem::new(),
            buffer_object,
            backend,
        }
    }

    /// Returns a reference to [`wgpu::Adapter`].
    ///
    pub fn adapter(&self) -> &wgpu::Adapter {
        self.window.adapter()
    }

    /// Resizes the Window.
    ///
    pub fn resize(
        &mut self,
        size: PhysicalSize<u32>,
    ) -> Result<(), GraphicsError> {
        self.window.resize(&self.device, size)
    }

    /// Returns a reference to the Optional [`wgpu::TextureView`]: frame buffer.
    ///
    pub fn frame_buffer(&self) -> &Option<wgpu::TextureView> {
        &self.framebuffer
    }

    /// Returns a reference to [`wgpu::TextureView`].
    ///
    pub fn depth_buffer(&self) -> &wgpu::TextureView {
        &self.depthbuffer
    }

    /// Returns the windows [`PhysicalSize`].
    ///
    pub fn size(&self) -> PhysicalSize<f32> {
        self.window.size
    }

    /// Returns the windows inner [`PhysicalSize`].
    ///
    pub fn inner_size(&self) -> PhysicalSize<u32> {
        self.window.inner_size
    }

    /// Returns a reference to [`wgpu::Surface`].
    ///
    pub fn surface(&self) -> &wgpu::Surface<'_> {
        &self.window.surface
    }

    /// Returns the surfaces [`wgpu::TextureFormat`].
    ///
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.window.surface_format
    }

    /// Called to update the Optional Framebuffer with a Buffer we use to render.
    /// Will return weither the frame buffer could have been processed or not.
    /// If not we should skip rendering till we can get a frame buffer.
    ///
    pub fn update(
        &mut self,
        event: &WindowEvent,
    ) -> Result<bool, GraphicsError> {
        let frame = match self.window.update(&self.device, event)? {
            Some(frame) => frame,
            _ => return Ok(false),
        };

        self.framebuffer = Some(
            frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
        self.frame = Some(frame);

        Ok(true)
    }

    /// Returns a reference to [`Window`].
    ///
    pub fn window(&self) -> &Window {
        &self.window.window
    }

    /// Updates the Internally Stored Depth Buffer.
    ///
    pub fn update_depth_texture(&mut self) {
        self.depthbuffer = self.window.create_depth_texture(&self.device);
    }

    /// Presents the Current frame Buffer to the Window if Some().
    /// If the frame buffer does not Exist will return a Error.
    ///
    pub fn present(&mut self) -> Result<(), GraphicsError> {
        self.framebuffer = None;

        match self.frame.take() {
            Some(frame) => {
                frame.present();
                Ok(())
            }
            None => Err(GraphicsError::Other(OtherError::new(
                "Frame does not Exist. Did you forget to update the renderer?",
            ))),
        }
    }

    /// Returns a reference to [`wgpu::Device`].
    ///
    pub fn device(&self) -> &wgpu::Device {
        &self.device.device
    }

    /// Returns a reference to [`GpuDevice`].
    ///
    pub fn gpu_device(&self) -> &GpuDevice {
        &self.device
    }

    /// Returns a reference to [`wgpu::Queue`].
    ///
    pub fn queue(&self) -> &wgpu::Queue {
        &self.device.queue
    }

    /// Returns a reference to [`FontSystem`].
    ///
    pub fn font_sys(&self) -> &FontSystem {
        &self.font_sys
    }

    /// Returns a mutable reference to [`FontSystem`].
    ///
    pub fn font_sys_mut(&mut self) -> &mut FontSystem {
        &mut self.font_sys
    }

    /// Creates a New [`BufferStore`] with set sizes for Rendering Object Data Storage and
    /// Returns its [`Index`] for Referencing it.
    ///
    pub fn new_buffer(
        &mut self,
        store_size: usize,
        index_size: usize,
    ) -> Index {
        self.buffer_stores
            .insert(BufferStore::new(store_size, index_size))
    }

    /// Creates a New [`BufferStore`] with default sizes for Rendering Object Data Storage and
    /// Returns its [`Index`] for Referencing it.
    ///
    pub fn default_buffer(&mut self) -> Index {
        self.buffer_stores.insert(BufferStore::default())
    }

    /// Removes a [`BufferStore`] using its [`Index`].
    ///
    pub fn remove_buffer(&mut self, index: Index) {
        let _ = self.buffer_stores.remove(index);
    }

    /// Gets a optional reference to [`BufferStore`] using its [`Index`].
    ///
    pub fn get_buffer(&self, index: Index) -> Option<&BufferStore> {
        self.buffer_stores.get(index)
    }

    /// Gets a optional mutable reference to [`BufferStore`] using its [`Index`].
    ///
    pub fn get_buffer_mut(&mut self, index: Index) -> Option<&mut BufferStore> {
        self.buffer_stores.get_mut(index)
    }

    /// Creates new BindGroupLayout from Generic K and Returns a Reference Counter to them.
    ///
    pub fn create_layout<K: Layout>(
        &mut self,
        layout: K,
    ) -> Arc<wgpu::BindGroupLayout> {
        self.layout_storage.create_layout(&mut self.device, layout)
    }

    /// Returns a Reference Counter to the layout Or None if it does not yet Exist.
    ///
    pub fn get_layout<K: Layout>(
        &self,
        layout: K,
    ) -> Option<Arc<wgpu::BindGroupLayout>> {
        self.layout_storage.get_layout(layout)
    }

    /// Creates each supported rendering objects pipeline.
    ///
    pub fn create_pipelines(
        &mut self,
        pipelined_enabled: EnabledPipelines,
        surface_format: wgpu::TextureFormat,
    ) {
        if pipelined_enabled.image_pipeline {
            self.pipeline_storage.create_pipeline(
                &mut self.device,
                &mut self.layout_storage,
                surface_format,
                crate::ImageRenderPipeline,
            );
        }

        if pipelined_enabled.anim_image_pipeline {
            self.pipeline_storage.create_pipeline(
                &mut self.device,
                &mut self.layout_storage,
                surface_format,
                crate::AnimImageRenderPipeline,
            );
        }

        if pipelined_enabled.map_pipeline {
            self.pipeline_storage.create_pipeline(
                &mut self.device,
                &mut self.layout_storage,
                surface_format,
                crate::MapRenderPipeline,
            );
        }

        if pipelined_enabled.text_pipeline {
            self.pipeline_storage.create_pipeline(
                &mut self.device,
                &mut self.layout_storage,
                surface_format,
                crate::TextRenderPipeline,
            );
        }

        if pipelined_enabled.mesh2d_pipeline {
            self.pipeline_storage.create_pipeline(
                &mut self.device,
                &mut self.layout_storage,
                surface_format,
                crate::Mesh2DRenderPipeline,
            );
        }

        if pipelined_enabled.light_pipeline {
            self.pipeline_storage.create_pipeline(
                &mut self.device,
                &mut self.layout_storage,
                surface_format,
                crate::LightRenderPipeline,
            );
        }

        if pipelined_enabled.rect_pipeline {
            self.pipeline_storage.create_pipeline(
                &mut self.device,
                &mut self.layout_storage,
                surface_format,
                crate::RectRenderPipeline,
            );
        }
    }

    /// Gets a optional reference of [`wgpu::RenderPipeline`]
    ///
    pub fn get_pipelines<K: PipeLineLayout>(
        &self,
        pipeline: K,
    ) -> Option<&wgpu::RenderPipeline> {
        self.pipeline_storage.get_pipeline(pipeline)
    }
}
