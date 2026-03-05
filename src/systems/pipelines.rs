use crate::{AHashMap, GpuDevice, LayoutStorage};
use bytemuck::{Pod, Zeroable};
use std::any::{Any, TypeId};

/// Trait used to Create and Load [`wgpu::RenderPipeline`] to and from a HashMap.
///
pub trait PipeLineLayout: Pod + Zeroable {
    /// Creates the [`wgpu::RenderPipeline`] to be added to the HashMap
    ///
    fn create_layout(
        &self,
        gpu_device: &mut GpuDevice,
        layouts: &mut LayoutStorage,
        surface_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline;

    /// Gives a Hashable Key of the [`wgpu::RenderPipeline`] to use to Retrieve it from the HashMap.
    ///
    fn layout_key(&self) -> (TypeId, Vec<u8>) {
        let type_id = self.type_id();
        let bytes: Vec<u8> =
            bytemuck::try_cast_slice(&[*self]).unwrap_or(&[]).to_vec();

        (type_id, bytes)
    }
}

/// [`wgpu::RenderPipeline`] Storage using a hashmap.
///
#[derive(Debug)]
pub struct PipelineStorage {
    pub(crate) map: AHashMap<(TypeId, Vec<u8>), wgpu::RenderPipeline>,
}

impl PipelineStorage {
    /// Creates a new [`PipelineStorage`] with default HashMap.
    ///
    pub fn new() -> Self {
        Self {
            map: AHashMap::default(),
        }
    }

    /// Creates a new [`wgpu::RenderPipeline`] from [`PipeLineLayout`] and adds it to the internal map.
    ///
    pub fn create_pipeline<K: PipeLineLayout>(
        &mut self,
        device: &mut GpuDevice,
        layout_storage: &mut LayoutStorage,
        surface_format: wgpu::TextureFormat,
        pipeline: K,
    ) {
        let key = pipeline.layout_key();

        self.map.insert(
            key,
            pipeline.create_layout(device, layout_storage, surface_format),
        );
    }

    /// Retrieves a Reference to a [`wgpu::RenderPipeline`] within the internal map for rendering.
    ///
    pub fn get_pipeline<K: PipeLineLayout>(
        &self,
        pipeline: K,
    ) -> Option<&wgpu::RenderPipeline> {
        let key = pipeline.layout_key();

        self.map.get(&key)
    }
}

impl Default for PipelineStorage {
    fn default() -> Self {
        Self::new()
    }
}
