use crate::{
    AsBufferPass, BufferData, BufferLayout, BufferPass, DrawOrder, GpuBuffer,
    GpuDevice, GpuRenderer, Index, parallel::*,
};
use std::{cmp::Ordering, ops::Range};

/// Details for the Objects Memory location within the Vertex Buffer and Index Buffers.
/// This is used to deturmine if the buffers location has changed or not for
/// reuploading the buffer.
#[derive(Copy, Clone, Debug)]
pub struct IndexDetails {
    /// Start location of the Index Buffer.
    pub indices_start: u32,
    /// End location of the Index Buffer.
    pub indices_end: u32,
    /// Start location of the vertex buffers base.
    pub vertex_base: i32,
}

/// OrderIndex Contains the information needed to Order the buffers and
/// to set the buffers up for rendering.
#[derive(Clone, Copy, Debug, Default)]
pub struct OrderedIndex {
    /// The Draw Order of the Buffer.
    pub(crate) order: DrawOrder,
    /// The Index to the Buffer.
    pub(crate) index: Index,
    /// Stores the VBO buffers indices count.
    pub(crate) index_count: u32,
    /// Stores the VBO buffers indices max count.
    pub(crate) index_max: u32,
}

impl PartialOrd for OrderedIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for OrderedIndex {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}

impl Eq for OrderedIndex {}

impl Ord for OrderedIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order.cmp(&other.order)
    }
}

impl OrderedIndex {
    /// Creates a OrderedIndex with DrawOrder, Buffer Index and Index Max.
    pub fn new(order: DrawOrder, index: Index, index_max: u32) -> Self {
        Self {
            order,
            index,
            index_count: 0,
            index_max,
        }
    }
}

/// VertexBuffer holds all the Details to render with Verticies and indicies.
/// This stores and handles the orders of all rendered objects to try and reduce the amount
/// of GPU uploads we make.
#[derive(Debug)]
pub struct VertexBuffer<K: BufferLayout> {
    /// Unprocessed Buffer Data.
    pub unprocessed: Vec<(usize, OrderedIndex)>,
    /// Buffers ready to Render
    pub buffers: Vec<(usize, IndexDetails)>,
    /// The main Vertex Buffer within GPU memory.
    pub vertex_buffer: GpuBuffer<K>,
    /// The main Index Buffer within GPU memory.
    pub index_buffer: GpuBuffer<K>,
    /// Used to Resize the vertex buffer if new data will not fit within.
    vertex_needed: usize,
    /// Used to Resize the index buffer if new data will not fit within.
    index_needed: usize,
}

impl<'a, K: BufferLayout> AsBufferPass<'a> for VertexBuffer<K> {
    fn as_buffer_pass(&'a self) -> BufferPass<'a> {
        BufferPass {
            vertex_buffer: &self.vertex_buffer.buffer,
            index_buffer: &self.index_buffer.buffer,
        }
    }
}

impl<K: BufferLayout> VertexBuffer<K> {
    /// Used to create a [`VertexBuffer`].
    /// Only use this for creating a reusable buffer.
    ///
    /// # Arguments
    /// - buffers: The (Vertex:Vec<u8>, Indices:Vec<u8>) to Create the Buffer with.
    ///
    pub fn create_buffer(gpu_device: &GpuDevice, buffers: &BufferData) -> Self {
        VertexBuffer {
            unprocessed: Vec::new(),
            buffers: Vec::new(),
            vertex_buffer: GpuBuffer::new(
                gpu_device,
                &buffers.vertexs,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                Some("Vertex Buffer"),
            ),
            vertex_needed: 0,
            index_buffer: GpuBuffer::new(
                gpu_device,
                &buffers.indexs,
                wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                Some("Index Buffer"),
            ),
            index_needed: 0,
        }
    }

    /// Used to create a [`VertexBuffer`].
    /// Only use this for creating a reusable buffer.
    ///
    /// # Arguments
    /// - buffers: The (Vertex:Vec<u8>, Indices:Vec<u8>) to Create the Buffer with.
    /// - capacity: the capacity of Elements to precreate.
    ///
    pub fn create_buffer_with(
        gpu_device: &GpuDevice,
        buffers: &BufferData,
        capacity: usize,
    ) -> Self {
        let size = capacity.max(1);

        VertexBuffer {
            unprocessed: Vec::with_capacity(size),
            buffers: Vec::with_capacity(size),
            vertex_buffer: GpuBuffer::new(
                gpu_device,
                &buffers.vertexs,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                Some("Vertex Buffer"),
            ),
            vertex_needed: 0,
            index_buffer: GpuBuffer::new(
                gpu_device,
                &buffers.indexs,
                wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                Some("Index Buffer"),
            ),
            index_needed: 0,
        }
    }

    /// Adds the Buffer to the unprocessed list so it can be processed in [`VertexBuffer::finalize`]
    /// This must be called in order to Render the Object.
    ///
    /// # Arguments
    /// - index: The Order Index of the Object we want to render.
    /// - buffer_layer: The Buffer Layer we want to add this Object too.
    ///
    pub fn add_buffer_store(
        &mut self,
        renderer: &GpuRenderer,
        mut index: OrderedIndex,
        buffer_layer: usize,
    ) {
        if let Some(store) = renderer.get_vbo_store(index.index) {
            self.vertex_needed += store.store.len();
            self.index_needed += store.indexs.len();

            index.index_count = store.indexs.len() as u32 / 4;

            self.unprocessed.push((buffer_layer, index));
        }
    }

    /// Processes all unprocessed listed buffers and uploads any changes to the gpu
    /// This must be called after [`VertexBuffer::add_buffer_store`] in order to Render the Objects.
    pub fn finalize(&mut self, renderer: &mut GpuRenderer) {
        let (
            mut changed,
            mut vertex_pos,
            mut index_pos,
            mut pos,
            mut base_vertex,
        ) = (false, 0, 0, 0, 0);

        if self.vertex_needed > self.vertex_buffer.max
            || self.index_needed > self.index_buffer.max
        {
            self.resize(
                renderer.gpu_device(),
                self.vertex_needed / K::stride(),
                self.index_needed,
            );
            changed = true;
        }

        self.vertex_buffer.count = self.vertex_needed / K::stride();
        self.vertex_buffer.len = self.vertex_needed;
        self.unprocessed.par_sort();
        self.buffers.clear();

        for (layer, buf) in self.unprocessed.iter() {
            let mut write_vertex = false;
            let mut write_index = false;
            let old_vertex_pos = vertex_pos as u64;
            let old_index_pos = index_pos as u64;

            if let Some(store) = renderer.get_vbo_store_mut(buf.index) {
                if store.indexs.is_empty() {
                    continue;
                }

                let vertex_range = vertex_pos..vertex_pos + store.store.len();
                let index_range = index_pos..index_pos + store.indexs.len();

                if store.store_pos != vertex_range || changed || store.changed {
                    store.store_pos = vertex_range;
                    write_vertex = true
                }

                if store.index_pos != index_range || changed || store.changed {
                    store.index_pos = index_range;
                    write_index = true
                }

                if write_index || write_vertex {
                    store.changed = false;
                }

                vertex_pos += store.store.len();
                index_pos += store.indexs.len();
            }

            if write_vertex
                && let Some(store) = renderer.get_ibo_store(buf.index)
            {
                self.vertex_buffer.write(
                    renderer.queue(),
                    &store.store,
                    old_vertex_pos,
                );
            }

            if write_index
                && let Some(store) = renderer.get_vbo_store(buf.index)
            {
                self.index_buffer.write(
                    renderer.queue(),
                    &store.indexs,
                    old_index_pos,
                );
            }

            let indices_start = pos;
            let indices_end = pos + buf.index_count;
            let vertex_base = base_vertex;

            base_vertex += buf.index_max as i32 + 1;
            pos += buf.index_count;

            self.buffers.push((
                *layer,
                IndexDetails {
                    indices_start,
                    indices_end,
                    vertex_base,
                },
            ));
        }

        self.unprocessed.clear();
        self.vertex_needed = 0;
        self.index_needed = 0;
    }

    //private but resizes the buffer on the GPU when needed.
    fn resize(
        &mut self,
        gpu_device: &GpuDevice,
        vertex_capacity: usize,
        index_capacity: usize,
    ) {
        let buffers = K::with_capacity(vertex_capacity, index_capacity);

        self.vertex_buffer = GpuBuffer::new(
            gpu_device,
            &buffers.vertexs,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            Some("Vertex Buffer"),
        );

        self.index_buffer = GpuBuffer::new(
            gpu_device,
            &buffers.indexs,
            wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            Some("Index Buffer"),
        )
    }

    /// Returns the index count.
    pub fn index_count(&self) -> usize {
        self.index_buffer.count
    }

    /// Returns the index maximum size.
    pub fn index_max(&self) -> usize {
        self.index_buffer.max
    }

    /// Returns [`wgpu::BufferSlice`] of indices.
    /// bounds is used to set a specific Range if needed.
    /// If bounds is None then range is 0..index_count.
    pub fn indices(&self, bounds: Option<Range<u64>>) -> wgpu::BufferSlice<'_> {
        let range = if let Some(bounds) = bounds {
            bounds
        } else {
            0..(self.index_buffer.count) as u64
        };

        self.index_buffer.buffer_slice(range)
    }

    /// Creates an [`VertexBuffer`] with a default buffer size.
    /// Buffer size is based on the initial [`crate::BufferLayout::default_buffer`] length.
    ///
    /// # Arguments
    /// - layer_size: The capacity allocated for any future elements per new Buffer Layer.
    ///
    pub fn new(device: &GpuDevice) -> Self {
        Self::create_buffer(device, &K::default_buffer())
    }

    /// Set the Index based on how many Vertex's Exist
    pub fn set_index_count(&mut self, count: usize) {
        self.index_buffer.count = count;
    }

    /// Returns the Vertex elements count.
    pub fn vertex_count(&self) -> usize {
        self.vertex_buffer.count
    }

    /// Returns if the vertex buffer is empty
    pub fn is_empty(&self) -> bool {
        self.vertex_buffer.count == 0
    }

    /// Returns vertex_buffer's max size in bytes.
    pub fn vertex_max(&self) -> usize {
        self.vertex_buffer.max
    }

    /// Returns vertex_buffer's vertex_stride.
    pub fn vertex_stride(&self) -> usize {
        K::stride()
    }

    /// Returns [`wgpu::BufferSlice`] of vertices.
    /// bounds is used to set a specific Range if needed.
    /// If bounds is None then range is 0..vertex_count.
    pub fn vertices(
        &self,
        bounds: Option<Range<u64>>,
    ) -> wgpu::BufferSlice<'_> {
        let range = if let Some(bounds) = bounds {
            bounds
        } else {
            0..self.vertex_buffer.count as u64
        };

        self.vertex_buffer.buffer_slice(range)
    }

    /// Creates an [`VertexBuffer`] with a buffer capacity.
    /// Buffer size is based on the initial [`crate::BufferLayout::default_buffer`] length.
    ///
    /// # Arguments
    /// - capacity: The capacity of the Buffers instances for future allocation * 2.
    ///
    pub fn with_capacity(gpu_device: &GpuDevice, capacity: usize) -> Self {
        Self::create_buffer(
            gpu_device,
            &K::with_capacity(capacity, capacity * 2),
        )
    }

    pub fn get_layer(&self, layer: usize) -> &[(usize, IndexDetails)] {
        let start = self.buffers.partition_point(|(l, _)| *l < layer);
        let end = self.buffers.partition_point(|(l, _)| *l <= layer);

        &self.buffers[start..end]
    }
}
