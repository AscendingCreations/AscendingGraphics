use crate::GpuDevice;
use std::{marker::PhantomData, ops::Range};
use wgpu::{Queue, util::DeviceExt};

/// BufferStore is Storage used to hold and modify the byte arrays that get sent to the GPU.
///
#[derive(Default, Debug)]
pub struct BufferStore {
    /// Storage used for Vertex or Indicies.
    pub store: Vec<u8>,
    /// Storage used for index's
    pub indexs: Vec<u8>,
    /// Boolean used to deturmine if it got changed to tell
    /// the system if we need to reupload the data to the gpu.
    pub changed: bool,
    /// Location Range within GPU this is Stored At
    /// if this does not match the current location internally we will resend
    /// the data to the gpu at the new location.
    pub store_pos: Range<usize>,
    /// Location Range within GPU this is Stored At
    /// if this does not match the current location internally we will resend
    /// the data to the gpu at the new location.
    pub index_pos: Range<usize>,
}

impl BufferStore {
    /// Used to create a [`BufferStore`].
    ///
    /// # Arguments
    /// - store_size: Preset and filled Size of the buffer to avoid reallocating.
    /// - index_size: Preset and filled Size of the buffer to avoid reallocating.
    ///
    pub fn new(store_size: usize, index_size: usize) -> Self {
        let mut store = Vec::with_capacity(store_size);
        let mut indexs = Vec::with_capacity(index_size);

        store.resize_with(store_size, || 0);
        indexs.resize_with(index_size, || 0);

        Self {
            store,
            indexs,
            changed: false,
            store_pos: Range::default(),
            index_pos: Range::default(),
        }
    }
}

/// Pass of Data from a Vertex or Static Vertex used to Set the
/// renderers Vertex and Index buffer Objects.
///
pub struct BufferPass<'a> {
    pub vertex_buffer: &'a wgpu::Buffer,
    pub index_buffer: &'a wgpu::Buffer,
}

/// Trait used to create Passing [`BufferPass`] from their Structs.
pub trait AsBufferPass<'a> {
    /// Creates a [`BufferPass`] from the Holding Object.
    fn as_buffer_pass(&'a self) -> BufferPass<'a>;
}

/// Hold for the Layouts in memory version of Vertex's, indices or Index's.
/// Data of Each object within the same Layout.
///
#[derive(Default)]
pub struct BufferData {
    pub vertexs: Vec<u8>,
    pub indexs: Vec<u8>,
}

/// GPU Buffer Management Struct. Used to keep track of Counts, Length and The Buffer in the GPU.
///
#[derive(Debug)]
pub struct Buffer<K: BufferLayout> {
    pub buffer: wgpu::Buffer,
    pub count: usize,
    pub len: usize,
    pub max: usize,
    phantom_data: PhantomData<K>,
}

impl<K: BufferLayout> Buffer<K> {
    /// Used to create a [`Buffer`].
    ///
    /// # Arguments
    /// - contents: The contents to Create the Buffer with.
    /// - usage: wgpu usage flags [`wgpu::BufferUsages`]
    /// - label: Label to be seen in GPU debugging.
    ///
    pub fn new(
        gpu_device: &GpuDevice,
        contents: &[u8],
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Self {
        Self {
            buffer: gpu_device.device().create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label,
                    contents,
                    usage,
                },
            ),
            count: 0,
            len: 0,
            max: contents.len(),
            phantom_data: PhantomData,
        }
    }

    /// Writes Data into the Buffer from its Position.
    ///
    /// # Panics
    /// - This method fails if data overruns the size of buffer starting at pos.
    ///
    /// # Arguments
    /// - data: the contents to write to the Buffer.
    /// - pos: Position to write to the buffer from.
    ///
    pub fn write(&self, queue: &Queue, data: &[u8], pos: u64) {
        if !data.is_empty() {
            queue.write_buffer(&self.buffer, pos, data);
        }
    }

    /// If the buffer len is empty.
    ///
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a [`wgpu::BufferSlice`] of the buffer to hand off to the GPU.
    ///
    pub fn buffer_slice(&self, range: Range<u64>) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(range)
    }
}

pub trait BufferLayout {
    ///WGPU's Shader Attributes
    fn attributes() -> Vec<wgpu::VertexAttribute>;

    ///Default Buffer set to a large size.
    fn default_buffer() -> BufferData;

    ///The size in bytes the vertex is
    fn stride() -> usize;

    /// Creates a Buffer at a capacity
    /// Capacity is a count of objects.
    fn with_capacity(
        vertex_capacity: usize,
        index_capacity: usize,
    ) -> BufferData;
}
