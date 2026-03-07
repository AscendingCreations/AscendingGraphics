use crate::{
    Bounds, Buffer, BufferLayout, CameraView, GpuDevice, GpuRenderer,
    OrderedIndex, parallel::*,
};
use std::ops::Range;

/// Details for the Objects Memory location within the instance Buffer.
/// This is used to deturmine if the buffers location has changed or not for
/// reuploading the buffer.
#[derive(Debug, Copy, Clone)]
pub struct InstanceDetails {
    /// Start location of the Buffer.
    pub start: u32,
    /// End location of the Buffer.
    pub end: u32,
}

/// Clipped buffers Tuple type.
pub type ClippedInstanceDetails = (InstanceDetails, Option<Bounds>, CameraView);

/// Instance buffer holds all the Details to render with instances with a Static VBO.
/// This stores and handles the orders of all rendered objects to try and reduce the amount
/// of GPU uploads we make.
#[derive(Debug)]
pub struct InstanceBuffer<K: BufferLayout> {
    /// Unprocessed Buffer Data. (layer, order_index)
    pub unprocessed: Vec<(usize, OrderedIndex)>,
    /// Buffers ready to Render
    pub buffers: Vec<Option<InstanceDetails>>,
    /// The main Buffer within GPU memory.
    pub buffer: Buffer<K>,
    /// Used to Resize the buffer if new data will not fit within.
    needed_size: usize,
}

impl<K: BufferLayout> InstanceBuffer<K> {
    /// Used to create a [`InstanceBuffer`].
    /// Only use this for creating a reusable buffer.
    ///
    /// # Arguments
    /// - data: The contents to Create the Buffer with.
    ///
    pub fn create_buffer(gpu_device: &GpuDevice, data: &[u8]) -> Self {
        InstanceBuffer {
            unprocessed: Vec::new(),
            buffers: Vec::new(),
            buffer: Buffer::new(
                gpu_device,
                data,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                Some("Instance Buffer"),
            ),
            needed_size: 0,
        }
    }

    /// Used to create a [`InstanceBuffer`] with predeturmined sizes.
    /// Only use this for creating a reusable buffer.
    ///
    /// # Arguments
    /// - data: The contents to Create the Buffer with.
    /// - capacity: the pre-capacity of objects to insert
    ///
    pub fn create_buffer_with(
        gpu_device: &GpuDevice,
        data: &[u8],
        capacity: usize,
    ) -> Self {
        let size = capacity.max(1);
        let unprocessed = Vec::with_capacity(size);

        InstanceBuffer {
            unprocessed,
            buffers: Vec::with_capacity(size),
            buffer: Buffer::new(
                gpu_device,
                data,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                Some("Instance Buffer"),
            ),
            needed_size: 0,
        }
    }

    /// Adds the Buffer to the unprocessed list so it can be processed in [`InstanceBuffer::finalize`]
    /// This must be called in order to Render the Object.
    ///
    /// # Arguments
    /// - index: The Order Index of the Object we want to render.
    /// - buffer_layer: The Buffer Layer we want to add this Object too.
    ///
    pub fn add_buffer_store(
        &mut self,
        renderer: &GpuRenderer,
        index: OrderedIndex,
        buffer_layer: usize,
    ) {
        if let Some(store) = renderer.get_buffer(index.index) {
            self.needed_size += store.store.len();
            self.unprocessed.push((buffer_layer, index));
        }
    }

    fn buffer_write(
        &self,
        renderer: &mut GpuRenderer,
        buf: &OrderedIndex,
        pos: &mut usize,
        count: &mut u32,
        changed: bool,
    ) {
        let mut write_buffer = false;
        let old_pos = *pos as u64;

        if let Some(store) = renderer.get_buffer_mut(buf.index) {
            let range = *pos..*pos + store.store.len();

            if store.store_pos != range || changed || store.changed {
                store.store_pos = range;
                store.changed = false;
                write_buffer = true;
            }

            *pos += store.store.len();
            *count += (store.store.len() / K::stride()) as u32;
        }

        if write_buffer && let Some(store) = renderer.get_buffer(buf.index) {
            self.buffer.write(renderer.queue(), &store.store, old_pos);
        }
    }

    /// Processes all unprocessed listed buffers and uploads any changes to the gpu
    /// This must be called after [`InstanceBuffer::add_buffer_store`] in order to Render the Objects.
    pub fn finalize(&mut self, renderer: &mut GpuRenderer) {
        let (mut changed, mut pos, mut count, mut last_layer, mut start_pos) =
            (false, 0, 0, 0, 0);

        if self.needed_size > self.buffer.max {
            self.resize(renderer.gpu_device(), self.needed_size / K::stride());
            changed = true;
        }

        self.buffer.count = self.needed_size / K::stride();
        self.buffer.len = self.needed_size;
        self.unprocessed.par_sort();
        self.buffers.clear();

        for processing in self.unprocessed.iter() {
            if last_layer != processing.0 {
                // set the buffer to the last known start and count.
                if count != 0 {
                    self.buffers.push(Some(InstanceDetails {
                        start: start_pos,
                        end: count,
                    }));
                }

                start_pos = count;

                //add in any empty layers for a faster lookup when rendering based on layer.
                if last_layer + 1 != processing.0 {
                    for _ in last_layer + 1..processing.0 {
                        self.buffers.push(None);
                    }
                }

                last_layer = processing.0;
            }

            self.buffer_write(
                renderer,
                &processing.1,
                &mut pos,
                &mut count,
                changed,
            );
        }

        if start_pos != count {
            self.buffers.push(Some(InstanceDetails {
                start: start_pos,
                end: count,
            }));
        }

        self.needed_size = 0;
        self.unprocessed.clear();
    }

    //private but resizes the buffer on the GPU when needed.
    fn resize(&mut self, gpu_device: &GpuDevice, capacity: usize) {
        let data = K::with_capacity(capacity, 0);

        self.buffer = Buffer::new(
            gpu_device,
            &data.vertexs,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            Some("Vertex Buffer"),
        );
    }

    /// Creates an [`InstanceBuffer`] with a default buffer size.
    /// Buffer size is based on the initial [`crate::BufferLayout::default_buffer`] length.
    ///
    pub fn new(gpu_device: &GpuDevice) -> Self {
        Self::create_buffer(gpu_device, &K::default_buffer().vertexs)
    }

    /// Returns the instances count.
    pub fn count(&self) -> u32 {
        self.buffer.count as u32
    }

    /// Returns the instances byte count.
    pub fn len(&self) -> u64 {
        self.buffer.len as u64
    }

    /// Returns if the instance buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns instance buffers max size in bytes.
    pub fn max(&self) -> usize {
        self.buffer.max
    }

    /// Returns buffer's stride.
    pub fn stride(&self) -> usize {
        K::stride()
    }

    /// Returns [`wgpu::BufferSlice`] of vertices.
    /// bounds is used to set a specific Range if needed.
    /// If bounds is None then range is 0..vertex_count.
    pub fn instances(
        &self,
        bounds: Option<Range<u64>>,
    ) -> wgpu::BufferSlice<'_> {
        let range = if let Some(bounds) = bounds {
            bounds
        } else {
            0..self.len()
        };

        self.buffer.buffer_slice(range)
    }

    /// Creates an InstanceBuffer with a buffer capacity.
    /// Buffer size is based on the initial [`crate::BufferLayout::default_buffer`] length.
    ///
    /// # Arguments
    /// - capacity: The capacity allocated for any future elements.
    ///
    pub fn with_capacity(gpu_device: &GpuDevice, capacity: usize) -> Self {
        Self::create_buffer(gpu_device, &K::with_capacity(capacity, 0).vertexs)
    }
}
