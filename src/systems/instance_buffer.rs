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
    /// Unprocessed Buffer Data.
    pub unprocessed: Vec<Vec<OrderedIndex>>,
    /// Buffers ready to Render
    pub buffers: Vec<Option<InstanceDetails>>,
    /// Clipped Buffers ready to Render.
    pub clipped_buffers: Vec<Vec<ClippedInstanceDetails>>,
    /// The main Buffer within GPU memory.
    pub buffer: Buffer<K>,
    /// Size each Buffer Layer gets allocated to for Future buffers.
    pub layer_size: usize,
    /// Used to Resize the buffer if new data will not fit within.
    needed_size: usize,
    /// Deturmines if we need to use clipped_buffers or Buffers for Rendering.
    is_clipped: bool,
}

impl<K: BufferLayout> InstanceBuffer<K> {
    /// Used to create a [`InstanceBuffer`].
    /// Only use this for creating a reusable buffer.
    ///
    /// # Arguments
    /// - data: The contents to Create the Buffer with.
    /// - layer_size: The capacity allocated for any future elements per new Buffer Layer.
    ///
    pub fn create_buffer(
        gpu_device: &GpuDevice,
        data: &[u8],
        layer_size: usize,
    ) -> Self {
        InstanceBuffer {
            unprocessed: Vec::new(),
            buffers: Vec::new(),
            clipped_buffers: Vec::new(),
            buffer: Buffer::new(
                gpu_device,
                data,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                Some("Instance Buffer"),
            ),
            layer_size: layer_size.max(32),
            needed_size: 0,
            is_clipped: false,
        }
    }

    /// Used to create a [`InstanceBuffer`] with predeturmined sizes.
    /// Only use this for creating a reusable buffer.
    ///
    /// # Arguments
    /// - data: The contents to Create the Buffer with.
    /// - layer_size: The capacity allocated for any future elements per new Buffer Layer.
    /// - capacity: the capacity of Layers to precreate.
    /// - layer_capacity: the capacity to which each layer will precreate.
    ///
    pub fn create_buffer_with(
        gpu_device: &GpuDevice,
        data: &[u8],
        layer_size: usize,
        capacity: usize,
        layer_capacity: usize,
    ) -> Self {
        let layer = layer_capacity.max(32);
        let size = capacity.max(1);
        let mut unprocessed = Vec::with_capacity(size);
        let mut clipped_buffers = Vec::with_capacity(size);

        for _ in 0..size {
            unprocessed.push(Vec::with_capacity(layer));

            clipped_buffers.push(Vec::with_capacity(layer));
        }

        InstanceBuffer {
            unprocessed,
            buffers: Vec::with_capacity(size),
            clipped_buffers,
            buffer: Buffer::new(
                gpu_device,
                data,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                Some("Instance Buffer"),
            ),
            layer_size: layer_size.max(32),
            needed_size: 0,
            is_clipped: false,
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
            let offset = buffer_layer.saturating_add(1);

            if self.unprocessed.len() < offset {
                let mut expansions = (self.unprocessed.len()..offset)
                    .into_par_iter()
                    .map(|_| Vec::with_capacity(self.layer_size))
                    .collect();

                self.unprocessed.append(&mut expansions);
            }

            self.needed_size += store.store.len();

            if let Some(unprocessed) = self.unprocessed.get_mut(buffer_layer) {
                unprocessed.push(index);
            }
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
        let (mut changed, mut pos, mut count) = (false, 0, 0);

        if self.needed_size > self.buffer.max {
            self.resize(renderer.gpu_device(), self.needed_size / K::stride());
            changed = true;
        }

        self.buffer.count = self.needed_size / K::stride();
        self.buffer.len = self.needed_size;

        self.unprocessed
            .par_iter_mut()
            .for_each(|processing| processing.par_sort());

        if self.is_clipped {
            self.clipped_buffers.par_iter_mut().for_each(|buffer| {
                buffer.clear();
            });

            if self.clipped_buffers.len() < self.unprocessed.len() {
                for i in self.clipped_buffers.len()..self.unprocessed.len() {
                    let count = self.unprocessed.get(i).unwrap().len();
                    self.clipped_buffers.push(Vec::with_capacity(count));
                }
            }
        } else {
            self.buffers.clear();
        }

        for (layer, processing) in self.unprocessed.iter().enumerate() {
            if processing.is_empty() {
                if !self.is_clipped {
                    self.buffers.push(None);
                }
                continue;
            }

            let mut start_pos = count;

            if !self.is_clipped {
                for buf in processing {
                    self.buffer_write(
                        renderer, buf, &mut pos, &mut count, changed,
                    );
                }

                self.buffers.push(Some(InstanceDetails {
                    start: start_pos,
                    end: count,
                }));
            } else {
                for buf in processing {
                    self.buffer_write(
                        renderer, buf, &mut pos, &mut count, changed,
                    );

                    if let Some(buffer) = self.clipped_buffers.get_mut(layer) {
                        buffer.push((
                            InstanceDetails {
                                start: start_pos,
                                end: count,
                            },
                            buf.bounds,
                            buf.camera_view,
                        ));
                    }

                    start_pos = count;
                }
            }
        }

        self.needed_size = 0;

        self.unprocessed.par_iter_mut().for_each(|buffer| {
            buffer.clear();
        });
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
    /// # Arguments
    /// - layer_size: The capacity allocated for any future elements per new Buffer Layer.
    ///
    pub fn new(gpu_device: &GpuDevice, layer_size: usize) -> Self {
        Self::create_buffer(
            gpu_device,
            &K::default_buffer().vertexs,
            layer_size,
        )
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

    /// Returns if the buffer is clipped or not to deturmine if you should use
    /// buffers or clipped_buffers.
    pub fn is_clipped(&self) -> bool {
        self.is_clipped
    }

    /// Sets the Buffer into Clipping mode.
    /// This will Produce a clipped_buffers instead of the buffers which
    /// will still be layered but a Vector of individual objects will Exist rather
    /// than a grouped object per layer. Will make it less Efficient but allows Bounds Clipping.
    pub fn set_as_clipped(&mut self) {
        self.is_clipped = true;
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
    /// - capacity: The capacity of the Buffers instances for future allocation.
    /// - layer_size: The capacity allocated for any future elements per new Buffer Layer.
    ///
    pub fn with_capacity(
        gpu_device: &GpuDevice,
        capacity: usize,
        layer_size: usize,
    ) -> Self {
        Self::create_buffer(
            gpu_device,
            &K::with_capacity(capacity, 0).vertexs,
            layer_size,
        )
    }
}
