use std::ops::Range;

/// BufferStore is Storage used to hold and modify the byte arrays that get sent to the GPU.
///
#[derive(Default, Debug)]
pub struct IBOStore {
    /// Storage used for Vertex or Indicies.
    pub store: Vec<u8>,
    /// Boolean used to deturmine if it got changed to tell
    /// the system if we need to reupload the data to the gpu.
    pub changed: bool,
    /// Location Range within GPU this is Stored At
    /// if this does not match the current location internally we will resend
    /// the data to the gpu at the new location.
    pub store_pos: Range<usize>,
}

impl IBOStore {
    /// Used to create a [`BufferStore`].
    ///
    /// # Arguments
    /// - store_size: Preset and filled Size of the buffer to avoid reallocating.
    ///
    pub fn new(store_size: usize) -> Self {
        let mut store = Vec::with_capacity(store_size);

        store.resize_with(store_size, || 0);

        Self {
            store,
            changed: false,
            store_pos: Range::default(),
        }
    }
}
