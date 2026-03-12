use std::ops::Range;

/// VboBufferStore is Storage used to hold and modify the byte arrays that get sent to the GPU. for the VertexBuffers
///
#[derive(Default, Debug)]
pub struct VBOStore {
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

impl VBOStore {
    /// Used to create a [`VboBufferStore`].
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
