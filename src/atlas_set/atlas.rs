pub use super::allocator::Allocator;
use crate::AIndexSet;

/// Atlas Layer within an [`AtlasSet`].
/// This Keeps track of the Individual Texture Layer.
pub struct Atlas {
    /// Handles the space allocation of the layer.
    pub allocator: Allocator,
    /// Stores each Index the allocations exist at for this layer.
    pub allocated: AIndexSet<usize>,
    /// use to avoid placing newly loaded images into
    /// if we are migrating images out of it.
    pub migrating: bool,
}

impl Atlas {
    /// Creates a new Atlas with Allocator texture size.
    ///
    pub fn new(size: u32) -> Self {
        Self {
            allocator: Allocator::new(size),
            allocated: AIndexSet::default(),
            migrating: false,
        }
    }

    /// Allocates a Spot within the Texture for uploading too.
    ///
    pub fn allocate(
        &mut self,
        width: u32,
        height: u32,
    ) -> Option<guillotiere::Allocation> {
        self.allocator.allocate(width, height)
    }

    /// Inserts [Allocation] Aquired Index for Back Mapping.
    ///
    pub fn insert_index(&mut self, index: usize) {
        self.allocated.insert(index);
    }

    /// Clears the internal Allocator and Allocated stores.
    ///
    pub fn clear(&mut self) {
        self.allocator.clear();
        self.allocated.clear();
        self.migrating = false;
    }

    /// Deallocates a [`Allocation`] returning it to the Allocator for reuse.
    ///
    pub fn deallocate(
        &mut self,
        index: usize,
        allocation: guillotiere::Allocation,
    ) {
        self.allocated.swap_remove(&index);
        self.allocator.deallocate(allocation);
    }

    /// Returns how many alloctions have been removed since the
    /// creation of the layer. this gets reset when the layer is purged.
    ///
    pub fn deallocations(&self) -> usize {
        self.allocator.deallocations()
    }

    /// Enables Migration of the Allocations inside the Texture.
    ///
    pub fn start_migration(&mut self) {
        self.migrating = true;
    }
}
