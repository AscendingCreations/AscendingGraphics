/// [`guillotiere::AtlasAllocator`] handler for [`crate::AtlasSet`].
///
pub struct Allocator {
    /// [`guillotiere::AtlasAllocator`] holding Allocations for Textures.
    allocator: guillotiere::AtlasAllocator,
    /// Amount of Allocations currently Held.
    allocations: usize,
    /// Amount of Current Deallocations since last clear or creation.
    deallocations: usize,
}

impl Allocator {
    /// Returns a new Allocation if Room exists within the Texture layer.
    ///
    pub fn allocate(
        &mut self,
        width: u32,
        height: u32,
    ) -> Option<guillotiere::Allocation> {
        let allocation = self
            .allocator
            .allocate(guillotiere::Size::new(width as i32, height as i32))?;

        self.allocations += 1;

        Some(allocation)
    }

    /// Clears the Allocator and its counters.
    ///
    pub fn clear(&mut self) {
        self.allocator.clear();
        self.allocations = 0;
        self.deallocations = 0;
    }

    /// Removes a Allocation to make it usable again.
    ///
    pub fn deallocate(&mut self, allocation: guillotiere::Allocation) {
        self.allocator.deallocate(allocation.id);

        self.allocations = self.allocations.saturating_sub(1);
        self.deallocations = self.deallocations.saturating_add(1);
    }

    /// If there are no Allocations.
    ///
    pub fn is_empty(&self) -> bool {
        self.allocations == 0
    }

    /// How many deallocations have been made. Used for defragmentation.
    ///
    pub fn deallocations(&self) -> usize {
        self.deallocations
    }

    /// Creates a new [`Allocator`] layer.
    ///
    pub fn new(size: u32) -> Self {
        let allocator = guillotiere::AtlasAllocator::new(
            guillotiere::Size::new(size as i32, size as i32),
        );

        Self {
            allocator,
            allocations: 0,
            deallocations: 0,
        }
    }
}
