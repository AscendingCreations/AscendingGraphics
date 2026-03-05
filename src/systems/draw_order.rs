use crate::{Bounds, CameraView, Vec3};
use slotmap::new_key_type;
use std::cmp::Ordering;

new_key_type! {
    pub struct AscendingKey;
}

/// Buffer Index Re-type.
pub type Index = AscendingKey;

/// Draw Order in which Buffers are sorted by for optimal rendering.
///
/// Positions are all calculated as (pos * 10000.0) as u32 to increase speed of sorting.
/// Sort Order is order_layer -> alpha -> y reversed -> x -> z reversed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Default)]
pub struct DrawOrder {
    /// Layer to sort the buffer by. This is not the same as buffer_layer.
    /// Sorted by lowest to highest. First to Sort by.
    pub order_layer: u32,
    /// If the Buffer includes any Alpha Rendering.
    /// This placed the buffer first above none Alpha in the Order Layer.
    pub alpha: bool,
    /// X Position on the Screen. Sorted After Y.
    /// Sorted by lowest to highest.
    pub x: u32,
    /// Y Position on the Screen. Sorted After Alpha.
    /// Sorted by highest to lowest.
    pub y: u32,
    /// Z Position on the Screen. Sorted After X.
    /// Sorted by highest to lowest.
    pub z: u32,
}

impl PartialOrd for DrawOrder {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DrawOrder {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order_layer
            .cmp(&other.order_layer)
            .then(self.alpha.cmp(&other.alpha))
            .then(self.y.cmp(&other.y).reverse())
            .then(self.x.cmp(&other.x))
            .then(self.z.cmp(&other.z).reverse())
    }
}

impl DrawOrder {
    /// Creates a DrawOrder with alpha, position and order_layer.
    pub fn new(alpha: bool, pos: Vec3, order_layer: u32) -> Self {
        Self {
            order_layer,
            alpha,
            x: (pos.x * 10000.0) as u32,
            y: (pos.y * 10000.0) as u32,
            z: (pos.z * 10000.0) as u32,
        }
    }

    pub fn set_pos(&mut self, pos: Vec3) {
        self.x = (pos.x * 10000.0) as u32;
        self.y = (pos.y * 10000.0) as u32;
        self.z = (pos.z * 10000.0) as u32;
    }
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
    /// Stores buffers optional Bounds for scissor clipping.
    pub(crate) bounds: Option<Bounds>,
    /// Stores the buffers Camera Type for Rendering Aspects.
    pub(crate) camera_view: CameraView,
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
            bounds: None,
            camera_view: CameraView::MainView,
        }
    }

    /// Creates a OrderedIndex with DrawOrder, Buffer Index and Index Max,
    /// Clip bounds and Camera Type.
    pub fn new_with_bounds(
        order: DrawOrder,
        index: Index,
        index_max: u32,
        bounds: Option<Bounds>,
        camera_view: CameraView,
    ) -> Self {
        Self {
            order,
            index,
            index_count: 0,
            index_max,
            bounds,
            camera_view,
        }
    }
}
