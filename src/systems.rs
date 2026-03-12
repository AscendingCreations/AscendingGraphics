mod arena;
mod bounds;
mod buffer;
mod device;
mod draw_order;
mod ibo_store;
pub mod instance_buffer;
mod layout;
mod pass;
mod pipelines;
mod renderer;
mod static_vbo;
mod system;
pub mod vbo;
mod vbo_store;

pub use arena::*;
pub use bounds::Bounds;
pub use buffer::{
    AsBufferPass, BufferData, BufferLayout, BufferPass, GpuBuffer,
};
pub use device::*;
pub use draw_order::{DrawOrder, Index};
pub use ibo_store::IBOStore;
pub use instance_buffer::{InstanceBuffer, InstanceDetails};
pub use layout::*;
pub use pass::*;
pub use pipelines::*;
pub use renderer::*;
pub use slotmap::KeyData;
pub use static_vbo::*;
pub use system::*;
pub use vbo::{IndexDetails, VertexBuffer};
pub use vbo_store::*;

pub(crate) use ahash::{AHashMap, AHashSet, AHasher};

pub(crate) type ABuildHasher = std::hash::BuildHasherDefault<AHasher>;
pub(crate) type AIndexSet<K> = indexmap::IndexSet<K, ABuildHasher>;

/// Type of Camera view and Scale to use within the Shader per rendered Object.
///
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CameraView {
    #[default]
    MainView,
    SubView1,
    SubView2,
    SubView3,
    SubView4,
    SubView5,
    SubView6,
    SubView7,
}

/// Type of Texture Flipping in shader.
///
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum FlipStyle {
    #[default]
    None,
    Horizontal,
    Vertical,
    Both,
}
