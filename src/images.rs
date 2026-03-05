mod pipeline;
mod render;
mod vertex;

pub use pipeline::*;
pub use render::*;
pub use vertex::*;

use crate::{
    AtlasSet, Bounds, CameraView, Color, DrawOrder, FlipStyle, GpuRenderer,
    Index, OrderedIndex, Vec2, Vec3, Vec4,
};

/// Basic and Fast Image Rendering Type. Best used for Sprites and Objects in the world.
///
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    /// Position of the object
    pub pos: Vec3,
    /// Height and Width
    pub size: Vec2,
    /// Static texture offsets or animation frame positions
    pub uv: Vec4,
    /// Color.
    pub color: Color,
    /// Global Camera the Shader will use to render the object with
    pub camera_view: CameraView,
    /// Texture area location in Atlas.
    pub texture: Option<usize>,
    /// Buffer's store Index.
    pub store_id: Index,
    /// Ordering Type, used to order the Stores in the buffers.
    pub order: DrawOrder,
    /// Clip bounds if enabled in the renderer.
    pub bounds: Option<Bounds>,
    /// Directional Flip.
    pub flip_style: FlipStyle,
    /// direct angle of rotation from the center Axis.
    pub rotation_angle: f32,
    /// When true tells system to update the buffers.
    pub changed: bool,
}

impl Image {
    /// Creates a new [`Image`] with rendering order layer, pos, size and uv.
    ///
    /// order_layer: Rendering Order Layer used in DrawOrder.
    pub fn new(
        texture: Option<usize>,
        renderer: &mut GpuRenderer,
        pos: Vec3,
        size: Vec2,
        uv: Vec4,
        order_layer: u32,
    ) -> Self {
        Self {
            pos,
            size,
            uv,
            camera_view: CameraView::default(),
            color: Color::rgba(255, 255, 255, 255),
            texture,
            store_id: renderer.new_buffer(
                bytemuck::bytes_of(&ImageVertex::default()).len(),
                0,
            ),
            order: DrawOrder::new(false, Vec3::default(), order_layer),
            bounds: None,
            flip_style: FlipStyle::None,
            rotation_angle: 0.0,
            changed: true,
        }
    }

    /// Unloads the [`Image`] from the Instance Buffers Store.
    ///
    pub fn unload(self, renderer: &mut GpuRenderer) {
        renderer.remove_buffer(self.store_id);
    }

    /// Updates the [`Image`]'s order_override.
    /// Use this after calls to set_position to set it to a specific rendering order.
    ///
    pub fn set_order_override(&mut self, order_override: Vec3) -> &mut Self {
        self.order.set_pos(order_override);
        self
    }

    /// Updates the [`Image`]'s Optional Clipping Bounds.
    ///
    pub fn update_bounds(&mut self, bounds: Option<Bounds>) -> &mut Self {
        self.bounds = bounds;
        self
    }

    /// Updates the [`Image`]'s [`FlipStyle`].
    ///
    pub fn set_flip_style(&mut self, flip_style: FlipStyle) -> &mut Self {
        self.changed = true;
        self.flip_style = flip_style;
        self
    }

    /// Updates the [`Image`]'s rotation.
    ///
    pub fn set_rotation_angle(&mut self, rotation_angle: f32) -> &mut Self {
        self.changed = true;
        self.rotation_angle = rotation_angle;
        self
    }

    /// Updates the [`Image`]'s position.
    ///
    pub fn set_pos(&mut self, pos: Vec3) -> &mut Self {
        self.changed = true;
        self.pos = pos;
        self.order.set_pos(pos);
        self
    }

    /// Updates the [`Image`]'s width and height.
    ///
    pub fn set_size(&mut self, size: Vec2) -> &mut Self {
        self.changed = true;
        self.size = size;
        self
    }

    /// Updates the [`Image`]'s texture x, y, w, h
    ///
    pub fn set_uv(&mut self, uv: Vec4) -> &mut Self {
        self.changed = true;
        self.uv = uv;
        self
    }

    /// Updates the [`Image`]'s rendering layer.
    ///
    pub fn set_order_layer(&mut self, order_layer: u32) -> &mut Self {
        self.order.order_layer = order_layer;
        self
    }

    /// Updates the [`Image`]'s Texture ID.
    ///
    pub fn set_texture(&mut self, texture: Option<usize>) -> &mut Self {
        self.changed = true;
        self.texture = texture;
        self
    }

    /// Updates the [`Image`]'s [`Color`].
    ///
    pub fn set_color(&mut self, color: Color) -> &mut Self {
        self.changed = true;
        self.order.alpha = color.a() < 255;
        self.color = color;
        self
    }

    /// Updates the [`Image`]'s [`DrawOrder`]'s is Alpha.
    /// Use this after set_color to overide the alpha sorting.
    ///
    pub fn set_order_alpha(&mut self, alpha: bool) -> &mut Self {
        self.order.alpha = alpha;
        self
    }

    /// Updates the [`Image`]'s [`CameraView`].
    ///
    pub fn set_camera_view(&mut self, camera_view: CameraView) -> &mut Self {
        self.changed = true;
        self.camera_view = camera_view;
        self
    }

    /// Updates the [`Image`]'s Buffers to prepare them for rendering.
    ///
    fn create_quad(
        &mut self,
        renderer: &mut GpuRenderer,
        atlas: &mut AtlasSet,
    ) {
        let allocation = match &self.texture {
            Some(id) => {
                if let Some(allocation) = atlas.get(*id) {
                    allocation
                } else {
                    return;
                }
            }
            None => return,
        };

        let (u, v, width, height) = allocation.rect();
        let tex_data = (
            self.uv.x + u as f32,
            self.uv.y + v as f32,
            self.uv.z.min(width as f32),
            self.uv.w.min(height as f32),
        );

        let instance = ImageVertex {
            pos: self.pos.to_array(),
            size: self.size.to_array(),
            tex_data: tex_data.into(),
            color: self.color.0,
            camera_view: self.camera_view as u32,
            layer: allocation.layer as i32,
            flip_style: self.flip_style as u32,
            angle: self.rotation_angle,
        };

        if let Some(store) = renderer.get_buffer_mut(self.store_id) {
            let bytes = bytemuck::bytes_of(&instance);

            if bytes.len() != store.store.len() {
                store.store.resize_with(bytes.len(), || 0);
            }

            store.store.copy_from_slice(bytes);
            store.changed = true;
        }
    }

    /// Used to check and update the vertex array.
    /// Returns a [`OrderedIndex`] used in Rendering.
    ///
    pub fn update(
        &mut self,
        renderer: &mut GpuRenderer,
        atlas: &mut AtlasSet,
    ) -> OrderedIndex {
        if self.changed {
            self.create_quad(renderer, atlas);
            self.changed = false;
        }

        OrderedIndex::new_with_bounds(
            self.order,
            self.store_id,
            0,
            self.bounds,
            self.camera_view,
        )
    }
}
