use crate::{
    AtlasSet, Bounds, CameraView, DrawOrder, GpuRenderer, GraphicsError, Index,
    OrderedIndex, OtherError, RectVertex, Texture, Vec2, Vec3, Vec4,
};
use cosmic_text::Color;

/// Rectangle to render to screen.
/// Can contain a Images otherwise just colors.
///
#[derive(Debug)]
pub struct Rect {
    /// Position on the Screen.
    pub pos: Vec3,
    /// Width and Height of the Rect.
    pub size: Vec2,
    /// Color of the Rect.
    pub color: Color,
    /// Optional Image Index.
    pub image: Option<usize>,
    /// Texture X, Y, W and H if any apply.
    pub uv: Vec4,
    /// Width of the Rects Border.
    pub border_width: f32,
    /// Color of the Rects Border.
    pub border_color: Color,
    /// Rectangle Radius.
    pub radius: f32,
    /// [`CameraView`] used to render with.
    pub camera_view: CameraView,
    /// Instance Buffers Store ID.
    pub store_id: Index,
    /// the draw order of the rect. created/updated when update is called.
    pub order: DrawOrder,
    /// Optional Bounds for Clipping the Rect too.
    pub bounds: Option<Bounds>,
    /// If anything got updated we need to update the buffers too.
    pub changed: bool,
}

impl Rect {
    /// Creates a new [`Rect`] with rendering order layer, position and size.
    ///
    /// order_layer: Rendering Order Layer of the rect used in DrawOrder.
    pub fn new(
        renderer: &mut GpuRenderer,
        pos: Vec3,
        size: Vec2,
        color: Color,
        order_layer: u32,
    ) -> Self {
        let rect_size = bytemuck::bytes_of(&RectVertex::default()).len();

        Self {
            pos,
            size,
            color,
            image: None,
            uv: Vec4::default(),
            border_width: 0.0,
            border_color: Color::rgba(0, 0, 0, 0),
            radius: 0.0,
            camera_view: CameraView::default(),
            store_id: renderer.new_buffer(rect_size, 0),
            order: DrawOrder::new(false, Vec3::default(), order_layer),
            bounds: None,
            changed: true,
        }
    }

    /// Creates a new [`Rect`] with rendering order layer, position and size, UV and Image
    ///
    /// order_layer: Rendering Order Layer of the rect used in DrawOrder.
    pub fn new_with(
        renderer: &mut GpuRenderer,
        image: Option<usize>,
        pos: Vec3,
        size: Vec2,
        uv: Vec4,
        order_layer: u32,
    ) -> Self {
        let rect_size = bytemuck::bytes_of(&RectVertex::default()).len();

        Self {
            pos,
            size,
            color: Color::rgba(255, 255, 255, 255),
            image,
            uv,
            border_width: 0.0,
            border_color: Color::rgba(0, 0, 0, 0),
            radius: 0.0,
            camera_view: CameraView::default(),
            store_id: renderer.new_buffer(rect_size, 0),
            order: DrawOrder::new(false, Vec3::default(), order_layer),
            bounds: None,
            changed: true,
        }
    }

    /// Unloads the [`Rect`] from the Instance Buffers Store.
    ///
    pub fn unload(self, renderer: &mut GpuRenderer) {
        renderer.remove_buffer(self.store_id);
    }

    /// Updates the [`Rect`]'s order to overide the last set position.
    /// Use this after calls to set_position to set it to a specific rendering order.
    ///
    pub fn set_order_pos(&mut self, order_override: Vec3) -> &mut Self {
        self.order.set_pos(order_override);
        self
    }

    /// Updates the [`Rect`]'s orders Render Layer.
    ///
    pub fn set_order_layer(&mut self, order_layer: u32) -> &mut Self {
        self.order.order_layer = order_layer;
        self
    }

    /// Updates the [`Rect`]'s [`DrawOrder`]'s is Alpha.
    /// Use this after set_color, set_border_color or set_radius to overide the alpha sorting.
    ///
    pub fn set_order_alpha(&mut self, alpha: bool) -> &mut Self {
        self.order.alpha = alpha;
        self
    }

    /// Updates the [`Rect`]'s Clipping Bounds.
    ///
    pub fn update_bounds(&mut self, bounds: Option<Bounds>) -> &mut Self {
        self.bounds = bounds;
        self
    }

    /// Sets the [`Rect`]'s [`CameraView`] for rendering.
    ///
    pub fn set_camera_view(&mut self, camera_view: CameraView) -> &mut Self {
        self.camera_view = camera_view;
        self.changed = true;
        self
    }

    /// Sets the [`Rect`]'s Color.
    ///
    pub fn set_color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self.order.alpha = self.border_color.a() < 255
            || self.radius > 0.0
            || self.color.a() < 255;
        self.changed = true;
        self
    }

    /// Sets the [`Rect`]'s Border Color.
    ///
    pub fn set_border_color(&mut self, color: Color) -> &mut Self {
        self.border_color = color;
        self.order.alpha = self.border_color.a() < 255
            || self.radius > 0.0
            || self.color.a() < 255;
        self.changed = true;
        self
    }

    /// Sets the [`Rect`]'s Texture.
    ///
    pub fn set_texture(
        &mut self,
        renderer: &GpuRenderer,
        atlas: &mut AtlasSet,
        path: String,
    ) -> Result<&mut Self, GraphicsError> {
        let (id, allocation) =
            Texture::upload_from_with_alloc(path, atlas, renderer)
                .ok_or_else(|| OtherError::new("failed to upload image"))?;

        let rect = allocation.rect();

        self.uv = Vec4::new(0.0, 0.0, rect.2 as f32, rect.3 as f32);
        self.image = Some(id);
        self.changed = true;
        Ok(self)
    }

    /// Sets the [`Rect`]'s Texture X,Y, W, H details.
    ///
    pub fn set_container_uv(&mut self, uv: Vec4) -> &mut Self {
        self.uv = uv;
        self.changed = true;
        self
    }

    /// Sets the [`Rect`]'s Position.
    ///
    pub fn set_pos(&mut self, pos: Vec3) -> &mut Self {
        self.pos = pos;
        self.order.set_pos(pos);
        self.changed = true;
        self
    }

    /// Sets the [`Rect`]'s Width and Height.
    ///
    pub fn set_size(&mut self, size: Vec2) -> &mut Self {
        self.size = size;
        self.changed = true;
        self
    }

    /// Sets the [`Rect`]'s Border Width.
    ///
    pub fn set_border_width(&mut self, size: f32) -> &mut Self {
        self.border_width = size;
        self.changed = true;
        self
    }

    /// Sets the [`Rect`]'s Corner Radius.
    ///
    pub fn set_radius(&mut self, radius: f32) -> &mut Self {
        self.radius = radius;
        self.order.alpha =
            self.border_color.a() < 255 || radius > 0.0 || self.color.a() < 255;
        self.changed = true;
        self
    }

    /// Updates the [`Rect`]'s Buffers to prepare them for rendering.
    ///
    pub fn create_quad(
        &mut self,
        renderer: &mut GpuRenderer,
        atlas: &mut AtlasSet,
    ) {
        let (uv, layer) = if let Some(id) = self.image {
            let tex = match atlas.get(id) {
                Some(tex) => tex,
                None => return,
            };
            let (u, v, width, height) = tex.rect();
            (
                [
                    self.uv.x + u as f32,
                    self.uv.y + v as f32,
                    self.uv.z.min(width as f32),
                    self.uv.w.min(height as f32),
                ],
                tex.layer as u32,
            )
        } else {
            ([0.0, 0.0, 0.0, 0.0], 0)
        };

        let instance = RectVertex {
            pos: self.pos.to_array(),
            size: self.size.to_array(),
            border_width: self.border_width,
            radius: self.radius,
            uv,
            layer,
            color: self.color.0,
            border_color: self.border_color.0,
            camera_view: self.camera_view as u32,
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
        // if points added or any data changed recalculate paths.
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

    /// Checks if the Mouse position is within the Rects location.
    ///
    pub fn check_mouse_bounds(&self, mouse_pos: Vec2) -> bool {
        if self.radius > 0.0 {
            let pos = [self.pos.x, self.pos.y];

            let inner_size = [
                self.size.x - self.radius * 2.0,
                self.size.y - self.radius * 2.0,
            ];
            let top_left = [pos[0] + self.radius, pos[1] + self.radius];
            let bottom_right =
                [top_left[0] + inner_size[0], top_left[1] + inner_size[1]];

            let top_left_distance =
                [top_left[0] - mouse_pos.x, top_left[1] - mouse_pos.y];
            let bottom_right_distance =
                [mouse_pos.x - bottom_right[0], mouse_pos.y - bottom_right[1]];

            let dist = [
                top_left_distance[0].max(bottom_right_distance[0]).max(0.0),
                top_left_distance[1].max(bottom_right_distance[1]).max(0.0),
            ];

            let dist = (dist[0] * dist[0] + dist[1] * dist[1]).sqrt();

            dist < self.radius
        } else {
            mouse_pos[0] > self.pos.x
                && mouse_pos[0] < self.pos.x + self.size.x
                && mouse_pos[1] > self.pos.y
                && mouse_pos[1] < self.pos.y + self.size.y
        }
    }
}
