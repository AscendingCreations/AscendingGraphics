use std::{cell::RefCell, mem};

use crate::{
    Bounds, CameraView, Color, DrawOrder, GpuRenderer, GraphicsError, Index,
    TextAtlas, TextRenderer, TextUniformRaw, TextVertex, Vec2, Vec3,
    instance_buffer::OrderedIndex, parallel::*,
};
use cosmic_text::{
    Align, Attrs, Buffer, Cursor, FontSystem, Metrics, Scroll, SwashCache,
    SwashContent, Wrap,
};
use wgpu::util::align_to;

/// [`Text`] Option Handler for [`Text::measure_string`].
///
#[derive(Debug)]
pub struct TextOptions {
    pub shaping: cosmic_text::Shaping,
    pub metrics: Option<Metrics>,
    pub buffer_width: Option<f32>,
    pub buffer_height: Option<f32>,
    pub scale: f32,
    pub wrap: Wrap,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            shaping: cosmic_text::Shaping::Advanced,
            metrics: Some(Metrics::new(16.0, 16.0).scale(1.0)),
            buffer_width: None,
            buffer_height: None,
            scale: 1.0,
            wrap: Wrap::None,
        }
    }
}

/// [`Text`] visible width and lines details
///
#[derive(Debug)]
pub struct VisibleDetails {
    /// Visible Width the Text can render as.
    pub width: f32,
    /// Visible Lines the Text can Render too.
    pub lines: usize,
    /// Max Height each line is rendered as.
    pub line_height: f32,
}

#[derive(Debug)]
pub enum TextShape {
    Scroll { scroll: Scroll, prune: bool },
    Cursor { cursor: Cursor, prune: bool },
    Line { line: usize, prune: bool },
}

impl Default for TextShape {
    fn default() -> Self {
        TextShape::Scroll {
            scroll: Scroll::default(),
            prune: false,
        }
    }
}

/// Text to render to screen.
///
#[derive(Debug)]
pub struct Text {
    /// Cosmic Text [`Buffer`].
    pub buffer: Buffer,
    /// Position on the Screen.
    pub pos: Vec3,
    /// Width and Height of the Text Area.
    pub size: Vec2,
    /// Scale of the Text.
    pub scale: f32,
    /// Default Text Font Color.
    pub default_color: Color,
    /// Clip Bounds of Text.
    pub bounds: Option<Bounds>,
    /// Instance Buffer Store Index of Text Buffer.
    pub store_id: Index,
    /// the draw order of the Text. created/updated when update is called.
    pub order: DrawOrder,
    /// Text Shaper to use defaults to
    pub shaper: TextShape,
    /// Word Wrap Type. Default is Wrap::Word.
    pub wrap: Wrap,
    /// [`CameraView`] used to render with.
    pub camera_view: CameraView,
    /// ID that points to the uniform if used to help optimize position.
    pub text_uniform_id: usize,
    /// If position changed then we will see if it is set to a uniformID > 0
    pub position_changed: bool,
    /// If any changes that need to be reuploaded to the GPU occur
    pub changed: bool,
    /// If the text was recently Shaped or not.
    pub was_shaped: bool,
}

thread_local! {
    static GLYPH_VERTICES: RefCell<Vec<TextVertex>> = RefCell::new(Vec::with_capacity(1024));
}

impl Text {
    /// Updates the [`Text`]'s Buffers to prepare them for rendering.
    ///
    pub fn create_quad(
        &mut self,
        cache: &mut SwashCache,
        atlas: &mut TextAtlas,
        renderer: &mut GpuRenderer,
    ) -> Result<(), GraphicsError> {
        self.shape_now(&mut renderer.font_sys);

        let count: usize = self
            .buffer
            .lines
            .par_iter()
            .map(|line| line.text().len())
            .sum();

        let mut is_alpha = false;

        GLYPH_VERTICES.with_borrow_mut(|vertices| {
            vertices.clear();

            if vertices.capacity() < count {
                vertices.reserve(count);
            }
        });

        // From Glyphon good optimization.
        let is_run_visible = |run: &cosmic_text::LayoutRun| {
            if let Some(bounds) = self.bounds {
                let start_y = if self.text_uniform_id > 0 {
                    0.0
                } else {
                    self.pos.y
                } + self.size.y
                    - run.line_top;
                let end_y = if self.text_uniform_id > 0 {
                    0.0
                } else {
                    self.pos.y
                } + self.size.y
                    - run.line_top
                    - (run.line_height * 0.5);

                start_y <= bounds.top + (run.line_height * 0.5)
                    && bounds.bottom <= end_y
            } else {
                true
            }
        };

        self.buffer
            .layout_runs()
            .skip_while(|run| !is_run_visible(run))
            .take_while(is_run_visible)
            .for_each(|run| {
                run.glyphs.iter().for_each(|glyph| {
                    let physical_glyph = glyph.physical(
                        if self.text_uniform_id > 0 {
                            (0.0, self.size.y)
                        } else {
                            (self.pos.x, self.pos.y + self.size.y)
                        },
                        self.scale,
                    );

                    let (allocation, is_color) =
                        if let Some((allocation, is_color)) =
                            atlas.get_by_key(&physical_glyph.cache_key)
                        {
                            (allocation, is_color)
                        } else {
                            let image = cache
                                .get_image_uncached(
                                    &mut renderer.font_sys,
                                    physical_glyph.cache_key,
                                )
                                .unwrap();

                            if image.placement.width > 0
                                && image.placement.height > 0
                            {
                                atlas
                                    .upload_with_alloc(
                                        renderer,
                                        image.content == SwashContent::Color,
                                        physical_glyph.cache_key,
                                        &image,
                                    )
                                    .unwrap()
                            } else {
                                return;
                            }
                        };

                    let position = allocation.data;
                    let (u, v, width, height) = allocation.rect();
                    let (mut u, mut v, mut width, mut height) = (
                        u as f32,
                        v as f32,
                        (width as f32 * self.scale).round(),
                        (height as f32 * self.scale).round(),
                    );
                    let (mut x, mut y) = (
                        position.x + physical_glyph.x as f32,
                        (physical_glyph.y as f32
                            + ((position.y - height)
                                - (run.line_y * self.scale).round())),
                    );
                    let color = if is_color {
                        Color::rgba(255, 255, 255, 255)
                    } else {
                        glyph.color_opt.unwrap_or(self.default_color)
                    };

                    if color.a() < 255 {
                        is_alpha = true;
                    }

                    if let Some(bounds) = self.bounds {
                        // Starts beyond right edge or ends beyond left edge
                        let max_x = x + width;
                        if x > bounds.right || max_x < bounds.left {
                            return;
                        }

                        // Clip left edge
                        if x < bounds.left {
                            let right_shift = bounds.left - x;

                            x = bounds.left;
                            width = max_x - bounds.left;
                            u += right_shift;
                        }

                        // Clip right edge
                        if x + width > bounds.right {
                            width = bounds.right - x;
                        }

                        // Clip top edge
                        if y < bounds.bottom {
                            height -= bounds.bottom - y;
                            y = bounds.bottom;
                        }

                        // Clip top edge
                        if y + height > bounds.top {
                            let bottom_shift = (y + height) - bounds.top;

                            v += bottom_shift;
                            height -= bottom_shift;
                        }
                    }

                    GLYPH_VERTICES.with_borrow_mut(|vertices| {
                        vertices.push(TextVertex {
                            pos: [x, y, self.pos.z],
                            size: [width, height],
                            tex_coord: [u, v],
                            layer: allocation.layer as u32,
                            color: color.0,
                            camera_view: self.camera_view as u32,
                            is_color: is_color as u32,
                            text_id: self.text_uniform_id as u32,
                        })
                    });
                });
            });

        if let Some(store) = renderer.get_ibo_store_mut(self.store_id) {
            GLYPH_VERTICES.with_borrow(|vertices| {
                let bytes: &[u8] = bytemuck::cast_slice(vertices);

                if bytes.len() != store.store.len() {
                    store.store.resize_with(bytes.len(), || 0);
                }

                store.store.copy_from_slice(bytes);
                store.changed = true;
            });
        }

        self.order.alpha = is_alpha;

        Ok(())
    }

    /// Creates a new [`Text`] without a uniform for position optimization.
    /// Useful for Text that Never changes its position.
    ///
    /// order_layer: Rendering Layer of the Text used in DrawOrder.
    pub fn new(
        renderer: &mut GpuRenderer,
        metrics: Option<Metrics>,
        pos: Vec3,
        size: Vec2,
        scale: f32,
        order_layer: u32,
    ) -> Self {
        let text_starter_size =
            bytemuck::bytes_of(&TextVertex::default()).len() * 64;

        Self {
            buffer: Buffer::new(
                &mut renderer.font_sys,
                metrics.unwrap_or(Metrics::new(16.0, 16.0).scale(scale)),
            ),
            pos,
            size,
            bounds: None,
            store_id: renderer.new_ibo_store(text_starter_size),
            order: DrawOrder::new(false, pos, order_layer),
            changed: true,
            was_shaped: false,
            default_color: Color::rgba(0, 0, 0, 255),
            camera_view: CameraView::default(),
            shaper: TextShape::default(),
            wrap: Wrap::Word,
            scale,
            text_uniform_id: 0,
            position_changed: true,
        }
    }

    /// Creates a new [`Text`].with a uniform buffer for position optimization.
    /// If no avaliable uniforms Exist it will use buffer 0 which means no optimizations.
    ///
    /// order_layer: Rendering Layer of the Text used in DrawOrder.
    pub fn new_with_buffer(
        renderer: &mut GpuRenderer,
        text_renderer: &mut TextRenderer,
        metrics: Option<Metrics>,
        pos: Vec3,
        size: Vec2,
        scale: f32,
        order_layer: u32,
    ) -> Self {
        let mut text =
            Self::new(renderer, metrics, pos, size, scale, order_layer);

        // Attempt to get a open slot, If not we default to the unchanged index of 0.
        // if set to 0 the text is less optimized and rebuilds each position update.
        text.text_uniform_id =
            text_renderer.unused_indexs.pop_front().unwrap_or_default();
        text
    }

    /// Sets the [`Text`]'s [`CameraView`] for rendering.
    ///
    pub fn set_camera_view(&mut self, camera_view: CameraView) {
        self.camera_view = camera_view;

        self.changed = true;
    }

    /// Unloads the [`Text`] from the Instance Buffers Store and its outline from the VBO Store.
    ///
    pub fn unload(self, renderer: &mut GpuRenderer) {
        renderer.remove_ibo_store(self.store_id);
    }

    /// Unloades the [`Text`]'s Uniform Index and sets the text uniform id to 0 without a uniform buffer.
    /// Run this to Reaquire Uniform Id's before unloading Text. This does nothing if id is already 0.
    ///
    pub fn unload_text_uniform_index(
        &mut self,
        text_renderer: &mut TextRenderer,
    ) {
        if self.text_uniform_id > 0 {
            text_renderer.unused_indexs.push_front(self.text_uniform_id);
        }

        self.text_uniform_id = 0;
    }

    /// Updates the [`Text`]'s order to overide the last set position.
    /// Use this after calls to set_position to set it to a specific rendering order.
    ///
    pub fn set_order_override(&mut self, order_override: Vec3) -> &mut Self {
        self.order.set_pos(order_override);
        self
    }

    /// Updates the [`Text`]'s orders Render Layer.
    ///
    pub fn set_order_layer(&mut self, order_layer: u32) -> &mut Self {
        self.order.order_layer = order_layer;
        self
    }

    /// Resets the [`Text`] to contain the new text only.
    ///
    pub fn set_text(
        &mut self,
        text: &str,
        attrs: &Attrs,
        shaping: cosmic_text::Shaping,
        alignment: Option<Align>,
    ) -> &mut Self {
        self.buffer.set_text(text, attrs, shaping, alignment);
        self.changed = true;
        self.was_shaped = false;
        self
    }

    /// Resets the [`Text`] to contain the new span of text only.
    ///
    pub fn set_rich_text<'r, 's, I>(
        &mut self,
        spans: I,
        default_attr: &Attrs,
        shaping: cosmic_text::Shaping,
        alignment: Option<Align>,
    ) -> &mut Self
    where
        I: IntoIterator<Item = (&'s str, Attrs<'r>)>,
    {
        self.buffer
            .set_rich_text(spans, default_attr, shaping, alignment);
        self.changed = true;
        self.was_shaped = false;
        self
    }

    /// For more advanced shaping and usage. Use [`Text::set_change`] to set if you need it to make changes or not.
    /// This will not set the change to true. when changes are made you must set changed to true.
    ///
    pub fn get_text_buffer(&mut self) -> &mut Buffer {
        &mut self.buffer
    }

    /// For more advanced shaping and usage. Use [`Text::set_change`] to set if you need it to make changes or not.
    /// This will not set the change to true. when changes are made you must set changed to true.
    ///
    ///
    pub fn shape_now(&mut self, font_system: &mut FontSystem) -> &mut Self {
        if !self.was_shaped {
            match self.shaper {
                TextShape::Cursor { cursor, prune } => {
                    self.buffer.shape_until_cursor(font_system, cursor, prune);
                }
                TextShape::Scroll { scroll, prune } => {
                    self.buffer.set_scroll(scroll);
                    self.buffer.shape_until_scroll(font_system, prune);
                }
                TextShape::Line { line, prune } => {
                    self.buffer.shape_until_cursor(
                        font_system,
                        Cursor::new(line, 0),
                        prune,
                    );
                }
            }

            self.was_shaped = true;
        }

        self
    }

    /// cursor shaping sets the [`Text`]'s location to shape from and sets the buffers scroll.
    ///
    pub fn set_until_cursor(
        &mut self,
        new_cursor: Cursor,
        prune: bool,
    ) -> &mut Self {
        match self.shaper {
            TextShape::Cursor { cursor, prune: _ } if cursor == new_cursor => {}
            _ => {
                self.shaper = TextShape::Cursor {
                    cursor: new_cursor,
                    prune,
                };
                self.changed = true;
                self.was_shaped = false;
            }
        }

        self
    }

    /// Gets he [`Text`]'s buffers current Scroll location. Must be called After shape_now or update to retrieve it.
    ///
    pub fn get_scroll(&mut self) -> Scroll {
        self.buffer.scroll()
    }

    /// cursor shaping sets the [`Text`]'s location to shape from.
    /// Shaping Must be called before update.
    ///
    pub fn set_until_line(
        &mut self,
        new_line: usize,
        prune: bool,
    ) -> &mut Self {
        match self.shaper {
            TextShape::Line { line, prune: _ } if line == new_line => {}
            _ => {
                self.shaper = TextShape::Line {
                    line: new_line,
                    prune,
                };
                self.changed = true;
                self.was_shaped = false;
            }
        }

        self
    }

    /// scroll shaping sets the [`Text`]'s location to shape from.
    /// Shaping Must be called before update.
    ///
    pub fn set_until_scroll(
        &mut self,
        new_scroll: cosmic_text::Scroll,
        prune: bool,
    ) -> &mut Self {
        match self.shaper {
            TextShape::Scroll { scroll, prune: _ } if scroll == new_scroll => {}
            _ => {
                self.shaper = TextShape::Scroll {
                    scroll: new_scroll,
                    prune,
                };
                self.changed = true;
                self.was_shaped = false;
            }
        }

        self
    }

    /// Sets the [`Text`] as changed for updating.
    ///
    pub fn set_change(&mut self, changed: bool) -> &mut Self {
        self.changed = changed;
        self
    }

    /// Sets the [`Text`] as position changed for updating.
    ///
    pub fn set_position_change(&mut self, changed: bool) -> &mut Self {
        self.position_changed = changed;
        self
    }

    /// Sets the [`Text`] so it will run the shaper the next time update or shape_now is run.
    ///
    pub fn reshape(&mut self) -> &mut Self {
        self.was_shaped = false;
        self
    }

    /// Sets the [`Text`] wrapping.
    ///
    pub fn set_wrap(&mut self, wrap: Wrap) -> &mut Self {
        if self.wrap != wrap {
            self.wrap = wrap;
            self.buffer.set_wrap(wrap);
            self.changed = true;
            self.was_shaped = false;
        }

        self
    }

    /// Sets the [`Text`]'s clipping bounds.
    ///
    pub fn set_bounds(&mut self, bounds: Option<Bounds>) -> &mut Self {
        self.bounds = bounds;
        self.changed = true;
        self
    }

    /// Sets the [`Text`]'s screen Posaition.
    ///
    pub fn set_pos(&mut self, pos: Vec3) -> &mut Self {
        self.pos = pos;
        self.order.set_pos(pos);
        self.position_changed = true;
        self
    }

    /// Sets the [`Text`]'s default color.
    ///
    pub fn set_default_color(&mut self, color: Color) -> &mut Self {
        self.default_color = color;
        self.changed = true;
        self
    }

    /// Sets the [`Text`]'s cosmic text buffer size.
    ///
    pub fn set_buffer_size(
        &mut self,
        width: Option<f32>,
        height: Option<f32>,
    ) -> &mut Self {
        self.buffer.set_size(width, height);
        self.changed = true;
        self.was_shaped = false;
        self
    }

    /// clears the [`Text`] buffer.
    ///
    pub fn clear(&mut self) -> &mut Self {
        self.buffer.set_text(
            "",
            &cosmic_text::Attrs::new(),
            cosmic_text::Shaping::Basic,
            None,
        );
        self.changed = true;
        self.was_shaped = false;
        self
    }

    // Used to check and update the vertex array.
    /// Returns a [`OrderedIndex`] used in Rendering.
    /// A Shaping function Must be called before update.
    ///
    pub fn update(
        &mut self,
        text_renderer: &mut TextRenderer,
        atlas: &mut TextAtlas,
        renderer: &mut GpuRenderer,
    ) -> Result<OrderedIndex, GraphicsError> {
        if self.position_changed && self.text_uniform_id > 0 {
            let queue = renderer.queue();
            let text_uniform = TextUniformRaw {
                pos: [self.pos.x, self.pos.y],
                padding: 0.0,
            };

            let text_alignment: usize =
                align_to(mem::size_of::<TextUniformRaw>(), 16) as usize;

            queue.write_buffer(
                &text_renderer.text_buffer,
                (self.text_uniform_id * text_alignment) as wgpu::BufferAddress,
                bytemuck::bytes_of(&text_uniform),
            );

            self.position_changed = false;
        }

        if self.changed || self.position_changed {
            self.create_quad(&mut text_renderer.swash_cache, atlas, renderer)?;
            self.changed = false;
            self.position_changed = false;
        }

        Ok(OrderedIndex::new(self.order, self.store_id))
    }

    /// Checks if mouse_pos is within the [`Text`]'s location.
    ///
    pub fn check_mouse_bounds(&self, mouse_pos: Vec2) -> bool {
        mouse_pos[0] > self.pos.x
            && mouse_pos[0] < self.pos.x + self.size.x
            && mouse_pos[1] > self.pos.y
            && mouse_pos[1] < self.pos.y + self.size.y
    }

    /// Returns Visible Width and Line details of the Rendered [`Text`].
    pub fn visible_details(&self) -> VisibleDetails {
        #[cfg(not(feature = "rayon"))]
        let (width, lines) = self.buffer.layout_runs().fold(
            (0.0, 0usize),
            |(width, total_lines), run| {
                (run.line_w.max(width), total_lines + 1)
            },
        );

        #[cfg(feature = "rayon")]
        let (width, lines) = self
            .buffer
            .layout_runs()
            .par_bridge()
            .fold(
                || (0.0, 0usize),
                |(width, total_lines), run| {
                    (run.line_w.max(width), total_lines + 1)
                },
            )
            .reduce(
                || (0.0, 0usize),
                |(w1, t1), (w2, t2)| (w1.max(w2), t1 + t2),
            );

        VisibleDetails {
            line_height: self.buffer.metrics().line_height,
            lines,
            width,
        }
    }

    /// shapes then measure's the [`Text`]'s Rendering Size.
    ///
    pub fn measure(&mut self, font_system: &mut FontSystem) -> Vec2 {
        let details = self.shape_now(font_system).visible_details();

        let (max_width, max_height) = self.buffer.size();
        let height = details.lines as f32 * details.line_height;

        Vec2::new(
            details
                .width
                .min(max_width.unwrap_or(0.0).max(details.width)),
            height.min(max_height.unwrap_or(0.0).max(height)),
        )
    }

    /// Allows measuring the String for how big it will be when Rendering.
    /// This will not create any buffers in the rendering system.
    ///
    pub fn measure_string(
        font_system: &mut FontSystem,
        text: &str,
        attrs: &Attrs,
        options: TextOptions,
        alignment: Option<Align>,
        text_shape: TextShape,
    ) -> Vec2 {
        let mut buffer = Buffer::new(
            font_system,
            options
                .metrics
                .unwrap_or(Metrics::new(16.0, 16.0).scale(options.scale)),
        );

        buffer.set_wrap(options.wrap);
        buffer.set_size(options.buffer_width, options.buffer_height);
        buffer.set_text(text, attrs, options.shaping, alignment);

        match text_shape {
            TextShape::Cursor { cursor, prune } => {
                buffer.shape_until_cursor(font_system, cursor, prune);
            }
            TextShape::Scroll { scroll, prune } => {
                buffer.set_scroll(scroll);
                buffer.shape_until_scroll(font_system, prune);
            }
            TextShape::Line { line, prune } => {
                buffer.shape_until_cursor(
                    font_system,
                    Cursor::new(line, 0),
                    prune,
                );
            }
        }

        buffer.shape_until_scroll(font_system, false);

        #[cfg(not(feature = "rayon"))]
        let (width, total_lines) = buffer.layout_runs().fold(
            (0.0, 0usize),
            |(width, total_lines), run| {
                (run.line_w.max(width), total_lines + 1)
            },
        );

        #[cfg(feature = "rayon")]
        let (width, total_lines) = buffer
            .layout_runs()
            .par_bridge()
            .fold(
                || (0.0, 0usize),
                |(width, total_lines), run| {
                    (run.line_w.max(width), total_lines + 1)
                },
            )
            .reduce(
                || (0.0, 0usize),
                |(w1, t1), (w2, t2)| (w1.max(w2), t1 + t2),
            );

        let (max_width, max_height) = buffer.size();
        let height = total_lines as f32 * buffer.metrics().line_height;

        Vec2::new(
            width.min(max_width.unwrap_or(0.0).max(width)),
            height.min(max_height.unwrap_or(0.0).max(height)),
        )
    }

    /// Allows measuring the String character's Glyph and returning a Vec of their Sizes per character.
    /// This will not create any buffers in the rendering system.
    ///
    pub fn measure_glyphs(
        font_system: &mut FontSystem,
        text: &str,
        attrs: &Attrs,
        options: TextOptions,
        alignment: Option<Align>,
    ) -> Vec<Vec2> {
        let mut buffer = Buffer::new(
            font_system,
            options
                .metrics
                .unwrap_or(Metrics::new(16.0, 16.0).scale(options.scale)),
        );

        buffer.set_wrap(options.wrap);
        buffer.set_size(options.buffer_width, options.buffer_height);

        text.char_indices()
            .map(|(_position, ch)| {
                //let mut buffer = buffer.clone();

                let n = ch.len_utf8();
                let mut buf = vec![0; n];
                let u = ch.encode_utf8(&mut buf);

                buffer.set_text(u, attrs, options.shaping, alignment);
                buffer.shape_until_scroll(font_system, false);

                #[cfg(not(feature = "rayon"))]
                let (width, total_lines) = buffer.layout_runs().fold(
                    (0.0, 0usize),
                    |(width, total_lines), run| {
                        (run.line_w.max(width), total_lines + 1)
                    },
                );

                #[cfg(feature = "rayon")]
                let (width, total_lines) = buffer
                    .layout_runs()
                    .par_bridge()
                    .fold(
                        || (0.0, 0usize),
                        |(width, total_lines), run| {
                            (run.line_w.max(width), total_lines + 1)
                        },
                    )
                    .reduce(
                        || (0.0, 0usize),
                        |(w1, t1), (w2, t2)| (w1.max(w2), t1 + t2),
                    );

                let (max_width, max_height) = buffer.size();
                let height = total_lines as f32 * buffer.metrics().line_height;

                Vec2::new(
                    width.min(max_width.unwrap_or(0.0).max(width)),
                    height.min(max_height.unwrap_or(0.0).max(height)),
                )
            })
            .collect()
    }
}
