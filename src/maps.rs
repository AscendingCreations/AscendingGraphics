mod pipeline;
mod render;
mod uniforms;
mod vertex;

use crate::{
    AtlasSet, CameraView, DrawOrder, GpuRenderer, Index, OrderedIndex, UVec2,
    UVec3, Vec2, Vec3, parallel::*,
};
use cosmic_text::Color;
pub use pipeline::*;
pub use render::*;
use std::{cell::RefCell, iter, mem};
pub use uniforms::*;
pub use vertex::*;
use wgpu::util::align_to;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialOrd, Ord, Eq, Hash, PartialEq)]
pub enum MapLayers {
    Ground,
    Mask,
    /// Mask 2 is the Z layer spacer for bridges.
    Mask2,
    Anim1,
    Anim2,
    Anim3,
    Anim4,
    /// always above player. \/
    Fringe,
    Fringe2,
    Count,
}

impl MapLayers {
    pub const LOWER_LAYERS: [Self; 7] = [
        Self::Ground,
        Self::Mask,
        Self::Mask2,
        Self::Anim1,
        Self::Anim2,
        Self::Anim3,
        Self::Anim4,
    ];

    pub const UPPER_LAYERS: [Self; 2] = [Self::Fringe, Self::Fringe2];

    pub fn indexed_layers(self, zlayer: &MapZLayers) -> f32 {
        match self {
            Self::Ground => zlayer.ground,
            Self::Mask => zlayer.mask,
            Self::Mask2 => zlayer.mask2,
            Self::Anim1 => zlayer.anim1,
            Self::Anim2 => zlayer.anim2,
            Self::Anim3 => zlayer.anim3,
            Self::Anim4 => zlayer.anim4,
            Self::Fringe => zlayer.fringe,
            _ => zlayer.fringe2,
        }
    }

    pub fn as_str<'a>(self) -> &'a str {
        match self {
            Self::Ground => "Ground",
            Self::Mask => "Mask",
            Self::Mask2 => "Mask 2",
            Self::Anim1 => "Anim 1",
            Self::Anim2 => "Anim 2",
            Self::Anim3 => "Anim 3",
            Self::Anim4 => "Anim 4",
            Self::Fringe => "Fringe",
            _ => "Fringe 2",
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapZLayers {
    pub ground: f32,
    pub mask: f32,
    pub mask2: f32,
    pub anim1: f32,
    pub anim2: f32,
    pub anim3: f32,
    pub anim4: f32,
    /// always above player. \/
    pub fringe: f32,
    pub fringe2: f32,
}

impl Default for MapZLayers {
    fn default() -> Self {
        Self {
            ground: 9.6,
            mask: 9.5,
            mask2: 9.4,
            anim1: 9.3,
            anim2: 9.2,
            anim3: 9.1,
            anim4: 9.0,
            fringe: 5.1,
            fringe2: 5.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, Ord, Eq, Hash, PartialEq)]
pub struct TileData {
    ///tiles allocation ID within the texture.
    pub id: usize,
    /// Color Offset of the Tile
    pub color: Color,
    /// Timer for animation switch. Note: Each Layer on the same Position must match to work correctly
    /// for Anim layer 1 - 4.
    pub anim_time: u32,
}

impl Default for TileData {
    fn default() -> Self {
        Self {
            id: 0,
            color: Color::rgba(255, 255, 255, 255),
            anim_time: 250,
        }
    }
}

/// Generic map upper and lower layer size defaults. will get overridden when using map::new_width
pub const TILE_COUNT: usize = 9216;
/// Generic map lower layers size defaults. will get overridden when using map::new_width
pub const LOWER_COUNT: usize = 7168;
/// Generic map upper layers size defaults. will get overridden when using map::new_width
pub const UPPER_COUNT: usize = 2048;

/// Handler for rendering Map to GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct Map {
    /// X, Y, GroupID for loaded map.
    /// Add this to the higher up Map struct.
    /// pub world_pos: Vec3,
    /// its render position. within the screen.
    pub pos: Vec2,
    /// Width and Height of a Map in tiles.
    pub size: UVec2,
    // tiles per layer.
    pub tiles: Vec<TileData>,
    /// Store index per each layer.
    pub stores: [Index; 2],
    /// the draw order of the maps. created when update is called.
    pub orders: [DrawOrder; 2],
    /// count if any Filled Tiles Exist. this is to optimize out empty maps in rendering.
    pub filled_tiles: [u16; MapLayers::Count as usize],
    /// The size of the Tile to render. for spacing tiles out upon
    /// vertex creation. Default will be 20.
    pub tilesize: u32,
    /// Used to deturmine if the map can be rendered or if its just a preload.
    pub can_render: bool,
    pub camera_view: CameraView,
    /// Each layers Z position. Default is 9.6-9.0 for lower levels and 5.1-5.0 for upper.
    pub zlayers: MapZLayers,
    /// If tiles vertex data got changed.
    pub tiles_changed: bool,
    /// If the uniform map data got changed.
    pub map_changed: bool,
    /// Index of map in uniform Data.
    pub map_index: usize,
}

// These are used to Reduce the Overall Memory usage of each and every map loaded and allow them all to process though
// a single point of memory which should help with cache locality.
thread_local! {
    static LOWER_BUFFER: RefCell<Vec<TileVertex>> = RefCell::new(Vec::with_capacity(LOWER_COUNT));
    static UPPER_BUFFER: RefCell<Vec<TileVertex>> = RefCell::new(Vec::with_capacity(UPPER_COUNT));
}

impl Map {
    fn generate_layer_vertexes(
        &self,
        vertexs: &mut Vec<TileVertex>,
        atlas: &AtlasSet,
        layer: MapLayers,
    ) {
        if self.filled_tiles[layer as usize] == 0 {
            return;
        }

        let z = layer.indexed_layers(&self.zlayers);
        let atlas_width = atlas.size().x / self.tilesize;
        let max_tiles = self.size.x * self.size.y;

        let mut data: Vec<TileVertex> = (0..max_tiles)
            .into_par_iter()
            .filter_map(|id| {
                let (x, y) = ((id % self.size.x), (id / self.size.x));
                let tile =
                    &self.tiles[(id + (layer as u32 * max_tiles)) as usize];

                if tile.id == 0 {
                    return None;
                }

                if let Some((allocation, _)) = atlas.peek(tile.id) {
                    let (posx, posy) = allocation.position();

                    Some(TileVertex {
                        pos: [
                            (x * self.tilesize) as f32,
                            (y * self.tilesize) as f32,
                            z,
                        ],
                        tile_id: (posx / self.tilesize)
                            + ((posy / self.tilesize) * atlas_width),
                        texture_layer: allocation.layer as u32,
                        color: tile.color.0,
                        map_layer: layer as u32,
                        map_index: self.map_index as u32,
                        anim_time: tile.anim_time,
                    })
                } else {
                    None
                }
            })
            .collect();

        vertexs.append(&mut data);
    }

    /// Updates the [`Map`]'s Buffers to prepare them for rendering.
    ///
    pub fn create_quad(
        &mut self,
        renderer: &mut GpuRenderer,
        atlas: &mut AtlasSet,
    ) {
        LOWER_BUFFER.with_borrow_mut(|lower_buffer| {
            lower_buffer.clear();

            MapLayers::LOWER_LAYERS.into_iter().for_each(|layer| {
                self.generate_layer_vertexes(lower_buffer, atlas, layer)
            });

            if let Some(store) = renderer.get_buffer_mut(self.stores[0]) {
                let bytes = bytemuck::cast_slice(lower_buffer);

                if bytes.len() != store.store.len() {
                    store.store.resize_with(bytes.len(), || 0);
                }

                store.store.copy_from_slice(bytes);
                store.changed = true;
            }
        });

        UPPER_BUFFER.with_borrow_mut(|upper_buffer| {
            upper_buffer.clear();

            MapLayers::UPPER_LAYERS.into_iter().for_each(|layer| {
                self.generate_layer_vertexes(upper_buffer, atlas, layer)
            });

            if let Some(store) = renderer.get_buffer_mut(self.stores[1]) {
                let bytes = bytemuck::cast_slice(upper_buffer);

                if bytes.len() != store.store.len() {
                    store.store.resize_with(bytes.len(), || 0);
                }

                store.store.copy_from_slice(bytes);
                store.changed = true;
            }
        });
    }

    pub fn set_visibility(
        &mut self,
        renderer: &mut GpuRenderer,
        visible: bool,
    ) {
        if !self.can_render && visible {
            for i in 0..=1 {
                if let Some(store) = renderer.get_buffer_mut(self.stores[i]) {
                    store.changed = true;
                }
            }
        }

        self.can_render = visible;
    }

    /// Creates a new [`Map`] with tilesize and a default size of [32, 32].
    ///
    pub fn new(
        renderer: &mut GpuRenderer,
        map_render: &mut MapRenderer,
        tilesize: u32,
        pos: Vec2,
        zlayers: MapZLayers,
    ) -> Option<Self> {
        let map_index = map_render.unused_indexs.pop_front()?;
        let map_vertex_size = bytemuck::bytes_of(&TileVertex::default()).len();
        let lower_index = renderer.new_buffer(map_vertex_size * LOWER_COUNT, 0);
        let upper_index = renderer.new_buffer(map_vertex_size * UPPER_COUNT, 0);
        let order1 =
            DrawOrder::new(false, Vec3::new(pos.x, pos.y, zlayers.anim4), 0);
        let order2 =
            DrawOrder::new(false, Vec3::new(pos.x, pos.y, zlayers.fringe2), 1);

        Some(Self {
            tiles: iter::repeat_n(TileData::default(), 9216).collect(),
            pos,
            stores: [lower_index, upper_index],
            filled_tiles: [0; MapLayers::Count as usize],
            orders: [order1, order2],
            tilesize,
            can_render: true,
            tiles_changed: true,
            map_changed: true,
            camera_view: CameraView::default(),
            zlayers,
            size: UVec2::new(32, 32),
            map_index,
        })
    }

    /// Creates a new [`Map`] with tilesize position, and size.
    ///
    pub fn new_with(
        renderer: &mut GpuRenderer,
        map_render: &mut MapRenderer,
        tilesize: u32,
        pos: Vec2,
        size: UVec2,
        zlayers: MapZLayers,
    ) -> Option<Self> {
        let map_index = map_render.unused_indexs.pop_front()?;
        let map_vertex_size = bytemuck::bytes_of(&TileVertex::default()).len();
        let lower_index = renderer
            .new_buffer(map_vertex_size * ((size.x * size.y) * 7) as usize, 0);
        let upper_index = renderer
            .new_buffer(map_vertex_size * ((size.x * size.y) * 2) as usize, 0);
        let order1 =
            DrawOrder::new(false, Vec3::new(pos.x, pos.y, zlayers.anim4), 0);
        let order2 =
            DrawOrder::new(false, Vec3::new(pos.x, pos.y, zlayers.fringe2), 1);

        //Since this is different than default we do want to resize the limit to avoid multiple resizes in a render loop.
        if ((size.x * size.y) * 7) as usize > LOWER_COUNT {
            LOWER_BUFFER.with_borrow_mut(|buffer| {
                if buffer.capacity() < ((size.x * size.y) * 7) as usize {
                    buffer.reserve_exact(
                        ((size.x * size.y) * 7) as usize - buffer.len(),
                    );
                }
            });
        }

        if ((size.x * size.y) * 2) as usize > UPPER_COUNT {
            UPPER_BUFFER.with_borrow_mut(|buffer| {
                if buffer.capacity() < ((size.x * size.y) * 2) as usize {
                    buffer.reserve_exact(
                        ((size.x * size.y) * 2) as usize - buffer.len(),
                    );
                }
            });
        }

        Some(Self {
            tiles: iter::repeat_n(
                TileData::default(),
                ((size.x * size.y) * 9) as usize,
            )
            .collect(),
            pos,
            stores: [lower_index, upper_index],
            filled_tiles: [0; MapLayers::Count as usize],
            orders: [order1, order2],
            tilesize,
            can_render: true,
            tiles_changed: true,
            map_changed: true,
            camera_view: CameraView::default(),
            size,
            zlayers,
            map_index,
        })
    }

    /// Updates the [`Map`]'s position.
    ///
    pub fn set_pos(&mut self, pos: Vec2) -> &mut Self {
        self.orders[0].set_pos(Vec3::new(pos.x, pos.y, self.zlayers.anim4));
        self.orders[1].set_pos(Vec3::new(pos.x, pos.y, self.zlayers.fringe2));
        self.pos = pos;
        self.map_changed = true;
        self
    }

    /// Updates the [`Map`]'s Tile layer z positions.
    ///
    pub fn set_zlayers(&mut self, zlayers: MapZLayers) -> &mut Self {
        self.orders[0].set_pos(Vec3::new(
            self.pos.x,
            self.pos.y,
            zlayers.anim4,
        ));
        self.orders[1].set_pos(Vec3::new(
            self.pos.x,
            self.pos.y,
            zlayers.fringe2,
        ));
        self.zlayers = zlayers;
        self.tiles_changed = true;
        self
    }

    /// Updates the [`Map`]'s orders to overide the last set position.
    /// Use this after calls to set_position to set it to a order.
    ///
    pub fn set_order_pos(&mut self, order_override: Vec2) -> &mut Self {
        self.orders[0].set_pos(Vec3::new(
            order_override.x,
            order_override.y,
            self.zlayers.anim4,
        ));
        self.orders[1].set_pos(Vec3::new(
            order_override.x,
            order_override.y,
            self.zlayers.fringe2,
        ));

        self
    }

    /// Updates one of the [`Map`]'s order Layer.
    ///
    /// Default Orders for Layer 1 is 0 Layer 2 is 1.
    pub fn set_order_layer(
        &mut self,
        index: usize,
        order_layer: u32,
    ) -> &mut Self {
        if let Some(order) = self.orders.get_mut(index) {
            order.order_layer = order_layer;
        }

        self
    }

    /// Unloades the [`Map`]'s buffer from the buffer store.
    ///
    pub fn unload(
        &self,
        renderer: &mut GpuRenderer,
        map_render: &mut MapRenderer,
    ) {
        for index in &self.stores {
            renderer.remove_buffer(*index);
        }

        map_render.unused_indexs.push_front(self.map_index);
    }

    /// Unloades the [`Map`]'s Index and sets can_render to false.
    ///
    pub fn unload_map_index(&mut self, map_render: &mut MapRenderer) {
        map_render.unused_indexs.push_front(self.map_index);
        self.can_render = false;
    }

    /// aquires a new [`Map`]'s Index and sets can_render, tiles and map to true.
    ///
    pub fn aquire_map_index(
        &mut self,
        map_render: &mut MapRenderer,
    ) -> Option<()> {
        let index = map_render.unused_indexs.pop_front()?;
        self.map_index = index;
        self.can_render = true;
        self.tiles_changed = true;
        self.map_changed = true;

        Some(())
    }

    /// gets the [`TileData`] based upon the tiles x, y, and [`MapLayers`].
    /// [`MapLayers::Ground`] is Layer 0.
    ///
    pub fn get_tile(&self, pos: UVec3) -> TileData {
        assert!(
            pos.x < self.size.x || pos.y < self.size.y || pos.z < 9,
            "pos is invalid. X < {}, y < {}, z < 9",
            self.size.x,
            self.size.y
        );

        self.tiles[(pos.x
            + (pos.y * self.size.y)
            + (pos.z * (self.size.x * self.size.y)))
            as usize]
    }

    /// Sets the [`CameraView`] this object will use to Render with.
    ///
    pub fn set_camera_view(&mut self, camera_view: CameraView) -> &mut Self {
        self.camera_view = camera_view;
        self.map_changed = true;
        self
    }

    /// This sets the tile's whole Data per layer.
    /// This also increments or deincrements a Filled tile count to help speed up shader Vertex generation.
    /// and avoid processing unused layers.
    /// This also will loop set the anim_timer for all layers if different from the current tiles.
    ///
    pub fn set_tile(&mut self, pos: UVec3, tile: TileData) {
        if pos.x >= self.size.x || pos.y >= self.size.y || pos.z >= 9 {
            return;
        }

        let tilepos = (pos.x
            + (pos.y * self.size.y)
            + (pos.z * (self.size.x * self.size.y)))
            as usize;
        let current_tile = self.tiles[tilepos];

        if current_tile.anim_time != tile.anim_time {
            for z in 0..9 {
                let tile_pos = (pos.x
                    + (pos.y * self.size.y)
                    + (z * (self.size.x * self.size.y)))
                    as usize;

                self.tiles[tile_pos].anim_time = tile.anim_time;
            }
        }

        if (current_tile.id > 0 && current_tile.color.a() > 0)
            && (tile.color.a() == 0 || tile.id == 0)
        {
            self.filled_tiles[pos.z as usize] =
                self.filled_tiles[pos.z as usize].saturating_sub(1);
        } else if tile.color.a() > 0
            && tile.id > 0
            && (current_tile.id == 0 || current_tile.color.a() == 0)
        {
            self.filled_tiles[pos.z as usize] =
                self.filled_tiles[pos.z as usize].saturating_add(1);
        }

        self.tiles[tilepos].color = tile.color;
        self.tiles[tilepos].id = tile.id;
        self.tiles_changed = true;
    }

    /// This sets the all layered Tiles anim_time within the X,Y location.
    /// This is to help prevent timing issues between tile layers.
    /// This does not increment or deincrement the tile data.
    ///
    pub fn set_tile_anim_timer(&mut self, pos: UVec2, anim_time: u32) {
        if pos.x >= self.size.x || pos.y >= self.size.y {
            return;
        }

        for z in 0..9 {
            let tilepos = (pos.x
                + (pos.y * self.size.y)
                + (z * (self.size.x * self.size.y)))
                as usize;

            self.tiles[tilepos].anim_time = anim_time;
        }

        self.tiles_changed = true;
    }

    /// Used to check and update the [`Map`]'s Buffer for Rendering.
    /// Returns an Optional vec![Lower, Upper] [`OrderedIndex`] to use in Rendering.
    ///
    pub fn update(
        &mut self,
        renderer: &mut GpuRenderer,
        atlas: &mut AtlasSet,
        map_buffer: &MapRenderer,
    ) -> Option<(OrderedIndex, OrderedIndex)> {
        if self.can_render {
            if self.tiles_changed {
                self.create_quad(renderer, atlas);
                self.tiles_changed = false;
            }

            if self.map_changed {
                let queue = renderer.queue();
                let map = MapRaw {
                    pos: self.pos.to_array(),
                    tilesize: self.tilesize as f32,
                    camera_view: self.camera_view as u32,
                };

                let map_alignment: usize =
                    align_to(mem::size_of::<MapRaw>(), 16) as usize;

                queue.write_buffer(
                    &map_buffer.map_buffer,
                    (self.map_index * map_alignment) as wgpu::BufferAddress,
                    bytemuck::bytes_of(&map),
                );

                self.map_changed = false;
            }

            Some((
                OrderedIndex::new(self.orders[0], self.stores[0], 0),
                OrderedIndex::new(self.orders[1], self.stores[1], 0),
            ))
        } else {
            None
        }
    }
}
