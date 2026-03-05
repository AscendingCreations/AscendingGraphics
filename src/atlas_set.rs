use crate::{
    AHashMap, AHashSet, GpuRenderer, GraphicsError, TextureGroup,
    TextureLayout, UVec3, parallel::*,
};
use lru::LruCache;
use slab::Slab;
use std::{hash::Hash, sync::Arc};
use wgpu::{BindGroup, BindGroupLayout, TextureUsages};

mod allocation;
mod allocator;
mod atlas;
mod migration;

pub use allocation::Allocation;
pub use allocator::Allocator;
pub use atlas::Atlas;
use migration::*;
/**
 * AtlasSet is used to hold and contain the data of many Atlas layers.
 * Each Atlas keeps track of the allocations allowed. Each allocation is a
 * given Width/Height as well as Position that a Texture image can fit within
 * the atlas.
 *
 * We try to use Store to keep all Allocations localized so if they need to be
 * unloaded, migrated or replaced then the system can prevent improper rendering
 * using a outdated Allocation. We will also attempt to keep track of reference counts
 * loading the Index and try to keep track of LRU cache and a list of last used Indexs.
 * This will help reduce errors and can help to reduce Vram and Later Reduce Fragmentation
 * of the Atlas.
 *
 * *******************************FRAGMENTATION********************************************
 * Fragmentation of a Atlas is when you Deallocate and Allocate new image textures into the
 * Atlas. As this occurs there is a possibility that Small spots that can not be used in the
 * Atlas to appear. These small Sections might get merged into larger Sections upon Deallocation
 * of neighboring Allocations, But in some Cases these might over run the Atlas cuasing use to
 * use way more Vram than is needed. To fix this we must migrate all loaded Allocations to a new
 * Atlas and either move the old atlas to the back of the list for reuse or unload it. We can accomplish
 * knowing when to migrate the atlas by setting a deallocations_limit. We also can know when to unload a
 * empty layer by using the layer_free_limit. This will allow us to control VRam usage.
 *
 * TODO Keep track of Indexs within an Atlas.
*/
pub struct AtlasSet<U: Hash + Eq + Clone = String, Data: Copy + Default = i32> {
    /// Texture in GRAM, Holds all the atlas layers.
    pub texture: wgpu::Texture,
    /// Layers of texture.
    pub layers: Vec<Atlas>,
    /// Holds the Texture's Size.
    pub size: u32,
    /// Store the Allocations se we can easily remove and update them.
    /// use a Generation id to avoid conflict if users use older allocation id's.
    /// Also stores the Key associated with the Allocation.
    pub store: Slab<(Allocation<Data>, U)>,
    /// for key to index lookups.
    pub lookup: AHashMap<U, usize>,
    /// keeps a list of least used allocations so we can unload them when need be.
    /// Also include the RefCount per ID lookup.
    /// we use this to keep track of when Fonts need to be unloaded.
    /// this only helps to get memory back but does not fix fragmentation of the Atlas.
    pub cache: LruCache<usize, usize>,
    /// List of allocations used in the last frame to ensure we dont unload what is
    /// in use.
    pub last_used: AHashSet<usize>,
    /// Format the Texture uses.
    pub format: wgpu::TextureFormat,
    /// When the System will Error if reached. This is the max allowed Layers
    /// Default is [`wgpu::Limits::max_texture_array_layers`]. Most GPU allow a max of 256.
    pub max_layers: usize,
    /// Limit of deallocations allowed before we attempt to migrate the textures
    /// allocations to fix fragmentation.
    pub deallocations_limit: usize,
    /// amount of layers in memory before we start checking for fragmentations.
    pub layer_check_limit: usize,
    /// When we should free empty layers. this must be more than 1 otherwise will cause
    /// issues.
    pub layer_free_limit: usize,
    /// uses the refcount to unload rather than the unused.
    /// must exist for fonts to unload correctly and must be set to false for them.
    pub use_ref_count: bool,
    /// Texture Bind group for Atlas Set
    pub texture_group: TextureGroup,
    /// Used to Migrate Textures to reduce Fragmentation.
    pub migration: Option<MigrationTask>,
}

impl<U: Hash + Eq + Clone, Data: Copy + Default> AtlasSet<U, Data> {
    fn allocate(
        &mut self,
        width: u32,
        height: u32,
        data: Data,
    ) -> Option<Allocation<Data>> {
        /* Check if the allocation would fit. */
        if width > self.size || height > self.size {
            return None;
        }

        /* Try allocating from an existing layer. */
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if layer.migrating {
                continue;
            }

            if let Some(allocation) = layer.allocator.allocate(width, height) {
                return Some(Allocation {
                    allocation,
                    layer: i,
                    data,
                });
            }
        }

        /* Try to see if we can clear out unused allocations first.
        It doesnt matter here if we are migrating  or not here this saves us some time.*/
        if !self.use_ref_count {
            loop {
                let (&id, _) = self.cache.peek_lru()?;

                //Check if ID has been used yet?
                if self.last_used.contains(&id) {
                    //Failed to find any unused allocations so lets try to add a layer.
                    break;
                }

                if let Some(layer_id) = self.remove(id) {
                    let layer = self.layers.get_mut(layer_id)?;

                    if let Some(allocation) =
                        layer.allocator.allocate(width, height)
                    {
                        return Some(Allocation {
                            allocation,
                            layer: layer_id,
                            data,
                        });
                    }
                }
            }
        }

        /* Add a new layer, as we found no layer to allocate from and could
        not retrieve any old allocations to use. */

        if self.layers.len() + 1 == self.max_layers {
            return None;
        }

        let mut layer = Atlas::new(self.size);

        if let Some(allocation) = layer.allocator.allocate(width, height) {
            self.layers.push(layer);

            if let Some(migrating) = &mut self.migration {
                migrating.avaliable.push(self.layers.len() - 1);
            }

            return Some(Allocation {
                allocation,
                layer: self.layers.len() - 1,
                data,
            });
        }

        /* We are out of luck. */
        None
    }

    //TODO Add shrink that takes layers using a unload boolean and also promote each layers.
    //TODO allocation layers to the new layer location. while removing the old empty layer.
    fn grow(&mut self, amount: usize, renderer: &GpuRenderer) {
        if amount == 0 {
            return;
        }

        let texture =
            renderer.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("Texture"),
                size: wgpu::Extent3d {
                    width: self.size,
                    height: self.size,
                    depth_or_array_layers: self.layers.len() as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[self.format],
            });

        let amount_to_copy = self.layers.len() - amount;

        let mut encoder = renderer.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Texture command encoder"),
            },
        );

        for (i, _) in self.layers.iter_mut().take(amount_to_copy).enumerate() {
            let origin = wgpu::Origin3d {
                x: 0,
                y: 0,
                z: i as u32,
            };

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.size,
                    height: self.size,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.texture = texture;
        let texture_view =
            self.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("Texture Atlas"),
                format: Some(self.format),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(self.layers.len() as u32),
                usage: Some(
                    TextureUsages::COPY_DST
                        | TextureUsages::COPY_SRC
                        | TextureUsages::TEXTURE_BINDING,
                ),
            });
        let atlas_layout: Arc<BindGroupLayout> = renderer
            .get_layout(TextureLayout)
            .expect("TextureLayout was never created.");
        self.texture_group =
            TextureGroup::from_view(renderer, texture_view, &atlas_layout);
        renderer.queue().submit(std::iter::once(encoder.finish()));
    }

    /// Creates a new [`AtlasSet`].
    ///
    /// # Arguments
    /// - format: [`wgpu::TextureFormat`] the texture layers will need to be.
    /// - use_ref_count: Mostly used for Glyph Storage and Auto Removal.
    /// - size: Used for both Width and Height. Limited to max of limits.max_texture_dimension_2d and min of 256.
    ///
    pub fn new(
        renderer: &mut GpuRenderer,
        format: wgpu::TextureFormat,
        use_ref_count: bool,
        size: u32,
    ) -> Self {
        let limits = renderer.device().limits();
        let size = size.clamp(256, limits.max_texture_dimension_2d);

        let extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: if renderer.backend == wgpu::Backend::Gl {
                2
            } else {
                1
            },
        };

        let texture =
            renderer.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("Texture"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[format],
            });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Texture Atlas"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: Some(
                TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC
                    | TextureUsages::TEXTURE_BINDING,
            ),
        });

        let atlas_layout: Arc<BindGroupLayout> =
            renderer.create_layout(TextureLayout);
        let texture_group =
            TextureGroup::from_view(renderer, texture_view, &atlas_layout);

        Self {
            texture,
            layers: if renderer.backend == wgpu::Backend::Gl {
                vec![Atlas::new(size), Atlas::new(size)]
            } else {
                vec![Atlas::new(size)]
            },
            store: Slab::with_capacity(512),
            lookup: AHashMap::new(),
            size,
            cache: LruCache::unbounded(),
            last_used: AHashSet::default(),
            format,
            max_layers: limits.max_texture_array_layers as usize,
            deallocations_limit: 32,
            layer_check_limit: (limits.max_texture_array_layers as f64 * 0.8)
                as usize,
            layer_free_limit: 3,
            use_ref_count,
            texture_group,
            migration: None,
        }
    }

    /// Uploads a new Texture Byte Array into the GPU AtlasSets Layer.
    ///
    pub fn upload_allocation(
        &mut self,
        buffer: &[u8],
        allocation: &Allocation<Data>,
        renderer: &GpuRenderer,
    ) {
        let (x, y) = allocation.position();
        let (width, height) = allocation.size();
        let layer = allocation.layer;

        renderer.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x,
                    y,
                    z: layer as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            buffer,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(
                    if self.format == wgpu::TextureFormat::Rgba8UnormSrgb {
                        4 * width
                    } else {
                        width
                    },
                ),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Clears all information of stored Textures and Allocations.
    ///
    /// This Does not Empty the [`AtlasSet`]s GPU Texture Buffer.
    /// As we normally just overwrite the buffer when we add new Allocations.
    ///
    pub fn clear(&mut self) {
        self.layers.par_iter_mut().for_each(|layer| {
            layer.allocator.clear();
        });

        self.store.clear();
        self.lookup.clear();
        self.cache.clear();
        self.last_used.clear();
    }

    /// Defragments Textures when they reach a specific threshhold.
    ///
    /// # Strategy
    /// This Function Will check to see if a migration task exists yet.
    /// If not, it will check to see if any texture layers meet the criteria of needing to
    /// be defragmented. If they are the system will generate a Migration Task and
    /// place each possibly fragmented layer into a list, defragmenting one layer per call,
    /// until all layers marked for migrating are no longer in need of migrating.
    ///
    /// # Suggested usage
    /// Call this function about once every so many frame cycles. It can be a bit slow and could
    /// Impact FPS during its processing so we only attempt to migrate one
    /// layer at a time per each call.
    ///
    /// # Returns
    /// bool indicating if any migrations took place during this run cycle so you can rebuild your
    /// Objects in GPU Memory. Always use this to know when to Rebuild Otherwise Graphical errors can
    /// Occur.
    ///
    pub fn defragment(
        &mut self,
        renderer: &GpuRenderer,
    ) -> Result<bool, GraphicsError> {
        Ok(if let Some(mut task) = self.migration.take() {
            let nlayers = self.layers.len();
            // Gather all the Texture ID's and what their new Allocations will be.
            // Also clears the old Texture to make it reusable for the next call.
            let migrate = self.migrate_reallocate(&mut task)?;

            // We need to Regrow to add any new layers that might of gotten Added due to the Defragmentation Gather.
            self.grow(self.layers.len() - nlayers, renderer);

            let mut encoder = renderer.device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Texture command encoder"),
                },
            );

            for (id, allocation) in migrate {
                let old_allocation =
                    if let Some((old_allocation, _hash)) = self.store.get(id) {
                        *old_allocation
                    } else {
                        return Err(GraphicsError::DefragFailed);
                    };

                self.migrate_allocation(
                    &old_allocation,
                    &allocation,
                    &mut encoder,
                );

                if let Some((old_allocation, _hash)) = self.store.get_mut(id) {
                    *old_allocation = allocation;
                } else {
                    return Err(GraphicsError::DefragFailed);
                }

                self.layers[allocation.layer].insert_index(id);
            }

            if !task.migrating.is_empty() {
                self.migration = Some(task);
            }

            true
        } else {
            let mut task = MigrationTask::default();
            let mut total = 0;

            let _ = self.layers.iter().inspect(|layer| {
                if layer.allocator.deallocations() >= self.deallocations_limit {
                    total += 1;
                }
            });

            if total == 0 {
                return Ok(false);
            }

            for (id, layer) in self.layers.iter_mut().enumerate() {
                if layer.allocator.deallocations() >= self.deallocations_limit {
                    task.migrating.push(id);
                    layer.start_migration();
                } else {
                    task.avaliable.push(id);
                }
            }

            if !task.migrating.is_empty() {
                self.migration = Some(task);
            }

            false
        })
    }

    /// Clears the last_used cache's.
    ///
    pub fn trim(&mut self) {
        self.last_used.clear();
    }

    /// Promotes the cache's Allocation by key making it recently used..
    ///
    pub fn promote_by_key(&mut self, key: U) {
        if let Some(id) = self.lookup.get(&key) {
            self.cache.promote(id);
            self.last_used.insert(*id);
        }
    }

    /// Promotes the cache's Allocation by index making it recently used..
    ///
    pub fn promote(&mut self, id: usize) {
        self.cache.promote(&id);
        self.last_used.insert(id);
    }

    /// Gets the [`Allocation`]'s index if it exists.
    ///
    pub fn lookup(&self, key: &U) -> Option<usize> {
        self.lookup.get(key).copied()
    }

    /// Gets using key the reference of [`Allocation`] with key if it exists.
    ///
    pub fn peek_by_key(&self, key: &U) -> Option<&(Allocation<Data>, U)> {
        if let Some(id) = self.lookup.get(key) {
            self.store.get(*id)
        } else {
            None
        }
    }

    /// Gets using index the reference of [`Allocation`] with key if it exists.
    ///
    pub fn peek(&self, id: usize) -> Option<&(Allocation<Data>, U)> {
        self.store.get(id)
    }

    /// If [`Allocation`] using key exists.
    ///
    pub fn contains_key(&self, key: &U) -> bool {
        self.lookup.contains_key(key)
    }

    /// If [`Allocation`] at id exists.
    ///
    pub fn contains(&self, id: usize) -> bool {
        self.store.contains(id)
    }

    /// Gets using key the [`Allocation`] if it exists.
    /// Also Increments the Cache and adds to last_used list.
    ///
    pub fn get_by_key(&mut self, key: &U) -> Option<Allocation<Data>> {
        let id = *self.lookup.get(key)?;
        if let Some((allocation, _)) = self.store.get(id) {
            self.cache.promote(&id);
            self.last_used.insert(id);
            return Some(*allocation);
        }

        None
    }

    /// Gets using index the [`Allocation`] if it exists.
    /// Also Increments the Cache and adds to last_used list.
    ///
    pub fn get(&mut self, id: usize) -> Option<Allocation<Data>> {
        if let Some((allocation, _)) = self.store.get(id) {
            self.cache.promote(&id);
            self.last_used.insert(id);
            return Some(*allocation);
        }

        None
    }

    /// Removed Texture by key.
    /// Removing will leave anything using the texture inable to load the correct texture if
    /// a new texture is loaded in the olds place.
    ///
    /// returns the layer id if removed otherwise None for everything else.
    ///
    pub fn remove_by_key(&mut self, key: &U) -> Option<usize> {
        let id = *self.lookup.get(key)?;
        let refcount = self.cache.pop(&id)?.saturating_sub(1);

        if self.use_ref_count && refcount > 0 {
            self.cache.push(id, refcount);
            return None;
        }

        let (allocation, _) = self.store.remove(id);
        self.last_used.remove(&id);
        self.lookup.remove(key);
        self.layers
            .get_mut(allocation.layer)?
            .deallocate(id, allocation.allocation);
        Some(allocation.layer)
    }

    /// Removed Texture by index.
    /// Removing will leave anything using the texture inable to load the correct texture if
    /// a new texture is loaded in the olds place.
    ///
    /// returns the layer id if removed otherwise None for everything else.
    ///
    pub fn remove(&mut self, id: usize) -> Option<usize> {
        let refcount = self.cache.pop(&id)?.saturating_sub(1);

        if self.use_ref_count && refcount > 0 {
            self.cache.push(id, refcount);
            return None;
        }

        let (allocation, key) = self.store.remove(id);
        self.last_used.remove(&id);
        self.lookup.remove(&key);
        self.layers
            .get_mut(allocation.layer)?
            .deallocate(id, allocation.allocation);
        Some(allocation.layer)
    }

    /// Uploads Texture byte array to the AtlasSet returning the created [`Allocation`]s Index.
    ///
    /// # Arguments
    /// - bytes: Textures Byte array.
    /// - width: Width of the Texture.
    /// - height: Height of the Texture.
    /// - data: any specail generic data for the texture.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &mut self,
        key: U,
        bytes: &[u8],
        width: u32,
        height: u32,
        data: Data,
        renderer: &GpuRenderer,
    ) -> Option<usize> {
        if let Some(&id) = self.lookup.get(&key) {
            Some(id)
        } else {
            let allocation = {
                let nlayers = self.layers.len();
                let allocation = self.allocate(width, height, data)?;
                self.grow(self.layers.len() - nlayers, renderer);

                allocation
            };

            self.upload_allocation(bytes, &allocation, renderer);
            let id = self.store.insert((allocation, key.clone()));
            self.layers[allocation.layer].insert_index(id);
            self.lookup.insert(key, id);
            self.cache.push(id, 1);
            Some(id)
        }
    }

    /// Uploads Texture byte array to the AtlasSet returning the created [`Allocation`] and Index.
    ///
    /// # Arguments
    /// - bytes: Textures Byte array.
    /// - width: Width of the Texture.
    /// - height: Height of the Texture.
    /// - data: any specail generic data for the texture.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn upload_with_alloc(
        &mut self,
        key: U,
        bytes: &[u8],
        width: u32,
        height: u32,
        data: Data,
        renderer: &GpuRenderer,
    ) -> Option<(usize, Allocation<Data>)> {
        if let Some(&id) = self.lookup.get(&key) {
            let (allocation, _) = self.store.get(id)?;
            Some((id, *allocation))
        } else {
            let allocation = {
                let nlayers = self.layers.len();
                let allocation = self.allocate(width, height, data)?;
                self.grow(self.layers.len() - nlayers, renderer);

                allocation
            };

            self.upload_allocation(bytes, &allocation, renderer);
            let id = self.store.insert((allocation, key.clone()));
            self.layers[allocation.layer].insert_index(id);
            self.lookup.insert(key.clone(), id);
            self.cache.push(id, 1);
            Some((id, allocation))
        }
    }

    /// Creates an [`Allocation`] without uploading bytes to the texture.
    /// Useful for Placeholders of stuff that doesnt render.
    /// used to avoid hitting extra cache writes. This will waste space
    /// and can be used to Precaching allocations for uploading later too.
    /// returns the created [`Allocation`] and Index
    ///
    /// # Arguments
    /// - width: Width of the Texture. if 0 will default to 1.
    /// - height: Height of the Texture. if 0 will default to 1.
    /// - data: any specail generic data for the texture.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn set_alloc(
        &mut self,
        key: U,
        width: u32,
        height: u32,
        data: Data,
        renderer: &GpuRenderer,
    ) -> Option<(usize, Allocation<Data>)> {
        let (width, height) = (width.max(1), height.max(1));

        if let Some(&id) = self.lookup.get(&key) {
            let (allocation, _) = self.store.get(id)?;
            Some((id, *allocation))
        } else {
            let allocation = {
                let nlayers = self.layers.len();
                let allocation = self.allocate(width, height, data)?;
                self.grow(self.layers.len() - nlayers, renderer);

                allocation
            };

            let id = self.store.insert((allocation, key.clone()));
            self.layers[allocation.layer].insert_index(id);
            self.lookup.insert(key.clone(), id);
            self.cache.push(id, 1);
            Some((id, allocation))
        }
    }

    /// Returns the Width and Height of the [`AtlasSet`] and how many Layers Exist.
    ///
    pub fn size(&self) -> UVec3 {
        UVec3::new(self.size, self.size, self.layers.len() as u32)
    }

    /// Returns a [`BindGroup`] Reference to the AtlasSets Texture Binding.
    pub fn bind_group(&self) -> &BindGroup {
        &self.texture_group.bind_group
    }

    /// Returns a [`BindGroup`] Reference to the AtlasSets Texture Binding.
    pub fn with_deallocations_limit(mut self, limit: usize) -> Self {
        self.deallocations_limit = limit;
        self
    }
}
