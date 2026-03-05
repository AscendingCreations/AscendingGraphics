use crate::{Allocation, Atlas, AtlasSet, GraphicsError};
use std::hash::Hash;
use wgpu::CommandEncoder;

#[derive(Debug, Default, Clone)]
/// Used to Migrate Textures from One Texture to Another to Eliminate possible fragmentation.
pub struct MigrationTask {
    /// Textures being migrated to an Avaliable Texture not migrating.
    pub migrating: Vec<usize>,
    /// Textures that dont need migration and Have extra space to try against.
    pub avaliable: Vec<usize>,
}

impl<U: Hash + Eq + Clone, Data: Copy + Default> AtlasSet<U, Data> {
    fn add_empty_layer(
        &mut self,
        task: &mut MigrationTask,
    ) -> Result<usize, GraphicsError> {
        if self.layers.len() + 1 == self.max_layers {
            return Err(GraphicsError::AtlasMaxLayers);
        }

        let layer = Atlas::new(self.size);

        self.layers.push(layer);
        task.avaliable.push(self.layers.len() - 1);
        Ok(self.layers.len() - 1)
    }

    pub(crate) fn migrate_reallocate(
        &mut self,
        task: &mut MigrationTask,
    ) -> Result<Vec<(usize, Allocation<Data>)>, GraphicsError> {
        let mut migrated = Vec::with_capacity(32);
        let migrating: Vec<(usize, Allocation<Data>)>;

        //Tis be fragmented heavily my lord. We need more layers!
        if task.avaliable.is_empty() {
            let _ = self.add_empty_layer(task)?;
        }

        let migrating_layer_id =
            task.migrating.pop().ok_or(GraphicsError::DefragFailed)?;

        // Lets Gather all the ones we want to Migrate that Exist within the Layer we
        // are working with in this round.
        if let Some(layer) = self.layers.get_mut(migrating_layer_id) {
            migrating = layer
                .allocated
                .clone()
                .into_iter()
                .filter_map(|alloc_id| {
                    if let Some((allocation, _hash)) = self.peek(alloc_id) {
                        Some((alloc_id, *allocation))
                    } else {
                        None
                    }
                })
                .collect();
        } else {
            migrating = vec![];
        }

        'outer: for (id, allocation) in migrating {
            // Try to place object in another created Texture thats not migrating.
            for layer_id in &task.avaliable {
                if let Some(layer) = self.layers.get_mut(*layer_id) {
                    let rect = allocation.allocation.rectangle;

                    if let Some(alloc) = layer
                        .allocator
                        .allocate(rect.width() as u32, rect.height() as u32)
                    {
                        migrated.push((
                            id,
                            Allocation {
                                allocation: alloc,
                                layer: *layer_id,
                                data: allocation.data,
                            },
                        ));
                        continue 'outer;
                    }
                }
            }

            // If failed to place then we do need a new layer.
            let layer_id = self.add_empty_layer(task)?;

            if let Some(layer) = self.layers.get_mut(layer_id) {
                let rect = allocation.allocation.rectangle;

                if let Some(alloc) = layer
                    .allocator
                    .allocate(rect.width() as u32, rect.height() as u32)
                {
                    migrated.push((
                        id,
                        Allocation {
                            allocation: alloc,
                            layer: layer_id,
                            data: allocation.data,
                        },
                    ));
                    continue 'outer;
                }
            }

            // We reached the end of the line. All hope is lost lets return None.
            return Err(GraphicsError::DefragFailed);
        }

        task.avaliable.push(migrating_layer_id);
        // we can clear this old layer now since we dont care about whats in here now.
        self.layers
            .get_mut(migrating_layer_id)
            .ok_or(GraphicsError::DefragFailed)?
            .clear();
        Ok(migrated)
    }

    pub fn migrate_allocation(
        &mut self,
        old_allocation: &Allocation<Data>,
        allocation: &Allocation<Data>,
        encoder: &mut CommandEncoder,
    ) {
        let (x, y) = allocation.position();
        let (width, height) = allocation.size();
        let layer = allocation.layer;

        let (o_x, o_y) = old_allocation.position();
        let o_layer = old_allocation.layer;

        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: o_x,
                    y: o_y,
                    z: o_layer as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
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
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}
