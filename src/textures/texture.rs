use crate::{Allocation, AtlasSet, GpuRenderer, GraphicsError, TileSheet};
use image::{DynamicImage, GenericImageView, ImageFormat};
use std::{hash::Hash, path::Path};
/// Holds the Textures information for Uploading to the GPU.
#[derive(Clone, Debug, Default)]
pub struct Texture {
    /// Loaded bytes of the Texture.
    pub bytes: Vec<u8>,
    /// Width and Height of the Texture.
    size: (u32, u32),
}

impl Texture {
    /// Returns a reference to bytes.
    ///
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Creates a [`Texture`] from loaded File.
    ///
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, GraphicsError> {
        Ok(Self::from_image(image::open(path)?))
    }

    /// Creates a [`Texture`] from loaded File and uploads it to an [`AtlasSet`].
    /// Returns Associated [`AtlasSet`] Index.
    ///
    pub fn upload_from<U: Hash + Eq>(
        name: U,
        path: impl AsRef<Path>,
        atlas: &mut AtlasSet<i32>,
        renderer: &GpuRenderer,
    ) -> Option<usize> {
        if let Some(id) = atlas.lookup(&name) {
            Some(id)
        } else {
            let texture = Texture::from_file(path).ok()?;
            let (width, height) = texture.size();
            atlas.upload(name, texture.bytes(), width, height, 0, renderer)
        }
    }

    /// Creates a [`Texture`] from Memory and uploads it to an [`AtlasSet`].
    /// Returns Associated [`AtlasSet`] Index.
    ///
    pub fn upload_from_memory<U: Hash + Eq>(
        name: U,
        data: &[u8],
        atlas: &mut AtlasSet<i32>,
        renderer: &GpuRenderer,
    ) -> Option<usize> {
        if let Some(id) = atlas.lookup(&name) {
            Some(id)
        } else {
            let texture = Texture::from_memory(data).ok()?;
            let (width, height) = texture.size();
            atlas.upload(name, texture.bytes(), width, height, 0, renderer)
        }
    }

    /// Creates a [`Texture`] from loaded File and uploads it to an [`AtlasSet`].
    /// Returns Associated [`AtlasSet`] Index and [`Allocation`].
    ///
    pub fn upload_from_with_alloc<U: Hash + Eq>(
        name: U,
        path: impl AsRef<Path>,
        atlas: &mut AtlasSet<i32>,
        renderer: &GpuRenderer,
    ) -> Option<(usize, Allocation)> {
        if let Some(id) = atlas.lookup(&name) {
            atlas.peek(id).map(|(allocation, _)| (id, *allocation))
        } else {
            let texture = Texture::from_file(path).ok()?;
            let (width, height) = texture.size();
            atlas.upload_with_alloc(
                name,
                texture.bytes(),
                width,
                height,
                0,
                renderer,
            )
        }
    }

    /// Creates a [`Texture`] from [`DynamicImage`].
    ///
    pub fn from_image(image: DynamicImage) -> Self {
        let size = image.dimensions();
        let bytes = image.into_rgba8().into_raw();

        Self { bytes, size }
    }

    /// Creates a [`Texture`] from Memory.
    ///
    pub fn from_memory(data: &[u8]) -> Result<Self, GraphicsError> {
        Ok(Self::from_image(image::load_from_memory(data)?))
    }

    /// Creates a [`Texture`] from Memory with [`ImageFormat`].
    ///
    pub fn from_memory_with_format(
        data: &[u8],
        format: ImageFormat,
    ) -> Result<Self, GraphicsError> {
        Ok(Self::from_image(image::load_from_memory_with_format(
            data, format,
        )?))
    }

    /// Uploads a [`Texture`] into an [`AtlasSet`].
    /// Returns Associated [`AtlasSet`] Index.
    ///
    pub fn upload<U: Hash + Eq>(
        &self,
        name: U,
        atlas: &mut AtlasSet<i32>,
        renderer: &GpuRenderer,
    ) -> Option<usize> {
        let (width, height) = self.size;
        atlas.upload(name, &self.bytes, width, height, 0, renderer)
    }

    /// Uploads a [`Texture`] into an [`AtlasSet`].
    /// Returns Associated [`AtlasSet`] Index and [`Allocation`].
    ///
    pub fn upload_with_alloc<U: Hash + Eq>(
        &self,
        name: U,
        atlas: &mut AtlasSet<i32>,
        renderer: &GpuRenderer,
    ) -> Option<(usize, Allocation)> {
        let (width, height) = self.size;
        atlas.upload_with_alloc(name, &self.bytes, width, height, 0, renderer)
    }

    /// Splits the Texture into Tiles.
    /// Returns a Optional new [`TileSheet`] upon completion.
    ///
    pub fn new_tilesheet(
        self,
        tileset_name: &str,
        atlas: &mut AtlasSet<i32>,
        renderer: &GpuRenderer,
        tilesize: u32,
    ) -> Option<TileSheet> {
        TileSheet::new(tileset_name, self, renderer, atlas, tilesize)
    }

    /// Splits the Texture into Tiles and Appends them to the tilesheet.
    /// Returns Some(()) upon completion.
    ///
    pub fn tilesheet_upload(
        self,
        tileset_name: &str,
        tilesheet: &mut TileSheet,
        atlas: &mut AtlasSet<i32>,
        renderer: &GpuRenderer,
        tilesize: u32,
    ) -> Option<()> {
        tilesheet.upload(tileset_name, self, renderer, atlas, tilesize)
    }

    /// Returns Width and Height of the Texture.
    ///
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
}
