use crate::{Allocation, AtlasSet, GpuRenderer, GraphicsError, TileSheet};
use image::{DynamicImage, GenericImageView, ImageFormat};
use std::{io::Error, path::Path};

/// Holds the Textures information for Uploading to the GPU.
#[derive(Clone, Debug, Default)]
pub struct Texture {
    /// full path.
    name: String,
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
        let name = path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::other("could not convert name to String"))?
            .to_owned();

        Ok(Self::from_image(name, image::open(path)?))
    }

    /// Creates a [`Texture`] from loaded File and uploads it to an [`AtlasSet`].
    /// Returns Associated [`AtlasSet`] Index.
    ///
    pub fn upload_from(
        path: impl AsRef<Path>,
        atlas: &mut AtlasSet<String, i32>,
        renderer: &GpuRenderer,
    ) -> Option<usize> {
        let name = path.as_ref().to_str()?.to_owned();

        if let Some(id) = atlas.lookup(&name) {
            Some(id)
        } else {
            let texture = Texture::from_file(path).ok()?;
            let (width, height) = texture.size();
            atlas.upload(name, texture.bytes(), width, height, 0, renderer)
        }
    }

    /// Creates a [`Texture`] from loaded File and uploads it to an [`AtlasSet`].
    /// Returns Associated [`AtlasSet`] Index and [`Allocation`].
    ///
    pub fn upload_from_with_alloc(
        path: impl AsRef<Path>,
        atlas: &mut AtlasSet<String, i32>,
        renderer: &GpuRenderer,
    ) -> Option<(usize, Allocation)> {
        let name = path.as_ref().to_str()?.to_owned();

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
    pub fn from_image(name: String, image: DynamicImage) -> Self {
        let size = image.dimensions();
        let bytes = image.into_rgba8().into_raw();

        Self { name, bytes, size }
    }

    /// Creates a [`Texture`] from Memory.
    ///
    pub fn from_memory(
        name: String,
        data: &[u8],
    ) -> Result<Self, GraphicsError> {
        Ok(Self::from_image(name, image::load_from_memory(data)?))
    }

    /// Creates a [`Texture`] from Memory with [`ImageFormat`].
    ///
    pub fn from_memory_with_format(
        name: String,
        data: &[u8],
        format: ImageFormat,
    ) -> Result<Self, GraphicsError> {
        Ok(Self::from_image(
            name,
            image::load_from_memory_with_format(data, format)?,
        ))
    }

    /// Uploads a [`Texture`] into an [`AtlasSet`].
    /// Returns Associated [`AtlasSet`] Index.
    ///
    pub fn upload(
        &self,
        atlas: &mut AtlasSet<String, i32>,
        renderer: &GpuRenderer,
    ) -> Option<usize> {
        let (width, height) = self.size;
        atlas.upload(self.name.clone(), &self.bytes, width, height, 0, renderer)
    }

    /// Uploads a [`Texture`] into an [`AtlasSet`].
    /// Returns Associated [`AtlasSet`] Index and [`Allocation`].
    ///
    pub fn upload_with_alloc(
        &self,
        atlas: &mut AtlasSet<String, i32>,
        renderer: &GpuRenderer,
    ) -> Option<(usize, Allocation)> {
        let (width, height) = self.size;
        atlas.upload_with_alloc(
            self.name.clone(),
            &self.bytes,
            width,
            height,
            0,
            renderer,
        )
    }

    /// Splits the Texture into Tiles.
    /// Returns a Optional new [`TileSheet`] upon completion.
    ///
    pub fn new_tilesheet(
        self,
        atlas: &mut AtlasSet<String, i32>,
        renderer: &GpuRenderer,
        tilesize: u32,
    ) -> Option<TileSheet> {
        TileSheet::new(self, renderer, atlas, tilesize)
    }

    /// Splits the Texture into Tiles and Appends them to the tilesheet.
    /// Returns Some(()) upon completion.
    ///
    pub fn tilesheet_upload(
        self,
        tilesheet: &mut TileSheet,
        atlas: &mut AtlasSet<String, i32>,
        renderer: &GpuRenderer,
        tilesize: u32,
    ) -> Option<()> {
        tilesheet.upload(self, renderer, atlas, tilesize)
    }

    /// Returns Path of the Texture.
    ///
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Returns Width and Height of the Texture.
    ///
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
}
