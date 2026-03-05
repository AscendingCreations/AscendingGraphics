use thiserror::Error;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct OtherError {
    details: String,
}

impl std::error::Error for OtherError {}

impl std::fmt::Display for OtherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl OtherError {
    pub fn new(msg: &str) -> OtherError {
        OtherError {
            details: msg.to_string(),
        }
    }
}

#[derive(Debug, Error)]
pub enum GraphicsError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Surface(#[from] wgpu::SurfaceError),
    #[error(transparent)]
    WGpu(#[from] wgpu::Error),
    #[error(transparent)]
    Device(#[from] wgpu::RequestDeviceError),
    #[error(transparent)]
    Adapter(#[from] wgpu::RequestAdapterError),
    #[error(transparent)]
    ImageError(#[from] image::ImageError),
    #[error("Image atlas has no more space.")]
    AtlasFull,
    #[error(transparent)]
    LyonTessellation(#[from] lyon::lyon_tessellation::TessellationError),
    #[error(transparent)]
    Other(#[from] OtherError),
    #[error(transparent)]
    EventLoop(#[from] winit::error::EventLoopError),
    #[error(transparent)]
    EventLoopExternal(#[from] winit::error::ExternalError),
    #[error(transparent)]
    OsError(#[from] winit::error::OsError),
    #[error(
        "Could not Defragment the Textures due to Failure in Wgpu or other issues."
    )]
    DefragFailed,
    #[error("The Atlas already has max layers and can not hold anymore.")]
    AtlasMaxLayers,
    #[error(
        "No Adpater could be created by the system. Maybe its not supported?"
    )]
    AdapterNotFound,
}
