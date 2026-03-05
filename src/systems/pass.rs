/// The `Pass` trait represents either a render or compute pass. This way the passes can be
/// implemented in a modular way.
pub trait Pass {
    /// Encodes the commands of the current pass. In addition this function has access to all the
    /// texture views such that it can use them as color attachments or as depth stencil
    /// attachment. In the future it should also be possible to use these textures as inputs.
    fn render(
        &mut self,
        renderer: &crate::GpuRenderer,
        encoder: &mut wgpu::CommandEncoder,
    );
}
