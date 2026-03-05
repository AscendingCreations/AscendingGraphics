use crate::{Bounds, CameraView, GpuDevice, GpuRenderer, Layout};
use bytemuck::{Pod, Zeroable};
use camera::Projection;
use glam::{Mat4, Vec2, Vec3, Vec4};
use time::FrameTime;
use wgpu::util::DeviceExt;

/// System Layout send to all the Shaders for struct Global.
#[repr(C)]
#[derive(Clone, Copy, Hash, Pod, Zeroable)]
pub struct SystemLayout;

impl Layout for SystemLayout {
    fn create_layout(
        &self,
        gpu_device: &mut GpuDevice,
    ) -> wgpu::BindGroupLayout {
        gpu_device.device().create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("system_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            },
        )
    }
}

/// System handler that keeps track of Data needed for the shaders struct Global.
#[derive(Debug)]
pub struct System<Controls: camera::controls::Controls> {
    /// Camera controller to use to get the
    /// projection, view, eye, inverse view and scale.
    camera: camera::Camera<Controls>,
    /// Screen Size used within the shaders.
    pub screen_size: [f32; 2],
    /// Buffer to shader struct Global
    global_buffer: wgpu::Buffer,
    /// Bind group for shader struct Global.
    bind_group: wgpu::BindGroup,
    /// Camera Views.
    views: [Mat4; 8],
    /// Camera Scales.
    scales: [f32; 8],
    /// If changed or not for uploading.
    changed: [bool; 8],
}

impl<Controls> System<Controls>
where
    Controls: camera::controls::Controls,
{
    /// Returns a reference too [`wgpu::BindGroup`].
    ///
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Returns a reference too [`Controls`].
    ///
    pub fn controls(&self) -> &Controls {
        self.camera.controls()
    }

    /// Returns a mutable reference too [`Controls`].
    ///
    pub fn controls_mut(&mut self) -> &mut Controls {
        self.camera.controls_mut()
    }

    /// Returns the eye positions.
    ///
    pub fn eye(&self) -> [f32; 3] {
        self.camera.eye()
    }

    /// Creates a new [`System`]
    ///
    pub fn new(
        renderer: &mut GpuRenderer,
        projection: Projection,
        controls: Controls,
        screen_size: [f32; 2],
    ) -> Self {
        let mut camera = camera::Camera::new(projection, controls);

        camera.update(0.0);

        // Create the camera uniform.
        let proj = camera.projection();
        let main_view = camera.view();
        let inverse_proj: Mat4 = (proj).inverse();
        let eye = camera.eye();
        let main_scale = camera.scale();
        let seconds = 0.0;
        let mut raw = [0f32; 204];

        raw[..16].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&main_view)[..]);

        for i in 1..8 {
            raw[(16 * i)..16 + (16 * i)].copy_from_slice(
                &AsRef::<[f32; 16]>::as_ref(&Mat4::IDENTITY)[..],
            );
        }

        raw[128] = main_scale;

        for i in 1..8 {
            raw[128 + (i * 4)] = 1.0;
        }

        raw[160..176].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&proj)[..]);
        raw[176..192]
            .copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&inverse_proj)[..]);
        raw[192..195].copy_from_slice(&eye);
        raw[196..198].copy_from_slice(&screen_size);
        raw[200] = seconds;

        // Create the uniform buffers.
        let global_buffer = renderer.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("camera buffer"),
                contents: bytemuck::cast_slice(&raw),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Create the bind group layout for the camera.
        let layout = renderer.create_layout(SystemLayout);

        // Create the bind group.
        let bind_group =
            renderer
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: global_buffer.as_entire_binding(),
                    }],
                    label: Some("system_bind_group"),
                });

        Self {
            camera,
            screen_size,
            global_buffer,
            bind_group,
            changed: [false, false, false, false, false, false, false, false],
            scales: [main_scale, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            views: [
                main_view,
                Mat4::IDENTITY,
                Mat4::IDENTITY,
                Mat4::IDENTITY,
                Mat4::IDENTITY,
                Mat4::IDENTITY,
                Mat4::IDENTITY,
                Mat4::IDENTITY,
            ],
        }
    }

    /// Returns the Projection Matrix 4x4.
    ///
    pub fn projection(&self) -> Mat4 {
        self.camera.projection()
    }

    /// Sets the Camera controls to a new controls.
    ///
    pub fn set_controls(&mut self, controls: Controls) -> Controls {
        self.camera.set_controls(controls)
    }

    /// Sets Camera Project to a new Projection.
    ///
    pub fn set_projection(&mut self, projection: Projection) {
        self.camera.set_projection(projection);
    }

    /// Sets Manual to a new View and Scale.
    /// This can set MainView but upon Update MainView gets reset by the camera.
    ///
    pub fn set_view(
        &mut self,
        camera_view: CameraView,
        view: Mat4,
        scale: f32,
    ) {
        let id = camera_view as usize;
        self.views[id] = view;
        self.scales[id] = scale;
        self.changed[id] = true;
    }

    /// Returns a views Matrix 4x4.
    ///
    pub fn get_view(&self, camera_view: CameraView) -> Mat4 {
        let id = camera_view as usize;
        self.views[id]
    }

    /// Returns mutable reference to a views Matrix 4x4.
    /// This can return MainView but upon Update MainView gets reset by the camera.
    ///
    pub fn get_view_mut(&mut self, camera_view: CameraView) -> &mut Mat4 {
        let id = camera_view as usize;

        self.changed[id] = true;
        &mut self.views[id]
    }

    /// Returns a Views Scale.
    ///
    pub fn get_scale(&self, camera_view: CameraView) -> f32 {
        let id = camera_view as usize;
        self.scales[id]
    }

    /// Returns mutable reference to a Views Scale.
    /// This can return MainView but upon Update MainView gets reset by the camera.
    ///
    pub fn get_scale_mut(&mut self, camera_view: CameraView) -> &mut f32 {
        let id = camera_view as usize;

        self.changed[id] = true;
        &mut self.scales[id]
    }

    /// Updates the GPU's shader struct Global with new Projections, Time, Views, and Scale changes.
    /// This will update the MainView automatically when camera gets changed.
    ///
    pub fn update(&mut self, renderer: &GpuRenderer, frame_time: &FrameTime) {
        if self.camera.update(frame_time.delta_seconds()) {
            let proj = self.camera.projection();
            let inv_proj: Mat4 = (proj).inverse();
            let eye = self.camera.eye();

            self.views[0] = self.camera.view();
            self.scales[0] = self.camera.scale();
            self.changed[0] = true;

            let mut raw = [0f32; 36];
            raw[..16].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&proj)[..]);
            raw[16..32]
                .copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&inv_proj)[..]);
            raw[32..35].copy_from_slice(&eye);

            renderer.queue().write_buffer(
                &self.global_buffer,
                640,
                bytemuck::cast_slice(&raw),
            );
        }

        renderer.queue().write_buffer(
            &self.global_buffer,
            800,
            bytemuck::bytes_of(&frame_time.seconds()),
        );

        for i in 0..8 {
            if self.changed[i] {
                let mut raw = [0f32; 16];

                raw[..16].copy_from_slice(
                    &AsRef::<[f32; 16]>::as_ref(&self.views[i])[..],
                );

                renderer.queue().write_buffer(
                    &self.global_buffer,
                    (i * 64) as u64,
                    bytemuck::cast_slice(&raw),
                );
            }
        }

        for i in 0..8 {
            if self.changed[i] {
                let mut raw = [0f32; 4];

                raw[0] = self.scales[i];

                renderer.queue().write_buffer(
                    &self.global_buffer,
                    512 + (i * 16) as u64,
                    bytemuck::cast_slice(&raw),
                );
            }
        }

        for i in 0..8 {
            self.changed[i] = false;
        }
    }

    /// Updates the GPU's shader struct Global with new screen size information.
    ///
    pub fn update_screen(
        &mut self,
        renderer: &GpuRenderer,
        screen_size: [f32; 2],
    ) {
        if self.screen_size != screen_size {
            self.screen_size = screen_size;

            renderer.queue().write_buffer(
                &self.global_buffer,
                784,
                bytemuck::cast_slice(&screen_size),
            );
        }
    }

    /// Returns the Internal Cameras view Matrix 4x4
    ///
    pub fn view(&self) -> Mat4 {
        self.camera.view()
    }

    /// Used to convert bounds information from World into Screen locations with view.
    ///
    pub fn projected_world_to_screen(
        &self,
        camera_view: CameraView,
        bounds: &Bounds,
    ) -> Vec4 {
        let height = f32::abs(bounds.top - bounds.bottom);
        let projection = self.camera.projection();
        let model = Mat4::IDENTITY;
        let view_id = camera_view as usize;
        let view = self.views[view_id];
        let scale = self.scales[view_id];

        let clip_coords = projection
            * view
            * model
            * Vec4::new(bounds.left, bounds.bottom, 1.0, 1.0);
        let coords = Vec3::from_slice(&clip_coords.to_array()) / clip_coords.w;

        let xy = Vec2::new(
            (coords.x + 1.0) * 0.5 * self.screen_size[0],
            (1.0 - coords.y) * 0.5 * self.screen_size[1],
        );

        let (bw, bh, objh) =
            (bounds.right * scale, bounds.top * scale, height * scale);

        Vec4::new(xy.x, xy.y - objh, bw, bh)
    }

    /// Used to convert bounds information from World into Screen locations without view.
    ///
    pub fn world_to_screen(
        &self,
        camera_view: CameraView,
        bounds: &Bounds,
    ) -> Vec4 {
        let height = f32::abs(bounds.top - bounds.bottom);
        let projection = self.camera.projection();
        let model = Mat4::IDENTITY;
        let clip_coords = projection
            * model
            * Vec4::new(bounds.left, bounds.bottom, 1.0, 1.0);
        let coords = Vec3::from_slice(&clip_coords.to_array()) / clip_coords.w;
        let view_id = camera_view as usize;
        let scale = self.scales[view_id];

        let xy = Vec2::new(
            (coords.x + 1.0) * 0.5 * self.screen_size[0],
            (1.0 - coords.y) * 0.5 * self.screen_size[1],
        );

        let (bw, bh, objh) =
            (bounds.right * scale, bounds.top * scale, height * scale);

        Vec4::new(xy.x, xy.y - objh, bw, bh)
    }

    /// Used to convert bounds information from World into Screen locations.
    ///
    pub fn world_to_screen_direct(
        screen_size: [f32; 2],
        scale: f32,
        projection: Mat4,
        left: f32,
        bottom: f32,
        right: f32,
        top: f32,
    ) -> Vec4 {
        let height = f32::abs(top - bottom);
        let model = Mat4::IDENTITY;
        let clip_coords =
            projection * model * Vec4::new(left, bottom, 1.0, 1.0);
        let coords = Vec3::from_slice(&clip_coords.to_array()) / clip_coords.w;

        let xy = Vec2::new(
            (coords.x + 1.0) * 0.5 * screen_size[0],
            (1.0 - coords.y) * 0.5 * screen_size[1],
        );

        let (bw, bh, objh) = if scale != 1.0 {
            (right * scale, top * scale, height * scale)
        } else {
            (right, top, height)
        };
        // We must minus the height to flip the Y location to window coords.
        // You might not need to do this based on how you handle your Y coords.
        Vec4::new(xy.x, xy.y - objh, bw, bh)
    }
}
