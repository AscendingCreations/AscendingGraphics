# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
## Unreleased

## 0.38.1 (25. March, 2026)

### Fixed
- Issue causing Layers to not be correctly inserted when empty spaces inbetween them within the Instance Buffer Store.

## 0.38.0 (19. March, 2026)
### Added
- pre_present_notify() to GpuWindow.

### Changes
- (Breaking) update to WGPU 29.0.0
- (Breaking) wgpu::Instance now Required for Window Update so we can Rebuild the Surface when it is Lost
- pre_present_notify() Moved into renderer present as it needs to be called just before for best optimization of the window.
- Optimized Storage Vectors to use Vec<IBOStore> instead of Vec<Vec<IBOStore>>
- (Breaking) Removed Scissor Support. Will plan to add this support later in a different way.
- (Breaking) There is no longer a generic Store Type VBO and IBO have their own. IBOStore/VBOStore. this helps increase speed and reduce size.
- (Breaking) GPUWindow resize no longer returns an error.

## 0.37.1 (5. March, 2026)

### Changes
- Migrate to its own Repo.

## 0.37.0 (26. Feb, 2026)

### Changes
- (Breaking) Updated Cosmic-text 0.18.2 and Glam 0.32.0

## 0.36.0 (31. Jan, 2026)

### Changes
- (Breaking) Updated Cosmic-text and Glam
- Refactored Code using parallel helper (thank you Stephan)

### Fixed
- Fixed issue where Metal Adapters return no Driver String which cuased Macs to not work.

## 0.35.1 (20. Jan, 2026)

### Fixes
-  Fix text width and Height not getting sized for correct output.

## 0.35.0 (2. Jan, 2026)

### Changes
- (Breaking) cosmic-text updated to 0.16.0.

## 0.34.0 (31. December, 2025)

### Added
- set_area_light_visibility and set_directional_light_visibility.

### Changes
- (Breaking) Add visibility to Area and Directional light.
- (Breaking) return Option<T> When removing a Area or Directional light.
- (Breaking) FrameTime moved to ascending_time crate.

## 0.33.0 (19. December, 2025)

### Changes
- (Breaking) Text Bounds is now Optional. If set to None Text will no longer Clip.
- (Breaking) Text:set_bounds now takes an Optional Argument,

## 0.32.0 (18. December, 2025)

### Changes
- (Breaking) get_adapters is now async.
- (Breaking) WGPU & Naga are updated to v28.0.0

## 0.31.2 (15. December, 2025)

### Fixed
- Image and Animation Image Shaders not doing rotation, flipping, scaling and Translation correctly when using Camera Zoom.

## 0.31.1 (14. December, 2025)

### Fixed
- Allow bounds to be set outside of the Screen limit so camera scrolling can still Render these text.

## 0.31.0 (12. December, 2025)

### Added
- get_scale, get_scale_mut, get_view, get_view_mut.
- Support for up to 7 additional Manual Views

### Changed
- (Breaking) CameraType is now CameraView with changes in its enum names adding main_view and sub_views.
- (Breaking) manual view and scale repalced with Array of views and scales. Functions renamed as set_view.
- (Breaking) Removed manual_view, mut_manual_view, manual_scale, and mut_manual_scale
- (Breaking) Aligned Camera View function names.
- (Breaking) changed all naming from camera_type to camera_view
- (Breaking) Shaders Simplfied under new Camera Views.
- (Breaking) System and Shader Global now all use an array of 8 camera views and scales. index 0 is MainView and auto Set by camera.

## 0.30.0 (25. November, 2025)

### Changed
- (Breaking) added individual anim timer per Tile for Map Anim layers.
- increased GPU memory to allow 9 maps worth of tiles to be Prebuffered in.

## 0.29.1 (10. November, 2025)

### Added
- Rayon Support to visible bounds Calculation in visible_details.


## 0.29.0 (3. November, 2025)

### Changed
- (Breaking)Updated Cosmic_text to 0.15.0
- Included Alignment Option for set_text, measure_string and measure_glyphs.

## 0.28.0 (7. October, 2025)

### Changed
- Wgpu & Naga to version 27.0.0
- Removed Animation and API from Image
- Renamed all Render *_update to be just update for better alignment.

### Added
- Animation Pipeline with AnimImage type use this for Animated Images types instead. 
- pipelined_enabled to create_device, create_renderer and create_pipelines.

## 0.27.0 (10. July, 2025)

### Changed
- Wgpu & Naga to version 26.0.0
- Updated LRU to version 0.16

## 0.26.0 (20. June, 2025)

### Added
- Light::set_pos() and set_size().
- MapZLayer to set layer Z, Default uses the old static set layers.
- unload_map_index and aquire_map_index for allowing map store and quicker reloads. will need to adjust later.
- map.set_visibility to enforce instance buffer updates if previously set to Hidden so they render correctly.

### Changed
- Light: replaced z with Pos and added Size.
- Rect::new, Expand to add Color.
- Map data now remove from tile and set to uniform. Should boost map position update speeds.
- Map new and new_with now requires &mut Map_renderer to get uniform indexs.
- Map internal tile vertex generation now uses rayon when enabled to boost vertex generation speed.
- All unloads now consume Self to drop Self after unload is complete.
- (Breaking) Graphics update no longer triggers redraw requests except when a state becomes invalid.
- (Breaking) Graphics Update will call window.pre_present_notify() upon successful frame buffer grab.
- Made logging tied behind a feature. Most Error logs will be panics instead if logging disabled.

### Removed 
- Light::set_z().
- use_clipping for types that dont support it.

## 0.25.0 (2. June, 2025)

### Added
- ThreadLocal Storage for Maps and Fonts. This should help with Cache locality.
- Added Derives to all supportable types
- Added Rayon support for TileSheets
- Add feature to enable static-dxc for WGPU
- Added ability to set Size of Map in Tiles.
- Rect::new_with to also include image and uv.

### Removed
- Removed the Async trait library as we will enfore Rust edition 1.85.

### Changed
- Map::Update now returns Option<(OrderedIndex, OrderedIndex)> instead of a Vec to avoid reallocation each loop.
- Change AtlasSet Peek functions to take an &self instead of &mut as well as contains.
- Map Create_quad redone to help reduce any cache issues.
- Changed set_tile and get_tile to use UVec3 instead of tuple.
- Map::new added position to Arguments.
- Image::new added pos, size and uv to Arguments.
- Rect::new added Position and size.
- Mesh2D::new and With_capacity get pos argument.
- All Positions renamed to pos to aligned it better across types.
- All hw renamed to size to align better across types.

## 0.24.0 (11. April, 2025)

- measure_glyphs removed cache from API(it was a bad push...)
- measure_glyphs outputs correct Vec of Sizes.

## 0.23.1 (11. April, 2025)

### Added
- measure_glyphs function to give an Array of Glyph Sizes for a String.

## 0.23.0 (10. April, 2025)

### Changed
- get_adapters now also returns the Backend ID.
- removed trace_path: Option<&Path>, from create_device and create_renderer. No longer needed.
- updated to WGPU to v25.0.0

### Added
- rayon feature to thread what can be threaded. This will boost speed.

## 0.22.0 (1. April, 2025)

### Changed
- Updated cosmic-text to 0.14.0
- Made Fonts Render using wgpu::BlendState::ALPHA_BLENDING instead of wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
- measure_string now takes an &Attrs.
- set_rich_text now takes an &Attrs for Default Attrs.
- set_text now takes an &Attrs.


## 0.21.0 (10. March, 2025)

### Changed
- Updated cosmic-text to 0.13.1
- text set_rich_text added alignment argument.


## 0.20.0 (7. March, 2025)

### Changed
- Removed Position Overide and Render Layer from being stored in Object Structs, Shrinking struct sizes.
- Moved all DrawOrder Updates inside of functions. set_pos, set_color, etc..
- DrawOrder update functions no longer set Change to true. preventing unnecessary updates to GPU.
- Made update only call resize_with if the len() actually changed reducing overhead.

### Added
- Created set_order_alpha function for everything but Light and Text.
- Updated render_layer to order_layer since it is a rendering layer but its how it is ordered.

## 0.19.4 (19. February, 2025)

### Fixed
- Removed BinaryHeap due to Ordering Being not as Clear with how it Orders data upon insert. (this yanks 0.19.3)

## 0.19.3 (18. February, 2025)

### Fixed
- BinaryHeap order needed to be Reversed for it to Sort correctly. (this yanks 0.19.2)

## 0.19.2 (10. February, 2025)

### Added
- Added create_buffer_with functions for IBO and VBO to precreate with a Set size for faster first runs.

### Changed
- VBO and IBO both use BinaryHeap instead of Sorted Vec.

## 0.19.1 (23. Janurary, 2025)
### Fixed
- Other Adapter was Ignored which is used by OpenGL contexts. It now will work correctly.

## 0.19.0 (16. Janurary, 2025)
### Changed
- (Breaking) Updated WGPU v24.0.0

## 0.18.0 (13. Janurary, 2025)
### Changed
- (Breaking) Updated Ascending_input to v0.4.0

## 0.17.2 (28. December, 2024)
### Fixed
- Map tile set fill counter being incorrect when inserting or clearing.
- Shader issue where Size was nto set correctly.

## 0.17.0 (6. December, 2024)
### Fixed
- (Breaking) Image and UI shader removes Anti Aliasing from shader. since this does cause rendering output we are making this a breaking change.
  we will now limit Zoom ranges for Camera within 0.5 number changes. if you try 1.3 for example it will render incorrectly hence the breaking change.

## 0.16.1 (5. December, 2024)
### Fixed
- Image Shader now uses the scale instead of global scale within flip_rotation_mat4 fixing a improper scaling issue where every image would scale to the control zoom even if not set to use it.

## 0.16.0 (23. November, 2024)
### Fixed
- VBO Buffer and Mesh2D now properly load and render the Meshes when they are appended to a Mesh2D.

### Changed
- (Breaking) Mesh2D Position now Offsets the Meshes locations and The Z is Set exactly as the Mesh2D Z.
- (Breaking) Mesh2DBuilder Now has a Offset that is applied when being built into Mesh2D. 
- (Breaking) Mesh2D now takes &Mesh2DBuilder References instead of Passing the Mesh2DBuilder.
- (Breaking) Mesh2D from_builder now clears the Mesh2D before Adding in the new mesh from Mesh2DBuilder.

### Added
- Mesh2DBuilder now has a clear function so it can be reused to Build new meshes.
- Mesh2D now has a append_from_builder function to append new Meshes to Mesh2D.
- Mesh2D now has a clear function to clear the previous meshes.

## 0.15.0 (30. October, 2024)
### Changed
- (Breaking) updated to Naga and Wgpu v23.0.0

## 0.14.0 (25. September, 2024)
### Changed
- (Breaking) update_bounds for Rect returns &mut Self now.
- Updated Crates.

### Added
- Order Override to Image, Mesh2D, Text, Rect

## 0.13.0 (30. August, 2024)
### Changed
- Removed Text Offsets use Positions directly for this. Reduces memory a bit.
- (Breaking) Made Bounds not Optional. Text always looks at its bounds now. 
- Bounds now can determine if Text should even be iterated in the first place. (boosts Vertices and Glyph insertions speeds for invisible lines.)
- Added Glyph Buffer to reduce allocations. This will help improve performance but will increase Ram usage. 
- Removed Y Glyph discard check since new iterator should discard them.
- Optimized Shader Code.

## 0.12.1 (7. August, 2024)
### Fixed
- Map upper layer is now set to Render Layer 1. They can not be on the same layer as ground tiles due to Z rendering transparency issues.

## 0.12.0 (18. July, 2024)
### Changed
- (Breaking) Updated to Wgpu and Naga v22.0.0

## 0.11.2 (16. July, 2024)
### Fixed
- Readme discord badge added.

## 0.11.1 (15. July, 2024)
### Fixed
- Readme image issue on crates.io

## 0.11.0 (15. July, 2024)
### Added
- Defragmentation to Atlas_set. Reduces Texture Fragmented DeadSpace allowing more textures to fit later.
- with_deallocations_limit to Atlas_set to allow setting a custom defrag ratio.

### Changed
- (Breaking) Added size that is clamped to 256 to limits.max_texture_dimension_2d.
- (Breaking) Removed TextureView from AtlasSet and placed into TextureGroup.
- Added Backend to GpuRenderer 

### Fixed
- Reduced Textures loaded to 1 if not Opengl being used as an adapter backend.
- Rebinding TextureGroup upon Grow allows Access to new Texture Layers.

### Added
- GpuRenderer::get_layout() to acquire already made layouts without the need for a &mut GpuRenderer. 

## 0.10.1 (25. June, 2024)
### Changed
- Added Feature PassThru to Give direct access to all usable internal crates.

### Fixed
- Atlas Grow had the wrong Format and mip_level_count incorrect. 

## 0.10.0 (19. June, 2024)
### Changed
- (Breaking) Removed Visible lines function in Text as Cosmic removed it as well.
- (Breaking) Updated to cosmic-text 0.12.0

### Added
- Text::visible_details() which will return the needed details to calculate the render texts size.

## 0.9.0 (6. June, 2024)
### Changed
- (Breaking) Added Rendering layer to Text, Mesh.
- (Breaking) DrawOrder Width, Height and DrawType Removed. 
- (Breaking) DrawType Removed.
- (Breaking) Rename GpuBuffer as VertexBuffer.
- (Breaking) Bounds functions now use Vec2 instead of Vec3 since we do not use Z.
- (Breaking) Removed tex_buf from Text to reduce Ram usage.

### Fixed
- Ensure all renderers use Alpha checks and Rendering Layer for Ordering

### Added
- More Documentation.

## 0.8.2 (30. May, 2024)
### Fixed
- sRGB to Linear color within shader to give same or closer results to Paint and other editing programs that use RGB

## 0.8.1 (17. May, 2024)
### Fixed
- fixed Y ordering offsets.

## 0.8.0 (24. April, 2024)
### Changed
- (Breaking) updated to support winit 0.30.0.
- (Breaking) renderer update now takes &WindowEvent.

### Added
- Z axis angle Rotation and Flip to image.

## 0.7.0 (24. April, 2024)
### Changed
- (Breaking) system not supports a secondary manual Mat4x4 and Scale.
- (Breaking) use_camera is now set as camera_type and uses a enumeration which tells the shader how to use the camera's.
- (Breaking) projected_world_to_screen and world_to_screen now both use CameraType instead of scale.
- (Breaking) shaders were rewritten to allow multiple views and scales.

### Fixed
- rect not rendering correctly due to scale was always being applied even when view was not.


## 0.5.0 (24. April, 2024)
### Changed
- (Breaking)  BufferStore::new now expects sizes for vertex and index's Vec

### Fixed
- Updated Rendering types to have persistent data to avoid recreation upon update, which avoids allocations.
