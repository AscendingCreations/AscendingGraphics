<h1 align="center">
Ascending Graphics
</h1>

[![https://crates.io/crates/ascending_graphics](https://img.shields.io/crates/v/ascending_graphics?style=plastic)](https://crates.io/crates/ascending_graphics)
[![Docs](https://docs.rs/ascending_graphics/badge.svg)](https://docs.rs/ascending_graphics)
[![PRs Welcomed](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Discord Server](https://img.shields.io/discord/81844480201728000?label=&labelColor=6A7EC2&logo=discord&logoColor=ffffff&color=7389D8)](https://discord.gg/gVXNDwpS3Z)

## 📑 Overview

<p align="center">
    A 2D extendable rendering library using WGPU and Winit.
</p>

- [`WGPU`](https://crates.io/crates/wgpu) Backend.
- [`Winit`](https://crates.io/crates/winit) for windows and inputs.
- Buffered Sorted Rendering. 
- Render Images.
- Render Shader Animated Images.
- Render Basic Lighting.
- Render Shapes via lyon.
- Render Optimized Rectangle with Corner Rounding and image support.
- Render Text using [`cosmic-text`](https://crates.io/crates/cosmic-text).
- Optimized Map Renderer. (see examples).
- Atlas System with Texture Arrays and texture defragmentation support.
- Tilesheet loader to help with 2D tiles to Textures.
- Supports both Instance Buffers and Vertex Buffers.
- Extendable.
- Rayon support for sorting, clearing and anything possible.
- Supports up to 7 custom Camera views with 1 Auto camera handled Main View.

## 🚨 Help

If you need help with this library or have suggestions or feature requests please go to our [Discord Group](https://discord.gg/gVXNDwpS3Z)

## 🔎 Examples

[`Ascending Client`](https://github.com/AscendingCreations/AscendingClient)
![Client showcase](./images/client.png)

[`Ascending Map Editor`](https://github.com/AscendingCreations/AscendingMapEditor)
![MapEditor showcase](./images/map_editor.png)

[Render Demo](https://github.com/AscendingCreations/render_demo)
![Demo showcase](./images/demo.png)