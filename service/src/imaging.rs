//! Image decoding and the model's input layout.
//!
//! The model takes NHWC float32 RGB in 0..255; normalisation happens inside the ONNX graph (it was part of the
//! Keras model), so this layer only decodes, checks sizes and lays out pixels.

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::RgbImage;

use crate::error::ApiError;

/// Side of the square patches the model was trained on (`PatchCamelyon`).
pub const PATCH: u32 = 96;
/// Floats per patch in the model input.
pub const PATCH_LEN: usize = (PATCH * PATCH * 3) as usize;

pub fn decode_base64_rgb(data: &str, max_side: u32) -> Result<RgbImage, ApiError> {
    let bytes = STANDARD
        .decode(data.trim())
        .map_err(|_| ApiError::BadRequest("image is not valid base64".into()))?;
    decode_rgb(&bytes, max_side)
}

pub fn decode_rgb(bytes: &[u8], max_side: u32) -> Result<RgbImage, ApiError> {
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| ApiError::BadRequest("unreadable image".into()))?;
    let format = reader
        .format()
        .ok_or_else(|| ApiError::BadRequest("image must be PNG or JPEG".into()))?;
    if !matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg) {
        return Err(ApiError::BadRequest("image must be PNG or JPEG".into()));
    }
    let (w, h) = reader
        .into_dimensions()
        .map_err(|_| ApiError::BadRequest("unreadable image".into()))?;
    if w > max_side || h > max_side {
        return Err(ApiError::BadRequest(format!(
            "image is {w}x{h}; the limit is {max_side}x{max_side}"
        )));
    }
    // Dimensions are checked before the pixels are decoded, so an oversized image costs nothing.
    let img = image::load_from_memory_with_format(bytes, format)
        .map_err(|_| ApiError::BadRequest("unreadable image".into()))?;
    Ok(img.to_rgb8())
}

/// Append one 96x96 patch to `out` in HWC order.
pub fn push_patch(img: &RgbImage, out: &mut Vec<f32>) -> Result<(), ApiError> {
    if img.width() != PATCH || img.height() != PATCH {
        return Err(ApiError::BadRequest(format!(
            "patches must be {PATCH}x{PATCH}, got {}x{}",
            img.width(),
            img.height()
        )));
    }
    out.extend(img.as_raw().iter().map(|&v| f32::from(v)));
    Ok(())
}

/// Grid geometry of a sliding window over a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub rows: u32,
    pub cols: u32,
    pub stride: u32,
}

pub fn grid(width: u32, height: u32, stride: u32) -> Result<Grid, ApiError> {
    if width < PATCH || height < PATCH {
        return Err(ApiError::BadRequest(format!(
            "tiles must be at least {PATCH}x{PATCH}"
        )));
    }
    Ok(Grid {
        rows: (height - PATCH) / stride + 1,
        cols: (width - PATCH) / stride + 1,
        stride,
    })
}

/// All windows of the grid, row-major, as one flat HWC buffer.
#[must_use]
pub fn windows(tile: &RgbImage, g: Grid) -> Vec<f32> {
    let mut out = Vec::with_capacity((g.rows * g.cols) as usize * PATCH_LEN);
    let raw = tile.as_raw();
    let row_len = tile.width() as usize * 3;
    for r in 0..g.rows {
        for c in 0..g.cols {
            let (y0, x0) = ((r * g.stride) as usize, (c * g.stride) as usize);
            for y in y0..y0 + PATCH as usize {
                let start = y * row_len + x0 * 3;
                out.extend(
                    raw[start..start + PATCH as usize * 3]
                        .iter()
                        .map(|&v| f32::from(v)),
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        })
    }

    #[test]
    fn grid_counts_windows() {
        assert_eq!(
            grid(96, 96, 48).unwrap(),
            Grid {
                rows: 1,
                cols: 1,
                stride: 48
            }
        );
        assert_eq!(
            grid(192, 96, 48).unwrap(),
            Grid {
                rows: 1,
                cols: 3,
                stride: 48
            }
        );
        assert_eq!(
            grid(960, 480, 96).unwrap(),
            Grid {
                rows: 5,
                cols: 10,
                stride: 96
            }
        );
        assert!(grid(95, 96, 48).is_err());
    }

    #[test]
    fn windows_copy_the_right_pixels() {
        let image = tile(192, 144);
        let g = grid(192, 144, 48).unwrap();
        let w = windows(&image, g);
        assert_eq!(w.len(), (g.rows * g.cols) as usize * PATCH_LEN);
        // Window (row 1, col 2) starts at y=48, x=96; its first pixel is tile(96, 48).
        let idx = (g.cols as usize + 2) * PATCH_LEN;
        let p = image.get_pixel(96, 48);
        assert_eq!(
            &w[idx..idx + 3],
            &[f32::from(p[0]), f32::from(p[1]), f32::from(p[2])]
        );
        // Last pixel of that window is tile(191, 143).
        let q = image.get_pixel(191, 143);
        let end = idx + PATCH_LEN;
        assert_eq!(
            &w[end - 3..end],
            &[f32::from(q[0]), f32::from(q[1]), f32::from(q[2])]
        );
    }

    #[test]
    fn push_patch_rejects_wrong_sizes() {
        let mut out = Vec::new();
        assert!(push_patch(&tile(96, 96), &mut out).is_ok());
        assert_eq!(out.len(), PATCH_LEN);
        assert!(push_patch(&tile(64, 96), &mut out).is_err());
    }

    #[test]
    fn decode_checks_format_and_size() {
        let mut png = Vec::new();
        tile(120, 100)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        assert_eq!(decode_rgb(&png, 2048).unwrap().dimensions(), (120, 100));
        assert!(
            matches!(decode_rgb(&png, 110), Err(ApiError::BadRequest(m)) if m.contains("limit"))
        );
        assert!(decode_rgb(b"not an image", 2048).is_err());
        assert!(decode_base64_rgb("%%%", 2048).is_err());
    }
}
