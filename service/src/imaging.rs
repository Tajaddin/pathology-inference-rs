//! Image decoding and the model's input layout.
//!
//! The model takes NHWC float32 RGB in 0..255; normalisation happens inside the ONNX graph (it was part of the
//! Keras model), so this layer only decodes, checks sizes and lays out pixels.

use std::ops::Range;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::RgbImage;

use crate::error::ApiError;

/// Side of the square patches the model was trained on (`PatchCamelyon`).
pub const PATCH: u32 = 96;
/// Floats per patch in the model input.
pub const PATCH_LEN: usize = (PATCH * PATCH * 3) as usize;

pub fn decode_base64(data: &str) -> Result<Vec<u8>, ApiError> {
    STANDARD
        .decode(data.trim())
        .map_err(|_| ApiError::BadRequest("image is not valid base64".into()))
}

pub fn decode_base64_rgb(data: &str, max_side: u32) -> Result<RgbImage, ApiError> {
    decode_rgb(&decode_base64(data)?, max_side)
}

/// What an image header says, read without decoding any pixels.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    format: image::ImageFormat,
    pub width: u32,
    pub height: u32,
}

/// Format and dimensions from the header alone; anything but PNG or JPEG up to `max_side` is refused.
pub fn read_header(bytes: &[u8], max_side: u32) -> Result<Header, ApiError> {
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| ApiError::BadRequest("unreadable image".into()))?;
    let format = reader
        .format()
        .ok_or_else(|| ApiError::BadRequest("image must be PNG or JPEG".into()))?;
    if !matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg) {
        return Err(ApiError::BadRequest("image must be PNG or JPEG".into()));
    }
    let (width, height) = reader
        .into_dimensions()
        .map_err(|_| ApiError::BadRequest("unreadable image".into()))?;
    if width > max_side || height > max_side {
        return Err(ApiError::BadRequest(format!(
            "image is {width}x{height}; the limit is {max_side}x{max_side}"
        )));
    }
    Ok(Header {
        format,
        width,
        height,
    })
}

impl Header {
    /// Decode the pixels of the image this header was read from.
    pub fn decode(&self, bytes: &[u8]) -> Result<RgbImage, ApiError> {
        let img = image::load_from_memory_with_format(bytes, self.format)
            .map_err(|_| ApiError::BadRequest("unreadable image".into()))?;
        Ok(img.into_rgb8())
    }
}

pub fn decode_rgb(bytes: &[u8], max_side: u32) -> Result<RgbImage, ApiError> {
    // Dimensions are checked before the pixels are decoded, so an oversized image costs nothing.
    read_header(bytes, max_side)?.decode(bytes)
}

/// Decode one model input patch, which must be exactly `PATCH` x `PATCH`.
pub fn decode_patch(data: &str) -> Result<RgbImage, ApiError> {
    let img = decode_base64_rgb(data, PATCH)?;
    if img.dimensions() != (PATCH, PATCH) {
        return Err(ApiError::BadRequest(format!(
            "patches must be {PATCH}x{PATCH}, got {}x{}",
            img.width(),
            img.height()
        )));
    }
    Ok(img)
}

/// Append one patch to `out` in HWC order.
pub fn push_patch(img: &RgbImage, out: &mut Vec<f32>) {
    debug_assert_eq!(img.dimensions(), (PATCH, PATCH));
    out.extend(img.as_raw().iter().map(|&v| f32::from(v)));
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

impl Grid {
    /// Number of windows.
    #[must_use]
    pub fn cells(&self) -> usize {
        self.rows as usize * self.cols as usize
    }
}

/// Windows `range` of the grid (row-major indices) as one flat HWC buffer. A caller lays out one batch at a time,
/// so a large tile never needs all of its windows in memory at once.
#[must_use]
pub fn windows(tile: &RgbImage, g: Grid, range: Range<usize>) -> Vec<f32> {
    let mut out = Vec::with_capacity(range.len() * PATCH_LEN);
    let raw = tile.as_raw();
    let row_len = tile.width() as usize * 3;
    let (cols, stride) = (g.cols as usize, g.stride as usize);
    for i in range {
        let (y0, x0) = (i / cols * stride, i % cols * stride);
        for y in y0..y0 + PATCH as usize {
            let start = y * row_len + x0 * 3;
            out.extend(
                raw[start..start + PATCH as usize * 3]
                    .iter()
                    .map(|&v| f32::from(v)),
            );
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
        let w = windows(&image, g, 0..g.cells());
        assert_eq!(w.len(), g.cells() * PATCH_LEN);
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
        // Any range is the same slice of the full layout, so batches can be laid out one at a time.
        assert_eq!(windows(&image, g, 3..6), w[3 * PATCH_LEN..6 * PATCH_LEN]);
    }

    #[test]
    fn decode_patch_rejects_wrong_sizes() {
        let encode = |img: RgbImage| {
            let mut png = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                .unwrap();
            STANDARD.encode(png)
        };
        let patch = decode_patch(&encode(tile(96, 96))).unwrap();
        let mut out = Vec::new();
        push_patch(&patch, &mut out);
        assert_eq!(out.len(), PATCH_LEN);
        assert!(decode_patch(&encode(tile(64, 96))).is_err());
        assert!(decode_patch(&encode(tile(97, 96))).is_err());
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
