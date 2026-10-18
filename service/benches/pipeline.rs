//! Criterion benchmarks for the request path that is not the model: decoding and window extraction.
//! (Model throughput is measured end to end with k6; see bench/.)

use std::hint::black_box;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use criterion::{Criterion, criterion_group, criterion_main};
use image::{Rgb, RgbImage};
use pathology_service::imaging::{decode_base64_rgb, grid, push_patch, windows};

fn png_b64(img: &RgbImage) -> String {
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    STANDARD.encode(buf)
}

fn textured(side: u32) -> RgbImage {
    RgbImage::from_fn(side, side, |x, y| {
        Rgb([
            ((x * 7 + y * 3) % 256) as u8,
            ((x * y) % 256) as u8,
            ((x ^ y) % 256) as u8,
        ])
    })
}

fn bench(c: &mut Criterion) {
    let patch = png_b64(&textured(96));
    c.bench_function("decode_and_layout_one_96px_patch", |b| {
        b.iter(|| {
            let img = decode_base64_rgb(black_box(&patch), 96).unwrap();
            let mut out = Vec::with_capacity(96 * 96 * 3);
            push_patch(&img, &mut out).unwrap();
            black_box(out)
        });
    });
    let tile = textured(1536);
    let g = grid(1536, 1536, 48).unwrap();
    c.bench_function("windows_1536px_tile_stride_48", |b| {
        b.iter(|| black_box(windows(black_box(&tile), g)));
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
