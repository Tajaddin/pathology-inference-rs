//! Configuration from environment variables (see `.env.example`).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub model_path: PathBuf,
    pub meta_path: PathBuf,
    /// `client:key` pairs; keys at least 16 characters.
    pub api_keys: Vec<(String, String)>,
    pub rate_limit_per_minute: u32,
    pub max_batch: usize,
    pub max_wait: Duration,
    pub workers: usize,
    pub intra_threads: usize,
    pub max_patches_per_request: usize,
    pub max_tile_side: u32,
    pub min_stride: u32,
    pub cors_origins: Vec<String>,
}

fn var(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn num<T: std::str::FromStr>(name: &str, default: &str) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    var(name, default)
        .parse::<T>()
        .with_context(|| format!("{name} is not a valid number"))
}

pub fn parse_keys(raw: &str) -> Result<Vec<(String, String)>> {
    let mut keys = Vec::new();
    for item in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let Some((client, key)) = item.split_once(':') else {
            bail!("API_KEYS entries must be client:key")
        };
        if client.is_empty() || key.len() < 16 {
            bail!("API_KEYS entries must be client:key with keys of at least 16 characters");
        }
        keys.push((client.to_string(), key.to_string()));
    }
    Ok(keys)
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let cpus = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        let workers: usize = num("WORKERS", "2")?;
        Ok(Self {
            port: num("PORT", "8080")?,
            model_path: var("MODEL_PATH", "models/pcam/1/model.onnx").into(),
            meta_path: var("MODEL_META_PATH", "models/pcam/1/model.json").into(),
            api_keys: parse_keys(&var("API_KEYS", ""))?,
            rate_limit_per_minute: num("RATE_LIMIT_PER_MINUTE", "6000")?,
            max_batch: num("MAX_BATCH", "32")?,
            max_wait: Duration::from_micros(num("MAX_WAIT_US", "2000")?),
            workers,
            intra_threads: num("INTRA_THREADS", &(cpus / workers.max(1)).max(1).to_string())?,
            max_patches_per_request: num("MAX_PATCHES_PER_REQUEST", "64")?,
            max_tile_side: num("MAX_TILE_SIDE", "2048")?,
            min_stride: num("MIN_STRIDE", "32")?,
            cors_origins: var(
                "CORS_ORIGINS",
                "http://localhost:5173,http://localhost:8088",
            )
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_parse_and_reject_short_or_malformed() {
        let keys = parse_keys("viewer:aaaaaaaaaaaaaaaa1, lab:bbbbbbbbbbbbbbbb2").unwrap();
        assert_eq!(
            keys,
            vec![
                ("viewer".into(), "aaaaaaaaaaaaaaaa1".into()),
                ("lab".into(), "bbbbbbbbbbbbbbbb2".into())
            ]
        );
        assert!(parse_keys("viewer:short").is_err());
        assert!(parse_keys("nocolon").is_err());
        assert!(parse_keys("").unwrap().is_empty());
    }
}
