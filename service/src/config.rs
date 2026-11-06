//! Configuration from environment variables (see `.env.example`).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::auth::ClientLimit;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    /// Serves `/metrics`; keep it off the public network (Cloud Run routes only `port`).
    pub metrics_port: u16,
    pub model_path: PathBuf,
    pub meta_path: PathBuf,
    /// `client:key` pairs; keys at least 16 characters.
    pub api_keys: Vec<(String, String)>,
    pub rate_limit_per_minute: u32,
    /// `client:patches_per_minute:max_windows` triples that replace the defaults for those clients.
    pub client_limits: Vec<ClientLimit>,
    pub max_batch: usize,
    pub max_wait: Duration,
    pub workers: usize,
    pub intra_threads: usize,
    /// Patches whose model inputs may be queued or running at once (about 108 KiB each).
    pub max_inflight_patches: usize,
    /// How long a request may wait for room in that budget before it is refused with 503.
    pub admit_timeout: Duration,
    pub max_patches_per_request: usize,
    pub max_windows_per_request: usize,
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

/// `CLIENT_LIMITS`: comma-separated `client:patches_per_minute:max_windows`, each naming a client in `keys`.
pub fn parse_client_limits(raw: &str, keys: &[(String, String)]) -> Result<Vec<ClientLimit>> {
    let mut limits = Vec::new();
    for item in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let parts: Vec<&str> = item.split(':').collect();
        let [client, per_minute, max_windows] = parts[..] else {
            bail!("CLIENT_LIMITS entries must be client:patches_per_minute:max_windows")
        };
        // A misspelt client would leave the real one on the default limits, so it stops the service instead.
        if !keys.iter().any(|(c, _)| c == client) {
            bail!("CLIENT_LIMITS names {client}, which has no key in API_KEYS");
        }
        limits.push(ClientLimit {
            client: client.to_string(),
            per_minute: per_minute
                .parse()
                .with_context(|| format!("CLIENT_LIMITS for {client}: bad patches per minute"))?,
            max_windows: max_windows
                .parse()
                .with_context(|| format!("CLIENT_LIMITS for {client}: bad window cap"))?,
        });
    }
    Ok(limits)
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let cpus = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        let workers: usize = num("WORKERS", "2")?;
        let api_keys = parse_keys(&var("API_KEYS", ""))?;
        Ok(Self {
            client_limits: parse_client_limits(&var("CLIENT_LIMITS", ""), &api_keys)?,
            port: num("PORT", "8080")?,
            metrics_port: num("METRICS_PORT", "9464")?,
            model_path: var("MODEL_PATH", "models/pcam/1/model.onnx").into(),
            meta_path: var("MODEL_META_PATH", "models/pcam/1/model.json").into(),
            api_keys,
            rate_limit_per_minute: num("RATE_LIMIT_PER_MINUTE", "6000")?,
            max_batch: num("MAX_BATCH", "32")?,
            max_wait: Duration::from_micros(num("MAX_WAIT_US", "2000")?),
            workers,
            intra_threads: num("INTRA_THREADS", &(cpus / workers.max(1)).max(1).to_string())?,
            max_inflight_patches: num("MAX_INFLIGHT_PATCHES", "512")?,
            admit_timeout: Duration::from_millis(num("ADMIT_TIMEOUT_MS", "1000")?),
            max_patches_per_request: num("MAX_PATCHES_PER_REQUEST", "64")?,
            // 2048 px at the minimum stride of 32 is 62 x 62 = 3,844 windows.
            max_windows_per_request: num("MAX_WINDOWS_PER_REQUEST", "4096")?,
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

    #[test]
    fn client_limits_parse_and_must_name_a_known_client() {
        let keys = parse_keys("viewer:aaaaaaaaaaaaaaaa1,lab:bbbbbbbbbbbbbbbb2").unwrap();
        assert_eq!(
            parse_client_limits("viewer:1200:256", &keys).unwrap(),
            vec![ClientLimit {
                client: "viewer".into(),
                per_minute: 1200,
                max_windows: 256
            }]
        );
        assert!(parse_client_limits("", &keys).unwrap().is_empty());
        assert!(parse_client_limits("veiwer:1200:256", &keys).is_err());
        assert!(parse_client_limits("viewer:1200", &keys).is_err());
        assert!(parse_client_limits("viewer:lots:256", &keys).is_err());
    }
}
