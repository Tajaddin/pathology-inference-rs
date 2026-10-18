//! API keys (constant-time comparison) and a per-client rate limit.

use std::num::NonZeroU32;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;
use governor::clock::{Clock, DefaultClock};
use governor::{DefaultKeyedRateLimiter, Quota, RateLimiter};
use subtle::ConstantTimeEq;

use crate::error::ApiError;

#[derive(Clone, Debug)]
pub struct Client(pub String);

pub struct Auth {
    keys: Vec<(String, String)>,
    limiter: DefaultKeyedRateLimiter<String>,
    clock: DefaultClock,
}

impl Auth {
    #[must_use]
    pub fn new(keys: Vec<(String, String)>, per_minute: u32) -> Self {
        let quota = Quota::per_minute(NonZeroU32::new(per_minute.max(1)).expect("non-zero"));
        Self {
            keys,
            limiter: RateLimiter::keyed(quota),
            clock: DefaultClock::default(),
        }
    }

    /// The client a presented key belongs to. Every key is compared, in constant time, so the response time
    /// does not reveal how much of a key matched or which keys exist.
    #[must_use]
    pub fn client_for(&self, presented: &str) -> Option<String> {
        let mut found = None;
        for (client, key) in &self.keys {
            if bool::from(presented.as_bytes().ct_eq(key.as_bytes())) {
                found = Some(client.clone());
            }
        }
        found
    }

    pub fn check_rate(&self, client: &str, cost: u32) -> Result<(), ApiError> {
        let cost = NonZeroU32::new(cost.max(1)).expect("non-zero");
        match self.limiter.check_key_n(&client.to_string(), cost) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(not_until)) => {
                metrics::counter!("pathology_rate_limited_total", "client" => client.to_string())
                    .increment(1);
                let wait = not_until.wait_time_from(self.clock.now());
                Err(ApiError::RateLimited {
                    retry_after_s: wait.as_secs() + 1,
                })
            }
            // The request costs more than the whole per-minute quota: it can never pass.
            Err(_) => Err(ApiError::BadRequest(
                "request is larger than the rate limit allows".into(),
            )),
        }
    }
}

fn presented_key(headers: &HeaderMap) -> Option<&str> {
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v);
    }
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

pub async fn require_key(
    State(auth): State<Arc<Auth>>,
    mut req: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let client = presented_key(req.headers())
        .and_then(|k| auth.client_for(k))
        .ok_or(ApiError::Unauthorized)?;
    req.extensions_mut().insert(Client(client));
    Ok(next.run(req).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_match_exactly() {
        let auth = Auth::new(
            vec![
                ("viewer".into(), "k".repeat(20)),
                ("lab".into(), "j".repeat(20)),
            ],
            60,
        );
        assert_eq!(auth.client_for(&"k".repeat(20)).as_deref(), Some("viewer"));
        assert_eq!(auth.client_for(&"k".repeat(19)), None);
        assert_eq!(auth.client_for(""), None);
    }

    #[test]
    fn rate_limit_counts_patches_per_client() {
        let auth = Auth::new(vec![], 10);
        assert!(auth.check_rate("a", 10).is_ok());
        assert!(matches!(
            auth.check_rate("a", 1),
            Err(ApiError::RateLimited { .. })
        ));
        assert!(auth.check_rate("b", 1).is_ok());
        assert!(matches!(
            auth.check_rate("c", 11),
            Err(ApiError::BadRequest(_))
        ));
    }
}
