//! API errors: one JSON shape, one status code per kind, nothing echoed from the request.

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("missing or invalid API key")]
    Unauthorized,
    #[error("rate limit exceeded")]
    RateLimited { retry_after_s: u64 },
    #[error("{0}")]
    BadRequest(String),
    #[error("model unavailable: {0}")]
    Unavailable(String),
    #[error("internal error")]
    Internal(#[from] anyhow::Error),
}

impl ApiError {
    fn status(&self) -> StatusCode {
        match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::RateLimited { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::BadRequest(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Unauthorized => "unauthorized",
            Self::RateLimited { .. } => "rate_limited",
            Self::BadRequest(_) => "invalid_request",
            Self::Unavailable(_) => "unavailable",
            Self::Internal(_) => "internal",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        if let Self::Internal(err) = &self {
            tracing::error!(error = %err, "internal error");
        }
        metrics::counter!("pathology_errors_total", "code" => self.code()).increment(1);
        let body = Json(json!({"error": {"code": self.code(), "message": self.to_string()}}));
        let mut response = (status, body).into_response();
        match self {
            Self::RateLimited { retry_after_s } => {
                if let Ok(v) = HeaderValue::from_str(&retry_after_s.max(1).to_string()) {
                    response.headers_mut().insert(header::RETRY_AFTER, v);
                }
            }
            Self::Unauthorized => {
                response
                    .headers_mut()
                    .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
            }
            _ => {}
        }
        response
    }
}
