# Cloud Run: the Rust service, the model mounted read-only from Cloud Storage. Scales to zero, so idle cost is nil.
# CPU is always allocated while a request runs; the dynamic batcher needs it between requests only briefly.

resource "google_cloud_run_v2_service" "service" {
  name                = "${local.name}-service"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.service.email
    max_instance_request_concurrency = 64
    timeout                          = "30s"

    scaling {
      min_instance_count = 0
      max_instance_count = var.max_instances
    }

    volumes {
      name = "models"
      gcs {
        bucket    = google_storage_bucket.models.name
        read_only = true
      }
    }

    containers {
      name  = "service"
      image = local.image
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu    = var.cpu
          memory = "2Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }
      env {
        name  = "MODEL_PATH"
        value = "/models/pcam/${var.model_version}/model.onnx"
      }
      env {
        name  = "MODEL_META_PATH"
        value = "/models/pcam/${var.model_version}/model.json"
      }
      # One inference worker per vCPU with one intra-op thread each, and a 0.5 ms batching window: the setting
      # that sustained 800 single-patch requests/s on 4 CPUs in reports/k6 (2 x 2 threads and 2 ms did not).
      env {
        name  = "WORKERS"
        value = var.cpu
      }
      env {
        name  = "INTRA_THREADS"
        value = "1"
      }
      env {
        name  = "MAX_WAIT_US"
        value = "500"
      }
      env {
        name  = "RUST_LOG"
        value = "info,ort=warn"
      }
      env {
        name = "API_KEYS"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.api_keys.secret_id
            version = "latest"
          }
        }
      }
      volume_mounts {
        name       = "models"
        mount_path = "/models"
      }
      # /ready turns true once the model is loaded and a warm-up batch has run.
      startup_probe {
        http_get {
          path = "/ready"
        }
        period_seconds    = 2
        failure_threshold = 30
      }
      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }
  }

  depends_on = [google_project_service.apis, google_secret_manager_secret_iam_member.service_reads_keys]
}

# Callers authenticate with API keys in the service, so Cloud Run accepts unauthenticated requests. A real
# deployment would add API Gateway or IAP in front.
resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.service.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
