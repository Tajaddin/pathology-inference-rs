output "service_url" {
  value = google_cloud_run_v2_service.service.uri
}

output "image" {
  description = "Where CI pushes the service image"
  value       = local.image
}

output "models_bucket" {
  description = "Upload models/pcam/<version>/ here"
  value       = google_storage_bucket.models.name
}

output "workload_identity_provider" {
  description = "For google-github-actions/auth in the deploy workflow"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_service_account" {
  value = google_service_account.deployer.email
}
