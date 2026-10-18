variable "project_id" {
  description = "GCP project that hosts everything"
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Artifact Registry and the model bucket"
  type        = string
  default     = "us-central1"
}

variable "github_repository" {
  description = "owner/name of the repository whose Actions may deploy (Workload Identity Federation)"
  type        = string
  default     = "Tajaddin/pathology-inference-rs"
}

variable "image_tag" {
  description = "Tag of the service image in Artifact Registry"
  type        = string
  default     = "latest"
}

variable "model_version" {
  description = "Folder under gs://<bucket>/pcam/ that holds model.onnx and model.json"
  type        = string
  default     = "1"
}

variable "cpu" {
  description = "vCPUs per instance; ONNX Runtime and the batcher scale with it"
  type        = string
  default     = "2"
}

variable "max_instances" {
  description = "Upper bound on Cloud Run instances; also a cost ceiling"
  type        = number
  default     = 3
}

variable "billing_account_id" {
  description = "Billing account for the budget alert; empty skips the budget"
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  description = "Budget alert threshold in USD"
  type        = number
  default     = 5
}

variable "alert_email" {
  description = "Where Cloud Monitoring sends alerts; empty skips the alert policies"
  type        = string
  default     = ""
}
