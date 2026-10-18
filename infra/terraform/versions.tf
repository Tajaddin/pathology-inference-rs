terraform {
  required_version = ">= 1.9"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.7"
    }
  }
  # State in a GCS bucket created once by hand (see docs/DEPLOY.md):
  # terraform init -backend-config="bucket=<project>-tfstate"
  backend "gcs" {
    prefix = "pathology-inference-rs"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# The Budgets API refuses user credentials without a quota project, so the budget bills its API quota to this
# project. Only the budget uses it: the override on every call would also need Resource Manager enabled first.
provider "google" {
  alias                 = "billing"
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}
