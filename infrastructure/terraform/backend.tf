# Backend configuration for Terraform state management
#
# State is stored in Google Cloud Storage (GCS) with:
# - Versioning enabled for state history
# - Encryption at rest
# - State locking to prevent concurrent modifications
#
# Prerequisites:
# 1. Create GCS bucket: gsutil mb gs://{project-id}-terraform-state
# 2. Enable versioning: gsutil versioning set on gs://{project-id}-terraform-state
# 3. Set appropriate IAM permissions

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Remote state backend
  # Uncomment and configure after creating the state bucket
  # 
  # backend "gcs" {
  #   bucket  = "vinasmol-rag-mlops-terraform-state"
  #   prefix  = "terraform/state"
  #   
  #   # Optional: Encrypt state with customer-managed encryption key
  #   # encryption_key = "projects/PROJECT_ID/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY"
  # }
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
