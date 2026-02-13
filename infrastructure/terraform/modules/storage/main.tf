# Storage Module - Cloud Storage Buckets

# Artifacts Bucket (for model weights, checkpoints)
resource "google_storage_bucket" "artifacts" {
  name          = "${var.project_id}-${var.environment}-artifacts"
  location      = var.region
  storage_class = var.storage_class
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# Data Bucket (for training data)
resource "google_storage_bucket" "data" {
  name          = "${var.project_id}-${var.environment}-data"
  location      = var.region
  storage_class = var.storage_class
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}

# MLflow Bucket (for MLflow artifacts)
resource "google_storage_bucket" "mlflow" {
  name          = "${var.project_id}-${var.environment}-mlflow"
  location      = var.region
  storage_class = var.storage_class
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}
