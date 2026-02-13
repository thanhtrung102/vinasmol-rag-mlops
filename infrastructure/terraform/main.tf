# Main Terraform Configuration for VinaSmol RAG MLOps

locals {
  common_labels = merge(
    {
      project     = "vinasmol-rag-mlops"
      environment = var.environment
      managed_by  = "terraform"
    },
    var.labels
  )
}

# Networking Module
module "networking" {
  source = "./modules/networking"

  project_id  = var.project_id
  region      = var.region
  environment = var.environment
  subnet_cidr = var.subnet_cidr
}

# Compute Module
module "compute" {
  source = "./modules/compute"

  project_id              = var.project_id
  environment             = var.environment
  zone                    = var.zone
  subnet_id               = module.networking.subnet_id
  api_machine_type        = var.api_machine_type
  api_disk_size_gb        = var.api_disk_size_gb
  training_machine_type   = var.training_machine_type
  training_gpu_type       = var.training_gpu_type
  training_gpu_count      = var.training_gpu_count
  enable_training_server  = var.enable_training_server

  depends_on = [module.networking]
}

# Storage Module
module "storage" {
  source = "./modules/storage"

  project_id    = var.project_id
  environment   = var.environment
  region        = var.region
  storage_class = var.storage_class
}

# Cloud SQL Instance (PostgreSQL for MLflow)
resource "google_sql_database_instance" "mlflow" {
  count = var.enable_cloud_sql ? 1 : 0

  name             = "${var.project_id}-${var.environment}-sql-mlflow"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = var.sql_tier

    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "all"
        value = "0.0.0.0/0"  # Restrict in production
      }
    }

    backup_configuration {
      enabled = true
    }
  }

  deletion_protection = false  # Set true for production
}

resource "google_sql_database" "mlflow_db" {
  count = var.enable_cloud_sql ? 1 : 0

  name     = "mlflow"
  instance = google_sql_database_instance.mlflow[0].name
}

# Memorystore Redis (for caching)
resource "google_redis_instance" "cache" {
  count = var.enable_redis ? 1 : 0

  name           = "${var.project_id}-${var.environment}-redis"
  tier           = "BASIC"
  memory_size_gb = var.redis_memory_size_gb
  region         = var.region
  redis_version  = "REDIS_7_0"

  authorized_network = module.networking.network_id

  depends_on = [module.networking]
}
