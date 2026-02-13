# Terraform Outputs

# Networking Outputs
output "vpc_network_name" {
  description = "VPC network name"
  value       = module.networking.network_name
}

output "subnet_name" {
  description = "Subnet name"
  value       = module.networking.subnet_name
}

# Compute Outputs
output "api_server_ip" {
  description = "Public IP of API server"
  value       = module.compute.api_instance_ip
}

output "training_server_ip" {
  description = "Public IP of training server"
  value       = module.compute.training_instance_ip
}

# Storage Outputs
output "artifacts_bucket" {
  description = "Artifacts bucket name"
  value       = module.storage.artifacts_bucket_name
}

output "data_bucket" {
  description = "Data bucket name"
  value       = module.storage.data_bucket_name
}

output "mlflow_bucket" {
  description = "MLflow bucket name"
  value       = module.storage.mlflow_bucket_name
}

# Cloud SQL Outputs
output "mlflow_db_connection" {
  description = "Cloud SQL connection string"
  value       = var.enable_cloud_sql ? google_sql_database_instance.mlflow[0].connection_name : null
}

output "mlflow_db_ip" {
  description = "Cloud SQL IP address"
  value       = var.enable_cloud_sql ? google_sql_database_instance.mlflow[0].public_ip_address : null
}

# Redis Outputs
output "redis_host" {
  description = "Redis host"
  value       = var.enable_redis ? google_redis_instance.cache[0].host : null
}

output "redis_port" {
  description = "Redis port"
  value       = var.enable_redis ? google_redis_instance.cache[0].port : null
}

# Access Instructions
output "access_instructions" {
  description = "Instructions to access deployed services"
  value = <<-EOT
  
  === VinaSmol RAG MLOps Infrastructure ===
  
  API Server: http://${module.compute.api_instance_ip}:8000
  MLflow UI:  http://${module.compute.api_instance_ip}:8080
  Grafana:    http://${module.compute.api_instance_ip}:3000
  Prometheus: http://${module.compute.api_instance_ip}:9090
  
  SSH to API Server:
    gcloud compute ssh ${var.project_id}-${var.environment}-vm-api --zone=${var.zone}
  
  SSH to Training Server:
    gcloud compute ssh ${var.project_id}-${var.environment}-vm-training --zone=${var.zone}
  
  EOT
}
