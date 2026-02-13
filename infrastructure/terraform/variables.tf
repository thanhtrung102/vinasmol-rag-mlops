# Terraform Variables for VinaSmol RAG MLOps Infrastructure

# PROJECT CONFIGURATION
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# NETWORKING
variable "network_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "vinasmol-vpc"
}

variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

# COMPUTE - API SERVER
variable "api_machine_type" {
  description = "Machine type for API server"
  type        = string
  default     = "n2-standard-4"
}

variable "api_disk_size_gb" {
  description = "Boot disk size for API server (GB)"
  type        = number
  default     = 50
}

# COMPUTE - TRAINING SERVER  
variable "training_machine_type" {
  description = "Machine type for training server"
  type        = string
  default     = "n1-standard-4"
}

variable "training_gpu_type" {
  description = "GPU type for training"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "training_gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 1
}

variable "enable_training_server" {
  description = "Whether to create training server"
  type        = bool
  default     = true
}

# STORAGE
variable "storage_class" {
  description = "Storage class for buckets"
  type        = string
  default     = "STANDARD"
}

# CLOUD SQL
variable "enable_cloud_sql" {
  description = "Whether to create Cloud SQL instance"
  type        = bool
  default     = true
}

variable "sql_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-f1-micro"
}

# REDIS
variable "enable_redis" {
  description = "Whether to create Redis instance"
  type        = bool
  default     = true
}

variable "redis_memory_size_gb" {
  description = "Redis memory size (GB)"
  type        = number
  default     = 1
}
