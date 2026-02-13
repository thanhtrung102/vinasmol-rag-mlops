variable "project_id" {
  type = string
}

variable "environment" {
  type = string
}

variable "zone" {
  type = string
}

variable "subnet_id" {
  type = string
}

variable "api_machine_type" {
  type = string
}

variable "api_disk_size_gb" {
  type = number
}

variable "training_machine_type" {
  type = string
}

variable "training_gpu_type" {
  type = string
}

variable "training_gpu_count" {
  type = number
}

variable "training_disk_size_gb" {
  type    = number
  default = 100
}

variable "enable_training_server" {
  type = bool
}
