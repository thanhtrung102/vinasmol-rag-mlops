output "api_instance_id" {
  value = google_compute_instance.api.id
}

output "api_instance_ip" {
  value = google_compute_instance.api.network_interface[0].access_config[0].nat_ip
}

output "training_instance_id" {
  value = var.enable_training_server ? google_compute_instance.training[0].id : null
}

output "training_instance_ip" {
  value = var.enable_training_server ? google_compute_instance.training[0].network_interface[0].access_config[0].nat_ip : null
}
