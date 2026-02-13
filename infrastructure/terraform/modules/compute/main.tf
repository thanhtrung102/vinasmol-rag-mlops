# Compute Module - API Server and Training Server

# API Server Instance
resource "google_compute_instance" "api" {
  name         = "${var.project_id}-${var.environment}-vm-api"
  machine_type = var.api_machine_type
  zone         = var.zone

  tags = ["api", "ssh", "mlflow", "monitoring"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.api_disk_size_gb
      type  = "pd-standard"
    }
  }

  network_interface {
    subnetwork = var.subnet_id
    
    access_config {
      # Ephemeral public IP
    }
  }

  metadata = {
    startup-script = file("${path.module}/startup-api.sh")
  }

  service_account {
    email  = google_service_account.api_sa.email
    scopes = ["cloud-platform"]
  }
}

# Training Server Instance (with GPU)
resource "google_compute_instance" "training" {
  count = var.enable_training_server ? 1 : 0

  name         = "${var.project_id}-${var.environment}-vm-training"
  machine_type = var.training_machine_type
  zone         = var.zone

  tags = ["ssh"]

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/pytorch-latest-gpu"
      size  = var.training_disk_size_gb
      type  = "pd-standard"
    }
  }

  guest_accelerator {
    type  = var.training_gpu_type
    count = var.training_gpu_count
  }

  scheduling {
    on_host_maintenance = "TERMINATE"  # Required for GPUs
  }

  network_interface {
    subnetwork = var.subnet_id
    
    access_config {
      # Ephemeral public IP
    }
  }

  metadata = {
    startup-script = file("${path.module}/startup-training.sh")
  }

  service_account {
    email  = google_service_account.training_sa.email
    scopes = ["cloud-platform"]
  }
}

# Service Account for API Server
resource "google_service_account" "api_sa" {
  account_id   = "${var.project_id}-${var.environment}-api-sa"
  display_name = "Service Account for API Server"
}

# Service Account for Training Server
resource "google_service_account" "training_sa" {
  account_id   = "${var.project_id}-${var.environment}-training-sa"
  display_name = "Service Account for Training Server"
}
