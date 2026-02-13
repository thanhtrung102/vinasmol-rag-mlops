# Networking Module - VPC, Subnets, Firewall Rules

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${var.project_id}-${var.environment}-vpc"
  auto_create_subnetworks = false
  description             = "VPC for VinaSmol RAG MLOps ${var.environment}"
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = "${var.project_id}-${var.environment}-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc.id
  description   = "Subnet for ${var.environment} environment"
}

# Firewall - Allow SSH
resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.project_id}-${var.environment}-allow-ssh"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]  # Restrict in production
  target_tags   = ["ssh"]
}

# Firewall - Allow API (8000)
resource "google_compute_firewall" "allow_api" {
  name    = "${var.project_id}-${var.environment}-allow-api"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["api"]
}

# Firewall - Allow MLflow (8080)
resource "google_compute_firewall" "allow_mlflow" {
  name    = "${var.project_id}-${var.environment}-allow-mlflow"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["mlflow"]
}

# Firewall - Allow Monitoring (3000, 9090)
resource "google_compute_firewall" "allow_monitoring" {
  name    = "${var.project_id}-${var.environment}-allow-monitoring"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["3000", "9090"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["monitoring"]
}

# Cloud NAT for outbound internet
resource "google_compute_router" "router" {
  name    = "${var.project_id}-${var.environment}-router"
  region  = var.region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name   = "${var.project_id}-${var.environment}-nat"
  router = google_compute_router.router.name
  region = var.region

  nat_ip_allocate_option = "AUTO_ONLY"

  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
