# Terraform Infrastructure as Code

This directory contains Terraform configurations for deploying the VinaSmol RAG MLOps infrastructure on Google Cloud Platform (GCP).

## Architecture Overview

The infrastructure provisions:

### Compute Resources
- **API Server**: VM instance running FastAPI (n2-standard-4)
- **Training Server**: GPU-enabled VM for model fine-tuning (n1-standard-4 + T4 GPU)

### Managed Services
- **Cloud Storage**: Buckets for model artifacts, data, and MLflow storage
- **Cloud SQL**: PostgreSQL instance for MLflow backend
- **Memorystore Redis**: Cache layer for RAG queries

### Networking
- **VPC**: Custom network with private subnets
- **Firewall Rules**: Security groups for API (8000), MLflow (8080), monitoring (3000, 9090)
- **Cloud NAT**: Outbound internet access for private instances
- **Load Balancer**: HTTP(S) load balancer for API traffic

### Container Infrastructure
- **Artifact Registry**: Docker image storage
- **GKE Cluster** (optional): Kubernetes for containerized workloads

## Directory Structure

```
infrastructure/terraform/
├── backend.tf              # Remote state configuration (GCS)
├── main.tf                 # Root module orchestration
├── variables.tf            # Input variables
├── outputs.tf              # Output values
├── terraform.tfvars.example # Example configuration
├── modules/
│   ├── networking/         # VPC, subnets, firewall rules
│   ├── compute/            # VM instances, startup scripts
│   ├── storage/            # Cloud Storage buckets
│   └── monitoring/         # Cloud Monitoring, alerting
└── environments/
    ├── dev.tfvars         # Development environment
    └── prod.tfvars        # Production environment
```

## Prerequisites

1. **Google Cloud SDK**:
   ```bash
   # Install gcloud
   curl https://sdk.cloud.google.com | bash
   
   # Initialize
   gcloud init
   
   # Authenticate
   gcloud auth application-default login
   ```

2. **Terraform**:
   ```bash
   # Install Terraform >= 1.6
   wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
   unzip terraform_1.6.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/
   ```

3. **GCP Project**:
   ```bash
   # Create project
   gcloud projects create vinasmol-rag-mlops --name="VinaSmol RAG MLOps"
   
   # Set project
   gcloud config set project vinasmol-rag-mlops
   
   # Enable billing
   gcloud beta billing accounts list
   gcloud beta billing projects link vinasmol-rag-mlops --billing-account=BILLING_ACCOUNT_ID
   ```

4. **Enable Required APIs**:
   ```bash
   gcloud services enable compute.googleapis.com \
     storage.googleapis.com \
     sqladmin.googleapis.com \
     redis.googleapis.com \
     artifactregistry.googleapis.com \
     container.googleapis.com
   ```

## Quick Start

### 1. Configure Variables

```bash
# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
vim terraform.tfvars
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Plan Deployment

```bash
# Development environment
terraform plan -var-file=environments/dev.tfvars

# Production environment
terraform plan -var-file=environments/prod.tfvars
```

### 4. Deploy Infrastructure

```bash
# Development
terraform apply -var-file=environments/dev.tfvars

# Production
terraform apply -var-file=environments/prod.tfvars
```

### 5. Access Outputs

```bash
# Get API server IP
terraform output api_server_ip

# Get all outputs
terraform output
```

## Environment Configurations

### Development (`dev.tfvars`)
- **Machine Types**: Smaller instances (n2-standard-2)
- **Storage**: Standard storage class
- **High Availability**: Single zone
- **Cost**: ~$200/month

### Production (`prod.tfvars`)
- **Machine Types**: Optimized instances (n2-standard-4)
- **Storage**: Multi-regional with versioning
- **High Availability**: Multi-zone with failover
- **Cost**: ~$500/month

## Resource Naming Convention

All resources follow the pattern: `{project}-{environment}-{resource_type}-{name}`

Examples:
- `vinasmol-dev-vm-api`
- `vinasmol-prod-bucket-artifacts`
- `vinasmol-dev-sql-mlflow`

## Cost Estimation

### Development Environment

| Resource | Type | Monthly Cost (USD) |
|----------|------|-------------------|
| API Server | n2-standard-2 | $50 |
| Training Server | n1-standard-4 + T4 | $150 |
| Cloud Storage | 100GB | $3 |
| Cloud SQL | db-f1-micro | $15 |
| Redis | 1GB | $30 |
| Network | Egress 100GB | $12 |
| **Total** | | **~$260** |

### Production Environment

| Resource | Type | Monthly Cost (USD) |
|----------|------|-------------------|
| API Server (HA) | 2x n2-standard-4 | $200 |
| Training Server | n1-standard-8 + V100 | $600 |
| Cloud Storage | 1TB | $30 |
| Cloud SQL | db-n1-standard-2 | $150 |
| Redis | 5GB | $150 |
| Load Balancer | HTTP(S) | $20 |
| Network | Egress 500GB | $60 |
| **Total** | | **~$1,210** |

## Terraform Commands Reference

```bash
# Initialize and download providers
terraform init

# Validate configuration
terraform validate

# Format code
terraform fmt -recursive

# Plan changes
terraform plan

# Apply changes
terraform apply

# Destroy infrastructure
terraform destroy

# Show current state
terraform show

# List resources
terraform state list

# Import existing resource
terraform import module.compute.google_compute_instance.api projects/PROJECT/zones/ZONE/instances/NAME
```

## Remote State Management

State is stored in GCS bucket: `gs://{project-id}-terraform-state`

**Features**:
- State locking with Cloud Storage
- Versioning enabled
- Encrypted at rest
- Lifecycle management (90-day retention)

## Security Considerations

1. **Firewall Rules**: Only required ports exposed
2. **IAM**: Least privilege service accounts
3. **Encryption**: All data encrypted at rest and in transit
4. **Secrets**: Use Secret Manager, never commit to git
5. **VPC**: Private subnets for databases
6. **Cloud Armor**: DDoS protection on load balancer (prod)

## Maintenance

### Update Infrastructure

```bash
# Pull latest Terraform code
git pull origin main

# Review changes
terraform plan -var-file=environments/prod.tfvars

# Apply updates
terraform apply -var-file=environments/prod.tfvars
```

### Backup State

```bash
# State is automatically backed up in GCS
# To manually download:
gsutil cp gs://vinasmol-rag-mlops-terraform-state/default.tfstate backup.tfstate
```

### Disaster Recovery

```bash
# Restore from state backup
terraform state pull > backup.tfstate

# Re-create infrastructure
terraform apply -var-file=environments/prod.tfvars
```

## Troubleshooting

### State Lock Issues

```bash
# Force unlock (use with caution)
terraform force-unlock LOCK_ID
```

### API Quota Exceeded

```bash
# Request quota increase
gcloud compute project-info describe --project=vinasmol-rag-mlops
```

### Resource Already Exists

```bash
# Import existing resource
terraform import module.compute.google_compute_instance.api PROJECT/ZONE/INSTANCE_NAME
```

## Next Steps

After infrastructure is deployed:

1. **Configure DNS**: Point domain to load balancer IP
2. **Setup CI/CD**: Configure GitHub Actions to deploy code
3. **Enable Monitoring**: Configure Cloud Monitoring alerts
4. **Setup Backups**: Configure automated backups for databases
5. **Security Hardening**: Implement Cloud Armor, IAP, VPC Service Controls

## Support

- **Issues**: https://github.com/thanhtrung102/vinasmol-rag-mlops/issues
- **Documentation**: See main project README.md
- **GCP Documentation**: https://cloud.google.com/docs

---

*Terraform configurations for VinaSmol RAG MLOps - Production-ready infrastructure for Vietnamese LLM and RAG systems*
