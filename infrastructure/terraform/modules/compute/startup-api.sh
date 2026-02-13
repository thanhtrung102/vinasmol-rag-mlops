#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Git
apt-get install -y git make

# Clone repository
cd /home/ubuntu
git clone https://github.com/thanthtrung102/vinasmol-rag-mlops.git
cd vinasmol-rag-mlops

# Start services
docker-compose up -d

echo "API server setup complete"
