#!/bin/bash

# Exit immediately if any command fails
set -e

# Airflow home directory
AIRFLOW_HOME=~/airflow

# Source path of Google Cloud credentials
SOURCE_GCLOUD_CREDENTIALS="/home/shah_darsha/gcp/key.json"

# Destination path for Google Cloud credentials
DEST_GCLOUD_CREDENTIALS="$AIRFLOW_HOME/config/key.json"

# Health check retries and wait time
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_WAIT_TIME=10

# Function to check if containers are healthy
check_containers_health() {
    echo "Checking container health status..."
    for i in $(seq 1 $HEALTH_CHECK_RETRIES); do
        if sudo docker compose ps | grep -q "unhealthy"; then
            echo "Attempt $i: Containers are not healthy, retrying in $HEALTH_CHECK_WAIT_TIME seconds..."
            sleep $HEALTH_CHECK_WAIT_TIME
        elif sudo docker compose ps | grep -q "healthy"; then
            echo "All containers are healthy!"
            return 0
        fi
    done
    echo "Containers failed to become healthy after $HEALTH_CHECK_RETRIES attempts."
    return 1
}

# Copy Google Cloud credentials
echo "Setting up Google Cloud credentials..."
if [ -f "$SOURCE_GCLOUD_CREDENTIALS" ]; then
    mkdir -p "$(dirname "$DEST_GCLOUD_CREDENTIALS")"
    cp "$SOURCE_GCLOUD_CREDENTIALS" "$DEST_GCLOUD_CREDENTIALS"
    echo "Google Cloud credentials copied to $DEST_GCLOUD_CREDENTIALS"
else
    echo "Google Cloud credentials not found at $SOURCE_GCLOUD_CREDENTIALS. Exiting."
    exit 1
fi

# Stop existing containers
echo "Stopping existing containers..."
sudo docker compose down || echo "No containers running to stop."

# Clean up unused Docker resources
echo "Pruning unused Docker resources..."
sudo docker system prune -af --volumes

# Build Airflow containers
echo "Building Airflow containers..."
sudo docker compose build --no-cache

# Initialize Airflow database
echo "Initializing Airflow..."
sudo docker compose up airflow-init || {
    echo "Airflow initialization failed!"
    exit 1
}

# Start Airflow services
echo "Starting Airflow services..."
sudo docker compose up -d

# Allow some time for containers to stabilize
echo "Waiting for containers to stabilize..."
sleep 30

# Perform health check
if ! check_containers_health; then
    echo "Container health check failed. Rolling back..."
    sudo docker compose down
    exit 1
fi

# Print success message
echo "Airflow deployment successful!"
