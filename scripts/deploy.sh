#!/bin/bash

# Exit immediately if any command fails
set -e

# Airflow home directory
AIRFLOW_HOME=~/airflow

# Health check retries and wait time
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_WAIT_TIME=10

# Google Cloud credentials path
GCLOUD_CREDENTIALS_PATH="$AIRFLOW_HOME/config/key.json"

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
if [ -f "./key.json" ]; then
    mkdir -p "$AIRFLOW_HOME/config"
    cp ./key.json "$GCLOUD_CREDENTIALS_PATH"
    echo "Google Cloud credentials copied to $GCLOUD_CREDENTIALS_PATH"
else
    echo "Google Cloud credentials (key.json) not found in the current directory. Exiting."
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
