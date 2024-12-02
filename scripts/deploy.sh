#!/bin/bash

# Exit immediately if any command fails
set -e

# Airflow home directory
AIRFLOW_HOME=~/airflow

# Health check retries and wait time
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_WAIT_TIME=10
AIRFLOW_DB_INITIALIZED_FLAG="$AIRFLOW_HOME/.db_initialized"

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

# Stop existing services
echo "Stopping existing services if running..."
if sudo docker compose ps | grep -q "Up"; then
    echo "Services are running. Stopping them now..."
    sudo docker compose down || {
        echo "Failed to stop running containers. Exiting."
        exit 1
    }
else
    echo "No running services found."
fi

# Clean up unused Docker resources
echo "Pruning unused Docker resources..."
sudo docker system prune -af --volumes

# Copy Google Cloud credentials
echo "Setting up Google Cloud credentials..."
SOURCE_GCLOUD_CREDENTIALS="/home/shah_darsha/gcp/key.json"
DEST_GCLOUD_CREDENTIALS="$AIRFLOW_HOME/config/key.json"

if [ -f "$SOURCE_GCLOUD_CREDENTIALS" ]; then
    mkdir -p "$(dirname "$DEST_GCLOUD_CREDENTIALS")"
    cp "$SOURCE_GCLOUD_CREDENTIALS" "$DEST_GCLOUD_CREDENTIALS"
    echo "Google Cloud credentials copied to $DEST_GCLOUD_CREDENTIALS"
else
    echo "Google Cloud credentials not found at $SOURCE_GCLOUD_CREDENTIALS. Exiting."
    exit 1
fi

# Build Airflow containers
echo "Building Airflow containers..."
sudo docker compose build --no-cache

# Initialize Airflow database only if needed
if [ ! -f "$AIRFLOW_DB_INITIALIZED_FLAG" ]; then
    echo "Initializing Airflow database for the first time..."
    sudo docker compose up airflow-init || {
        echo "Airflow initialization failed!"
        exit 1
    }
    touch "$AIRFLOW_DB_INITIALIZED_FLAG"
else
    echo "Airflow database already initialized. Skipping initialization."
fi

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
