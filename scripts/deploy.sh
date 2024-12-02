#!/bin/bash

# Exit immediately if any command fails
set -e

# Airflow home directory
AIRFLOW_HOME=~/airflow

# Health check retries and wait time
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_WAIT_TIME=10

# Function to check if containers are healthy
check_containers_health() {
    echo "Checking container health status..."
    for i in $(seq 1 $HEALTH_CHECK_RETRIES); do
        if docker compose ps | grep -q "unhealthy"; then
            echo "Attempt $i: Containers are not healthy, retrying in $HEALTH_CHECK_WAIT_TIME seconds..."
            sleep $HEALTH_CHECK_WAIT_TIME
        elif docker compose ps | grep -q "healthy"; then
            echo "All containers are healthy!"
            return 0
        fi
    done
    echo "Containers failed to become healthy after $HEALTH_CHECK_RETRIES attempts."
    return 1
}

# Stop existing containers
echo "Stopping existing containers..."
docker compose down || echo "No containers running to stop."

# Clean up unused Docker resources
echo "Pruning unused Docker resources..."
docker system prune -af --volumes

# Build Airflow containers
echo "Building Airflow containers..."
docker compose build --no-cache

# Initialize Airflow database
echo "Initializing Airflow..."
docker compose up airflow-init || {
    echo "Airflow initialization failed!"
    exit 1
}

# Start Airflow services
echo "Starting Airflow services..."
docker compose up -d

# Allow some time for containers to stabilize
echo "Waiting for containers to stabilize..."
sleep 30

# Perform health check
if ! check_containers_health; then
    echo "Container health check failed. Rolling back..."
    docker compose down
    exit 1
fi

# Print success message
echo "Airflow deployment successful!"
