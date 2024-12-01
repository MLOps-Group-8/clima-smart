#!/bin/bash
set -e

AIRFLOW_HOME=~/airflow

# Function to check if containers are healthy
check_containers_health() {
    local retries=5
    local wait_time=10
    
    for i in $(seq 1 $retries); do
        if docker compose ps | grep -q "unhealthy"; then
            echo "Containers are not healthy, waiting..."
            sleep $wait_time
        else
            return 0
        fi
    done
    return 1
}

echo "Stopping existing containers..."
docker compose down

echo "Pruning unused Docker resources..."
docker system prune -f

echo "Building Airflow containers..."
docker compose build --no-cache

echo "Initializing Airflow..."
docker compose up airflow-init

echo "Starting Airflow services..."
docker compose up -d

echo "Waiting for containers to be healthy..."
sleep 30

if ! check_containers_health; then
    echo "Container health check failed. Rolling back..."
    docker compose down
    exit 1
fi

echo "Deployment successful!"