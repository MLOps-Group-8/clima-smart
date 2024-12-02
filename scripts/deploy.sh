#!/bin/bash

# Exit immediately if any command fails
set -e

# Airflow home directory
AIRFLOW_HOME=~/clima-smart

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

# Check if containers are already running
if sudo docker compose ps | grep -q "Up"; then
    echo "Containers are already running. Updating DAGs..."

    # Remove old DAGs inside the containers
    echo "Removing old DAGs from the containers..."
    sudo docker exec -it airflow-webserver bash -c "rm -rf /opt/airflow/dags/*"
    sudo docker exec -it airflow-scheduler bash -c "rm -rf /opt/airflow/dags/*"

    # Copy updated DAGs into the containers
    echo "Copying updated DAGs into the containers..."
    sudo docker cp dags/. airflow-webserver:/opt/airflow/dags/
    sudo docker cp dags/. airflow-scheduler:/opt/airflow/dags/

    # Restart Airflow services to pick up changes
    echo "Restarting Airflow services to apply changes..."
    sudo docker compose restart webserver scheduler

    # Trigger DAGs via REST API
    echo "Triggering DAGs via REST API..."
    for dag in $(ls dags/*.py | xargs -n 1 basename | sed 's/.py//'); do
        echo "Triggering DAG: $dag"
        curl -X POST "http://localhost:8080/api/v1/dags/$dag/dagRuns" \
             -H "Content-Type: application/json" \
             --user "airflow:airflow" \
             -d '{"conf": {}}'
    done

    echo "DAGs updated and triggered successfully."
    exit 0
fi

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
sudo docker compose build

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
sudo docker compose up

# Allow some time for containers to stabilize
echo "Waiting for containers to stabilize..."
sleep 100

# Perform health check
if ! check_containers_health; then
    echo "Container health check failed. Rolling back..."
    sudo docker compose down
    exit 1
fi

# Print success message
echo "Airflow deployment successful!"
