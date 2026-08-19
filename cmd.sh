#!/bin/bash

# =========================
# Configurations
# =========================

# Icons
ICON_START="▶"     # U+25B6
ICON_STOP="■"      # U+25A0
ICON_SETUP="⚙"     # U+2699
ICON_DOWNLOAD="↓"  # U+2193
ICON_CLEAN="♻"     # U+267B
ICON_OK="✓"        # U+2713
ICON_ERR="✗"       # U+2717

# Colors
RESET="\033[0m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"


# =========================
# Constants
# =========================
# App
APP_IMAGE="app-larionow"
APP_CONTAINER="larionow"
APP_PORT="8501"

# GCP
PROJECT_ID="uninsubria-data-science"
DATASET_ID="larionow-dataset"
BQ_DATASET="larionow_dataset"
BQ_TABLE_NAME="measurements"
GCS_BUCKET="uninsubria-data-science-models"
GCS_PREFIX_MODELS="models/"
REGION_RUN="europe-west8"
JOB_COLLECTOR="collector"
JOB_RETRAINING="retraining"
SERVICE_ACCOUNT="289545143980-compute@developer.gserviceaccount.com"
ARGS_COLLECTOR=(
    --image="${REGION_RUN}-docker.pkg.dev/${PROJECT_ID}/${DATASET_ID}/${JOB_COLLECTOR}:latest"
    --region="${REGION_RUN}"
    --memory=2Gi
    --cpu=2
    --task-timeout=30m
    --service-account="${SERVICE_ACCOUNT}"
    --set-env-vars="GCP_PROJECT=${PROJECT_ID},BQ_DATASET=${BQ_DATASET},BQ_TABLE_NAME=${BQ_TABLE_NAME}"
)
ARGS_RETRAINING=(
    --image="${REGION_RUN}-docker.pkg.dev/${PROJECT_ID}/${DATASET_ID}/${JOB_RETRAINING}:latest"
    --region="${REGION_RUN}"
    --memory=4Gi
    --cpu=2
    --task-timeout=60m
    --service-account="${SERVICE_ACCOUNT}"
    --set-env-vars="GCP_PROJECT=${PROJECT_ID},BQ_DATASET=${BQ_DATASET},BQ_TABLE_NAME=${BQ_TABLE_NAME},GCS_BUCKET=${GCS_BUCKET},GCS_PREFIX_MODELS=${GCS_PREFIX_MODELS}"
)

# =========================
# Methods
# =========================

setup() {

    printer -setup "Setting up the project..."
    case "$1" in

        --dev)

            # Environment
            uv python install 3.12.10
            uv venv --python 3.12.10

            # Requirements
            uv pip install -r packages/etl.txt
            uv pip install -r packages/train.txt
            uv pip install -r packages/notebook.txt
            ;;

        --app)

            # Docker
            docker build -f docker/Dockerfile.app -t "${APP_IMAGE}:latest" . || {
                handler $?
                return
            }
            docker rm -f "${APP_CONTAINER}" >/dev/null 2>&1 || true
            docker run -d \
                --name "${APP_CONTAINER}" \
                -p "${APP_PORT}:8501" \
                "${APP_IMAGE}:latest"
            ;;

        *)
            usage
            ;;
    esac

    # Handler
    STATUS=$?
    handler $STATUS
}

start() {

    # Docker
    printer -start "Starting the project..."
    if docker ps -a --format '{{.Names}}' | grep -qx "${APP_CONTAINER}"; then
        docker start "${APP_CONTAINER}" >/dev/null
    else
        docker run -d \
            --name "${APP_CONTAINER}" \
            -p "${APP_PORT}:8501" \
            "${APP_IMAGE}:latest"
    fi

    # Handler
    STATUS=$?
    handler $STATUS
}

stop() {

    # Docker
    printer -stop "Stopping the project..."
    if docker ps -a --format '{{.Names}}' | grep -qx "${APP_CONTAINER}"; then
        docker stop "${APP_CONTAINER}"
    else
        printer -success "No app container to stop"
        return
    fi

    # Handler
    STATUS=$?
    handler $STATUS
}

debug() {

    # Docker
    printer -setup "Starting debug..."
    docker rm -f "${APP_CONTAINER}" >/dev/null 2>&1 || true
    docker build --no-cache -f docker/Dockerfile.app -t "${APP_IMAGE}:latest" . || {
        handler $?
        return
    }
    docker run -d \
        --name "${APP_CONTAINER}" \
        -p "${APP_PORT}:8501" \
        "${APP_IMAGE}:latest"

    # Handler
    STATUS=$?
    handler $STATUS
}

collector() {

    # JOB
    printer -start "Starting the data collection..."
    cd jobs || exit 1
    uv run python collector.py
    STATUS=$?
    cd - >/dev/null || exit 1

    # Handler
    handler $STATUS
}

retraining() {

    # RETRAINING
    printer -start "Starting the model retraining..."
    cd jobs || exit 1
    uv run python retraining.py
    STATUS=$?
    cd - >/dev/null || exit 1

    # Handler
    handler $STATUS
}

deploy_jobs() {

    # BUILD
    printer -setup "Deploying jobs on Google Cloud Run..."
    gcloud builds submit --config cloudbuild.yaml . || {
        handler $?
        return
    }

    # DEPLOY
    gcloud run jobs deploy "${JOB_COLLECTOR}" \
        "${ARGS_COLLECTOR[@]}" || {
            handler $?
            return
        }

    gcloud run jobs deploy "${JOB_RETRAINING}" \
        "${ARGS_RETRAINING[@]}"

    # Handler
    handler $?
}

# =========================
# Handlers
# =========================

usage() {

    cat <<EOF

1. Usage:
    - bash $0 <command>

2. Commands:
    - [${ICON_START}] start
    - [${ICON_STOP}] stop
    - [${ICON_SETUP}] setup [--dev|--app]
    - [${ICON_SETUP}] debug
    - [${ICON_START}] collector
    - [${ICON_START}] retraining
    - [${ICON_SETUP}] deploy_jobs

EOF
    exit 1
}

printer() {

    local STATUS="$1"
    local MESSAGE="$2"
    local ICON=""
    local COLOR=""
    case "$STATUS" in
        -start)
            ICON="$ICON_START"
            COLOR="$BLUE"
            ;;
        -stop)
            ICON="$ICON_STOP"
            COLOR="$RED"
            ;;
        -debug)
            ICON="$ICON_START"
            COLOR="$CYAN"
            ;;
        -setup)
            ICON="$ICON_SETUP"
            COLOR="$MAGENTA"
            ;;
        -clean)
            ICON="$ICON_CLEAN"
            COLOR="$YELLOW"
            ;;
        -success)
            ICON="$ICON_OK"
            COLOR="$GREEN"
            ;;
        -error)
            ICON="$ICON_ERR"
            COLOR="$RED"
            ;;
        *)
            ICON="$ICON_ERR"
            COLOR="$RED"
            ;;
    esac
    echo ""
    echo -e "${COLOR}[${ICON}] ${MESSAGE}${RESET}"
    echo ""
}

handler() {

    local STATUS=$1
    if [ $STATUS -eq 0 ]; then
        printer -success "Process completed successfully"
    else
        printer -error "An unexpected error occurred"
        exit 1
    fi
}

case $1 in
    start)
        start
        ;;
    stop)
        stop
        ;;
    setup)
        setup "$2"
        ;;
    debug)
        debug
        ;;
    collector)
        collector
        ;;
    retraining)
        retraining
        ;;
    deploy_jobs)
        deploy_jobs
        ;;
    *)
        usage
        ;;
esac

# -------------------------
