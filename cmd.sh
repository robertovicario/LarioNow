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
APP_IMAGE="larionow"
APP_CONTAINER="larionow"
APP_PORT="8501"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GCP
PROJECT_ID="uninsubria-data-science"
DATASET_ID="larionow-dataset"
BQ_DATASET="larionow_dataset"
BQ_TABLE_NAME="measurements"
GCS_BUCKET="uninsubria-data-science-models"
GCS_PREFIX_MODELS="models/"
REGION_RUN="europe-west8"
SERVICE_ACCOUNT="289545143980-compute@developer.gserviceaccount.com"
ADC_HOST_PATH="${HOME}/.config/gcloud/application_default_credentials.json"
ADC_CONTAINER_PATH="/tmp/gcp-credentials.json"
WS_APP="larionow"
ARGS_APP=(
    --image="${REGION_RUN}-docker.pkg.dev/${PROJECT_ID}/${DATASET_ID}/app:latest"
    --region="${REGION_RUN}"
    --port=8501
    --memory=2Gi
    --cpu=1
    --service-account="${SERVICE_ACCOUNT}"
    --allow-unauthenticated
    --set-env-vars="GCP_PROJECT=${PROJECT_ID},BQ_DATASET=${BQ_DATASET},BQ_TABLE_NAME=${BQ_TABLE_NAME}"
)
JOB_COLLECTOR="collector"
ARGS_COLLECTOR=(
    --image="${REGION_RUN}-docker.pkg.dev/${PROJECT_ID}/${DATASET_ID}/${JOB_COLLECTOR}:latest"
    --region="${REGION_RUN}"
    --memory=2Gi
    --cpu=2
    --task-timeout=30m
    --service-account="${SERVICE_ACCOUNT}"
    --set-env-vars="GCP_PROJECT=${PROJECT_ID},BQ_DATASET=${BQ_DATASET},BQ_TABLE_NAME=${BQ_TABLE_NAME}"
)
JOB_RETRAINING="retraining"
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

start() {

    # Docker
    printer -start "Starting the project..."
    if docker ps -a --format '{{.Names}}' | grep -qx "${APP_CONTAINER}"; then
        docker start "${APP_CONTAINER}" >/dev/null || {
            handler $?
            return
        }
    else
        docker run -d \
            --name "${APP_CONTAINER}" \
            -p "${APP_PORT}:8501" \
            -v "${ADC_HOST_PATH}:${ADC_CONTAINER_PATH}:ro" \
            -e "GOOGLE_APPLICATION_CREDENTIALS=${ADC_CONTAINER_PATH}" \
            "${APP_IMAGE}:latest" || {
            handler $?
            return
        }
    fi

    # Handler
    handler 0
}

stop() {

    # Docker
    printer -stop "Stopping the project..."
    if docker ps -a --format '{{.Names}}' | grep -qx "${APP_CONTAINER}"; then
        docker stop "${APP_CONTAINER}" || {
            handler $?
            return
        }
    fi

    # Handler
    handler 0
}

build() {

    # Docker
    printer -setup "Building the project..."
    docker build -f docker/Dockerfile.app -t "${APP_IMAGE}:latest" . || {
        handler $?
        return
    }
    docker rm -f "${APP_CONTAINER}" >/dev/null 2>&1 || true
    docker run -d \
        --name "${APP_CONTAINER}" \
        -p "${APP_PORT}:8501" \
        -v "${ADC_HOST_PATH}:${ADC_CONTAINER_PATH}:ro" \
        -e "GOOGLE_APPLICATION_CREDENTIALS=${ADC_CONTAINER_PATH}" \
        "${APP_IMAGE}:latest" || {
        handler $?
        return
    }

    # Handler
    handler 0
}

setup() {

    # Virtual Environment
    printer -setup "Set up the project..."
    uv python install 3.12.10 || {
        handler $?
        return
    }
    uv venv --python 3.12.10 || {
        handler $?
        return
    }
    uv pip install -r packages/etl.txt || {
        handler $?
        return
    }
    uv pip install -r packages/train.txt || {
        handler $?
        return
    }
    uv pip install -r packages/notebook.txt || {
        handler $?
        return
    }

    # Handler
    handler 0
}

clean() {

    # TARGET
    printer -clean "Cleaning the project..."
    case "$1" in
        --env|--docker)
            ;;
        *)
            usage
            ;;
    esac

    # CLEAN
    case "$1" in
        --env)
            if [ -d "${PROJECT_ROOT}/.venv" ]; then
                rm -fv "${PROJECT_ROOT}/.venv" || {
                    handler $?
                    return
                }
            fi
            ;;
        --docker)
            if docker ps -a --format '{{.Names}}' | grep -qx "${APP_CONTAINER}"; then
                docker rm -f "${APP_CONTAINER}" >/dev/null 2>&1 || {
                    handler $?
                    return
                }
            fi
            if docker image inspect "${APP_IMAGE}:latest" >/dev/null 2>&1; then
                docker rmi "${APP_IMAGE}:latest" >/dev/null 2>&1 || {
                    handler $?
                    return
                }
            fi
            ;;
    esac

    # Handler
    handler 0
}

collector() {

    # COLLECTOR
    printer -start "Collecting data..."
    cd jobs || {
        handler $?
        return
    }
    uv run python job_collector.py
    STATUS=$?
    cd - >/dev/null || {
        handler $?
        return
    }

    # Handler
    handler $STATUS
}

retraining() {

    # RETRAINING
    printer -start "Retraining the model..."
    cd jobs || {
        handler $?
        return
    }
    uv run python job_retraining.py
    STATUS=$?
    cd - >/dev/null || {
        handler $?
        return
    }

    # Handler
    handler $STATUS
}

deploy() {

    # TARGET
    printer -setup "Deploying jobs on Google Cloud Run..."
    case "$1" in
        --app|--jobs)
            ;;
        *)
            usage
            ;;
    esac

    # BUILD
    gcloud builds submit --config cloudbuild.yaml . || {
        handler $?
        return
    }

    # DEPLOY
    case $1 in
        --app)
            gcloud run deploy "${WS_APP}" \
                "${ARGS_APP[@]}" || {
                    handler $?
                    return
                }
            ;;
        --jobs)
            gcloud run jobs deploy "${JOB_COLLECTOR}" \
                "${ARGS_COLLECTOR[@]}" || {
                    handler $?
                    return
                }

            gcloud run jobs deploy "${JOB_RETRAINING}" \
                "${ARGS_RETRAINING[@]}" || {
                    handler $?
                    return
                }
            ;;
        *)
            usage
            ;;
    esac

    # Handler
    handler 0
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
    - [${ICON_SETUP}] build
    - [${ICON_SETUP}] setup
    - [${ICON_CLEAN}] clean [--env|--docker]
    - [${ICON_START}] collector
    - [${ICON_START}] retraining
    - [${ICON_SETUP}] deploy [--app|--jobs]

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
    build)
        build
        ;;
    setup)
        setup
        ;;
    clean)
        clean $2
        ;;
    collector)
        collector
        ;;
    retraining)
        retraining
        ;;
    deploy)
        deploy $2
        ;;
    *)
        usage
        ;;
esac

# -------------------------
