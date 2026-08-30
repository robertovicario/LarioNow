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

# Configurations
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GCP
GCP_PROJECT="uninsubria-data-science"
GCP_SERVICE_ACCOUNT="289545143980-compute@developer.gserviceaccount.com"
GCR_REGION="europe-west8"
GCS_BUCKET="uninsubria-data-science-models"
GCS_PREFIX_MODELS="models/"
BQ_DATASET="larionow_dataset"
BQ_TABLE_NAME="measurements"
ADC_HOST_PATH="${HOME}/.config/gcloud/application_default_credentials.json"
ADC_CONTAINER_PATH="/tmp/gcp-credentials.json"

# Docker
DOCKER_GCP_REGISTRY="${GCR_REGION}-docker.pkg.dev"
DOCKER_GCP_REPOSITORY="uninsubria-data-science/larionow"
DOCKER_PLATFORM="linux/amd64"

# App
WS_APP="larionow-app"
PORT_APP="8501"
DOCKERFILE_APP="docker/Dockerfile.app"
IMAGE_APP_NAME="app:latest"
IMAGE_APP="${DOCKER_GCP_REGISTRY}/${DOCKER_GCP_REPOSITORY}/${IMAGE_APP_NAME}"
ARGS_APP=(
    --image="${IMAGE_APP}"
    --region="${GCR_REGION}"
    --port="${PORT_APP}"
    --memory=2Gi
    --cpu=2
    --service-account="${GCP_SERVICE_ACCOUNT}"
    --allow-unauthenticated
    --set-env-vars="GCP_PROJECT=${GCP_PROJECT},BQ_DATASET=${BQ_DATASET},BQ_TABLE_NAME=${BQ_TABLE_NAME},GCS_BUCKET=${GCS_BUCKET},GCS_PREFIX_MODELS=${GCS_PREFIX_MODELS}"
)

# API
WS_API="larionow-api"
PORT_API="8080"
DOCKERFILE_API="docker/Dockerfile.api"
IMAGE_API_NAME="api:latest"
IMAGE_API="${DOCKER_GCP_REGISTRY}/${DOCKER_GCP_REPOSITORY}/${IMAGE_API_NAME}"
ARGS_API=(
    --image="${IMAGE_API}"
    --region="${GCR_REGION}"
    --port="${PORT_API}"
    --memory=2Gi
    --cpu=2
    --service-account="${GCP_SERVICE_ACCOUNT}"
    --allow-unauthenticated
    --set-env-vars="GCP_PROJECT=${GCP_PROJECT},BQ_DATASET=${BQ_DATASET},BQ_TABLE_NAME=${BQ_TABLE_NAME},GCS_BUCKET=${GCS_BUCKET},GCS_PREFIX_MODELS=${GCS_PREFIX_MODELS}"
)

# Collector
JOB_ETL="larionow-collector"
DOCKERFILE_ETL="docker/Dockerfile.etl"
IMAGE_ETL_NAME="collector:latest"
IMAGE_ETL="${DOCKER_GCP_REGISTRY}/${DOCKER_GCP_REPOSITORY}/${IMAGE_ETL_NAME}"
ARGS_ETL=(
    --image="${IMAGE_ETL}"
    --region="${GCR_REGION}"
    --memory=2Gi
    --cpu=2
    --task-timeout=30m
    --service-account="${GCP_SERVICE_ACCOUNT}"
    --set-env-vars="GCP_PROJECT=${GCP_PROJECT},BQ_DATASET=${BQ_DATASET},BQ_TABLE_NAME=${BQ_TABLE_NAME}"
)

# Retraining
JOB_TRAIN="larionow-retraining"
DOCKERFILE_TRAIN="docker/Dockerfile.train"
IMAGE_TRAIN_NAME="retraining:latest"
IMAGE_TRAIN="${DOCKER_GCP_REGISTRY}/${DOCKER_GCP_REPOSITORY}/${IMAGE_TRAIN_NAME}"
ARGS_TRAIN=(
    --image="${IMAGE_TRAIN}"
    --region="${GCR_REGION}"
    --memory=4Gi
    --cpu=2
    --task-timeout=60m
    --service-account="${GCP_SERVICE_ACCOUNT}"
    --set-env-vars="GCP_PROJECT=${GCP_PROJECT},BQ_DATASET=${BQ_DATASET},BQ_TABLE_NAME=${BQ_TABLE_NAME},GCS_BUCKET=${GCS_BUCKET},GCS_PREFIX_MODELS=${GCS_PREFIX_MODELS}"
)
# =========================
# Helpers
# =========================

exec_docker_build() {

    local DOCKERFILE="$1"
    local IMAGE_NAME="$2"

    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        docker build \
            --platform "${DOCKER_PLATFORM}" \
            -f "${DOCKERFILE}" \
            -t "${IMAGE_NAME}" \
            . || {
            handler $?
            return
        }
    fi
}

# =========================
# Methods
# =========================

start() {

    # Helpers
    start_fx() {

        if docker ps -a --format '{{.Names}}' | grep -qx "$1"; then
            docker start "$1" >/dev/null || {
                handler $?
                return
            }
        else
            docker run -d \
                --platform "${DOCKER_PLATFORM}" \
                --name "$1" \
                -p "$2:$2" \
                -v "${ADC_HOST_PATH}:${ADC_CONTAINER_PATH}:ro" \
                -e "GOOGLE_APPLICATION_CREDENTIALS=${ADC_CONTAINER_PATH}" \
                "$3" || {
                handler $?
                return
            }
        fi
    }

    # START
    printer -start "Starting the application..."
    start_fx "${WS_APP}" "${PORT_APP}" "${IMAGE_APP_NAME}" "${DOCKERFILE_APP}"
    start_fx "${WS_API}" "${PORT_API}" "${IMAGE_API_NAME}" "${DOCKERFILE_API}"

    # Handler
    handler 0
}

stop() {

    # Helpers
    stop_fx() {

        if docker ps -a --format '{{.Names}}' | grep -qx "$1"; then
            docker stop "$1" >/dev/null || {
                handler $?
                return
            }
        fi
    }

    # STOP
    printer -stop "Stopping the application..."
    stop_fx "${WS_APP}"
    stop_fx "${WS_API}"

    # Handler
    handler 0
}

build() {

    # BUILD
    printer -setup "Building the application..."
    exec_docker_build "${DOCKERFILE_APP}" "${IMAGE_APP_NAME}" || {
        handler $?
        return
    }
    exec_docker_build "${DOCKERFILE_API}" "${IMAGE_API_NAME}" || {
        handler $?
        return
    }

    # Handler
    handler 0

    # START
    start
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
    for package in packages/*.txt; do
        uv pip install -r "$package" || {
            handler $?
            return
        }
    done

    # Handler
    handler 0
}

clean() {

    # Helpers
    clean_env() {

        if [ -d "${PROJECT_ROOT}/.venv" ]; then
            rm -rfv "${PROJECT_ROOT}/.venv" || {
                handler $?
                return
            }
        fi
    }

    clean_docker() {
        helper_fx() {

            if docker ps -a --format '{{.Names}}' | grep -qx "$1"; then
                docker rm -f "$1" >/dev/null 2>&1 || {
                    handler $?
                    return
                }
            fi

            local IMAGE
            for IMAGE in "$@"; do
                if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
                    docker rmi -f "${IMAGE}" >/dev/null 2>&1 || {
                        handler $?
                        return
                    }
                fi
            done
        }

        helper_fx "${WS_APP}" "${WS_APP}" "${IMAGE_APP_NAME}" "${IMAGE_APP}"
        helper_fx "${WS_API}" "${WS_API}" "${IMAGE_API_NAME}" "${IMAGE_API}"
        helper_fx "${JOB_ETL}" "${JOB_ETL}" "${IMAGE_ETL_NAME}" "${IMAGE_ETL}"
        helper_fx "${JOB_TRAIN}" "${JOB_TRAIN}" "${IMAGE_TRAIN_NAME}" "${IMAGE_TRAIN}"
    }

    # CLEAN
    printer -clean "Cleaning the project..."
    case "$1" in
        --env)
            clean_env
        ;;
        --docker)
            clean_docker
        ;;
        --all)
            clean_env
            clean_docker
        ;;
    esac

    # Handler
    handler 0
}

deploy() {

    # Helpers
    deploy_fx() {

        local TARGET="$1"
        local DOCKERFILE="$2"
        local IMAGE="$3"
        local NAME="$4"
        shift 4

        exec_docker_build "${DOCKERFILE}" "${IMAGE}" || {
            handler $?
            return
        }
        docker push "${IMAGE}" || {
            handler $?
            return
        }

        if [ "${TARGET}" = "jobs" ]; then
            gcloud run jobs deploy "${NAME}" "${@:1}" || {
                handler $?
                return
            }
        else
            gcloud run deploy "${NAME}" "${@:1}" || {
                handler $?
                return
            }
        fi
    }

    deploy_app() {

        deploy_fx service "${DOCKERFILE_APP}" "${IMAGE_APP}" "${WS_APP}" "${ARGS_APP[@]}"
        deploy_fx service "${DOCKERFILE_API}" "${IMAGE_API}" "${WS_API}" "${ARGS_API[@]}"
    }

    deploy_jobs() {

        deploy_fx jobs "${DOCKERFILE_ETL}" "${IMAGE_ETL}" "${JOB_ETL}" "${ARGS_ETL[@]}"
        deploy_fx jobs "${DOCKERFILE_TRAIN}" "${IMAGE_TRAIN}" "${JOB_TRAIN}" "${ARGS_TRAIN[@]}"
    }

    # TARGET
    printer -setup "Deploying the web instances..."
    case "$1" in
        --app)
            deploy_app
            ;;
        --jobs)
            deploy_jobs
            ;;
        --all)
            deploy_app
            deploy_jobs
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

    # Operations
    cat <<EOF

1. Usage:
    - bash $0 <command>

2. Commands:
    - [${ICON_START}] start
    - [${ICON_STOP}] stop
    - [${ICON_SETUP}] build
    - [${ICON_SETUP}] setup
    - [${ICON_CLEAN}] clean <target>
       ├──  --env        |> environment resources
       ├──  --docker     |> docker resources
       └──  --all        |> all related resources
    - [${ICON_SETUP}] deploy [option] <target>
       ├──  --app        |> web services
       ├──  --jobs       |> job services
       └──  --all        |> all web instances

EOF
    exit 1
}

printer() {

    # Operations
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

    # Operations
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
    deploy)
        shift
        deploy "$@"
        ;;
    *)
        usage
        ;;
esac

# -------------------------
