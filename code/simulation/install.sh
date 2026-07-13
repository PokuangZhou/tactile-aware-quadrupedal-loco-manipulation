#!/usr/bin/env bash

set -Eeuo pipefail

ENV_NAME="preview_tac_quad"
PYTHON_VERSION="3.8"
ARCHIVE_NAME="IsaacGym_Preview_4_Modified.tar.gz"
GDRIVE_FILE_ID="1fH_bSIgfvtkPTUbnkNu0rhX7bVdJrdEt"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
THIRDPARTY_DIR="${SCRIPT_DIR}/thirdparty"
ARCHIVE_PATH="${THIRDPARTY_DIR}/${ARCHIVE_NAME}"
ISAACGYM_DIR="${THIRDPARTY_DIR}/IsaacGym_Preview_TacSL_Package_modify"
ISAACGYM_PYTHON_DIR="${ISAACGYM_DIR}/isaacgym/python"
STANDARD_ISAACGYM_PYTHON_DIR="${THIRDPARTY_DIR}/isaacgym/python"
RSL_RL_DIR="${THIRDPARTY_DIR}/rsl_rl"
SKRL_DIR="${THIRDPARTY_DIR}/skrl"
SYSTEM_COMPATIBLE=true

info() {
    printf '[INFO] %s\n' "$*"
}

warn() {
    printf '[WARNING] %s\n' "$*" >&2
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

check_system() {
    info "Checking system compatibility with Isaac Gym RC4..."

    if [[ "$(uname -s)" != "Linux" ]]; then
        warn "Isaac Gym RC4 requires Linux; detected $(uname -s)."
        SYSTEM_COMPATIBLE=false
    fi

    if [[ "$(uname -m)" != "x86_64" ]]; then
        warn "Isaac Gym RC4 targets x86_64; detected $(uname -m)."
        SYSTEM_COMPATIBLE=false
    fi

    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
        if [[ "${ID:-}" != "ubuntu" ]]; then
            warn "Isaac Gym RC4 officially targets Ubuntu; detected ${PRETTY_NAME:-unknown Linux distribution}."
            SYSTEM_COMPATIBLE=false
        elif [[ "${VERSION_ID:-}" != "18.04" && "${VERSION_ID:-}" != "20.04" ]]; then
            warn "Ubuntu ${VERSION_ID:-unknown} is not an officially supported Isaac Gym RC4 release (18.04 or 20.04)."
            SYSTEM_COMPATIBLE=false
        else
            info "Operating system: ${PRETTY_NAME}."
        fi
    else
        warn "Cannot determine the Linux distribution because /etc/os-release is unavailable."
        SYSTEM_COMPATIBLE=false
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi was not found. Isaac Gym requires a supported NVIDIA GPU and driver."
        SYSTEM_COMPATIBLE=false
    elif ! nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi is installed but cannot communicate with the NVIDIA driver."
        SYSTEM_COMPATIBLE=false
    else
        local gpu_names
        gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"

        if grep -Eiq 'RTX[[:space:]]+50(80|90)([^0-9]|$)' <<< "${gpu_names}"; then
            die "RTX 5080/5090 series GPUs are incompatible with Isaac Gym RC4. Installation stopped."
        fi

        info "NVIDIA hardware and driver detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    fi
}

select_environment_manager() {
    if command -v mamba >/dev/null 2>&1; then
        ENV_MANAGER="mamba"
    elif command -v conda >/dev/null 2>&1; then
        ENV_MANAGER="conda"
        warn "mamba was not found; using conda instead."
    else
        die "Neither mamba nor conda was found. Install Miniforge, Mambaforge, or Conda and rerun this script."
    fi

    info "Environment manager: ${ENV_MANAGER} ($("${ENV_MANAGER}" --version))."
}

download_and_extract() {
    mkdir -p "${THIRDPARTY_DIR}"

    if [[ -d "${ISAACGYM_DIR}" ]]; then
        printf '👉 Isaac Gym is already extracted; skipping download and extraction: %s\n' "${ISAACGYM_DIR}"
        return
    fi

    if [[ -f "${ARCHIVE_PATH}" ]]; then
        info "Using existing archive: ${ARCHIVE_PATH}"
    else
        if ! command -v python3 >/dev/null 2>&1; then
            die "python3 is required to install and run gdown."
        fi

        if ! python3 -c 'import gdown' >/dev/null 2>&1; then
            info "Installing gdown..."
            python3 -m pip install --user gdown
        fi

        info "Downloading the modified Isaac Gym RC4 package to ${THIRDPARTY_DIR}..."
        python3 -m gdown "${GDRIVE_FILE_ID}" --output "${ARCHIVE_PATH}"
    fi

    if command -v unzip >/dev/null 2>&1 && unzip -tq "${ARCHIVE_PATH}" >/dev/null 2>&1; then
        info "Detected ZIP archive. Extracting into ${THIRDPARTY_DIR}..."
        unzip -o "${ARCHIVE_PATH}" -d "${THIRDPARTY_DIR}"
    elif command -v python3 >/dev/null 2>&1 && python3 -c \
        'import sys, zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' \
        "${ARCHIVE_PATH}"; then
        die "unzip is required to extract ${ARCHIVE_NAME}."
    elif tar -tzf "${ARCHIVE_PATH}" >/dev/null 2>&1; then
        info "Detected gzip-compressed tar archive. Extracting into ${THIRDPARTY_DIR}..."
        tar -xzf "${ARCHIVE_PATH}" -C "${THIRDPARTY_DIR}"
    else
        die "The downloaded file is not a valid ZIP or gzip-compressed tar archive: ${ARCHIVE_PATH}"
    fi
}

create_environment() {
    if "${ENV_MANAGER}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
        info "Environment ${ENV_NAME} already exists; reusing it."
    else
        info "Creating environment ${ENV_NAME} with Python ${PYTHON_VERSION}..."
        "${ENV_MANAGER}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    fi

    if environment_has_module torch; then
        printf '👉 PyTorch is already installed in %s; skipping installation.\n' "${ENV_NAME}"
    else
        info "Installing PyTorch packages in ${ENV_NAME}..."
        "${ENV_MANAGER}" run -n "${ENV_NAME}" python -m pip install torch torchvision torchaudio
    fi
}

environment_has_module() {
    local module_name="$1"

    "${ENV_MANAGER}" run -n "${ENV_NAME}" python -c \
        'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)' \
        "${module_name}" >/dev/null 2>&1
}

install_isaac_gym() {
    if environment_has_module isaacgym_tactile; then
        printf '👉 isaacgym_tactile is already installed in %s; skipping installation.\n' "${ENV_NAME}"
        return
    fi

    [[ -f "${ISAACGYM_PYTHON_DIR}/setup.py" ]] || \
        die "Isaac Gym setup.py was not found: ${ISAACGYM_PYTHON_DIR}/setup.py"

    info "Installing isaacgym_tactile in ${ENV_NAME}..."
    "${ENV_MANAGER}" run -n "${ENV_NAME}" python -m pip install -e "${ISAACGYM_PYTHON_DIR}"
}

install_standard_isaac_gym() {
    if environment_has_module isaacgym; then
        printf '👉 isaacgym is already installed in %s; skipping installation.\n' "${ENV_NAME}"
        return
    fi

    [[ -f "${STANDARD_ISAACGYM_PYTHON_DIR}/setup.py" ]] || \
        die "Standard Isaac Gym setup.py was not found: ${STANDARD_ISAACGYM_PYTHON_DIR}/setup.py"

    info "Installing standard isaacgym in ${ENV_NAME}..."
    "${ENV_MANAGER}" run -n "${ENV_NAME}" python -m pip install -e "${STANDARD_ISAACGYM_PYTHON_DIR}"
}

install_project_dependencies() {
    [[ -f "${RSL_RL_DIR}/setup.py" ]] || die "Local rsl_rl setup.py was not found: ${RSL_RL_DIR}/setup.py"
    [[ -f "${SKRL_DIR}/pyproject.toml" ]] || die "Local skrl pyproject.toml was not found: ${SKRL_DIR}/pyproject.toml"
    [[ -f "${SCRIPT_DIR}/low-level/setup.py" ]] || die "Low-level setup.py was not found."

    if environment_has_module rsl_rl; then
        printf '👉 rsl_rl is already installed in %s; skipping installation.\n' "${ENV_NAME}"
    else
        info "Installing local rsl_rl in editable mode..."
        "${ENV_MANAGER}" run -n "${ENV_NAME}" python -m pip install -e "${RSL_RL_DIR}"
    fi

    if environment_has_module skrl; then
        printf '👉 skrl is already installed in %s; skipping installation.\n' "${ENV_NAME}"
    else
        info "Installing local skrl in editable mode..."
        "${ENV_MANAGER}" run -n "${ENV_NAME}" python -m pip install -e "${SKRL_DIR}"
    fi

    if environment_has_module legged_gym; then
        printf '👉 legged_gym is already installed in %s; skipping installation.\n' "${ENV_NAME}"
    else
        info "Installing low-level in editable mode..."
        "${ENV_MANAGER}" run -n "${ENV_NAME}" python -m pip install --no-deps -e "${SCRIPT_DIR}/low-level"
    fi

    info "Installing NumPy 1.23.3 in ${ENV_NAME}..."
    "${ENV_MANAGER}" install -y -n "${ENV_NAME}" "numpy=1.23.3"

    info "Installing Python dependencies in ${ENV_NAME}..."
    "${ENV_MANAGER}" run -n "${ENV_NAME}" python -m pip install \
        pydelatin \
        tqdm \
        imageio-ffmpeg \
        opencv-python \
        wandb \
        matplotlib \
        torchinfo \
        omegaconf \
        trimesh \
        urdfpy \
        hidapi \
        rtree \
        pyzmq \
        msgpack \
        msgpack-numpy \
        "zarr<3" \
        rerun-sdk==0.19.1 \
        lightning
}

main() {
    check_system
    select_environment_manager

    if [[ "${SYSTEM_COMPATIBLE}" == true ]]; then
        printf '✅ System compatibility check passed.\n'
    else
        warn "System compatibility checks reported issues. Isaac Gym RC4 may not work correctly."
    fi

    download_and_extract
    create_environment
    install_isaac_gym
    install_standard_isaac_gym
    install_project_dependencies

    info "Installation completed. Activate the environment with: ${ENV_MANAGER} activate ${ENV_NAME}"
}

main "$@"
