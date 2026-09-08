#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRDPARTY_DIR="${REPO_ROOT}/thirdparty"
LCM_DIR="${THIRDPARTY_DIR}/lcm"
LCM_BUILD_DIR="${LCM_DIR}/build"
UNITREE_SDK2_PYTHON_DIR="${THIRDPARTY_DIR}/unitree_sdk2_python"
DEPLOY_BUILD_DIR="${SCRIPT_DIR}/build"
UNITREE_THIRDPARTY_LIB="${SCRIPT_DIR}/unitree_sdk2_bin/library/unitree_sdk2/thirdparty/lib/x86_64"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(command -v python)}"
PYTHON_INCLUDE_DIR="$("${PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['include'])")"

LCM_REPO_URL="${LCM_REPO_URL:-https://github.com/lcm-proj/lcm.git}"
UNITREE_SDK2_PYTHON_REPO_URL="${UNITREE_SDK2_PYTHON_REPO_URL:-https://github.com/unitreerobotics/unitree_sdk2_python.git}"
LCM_CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}"
    -DPython3_EXECUTABLE="${PYTHON_EXECUTABLE}"
    -DPython3_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}"
    -DPython3_INCLUDE_DIRS="${PYTHON_INCLUDE_DIR}"
    -DCMAKE_C_FLAGS="-I${PYTHON_INCLUDE_DIR}"
    -DLCM_ENABLE_JAVA=OFF
    -DLCM_ENABLE_LUA=OFF
    -DLCM_ENABLE_PYTHON=ON
    -DLCM_ENABLE_TESTS=OFF
    -DLCM_ENABLE_EXAMPLES=OFF
)

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

need_cmd git
need_cmd cmake
need_cmd make
need_cmd python

mkdir -p "${THIRDPARTY_DIR}"

if [ ! -d "${LCM_DIR}/.git" ]; then
    if [ -e "${LCM_DIR}" ]; then
        echo "Found ${LCM_DIR}, but it is not a git checkout." >&2
        echo "Move it away or set LCM_REPO_URL to an existing lcm git repository." >&2
        exit 1
    fi

    echo "Cloning LCM into ${LCM_DIR}"
    git clone --depth 1 "${LCM_REPO_URL}" "${LCM_DIR}"
else
    echo "Using existing LCM checkout: ${LCM_DIR}"
fi

if [ ! -d "${UNITREE_SDK2_PYTHON_DIR}/.git" ]; then
    if [ -e "${UNITREE_SDK2_PYTHON_DIR}" ]; then
        echo "Found ${UNITREE_SDK2_PYTHON_DIR}, but it is not a git checkout." >&2
        echo "Move it away or set UNITREE_SDK2_PYTHON_REPO_URL to an existing unitree_sdk2_python git repository." >&2
        exit 1
    fi

    echo "Cloning unitree_sdk2_python into ${UNITREE_SDK2_PYTHON_DIR}"
    git clone --depth 1 "${UNITREE_SDK2_PYTHON_REPO_URL}" "${UNITREE_SDK2_PYTHON_DIR}"
else
    echo "Using existing unitree_sdk2_python checkout: ${UNITREE_SDK2_PYTHON_DIR}"
fi

mkdir -p "${LCM_BUILD_DIR}"
cmake -S "${LCM_DIR}" -B "${LCM_BUILD_DIR}" "${LCM_CMAKE_ARGS[@]}"
cmake --build "${LCM_BUILD_DIR}" --parallel

PYTHON_SITE_PACKAGES="$("${PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")"
LCM_PYTHON_BUILD_DIR="${LCM_BUILD_DIR}/python/lcm"

if [ ! -f "${LCM_PYTHON_BUILD_DIR}/__init__.py" ] || [ ! -f "${LCM_PYTHON_BUILD_DIR}/_lcm.so" ]; then
    echo "LCM Python module was not built under ${LCM_PYTHON_BUILD_DIR}" >&2
    exit 1
fi

mkdir -p "${PYTHON_SITE_PACKAGES}/lcm"
cp -a "${LCM_PYTHON_BUILD_DIR}/." "${PYTHON_SITE_PACKAGES}/lcm/"

# The Unitree DDS libraries have SONAMEs ending in .so.0. Keep local symlinks
# next to the checked-in libraries so deploy binaries resolve them from rpath.
if [ -f "${UNITREE_THIRDPARTY_LIB}/libddsc.so" ] && [ ! -e "${UNITREE_THIRDPARTY_LIB}/libddsc.so.0" ]; then
    ln -s libddsc.so "${UNITREE_THIRDPARTY_LIB}/libddsc.so.0"
fi

if [ -f "${UNITREE_THIRDPARTY_LIB}/libddscxx.so" ] && [ ! -e "${UNITREE_THIRDPARTY_LIB}/libddscxx.so.0" ]; then
    ln -s libddscxx.so "${UNITREE_THIRDPARTY_LIB}/libddscxx.so.0"
fi

cmake -S "${SCRIPT_DIR}" -B "${DEPLOY_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${DEPLOY_BUILD_DIR}" --parallel

"${PYTHON_EXECUTABLE}" -m pip install -e "${SCRIPT_DIR}"
"${PYTHON_EXECUTABLE}" -m pip install -e "${UNITREE_SDK2_PYTHON_DIR}"

echo
echo "Done. Deploy binaries are in:"
echo "  ${DEPLOY_BUILD_DIR}"
echo "LCM Python package installed to:"
echo "  ${PYTHON_SITE_PACKAGES}/lcm"
echo "Python package installed in editable mode:"
echo "  ${SCRIPT_DIR}"
echo "Unitree SDK2 Python package installed in editable mode:"
echo "  ${UNITREE_SDK2_PYTHON_DIR}"
