#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# SenseForge — NVIDIA AI Aerial SDK Setup Script
# ═══════════════════════════════════════════════════════════════
# This script:
#   1. Checks prerequisites (docker, git, git-lfs, nvidia-smi)
#   2. Confirms NVIDIA Docker runtime works
#   3. Clones aerial-cuda-accelerated-ran (with submodules)
#   4. Prompts NGC login instructions
#   5. Pulls the Aerial container image
#   6. Runs container with project + pyaerial mounted
#   7. Installs deps, trains fusion model, starts backend
# ═══════════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

AERIAL_REPO="https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git"
AERIAL_IMAGE="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb"
AERIAL_DIR="${HOME}/aerial-cuda-accelerated-ran"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${CYAN}  SenseForge — NVIDIA AI Aerial SDK Setup${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
echo ""

# ── 1. Check prerequisites ───────────────────────────────────────────────

echo -e "${BOLD}[1/6] Checking prerequisites...${RESET}"

check_cmd() {
    if command -v "$1" &> /dev/null; then
        echo -e "  ${GREEN}✓${RESET} $1 found: $(command -v "$1")"
        return 0
    else
        echo -e "  ${RED}✗${RESET} $1 NOT found"
        return 1
    fi
}

PREREQS_OK=true
check_cmd docker     || PREREQS_OK=false
check_cmd git        || PREREQS_OK=false
check_cmd git-lfs    || PREREQS_OK=false
check_cmd nvidia-smi || PREREQS_OK=false

if [ "$PREREQS_OK" = false ]; then
    echo ""
    echo -e "${RED}ERROR: Missing prerequisites. Install the above tools first.${RESET}"
    echo "  - Docker: https://docs.docker.com/engine/install/"
    echo "  - Git LFS: https://git-lfs.com/"
    echo "  - NVIDIA Drivers: https://www.nvidia.com/drivers"
    exit 1
fi

# ── 2. Verify NVIDIA Docker runtime ─────────────────────────────────────

echo ""
echo -e "${BOLD}[2/6] Verifying NVIDIA Docker runtime...${RESET}"

if docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "  ${GREEN}✓${RESET} NVIDIA Docker runtime works"
else
    echo -e "  ${RED}✗${RESET} NVIDIA Docker runtime failed"
    echo ""
    echo "  Install nvidia-container-toolkit:"
    echo "    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    echo "  Then restart Docker:"
    echo "    sudo systemctl restart docker"
    exit 1
fi

# ── 3. Clone Aerial SDK ─────────────────────────────────────────────────

echo ""
echo -e "${BOLD}[3/6] Cloning NVIDIA Aerial SDK...${RESET}"

if [ -d "$AERIAL_DIR" ]; then
    echo -e "  ${YELLOW}○${RESET} $AERIAL_DIR already exists. Pulling latest..."
    cd "$AERIAL_DIR" && git pull && git submodule update --init --recursive
else
    echo "  Cloning to $AERIAL_DIR..."
    git clone --recurse-submodules "$AERIAL_REPO" "$AERIAL_DIR"
fi
echo -e "  ${GREEN}✓${RESET} Aerial SDK ready"

# ── 4. NGC Login ─────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}[4/6] NGC Container Registry Login${RESET}"
echo ""
echo -e "  ${YELLOW}If you haven't logged into NGC yet:${RESET}"
echo ""
echo "  1. Create an NGC account at: https://ngc.nvidia.com/"
echo "  2. Generate an API key at: https://ngc.nvidia.com/setup/api-key"
echo "  3. Run: docker login nvcr.io"
echo "     Username: \$oauthtoken"
echo "     Password: <your-api-key>"
echo ""
read -p "  Press ENTER when ready (or Ctrl+C to cancel)..."

# ── 5. Pull Aerial container image ───────────────────────────────────────

echo ""
echo -e "${BOLD}[5/6] Pulling Aerial container image...${RESET}"
echo "  Image: $AERIAL_IMAGE"
docker pull "$AERIAL_IMAGE"
echo -e "  ${GREEN}✓${RESET} Image pulled successfully"

# ── 6. Run container ────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}[6/6] Starting Aerial container...${RESET}"
echo ""
echo "  Mounting:"
echo "    - Project: $PROJECT_DIR → /workspace/SenseForge"
echo "    - pyAerial: $AERIAL_DIR/pyaerial → /workspace/pyaerial"
echo ""

CONTAINER_NAME="senseforge-aerial"

# Stop existing container if running
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --network host \
    -v "$PROJECT_DIR:/workspace/SenseForge" \
    -v "$AERIAL_DIR/pyaerial:/workspace/pyaerial" \
    -w /workspace/SenseForge \
    "$AERIAL_IMAGE" \
    bash -c "
        echo '═══ Installing pyAerial ═══'
        pip install -e /workspace/pyaerial/ 2>/dev/null || true
        echo '═══ Installing SenseForge dependencies ═══'
        pip install -r requirements.txt
        echo '═══ Training fusion model ═══'
        python -m fusion.train
        echo '═══ Running 3GPP validation ═══'
        python aerial_validate.py
        echo '═══ Starting backend ═══'
        uvicorn backend.main:app --host 0.0.0.0 --port 8000
    "

echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  Container '$CONTAINER_NAME' is starting.${RESET}"
echo -e "${GREEN}  Backend will be available at http://localhost:8000${RESET}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
echo ""
echo "  Useful commands:"
echo "    docker logs -f $CONTAINER_NAME     # View logs"
echo "    docker exec -it $CONTAINER_NAME bash  # Shell into container"
echo "    docker stop $CONTAINER_NAME        # Stop"
echo ""
