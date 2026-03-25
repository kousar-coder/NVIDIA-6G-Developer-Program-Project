#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# SenseForge — Build Script
# ═══════════════════════════════════════════════════════════════
# Checks SDK availability, installs deps, trains fusion model.
# ═══════════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  SenseForge — Build${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo ""

# ── Check pyAerial ──────────────────────────────────────────────────────
echo -e "${BOLD}[1/4] Checking pyAerial...${RESET}"
if python -c "import pyaerial; print(f'  pyAerial {getattr(pyaerial, \"__version__\", \"found\")}')" 2>/dev/null; then
    echo -e "  ${GREEN}✓${RESET} pyAerial available"
else
    echo -e "  ${RED}✗${RESET} pyAerial NOT found"
    echo ""
    echo "  pyAerial must be installed from the NVIDIA Aerial SDK."
    echo "  Run: bash aerial_setup.sh"
    echo ""
    exit 1
fi

# ── Check Sionna ────────────────────────────────────────────────────────
echo -e "${BOLD}[2/4] Checking Sionna...${RESET}"
if python -c "import sionna; print(f'  Sionna {sionna.__version__}')" 2>/dev/null; then
    echo -e "  ${GREEN}✓${RESET} Sionna available"
else
    echo -e "  ${RED}✗${RESET} Sionna NOT found"
    echo ""
    echo "  Install: pip install sionna>=0.18.0"
    echo ""
    exit 1
fi

# ── Install remaining deps ──────────────────────────────────────────────
echo -e "${BOLD}[3/4] Installing dependencies...${RESET}"
pip install -r requirements.txt

# ── Train fusion model ──────────────────────────────────────────────────
echo -e "${BOLD}[4/4] Training fusion model...${RESET}"
python -m fusion.train

echo ""
echo -e "${GREEN}${BOLD}Build complete ✓${RESET}"
echo ""
