#!/usr/bin/env bash

# ==========================================
# NLM-1 Launcher Script
# ==========================================

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
DIM='\033[2m'
NC='\033[0m' # No Color

# 1. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed or not in PATH.${NC}"
    exit 1
fi

# 2. Check for NumPy
if ! python3 -c "import numpy" &> /dev/null; then
    echo -e "${YELLOW}Missing dependency: numpy${NC}"
    echo -e "Please install it by running: ${GREEN}pip install numpy${NC}"
    exit 1
fi

# 3. Check for required files
if [ ! -f "model.json" ]; then
    echo -e "${RED}Error: model.json not found in the current directory.${NC}"
    exit 1
fi

if [ ! -f "__main__.py" ]; then
    echo -e "${RED}Error: nlm.py not found in the current directory.${NC}"
    exit 1
fi

# 4. Help menu
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./app.sh [prompt] [length] [temperature] [top_p]"
    echo ""
    echo -e "${YELLOW}Arguments:${NC}"
    echo "  prompt       Seed text (default: \"the sun rose\")"
    echo "  length       Number of characters to generate (default: 300)"
    echo "  temperature  Randomness level (default: 0.8)"
    echo "  top_p        Nucleus sampling threshold (default: 0.92)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./app.sh"
    echo "  ./app.sh \"she walked down\""
    echo "  ./app.sh \"the station was\" 500 0.6 0.9"
    exit 0
fi

# 5. Launch
echo -e "${DIM}Starting NLM-1...${NC}"
python3 __main__.py "$@"