#!/bin/bash

# --- Styling Constants ---
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Header ---
clear
echo -e "${BLUE}${BOLD}"
echo "  _______ _    _ _       "
echo " |__   __| |  | | |      "
echo "    | |  | |__| | |      "
echo "    | |  |  __  | |      "
echo "    | |  | |  | | |____  "
echo "    |_|  |_|  |_|______| "
echo -e "${NC}"
echo -e "${CYAN}${BOLD}Running THL Test Suite...${NC}\n"

# Export project root to pythonpath
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- Helper Function ---
run_test_category() {
    category_name=$1
    test_path=$2
    
    echo -e "${BOLD}▶ Running ${category_name} tests...${NC}"
    
    # Run pytest with custom formatting
    # -q: quiet
    # --tb=short: shorter tracebacks
    if [ -f ".venv/bin/python3" ]; then
        PYTHON_CMD=".venv/bin/python3"
    else
        PYTHON_CMD="python3"
    fi
    
    $PYTHON_CMD -m pytest $test_path -q --tb=short
    
    status=$?
    
    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✔ ${category_name} tests passed!${NC}\n"
        return 0
    else
        echo -e "${RED}✘ ${category_name} tests failed!${NC}\n"
        return 1
    fi
}

# --- Execution ---

start_time=$(date +%s)
failures=0

# Clean pycache
find . -type d -name "__pycache__" -exec rm -rf {} + > /dev/null 2>&1

# Unit Tests
run_test_category "Unit: Memory" "tests/memory/" || ((failures++))
run_test_category "Unit: Tiers" "tests/tiers/" || ((failures++))
run_test_category "Unit: Training" "tests/training/" || ((failures++))
run_test_category "Unit: Utils" "tests/utils/" || ((failures++))

# Inference & Integration
run_test_category "Inference" "tests/inference/" || ((failures++))
run_test_category "Integration" "tests/integration/" || ((failures++))

end_time=$(date +%s)
duration=$((end_time - start_time))

# --- Summary ---
echo -e "----------------------------------------"
if [ $failures -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All tests passed in ${duration}s! 🚀${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}${failures} test categories failed. See above for details.${NC}"
    exit 1
fi
