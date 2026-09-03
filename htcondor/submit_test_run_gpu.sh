#!/bin/bash
# Convenience wrapper to submit the simulator_gpu.py smoke test (5 samples).
# Run on the cluster's login node, from anywhere:
#   ~/cluster/../lisa_modulation_test/htcondor/submit_test_run_gpu.sh
# (or `cd` into the project first and run ./htcondor/submit_test_run_gpu.sh)
set -euo pipefail

cd "$(dirname "$0")/.."   # project root, matches initialdir in test_run_gpu.sub
mkdir -p outputs          # HTCondor requires this to exist before submission
condor_submit_bid 50 htcondor/test_run_gpu.sub
