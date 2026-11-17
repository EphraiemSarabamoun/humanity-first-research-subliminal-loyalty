#!/usr/bin/env bash
# Edited for release.
# The leg's environment, run once on the rented GPU: a CLEAN venv from the pinned lock, then the preflight. The recipe:
# never --system-site-packages, never user-site installs; vLLM 0.16 with its torch 2.9.1.
set -euo pipefail
cd ~/work
python3 -m venv rvenv
. rvenv/bin/activate
export PYTHONNOUSERSITE=1
pip -q install --upgrade pip
pip -q install -r leg/requirements.txt
python leg/preflight.py "$@"           # "person": also the second study's inputs
