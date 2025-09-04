#!/bin/bash

# Create necessary directories
mkdir -p /home/jupyter/.local/bin
mkdir -p /home/jupyter/src

# Install geoips-aviris-ng if not already installed
git clone https://github.com/biosafetylvl5/geoips-aviris-ng.git /home/jupyter/src 
pip install -e /home/jupyter/src/[all]

# If no command is provided, keep container running
if [ $# -eq 0 ]; then
    echo "Running Jupyter with default token"
    cd /home/jupyter/src
    jupyter lab     --ip=0.0.0.0     --port=8888     --no-browser     --ServerApp.token='Q4xfVZAmcR42H6ofrtRDEZePVQ3B6REgr6obsUas'     --ServerApp.password=''     --ServerApp.allow_origin='*'     --ServerApp.disable_check_xsrf=True     --ServerApp.base_url='/jupyter/'     --ServerApp.allow_remote_access=True     --ServerApp.trust_xheaders=True
else
    exec "$@"
fi
