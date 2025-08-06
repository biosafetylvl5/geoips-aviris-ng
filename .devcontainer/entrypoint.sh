#!/bin/bash

# Create necessary directories
mkdir -p /home/jupyter/.local/bin
mkdir -p /home/jupyter/src

# Install geoips-aviris-ng if not already installed
if ! pip list | grep -q geoips-aviris-ng; then
    echo "Installing geoips-aviris-ng..."
        if ! pip install --user --src /home/jupyter/src -e "git+https://github.com/biosafetylvl5/geoips-aviris-ng.git#egg=geoips-aviris-ng[all]"; then
        echo "Installation failed, continuing without geoips-aviris-ng..."
    fi
fi

# If no command is provided, keep container running
if [ $# -eq 0 ]; then
    echo "Running Jupyter with default token"
    cd /home/jupyter/src
    jupyter lab     --ip=0.0.0.0     --port=8888     --no-browser     --ServerApp.token='Q4xfVZAmcR42H6ofrtRDEZePVQ3B6REgr6obsUas'     --ServerApp.password=''     --ServerApp.allow_origin='*'     --ServerApp.disable_check_xsrf=True     --ServerApp.base_url='/jupyter/'     --ServerApp.allow_remote_access=True     --ServerApp.trust_xheaders=True
else
    exec "$@"
fi
