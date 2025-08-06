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
    echo "Container is ready. Keeping it alive..."
    exec sleep infinity
else
    exec "$@"
fi
