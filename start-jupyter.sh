#!/bin/bash
set -e

# Install geoips-aviris-ng if not already installed
if ! pip list | grep -q geoips-aviris-ng; then
    echo "Installing geoips-aviris-ng..."
    pip install --user -e "git+https://github.com/biosafetylvl5/geoips-aviris-ng.git#egg=geoips-aviris-ng[all]"
fi

# Start JupyterLab
exec jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --LabApp.token='Q4xfVZAmcR42H6ofrtRDEZePVQ3B6REgr6obsUas' \
    --LabApp.password='' \
    --ServerApp.allow_origin='*' \
    --ServerApp.disable_check_xsrf=True
