FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Install dependencies for building GDAL
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    libsqlite3-dev \
    libproj-dev \
    libgeos-dev \
    libexpat1-dev \
    zlib1g-dev \
    liblzma-dev \
    libcurl4-gnutls-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Download and install GDAL 3.6.4 (a more recent stable version)
WORKDIR /tmp
RUN wget https://github.com/OSGeo/gdal/releases/download/v3.6.4/gdal-3.6.4.tar.gz \
    && tar -xzf gdal-3.6.4.tar.gz \
    && cd gdal-3.6.4 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /tmp \
    && rm -rf gdal-3.6.4 gdal-3.6.4.tar.gz

# Install Python GDAL bindings
RUN pip install GDAL==3.6.4

RUN echo "Installing GeoIPS" \
    && git clone https://github.com/NRLMMD-GEOIPS/geoips.git /workspaces/geoips \
    && mkdir /workspaces/testdata