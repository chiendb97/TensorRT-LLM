#!/bin/bash

set -ex

install_boost() {
  # Install boost version >= 1.78 for boost::span
  # Current libboost-dev apt packages are < 1.78, so install from tar.gz
  wget -O /tmp/boost.tar.gz --timeout=180 --tries=3 https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz \
    && tar xzf /tmp/boost.tar.gz -C /tmp \
    && cd /tmp/boost_1_86_0 && ./bootstrap.sh && ./b2 --with=all -j16 install \
    && rm -rf /tmp/boost_1_86_0 /tmp/boost.tar.gz
}

install_triton_deps() {
  apt-get update \
    && apt-get install -y --no-install-recommends \
      pigz \
      libxml2-dev \
      libre2-dev \
      libnuma-dev \
      python3-build \
      libb64-dev \
      libarchive-dev \
      datacenter-gpu-manager=1:3.3.6 \
    && install_boost \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
}

# Install Triton only if base image is Ubuntu
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [ "$ID" == "ubuntu" ]; then
  install_triton_deps
else
  rm -rf /opt/tritonserver
  echo "Skip Triton installation for non-Ubuntu base image"
fi
