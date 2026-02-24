#!/bin/bash
# NeuralAtoms — System Core Dependencies
# For use on Linux/Colab runtimes.

set -e

echo "Installing Linux System Dependencies for Vision and Physics..."

# Industrial-grade GL and display libraries for MuJoCo and OpenCV
apt-get update
apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    ffmpeg

echo "System Dependencies Primed ✓"
