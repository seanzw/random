#!/bin/bash

echo "Building bandwidth test..."
make

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Setting GPU to 2GHz for consistent benchmarking..."
sudo nvidia-smi -i 1 --lock-gpu-clocks=2011

echo ""
echo "Running bandwidth comparison test..."
./bandwidth_test.out

echo "Resetting GPU clocks to default..."
sudo nvidia-smi -i 1 --reset-gpu-clocks

echo ""
echo "Test completed!"