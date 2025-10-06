#!/bin/bash

echo "Building bandwidth test..."
make

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Setting GPU to 2GHz for consistent benchmarking..."
sudo nvidia-smi -pm 1
sudo nvidia-smi --lock-gpu-clocks=2000,2000

echo ""
echo "Running bandwidth comparison test..."
./bandwidth_test.out | tee bw_result.log

echo "Resetting GPU clocks to default..."
sudo nvidia-smi --reset-gpu-clocks
sudo nvidia-smi -pm 0

python process.py -i bw_result.log -o bw_result.png --peak-bw 5.6

echo ""
echo "Test completed!"