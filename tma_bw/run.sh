#!/bin/bash

echo "Building bandwidth test..."
make

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "Running bandwidth comparison test..."
./bandwidth_test.out > bw_result.log
python process.py -i bw_result.log -o bw_result.png --peak-bw 5.6

echo ""
echo "Test completed!"