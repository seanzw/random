GPU_ID=1
CLOCK=2010
cmd="./bandwidth_test.out"

# Check for l2/dram argument
if [[ "$1" != "l2" && "$1" != "dram" ]]; then
    echo "Usage: source profile.sh [l2|dram]"
    return 1
fi

MODE=$1
workload="bw_test_${MODE}"

echo "Running in ${MODE} mode..."
echo "Command: ${cmd}"
echo "Output report: ${workload}.ncu-rep"

# Lock GPU clock
sudo nvidia-smi -i ${GPU_ID} -lgc $((CLOCK + 1))

if [[ "${MODE}" == "l2" ]]; then
    # L2 benchmark command
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ncu -f -o ${workload} \
      --section MemoryWorkloadAnalysis_Chart \
      --section SpeedOfLight \
      --target-processes all \
      --clock-control none \
      --cache-control none \
      --replay-mode application \
    ${cmd}
elif [[ "${MODE}" == "dram" ]]; then
    # DRAM benchmark command 
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ncu -o ${workload} \
      --section MemoryWorkloadAnalysis_Chart \
      --section SpeedOfLight \
      --target-processes all \
      --clock-control none \
    ${cmd}
fi

# Reset GPU clock
sudo nvidia-smi -i ${GPU_ID} -rgc

echo "Profiling finished. Report saved to ${workload}.ncu-rep"