# rm nsys_torch.nsys-rep nsys_torch.sqlite
# nsys profile -t nvtx,cuda --stats true \
# --capture-range nvtx \
# --nvtx-capture "Compute Section" \
# --output nsys_torch \
# python megakernels/scripts/generate.py mode=torch prompt="write me a 90 word story about Trump" ntok=10 num_warmup=0 num_iters=1

cmd="./bandwidth_test.out"
cmd="./normal_load_test.out"
echo $cmd
workload="bw_test"

# rm ${workload}.nsys-rep ${workload}.sqlite
# nsys profile -t nvtx,cuda --stats true \
# --capture-range nvtx \
# --nvtx-capture "Compute Section" \
# --output ${workload} \
# ${cmd}

rm ${workload}.ncu-rep
ncu -o ${workload} \
  --section MemoryWorkloadAnalysis_Chart \
  --section SpeedOfLight \
  --target-processes all \
${cmd}
  # --cache-control none \
  # --replay-mode application \
# --nvtx --nvtx-include "Compute Section/" \
# --section SpeedOfLight \
  # --set full \
# --section MemoryWorkloadAnalysis_Chart \

# rm ncu_torch.ncu-rep
# ncu -o ncu_torch \
#   --section SpeedOfLight \
#   --nvtx --nvtx-include "Compute Section/" \
#   --target-processes all \
#   python megakernels/scripts/generate.py mode=torch prompt="write me a 90 word story about Trump" ntok=10 num_warmup=0 num_iters=1

# rm ncu_torch_compile.ncu-rep
# ncu -o ncu_torch_compile \
#   --section SpeedOfLight \
#   --nvtx --nvtx-include "Compute Section/" \
#   --target-processes all \
#   python megakernels/scripts/generate.py mode=torch_compile prompt="write me a 90 word story about Trump" ntok=10 num_warmup=0 num_iters=1
