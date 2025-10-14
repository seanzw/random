GPGPU_SIM_ROOT="/home/syu/Workspace/arch/gpgpu-sim_distribution"
CONFIG_DIR="${GPGPU_SIM_ROOT}/configs/SM120_RTX5090"

# Source environment
source "${GPGPU_SIM_ROOT}/setup_environment"

# Copy GPGPU-Sim config files, overwriting if they exist
# echo "Copying config files from ${CONFIG_DIR}..."
cp -f "${CONFIG_DIR}/gpgpusim.config" .
cp -f "${CONFIG_DIR}/config_ampere_islip.icnt" .

# Build the project from the parent directory
echo "Building the project..."
make

# Run the benchmark with GPGPU-Sim
echo "Running the benchmark..."
./bandwidth_test.out > bw_test.log