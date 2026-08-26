V=/home/tusshar_agarwaal/turboquant-llm/.venv/lib/python3.12/site-packages
$V/nvidia/cu13/bin/nvcc --version | tail -2
grep -h "define CUDA_VERSION" $V/nvidia/cu13/include/cuda.h
ls -d $V/nvidia/cu1*
