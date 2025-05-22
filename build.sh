s114:
NVIDIA Tesla V100-SXM2
make -j src.build CUDA_HOME=/usr/lib/cuda NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"

