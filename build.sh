# s114:
# NVIDIA Tesla V100-SXM2
# make -j src.build CXXFLAGS="-g -O0" DEBUG=1 CUDA_HOME=/usr/lib/cuda NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
make -j src.build CXXFLAGS="-I/root/project/ai/nccl/src/include -g1" DEBUG=1 CUDA_HOME=/usr/lib/cuda NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"

# export LD_LIBRARY=/root/project/ai/nccl/build/lib