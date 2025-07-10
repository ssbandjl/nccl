# s114:
# NVIDIA Tesla V100-SXM2
# make -j src.build  DEBUG=1 CUDA_HOME=/usr/lib/cuda NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"

# v100
# make -j src.build \
#   BUILDDIR=/root/project/ai/nccl-tests/nccl/build \
#   CXXFLAGS="-I/root/project/ai/nccl-tests/nccl/src/include -g -O0 -fPIC" \
#   DEBUG=1 \
#   ENABLE_TRACE=1 \
#   CUDA_HOME=/usr/lib/cuda \
#   NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"

# a6000
make -j src.build \
  BUILDDIR=/root/project/ai/nccl-tests/nccl/build \
  CXXFLAGS="-I/root/project/ai/nccl-tests/nccl/src/include -g -O0 -fPIC" \
  DEBUG=1 \
  ENABLE_TRACE=1 \
  CUDA_HOME=/usr/lib/cuda \
  NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86"

# export LD_LIBRARY=/root/project/ai/nccl-tests/nccl/build/lib


# make -j src.build \
#   DEBUG=1 \
#   CXXFLAGS="-g -O0 -fPIC" \
#   NVCCFLAGS="-g -Xcompiler -fPIC" \
#   NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
