CUDA_HOME   = /Soft/cuda/8.0.61

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DSIZE=32

PJ1    = proj1.exe

OBJ1    = proj1.o

default: $(PJ1) $(PJ0)

proj1.o: BYNFila.cu
	$(NVCC) -c -o $@ BYNFila.cu $(NVCC_FLAGS) $(PROG_FLAGS)

$(PJ1): $(OBJ1)
	$(NVCC) $(OBJ1) -o $(PJ1) $(LD_FLAGS)


all:	$(PJ1)	$(PJ0)

clean:
	rm -rf *.o* *.e*

