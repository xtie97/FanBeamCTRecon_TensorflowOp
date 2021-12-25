CUDA_ROOT = /usr/local/cuda-10.0
CUDA_SDK = /usr/local/cuda-10.0/samples
TF_INC = /home/xintie/Python_env/FPTF/lib/python3.7/site-packages/tensorflow/include
TF_LIB = /home/xintie/Python_env/FPTF/lib/python3.7/site-packages/tensorflow
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
IPP_INC = /home/xintie/intel/ipp/include64
IPP_LIB = /home/xintie/intel/ipp/lib64

NVCC := $(CUDA_ROOT)/bin/nvcc
CXX := g++

ERASE := rm -rf

gpu:
	$(NVCC) -std=c++11 -c -o transform_cu.o transform.cu -L$(CUDA_ROOT)/lib64 -I$(CUDA_ROOT)/include -L$(CUDA_SDK)/lib -I$(CUDA_SDK)/common/inc -L$(IPP_LIB) -I$(IPP_INC) -I$(TF_INC) -L$(TF_LIB) -ltensorflow_framework ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_61 -lippsmergedem64t -lippcoreem64t -liomp5
	$(CXX) -std=c++11 -shared -o FBP.so transform.cc transform.h transform_cu.o ${TF_CFLAGS[@]} -fPIC -lcuda -lcudart -lcublas -lippsmergedem64t -lippcoreem64t -pthread -liomp5 ${TF_LFLAGS[@]} -L$(CUDA_ROOT)/lib64 -I$(CUDA_ROOT)/include -L$(CUDA_SDK)/lib -I$(CUDA_SDK)/common/inc -L$(IPP_LIB) -I$(IPP_INC) -I$(TF_INC) -L$(TF_LIB) -ltensorflow_framework 

clean:
	$(ERASE) *.so *.o Debug/

#-ltensorflow_framework
#-D_GLIBCXX_USE_CXX11_ABI=0
