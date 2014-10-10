nvcc -g -G -I/usr/local/cuda/include -Xcompiler "-I/Library/Frameworks/R.framework/Resources/include -fpic" -c matmult.cu -o matmult.o -arch=sm_30
nvcc -shared -Xlinker "-L/Library/Frameworks/R.framework/Resources/lib -lR" -L/usr/local/cuda/lib -lcublas matmult.o -o matmult.so
