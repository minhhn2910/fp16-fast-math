rm mathfunc
nvcc -arch=sm_70 mathfloat.cu -use_fast_math -o mathfunc
