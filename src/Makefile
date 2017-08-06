NVCC=nvcc -arch=sm_20

all:
	$(NVCC) main.cu wrappers.cu -lm -o ../bin/Backpropagation

clean:
	rm -f *~ *.o *.bin ../bin/Backpropagation
