#See the params.h for parameters

#To run: ./main <dataset> <epsilon> <minpts> <data dimensionality> <partitions>
#Some of these are constants for the GPU densebox algorithm: ./main <dataset> <epsilon> <minpts> 2 <partitions>

#The output for testing will appear in gpu_stats.txt

#For example, if you run: ./main gaia_dr2_ra_dec_50M.txt 0.06 4 2 10
#The output (on my machine) is below, which shows that it took 20.9486 s to execute, epsilon is 0.06, minpts is 4, there are 10 partitions, 9155 clusters were generated and 0.09337% of points were assigned to noise

#Sample output in gpu_stats.txt: 20.9486, gaia_dr2_ra_dec_50M.txt, 0.06, 4, 10, 9155, 0.0009337, DENSEBOX/MU/PARCHUNKS/NUMGPU/GPUSTREAMS/PARTITIONMODE/SCHEDULE/NTRIALS/DTYPE(float/double): 2, 0.25, 10, 1, 3, 1, 0, 1, float


#You can leave the rest of the parameters as is 
#The dynamic densebox algorithm is enabled with mu=0.25, which obtained good performance  

#I recommend use the following for testing:
#./main gaia_dr2_ra_dec_50M.txt 0.01 4 2 10
#./main gaia_dr2_ra_dec_50M.txt 0.06 4 2 10

#This give a range of noise point percentages: 36%-0.09337% corresponding to epsilon=0.01 and 0.06, respectively.



#Change to your CUDA Compute Capability:
# COMPUTE_CAPABILITY = 80
# COMPUTE_CAPABILITY = 90
COMPUTE_CAPABILITY = 120
COMPUTE_CAPABILITY_FLAGS = -arch=compute_$(COMPUTE_CAPABILITY) -code=sm_$(COMPUTE_CAPABILITY)


SOURCES = import_dataset.cu main.cu GPU.cu kernel.cu cluster.cu DBScan.cu tree_functions.cu globalsDBSCAN.cu par_sort.cu
OBJECTS = import_dataset.o cluster.o DBScan.o tree_functions.o par_sort.o globalsDBSCAN.o
CUDAOBJECTS = GPU.o kernel.o main.o
CC = nvcc
EXECUTABLE = main


FLAGS = -O3 -std=c++20 -Xcompiler -fopenmp -lcuda -lineinfo 
CFLAGS = -c 



all: $(EXECUTABLE)



.cpp.o:
	$(CC) $(CFLAGS) $(FLAGS) $<


#CUDA/GPU source files:
main.o: main.cu params.h structs.h
	$(CC) $(CFLAGS) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) main.cu 

kernel.o: kernel.cu params.h structs.h
	$(CC) $(CFLAGS) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) kernel.cu 		

GPU.o: GPU.cu params.h structs.h
	$(CC) $(CFLAGS) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) GPU.cu 	


#C/C++ source files:
cluster.o: cluster.cu params.h structs.h
	$(CC) $(CFLAGS) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) cluster.cu 

par_sort.o: par_sort.cu params.h structs.h
	$(CC) $(CFLAGS) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) par_sort.cu

import_dataset.o: import_dataset.cu params.h structs.h
	$(CC) $(CFLAGS) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) import_dataset.cu 

DBScan.o: DBScan.cu params.h structs.h
	$(CC) $(CFLAGS) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) DBScan.cu 			

tree_functions.o: tree_functions.cu params.h structs.h
	$(CC) $(CFLAGS) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) tree_functions.cu 

globalsDBSCAN.o: globalsDBSCAN.cu params.h structs.h
	$(CC) $(CFLAGS) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) globalsDBSCAN.cu 	






$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS) 
	$(CC) $(FLAGS) $(COMPUTE_CAPABILITY_FLAGS) $^ -o $@




clean:
	rm $(OBJECTS)
	rm $(CUDAOBJECTS)
	rm main



