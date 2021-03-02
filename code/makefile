#See the params.h for parameters

#To run: ./main <dataset> <epsilon> <minpts> <data dimensionality> <partitions> <search mode> 
#Some of these are constants for the GPU densebox algorithm: ./main <dataset> <epsilon> <minpts> 2 <partitions> 3

#The output for testing will appear in gpu_stats.txt

#For example, if you run: ./main gaia_dr2_ra_dec_50M.txt 0.06 4 2 10 3
#The output (on my machine) is below, which shows that it took 20.9486 s to execute, epsilon is 0.06, minpts is 4, there are 10 partitions, 9155 clusters were generated and 0.09337% of points were assigned to noise

#Sample output in gpu_stats.txt: 20.9486, gaia_dr2_ra_dec_50M.txt, 0.06, 4, 10, 9155, 0.0009337, DENSEBOX/MU/PARCHUNKS/NUMGPU/GPUSTREAMS/PARTITIONMODE/SCHEDULE/NTRIALS/DTYPE(float/double): 2, 0.25, 10, 1, 3, 1, 0, 1, float


#You can leave the rest of the parameters as is 
#The dynamic densebox algorithm is enabled with mu=0.25, which obtained good performance  

#I would use the following for testing:
#./main gaia_dr2_ra_dec_50M.txt 0.01 4 2 10 3
#./main gaia_dr2_ra_dec_50M.txt 0.06 4 2 10 3

#This give a range of noise point percentages: 36%-0.09337% corresponding to epsilon=0.01 and 0.06, respectively.




SOURCES = import_dataset.cpp main.cu GPU.cu kernel.cu cluster.cpp DBScan.cpp tree_functions.cpp globalsDBSCAN.cpp par_sort.cpp
OBJECTS = import_dataset.o cluster.o DBScan.o tree_functions.o par_sort.o globalsDBSCAN.o
CUDAOBJECTS = GPU.o kernel.o main.o
CC = nvcc
EXECUTABLE = main


FLAGS = -ccbin g++ -O3 -std=c++11 -Xcompiler -fopenmp -arch=compute_60 -code=sm_60 -lcuda -lineinfo 
CFLAGS = -c 



all: $(EXECUTABLE)



.cpp.o:
	$(CC) $(CFLAGS) $(FLAGS) $<



main.o: main.cu params.h
	$(CC) $(CFLAGS) $(FLAGS) main.cu 

kernel.o: kernel.cu params.h
	$(CC) $(CFLAGS) $(FLAGS) kernel.cu 		

GPU.o: GPU.cu params.h
	$(CC) $(CFLAGS) $(FLAGS) GPU.cu 	


cluster.o: cluster.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) cluster.cpp 

import_dataset.o: import_dataset.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) import_dataset.cpp 

DBScan.o: DBScan.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) DBScan.cpp 			

tree_functions.o: tree_functions.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) tree_functions.cpp 

globalsDBSCAN.o: globalsDBSCAN.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) globalsDBSCAN.cpp 	

par_sort.o: par_sort.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) par_sort.cpp 		



$(EXECUTABLE): $(OBJECTS) $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(OBJECTS)
	rm $(CUDAOBJECTS)
	rm main



