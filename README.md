# BPS-HDBSCAN
Hybrid Parallel CPU+GPU DBSCAN Algorithm

This algorithm is described in the following paper: Michael Gowanlock (2019) "Hybrid CPU/GPU Clustering in Shared Memory on the Billion Point Scale" *In Proceedings of the International Conference on Supercomputing 2019 (ICS 2019)*, June 26–28, 2019, Phoenix, AZ, USA.

The preprint of the paper can be found at the following address: https://jan.ucc.nau.edu/mg2745/discoframe/

# Acknowledgements
This material is based upon work supported by the National Science Foundation under Grant No. 1849559.

# Overview
The code performs DBSCAN on 2-D datasets. As described in the ICS'19 paper above, the algorithm splits the work between the CPU and GPU, where the GPU finds all neighbors within epsilon and the CPU uses these neighbors as input to cluster the data. This gives the irregular instruction flow to the CPU while allowing the GPU to perform the (more) regularized computation.

The algorithm splits the input dataset into several partitions. Each partition is independently clustered, and partitions are merged at the end to compute the final set of clusters. Each partition is processed by a single CPU core, thus exploiting parallelism on the CPU. For example, if you had a 8 core CPU and set the algorithm to concurrently cluster 8 partitions, then all 8 cores could be utilized. 


# Constraints and Testing
- The code requires at least one GPU (and a CPU)
- The code has been tested using 2 GPUs
- The code has been compiled using GCC v.5.4.0 using CUDA 9
- There are no special libraries. The code uses OpenMP and Thrust.

# Future Plans
- You will notice that there are hooks for changing the data dimensionality. I hope to incorporate >2 dimensions in the future.

# Usage and Example Execution
To execute the program, you must give the following as input on the command line: The input dataset as a csv (2-D datasets only), epsilon, min. points, the data dimensionality (this will be 2), and the number of partitions to generate from the input dataset.

./main \<dataset\> \<epsilon\> \<minpts\> \<data dimensionality\> \<partitions\>
  
Using the dataset in the "datasets" directory of this repository, the following takes the input dataset and sets epsilon=0.01 min. points=4, 2-D, and the data is split into 10 partitions.

./main gaia_dr2_ra_dec_50M.txt 0.01 4 2 10


Here's the same as the above but with epsilon=0.06.

./main gaia_dr2_ra_dec_50M.txt 0.06 4 2 10

Each execution will output information to gpu_stats.txt

The output below shows the following for the epsilon=0.01 and epsilon=0.06 executions. The output shows the execution time, dataset, epsilon, min. points, the number of partitions, the number of clusters and the fraction of points that were detected as noise. The parameters will be described in the next section.

20.7339, ../datasets/gaia_dr2_ra_dec_50M.txt, 0.01, 4, 10, 1116232, 0.3627, DENSEBOX/MU/PARCHUNKS/NUMGPU/GPUSTREAMS/PARTITIONMODE/SCHEDULE/DTYPE(float/double): 2, 0.25, 10, 1, 3, 1, 0, float

23.0778, ../datasets/gaia_dr2_ra_dec_50M.txt, 0.06, 4, 10, 9155, 0.0009337, DENSEBOX/MU/PARCHUNKS/NUMGPU/GPUSTREAMS/PARTITIONMODE/SCHEDULE/DTYPE(float/double): 2, 0.25, 10, 1, 3, 1, 0, float

From the above, this yields a range of noise point percentages: 36%-0.09337% corresponding to epsilon=0.01 and 0.06, respectively. These values bracket useful ranges of the fraction of noise points.

# Parameters
- The parameters can be changed in params.h
- If you are only interested in running the algorithm on a single GPU, you can likely use the default values of the parameters. However you will need to change the parameters if you would like to use multiple GPUs, change the number of cores that concurrently partition the data, and other performance tuning options.

The parameters (default values) that can be changed are described as follows.
- DTYPE (float): float or double. This is the floating point precision of the input dataset. 
- BLOCKSIZE (256): This is the CUDA block size of the main GPU kernels. You may change this to tune performance.  
- NTHREADS (16): This is the number of physical CPU cores. Several parts of the algorithm are parallelized on the CPU and use multiple CPU threads. Set this to the number of physical cores on your platform.
- DENSEBOXMODE (2): This refers to the dense box optimization in the paper. 0- disable dense box mode on all partitions; 1- enable dense box mode on all partitions; 2- dynamically enable the dense box algorithm based on the data density on each partition.
- MU (0.25): The parameter used when using the dynamic dense box mode. 0.25 performed well in the experimental evaluation of the paper.
- PARCHUNKS (10): The number of partitions that should be concurrently clustered. Do not use more than NTHREADS.
- NUMGPU (1): The number of GPUs used to compute the epsilon-neighborhoods of all points in each partition.
- GPUSTREAMS (3): The number of GPU streams for each GPU. Used to overlap computation and communication over PCIe. 3 streams is likely sufficient.
- PARTITIONMODE (1): The order in which partitions will be processed. 1- default; 2- move partition boundaries to exploit local density minima to reduce the overhead of merging adjacent partitions.
- SCHEDULE (0): 0- default order of processing partitions; 1- order the partitions to be processed from most work to least work to improve load balancing at the end of the computation.
- GPUBUFFERSIZE (100000000): GPU result set size for each stream. If your GPU exceeds memory capacity, you may want to lower this value.



