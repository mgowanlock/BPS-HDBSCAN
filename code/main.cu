// MIT License

// Copyright (c) 2021 Mike Gowanlock

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <semaphore.h>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include <algorithm> 
#include <string.h>
#include <fstream>
#include <iostream>
#include <string>
#include "GPU.h"
#include "kernel.h"
#include "cluster.h"
#include "par_sort.h" // for parallel sort with parallel mode extensions (nvcc doesn't like it)
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include "UF.hpp"

//for printing defines as strings
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

//sort descending
bool compareByDimVariance(const dim_reorder_sort &a, const dim_reorder_sort &b)
{
    return a.variance > b.variance;
}


//sort ascending
bool compareLexicographicCellCoords(const cellCoords &a, const cellCoords &b)
{
    if(a.dim1 < b.dim1)
    {
    	return 1;
    }
    else if (a.dim1 > b.dim1)
    {
    	return 0;
    }
    
    return (a.dim2<b.dim2);
 
}

using namespace std;

//function prototypes
void generateNeighborTableCPUPrototype(std::vector<std::vector <DTYPE> > *NDdataPoints, unsigned int queryPoint, DTYPE epsilon, grid * index, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, std::vector<uint64_t> * cellsToCheck, table * neighborTableCPUPrototype);
uint64_t getLinearID_nDimensions(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions);
void populateNDGridIndexAndLookupArray(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells, unsigned int ** gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems);
void populateNDGridIndexAndLookupArrayParallel(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells);
void generateNDGridDimensions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells);
void importNDDataset(std::vector<std::vector <DTYPE> > *dataPoints, char * fname);
void generateSyntheticData(std::vector<std::vector <DTYPE> > *dataPoints, unsigned int datasetsize);
void CPUBruteForceTable(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, table * neighborTable, unsigned int * totalNeighbors);
void sortInNDBins(std::vector<std::vector <DTYPE> > *dataPoints);
void ReorderByDimension(std::vector<std::vector <DTYPE> > *NDdataPoints);
void test_disjoint_sets();
void test_disjoint_sets2();
void densebox(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, unsigned int minPts, std::vector<int> *clusterIDs, std::vector<unsigned int> *queryVect, std::vector<bool> *denseBoxPoints);
void denseboxWithoutIndex(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, unsigned int minPts, std::vector<int> *clusterIDs, std::vector<unsigned int> *queryVect, std::vector<bool> *denseBoxPoints);
void PrintSets(std::vector<std::vector<unsigned int> > * disjointSetList, std::vector<unsigned int> * denseBoxCells, 
	std::vector<std::vector <DTYPE> > *NDdataPoints, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index);
void generatePointLists(std::vector<int>  * clusterIDs, std::vector<unsigned int>  * queryVect, std::vector<std::vector<unsigned int> > * disjointSetList, std::vector<unsigned int> * denseBoxCells, 
	std::vector<std::vector <DTYPE> > *NDdataPoints, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index, std::vector<bool> *denseBoxPoints);
void processNeighborTableForClusters(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDs, neighborTableLookup * neighborTable, unsigned int minPts, unsigned long int sizeDB, std::vector<unsigned int> *queryVectForNeighborTableDBSCAN, std::vector<bool> *denseBoxPoints);


int getUniqueClusters(unsigned int queryId, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDs, neighborTableLookup * neighborTable, std::vector<int> * outputUniqueIds);
void printclusters(std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<int> *clusterIDs, char * fname);
//original in ICS'19 paper
void generateReferencePoints(DTYPE * minArr, DTYPE epsilon,unsigned int * nCells, std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDs, struct gridCellLookup * gridCellLookupArr, unsigned int nNonEmptyCells); 
//With better reference point generation for shadow region
void generateReferencePointsShadowRegionOnly(DTYPE * minArr, DTYPE epsilon, unsigned int * nCells, std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDs, struct gridCellLookup * gridCellLookupArr, unsigned int nNonEmptyCells);


unsigned int getNumClusters(std::vector<int> *clusterIDs);
int computeMergesRefPointDistanceCalculations(std::vector<std::vector <DTYPE> > *NDdataPoints, double epsilon, unsigned int queryId, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDs, neighborTableLookup * neighborTable, std::vector<int> * uniqueIds, std::vector<bool> *denseBoxPoints, std::vector<int> * outputMergeIdsRefPnts);
void generateDBSCANQueryVect(std::vector<unsigned int> *queryVect, std::vector<unsigned int> *queryVectForNeighborTableDBSCAN, unsigned int sizeDB);
void initNeighbortable(neighborTableLookup * neighborTable, unsigned int numElems);
int getGPU(int * GPUresources, omp_lock_t * gpuwritelock);
void returnGPU(int * GPUresources, int gpuid, omp_lock_t * gpuwritelock);
void generatePointListsWithoutIndex(std::vector<int>  * clusterIDs, std::vector<unsigned int>  * queryVect, 
	std::vector<std::vector<unsigned int> > * disjointSetList, std::vector<unsigned int> * denseBoxCells, 
	std::vector<std::vector <DTYPE> > *NDdataPoints, unsigned int * nNonEmptyCells, std::vector<bool> *denseBoxPoints, std::vector<DenseBoxPointIDStruct> *DenseBoxPointIDs);
unsigned long int updateAndComputeNumCellsInWindow(unsigned long int * buffer, unsigned int cnt, unsigned long int newVal);





//to partition the dataset-- Default in the ICS'19 paper
void generatePartitions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells, unsigned int * binBounaries, const unsigned int CHUNKS);

//partition by minimizing the points in the shadow regions
void generatePartitionsMinimizeShadowRegionPoints(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells, unsigned int * binBounaries, const unsigned int CHUNKS);



//generates a schedule for load balancing
void generateScheduleLoadBalance(unsigned int * binBounaries, const unsigned int CHUNKS, unsigned int * schedule, std::vector<std::vector <DTYPE> > *PartitionedNDdataPoints);


//generate datasets for each partition
void partitionDataset(std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<std::vector <DTYPE> > *PartitionedNDdataPoints, 
		std::vector<struct pointChunkLookupArr> *pointChunkMapping, std::vector<unsigned int> * pointsIDsInShadowRegion, 
		std::vector<std::vector <DTYPE> > *NDdataPointsInShadowRegion, DTYPE epsilon, DTYPE* minArr, unsigned int * nCells, 
		unsigned int * binBounaries, const unsigned int CHUNKS);




//merges clusters across partitions at the end
void processNeighborTableForPartitions(std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<unsigned int> *pointsIDsInShadowRegion, DTYPE epsilon, std::vector<unsigned int> *queryVect,  std::vector<int> *clusterIDsShadow, neighborTableLookup * neighborTable, unsigned int minPts, unsigned long int sizeDB, std::vector<int> *clusterIDsAcrossPartitions, const unsigned int CHUNKS, std::vector<struct pointChunkLookupArr> *pointChunkMapping, std::vector<int> * finalClusterIDs);

//gets the unique clusters for the shadow regions
int getUniqueClustersForShadowRegions(unsigned int queryId, std::vector<unsigned int> *queryVectRefPoints, std::vector<int> *clusterIDsMergedFromPartitions, 
				std::vector<unsigned int> *pointsIDsInShadowRegion, neighborTableLookup * neighborTable, std::vector<int> * outputUniqueIds);


int computeMergesRefPointDistanceCalculationsShadowRegion(std::vector<std::vector <DTYPE> > *NDdataPoints, double epsilon, unsigned int minPts, unsigned int queryId, 
					std::vector<int> *clusterIDsMergedFromPartitions,std::vector<unsigned int> *pointsIDsInShadowRegion,  neighborTableLookup * neighborTable, 
					std::vector<int> * uniqueIds, std::vector<int> * outputMergeIdsRefPnts);



int main(int argc, char *argv[])
{
	omp_set_nested(1);	



	/////////////////////////
	// Get information from command line
	//1) the dataset, 2) epsilon, 3) number of dimensions
	/////////////////////////

	//Read in parameters from file:
	//dataset filename and cluster instance file
	if (argc!=6)
	{
	cout <<"\n\nIncorrect number of input parameters.  \nShould be dataset file, epsilon, minpts, number of dimensions, #Partitions\n";
	return 0;
	}
	
	

	//copy parameters from commandline:
	//char inputFname[]="data/test_data_removed_nan.txt";	
	char inputFname[500];
	char inputEpsilon[500];
	char inputMinpts[500];
	char inputnumdim[500];

	strcpy(inputFname,argv[1]);
	strcpy(inputEpsilon,argv[2]);
	strcpy(inputMinpts,argv[3]);
	strcpy(inputnumdim,argv[4]);

	const unsigned int NCHUNKS = atoi(argv[5]); 
	

	DTYPE epsilon=atof(inputEpsilon);
	unsigned int minPts=atoi(inputMinpts);
	unsigned int NDIM=atoi(inputnumdim);

	if (GPUNUMDIM!=NDIM){
		printf("\nERROR: The number of dimensions defined for the GPU is not the same as the number of dimensions\n \
		 passed into the computer program on the command line. GPUNUMDIM=%d, NDIM=%d Exiting!!!",GPUNUMDIM,NDIM);
		return 0;
	}

	printf("\nDataset file: %s",inputFname);
	printf("\nEpsilon: %f",epsilon);
	printf("\nMinPts: %u",minPts);
	printf("\nNumber of dimensions (NDIM): %d",NDIM);
	printf("\nNumber of Chunks/Partitions: %u\n",NCHUNKS);
	printf("\nPartition mode: %u", PARTITIONMODE);

	//////////////////////////////
	//import the dataset:
	/////////////////////////////
	
	//one vector for each dimension	
	std::vector<std::vector <DTYPE> > NDdataPoints;
	NDdataPoints.resize(GPUNUMDIM);
	importNDDataset(&NDdataPoints, inputFname);

	unsigned long int sizeDB=NDdataPoints[0].size();
	

	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	

	printf("\n*****************\nWarming up GPU:\n*****************\n");
	
	for (int i=0; i<NUMGPU; i++)
	{
	cudaSetDevice(i);	
	warmUpGPU();
	}
	printf("\n*****************\n");

	
	//GPU resource control
	//semaphore for accessing GPUs
	sem_t gpu_sem;
	sem_init(&gpu_sem, 0, NUMGPU);

	//lock for accessing/updating the GPUresources array
	omp_lock_t gpuwritelock;
	omp_init_lock(&gpuwritelock);

	int GPUresources[NUMGPU];

	//init resource array
	for (int i=0; i<NUMGPU; i++)
	{
		GPUresources[i]=0;
	}

	//Cluster ids for the dense box algorithm
	//One set of cluster IDs per chunk that we reconcile later
	std::vector<int>clusterIDs[NCHUNKS];
	

	//after all partitions have been merged, these are the final cluster assignments
	std::vector<int>finalClusterIDs;

	double tstarttotal=omp_get_wtime();

	//Testing time to partition dataset when we allocate here (and not outside of trials loop)	
	std::vector<std::vector <DTYPE> > PartitionedNDdataPoints[NCHUNKS];	
	
	for (int y=0; y<NCHUNKS; y++)
	{
		clusterIDs[y].clear();
		clusterIDs[y].shrink_to_fit();
	}
	finalClusterIDs.clear();
	finalClusterIDs.shrink_to_fit();
	
	

	DTYPE * minArr= new DTYPE[NUMINDEXEDDIM];
	DTYPE * maxArr= new DTYPE[NUMINDEXEDDIM];
	unsigned int * nCells= new unsigned int[NUMINDEXEDDIM];
	uint64_t totalCells=0;

	generateNDGridDimensions(&NDdataPoints,epsilon, minArr, maxArr, nCells, &totalCells);
	printf("\nGrid: total cells (including empty) %lu",totalCells);cout.flush();
	printf("\nFinished generating grid dimensions for partitioning");cout.flush();	
	

	//disable sensebox
	#if DENSEBOXMODE==0
	bool ENABLEDENSEBOX=false;
	#endif

	//enable densebox
	#if DENSEBOXMODE==1
	bool ENABLEDENSEBOX=true;
	#endif

	//dynamic densebox in main loop

	


	//Allocate pinned memory:
	//Buffers for each stream for each GPU. There are GPUSTREAMS streams per GPU.
	int * pointIDKey[NUMGPU][GPUSTREAMS];
	int * pointInDistValue[NUMGPU][GPUSTREAMS];
	
	

	//We count the bin boundaries on the edges of the grid cells, so add 1
	unsigned int * binBounaries=new unsigned int[NCHUNKS+1];

	//Make NCHUNKS datasets
	//pointChunkMapping- Mapping of the datapoints to the points in each chunk
	//shadowregion- data points in the shadow region for merging the datasets later
	std::vector<struct pointChunkLookupArr> pointChunkMapping; //maps the indices of the global data to the chunks 	
	std::vector<unsigned int> pointsIDsInShadowRegion;	//point ids in the shadow region
	std::vector<std::vector <DTYPE> > NDdataPointsInShadowRegion;
	NDdataPointsInShadowRegion.resize(GPUNUMDIM);


	#if SCHEDULE==1
	unsigned int *partitionSchedule= new unsigned int[NCHUNKS];
	#endif
	
	//sections for allocating pinned memory with generating partitions
	//Hide some of the pinned memory allocation time
	#pragma omp parallel sections
	{
		#pragma omp section
		{	
		double tstart_gen_partitions=omp_get_wtime();
		#if PARTITIONMODE==1
		generatePartitions(&NDdataPoints,epsilon, minArr, maxArr, nCells, &totalCells, binBounaries, NCHUNKS);
		#endif

		#if PARTITIONMODE==2
		generatePartitionsMinimizeShadowRegionPoints(&NDdataPoints,epsilon, minArr, maxArr, nCells, &totalCells, binBounaries, NCHUNKS);
		#endif

		

		

		double tend_gen_partitions=omp_get_wtime();
		printf("\nTime to generate the partitions: %f", tend_gen_partitions - tstart_gen_partitions);
		

		double tstart_partition_dataset=omp_get_wtime();
		partitionDataset(&NDdataPoints, PartitionedNDdataPoints, &pointChunkMapping, &pointsIDsInShadowRegion, &NDdataPointsInShadowRegion, epsilon, minArr, nCells, binBounaries, NCHUNKS);
		double tend_partition_dataset=omp_get_wtime();
		printf("\nTime to partition the dataset: %f",tend_partition_dataset - tstart_partition_dataset);
		
		printf("\n[Time total to generate partitions and partition the dataset]: %f", tend_partition_dataset- tstart_gen_partitions);
		
		//If we modify the order of partitioning processing for load balancing
		#if SCHEDULE==1
		generateScheduleLoadBalance(binBounaries, NCHUNKS, partitionSchedule, PartitionedNDdataPoints);
		#endif
	
		}


		#pragma omp section
		{
			allocPinnedmemory(pointIDKey, pointInDistValue);		
		}
	}



	double timeWaitingGPU[PARCHUNKS];
	for (int i=0; i<PARCHUNKS; i++)
	{
		timeWaitingGPU[i]=0;
	}



	double tstart_dbscan=omp_get_wtime();


	// printf("\nOnly processing CHUNK 1");
	//begin processing chunks of the dataset
	#pragma omp parallel for num_threads(PARCHUNKS) shared(clusterIDs,PartitionedNDdataPoints,timeWaitingGPU,pointIDKey,pointInDistValue) schedule(dynamic,1)
	for (unsigned int i=0; i<NCHUNKS; i++)
	{

		

	//by default we execute the partitions in order
	unsigned int partitionID=i;
	
	//If we use SCHEDULE==1, we order the partitions by most work to least work to improve load imbalance
	#if SCHEDULE==1	
	partitionID=partitionSchedule[i];
	#endif


	int tid=omp_get_thread_num();	


	//make a copy of the partitioned dataset 
	//with densebox, we need to write to the partitioned dataset
	std::vector<std::vector <DTYPE> > PartitionedData=PartitionedNDdataPoints[partitionID]; 

	unsigned long int sizeDBPartition=PartitionedData[0].size();	


	//local copy of the cluster ids:
	std::vector<int>PartitionClusterIDs;


	printf("\nData points in partition #%d: %lu",partitionID,sizeDBPartition);cout.flush();	

	DTYPE * minArr= new DTYPE[NUMINDEXEDDIM];
	DTYPE * maxArr= new DTYPE[NUMINDEXEDDIM];
	unsigned int * nCells= new unsigned int[NUMINDEXEDDIM];
	uint64_t totalCells=0;
	unsigned int nNonEmptyCells=0;
	uint64_t totalNeighbors =0;
	
	
	int maxClusterId=-1; //used to append the cluster ids for DBSCAN leftovers after densebox

	generateNDGridDimensions(&PartitionedData,epsilon, minArr, maxArr, nCells, &totalCells);
	printf("\nGrid: total cells (including empty) %lu",totalCells);cout.flush();

	// allocate memory for index now that we know the number of cells
	//the grid struct itself
	//the grid lookup array that accompanys the grid -- so we only send the non-empty cells
	struct grid * index; //allocate in the populateDNGridIndexAndLookupArray -- only index the non-empty cells
	struct gridCellLookup * gridCellLookupArr; //allocate in the populateDNGridIndexAndLookupArray -- list of non-empty cells

	//the grid cell mask tells you what cells are non-empty in each dimension
	//used for finding the non-empty cells that you want
	

	//ids of the elements in the database that are found in each grid cell
	unsigned int * indexLookupArr=new unsigned int[sizeDBPartition]; 

	populateNDGridIndexAndLookupArrayParallel(&PartitionedData, epsilon, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells);

	//DISABLE DENSEBOX BASED ON THE DATASET CHARACTERISTICS 
	#if DENSEBOXMODE==2
	double meanPointsHeuristic=sizeDBPartition/(8.0*nNonEmptyCells);
	bool ENABLEDENSEBOX=true;
	if (meanPointsHeuristic<((1.0*minPts)*(MU*1.0)))
	{
		ENABLEDENSEBOX=false;
	}
	printf("\nDynamic densebox: Partition: %d, Heuristic value: %f, MU: %f, DENSEBOX: %d",partitionID, meanPointsHeuristic, MU, ENABLEDENSEBOX);
	#endif
	

	//Point ids that need to be searched on the GPU (weren't found by dense box)
	std::vector<unsigned int> queryVect;
	//Point ids that need to be searched using DBSCAN that takes as input the neighbor tbale
	//These points are in sparse regions, that have neighbors, but can't be added to preexisting sets
	std::vector<unsigned int> queryVectForNeighborTableDBSCAN;

	//vector of densebox point ids used to make sure we only merge
	//densebox-densebox
	//dbscan-densebox
	//NOT dbscan-dbscan
	std::vector<bool> denseBoxPoints;

	//Neighbortable storage -- the result for the neighbors of each point to be generated on the GPU
	neighborTableLookup * neighborTable;

	
	if (ENABLEDENSEBOX)	
	{
	printf("\nBefore Densebox");cout.flush();

	//densebox algorithm
	double tstartdb=omp_get_wtime();	

	
	// denseboxWithoutIndex(&PartitionedData, epsilon, minPts, &clusterIDs[i], &queryVect, &denseBoxPoints);
	denseboxWithoutIndex(&PartitionedData, epsilon, minPts, &PartitionClusterIDs, &queryVect, &denseBoxPoints);
	

	double tenddb=omp_get_wtime();	
	printf("\nTime to densebox points: %f", tenddb - tstartdb);cout.flush();

	//generate reference points for merging denseboxes
	//appends PartitionedData, queryVect and clusterIDs

	double tstartrefpnts=omp_get_wtime();	
	
	generateReferencePoints(minArr, epsilon, nCells, &PartitionedData, &queryVect, &PartitionClusterIDs, gridCellLookupArr, nNonEmptyCells);
	
	double tendrefpnts=omp_get_wtime();	
	printf("\nTime to generate reference points: %f",tendrefpnts - tstartrefpnts);
	}
	
	 
	//IF densebox, we need to allocate memory that includes reference points
	if (ENABLEDENSEBOX)
	{
		//Neighbortable storage -- the result
		//need storage for the reference points as well.
		//To avoid the lookup for ref points, we allocate space for even the reference points in the empty cells 
		printf("\nsize db before alloc neighbor table: %lu ", sizeDBPartition);
		printf("\nSize number of neighbortable elements (data points+ non-empty cell ref. points): %lu", sizeDBPartition+nNonEmptyCells);
		neighborTable= new neighborTableLookup[sizeDBPartition+nNonEmptyCells];
		//initialize neighbortable
		initNeighbortable(neighborTable, sizeDBPartition+nNonEmptyCells);

		//compute max cluster id for DBSCAN leftovers
		for (int k=0; k<PartitionClusterIDs.size(); k++)
		{
			// printf("\nPoint %d, cluster id: %d",i,clusterIDs[i]);
			if (PartitionClusterIDs[k]>maxClusterId)
			{
				maxClusterId=PartitionClusterIDs[k];
			}
		}
	}
	//if we don't densebox, then we don't need to allocate reference points in the neighbortable
	//Need to iniitalize the cluster ids and query Vect
	else
	{
		printf("\nsize db before alloc neighbor table: %lu ", sizeDBPartition);
		printf("\nSize number of neighbortable elements (data points): %lu", sizeDBPartition);
		neighborTable= new neighborTableLookup[sizeDBPartition];
		//initialize neighbortable
		initNeighbortable(neighborTable, sizeDBPartition);	

		//need to initialize the query vect
		queryVect.resize(sizeDBPartition);
		for (unsigned int x=0; x<sizeDBPartition; x++)
		{
			queryVect[x]=x;
		}

		//initialize the cluster ids
		PartitionClusterIDs.resize(sizeDBPartition);

		for (unsigned int x=0; x<sizeDBPartition; x++)
		{
		PartitionClusterIDs[x]=-1;
		}

		//max cluster id for DBSCAN leftovers:
		//0 if we don't densebox
		maxClusterId=0;

	}

	

	std::vector<struct neighborDataPtrs> pointersToNeighbors;


	
	//////////////////////////////////////
	//GPU

	//for getting cell and point comparisons on the GPU
	CTYPE* workCounts = (CTYPE*)malloc(2*sizeof(CTYPE)); workCounts[0]=0;workCounts[1]=0;	

	double tstartwait=omp_get_wtime();

	//Only NUMGPUs allowed in this section
	sem_wait(&gpu_sem);

	
	//get a GPU
	int gpuid=getGPU(GPUresources,&gpuwritelock);
	printf("\nPartition %d, got GPU: %d",i,gpuid);

	double tendwait=omp_get_wtime();	
	timeWaitingGPU[tid]+=tendwait - tstartwait;

	
	double tstart=omp_get_wtime();	
	distanceTableNDGridBatches(&PartitionedData, &queryVect, &epsilon, index, gridCellLookupArr, &nNonEmptyCells,  minArr, nCells, indexLookupArr, neighborTable, &pointersToNeighbors, &totalNeighbors, workCounts, sizeDBPartition, pointIDKey, pointInDistValue, ENABLEDENSEBOX, gpuid);
	double tend=omp_get_wtime();		

	
	//return GPU
	returnGPU(GPUresources, gpuid, &gpuwritelock);


	sem_post(&gpu_sem);
	


	//END GPU
	////////////////////////////


	//need to assign DBScan leftovers a clusterId that's larger than any cluster Id assigned
	//so that it can simply assign a number of cluster ids in a row
	//but the cluster ids are not 0,1,2,3, because they were assigned based on the dense boxes
	//find the maximum cluster id and assign dbscan leftovers that cluster id+1
	
	
	
	printf("\nStarting DBSCAN leftovers cluster id at: %d", maxClusterId+1);

	//generate query vect if we densebox based on the reference points
	if (ENABLEDENSEBOX)
	{
		generateDBSCANQueryVect(&queryVect, &queryVectForNeighborTableDBSCAN, sizeDBPartition);
	}
	//if we don't desebox then we need to initialize the query vect to be all of the data points in the partition
	else
	{
		queryVectForNeighborTableDBSCAN.resize(sizeDBPartition);
		for (unsigned int x=0; x<sizeDBPartition; x++)
		{
			queryVectForNeighborTableDBSCAN[x]=x;
		}
	}


	double tstartleftovers=omp_get_wtime();	
	
	// dbscanTableLeftovers(neighborTable, &queryVectForNeighborTableDBSCAN, &clusterIDs[i], maxClusterId+1, minPts, sizeDBPartition);
	dbscanTableLeftovers(neighborTable, &queryVectForNeighborTableDBSCAN, &PartitionClusterIDs, maxClusterId+1, minPts, sizeDBPartition);
	double tendleftovers=omp_get_wtime();	
	printf("\nTime to DBSCAN leftovers for partition #%u: %f",partitionID, tendleftovers - tstartleftovers);
	
	
	//resolve conflicts etc.
	//use the original database size to discern between real data points and reference points
	//We join clusters using the reference points here
	if (ENABLEDENSEBOX)
	{
	double tstartprocneighbortable=omp_get_wtime();	
	// processNeighborTableForClusters(&PartitionedData, epsilon, &queryVect, &clusterIDs[i], neighborTable, minPts, sizeDBPartition, &queryVectForNeighborTableDBSCAN, &denseBoxPoints);
	processNeighborTableForClusters(&PartitionedData, epsilon, &queryVect, &PartitionClusterIDs, neighborTable, minPts, sizeDBPartition, &queryVectForNeighborTableDBSCAN, &denseBoxPoints);
	double tendprocneighbortable=omp_get_wtime();	
	printf("\nTime to process neighbortable for ref points: %f", tendprocneighbortable - tstartprocneighbortable);
	

	//need to use the cluster ids of the data points (remove reference points)
	//need to use the initial size of the database (remove reference points)
	// clusterIDs[i].resize(sizeDBPartition);
	PartitionClusterIDs.resize(sizeDBPartition);
		for (int x=0; x<GPUNUMDIM; x++)
		{
		PartitionedData[x].resize(sizeDBPartition);
		}
	}

	
	//update the cluster ids for the partition
	#pragma omp critical
	{
	clusterIDs[partitionID]=PartitionClusterIDs;
	}

	// unsigned int numClusters=getNumClusters(&clusterIDs[i]);
	// printf("\nNum clusters partition: %d, densebox: %u",i,numClusters);

	//free memory in neighbortable
	for (int x=0; x<pointersToNeighbors.size(); x++)
	{
		delete [] pointersToNeighbors[x].dataPtr;
	}

	free(workCounts);
	delete [] indexLookupArr;
	delete [] neighborTable;
	delete [] gridCellLookupArr;
	delete [] index;
	delete [] minArr;
	delete [] maxArr;
	delete [] nCells;


	double tendpartition=omp_get_wtime();
	printf("\nTime partition #%d finishes from the start (before loop): %f", partitionID, tendpartition - tstart_dbscan);

	} //END loop over chunks


	double tend_dbscan=omp_get_wtime();
	printf("\n[Time total to DBSCAN]: %f",tend_dbscan - tstart_dbscan);




	//MERGE DATA PARTITIONS HERE
	//STEPS:
	//1) WHEN PARTITIONING THE DATASET, STORE THE POINTS WITHIN THE SHADOW REGION (NDdataPointsInShadowRegion)
	//2) STORE THE POINT INDEXES OF THE SHADOW REGION POINTS ACROSS THE ENTIRE DATASET (pointsIDsInShadowRegion)
	//SO WE KNOW THE REAL POINT IDS
	//3) THEN INDEX THE POINTS
	//4) GENERATE REFERENCE POINTS IN NON-EMPTY CELLS
	//5) RESOLVE CONFLICTS USING THE ORIGINAL POINT IDS IN THE DATASET



	//free memory -- partitioned data
	for (int i=0; i<NCHUNKS; i++)
	{
		for (int j=0; j<GPUNUMDIM; j++)
		{
		PartitionedNDdataPoints[i][j].resize(0);
		PartitionedNDdataPoints[i][j].shrink_to_fit();
		}
		// PartitionedNDdataPoints[i] = std::vector<std::vector <DTYPE>>();
	}


	// printf("Returning before reconciling partitions");
	// return 0;


	double tstart_reconcile_partitions=omp_get_wtime();	
	if (NCHUNKS>1)
	{

	


	printf("\n**************\nReconciling the partitions\n**************\n");cout.flush();
	//Step 3: Index the points:
	//generateNDGridDimensions() has already been executed for the entire dataset to get statistics
	//for the partitioning-- reuse the values, e.g., totalCells, minArr, nCells etc.
	
	printf("\nNum. Data points in shadow region: %lu",NDdataPointsInShadowRegion[0].size());	

	unsigned long int sizeDBShadow=NDdataPointsInShadowRegion[0].size();

	struct grid * index; //allocate in the populateDNGridIndexAndLookupArray -- only index the non-empty cells
	struct gridCellLookup * gridCellLookupArr; //allocate in the populateDNGridIndexAndLookupArray -- list of non-empty cells

	//the grid cell mask tells you what cells are non-empty in each dimension
	//used for finding the non-empty cells that you want

	//ids of the elements in the database that are found in each grid cell
	unsigned int * indexLookupArr=new unsigned int[NDdataPoints[0].size()]; 
	unsigned int nNonEmptyCells=0;
	uint64_t totalNeighbors =0;

	
	std::vector<int>clusterIDsShadow; //MIGHT NOT BE NEEDED -- CHECK LATER

	//add datapoints to the queryVect fpr the shadow region and initialize
	std::vector<unsigned int> queryVectShadow;
	for (unsigned int i=0; i<NDdataPointsInShadowRegion[0].size(); i++){
		queryVectShadow.push_back(i);
	}

	
		

	//populate the index-- minArr, nCells etc already computed when we determined the dimensions of the grid for partitioning
	double tstartpopulateindexpartition=omp_get_wtime();
	populateNDGridIndexAndLookupArrayParallel(&NDdataPointsInShadowRegion, epsilon, &gridCellLookupArr, &index, indexLookupArr, minArr, nCells, totalCells, &nNonEmptyCells);
	double tendpopulateindexpartition=omp_get_wtime();
	printf("\nTime to populate the index (component of partition reconcilation): %f", tendpopulateindexpartition - tstartpopulateindexpartition);fflush(stdout);
	

	//generate reference points
	double tstartgenrefpntspartition=omp_get_wtime();
	//original
	// generateReferencePoints(minArr, epsilon, nCells, &NDdataPointsInShadowRegion, &queryVectShadow, &clusterIDsShadow, gridCellLookupArr, nNonEmptyCells);
	generateReferencePointsShadowRegionOnly(minArr, epsilon, nCells, &NDdataPointsInShadowRegion, &queryVectShadow, &clusterIDsShadow, gridCellLookupArr, nNonEmptyCells);
	double tendgenrefpntspartition=omp_get_wtime();
	printf("\nTime to generate reference points (component of partition reconcilation): %f", tendgenrefpntspartition - tstartgenrefpntspartition);fflush(stdout);


	//allocate the neighbortable
	neighborTableLookup * neighborTable= new neighborTableLookup[NDdataPointsInShadowRegion[0].size()];
	//initialize neighbortable
	printf("\nElems in neighbortable (shadow region data points + ref points): %lu",NDdataPointsInShadowRegion[0].size());fflush(stdout);
	
	double tstartinitneighbortablepartition=omp_get_wtime();
	initNeighbortable(neighborTable, NDdataPointsInShadowRegion[0].size());
	double tendinitneighbortablepartition=omp_get_wtime();
	printf("\nTime to init neighbortable (component of partition reconcilation): %f", tendinitneighbortablepartition - tstartinitneighbortablepartition);fflush(stdout);		

	//pointers to the neighbors
	std::vector<struct neighborDataPtrs> pointersToNeighbors;
	pointersToNeighbors.clear();

	//for getting cell and point comparisons on the GPU
	CTYPE* workCounts = (CTYPE*)malloc(2*sizeof(CTYPE));
	workCounts[0]=0; workCounts[1]=0;
	

	distanceTableNDGridBatches(&NDdataPointsInShadowRegion, &queryVectShadow, &epsilon, index, gridCellLookupArr, &nNonEmptyCells,  minArr, nCells, indexLookupArr, neighborTable, &pointersToNeighbors, &totalNeighbors, workCounts, sizeDBShadow, pointIDKey, pointInDistValue, 1, 0);
	

	//Now we need to process the reference points to merge the clusters between the partitions	
	//Also, need to create clusters for noise points that may be noise in different partitions but form clusters
	double tstartprocpartitions=omp_get_wtime();
	processNeighborTableForPartitions(&NDdataPoints, &pointsIDsInShadowRegion, epsilon, &queryVectShadow, &clusterIDsShadow, neighborTable, minPts, sizeDBShadow, clusterIDs, NCHUNKS, &pointChunkMapping, &finalClusterIDs);
	double tendprocpartitions=omp_get_wtime();	
	printf("\nTime to process neighbortable for partitions (component of partition reconcilation): %f",tendprocpartitions - tstartprocpartitions);

	// printf("\nReturning after proc neighbor table for partitions");
	// return 0;
	

	//free memory in neighbortable
	for (int i=0; i<pointersToNeighbors.size(); i++)
	{
		delete [] pointersToNeighbors[i].dataPtr;
	}

	delete [] neighborTable;
	delete [] indexLookupArr;
	delete [] gridCellLookupArr;
	delete [] minArr;
	delete [] maxArr;
	delete [] nCells;
	free(workCounts);


	} //end reconcile partitions
	double tend_reconcile_partitions=omp_get_wtime();
	printf("\n[Time to reconcile partitions]: %f", tend_reconcile_partitions-tstart_reconcile_partitions);


	deAllocPinnedMemory(pointIDKey, pointInDistValue);

	delete [] binBounaries;

	for (int i=0; i<PARCHUNKS; i++)
	{
		printf("\nTime tid: %d spent waiting to use GPU: %f",i, timeWaitingGPU[i]);
	}
	



	double tendtotal=omp_get_wtime();
	double totaltime=tendtotal - tstarttotal;
	

	
	printf("\nTotal time densebox alg: %f", totaltime);cout.flush();


	
	unsigned int numClusters=0;	
	unsigned int cntnoise=0;
	unsigned int cntunassigned=0;
	if (NCHUNKS==1)
	{
		numClusters=getNumClusters(&clusterIDs[0]);
		printf("\nDensebox total clusters: %u",numClusters);cout.flush();

		for (int i=0; i<sizeDB; i++)
		{
			if (clusterIDs[0][i]==0)
			cntnoise++;

			if (clusterIDs[0][i]==-1)
				cntunassigned++;

		}
		printf("\nNum noise (check): %u",cntnoise);cout.flush();
		printf("\nNum unassigned (check): %u",cntunassigned);cout.flush();
	}

	if (NCHUNKS>1)
	{
	
	numClusters=getNumClusters(&finalClusterIDs);
	printf("\nDensebox total clusters: %u",numClusters);cout.flush();


	
	for (int i=0; i<sizeDB; i++)
	{
		if (finalClusterIDs[i]==0)
		cntnoise++;

		if (finalClusterIDs[i]==-1)
			cntunassigned++;

	}
	printf("\nNum noise (check): %u",cntnoise);cout.flush();
	printf("\nNum unassigned (check): %u",cntunassigned);cout.flush();

	}
	
	gpu_stats<<totaltime<<", "<< inputFname<<", "<<epsilon<<", "<<minPts<<", "<<NCHUNKS<<", "<<numClusters<<", "<<std::setprecision(4)<<((cntnoise*1.0)/(sizeDB*1.0))<<", DENSEBOX/MU/PARCHUNKS/NUMGPU/GPUSTREAMS/PARTITIONMODE/SCHEDULE/DTYPE(float/double): "<<DENSEBOXMODE<<", "<<MU<<", "<<PARCHUNKS<<", "<<NUMGPU<<", "<<GPUSTREAMS<< ", "<<PARTITIONMODE<<", "<<SCHEDULE<<", "<<STR(DTYPE)<< endl;
	gpu_stats.close();


	
	//Test printing clusters for visual inspection (not all clusters printed)
	#if PRINTCLUSTERS==1
	printclusters(&NDdataPoints, &finalClusterIDs, inputFname);
	#endif
	

	printf("\n\n\n");
	return 0;
}




int getGPU(int * GPUresources, omp_lock_t * gpuwritelock)
{

	int mygpu=-1;
	
	omp_set_lock(gpuwritelock);

		for (int i=0; i<NUMGPU; i++)
		{
			if (GPUresources[i]==0 && mygpu==-1)
			{
				mygpu=i;
				GPUresources[i]=1;
			}
		}

	omp_unset_lock(gpuwritelock);

	return mygpu;

}

void returnGPU(int * GPUresources, int gpuid, omp_lock_t * gpuwritelock)
{
		omp_set_lock(gpuwritelock);
		GPUresources[gpuid]=0;
		omp_unset_lock(gpuwritelock);		
	
}




//NDdataPoints- all data points for resolving merges across all partitions
//pointsIDsInShadowRegion- the global point ids in the shadow region
//epsilon- epsilon
//queryVect- shadow region queryVect
//clusterIDsShadow- the ids of the clusters in the shadow region [NOT NEEDED?]
//neighborTableLookup * neighborTable - neighbortable of the shadow region points
//minPts- minpts
//sizeDB- the number of data points, the others are reference points
//clusterIDsAcrossPartitions- the cluster ids in each partition (array of vectors)
//CHUNKS- the number of chunks/partitions
//pointChunkMapping- mapping of each point to its partition and idx in clusterIDsAcrossPartitions
//finalClusterIDs- final cluster assignments after all merging of clusters in partitions
void processNeighborTableForPartitions(std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<unsigned int> *pointsIDsInShadowRegion, 
	DTYPE epsilon, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDsShadow,  neighborTableLookup * neighborTable, 
	unsigned int minPts, unsigned long int sizeDB, std::vector<int> *clusterIDsAcrossPartitions, const unsigned int CHUNKS, 
	std::vector<struct pointChunkLookupArr> *pointChunkMapping, std::vector<int> * finalClusterIDs)
{

	//first, we create a vector of all of the cluster ids from the partitions
	//They correspond to the point ids in the global dataset
	std::vector<int> clusterIDsMergedFromPartitions;
	
	unsigned long int numPointsTotal=0;
	for (unsigned int i=0; i<CHUNKS; i++)
	{
		numPointsTotal+=clusterIDsAcrossPartitions[i].size();
	}	

	clusterIDsMergedFromPartitions.resize(numPointsTotal);
	printf("\nNum points across all partitions (sanity check, should be |D|): %lu",clusterIDsMergedFromPartitions.size());

	// unsigned int maxclusterId=0;
	// for (unsigned int i=0; i<CHUNKS; i++)
	// {
	// 	for (unsigned int j=0; j<clusterIDsAcrossPartitions[i].size(); j++)
	// 	{
	// 		XXX here
	// 		unsigned int idx=clusterIDsAcrossPartitions
	// 		pointChunkMapping[i]
	// 		clusterIDsMergedFromPartitions[]=clusterIDsAcrossPartitions[i][j];
	// 	}
	// 	//compute max here
	// }


	//find the maximum clusterID value in each chunk
	int maxClusterID[CHUNKS]; 

	for (unsigned int i=0; i<CHUNKS; i++){
		maxClusterID[i]=-2;
		for (unsigned int j=0; j<clusterIDsAcrossPartitions[i].size(); j++){
			if (clusterIDsAcrossPartitions[i][j]>maxClusterID[i])
			{
				maxClusterID[i]=clusterIDsAcrossPartitions[i][j];
			}
		}
		printf("\nMax cluster ids chunk: %u: %d ",i,maxClusterID[i]);
		
	}

	//compute clusterID offsets for assigning unique cluster ids
	int clusterIDOffsets[CHUNKS];
	clusterIDOffsets[0]=0;
	for (unsigned int i=1; i<CHUNKS; i++){
		clusterIDOffsets[i]=clusterIDOffsets[i-1]+maxClusterID[i-1]+1; //add 1 so there's no overlap
		printf("\nCluster id offsets for chunk: %u: %d ",i,clusterIDOffsets[i]);
	}

	
	
	//Map the partitioned cluster ids to globally mapped cluster ids
	for (unsigned int i=0; i<numPointsTotal; i++)
	{
		int chunk=(*pointChunkMapping)[i].chunkID;
		int idxInChunk=(*pointChunkMapping)[i].idxInChunk;
		int newClusterID=0;
		//if not noise, then we assign a new cluster id
		if(clusterIDsAcrossPartitions[chunk][idxInChunk]!=0)
		{
			newClusterID=clusterIDsAcrossPartitions[chunk][idxInChunk]+clusterIDOffsets[chunk];
		}
		//new cluster ID
		clusterIDsMergedFromPartitions[i]=newClusterID; 
	}

	//testing:
	// printf("\n****************");
	// for (unsigned int i=0; i<CHUNKS; i++){
	// 	printf("\nPartition: %u",i);
	// 	for (unsigned int j=0; j<clusterIDsAcrossPartitions[i].size(); j++){
	// 			printf("\n%d",clusterIDsAcrossPartitions[i][j]);
	// 	}
	// }
	// printf("\n****************");
	// for (unsigned int i=0; i<clusterIDsMergedFromPartitions.size(); i++){
	// 			printf("\n%u: %d",i,clusterIDsMergedFromPartitions[i]);
	// }	


	
	//
	
	
	printf("\nsize of clusterIDs merged from all partitions (sanity check, should be |D|): %lu",clusterIDsMergedFromPartitions.size());fflush(stdout);
	printf("\nsize of query vect for shadow regions (data points + ref points): %lu",queryVect->size());fflush(stdout);
	
	
	std::vector<unsigned int> queryVectRefPoints;
	for (unsigned int i=0; i<queryVect->size(); i++)
	{
		if((*queryVect)[i]>=sizeDB)
		{
			queryVectRefPoints.push_back((*queryVect)[i]);
		}
	}

	printf("\nSize vect ref points in shadow regions: %lu",queryVectRefPoints.size());



	// unsigned int tmp=0;
	
	//ADDED FOR PROCESSING NOISE POINTS IN SHADOW REGION
	//Noise points 
	std::vector<unsigned int> queryVectNoisePoints;
	

	for (unsigned int i=0; i<queryVect->size(); i++)
	{
		if (i<pointsIDsInShadowRegion->size() && (*queryVect)[i]<sizeDB)
		{
			unsigned int globalid=(*pointsIDsInShadowRegion)[i];
			if(clusterIDsMergedFromPartitions[globalid]==0)
			{
				queryVectNoisePoints.push_back(i); 
			}

		}
	}

	

	//vector of clusters that need to be merged
	//each vector is a list that need to be merged
	std::vector<std::vector<int> > mergeList;

	// unsigned int numNoise=0;
	// unsigned int numWithOnlyOneCluster=0;
	// unsigned int numBorderPointMultipleClusters=0;
	// unsigned int numThatNeedToMergeWithClusters=0;
	unsigned int numThatNeedToMergeRefPoints=0;
	std::vector<int> outputUniqueIds;

	std::vector<int> outputMergeIdsRefPnts;



	unsigned int cntRefPntsNonEmpty=0;
	unsigned int cntRefPnts=0;


	double tstartrefpts=omp_get_wtime();
	
	//reference points: 
	#pragma omp parallel for num_threads(NTHREADS) schedule(guided) shared(neighborTable, queryVectRefPoints, mergeList) private(outputUniqueIds, outputMergeIdsRefPnts) reduction(+:cntRefPnts,numThatNeedToMergeRefPoints, cntRefPntsNonEmpty)
	for (int i=0; i<queryVectRefPoints.size(); i++)
	{
		unsigned int idx=queryVectRefPoints[i];	
		
		outputUniqueIds.clear();

		cntRefPnts++;
				
		//check that the ref point has neighbors
		if (neighborTable[idx].indexmin!=0 && neighborTable[idx].indexmax!=0)
		{
			
			

			

			int numUnqiueClusters=getUniqueClustersForShadowRegions(idx, &queryVectRefPoints, &clusterIDsMergedFromPartitions, pointsIDsInShadowRegion, neighborTable, &outputUniqueIds);
			
			//if the number of unique clusters is >1
			if(numUnqiueClusters>1) 
			{
				outputMergeIdsRefPnts.clear();



				computeMergesRefPointDistanceCalculationsShadowRegion(NDdataPoints, epsilon, minPts, idx, &clusterIDsMergedFromPartitions, pointsIDsInShadowRegion, neighborTable, &outputUniqueIds, &outputMergeIdsRefPnts);

				if (outputMergeIdsRefPnts.size()>0)
				{
					//Need to merge the clusters in pairs
					//A single reference point can detect multiple merges, but which are separate from each other
					//I.e., 4 clusters which get merged into two distinct clusters
					//Therefore all of the merges don't get merged together
					
					std::vector<int> tmpoutputMergeIdsRefPntsPairs;

					for (int j=0; j<outputMergeIdsRefPnts.size(); j+=2)
					{
						tmpoutputMergeIdsRefPntsPairs.clear();
						tmpoutputMergeIdsRefPntsPairs.push_back(outputMergeIdsRefPnts[j]);
						tmpoutputMergeIdsRefPntsPairs.push_back(outputMergeIdsRefPnts[j+1]);
						
						#pragma omp critical
						{
						mergeList.push_back(tmpoutputMergeIdsRefPntsPairs);
						}
					}

					
					numThatNeedToMergeRefPoints++;
				}

			}	
			
			cntRefPntsNonEmpty++;			
			
		}
		

	} //end ref points
	

	double tendrefpts=omp_get_wtime();
	printf("\nReconcile partitions (time reference points): %f",tendrefpts - tstartrefpts);
	
	
	printf("\nNum of merges for shadow region clusters: %u",numThatNeedToMergeRefPoints);

	printf("\nNum of non-empty ref points: %u",cntRefPntsNonEmpty);


	


	
	//NOISE DATA POINTS
	//Points can be noise in two partitions but be part of a cluster
	
	unsigned int cntNoiseToCluster=0;
	unsigned int clusterIDNoiseStart=(*NDdataPoints)[0].size();
	unsigned int clusterIDNoise=(*NDdataPoints)[0].size();
	

	//visited array so that we don't revisit the same points
	//make the array the same size as the queryVect so that we can
	//just access by index
	std::vector<bool>queryVectNoisePointsVisited;
	queryVectNoisePointsVisited.resize(sizeDB);
	for (unsigned int i=0; i<queryVectNoisePointsVisited.size(); i++)
	{
		queryVectNoisePointsVisited[i]=false;
	}

	double tstartshadownoise=omp_get_wtime();
	printf("\nNum noise points in shadow region: %lu",queryVectNoisePoints.size());
	printf("\nStarting the cluster id for noise points (that may turn into clusters) at the dataset size: %u", clusterIDNoiseStart);

	#pragma omp parallel for num_threads(NTHREADS) schedule(guided) shared(clusterIDNoise,clusterIDsMergedFromPartitions,neighborTable, queryVectNoisePoints, mergeList, queryVectNoisePointsVisited, pointsIDsInShadowRegion) reduction(+:cntNoiseToCluster)
	for (int i=0; i<queryVectNoisePoints.size(); i++)
	{
		unsigned int idx=queryVectNoisePoints[i];	
		
		//check that the noise point has at least minpts neighbors

		unsigned int numNeighbors=(neighborTable[idx].indexmax-neighborTable[idx].indexmin)+1;

		if (numNeighbors>=minPts)
		{

			int clusterid;
			#pragma omp critical
			{
			clusterIDNoise++;
			clusterid=clusterIDNoise;
			}
						

			unsigned int globalid=(*pointsIDsInShadowRegion)[idx];	
			clusterIDsMergedFromPartitions[globalid]=clusterid;
			queryVectNoisePointsVisited[idx]=true;

			//Add all points to a cluster (this includes the point itself, since it will be in the neighbortable)
			for (int k=neighborTable[idx].indexmin; k<=neighborTable[idx].indexmax; k++)
			{
				//idx in shadow region
				unsigned int idx2=neighborTable[idx].dataPtr[k];
				unsigned int globalid2=(*pointsIDsInShadowRegion)[idx2];	

				//if the noise point has not been visited, then we assign it a cluster
				//check to make sure it's a noise point
				if (queryVectNoisePointsVisited[idx2]==false && clusterIDsMergedFromPartitions[globalid2]==0)
				{
				queryVectNoisePointsVisited[idx2]=true;
				unsigned int globalid=(*pointsIDsInShadowRegion)[idx2];	
				clusterIDsMergedFromPartitions[globalid]=clusterid;
				cntNoiseToCluster++;
				}
				//if the noise point has been visited, then we check to see if its cluster id is not the same
				//if they are in different clusters, then we merge
				//But only if it's not a cluster from a non-shadow region
				//These clusters should not be changed
				if (queryVectNoisePointsVisited[idx2]==true && 
					(clusterIDsMergedFromPartitions[globalid]!=clusterIDsMergedFromPartitions[globalid2])
					&& (clusterIDsMergedFromPartitions[globalid2]>clusterIDNoiseStart))
				{
					std::vector<int> tmpoutputMergeIdsNoisePntsPairs;
					tmpoutputMergeIdsNoisePntsPairs.push_back(clusterIDsMergedFromPartitions[globalid]);
					tmpoutputMergeIdsNoisePntsPairs.push_back(clusterIDsMergedFromPartitions[globalid2]);
					#pragma omp critical
					{
					mergeList.push_back(tmpoutputMergeIdsNoisePntsPairs);
					}

				}

				
			
			}	
			

								
					// std::vector<int> tmpoutputMergeIdsRefPntsPairs;

					// for (int j=0; j<outputMergeIdsRefPnts.size(); j+=2)
					// {
					// 	tmpoutputMergeIdsRefPntsPairs.clear();
					// 	tmpoutputMergeIdsRefPntsPairs.push_back(outputMergeIdsRefPnts[j]);
					// 	tmpoutputMergeIdsRefPntsPairs.push_back(outputMergeIdsRefPnts[j+1]);
						
					// 	#pragma omp critical
					// 	{
					// 	mergeList.push_back(tmpoutputMergeIdsRefPntsPairs);
					// 	}
					// }

					

		} 
		

	} //end noise points shadow region
	double tendshadownoise=omp_get_wtime();	

	printf("\nNumber of points noise -> cluster: %u", cntNoiseToCluster);
	printf("\nReconcile partitions (time to reclassify noise points in shadow region): %f", tendshadownoise - tstartshadownoise);





	
	
	
	

	


	std::set<int> uniqueClusterIdsAcrossAllMerges;
	for (int i=0; i<mergeList.size(); i++)
	{
		for (int j=0; j<mergeList[i].size(); j++)
		{
		uniqueClusterIdsAcrossAllMerges.insert(mergeList[i][j]);
		}
	}

	

	// printf("\nUnique cluster ids across all merges: %lu",uniqueClusterIdsAcrossAllMerges.size());
	//ENUMERATE ALL OF THE UNIQUE IDS AND THEN DISJOINT SET THEM TO MERGE
	std::vector<int> uniqueClusterIdsAcrossAllMergesVect;
	std::copy(uniqueClusterIdsAcrossAllMerges.begin(), uniqueClusterIdsAcrossAllMerges.end(), std::back_inserter(uniqueClusterIdsAcrossAllMergesVect));
	printf("\nUnique cluster ids across all merges: %lu",uniqueClusterIdsAcrossAllMergesVect.size());

	UF disjointsets(uniqueClusterIdsAcrossAllMergesVect.size());

	
	unsigned int cntMergePairs=0;

	for (int i=0; i<mergeList.size(); i++)
	{
		for (int j=1; j<mergeList[i].size(); j++)
		{
			//merge pairs of cluster ids
			//find the index of the cluster id 1
			auto it = std::lower_bound(uniqueClusterIdsAcrossAllMergesVect.begin(), uniqueClusterIdsAcrossAllMergesVect.end(),
			mergeList[i][j-1]);
			uint64_t idx = std::distance(uniqueClusterIdsAcrossAllMergesVect.begin(), it);

			//find the index of the cluster id 2
			auto it2 = std::lower_bound(uniqueClusterIdsAcrossAllMergesVect.begin(), uniqueClusterIdsAcrossAllMergesVect.end(),
			mergeList[i][j]);
			uint64_t idx2 = std::distance(uniqueClusterIdsAcrossAllMergesVect.begin(), it2);

			//merge in ds
			disjointsets.merge(idx,idx2);
			cntMergePairs++;

		}
	}


	printf("\nNum of pairs of clusters merged: %u",cntMergePairs);


	//need to scan all elements in disjoint set to make sure we have the correct mapping
	for (int i=0; i<uniqueClusterIdsAcrossAllMergesVect.size(); i++)
	{
	disjointsets.connected(i,0);	
	}


	//Now uniqueClusterIdsAcrossAllMergesVect contains a mapping of each cluster to a possibly new (merged) cluster
	//for all points
	//Mapping shown below
	//test print clusters and membership after merges
	// printf("\n**********************\n");
	// for (int i=0; i<uniqueClusterIdsAcrossAllMergesVect.size(); i++)
	// {
	// printf("\niter: %d, original cluster id: %d, merged cluster id: %d",i,uniqueClusterIdsAcrossAllMergesVect[i],uniqueClusterIdsAcrossAllMergesVect[disjointsets.id[i]]);	
	// }		
	// printf("\n**********************\n");

	//reassign all cluster IDs such that they are now merged

	


	
	double tstartclusterreassign=omp_get_wtime();
	

	#pragma omp parallel for num_threads(NTHREADS)
	// for (int i=0; i<sizeDB; i++)
	for (int i=0; i<(*NDdataPoints)[0].size(); i++)
	{
		//search to see if the cluster id is in the array
		auto it = std::lower_bound(uniqueClusterIdsAcrossAllMergesVect.begin(), uniqueClusterIdsAcrossAllMergesVect.end(),
			clusterIDsMergedFromPartitions[i]);


				
			if(!(it == uniqueClusterIdsAcrossAllMergesVect.end() || *it != clusterIDsMergedFromPartitions[i]))
			{
				//only update cluster if it's been assigned
				if (clusterIDsMergedFromPartitions[i]!=-1 && clusterIDsMergedFromPartitions[i]!=0)
				{
					uint64_t ind = std::distance(uniqueClusterIdsAcrossAllMergesVect.begin(), it);
					// printf("\npnt: %d, Original cluster id: %d, updated cluster id: %d",i,clusterIDsMergedFromPartitions[i], uniqueClusterIdsAcrossAllMergesVect[disjointsets.id[ind]]);				
				clusterIDsMergedFromPartitions[i]=uniqueClusterIdsAcrossAllMergesVect[disjointsets.id[ind]];	
				}


			}
			// else
			// 	{
			// 		// printf("\npnt: %d, Original cluster id: %d",i,(*clusterIDs)[i]);				
			// 	}

	}

	double tendclusterreassign=omp_get_wtime();
	printf("\nReconcile partitions (time to reassign clusters): %f", tendclusterreassign - tstartclusterreassign);
	
	

	for (unsigned int i=0; i<clusterIDsMergedFromPartitions.size(); i++)
	{
		finalClusterIDs->push_back(clusterIDsMergedFromPartitions[i]);	
	}

	printf("\nNum. elems final cluster ids: %lu",finalClusterIDs->size());

	
	
	

printf("\n\n");
return;

}

















void initNeighbortable(neighborTableLookup * neighborTable, unsigned int numElems)
{
	for (unsigned int i=0; i<numElems; i++)
	{
		neighborTable[i].pointID=0;
		neighborTable[i].indexmin=0;
		neighborTable[i].indexmax=0;
		neighborTable[i].dataPtr=NULL;
	}

}



void generateDBSCANQueryVect(std::vector<unsigned int> *queryVect, std::vector<unsigned int> *queryVectForNeighborTableDBSCAN, unsigned int sizeDB)
{
	for (int i=0; i<queryVect->size(); i++)
	{
		if((*queryVect)[i]<sizeDB)
		{
			queryVectForNeighborTableDBSCAN->push_back((*queryVect)[i]);
		}
	}

	printf("\nSize query points to DBSCANLeftovers: %lu",queryVectForNeighborTableDBSCAN->size());	
}

//append the dataset with reference points that can be used to merge the dense boxes
//updates NDdataPoints, queryVect, clusterIDs (add all of the reference points)
void generateReferencePoints(DTYPE * minArr, DTYPE epsilon, unsigned int * nCells, std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDs, struct gridCellLookup * gridCellLookupArr, unsigned int nNonEmptyCells)
{

	//one reference point at the center of each cell

	// printf("\n********\nFor testing: using all reference points, not just the non-empty cell ones\n********\n");
	
	unsigned long int initDBSize=(*NDdataPoints)[0].size();

	std::vector<DTYPE> tmppntthreads[NTHREADS];		
	
	std::vector<std::vector<DTYPE> > tmppntthreadsvect[NTHREADS];	
	
	
	
	unsigned long int numRefPts=0;

	#pragma omp parallel shared(tmppntthreads, tmppntthreadsvect) num_threads(NTHREADS)
	{
		int tid=omp_get_thread_num();
		#pragma omp for schedule(guided) reduction(+:numRefPts) 
		for (int i=0; i<nCells[0]; i++)
		{
			for (int j=0; j<nCells[1]; j++)
			{
					//Only populate reference points in non-empty cells

					DTYPE tmprefpoint[2];			
					tmprefpoint[0]=minArr[0]+(epsilon/2.0)+(i*epsilon);
					tmprefpoint[1]=minArr[1]+(epsilon/2.0)+(j*epsilon);

					unsigned int indexes[2];	
					indexes[0]=(tmprefpoint[0]-minArr[0])/epsilon;
					indexes[1]=(tmprefpoint[1]-minArr[1])/epsilon;
					uint64_t linearID=getLinearID_nDimensions(indexes,nCells,2);
					
					struct gridCellLookup tmp;
					tmp.gridLinearID=linearID;
		        	// struct gridCellLookup * resultBinSearch=;
		            // unsigned int GridIndex=resultBinSearch->idx;

					//if the point is in a non-empty cell
					if(std::binary_search(gridCellLookupArr, gridCellLookupArr+(nNonEmptyCells), gridCellLookup(tmp)))
					{

						//original single threaded
						// std::vector<DTYPE> tmppnt;	
						// tmppnt.push_back(tmprefpoint[0]);
						// tmppnt.push_back(tmprefpoint[1]);
						// queryVect->push_back(NDdataPoints->size()); //add the point to the query vect
						// NDdataPoints->push_back(tmppnt);
						// numRefPts++;

						//parallel: perform all insertions into vectors for each thread-- add to the query vector/data points
						//at the end
						
						tmppntthreads[tid].clear();
						tmppntthreads[tid].push_back(tmprefpoint[0]);
						tmppntthreads[tid].push_back(tmprefpoint[1]);
						tmppntthreadsvect[tid].push_back(tmppntthreads[tid]);
						numRefPts++;

					}
					
					
					

					//ORIGINAL WITH ALL REFERENCE POINTS
					// tmppnt.clear();	
					// tmppnt.push_back(minArr[0]+(epsilon/2.0)+(i*epsilon));
					// tmppnt.push_back(minArr[1]+(epsilon/2.0)+(j*epsilon));
					// queryVect->push_back(NDdataPoints->size()); //add the point to the query vect
					// NDdataPoints->push_back(tmppnt);
					// numRefPts++;
				

			}

		}
	}


	//add the temporary arrays generated by the threads to the  queryVect and NDdataPoints
	//ORIGINAL SEQUENTIAL VALIDATED

	// for (unsigned int i=0; i<NTHREADS; i++)
	// {
	// 	for (unsigned int j=0; j<tmppntthreadsvect[i].size(); j++)
	// 	{
	// 		queryVect->push_back(NDdataPoints->size()); //add the point to the query vect
	// 		NDdataPoints->push_back(tmppntthreadsvect[i][j]);
	// 	}	
	// }

	
	/////////////////////////////////////////////
	//Parallel copy

	//first get the total size of the reference points
	unsigned long int numRefPntsCnt=0;
	unsigned long int insertOffsets[NTHREADS];
	for (unsigned int i=0; i<NTHREADS; i++)
	{
		numRefPntsCnt+=tmppntthreadsvect[i].size();
		// printf("\nSize tmp threads vect: %lu", tmppntthreadsvect[i].size());
	}

	//get offsets for inserting into the data points vector
	//first offset is the number of data points
	insertOffsets[0]=initDBSize;
	for (unsigned int i=1; i<NTHREADS; i++)
	{
	insertOffsets[i]=insertOffsets[i-1]+tmppntthreadsvect[i-1].size();
	// printf("\nInsert offsets: %lu", insertOffsets[i]);
	}

	printf("\nNum ref pnts cnt: %lu", numRefPntsCnt);

	
	for (int i=0; i<GPUNUMDIM; i++)
	{
		(*NDdataPoints)[i].resize(initDBSize+numRefPntsCnt);	
	}
	

	//element wise update data points
	#pragma omp parallel for num_threads(4) schedule(dynamic) shared(NDdataPoints,insertOffsets,tmppntthreadsvect)
	for (unsigned int i=0; i<NTHREADS; i++)
	{
		for (unsigned int j=0; j<tmppntthreadsvect[i].size(); j++)
		{
			for (unsigned int k=0; k<GPUNUMDIM; k++)
			{
				(*NDdataPoints)[k][insertOffsets[i]+j]=tmppntthreadsvect[i][j][k];	
			}
			// (*NDdataPoints)[insertOffsets[i]+j]=tmppntthreadsvect[i][j];	
		}	
	}


	//element wise- sequential for the queryVect
	unsigned long int cntForQueryVect=initDBSize;
	for (unsigned int i=0; i<NTHREADS; i++)
	{
		for (unsigned int j=0; j<tmppntthreadsvect[i].size(); j++)
		{
			queryVect->push_back(cntForQueryVect); //add the point to the query vect
			cntForQueryVect++;
		}	
	}


	//end parallel copy
	/////////////////////////////////////////////
	

	//initialize cluster ids for the reference points
	for (int i=initDBSize; i<(*NDdataPoints)[0].size(); i++)
	{
		clusterIDs->push_back(-1);
	}


	printf("\nNum Ref points (ref points in non-empty cells only): %lu\n", numRefPts);
	

	



	//print ref points
	
	
	
	// printf("\nrefArr_x=[");
	// for (int i=initDBSize; i<NDdataPoints->size(); i++)
	// {
	// 	printf("%f,",(*NDdataPoints)[i][0]);
	// }
	// printf("]");

	// printf("\nrefArr_y=[");
	// for (int i=initDBSize; i<NDdataPoints->size(); i++)
	// {
	// 	printf("%f,",(*NDdataPoints)[i][1]);
	// }
	// printf("]");

	// printf("\nNum ref points: %lu", NDdataPoints->size()-initDBSize+1);
	

}



//append the dataset with reference points that can be used to merge the dense boxes
//updates NDdataPoints, queryVect, clusterIDs (add all of the reference points)
//Only for the shadow region --- this is so that we avoid populating the entire data space of the dataset and then finding the reference points
void generateReferencePointsShadowRegionOnly(DTYPE * minArr, DTYPE epsilon, unsigned int * nCells, std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDs, struct gridCellLookup * gridCellLookupArr, unsigned int nNonEmptyCells)
{

	//one reference point at the center of each cell

	// printf("\n********\nFor testing: using all reference points, not just the non-empty cell ones\n********\n");
	
	unsigned long int initDBSize=(*NDdataPoints)[0].size();

	std::vector<DTYPE> tmppntthreads[NTHREADS];		
	


	
	std::vector<std::vector<DTYPE> > tmppntthreadsvect[NTHREADS];	
	
	
	unsigned long int numRefPts=0;

	/*

	#pragma omp parallel shared(tmppntthreads, tmppntthreadsvect) num_threads(NTHREADS)
	{
		int tid=omp_get_thread_num();
		#pragma omp for schedule(guided) reduction(+:numRefPts) 
		for (int i=0; i<nCells[0]; i++)
		{
			for (int j=0; j<nCells[1]; j++)
			{
					//Only populate reference points in non-empty cells

					DTYPE tmprefpoint[2];			
					tmprefpoint[0]=minArr[0]+(epsilon/2.0)+(i*epsilon);
					tmprefpoint[1]=minArr[1]+(epsilon/2.0)+(j*epsilon);

					unsigned int indexes[2];	
					indexes[0]=(tmprefpoint[0]-minArr[0])/epsilon;
					indexes[1]=(tmprefpoint[1]-minArr[1])/epsilon;
					uint64_t linearID=getLinearID_nDimensions(indexes,nCells,2);
					
					struct gridCellLookup tmp;
					tmp.gridLinearID=linearID;
		        	// struct gridCellLookup * resultBinSearch=;
		            // unsigned int GridIndex=resultBinSearch->idx;

					//if the point is in a non-empty cell
					if(std::binary_search(gridCellLookupArr, gridCellLookupArr+(nNonEmptyCells), gridCellLookup(tmp)))
					{

						//original single threaded
						// std::vector<DTYPE> tmppnt;	
						// tmppnt.push_back(tmprefpoint[0]);
						// tmppnt.push_back(tmprefpoint[1]);
						// queryVect->push_back(NDdataPoints->size()); //add the point to the query vect
						// NDdataPoints->push_back(tmppnt);
						// numRefPts++;

						//parallel: perform all insertions into vectors for each thread-- add to the query vector/data points
						//at the end
						
						tmppntthreads[tid].clear();
						tmppntthreads[tid].push_back(tmprefpoint[0]);
						tmppntthreads[tid].push_back(tmprefpoint[1]);
						tmppntthreadsvect[tid].push_back(tmppntthreads[tid]);
						numRefPts++;

					}
					
					
					

					//ORIGINAL WITH ALL REFERENCE POINTS
					// tmppnt.clear();	
					// tmppnt.push_back(minArr[0]+(epsilon/2.0)+(i*epsilon));
					// tmppnt.push_back(minArr[1]+(epsilon/2.0)+(j*epsilon));
					// queryVect->push_back(NDdataPoints->size()); //add the point to the query vect
					// NDdataPoints->push_back(tmppnt);
					// numRefPts++;
				

			}

		}
	}

		
	*/	

	
	//New -- use shadow region points to determine the reference points
	//Need to remove duplicates
	/*
	
	int tid=omp_get_thread_num();
	//iterate over all shadow region points
	for (unsigned long int i=0; i<initDBSize; i++)
	{
		//Get the 2-D coords of each shadow region point's cell
		unsigned int indexes[2];	
		indexes[0]=((*NDdataPoints)[0][i]-minArr[0])/epsilon;
		indexes[1]=((*NDdataPoints)[1][i]-minArr[1])/epsilon;

		DTYPE tmprefpoint[2];			
		tmprefpoint[0]=minArr[0]+(epsilon/2.0)+(indexes[0]*epsilon);
		tmprefpoint[1]=minArr[1]+(epsilon/2.0)+(indexes[1]*epsilon);

		tmppntthreads[tid].clear();
		tmppntthreads[tid].push_back(tmprefpoint[0]);
		tmppntthreads[tid].push_back(tmprefpoint[1]);
		tmppntthreadsvect[tid].push_back(tmppntthreads[tid]);
		numRefPts++;
	}
	*/

	//New -- use shadow region points to determine the reference points
	//Fix the new generation of reference points because it creates duplicates


	

	cellCoords * cellCoordsToFilter=new cellCoords[initDBSize];

	for (unsigned long int i=0; i<initDBSize; i++)
	{
		//Get the 2-D coords of each shadow region point's cell
		// unsigned int indexes[2];	
		cellCoordsToFilter[i].dim1=((*NDdataPoints)[0][i]-minArr[0])/epsilon;
		cellCoordsToFilter[i].dim2=((*NDdataPoints)[1][i]-minArr[1])/epsilon;
		
		// DTYPE tmprefpoint[2];			
		// refPointsToFilter[i].dim1=minArr[0]+(epsilon/2.0)+(indexes[0]*epsilon);
		// refPointsToFilter[i].dim2=minArr[1]+(epsilon/2.0)+(indexes[1]*epsilon);

		// tmppntthreads[tid].clear();
		// tmppntthreads[tid].push_back(tmprefpoint[0]);
		// tmppntthreads[tid].push_back(tmprefpoint[1]);
		// tmppntthreadsvect[tid].push_back(tmppntthreads[tid]);
		// numRefPts++;
	}

	//sort for filtering unique
	std::sort(cellCoordsToFilter, cellCoordsToFilter+initDBSize, compareLexicographicCellCoords);

	// printf("\n************\nSorted cell coords: ");
	// for (unsigned long int i=0; i<initDBSize; i++)
	// {
	// 	printf("\n%u, %u", cellCoordsToFilter[i].dim1,cellCoordsToFilter[i].dim2);
	// }

	//Unique the cell coords

	std::vector<unsigned int> uniqueCellsDim1;
	std::vector<unsigned int> uniqueCellsDim2;
	
	uniqueCellsDim1.push_back(cellCoordsToFilter[0].dim1);
	uniqueCellsDim2.push_back(cellCoordsToFilter[0].dim2);

	for (unsigned long int i=1; i<initDBSize; i++)
	{
		unsigned long int idx = uniqueCellsDim1.size(); 

		//if different on the first coord, then add to unique cells
		if (uniqueCellsDim1[idx-1]!=cellCoordsToFilter[i].dim1)
		{
			uniqueCellsDim1.push_back(cellCoordsToFilter[i].dim1);
			uniqueCellsDim2.push_back(cellCoordsToFilter[i].dim2);
		}
		//if equal first coord, check second coord
		else if (uniqueCellsDim2[idx-1]!=cellCoordsToFilter[i].dim2)
		{
			uniqueCellsDim1.push_back(cellCoordsToFilter[i].dim1);
			uniqueCellsDim2.push_back(cellCoordsToFilter[i].dim2);
		}
	}



	int tid=omp_get_thread_num();

	//copy the unique reference points to the output array
	for (unsigned long int i=0; i<uniqueCellsDim1.size(); i++)
	{
		
		

		DTYPE tmprefpoint[2];			
		tmprefpoint[0]=minArr[0]+(epsilon/2.0)+(uniqueCellsDim1[i]*epsilon);
		tmprefpoint[1]=minArr[1]+(epsilon/2.0)+(uniqueCellsDim2[i]*epsilon);

		tmppntthreads[tid].clear();
		tmppntthreads[tid].push_back(tmprefpoint[0]);
		tmppntthreads[tid].push_back(tmprefpoint[1]);
		tmppntthreadsvect[tid].push_back(tmppntthreads[tid]);
		numRefPts++;
	}


	// printf("\n************\nUnqiue cell coords: ");
	// for (unsigned long int i=0; i<uniqueCellsDim1.size(); i++)
	// {
	// 	printf("\n%u, %u", uniqueCellsDim1[i],uniqueCellsDim2[i]);
	// }





	//add the temporary arrays generated by the threads to the  queryVect and NDdataPoints
	//ORIGINAL SEQUENTIAL VALIDATED

	// for (unsigned int i=0; i<NTHREADS; i++)
	// {
	// 	for (unsigned int j=0; j<tmppntthreadsvect[i].size(); j++)
	// 	{
	// 		queryVect->push_back(NDdataPoints->size()); //add the point to the query vect
	// 		NDdataPoints->push_back(tmppntthreadsvect[i][j]);
	// 	}	
	// }

	
	/////////////////////////////////////////////
	//Parallel copy

	//first get the total size of the reference points
	unsigned long int numRefPntsCnt=0;
	unsigned long int insertOffsets[NTHREADS];
	for (unsigned int i=0; i<NTHREADS; i++)
	{
		numRefPntsCnt+=tmppntthreadsvect[i].size();
		// printf("\nSize tmp threads vect: %lu", tmppntthreadsvect[i].size());
	}

	//get offsets for inserting into the data points vector
	//first offset is the number of data points
	insertOffsets[0]=initDBSize;
	for (unsigned int i=1; i<NTHREADS; i++)
	{
	insertOffsets[i]=insertOffsets[i-1]+tmppntthreadsvect[i-1].size();
	// printf("\nInsert offsets: %lu", insertOffsets[i]);
	}

	printf("\nNum ref pnts cnt: %lu", numRefPntsCnt);

	
	for (int i=0; i<GPUNUMDIM; i++)
	{
		(*NDdataPoints)[i].resize(initDBSize+numRefPntsCnt);	
	}
	

	//element wise update data points
	#pragma omp parallel for num_threads(4) schedule(dynamic) shared(NDdataPoints,insertOffsets,tmppntthreadsvect)
	for (unsigned int i=0; i<NTHREADS; i++)
	{
		for (unsigned int j=0; j<tmppntthreadsvect[i].size(); j++)
		{
			for (unsigned int k=0; k<GPUNUMDIM; k++)
			{
				(*NDdataPoints)[k][insertOffsets[i]+j]=tmppntthreadsvect[i][j][k];	
			}
			// (*NDdataPoints)[insertOffsets[i]+j]=tmppntthreadsvect[i][j];	
		}	
	}


	//element wise- sequential for the queryVect
	unsigned long int cntForQueryVect=initDBSize;
	for (unsigned int i=0; i<NTHREADS; i++)
	{
		for (unsigned int j=0; j<tmppntthreadsvect[i].size(); j++)
		{
			queryVect->push_back(cntForQueryVect); //add the point to the query vect
			cntForQueryVect++;
		}	
	}


	//end parallel copy
	/////////////////////////////////////////////
	

	//initialize cluster ids for the reference points
	for (int i=initDBSize; i<(*NDdataPoints)[0].size(); i++)
	{
		clusterIDs->push_back(-1);
	}


	printf("\nNum Ref points (ref points in non-empty cells only): %lu\n", numRefPts);
	

	



	//print ref points
	
	
	
	// printf("\nrefArr_x=[");
	// for (int i=initDBSize; i<NDdataPoints->size(); i++)
	// {
	// 	printf("%f,",(*NDdataPoints)[i][0]);
	// }
	// printf("]");

	// printf("\nrefArr_y=[");
	// for (int i=initDBSize; i<NDdataPoints->size(); i++)
	// {
	// 	printf("%f,",(*NDdataPoints)[i][1]);
	// }
	// printf("]");

	// printf("\nNum ref points: %lu", NDdataPoints->size()-initDBSize+1);
	

}


//cluster ids are from the dense box algorithm -- cluster ids for all points
//query vect- the queries resolved using the neighbortable on the GPU, which need to be resolved: assigned a preexisting cluster from densebox, assigned as noise, merge two or more dense box clusters

void processNeighborTableForClusters(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, std::vector<unsigned int> *queryVect, 
	std::vector<int> *clusterIDs, neighborTableLookup * neighborTable, unsigned int minPts, unsigned long int sizeDB, std::vector<unsigned int> *queryVectForNeighborTableDBSCAN, std::vector<bool> *denseBoxPoints)
{



	queryVectForNeighborTableDBSCAN->clear();
	

	printf("\nsize of clusterIDs: %lu",clusterIDs->size());
	printf("\nsize of query vect: %lu",queryVect->size());
	
	
	//vector storing the ids of points that cannot be merged with any of the densebox points, and are not noise
	//need to DBSCAN these ones with the neighbortable (like previous paper)



	std::vector<unsigned int> queryVectRefPoints;
	for (int i=0; i<queryVect->size(); i++)
	{
		if((*queryVect)[i]>=sizeDB)
		{
			queryVectRefPoints.push_back((*queryVect)[i]);
		}
	}

	printf("\nSize vect ref points: %lu",queryVectRefPoints.size());
	
	

	// return;	

	//vector of clusters that need to be merged
	//each vector is a list that need to be merged
	std::vector<std::vector<int> > mergeList;

	unsigned int numNoise=0;
	unsigned int numWithOnlyOneCluster=0;
	unsigned int numBorderPointMultipleClusters=0;
	unsigned int numThatNeedToMergeWithClusters=0;
	unsigned int numThatNeedToMergeRefPoints=0;
	std::vector<int> outputUniqueIds;

	std::vector<int> outputMergeIdsRefPnts;



	unsigned int cntRefPntsNonEmpty=0;
	unsigned int cntRefPnts=0;


	
	
	//reference points: 
	#pragma omp parallel for num_threads(NTHREADS) schedule(guided) shared(neighborTable, queryVectRefPoints, mergeList) private(outputUniqueIds, outputMergeIdsRefPnts) reduction(+:cntRefPnts,numThatNeedToMergeRefPoints, cntRefPntsNonEmpty)
	for (int i=0; i<queryVectRefPoints.size(); i++)
	{
		unsigned int idx=queryVectRefPoints[i];	
		
		outputUniqueIds.clear();

		cntRefPnts++;
				
		//check that the ref point has neighbors
		if (neighborTable[idx].indexmin!=0 && neighborTable[idx].indexmax!=0)
		{
			
			

			int numUnqiueClusters=getUniqueClusters(idx, &queryVectRefPoints, clusterIDs, neighborTable, &outputUniqueIds);

			//new, if the number of unique clusters is >1
			if(numUnqiueClusters>1) 
			{
				outputMergeIdsRefPnts.clear();
				computeMergesRefPointDistanceCalculations(NDdataPoints, epsilon, idx, &queryVectRefPoints, clusterIDs, neighborTable, &outputUniqueIds,denseBoxPoints, &outputMergeIdsRefPnts);

				if (outputMergeIdsRefPnts.size()>0)
				{
					//Need to merge the clusters in pairs
					//A single reference point can detect multiple merges, but which are separate from each other
					//I.e., 4 clusters which get merged into two distinct clusters
					//Therefore all of the merges don't get merged together
					
					std::vector<int> tmpoutputMergeIdsRefPntsPairs;

					for (int j=0; j<outputMergeIdsRefPnts.size(); j+=2)
					{
						tmpoutputMergeIdsRefPntsPairs.clear();
						tmpoutputMergeIdsRefPntsPairs.push_back(outputMergeIdsRefPnts[j]);
						tmpoutputMergeIdsRefPntsPairs.push_back(outputMergeIdsRefPnts[j+1]);
						
						#pragma omp critical
						{
						mergeList.push_back(tmpoutputMergeIdsRefPntsPairs);
						}
					}

					
					numThatNeedToMergeRefPoints++;
				}
			}	
			
			cntRefPntsNonEmpty++;			
			
		} //END ELSE REFERENCE POINTS
		

	} //end ref points
	
	
	

	printf("\nNumber of noise points: %u",numNoise);
	printf("\nNumber with only 1 unique cluster: %u",numWithOnlyOneCluster);
	printf("\nNumber with >1 unique cluster, but fewer than minpts (border point for one of the clusters): %u",numBorderPointMultipleClusters);
	printf("\nNumber that need to be clustered using DBSCAN that takes as input the neighbor table: %lu", queryVectForNeighborTableDBSCAN->size());
	printf("\nNumber that merge two or more clusters: %u", numThatNeedToMergeWithClusters);
	printf("\nNumber that merge two or more clusters because of reference points: %u", numThatNeedToMergeRefPoints);

	printf("\nNum of ref points: %u",cntRefPnts);
	printf("\nNum of non-empty ref points: %u",cntRefPntsNonEmpty);


	unsigned int totalAcrossAllCases=numNoise+numWithOnlyOneCluster+numBorderPointMultipleClusters+(unsigned int)queryVectForNeighborTableDBSCAN->size()+numThatNeedToMergeWithClusters+cntRefPnts;
	printf("\nTotal query points across all cases: %u",totalAcrossAllCases);

	printf("\nChecking that the total number of query points is equal to the number of points in the query points vector.\nThere isn't an else statement, so if the cases don't capture all possibilities, then we've forgotten something");
	
	
	// assert(totalAcrossAllCases==queryVect->size());

	/*	
	int cntnoise=0;
	int cntunassigned=0;
	for (int i=0; i<sizeDB; i++)
	{
		if ((*clusterIDs)[i]==0)
		cntnoise++;

		if ((*clusterIDs)[i]==-1)
			cntunassigned++;

	}

		printf("\nNum noise (check): %d",cntnoise);
		printf("\nNum unassigned (check): %d",cntunassigned);

	*/

	//Need to merge clusters:

	//testing merge list
	/*
	for (int i=0; i<mergeList.size(); i++)
	{
		printf("\nList %d: ", i);
		for (int j=0; j<mergeList[i].size(); j++)
		{
			printf("%d, ", mergeList[i][j]);	
		}
	}
	*/




	std::set<int> uniqueClusterIdsAcrossAllMerges;
	for (int i=0; i<mergeList.size(); i++)
	{
		for (int j=0; j<mergeList[i].size(); j++)
		{
		uniqueClusterIdsAcrossAllMerges.insert(mergeList[i][j]);
		}
	}

	

	// printf("\nUnique cluster ids across all merges: %lu",uniqueClusterIdsAcrossAllMerges.size());
	//ENUMERATE ALL OF THE UNIQUE IDS AND THEN DISJOINT SET THEM TO MERGE
	std::vector<int> uniqueClusterIdsAcrossAllMergesVect;
	std::copy(uniqueClusterIdsAcrossAllMerges.begin(), uniqueClusterIdsAcrossAllMerges.end(), std::back_inserter(uniqueClusterIdsAcrossAllMergesVect));
	printf("\nUnique cluster ids across all merges: %lu",uniqueClusterIdsAcrossAllMergesVect.size());

	UF disjointsets(uniqueClusterIdsAcrossAllMergesVect.size());

	

	for (int i=0; i<mergeList.size(); i++)
	{
		for (int j=1; j<mergeList[i].size(); j++)
		{
			//merge pairs of cluster ids
			//find the index of the cluster id 1
			auto it = std::lower_bound(uniqueClusterIdsAcrossAllMergesVect.begin(), uniqueClusterIdsAcrossAllMergesVect.end(),
			mergeList[i][j-1]);
			uint64_t idx = std::distance(uniqueClusterIdsAcrossAllMergesVect.begin(), it);

			//find the index of the cluster id 2
			auto it2 = std::lower_bound(uniqueClusterIdsAcrossAllMergesVect.begin(), uniqueClusterIdsAcrossAllMergesVect.end(),
			mergeList[i][j]);
			uint64_t idx2 = std::distance(uniqueClusterIdsAcrossAllMergesVect.begin(), it2);

			//merge in ds
			disjointsets.merge(idx,idx2);

		}
	}



	//need to scan all elements in disjoint set to make sure we have the correct mapping
	for (int i=0; i<uniqueClusterIdsAcrossAllMergesVect.size(); i++)
	{
	disjointsets.connected(i,0);	
	}


	//Now uniqueClusterIdsAcrossAllMergesVect contains a mapping of each cluster to a possibly new (merged) cluster
	//for all points
	//Mapping shown below
	//test print clusters and membership after merges
	// printf("\n**********************\n");
	// for (int i=0; i<uniqueClusterIdsAcrossAllMergesVect.size(); i++)
	// {
	// printf("\niter: %d, original cluster id: %d, merged cluster id: %d",i,uniqueClusterIdsAcrossAllMergesVect[i],uniqueClusterIdsAcrossAllMergesVect[disjointsets.id[i]]);	
	// }		
	// printf("\n**********************\n");

	//reassign all cluster IDs such that they are now merged

	

	#pragma omp parallel for num_threads(NTHREADS)
	for (int i=0; i<sizeDB; i++)
	{
		//search to see if the cluster id is in the array
		auto it = std::lower_bound(uniqueClusterIdsAcrossAllMergesVect.begin(), uniqueClusterIdsAcrossAllMergesVect.end(),
			(*clusterIDs)[i]);


			// if(*it == (*clusterIDs)[i])
			if(!(it == uniqueClusterIdsAcrossAllMergesVect.end() || *it != (*clusterIDs)[i]))
			{
				//only update cluster if it's been assigned
				if ((*clusterIDs)[i]!=-1 && (*clusterIDs)[i]!=0)
				{
					uint64_t ind = std::distance(uniqueClusterIdsAcrossAllMergesVect.begin(), it);
					// printf("\npnt: %d, Original cluster id: %d, updated cluster id: %d",i,(*clusterIDs)[i], uniqueClusterIdsAcrossAllMergesVect[disjointsets.id[ind]]);				
				(*clusterIDs)[i]=uniqueClusterIdsAcrossAllMergesVect[disjointsets.id[ind]];	
				}


			}
			// else
			// 	{
			// 		// printf("\npnt: %d, Original cluster id: %d",i,(*clusterIDs)[i]);				
			// 	}

	}

	

	
	

printf("\n\n");

}

unsigned int getNumClusters(std::vector<int> *clusterIDs)
{
	std::set<int> setOfClusterIds;

	for (int i=0; i<clusterIDs->size(); i++)
	{
		setOfClusterIds.insert((*clusterIDs)[i]);
	}

	return (setOfClusterIds.size());

}



void printclusters(std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<int> *clusterIDs, char * fname)
{



	//testing: printing clusters only if the total number of points in the cluster are within bounds

	unsigned int lbound=100;
	unsigned int ubound=1700000000;

	//print every sample_points points in the cluster
	unsigned int sample_points=10;

	//Print only data points with certain coordinates: e.g., point the galactic center 
	//set FILTERCOORDS to true
	bool FILTERCOORDS=true;
	DTYPE x_min=0;
	DTYPE x_max=360;
	DTYPE y_min=-25;
	DTYPE y_max=25;


		
	// unsigned int lbound=100;
	// unsigned int ubound=1700000000;

	// char fname[]="gpu_stats.txt";
	
	// sprintf( lower_bound, "%u", lbound);

	

	


	std::set<int> setOfClusterIds;

	for (int i=0; i<clusterIDs->size(); i++)
	{
		setOfClusterIds.insert((*clusterIDs)[i]);
	}

	printf("\nPrinting %lu clusters\n**********\n",setOfClusterIds.size());





	//output to file:
	char output_fname[200];
	sprintf( output_fname, "Max_clusters_%lu_lbound_%u_ubound_%u.py", setOfClusterIds.size(),lbound,ubound);
	printf("\noutput name: %s",output_fname);

	ofstream plot;
	plot.open(output_fname,ios::out);			
	

	// printf("#!/usr/bin/python\n");
	// printf("import numpy as np\n");
	// printf("import matplotlib.pyplot as plt\n");
	// printf("import csv\n");
	// printf("from matplotlib.ticker import FormatStrFormatter\n");
	// printf("fig = plt.figure(figsize=(20,16))\n");
	// printf("ax1 = fig.add_subplot(111)\n");
	// printf("ax1.tick_params(which='major', labelsize=14)\n");

	

	plot<<"#!/usr/bin/python\n";
	plot<<"import numpy as np\n";
	plot<<"import matplotlib.pyplot as plt\n";
	plot<<"import csv\n";
	plot<<"import itertools\n";
	plot<<"marker = itertools.cycle((',', '.','*'))\n";
	plot<<"from matplotlib.ticker import FormatStrFormatter\n";
	plot<<"fig = plt.figure(figsize=(30,15))\n";
	plot<<"ax1 = fig.add_subplot(111)\n";
	plot<<"ax1.tick_params(which='major', labelsize=14)\n";

	plot<<"#Sample points: "<<sample_points<<"\n";

	std::vector<int> setOfClusterIdsVect;

	std::copy(setOfClusterIds.begin(), setOfClusterIds.end(), std::back_inserter(setOfClusterIdsVect));	

	// std::vector<DTYPE> tmp_clusterx;
	// std::vector<DTYPE> tmp_clustery;


	//Create vectors of point ids for each cluster 
	//The cluster ids have not been relabeled, so we need to find the maximum cluster id, and then allocate that many vectors (most won't be used)

	// std::cout << "The largest element is "  << *std::max_element(myints,myints+7) << '\n';
	int maxClusterID=*std::max_element(setOfClusterIdsVect.begin(),setOfClusterIdsVect.end());
	// printf("\nMax cluster id: %d",maxClusterID);

	std::vector<int> * pointIdsInCluster=new std::vector<int>[maxClusterID];



	//Do one scan through the cluster ids and assign to the correct vector:


	// for (unsigned int i=0; i<clusterIDs->size(); i++)
	// {
	// 	printf("\ncluster id: %d",(*clusterIDs)[i]);
	// }

	

	for (unsigned int i=0; i<clusterIDs->size(); i++)
	{
		int clusterId=(*clusterIDs)[i];
		pointIdsInCluster[clusterId].push_back(i);
	}



	// printf("\nCluster id: %d",setOfClusterIdsVect[0]);

	// return;
	//Scan for cluster ids, could make this more efficient, but it's for debugging 
		

	//start at 1 because we do not want the noise points to be plotted	
	for (int i=1; i<setOfClusterIdsVect.size(); i++)
	{

		//if the cluster has enough points within the bounds, then we print that cluster
		if (pointIdsInCluster[i].size()>=lbound && pointIdsInCluster[i].size()<=ubound)
		{

			plot<<"Cluster"<<i<<"_x=[";
			for (unsigned int j=0; j<pointIdsInCluster[i].size(); j+=sample_points)
			{
				unsigned int pointId=pointIdsInCluster[i][j];
				//need to filter on both x and y so that it doesn't print the y later when the x coord isn't printed
				if (FILTERCOORDS==true && (*NDdataPoints)[0][pointId]>=x_min && (*NDdataPoints)[0][pointId]<=x_max && (*NDdataPoints)[1][pointId]>=y_min && (*NDdataPoints)[1][pointId]<=y_max)
					plot<<(*NDdataPoints)[0][pointId]<<",";	
				else if (FILTERCOORDS==false)
				{
					plot<<(*NDdataPoints)[0][pointId]<<",";
				}
			}
			plot<<"]\n";

			plot<<"Cluster"<<i<<"_y=[";

			for (unsigned int j=0; j<pointIdsInCluster[i].size(); j+=sample_points)
			{
				unsigned int pointId=pointIdsInCluster[i][j];
				if (FILTERCOORDS==true && (*NDdataPoints)[0][pointId]>=x_min && (*NDdataPoints)[0][pointId]<=x_max && (*NDdataPoints)[1][pointId]>=y_min && (*NDdataPoints)[1][pointId]<=y_max)
					plot<<(*NDdataPoints)[1][pointId]<<",";	
				else if (FILTERCOORDS==false)	
				{
					plot<<(*NDdataPoints)[1][pointId]<<",";
				}
			}
			plot<<"]\n";

			plot<<"ax1.scatter(Cluster"<<i<<"_x, Cluster"<<i<<"_y,     marker=marker.next(), s=5, c=np.random.rand(3,), linewidth=0, label='')\n";

		}

		
		/*
		tmp_clusterx.clear();
		tmp_clustery.clear();
		
		

		for (int j=0; j<clusterIDs->size(); j++)	
		{
			if ((*clusterIDs)[j]==setOfClusterIdsVect[i])
			{
				tmp_clusterx.push_back((*NDdataPoints)[0][j]);
				tmp_clustery.push_back((*NDdataPoints)[1][j]);
			}
		}
		
		

		if (tmp_clusterx.size()>=lbound && tmp_clusterx.size()<=ubound)
		{	
			// printf("Cluster%d_x=[",i);
			plot<<"Cluster"<<i<<"_x=[";
			for (int k=0; k<tmp_clusterx.size(); k+=sample_points)
			{
				// printf("%f,",tmp_clusterx[k]);
				plot<<tmp_clusterx[k]<<",";
			}
			// printf("]\n");
			plot<<"]\n";

			// printf("Cluster%d_y=[",i);
			plot<<"Cluster"<<i<<"_y=[";
			for (int k=0; k<tmp_clustery.size(); k+=sample_points)
			{
				// printf("%f,",tmp_clustery[k]);
				plot<<tmp_clustery[k]<<",";
			}
			// printf("]\n");
			plot<<"]\n";

			// printf("ax1.scatter(Cluster%d_x, Cluster%d_y,     marker='.', s=5, c=np.random.rand(3,), linewidth=0, label='')\n",i,i);
			plot<<"ax1.scatter(Cluster"<<i<<"_x, Cluster"<<i<<"_y,     marker='.', s=5, c=np.random.rand(3,), linewidth=0, label='')\n";
		}

		*/
	}

	// printf("print \"Saving figure: test_dense_box.pdf\"\n");
	// printf("fig.savefig(\"test_dense_box.pdf\", bbox_inches='tight')\n");

	plot<<"print \"Saving figure: "<<output_fname<<".pdf\"\n";
	plot<<"fig.savefig(\""<<output_fname<<".pdf\", bbox_inches='tight')\n";


	plot.close();
	
}	


//Takes the neighbors of the reference points and returns a list of clusters to merge
//if they meet the merge criteria (core point overlap between two clusters)
//detect this using distance calulations, but once a merge is detected between a pair of clusters
//abort the distance calculations 

//Make sure that we merge between densebox-densebox points, densebox-dbscan points, NOT dbscan-densebox

int computeMergesRefPointDistanceCalculations(std::vector<std::vector <DTYPE> > *NDdataPoints, double epsilon, unsigned int queryId, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDs, neighborTableLookup * neighborTable, std::vector<int> * uniqueIds, std::vector<bool> *denseBoxPoints, std::vector<int> * outputMergeIdsRefPnts)
{

	// unsigned int pointToTest1=0;
	// unsigned int pointToTest2=0;

	


	//create a temp vector array with the points within each cluster
	std::vector<int> clusterIdsPoints[uniqueIds->size()];

	for (int k=neighborTable[queryId].indexmin; k<=neighborTable[queryId].indexmax; k++)
	{
		unsigned int pointToInsert=neighborTable[queryId].dataPtr[k];

		//make sure it isn't a noise point or an unassigned point
		if ((*clusterIDs)[pointToInsert]!=-1 && (*clusterIDs)[pointToInsert]!=0 )
		{
			for (int l=0; l<uniqueIds->size(); l++)
			{
				
				if ((*clusterIDs)[pointToInsert]==(*uniqueIds)[l])
				{
					clusterIdsPoints[l].push_back(pointToInsert);
					break;
				}
			}
		}
	}


	//testing printing the clusters and points
	// for (int i=0; i<uniqueIds->size(); i++)
	// {
	// 	printf("\nCluster id: %d, data pnts: ",(*uniqueIds)[i]);
	// 	for (int j=0; j<clusterIdsPoints[i].size(); j++)
	// 	{
	// 	printf("%d, ",clusterIdsPoints[i][j]);
	// 	}
	// }





	unsigned int num_iter=0;
	unsigned int num_merges=0;
	unsigned int max_merges=0;

	// bool flag1=0;
	// bool flag2=0;
	


	//loop over pairs of cluster ids 
	for (int i=0; i<uniqueIds->size(); i++)
	{
		for (int j=i+1; j<uniqueIds->size(); j++)
		{
			// flag1=0;
			// flag2=0;
			
			max_merges++;
			

			for (int k=0; k<clusterIdsPoints[i].size(); k++){
				unsigned int pointIdxToTest1=clusterIdsPoints[i][k];

				 


				for (int l=0; l<clusterIdsPoints[j].size(); l++){
					unsigned int pointIdxToTest2=clusterIdsPoints[j][l];

					//make sure one of the points was originally a densebox point:
					if ((*denseBoxPoints)[pointIdxToTest1]==true || (*denseBoxPoints)[pointIdxToTest2]==true )
					{

						//distance calculation
						// double dist1=((*NDdataPoints)[pointIdxToTest1][0]-(*NDdataPoints)[pointIdxToTest2][0])*((*NDdataPoints)[pointIdxToTest1][0]-(*NDdataPoints)[pointIdxToTest2][0]);
						// double dist2=((*NDdataPoints)[pointIdxToTest1][1]-(*NDdataPoints)[pointIdxToTest2][1])*((*NDdataPoints)[pointIdxToTest1][1]-(*NDdataPoints)[pointIdxToTest2][1]);

						double dist1=((*NDdataPoints)[0][pointIdxToTest1]-(*NDdataPoints)[0][pointIdxToTest2])*((*NDdataPoints)[0][pointIdxToTest1]-(*NDdataPoints)[0][pointIdxToTest2]);
						double dist2=((*NDdataPoints)[1][pointIdxToTest1]-(*NDdataPoints)[1][pointIdxToTest2])*((*NDdataPoints)[1][pointIdxToTest1]-(*NDdataPoints)[1][pointIdxToTest2]);



						if (sqrt(dist1+dist2)<=epsilon)
						{
							//merge the two clusters
							outputMergeIdsRefPnts->push_back((*clusterIDs)[pointIdxToTest1]);
							outputMergeIdsRefPnts->push_back((*clusterIDs)[pointIdxToTest2]);

							// printf("\nPoints: %d, %d, Merging: %d, %d",pointIdxToTest1,pointIdxToTest2,(*clusterIDs)[pointIdxToTest1],(*clusterIDs)[pointIdxToTest2]);

							//set loop counters so that the two loops end (begin at the next jth loop iteration)
							k=clusterIdsPoints[i].size();
							l=clusterIdsPoints[j].size();
					
							num_merges++;	
						}
						num_iter++;
					}
				}
			}
		}
	}

	// printf("\nEps: %f, Num iterations of distance calc: %u, Num merges: %u, Num unique clusters: %lu, max number of merges: %u",epsilon, num_iter, num_merges, uniqueIds->size(), max_merges);


	
	

	// printf("\nNum merges: %lu", outputMergeIdsRefPnts->size());
	return (outputMergeIdsRefPnts->size());


}


//Takes the neighbors of the reference points and returns a list of clusters to merge
//if they meet the merge criteria (core point overlap between two clusters)
//detect this using distance calulations

//this is for the shadow region so all clusers in each partition have already been merged (i.e., densebox-densebox and dbscan-densebox)
//Need to check that the clusters have a core point overlap to merge
//queryId- id of the reference point
// NDdataPoints- all data points
//pointsIDsInShadowRegion- the point ids in the shadow region in the global dataset 
//neighborTable- the neoghbortable for these points (mapped to pointsIDsInShadowRegion)
int computeMergesRefPointDistanceCalculationsShadowRegion(std::vector<std::vector <DTYPE> > *NDdataPoints, double epsilon, unsigned int minPts, unsigned int queryId, 
					std::vector<int> *clusterIDsMergedFromPartitions,std::vector<unsigned int> *pointsIDsInShadowRegion,  neighborTableLookup * neighborTable, 
					std::vector<int> * uniqueIds, std::vector<int> * outputMergeIdsRefPnts)
{


	




	//create a temp vector array with the points within each cluster
	std::vector<int> clusterIdsPoints[uniqueIds->size()];

	for (int k=neighborTable[queryId].indexmin; k<=neighborTable[queryId].indexmax; k++)
	{
		//unsigned int pointToInsert=neighborTable[queryId].dataPtr[k];
		//this is the index of a point in the neighbortable, but needs to be mapped into its global point id	
		unsigned int idx=neighborTable[queryId].dataPtr[k];
		unsigned int pointToInsert=(*pointsIDsInShadowRegion)[idx];

		//make sure it isn't a noise point or an unassigned point
		if ((*clusterIDsMergedFromPartitions)[pointToInsert]!=-1 && (*clusterIDsMergedFromPartitions)[pointToInsert]!=0 )
		{
			for (int l=0; l<uniqueIds->size(); l++)
			{
				
				if ((*clusterIDsMergedFromPartitions)[pointToInsert]==(*uniqueIds)[l])
				{
					clusterIdsPoints[l].push_back(pointToInsert);
					break;
				}
			}
		}
	}


	//testing printing the clusters and points
	// for (int i=0; i<uniqueIds->size(); i++)
	// {
	// 	printf("\nCluster id: %d, data pnts: ",(*uniqueIds)[i]);
	// 	for (int j=0; j<clusterIdsPoints[i].size(); j++)
	// 	{
	// 	printf("%d, ",clusterIdsPoints[i][j]);
	// 	}
	// }


	// printf("\nFOR TESTING, MERGING ALL CLUSTERS IF THEY ONLY HAVE 1 NEIGHBOR");

	

	unsigned int num_iter=0;
	unsigned int num_merges=0;
	unsigned int max_merges=0;

	// bool flag1=0;
	// bool flag2=0;
	
	

	//loop over pairs of cluster ids 
	for (int i=0; i<uniqueIds->size(); i++)
	{
		for (int j=i+1; j<uniqueIds->size(); j++)
		{
	
			
			max_merges++;
			

			for (int k=0; k<clusterIdsPoints[i].size(); k++){
				unsigned int pointIdxToTest1=clusterIdsPoints[i][k];

				 


				for (int l=0; l<clusterIdsPoints[j].size(); l++){
					unsigned int pointIdxToTest2=clusterIdsPoints[j][l];


					
						//search for the point id to test1
						auto it = std::lower_bound(pointsIDsInShadowRegion->begin(), pointsIDsInShadowRegion->end(), pointIdxToTest1);
						uint64_t ind = std::distance(pointsIDsInShadowRegion->begin(), it);
						unsigned int numNeighborsPoint1=0;
						if (neighborTable[ind].indexmin!=0 && neighborTable[ind].indexmax!=0)
						{
							numNeighborsPoint1=neighborTable[ind].indexmax-neighborTable[ind].indexmin+1;
						}


						//search for the point id to test2
						it = std::lower_bound(pointsIDsInShadowRegion->begin(), pointsIDsInShadowRegion->end(), pointIdxToTest2);
						ind = std::distance(pointsIDsInShadowRegion->begin(), it);
						unsigned int numNeighborsPoint2=0;
						if (neighborTable[ind].indexmin!=0 && neighborTable[ind].indexmax!=0)
						{
							numNeighborsPoint2=neighborTable[ind].indexmax-neighborTable[ind].indexmin+1;
						}

						
						//if the point has at least minpts, then it's a core point which means it is valid to be merged
						if (numNeighborsPoint1>=minPts || numNeighborsPoint2>=minPts)
						{
							//distance calculation
							double dist1=((*NDdataPoints)[0][pointIdxToTest1]-(*NDdataPoints)[0][pointIdxToTest2])*((*NDdataPoints)[0][pointIdxToTest1]-(*NDdataPoints)[0][pointIdxToTest2]);
							double dist2=((*NDdataPoints)[1][pointIdxToTest1]-(*NDdataPoints)[1][pointIdxToTest2])*((*NDdataPoints)[1][pointIdxToTest1]-(*NDdataPoints)[1][pointIdxToTest2]);

							//original
							if (sqrt(dist1+dist2)<=epsilon){
								//merge the two clusters
								outputMergeIdsRefPnts->push_back((*clusterIDsMergedFromPartitions)[pointIdxToTest1]);
								outputMergeIdsRefPnts->push_back((*clusterIDsMergedFromPartitions)[pointIdxToTest2]);
								//set loop counters so that the two loops end (begin at the next jth loop iteration)
								k=clusterIdsPoints[i].size();
								l=clusterIdsPoints[j].size();
								num_merges++;	
							}

						}
						


						num_iter++;
				}
			}
		}
	}

	// printf("\nEps: %f, Num iterations of distance calc: %u, Num merges: %u, Num unique clusters: %lu, max number of merges: %u",epsilon, num_iter, num_merges, uniqueIds->size(), max_merges);


	
	

	// printf("\nNum merges: %lu", outputMergeIdsRefPnts->size());
	return (outputMergeIdsRefPnts->size());

	
}





//returns the number of unique clusters (and the unique clusters) that a point's neighbors have  
//checks each neighbor and determines what cluster it is located in
//Doesn't count noise points
int getUniqueClusters(unsigned int queryId, std::vector<unsigned int> *queryVect, std::vector<int> *clusterIDs, neighborTableLookup * neighborTable, std::vector<int> * outputUniqueIds)
{
	std::set<int> setOfClusterIds;

	unsigned int neighbor;

	for (int j=neighborTable[queryId].indexmin; j<=neighborTable[queryId].indexmax; j++){
		
		
		neighbor=neighborTable[queryId].dataPtr[j];
		//only insert if it isn't noise and the neighbor has been assigned a cluster id
		if ((*clusterIDs)[neighbor]!=-1 && (*clusterIDs)[neighbor]!=0)
		{
		setOfClusterIds.insert((*clusterIDs)[neighbor]);
		}
		
	}

	//copy unique clusters to vector 
	std::copy(setOfClusterIds.begin(), setOfClusterIds.end(), std::back_inserter((*outputUniqueIds)));

	return (setOfClusterIds.size());
}


//returns the number of unique clusters (and the unique clusters) that a point's neighbors have  
//checks each neighbor and determines what cluster it is located in
//Doesn't count noise points
//Since it is for the shadow region, we need to use the global cluster IDs across all partitions and a lookup array 
//To map the queryId to the actual point id in the global dataset
int getUniqueClustersForShadowRegions(unsigned int queryId, std::vector<unsigned int> *queryVectRefPoints, std::vector<int> *clusterIDsMergedFromPartitions, 
				std::vector<unsigned int> *pointsIDsInShadowRegion, neighborTableLookup * neighborTable, std::vector<int> * outputUniqueIds)
{
	std::set<int> setOfClusterIds;

	for (int j=neighborTable[queryId].indexmin; j<=neighborTable[queryId].indexmax; j++){
		
		//this is the index of a point in the neighbortable, but needs to be mapped into it's actual point id	
		unsigned int idx=neighborTable[queryId].dataPtr[j];
		unsigned int neighbor=(*pointsIDsInShadowRegion)[idx];
		//only insert if it isn't noise and the neighbor has been assigned a cluster id
		if ((*clusterIDsMergedFromPartitions)[neighbor]!=-1 && (*clusterIDsMergedFromPartitions)[neighbor]!=0)
		{
		setOfClusterIds.insert((*clusterIDsMergedFromPartitions)[neighbor]);
		}
		
	}

	//copy unique clusters to vector 
	std::copy(setOfClusterIds.begin(), setOfClusterIds.end(), std::back_inserter((*outputUniqueIds)));

	return (setOfClusterIds.size());
}



//EVERYWHERE WE SEE "2" IN THIS FUNCTION ,IT'S THE NUMBER OF INDEXED DIMENSIONS
void densebox(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, unsigned int minPts, std::vector<int> *clusterIDs, std::vector<unsigned int> *queryVect, std::vector<bool> *denseBoxPoints)
{

	

	DTYPE denseboxepsilon=epsilon/(2.0*sqrt(2.0));

	printf("\n[Dense box] Epsilon: %f",denseboxepsilon);cout.flush();

	DTYPE * minArr= new DTYPE[NUMINDEXEDDIM];
	DTYPE * maxArr= new DTYPE[NUMINDEXEDDIM];
	unsigned int * nCells= new unsigned int[NUMINDEXEDDIM];
	uint64_t totalCells=0;
	unsigned int nNonEmptyCells=0;
	

	generateNDGridDimensions(NDdataPoints,denseboxepsilon, minArr, maxArr, nCells, &totalCells);
	printf("\nGrid: total cells (including empty) %lu",totalCells);cout.flush();

		
	// allocate memory for index now that we know the number of cells
	//the grid struct itself
	//the grid lookup array that accompanys the grid -- so we only send the non-empty cells
	struct grid * index; //allocate in the populateDNGridIndexAndLookupArray -- only index the non-empty cells
	struct gridCellLookup * gridCellLookupArr; //allocate in the populateDNGridIndexAndLookupArray -- list of non-empty cells


	
	double tstartindex=omp_get_wtime();

	//ids of the elements in the database that are found in each grid cell
	unsigned int * indexLookupArr=new unsigned int[NDdataPoints->size()]; 

	printf("\nBefore Generating the Index/lookup array for densebox");cout.flush();

	// populateNDGridIndexAndLookupArray(NDdataPoints, denseboxepsilon, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells, &gridCellNDMask, gridCellNDMaskOffsets, nNDMaskElems);
	populateNDGridIndexAndLookupArrayParallel(NDdataPoints, denseboxepsilon, &gridCellLookupArr, &index, indexLookupArr, minArr,  nCells, totalCells, &nNonEmptyCells);
	
	double tendindex=omp_get_wtime();

	printf("\nTime to index for densebox: %f", tendindex - tstartindex);cout.flush();


	unsigned int totalPnts=0;
	unsigned int totalCellsWithMinptsPts=0;
	unsigned int totalPointsThatCanBeSkippedDenseBoxes=0;


	//the cells that meet the dense box criteria
	std::vector<unsigned int> denseBoxCells;

	//cells that don't meet the criteria but may be merged into other dense boxes
	std::vector<unsigned int> nonDenseBoxCells;
	std::vector<unsigned int> nonDenseBoxCellsToMarkAsDense;

	for (unsigned int i=0; i<nNonEmptyCells; i++)
	{
		unsigned int numInCell=index[i].indexmax-index[i].indexmin+1;
		totalPnts+=numInCell;
		if (numInCell>=minPts)
		{
			totalCellsWithMinptsPts++;
			totalPointsThatCanBeSkippedDenseBoxes+=numInCell;
			denseBoxCells.push_back(gridCellLookupArr[i].gridLinearID);
		}
		else
		{
			nonDenseBoxCells.push_back(gridCellLookupArr[i].gridLinearID);
		}
	}


	printf("\nTotal points in cells (sanity check): %u", totalPnts);cout.flush();
	printf("\nTotal cells with at least minpts (dense boxes): %lu", denseBoxCells.size());cout.flush();
	printf("\nTotal points that can be skipped in dense boxes only: %u (fraction of |D|: %f)", totalPointsThatCanBeSkippedDenseBoxes, totalPointsThatCanBeSkippedDenseBoxes*1.0/totalPnts*1.0);cout.flush();
	
	
	
	printf("\n*********************************************\n");

	

	UF disjointsets(denseBoxCells.size());cout.flush();	

	printf("\nSize of ds: %u, num of total dense boxes: %lu",disjointsets.count(), denseBoxCells.size());	cout.flush();
	
	//merge dense boxes

	//first, see if the cells are in the same row, and then if they are adjacent then merge
	for (int i=1; i<denseBoxCells.size(); i++)
	{
		unsigned int x_cell_coord1=denseBoxCells[i-1]%nCells[0];
		unsigned int y_cell_coord1=denseBoxCells[i-1]/nCells[0];

		unsigned int x_cell_coord2=denseBoxCells[i]%nCells[0];
		unsigned int y_cell_coord2=denseBoxCells[i]/nCells[0];

		//if in the same row and adjacent
		if ((y_cell_coord2==y_cell_coord1) && ((x_cell_coord2-x_cell_coord1)==1))
		{
			// printf("\nUnioning: iter: %d, linear ids: %u, %u, idx: %u, %u",i,denseBoxCells[i-1], denseBoxCells[i], i-1,i);
			disjointsets.merge(i-1,i); //union by idx
			
		}
	}	

	printf("\nSize of ds: %u, num of total dense boxes: %lu",disjointsets.count(), denseBoxCells.size());	cout.flush();

	//next, merge based on the next row, including the one directly below, and to the left and right 
	unsigned int * indexes=new unsigned int[2];
	for (int i=0; i<denseBoxCells.size(); i++)
	{
		unsigned int x_cell_origin_coord1=denseBoxCells[i]%nCells[0];
		unsigned int y_cell_origin_coord1=denseBoxCells[i]/nCells[0];

		indexes[0]=x_cell_origin_coord1;
		indexes[1]=y_cell_origin_coord1;	
		// uint64_t linearid= getLinearID_nDimensions(indexes, nCells,2);		
		// printf("\n2-d coords: %u, %u, Linear id computed origin cell: %lu,from array: %u, nCells: %u",indexes[0],indexes[1],linearid, denseBoxCells[i],nCells[0]);

		//always test the next row, unless it's the last row, then we go to the next iteration
		unsigned int x_cell_coord_test1=x_cell_origin_coord1+1;
		if (x_cell_coord_test1>=nCells[0])
		{
			continue;
		}

		//the three options in the next row
		unsigned int y_cell_coord_test[3];

		//make sure we don't go "off of the grid" with the -1 or +1
		//if it goes off, we just test the one below twice (merging will have no effect)
		y_cell_coord_test[0]=max(0,y_cell_origin_coord1-1);
		y_cell_coord_test[1]=y_cell_origin_coord1;
		y_cell_coord_test[2]=min(nCells[1]-1,y_cell_origin_coord1+1);


		
		indexes[0]=x_cell_coord_test1;
		//if there is a dense box in the next row of the 3 options
		for (int j=0; j<3; j++)
		{
			
			indexes[1]=y_cell_coord_test[j];

			uint64_t linearid= getLinearID_nDimensions(indexes, nCells,2);

			//if a densebox cell exists with the linear id, merge
			auto it = std::lower_bound(denseBoxCells.begin(), denseBoxCells.end(), (unsigned int)linearid);

			if(!(it == denseBoxCells.end() || *it != linearid))
			{
				uint64_t ind = std::distance(denseBoxCells.begin(), it);
	    		// std::cout << linearid << " was found at " << ind << '\n';
	    		// printf("\nLinearid %lu was found at: %lu, sanity check: %u",linearid,ind,denseBoxCells[ind]);

				//sanity check:
	    		// if(linearid!=denseBoxCells[ind])
	    		// {
	    		// 	printf("\nERROR linear id/lookup don't match");
	    		// }

				disjointsets.merge(i,ind);
				// printf("\nUnioning2,%d: iter: %d, idx: %lu, lid1: %u, lid2: %u (searched linear id %u)",j,i,ind,denseBoxCells[i], denseBoxCells[(unsigned int)ind], (unsigned int)linearid);



			}


		}
	}
	

	//assemble the list of disjoint sets representing the chain of dense boxes
	std::vector<std::vector<unsigned int> > disjointSetList;
	disjointsets.assembleSetList(&disjointSetList, denseBoxCells.size());
	
	printf("\nSize of ds: %u, num of total dense boxes: %lu",disjointsets.count(), denseBoxCells.size());	cout.flush();	

	unsigned int cnt=0;
	for (int i=0; i<disjointSetList.size(); i++)
	{
		//print if not empty
		//empty if only one element
		if (disjointSetList[i][0]!=disjointSetList.size())
		{
			cnt++;
		}
	}

	printf("\ncnt non empty: %u",cnt);cout.flush();


	//testing to ensure that adding the indices together are equal after the indices are merged

	uint64_t test_total_size=0;
	uint64_t sanity_check_size=0;

	for (int i=0; i<disjointSetList.size(); i++)
	{
		sanity_check_size+=i;	
		//if non-empty
		if (disjointSetList[i][0]!=disjointSetList.size())
		{
			// printf("\nRepresentative: %d: ",i);
			for (int j=0; j<disjointSetList[i].size(); j++)
			{
				// printf("%u, ",disjointSetList[i][j]);
				test_total_size+=disjointSetList[i][j];
			}
		}
	}

	assert(test_total_size==sanity_check_size);
	printf("\nTest total size: %lu, sanity check total size: %lu", test_total_size, sanity_check_size);cout.flush();

	/*
	//same as above but with the linear ids 
	for (int i=0; i<disjointSetList.size(); i++)
	{
		
		//if non-empty
		if (disjointSetList[i][0]!=disjointSetList.size())
		{
			// printf("\nRepresentative [linearid]: %u: ",denseBoxCells[i]);
			for (int j=0; j<disjointSetList[i].size(); j++)
			{
				unsigned int idx=disjointSetList[i][j];
				// printf("%u, ",denseBoxCells[idx]);
				
			}
		}
	}
	*/


	//testing print sets

	// PrintSets(&disjointSetList, &denseBoxCells, NDdataPoints, gridCellLookupArr, &nNonEmptyCells, indexLookupArr, index);

	//generate the cluster assignments from the dense boxes
	//generate the list of points that need neighborhood searches (queryVect)
	generatePointLists(clusterIDs,queryVect, &disjointSetList, &denseBoxCells, NDdataPoints, gridCellLookupArr, &nNonEmptyCells, indexLookupArr, index, denseBoxPoints);


	
	delete [] indexLookupArr;
	delete [] minArr;
	delete [] maxArr;
	delete [] nCells;
	delete [] indexes;


}





//TESTING not generating the index to densebox the points
//Idea: generate key/value pairs (key: linearID, value: pointID in the box)
//Then we find if there are minPts within each "dense box" (at least minPts keys)
//And then assign these points to the dense boxes
void denseboxWithoutIndex(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, unsigned int minPts, 
	std::vector<int> *clusterIDs, std::vector<unsigned int> *queryVect, std::vector<bool> *denseBoxPoints)
{

	

	DTYPE denseboxepsilon=epsilon/(2.0*sqrt(2.0));

	printf("\n[Dense box] Epsilon: %f",denseboxepsilon);cout.flush();

	DTYPE * minArr= new DTYPE[NUMINDEXEDDIM];
	DTYPE * maxArr= new DTYPE[NUMINDEXEDDIM];
	unsigned int * nCells= new unsigned int[NUMINDEXEDDIM];
	uint64_t totalCells=0;
	unsigned int nNonEmptyCells=0;
	

	generateNDGridDimensions(NDdataPoints,denseboxepsilon, minArr, maxArr, nCells, &totalCells);
	printf("\nGrid: total cells (including empty) %lu",totalCells);cout.flush();

	//new: use key/value pairs to keep track of what points are in which dense boxes
	std::vector <keyValDenseBoxStruct> denseBoxKeyValue;
	denseBoxKeyValue.resize((*NDdataPoints)[0].size());


	unsigned int tmpNDCellIdx[NUMINDEXEDDIM];

	//create key/value pairs of linearIDs of cells and points inside the cells



	
	
	#pragma omp parallel for schedule(static) num_threads(3)
	for (unsigned long int i=0; i<(*NDdataPoints)[0].size(); i++){
		
		for (unsigned long int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellIdx[j]=(((*NDdataPoints)[j][i]-minArr[j])/denseboxepsilon);
		}
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);

		denseBoxKeyValue[i].linearID=linearID;
		denseBoxKeyValue[i].pointID=i;
	}

	

	//sort by key (linearID)

	

	//NEEDS TO BE PARALLELIZED ~half of dense box resp time
	//Parallelize sorting the key/value pairs
	sortDenseBoxKeyVal(&denseBoxKeyValue);

	

	//the cells that meet the dense box criteria
	std::vector<unsigned int> denseBoxCells;



	unsigned int totalCellsWithMinptsPts=0;
	unsigned int totalPointsThatCanBeSkippedDenseBoxes=0;


	//keep track of the last linearID initialize to the first linearID
	uint64_t lastLinearID=denseBoxKeyValue[0].linearID;
	unsigned long int numPointsCnt=1;

	unsigned int totalPnts=1;




	//start at 1 since we initialize to the lastLinearID
	//Cannot be parallelized
	for (unsigned long int i=1; i<denseBoxKeyValue.size(); i++)
	{
		if (denseBoxKeyValue[i].linearID==lastLinearID)
		{
			numPointsCnt++;
			totalPnts++; //sanity check
		}
		//if it's not the last linearID, then we check to see if there are at least MinPts
		//otherwise we reset 
		else
		{

			//if it's a dense box, append the last Linear ID
			if (numPointsCnt>=minPts)
			{
				denseBoxCells.push_back(lastLinearID);
				totalCellsWithMinptsPts++;
				totalPointsThatCanBeSkippedDenseBoxes+=numPointsCnt;
			}
			//reset the variables for the new linearID
			lastLinearID=denseBoxKeyValue[i].linearID;
			numPointsCnt=1;

			totalPnts++; //sanity check
		}
	}

	//edge case: check the last linearID
	if (numPointsCnt>=minPts)
	{
		denseBoxCells.push_back(lastLinearID);	
	}

	

	printf("\nNum dense boxes: %lu",denseBoxCells.size());


	//store the dense box IDs and their points
	std::vector<DenseBoxPointIDStruct> DenseBoxPointIDs; 
	DenseBoxPointIDs.resize(denseBoxCells.size());

	// unsigned long int nextIdx=0;	

	//ORIGINAL
	/*
	for (unsigned long int i=0; i<denseBoxCells.size(); i++)
	{
		DenseBoxPointIDs[i].linearID=denseBoxCells[i];

		//since the densebox point ids are sorted, we can scan through a total of once to get all of the points assigned
		//to each densebox (it's not O n**2)
		for (unsigned long int j=nextIdx; j<denseBoxKeyValue.size(); j++)
		{
			if (DenseBoxPointIDs[i].linearID==denseBoxKeyValue[j].linearID)
			{
				DenseBoxPointIDs[i].pointIDs.push_back(denseBoxKeyValue[j].pointID);	
			}
			else if (DenseBoxPointIDs[i].linearID<denseBoxKeyValue[j].linearID)
			{
				nextIdx=j;
				break;
			}
		}
	}
	*/	

	// 

	
	
	
	//to parallelize the algorithm above need to do binary searches instead of only "looking forward" in the array
	#pragma omp parallel 
	{
		std::vector<keyValDenseBoxStruct>::iterator lowerBoundIt;
		std::vector<keyValDenseBoxStruct> tmpLinearIdx;
		tmpLinearIdx.resize(1);

		#pragma omp for 
		for (unsigned long int i=0; i<denseBoxCells.size(); i++)
		{

			
			DenseBoxPointIDs[i].linearID=denseBoxCells[i];


			//need to do a lower bound and find the index of the first element with the linearID here
			
			tmpLinearIdx[0].linearID=DenseBoxPointIDs[i].linearID;
			lowerBoundIt = std::lower_bound(denseBoxKeyValue.begin(), denseBoxKeyValue.end(), keyValDenseBoxStruct(tmpLinearIdx[0]));

			unsigned long int idxDenseBoxKeyVal=std::distance(denseBoxKeyValue.begin(),lowerBoundIt);

		
			for (unsigned long int j=idxDenseBoxKeyVal; j<denseBoxKeyValue.size(); j++)
			{
				if (DenseBoxPointIDs[i].linearID==denseBoxKeyValue[j].linearID)
				{
					DenseBoxPointIDs[i].pointIDs.push_back(denseBoxKeyValue[j].pointID);	
				}
				else if (DenseBoxPointIDs[i].linearID<denseBoxKeyValue[j].linearID)
				{
					break;
				}
			}
		}
	}
	
	

	

	//testing the total number of points in dense boxes
	// unsigned long int NumPntsDenseBoxSanityCheck=0;
	// for (unsigned long int i=0; i<DenseBoxPointIDs.size(); i++)
	// {
	// 	NumPntsDenseBoxSanityCheck+=DenseBoxPointIDs[i].pointIDs.size();
	// }

	// printf("\nSanity check num dense box points: %lu", NumPntsDenseBoxSanityCheck);


	
	printf("\nTotal points in cells (sanity check): %u", totalPnts);cout.flush();
	printf("\nTotal cells with at least minpts (dense boxes): %lu", denseBoxCells.size());cout.flush();
	printf("\nTotal points that can be skipped in dense boxes only: %u (fraction of |D|: %f)", totalPointsThatCanBeSkippedDenseBoxes, totalPointsThatCanBeSkippedDenseBoxes*1.0/totalPnts*1.0);cout.flush();
	

	
	
	UF disjointsets(denseBoxCells.size());cout.flush();	

	printf("\nSize of ds: %u, num of total dense boxes: %lu",disjointsets.count(), denseBoxCells.size());	cout.flush();
	
	

	//merge dense boxes

	//first, see if the cells are in the same row, and then if they are adjacent then merge
	
	
	for (int i=1; i<denseBoxCells.size(); i++)
	{
		unsigned int x_cell_coord1=denseBoxCells[i-1]%nCells[0];
		unsigned int y_cell_coord1=denseBoxCells[i-1]/nCells[0];

		unsigned int x_cell_coord2=denseBoxCells[i]%nCells[0];
		unsigned int y_cell_coord2=denseBoxCells[i]/nCells[0];

		//if in the same row and adjacent
		if ((y_cell_coord2==y_cell_coord1) && ((x_cell_coord2-x_cell_coord1)==1))
		{
			// printf("\nUnioning: iter: %d, linear ids: %u, %u, idx: %u, %u",i,denseBoxCells[i-1], denseBoxCells[i], i-1,i);
			disjointsets.merge(i-1,i); //union by idx
			
		}
	}	



	printf("\nSize of ds: %u, num of total dense boxes: %lu",disjointsets.count(), denseBoxCells.size());	cout.flush();

	//next, merge based on the next row, including the one directly below, and to the left and right 
	unsigned int * indexes=new unsigned int[2];

	

	//use this so all threads keep track of merges, then we update at the end
	struct mergeList{
		unsigned long int id1;
		unsigned long int id2;
	};

	std::vector<struct mergeList> mergeListThreads[NTHREADS];





	#pragma omp parallel 
	{
		unsigned int tid=omp_get_thread_num();
		mergeList tmpMergeItem;
		#pragma omp for
		for (int i=0; i<denseBoxCells.size(); i++)
		{
			unsigned int indexes[2];
			unsigned int x_cell_origin_coord1=denseBoxCells[i]%nCells[0];
			unsigned int y_cell_origin_coord1=denseBoxCells[i]/nCells[0];

			indexes[0]=x_cell_origin_coord1;
			indexes[1]=y_cell_origin_coord1;	
			// uint64_t linearid= getLinearID_nDimensions(indexes, nCells,2);		
			// printf("\n2-d coords: %u, %u, Linear id computed origin cell: %lu,from array: %u, nCells: %u",indexes[0],indexes[1],linearid, denseBoxCells[i],nCells[0]);

			//always test the next row, unless it's the last row, then we go to the next iteration
			unsigned int x_cell_coord_test1=x_cell_origin_coord1+1;
			if (x_cell_coord_test1>=nCells[0])
			{
				continue;
			}

			//the three options in the next row
			unsigned int y_cell_coord_test[3];

			//make sure we don't go "off of the grid" with the -1 or +1
			//if it goes off, we just test the one below twice (merging will have no effect)
			y_cell_coord_test[0]=max(0,y_cell_origin_coord1-1);
			y_cell_coord_test[1]=y_cell_origin_coord1;
			y_cell_coord_test[2]=min(nCells[1]-1,y_cell_origin_coord1+1);


			
			indexes[0]=x_cell_coord_test1;
			//if there is a dense box in the next row of the 3 options
			for (int j=0; j<3; j++)
			{
				
				indexes[1]=y_cell_coord_test[j];

				uint64_t linearid= getLinearID_nDimensions(indexes, nCells,2);

				//if a densebox cell exists with the linear id, merge
				auto it = std::lower_bound(denseBoxCells.begin(), denseBoxCells.end(), (unsigned int)linearid);

				if(!(it == denseBoxCells.end() || *it != linearid))
				{
					uint64_t ind = std::distance(denseBoxCells.begin(), it);
		    		// std::cout << linearid << " was found at " << ind << '\n';
		    		// printf("\nLinearid %lu was found at: %lu, sanity check: %u",linearid,ind,denseBoxCells[ind]);

					//sanity check:
		    		// if(linearid!=denseBoxCells[ind])
		    		// {
		    		// 	printf("\nERROR linear id/lookup don't match");
		    		// }

					//original before parallelizing
					// disjointsets.merge(i,ind);

					//for threads
					tmpMergeItem.id1=i;
					tmpMergeItem.id2=ind;
					mergeListThreads[tid].push_back(tmpMergeItem);
					
					

					// printf("\nUnioning2,%d: iter: %d, idx: %lu, lid1: %u, lid2: %u (searched linear id %u)",j,i,ind,denseBoxCells[i], denseBoxCells[(unsigned int)ind], (unsigned int)linearid);



				}


			}
		}

	}	


	//merge using all mergelists for the threads
	for (unsigned int i=0; i<NTHREADS; i++)
	{
		for (unsigned int j=0; j<mergeListThreads[i].size(); j++)
		{
			disjointsets.merge(mergeListThreads[i][j].id1,mergeListThreads[i][j].id2);
		}

	}	
	
	
	

	

	//assemble the list of disjoint sets representing the chain of dense boxes
	std::vector<std::vector<unsigned int> > disjointSetList;
	disjointsets.assembleSetList(&disjointSetList, denseBoxCells.size());


	
	
	
	printf("\nSize of ds: %u, num of total dense boxes: %lu",disjointsets.count(), denseBoxCells.size());	cout.flush();	

	unsigned int cnt=0;
	for (int i=0; i<disjointSetList.size(); i++)
	{
		//print if not empty
		//empty if only one element
		if (disjointSetList[i][0]!=disjointSetList.size())
		{
			cnt++;
		}
	}

	printf("\ncnt non empty: %u",cnt);cout.flush();


	//testing to ensure that adding the indices together are equal after the indices are merged

	uint64_t test_total_size=0;
	uint64_t sanity_check_size=0;

	for (int i=0; i<disjointSetList.size(); i++)
	{
		sanity_check_size+=i;	
		//if non-empty
		if (disjointSetList[i][0]!=disjointSetList.size())
		{
			// printf("\nRepresentative: %d: ",i);
			for (int j=0; j<disjointSetList[i].size(); j++)
			{
				// printf("%u, ",disjointSetList[i][j]);
				test_total_size+=disjointSetList[i][j];
			}
		}
	}

	assert(test_total_size==sanity_check_size);
	printf("\nTest total size: %lu, sanity check total size: %lu", test_total_size, sanity_check_size);cout.flush();

	
	
	//testing print sets

	// PrintSets(&disjointSetList, &denseBoxCells, NDdataPoints, gridCellLookupArr, &nNonEmptyCells, indexLookupArr, index);

	//generate the cluster assignments from the dense boxes
	//generate the list of points that need neighborhood searches (queryVect)

	// generatePointLists(clusterIDs,queryVect, &disjointSetList, &denseBoxCells, NDdataPoints, gridCellLookupArr, &nNonEmptyCells, indexLookupArr, index, denseBoxPoints);




	generatePointListsWithoutIndex(clusterIDs, queryVect, &disjointSetList, &denseBoxCells, NDdataPoints, &nNonEmptyCells, denseBoxPoints, &DenseBoxPointIDs);


	
	// void generatePointListsWithoutIndex(std::vector<int>  * clusterIDs, std::vector<unsigned int>  * queryVect, 
	// std::vector<std::vector<unsigned int> > * disjointSetList, std::vector<unsigned int> * denseBoxCells, 
	// std::vector<std::vector <DTYPE> > *NDdataPoints, unsigned int * nNonEmptyCells, std::vector<bool> *denseBoxPoints)


	delete [] minArr;
	delete [] maxArr;
	delete [] nCells;
	delete [] indexes;
	

}






//generate the list of assigned clusters from the densebox algorithm
//generate the list of query points that were not found by the densebox algorithm
void generatePointListsWithoutIndex(std::vector<int>  * clusterIDs, std::vector<unsigned int>  * queryVect, 
	std::vector<std::vector<unsigned int> > * disjointSetList, std::vector<unsigned int> * denseBoxCells, 
	std::vector<std::vector <DTYPE> > *NDdataPoints, unsigned int * nNonEmptyCells, std::vector<bool> *denseBoxPoints, 
	std::vector<DenseBoxPointIDStruct> *DenseBoxPointIDs)
{

	clusterIDs->reserve((*NDdataPoints)[0].size());

	for (int i=0; i<(*NDdataPoints)[0].size(); i++)
	{
		clusterIDs->push_back(-1);
	}

	//0-will be noise, start at 1
	//-1 is no cluster assigned
	unsigned int clusternum=1;

	//ORIGINAL
	/*
	for (int i=0; i<disjointSetList->size();i++)
	{
		//if non-empty
		if ((*disjointSetList)[i][0]!=(*disjointSetList).size())
		{
			
				for (int j=0; j<(*disjointSetList)[i].size(); j++)
				{
					unsigned int idx=(*disjointSetList)[i][j];	

	        		struct gridCellLookup tmp;
	        		tmp.gridLinearID=(uint64_t)(*denseBoxCells)[idx];
	        		struct gridCellLookup * resultBinSearch=std::lower_bound(gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
	                unsigned int GridIndex=resultBinSearch->idx;

	                //loop over points here-
	                //x array	
	                for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
	                unsigned int dataIdx=indexLookupArr[k];
	                (*clusterIDs)[dataIdx]=clusternum;
	                }
	                
					
				}			
		}
		clusternum++;	
	}

	*/


	//testing
	// unsigned long int maxDenseBox=0;
	// for (unsigned long int i=0; i<(*denseBoxCells).size(); i++)
	// {
	// 	if (maxDenseBox<(*denseBoxCells)[i])
	// 		maxDenseBox=(*denseBoxCells)[i];
	// }
	// printf("\nMax dense box: %lu", maxDenseBox);


	// printf("\nBefore disjoint set list loop");cout.flush();
	
	for (unsigned long int i=0; i<disjointSetList->size();i++)
	{
		
		//if non-empty
		if ((*disjointSetList)[i][0]!=(*disjointSetList).size())
		{
				// printf("\nIter: %lu/%lu, Val first elem: %u",i,disjointSetList->size(), (*disjointSetList)[i][0]);cout.flush();
			

				
				for (unsigned long int j=0; j<(*disjointSetList)[i].size(); j++)
				{
					unsigned long int idx=(*disjointSetList)[i][j];	

					uint64_t linearID=(uint64_t)(*denseBoxCells)[idx];

					
					struct DenseBoxPointIDStruct tmp;
					tmp.linearID=linearID;

					std::vector<DenseBoxPointIDStruct>::iterator resultBinSearch;
					resultBinSearch=std::lower_bound(DenseBoxPointIDs->begin(), DenseBoxPointIDs->end(), DenseBoxPointIDStruct(tmp));					



					// unsigned long int indexInDenseBoxPointIDs=resultBinSearch-DenseBoxPointIDs->begin();
					
					uint64_t indexInDenseBoxPointIDs=std::distance(DenseBoxPointIDs->begin(), resultBinSearch);

					
					// if (indexInDenseBoxPointIDs>=(*DenseBoxPointIDs).size())
					// {
					// 	printf("\nOut of range! idx: %lu, size vect: %lu", indexInDenseBoxPointIDs,(*DenseBoxPointIDs).size());cout.flush();
					// }	
					// if (indexInDenseBoxPointIDs<(*DenseBoxPointIDs).size())
					// {

					
					
					if(!(resultBinSearch == DenseBoxPointIDs->end()))
					{
						for (unsigned long int k=0; k<(*DenseBoxPointIDs)[indexInDenseBoxPointIDs].pointIDs.size(); k++)
						{
							unsigned long int dataIdx=(*DenseBoxPointIDs)[indexInDenseBoxPointIDs].pointIDs[k];
							(*clusterIDs)[dataIdx]=clusternum;
						}
					}
					
	                
					
				}
						
		}
		clusternum++;	
	}
	
	// printf("\nAfter disjoint set list loop");cout.flush();


	//find remaining query points to be computed by the GPU/neighbortable and DBSCAN leftovers

	//assign denseBoxPoints points that have been found with the densebox points
	//set the bool to true if its a densebox point
	unsigned int numDenseBoxPoints=0;
	for (int i=0; i<clusterIDs->size(); i++)
	{
		if ((*clusterIDs)[i]==-1)
		{
			queryVect->push_back(i);
		}

		//if its been assigned, then its a densebox point
		if ((*clusterIDs)[i]>0)
		{
			denseBoxPoints->push_back(true);
			numDenseBoxPoints++;
		}
		else
		{
			denseBoxPoints->push_back(false);	
		}
	}

	printf("\nNumber of points that were found within dense boxes (sanity check): %u",numDenseBoxPoints);cout.flush();
	printf("\nNumber of points that were not found within dense boxes: %lu",queryVect->size());cout.flush();

} 









//generate the list of assigned clusters from the densebox algorithm
//generate the list of query points that were not found by the densebox algorithm
void generatePointLists(std::vector<int>  * clusterIDs, std::vector<unsigned int>  * queryVect, std::vector<std::vector<unsigned int> > * disjointSetList, std::vector<unsigned int> * denseBoxCells, 
	std::vector<std::vector <DTYPE> > *NDdataPoints, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index, std::vector<bool> *denseBoxPoints)
{

	clusterIDs->reserve(NDdataPoints->size());

	for (int i=0; i<NDdataPoints->size(); i++)
	{
		clusterIDs->push_back(-1);
	}

	//0-will be noise, start at 1
	//-1 is no cluster assigned
	unsigned int clusternum=1;

	for (int i=0; i<disjointSetList->size();i++)
	{
		//if non-empty
		if ((*disjointSetList)[i][0]!=(*disjointSetList).size())
		{
			
				for (int j=0; j<(*disjointSetList)[i].size(); j++)
				{
					unsigned int idx=(*disjointSetList)[i][j];	

	        		struct gridCellLookup tmp;
	        		tmp.gridLinearID=(uint64_t)(*denseBoxCells)[idx];
	        		struct gridCellLookup * resultBinSearch=std::lower_bound(gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
	                unsigned int GridIndex=resultBinSearch->idx;

	                //loop over points here-
	                //x array	
	                for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
	                unsigned int dataIdx=indexLookupArr[k];
	                (*clusterIDs)[dataIdx]=clusternum;
	                }
	                
					
				}			
		}
		clusternum++;	
	}




	//find remaining query points to be computed by the GPU/neighbortable and DBSCAN leftovers

	//assign denseBoxPoints points that have been found with the densebox points
	//set the bool to true if its a densebox point
	unsigned int numDenseBoxPoints=0;
	for (int i=0; i<clusterIDs->size(); i++)
	{
		if ((*clusterIDs)[i]==-1)
		{
			queryVect->push_back(i);
		}

		//if its been assigned, then its a densebox point
		if ((*clusterIDs)[i]>0)
		{
			denseBoxPoints->push_back(true);
			numDenseBoxPoints++;
		}
		else
		{
			denseBoxPoints->push_back(false);	
		}
	}

	printf("\nNumber of points that were found within dense boxes (sanity check): %u",numDenseBoxPoints);
	printf("\nNumber of points that were not found within dense boxes: %lu",queryVect->size());

} 



//for testing,
//print the points inside each densebox chain for plotting
//see the point thresholds for printing
void PrintSets(std::vector<std::vector<unsigned int> > * disjointSetList, std::vector<unsigned int> * denseBoxCells, 
	std::vector<std::vector <DTYPE> > *NDdataPoints, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, struct grid * index)
{






	std::vector<DTYPE> tmpx_coord;
	std::vector<DTYPE> tmpy_coord;

	unsigned int listCnt=0;

	for (int i=0; i<disjointSetList->size();i++)
	{
		//if non-empty
		if ((*disjointSetList)[i][0]!=(*disjointSetList).size())
		{
			
			//****************
			//only print out a fraction of the lists: criteria- if they have between the bounded number of points
			int lower_pnt_thresh=50;
			int upper_pnt_thresh=3000;
			//****************

			if ((*disjointSetList)[i].size()>=lower_pnt_thresh && (*disjointSetList)[i].size()<=upper_pnt_thresh)
			{
			
				for (int j=0; j<(*disjointSetList)[i].size(); j++)
				{
					unsigned int idx=(*disjointSetList)[i][j];	

	        		struct gridCellLookup tmp;
	        		tmp.gridLinearID=(uint64_t)(*denseBoxCells)[idx];
	        		struct gridCellLookup * resultBinSearch=std::lower_bound(gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
	                unsigned int GridIndex=resultBinSearch->idx;

	                //loop over points here-
	                //x array	
	                for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
	                unsigned int dataIdx=indexLookupArr[k];
	                // printf("[%f, %f], ",(*NDdataPoints)[dataIdx][0],(*NDdataPoints)[dataIdx][1]);	
	                tmpx_coord.push_back((*NDdataPoints)[dataIdx][0]);
	                tmpy_coord.push_back((*NDdataPoints)[dataIdx][1]);
	                }
	                
					
				}

				//print x and y arrays:

				printf("arr%d_x=[",listCnt);
				for (int l=0; l<tmpx_coord.size(); l++)
				{
				printf("%f, ",tmpx_coord[l]);
				}
				printf("]");

				printf("\narr%d_y=[",listCnt);
				for (int l=0; l<tmpy_coord.size(); l++)
				{
				printf("%f, ",tmpy_coord[l]);
				}
				printf("]");

				printf("\n");
				listCnt++;

				tmpx_coord.clear();
				tmpy_coord.clear();

			}
		}	
	}

}



//use the index to compute the neighbors
void generateNeighborTableCPUPrototype(std::vector<std::vector <DTYPE> > *NDdataPoints, unsigned int queryPoint, DTYPE epsilon, grid * index, struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, unsigned int * indexLookupArr, std::vector<uint64_t> * cellsToCheck, table * neighborTableCPUPrototype)
{
	
	for (int i=0; i<cellsToCheck->size(); i++){
		//find the id in the compressed grid index of the cell:
		//CHANGE TO BINARY SEARCH
		uint64_t GridIndex=0;

		struct gridCellLookup tmp;
		tmp.gridLinearID=(*cellsToCheck)[i];
		// struct gridCellLookup resultBinSearch;
		struct gridCellLookup * resultBinSearch=std::lower_bound(gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
		GridIndex=resultBinSearch->idx;


		for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
			DTYPE runningTotalDist=0;
			//printf("\nPoint id for dist calc: %d",k);
			unsigned int dataIdx=indexLookupArr[k];

			// printf("\nqueryPoint: %d, dataidx: %d",queryPoint,dataIdx);

			for (int l=0; l<GPUNUMDIM; l++){
			runningTotalDist+=((*NDdataPoints)[dataIdx][l]-(*NDdataPoints)[queryPoint][l])*((*NDdataPoints)[dataIdx][l]-(*NDdataPoints)[queryPoint][l]);
			}

			if (sqrt(runningTotalDist)<=epsilon){
				neighborTableCPUPrototype[queryPoint].neighbors.push_back(dataIdx);
			}
		}
}

	return;
}


struct cmpStruct {
	cmpStruct(std::vector <std::vector <DTYPE>> points) {this -> points = points;}
	bool operator() (int a, int b) {
		return points[a][0] < points[b][0];
	}

	std::vector<std::vector<DTYPE>> points;
};



void populateNDGridIndexAndLookupArray(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells, unsigned int ** gridCellNDMask, unsigned int * gridCellNDMaskOffsets, unsigned int * nNDMaskElems)
{

	/////////////////////////////////
	//Populate grid lookup array
	//and corresponding indicies in the lookup array
	/////////////////////////////////
	printf("\n\n*****************************\nPopulating Grid Index and lookup array:\n*****************************\n");
	// printf("\nSize of dataset: %lu", NDdataPoints->size());


	///////////////////////////////
	//First, we need to figure out how many non-empty cells there will be
	//For memory allocation
	//Need to do a scan of the dataset and calculate this
	//Also need to keep track of the list of uniquie linear grid cell IDs for inserting into the grid
	///////////////////////////////
	std::set<uint64_t> uniqueGridCellLinearIds;
	std::vector<uint64_t>uniqueGridCellLinearIdsVect; //for random access

	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellIdx[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);
		}
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);
		uniqueGridCellLinearIds.insert(linearID);

	}

	// printf("uniqueGridCellLinearIds: %d",uniqueGridCellLinearIds.size());

	//copy the set to the vector (sets can't do binary searches -- no random access)
	std::copy(uniqueGridCellLinearIds.begin(), uniqueGridCellLinearIds.end(), std::back_inserter(uniqueGridCellLinearIdsVect));
	



	///////////////////////////////////////////////


	std::vector<uint64_t> * gridElemIDs;
	gridElemIDs = new std::vector<uint64_t>[uniqueGridCellLinearIds.size()];

	//Create ND array mask:
	//This mask determines which cells in each dimension has points in them.
	std::set<unsigned int> NDArrMask[NUMINDEXEDDIM];
	
	vector<uint64_t>::iterator lower;
	

	for (int i=0; i<NDdataPoints->size(); i++){
		unsigned int tmpNDCellID[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellID[j]=(((*NDdataPoints)[i][j]-minArr[j])/epsilon);

			//add value to the ND array mask
			NDArrMask[j].insert(tmpNDCellID[j]);
		}

		//get the linear id of the cell
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellID, nCells, NUMINDEXEDDIM);
		//printf("\nlinear id: %d",linearID);
		if (linearID > totalCells){

			printf("\n\nERROR Linear ID is: %lu, total cells is only: %lu\n\n", linearID, totalCells);
		}

		//find the index in gridElemIds that corresponds to this grid cell linear id
		
		lower=std::lower_bound(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(),linearID);
		uint64_t gridIdx=lower - uniqueGridCellLinearIdsVect.begin();
		gridElemIDs[gridIdx].push_back(i);
	}

	
	

	///////////////////////////////
	//Here we fill a temporary index with points, and then copy the non-empty cells to the actual index
	///////////////////////////////
		


	struct grid * tmpIndex=new grid[uniqueGridCellLinearIdsVect.size()];



	int cnt=0;

	

	//populate temp index and lookup array

	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++)
	{
			tmpIndex[i].indexmin=cnt;
			for (int j=0; j<gridElemIDs[i].size(); j++)
			{
				if (j>((NDdataPoints->size()-1)))
				{
					printf("\n\n***ERROR Value of a data point is larger than the dataset! %d\n\n", j);
					return;
				}
				indexLookupArr[cnt]=gridElemIDs[i][j]; 
				cnt++;
			}
			tmpIndex[i].indexmax=cnt-1;
	}

	// printf("\nExiting grid populate method early!");
	// return;

	printf("\nFull cells: %d (%f, fraction full)",(unsigned int)uniqueGridCellLinearIdsVect.size(), uniqueGridCellLinearIdsVect.size()*1.0/double(totalCells));
	printf("\nEmpty cells: %ld (%f, fraction empty)",totalCells-(unsigned int)uniqueGridCellLinearIdsVect.size(), (totalCells-uniqueGridCellLinearIdsVect.size()*1.0)/double(totalCells));
	
	*nNonEmptyCells=uniqueGridCellLinearIdsVect.size();


	printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid)*(totalCells)/(1024.0*1024.0*1024.0));
	printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid)*(uniqueGridCellLinearIdsVect.size()*1.0)/(1024.0*1024.0*1024.0));


	/////////////////////////////////////////
	//copy the tmp index into the actual index that only has the non-empty cells

	//allocate memory for the index that will be sent to the GPU
	*index=new grid[uniqueGridCellLinearIdsVect.size()];
	*gridCellLookupArr= new struct gridCellLookup[uniqueGridCellLinearIdsVect.size()];

	cmpStruct theStruct(*NDdataPoints);

	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++){
			(*index)[i].indexmin=tmpIndex[i].indexmin;
			(*index)[i].indexmax=tmpIndex[i].indexmax;
			(*gridCellLookupArr)[i].idx=i;
			(*gridCellLookupArr)[i].gridLinearID=uniqueGridCellLinearIdsVect[i];
	}

	printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu",uniqueGridCellLinearIdsVect.size());
	



	//copy NDArrMask from set to an array

	//find the total size and allocate the array
	
	unsigned int cntNDOffsets=0;
	unsigned int cntNonEmptyNDMask=0;
	for (int i=0; i<NUMINDEXEDDIM; i++){
		cntNonEmptyNDMask+=NDArrMask[i].size();
	}	
	*gridCellNDMask = new unsigned int[cntNonEmptyNDMask];
	
	*nNDMaskElems=cntNonEmptyNDMask;

	
	//copy the offsets to the array
	for (int i=0; i<NUMINDEXEDDIM; i++){
		//Min
		gridCellNDMaskOffsets[(i*2)]=cntNDOffsets;
		for (std::set<unsigned int>::iterator it=NDArrMask[i].begin(); it!=NDArrMask[i].end(); ++it){
    		(*gridCellNDMask)[cntNDOffsets]=*it;
    		cntNDOffsets++;
		}
		//max
		gridCellNDMaskOffsets[(i*2)+1]=cntNDOffsets-1;
	}
	
	


	delete [] gridElemIDs;

	delete [] tmpIndex;
		


} //end function populate grid index and lookup array



//determines the linearized ID for a point in n-dimensions
//indexes: the indexes in the ND array: e.g., arr[4][5][6]
//dimLen: the length of each array e.g., arr[10][10][10]
//nDimensions: the number of dimensions


uint64_t getLinearID_nDimensions(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions) {
    // int i;
    // uint64_t offset = 0;
    // for( i = 0; i < nDimensions; i++ ) {
    //     offset += (uint64_t)pow(dimLen[i],i) * (uint64_t)indexes[nDimensions - (i + 1)];
    // }
    // return offset;

    uint64_t index = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i<nDimensions; i++){
  	index += (uint64_t)indexes[i] * multiplier;
  	multiplier *= dimLen[i];
	}

	return index;
}





//testing this one
// uint64_t getLinearID_nDimensions(uint64_t * indexes, unsigned int * dimLen, unsigned int nDimensions) {
// 	uint64_t index = 0;
// 	uint64_t multiplier = 1;
// 	for (int i = 0;i<nDimensions;i++)
// 	{
// 	  index += indexes[i] * multiplier;
// 	  multiplier *= (uint64_t)dimLen[i];
// 	}
// 	// printf("\nLinear Index: %lld",index);
// 	return index;
// }




//min arr- the minimum value of the points in each dimensions - epsilon
//we can use this as an offset to calculate where points are located in the grid
//max arr- the maximum value of the points in each dimensions + epsilon 
//returns the time component of sorting the dimensions when SORT=1
void generateNDGridDimensions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells)
{

	printf("\n\n*****************************\nGenerating grid dimensions.\n*****************************\n");
	
	//First, reorder the points by variance in dimension:



	printf("\nNumber of dimensions data: %d, Number of dimensions indexed: %d", GPUNUMDIM, NUMINDEXEDDIM);
	
	//make the min/max values for each grid dimension the first data element
	for (int j=0; j<NUMINDEXEDDIM; j++){
		minArr[j]=(*NDdataPoints)[j][0];
		maxArr[j]=(*NDdataPoints)[j][0];
	}



	for (int i=1; i<(*NDdataPoints)[0].size(); i++)
	{
		for (int j=0; j<NUMINDEXEDDIM; j++){
		if ((*NDdataPoints)[j][i]<minArr[j]){
			minArr[j]=(*NDdataPoints)[j][i];
		}
		if ((*NDdataPoints)[j][i]>maxArr[j]){
			maxArr[j]=(*NDdataPoints)[j][i];
		}	
		}
	}	
		

	printf("\n");
	for (int j=0; j<NUMINDEXEDDIM; j++){
		printf("Data Dim: %d, min/max: %f,%f\n",j,minArr[j],maxArr[j]);
	}	

	//add buffer around each dim so no weirdness later with putting data into cells
	for (int j=0; j<NUMINDEXEDDIM; j++){
		minArr[j]-=epsilon;
		maxArr[j]+=epsilon;
	}	

	for (int j=0; j<NUMINDEXEDDIM; j++){
		printf("Appended by epsilon Dim: %d, min/max: %f,%f\n",j,minArr[j],maxArr[j]);
	}	
	
	//calculate the number of cells:
	for (int j=0; j<NUMINDEXEDDIM; j++){
		nCells[j]=ceil((maxArr[j]-minArr[j])/epsilon);
		printf("Number of cells dim: %d: %d\n",j,nCells[j]);
	}

	//calc total cells: num cells in each dim multiplied
	uint64_t tmpTotalCells=nCells[0];
	for (int j=1; j<NUMINDEXEDDIM; j++){
		tmpTotalCells*=nCells[j];
	}

	*totalCells=tmpTotalCells;

}



//CPU brute force
void CPUBruteForceTable(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, table * neighborTable, unsigned int * totalNeighbors)
{
	DTYPE runningDist=0;
	unsigned int runningNeighbors=0;
	for (int i=0; i<NDdataPoints->size(); i++)
	{
		neighborTable[i].pointID=i;
		for (int j=0; j<NDdataPoints->size(); j++)
		{
			runningDist=0;
			for (int k=0; k<GPUNUMDIM; k++){
				runningDist+=((*NDdataPoints)[k][i]-(*NDdataPoints)[k][j])*((*NDdataPoints)[k][i]-(*NDdataPoints)[k][j]);
			}
			
			//if within epsilon:
			if ((sqrt(runningDist))<=epsilon){
				neighborTable[i].neighbors.push_back(j);
				runningNeighbors++;
			}
		}

	}
	//update the total neighbor count
	(*totalNeighbors)=runningNeighbors;

}



void test_disjoint_sets2()
{
	///////////
	// testing disjoint sets

	unsigned int N=10;

	UF disjointsets(N);	

	int ret=disjointsets.find(0);
	printf("\nRet: %d",ret);

	printf("\ncnt: %d",disjointsets.count());
	disjointsets.merge(0,1);
	printf("\ncnt: %d",disjointsets.count());

	disjointsets.merge(5,6);
	disjointsets.merge(6,7);
	disjointsets.merge(0,5);



	// disjointsets.merge(51,1);
	printf("\ncnt: %d",disjointsets.count());

	/*
	ret=disjointsets.connected(0,1);
	printf("\nret- connected: %d",ret);

	ret=disjointsets.connected(1,5);
	printf("\nret- connected: %d",ret);

	ret=disjointsets.connected(0,5);
	printf("\nret- connected: %d",ret);

	ret=disjointsets.connected(0,6);
	printf("\nret- connected: %d",ret);

	ret=disjointsets.connected(0,7);
	printf("\nret- connected: %d",ret);

	ret=disjointsets.connected(8,9);
	printf("\nret- connected: %d",ret);
	
	
	*/

	printf("\nTo get the updated list of connections (the root node for each member), need to see if the nodes are connected.");
	printf("\nCan probably obtain this when searching");

	for (unsigned int i=0; i<N; i++)
	{
		disjointsets.connected(i,0);
	}	


	for (unsigned int i=0; i<N; i++)
	{
		printf("\n%d: %d, %d",i, disjointsets.id[i], disjointsets.sz[i]);
	}


	//end testing disjoint sets
	/////////////
}



/*
void test_disjoint_sets()
{
	///////////
	// testing disjoint sets

		std::vector<unsigned int> rank;
		std::vector<unsigned int> parent;

		for (int i=0; i<10; i++)
		{
			rank.push_back(i);
			parent.push_back(i);
		}

	boost::disjoint_sets<unsigned int*,unsigned int*> ds(&rank[0], &parent[0]);

	ds.union_set(0,1);
	ds.union_set(1,2);

	

	
	ds.union_set(4,5);


	unsigned int a=ds.find_set(5);
	printf("\n%u",a);


	ds.union_set(5,6);
	ds.union_set(6,7);



	a=ds.find_set(5);
	printf("\n%u",a);
	a=ds.find_set(6);
	printf("\n%u",a);
	a=ds.find_set(7);
	printf("\n%u",a);


	return;

	//end testing disjoint sets
	/////////////
}
*/


//generates partitions by putting nearly the same number of points in each partition
void generatePartitions(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells, unsigned int * binBounaries, const unsigned int CHUNKS)
{

	double DBfraction=1.0;
	unsigned int stride=1.0/DBfraction;

	//histogram the data in the first dimension
	unsigned long int *bins=new unsigned long int [nCells[0]];

	//find the Bin IDs with the cumulative fraction
	double *cumulativeFrac=new double [nCells[0]];

	for (unsigned int i=0; i<nCells[0]; i++)
	{
		cumulativeFrac[i]=0;
		bins[i]=0;
	}

	unsigned int binidx=0;
	unsigned int approxTotalPointsInserted=((*NDdataPoints)[0].size()*1.0)*DBfraction;
	printf("\nApprox points inserted: %u", approxTotalPointsInserted);

	for (unsigned long int i=0; i<(*NDdataPoints)[0].size(); i+=stride)
	{
		binidx=((*NDdataPoints)[0][i]-minArr[0])/epsilon;
		bins[binidx]++;
	}

	
	

	//the first cumulative fraction is the number of points in the bim
	cumulativeFrac[0]=bins[0];
	//get the number
	for (unsigned long int i=1; i<nCells[0]; i++)
	{
		cumulativeFrac[i]=bins[i]+cumulativeFrac[i-1];
		// printf("\nbin: %lu, cumulative frac (num): %f",i,cumulativeFrac[i]);
	}

	//convert to fraction
	for (unsigned long int i=0; i<nCells[0]; i++)
	{
		cumulativeFrac[i]=cumulativeFrac[i]/(approxTotalPointsInserted*1.0);
		// printf("\nbin: %lu, cumulative frac (fraction): %f",i,cumulativeFrac[i]);
	}

	//find bin boundaries
	double fractionDataPerPartition=1.0/(CHUNKS*1.0);
	double boundary=fractionDataPerPartition;
	// printf("\nFrac data per partition: %f", fractionDataPerPartition);

	binBounaries[0]=0;
	unsigned int cntBin=1;
	for (unsigned long int i=0; i<nCells[0]-1; i++)
	{
		
		// printf("\nBoundary: %f", boundary);
		// printf("\ni: %lu, cntbin: %u",i,cntBin);
		if (boundary>=cumulativeFrac[i] && boundary<cumulativeFrac[i+1])
		{
			binBounaries[cntBin]=i;
			cntBin++;
			boundary+=fractionDataPerPartition;
		}

			
	}


	binBounaries[CHUNKS]=nCells[0];

	// for (unsigned int i=0; i<CHUNKS+1; i++)
	// {
	// 	printf("\nBin boundaries (cell of total num cells): %u/%u",binBounaries[i],nCells[0]);
	// }	


	delete [] bins;
	delete [] cumulativeFrac;
	
		

}







//Returns a schedule for the ordering in which the partitions should be executed to
//fill the pipeline fast and achieve good load balancing at the end
//Forces the small workloads to be at the end 
//Uses the number of points in each partition to compute the density -- if we use a different partitioning strategy, we 
//need to consider the number of points in the partition
void generateScheduleLoadBalance(unsigned int * binBounaries, const unsigned int CHUNKS, unsigned int * schedule, std::vector<std::vector <DTYPE> > *PartitionedNDdataPoints)
{

	
	// for (unsigned int i=0; i<CHUNKS+1; i++)
	// {
	// 	printf("\nBin boundaries (cell of total num cells): %u/%u",binBounaries[i],nCells[0]);
	// }	


	unsigned int * partitionDensity=new unsigned int[CHUNKS];


	//The density of each partition in points in the partition/number of bins
	//
	for (int i=0; i<CHUNKS; i++)
	{
		partitionDensity[i]=PartitionedNDdataPoints[i][0].size()/(binBounaries[i+1]-binBounaries[i]);
		printf("\n[SCHEDULE==1] Partition: %d, Size (number of bins): %u, Number of points in partition: %lu, Density: %u",i,(binBounaries[i+1]-binBounaries[i]), PartitionedNDdataPoints[i][0].size(),partitionDensity[i]);
	}

	//We now generate a schedule for the execution of the partitions
	//We don't sort, use a scan because the number of partitions is small

	//Want the number of GPUs to be the number we put at the start
	//Want the number of parallel chunks to be the number at the end so they have similar workloads


	// unsigned int numPartitionsStart=NUMGPU;
	// unsigned int numPartitionsEnd=PARCHUNKS;

	//ordering from most to least work
	unsigned int * partitionOrder=new unsigned int[CHUNKS];

	
	unsigned int largestDensityIdx=0;
	
	for (int i=0; i<CHUNKS; i++)
	{
		unsigned int largestDensity=partitionDensity[0];
		for (int j=0; j<CHUNKS; j++)
		{
			if (partitionDensity[j]>=largestDensity){
				largestDensity=partitionDensity[j];
				largestDensityIdx=j;
			}
		}

		partitionOrder[i]=largestDensityIdx;
		//set partition size to be 0 so it won't be selected next iteration
		partitionDensity[largestDensityIdx]=0;
	}





	//for now we copy from tmp into schedule in case we change it later
	for (int i=0; i<CHUNKS; i++)
	{
		schedule[i]=partitionOrder[i];
		printf("\n[SCHEDULE==1] Partition order (highest to smallest density): %u", schedule[i]);
	}



	
	
	delete [] partitionOrder;
	delete [] partitionDensity;
	
		

}








void generatePartitionsMinimizeShadowRegionPoints(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, DTYPE* minArr, DTYPE* maxArr, unsigned int * nCells, uint64_t * totalCells, unsigned int * binBounaries, const unsigned int CHUNKS)
{

	double DBfraction=1.0;
	unsigned int stride=1.0/DBfraction;

	//histogram the data in the first dimension
	unsigned long int *bins=new unsigned long int [nCells[0]];

	//find the Bin IDs with the cumulative fraction
	double *cumulativeFrac=new double [nCells[0]];

	for (unsigned int i=0; i<nCells[0]; i++)
	{
		cumulativeFrac[i]=0;
		bins[i]=0;
	}

	unsigned int binidx=0;
	unsigned int approxTotalPointsInserted=((*NDdataPoints)[0].size()*1.0)*DBfraction;
	printf("\nApprox points inserted: %u", approxTotalPointsInserted);

	for (unsigned long int i=0; i<(*NDdataPoints)[0].size(); i+=stride)
	{
		binidx=((*NDdataPoints)[0][i]-minArr[0])/epsilon;
		bins[binidx]++;
	}

	
	

	//the first cumulative fraction is the number of points in the bim
	cumulativeFrac[0]=bins[0];
	//get the number
	for (unsigned long int i=1; i<nCells[0]; i++)
	{
		cumulativeFrac[i]=bins[i]+cumulativeFrac[i-1];
		// printf("\nbin: %lu, points in bin: %lu, cumulative frac (num): %f",i,bins[i],cumulativeFrac[i]);
	}


	//convert to fraction
	for (unsigned long int i=0; i<nCells[0]; i++)
	{
		cumulativeFrac[i]=cumulativeFrac[i]/(approxTotalPointsInserted*1.0);
		// printf("\nbin: %lu, points in bin: %lu, cumulative frac (fraction): %f",i,bins[i],cumulativeFrac[i]);
	}


	//for making histogram:
	// printf("\n******histogram\n");
	// for (unsigned long int i=0; i<nCells[0]; i++)
	// {
	// 	printf("\n%lu, %lu",i,bins[i]);
	// }
	// printf("\n******histogram\n");

	//find bin boundaries
	double fractionDataPerPartition=1.0/(CHUNKS*1.0);
	double boundary=fractionDataPerPartition;
	// printf("\nFrac data per partition: %f", fractionDataPerPartition);

	binBounaries[0]=0;
	unsigned int cntBin=1;
	for (unsigned long int i=0; i<nCells[0]-1; i++)
	{
		
		// printf("\nBoundary: %f", boundary);
		// printf("\ni: %lu, cntbin: %u",i,cntBin);
		if (boundary>=cumulativeFrac[i] && boundary<cumulativeFrac[i+1])
		{
			binBounaries[cntBin]=i;
			cntBin++;
			boundary+=fractionDataPerPartition;
		}

			
	}


	binBounaries[CHUNKS]=nCells[0];

	for (unsigned int i=0; i<CHUNKS+1; i++)
	{
		printf("\nBin boundaries (cell of total num cells): %u/%u",binBounaries[i],nCells[0]);
	}


	//////////////////////////////////
	//Up until here is the original algorithm
	//////////////////////////////////

	//Now we do a lookahead by XXX% of each partition width (number of bins) to see if we can find a bin with fewer points
	
	unsigned int * partitionSize=new unsigned int[CHUNKS];


	//The size of each partition in cells:
	for (int i=0; i<CHUNKS; i++)
	{
		partitionSize[i]=binBounaries[i+1]-binBounaries[i];
		// printf("\nPartition size in cells: %u",partitionSize[i]);
	}



	//Now we move the bin boundaries by scanning forward in the next bin to see if we can decrease the number of points in the bin
	//start at bin boundary 1 (not 0, because the first one is the start of the dataset)
	//don't move the last bin boundary either (the end of the dataset)

	const double FRACFORWARD=0.75;
	for (unsigned int i=1; i<CHUNKS; i++)
	{
		unsigned int numCellsCanCheck=partitionSize[i]*FRACFORWARD;
		printf("\nPartitiong size in cells: %u, Num cells can check: %u", partitionSize[i], numCellsCanCheck);fflush(stdout);

		printf("\nOriginal Bin points: %lu", bins[binBounaries[i]]);
		unsigned int idxOfBetterBin=binBounaries[i];
		for (unsigned int j=binBounaries[i]; j<binBounaries[i]+numCellsCanCheck; j++)
		{

			//if the next bin has fewer points, then we move the partition over
			if (bins[j]<bins[idxOfBetterBin])
			{
				idxOfBetterBin=j;
			}
		}

		binBounaries[i]=idxOfBetterBin;
		printf("\nBetter Bin points: %lu", bins[binBounaries[i]]);

	}





	//sanity check that the modified bin boundaries are monotonically increasing: e.g., boundary i must be smaller than i+1

	bool monotonic_flag=0;
	for (unsigned int i=0; i<CHUNKS; i++)
	{
		if(binBounaries[i]>binBounaries[i+1])
		{
			monotonic_flag=1;
		}
	} 

	assert(monotonic_flag==0);


	delete [] partitionSize;
	delete [] bins;
	delete [] cumulativeFrac;



	
		

}


unsigned long int updateAndComputeNumCellsInWindow(unsigned long int * buffer, unsigned int cnt, unsigned long int newVal)
{
	buffer[cnt%4]=newVal;

	unsigned int sum=0;
	for (int i=0; i<4; i++)
	{
		sum+=buffer[i];
	}

	return sum;
}


//generate datasets for each partition
//input the dataset
//output the dataset partitions
//pointIDs in the shadow region for the final merging of the clusters
void partitionDataset(std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<std::vector <DTYPE> > *PartitionedNDdataPoints, 
		std::vector<struct pointChunkLookupArr> *pointChunkMapping, std::vector<unsigned int> * pointsIDsInShadowRegion, 
		std::vector<std::vector <DTYPE> > *NDdataPointsInShadowRegion, DTYPE epsilon, DTYPE* minArr, unsigned int * nCells, 
		unsigned int * binBounaries, const unsigned int CHUNKS)

{

	

	std::vector<unsigned int> shadowBins;
	
	//compute the binids that are in the shadow region
	if (CHUNKS>1)
	{
		//left-most partition doesn't have a region on the left grid edge
		shadowBins.push_back(binBounaries[1]-1);
		shadowBins.push_back(binBounaries[1]);
		
		

		//middle partitions
		for (unsigned int i=1; i<CHUNKS-1; i++)
		{
			//add left shadow regions
			shadowBins.push_back(binBounaries[i]+1);
			shadowBins.push_back(binBounaries[i]+2);

			//right shadow region
			shadowBins.push_back(binBounaries[i+1]-1);
			shadowBins.push_back(binBounaries[i+1]);
		}


		//right-most partition doesn't have a region on the right grid edge
		shadowBins.push_back(binBounaries[CHUNKS-1]+1);
		shadowBins.push_back(binBounaries[CHUNKS-1]+2);

		// for (unsigned int i=0; i<shadowBins.size(); i++)
		// {
		// 	printf("\nShadow region cell id: %u",shadowBins[i] );
		// }


		//check to make sure that the shadow regions are disjoint
		//i.e., no two identical bins in the shadow region
		//if not we need to quit

		bool shadow_disjoint_flag=false;
		for (unsigned int i=1; i<shadowBins.size(); i++)
		{
			if (shadowBins[i]==shadowBins[i-1])
			{
				shadow_disjoint_flag=true;
				printf("\nOverlap in the shadow region!! At least two bins are the same.");
			}
		}

		assert(shadow_disjoint_flag==false);

	}



	////////////////////////////////
	//END COMPUTING SHADOW REGION BINIDS
	////////////////////////////////


	//ORIGINAL-- sequential
	
	// double tstarts3=omp_get_wtime();	
	/*
	for (unsigned int i=0; i<NDdataPoints->size(); i++)
	{
		unsigned int partition=0;
		unsigned int binidx=((*NDdataPoints)[i][0]-minArr[0])/epsilon;
		for (unsigned int j=0; j<CHUNKS+1; j++){
			if (binidx>=binBounaries[j] && binidx<binBounaries[j+1]){
				partition=j;
				break;
			}		
		}

		//add points to the shadow region if it falls in the region
		//These are cells in two cells on each border on the left and right of
		//each partition, except the left and right-most partitions
		for (unsigned int j=0; j<shadowBins.size(); j++){
			if (shadowBins[j]==binidx){
				pointsIDsInShadowRegion->push_back(i);
				NDdataPointsInShadowRegion->push_back((*NDdataPoints)[i]);
				break;
			}
		}

		PartitionedNDdataPoints[partition].push_back((*NDdataPoints)[i]);

		//update mapping for the point in the entire (global) dataset
		pointChunkLookupArr tmp;
		tmp.pointID=i;
		tmp.chunkID=partition;
		tmp.idxInChunk=PartitionedNDdataPoints[partition].size()-1;
		pointChunkMapping->push_back(tmp);

	
	}

	// double tends3=omp_get_wtime();
	// printf("\nTime S3 (original): %f", tends3-tstarts3);
	*/


	//original, don't want this because then its the length of the data points
	// for (int i=0; i<GPUNUMDIM; i++)
	// {
	// 	(*NDdataPointsInShadowRegion)[i].resize((*NDdataPoints)[0].size());
	// }

	
	
	//add points to shadow region
	for (unsigned int i=0; i<(*NDdataPoints)[0].size(); i++)
	{
		
		unsigned int binidx=((*NDdataPoints)[0][i]-minArr[0])/epsilon;
		//add points to the shadow region if it falls in the region
		//These are cells in two cells on each border on the left and right of
		//each partition, except the left and right-most partitions
		auto it = std::lower_bound(shadowBins.begin(), shadowBins.end(),binidx);
		if(!(it == shadowBins.end() || *it != binidx))
		{
			pointsIDsInShadowRegion->push_back(i);
			for (int k=0; k<GPUNUMDIM; k++)
			{
				// NDdataPointsInShadowRegion[k][i].push_back((*NDdataPoints)[k][i]);
				// (*NDdataPointsInShadowRegion)[k][i]=(*NDdataPoints)[k][i];
				(*NDdataPointsInShadowRegion)[k].push_back((*NDdataPoints)[k][i]);

			}
			

			
		}
	}
	
	

	//Step1: for each point, compute its partition -- can be done in parallel
	std::vector<unsigned int> mapPointToPartition; 
	mapPointToPartition.resize((*NDdataPoints)[0].size());

	
	

	//Some threads may get lots of inner loop iterations, need to address load imbalance
	#pragma omp parallel for num_threads(NTHREADS) schedule(guided)
	for (unsigned int i=0; i<(*NDdataPoints)[0].size(); i++)
	{
			unsigned int binidx=((*NDdataPoints)[0][i]-minArr[0])/epsilon;
			for (unsigned int j=0; j<CHUNKS+1; j++){
				if (binidx>=binBounaries[j] && binidx<binBounaries[j+1]){
					mapPointToPartition[i]=j;
					break;
				}		
			}

	}


	

	//step2: 
	//Num threads is either the chunks or the num threads, whichever is lower
	const unsigned int NTHREADSPARTITION=min(NTHREADS,CHUNKS);
	std::vector<unsigned int> prefixSumPartition[CHUNKS];
	std::vector<unsigned int> pointIDsPartition[CHUNKS];
	unsigned int prefixSums[CHUNKS]; 

	for (unsigned int i=0; i<CHUNKS; i++)
	{
		prefixSums[i]=0;
	}

	//prefix sum for each CHUNK
	for (unsigned int i=0; i<(*NDdataPoints)[0].size(); i++)
	{
		unsigned int partition=mapPointToPartition[i];
		prefixSumPartition[partition].push_back(prefixSums[partition]);
		pointIDsPartition[partition].push_back(i);
		prefixSums[partition]++;
	}



	

	//resize arrays
	for (unsigned int i=0; i<CHUNKS; i++)
	{
		PartitionedNDdataPoints[i].resize(GPUNUMDIM);
		for (int k=0; k<GPUNUMDIM; k++)
		{
		PartitionedNDdataPoints[i][k].resize(pointIDsPartition[i].size());
		}
	}
	pointChunkMapping->resize((*NDdataPoints)[0].size());

	

	
	//write the results using prefix sums in parallel

	for (unsigned int i=0; i<CHUNKS; i++)
	{
		#pragma omp parallel for num_threads(NTHREADS) shared(i)
		for (unsigned int j=0; j<pointIDsPartition[i].size(); j++)
		{
			unsigned int idx=prefixSumPartition[i][j];
			unsigned int dataidx=pointIDsPartition[i][j];
			
			for (int k=0; k<GPUNUMDIM; k++)
			{
			PartitionedNDdataPoints[i][k][idx]=(*NDdataPoints)[k][dataidx];	
			}

			//update mapping for the point in the entire (global) dataset
			pointChunkLookupArr tmp;
			tmp.pointID=dataidx;
			tmp.chunkID=i;
			tmp.idxInChunk=idx;
			(*pointChunkMapping)[dataidx]=tmp;	
		}

	}


	//write results in parallel using prefix sums
	// for (unsigned int i=0; i<NDdataPoints->size(); i++)
	// {
	// 	unsigned int partition=mapPointToPartition[i];
	// 	unsigned int idx=prefixSumPartition[partition];
	// 	PartitionedNDdataPoints[partition][idx]=((*NDdataPoints)[i]);

	// 	//update mapping for the point in the entire (global) dataset
	// 	pointChunkLookupArr tmp;
	// 	tmp.pointID=i;
	// 	tmp.chunkID=partition;
	// 	tmp.idxInChunk=PartitionedNDdataPoints[partition].size()-1;
	// 	(*pointChunkMapping)[i]=tmp;	
	// }




} //end function




//generate datasets for each partition
//input the dataset
//output the dataset partitions
//pointIDs in the shadow region for the final merging of the clusters
void partitionDatasetORIGINAL(std::vector<std::vector <DTYPE> > *NDdataPoints, std::vector<std::vector <DTYPE> > *PartitionedNDdataPoints, 
		std::vector<struct pointChunkLookupArr> *pointChunkMapping, std::vector<unsigned int> * pointsIDsInShadowRegion, 
		std::vector<std::vector <DTYPE> > *NDdataPointsInShadowRegion, DTYPE epsilon, DTYPE* minArr, unsigned int * nCells, 
		unsigned int * binBounaries, const unsigned int CHUNKS)

{

	pointChunkMapping->resize(NDdataPoints->size());

	std::vector<unsigned int> shadowBins;
	
	//compute the binids that are in the shadow region
	if (CHUNKS>1)
	{
		//left-most partition doesn't have a region on the left grid edge
		shadowBins.push_back(binBounaries[1]-1);
		shadowBins.push_back(binBounaries[1]);
		
		

		//middle partitions
		for (unsigned int i=1; i<CHUNKS-1; i++)
		{
			//add left shadow regions
			shadowBins.push_back(binBounaries[i]+1);
			shadowBins.push_back(binBounaries[i]+2);

			//right shadow region
			shadowBins.push_back(binBounaries[i+1]-1);
			shadowBins.push_back(binBounaries[i+1]);
		}


		//right-most partition doesn't have a region on the right grid edge
		shadowBins.push_back(binBounaries[CHUNKS-1]+1);
		shadowBins.push_back(binBounaries[CHUNKS-1]+2);

		// for (unsigned int i=0; i<shadowBins.size(); i++)
		// {
		// 	printf("\nShadow region cell id: %u",shadowBins[i] );
		// }


	}



	////////////////////////////////
	//END COMPUTING SHADOW REGION BINIDS
	////////////////////////////////


	//ORIGINAL-- sequential
	
	// double tstarts3=omp_get_wtime();	
	/*
	for (unsigned int i=0; i<NDdataPoints->size(); i++)
	{
		unsigned int partition=0;
		unsigned int binidx=((*NDdataPoints)[i][0]-minArr[0])/epsilon;
		for (unsigned int j=0; j<CHUNKS+1; j++){
			if (binidx>=binBounaries[j] && binidx<binBounaries[j+1]){
				partition=j;
				break;
			}		
		}

		//add points to the shadow region if it falls in the region
		//These are cells in two cells on each border on the left and right of
		//each partition, except the left and right-most partitions
		for (unsigned int j=0; j<shadowBins.size(); j++){
			if (shadowBins[j]==binidx){
				pointsIDsInShadowRegion->push_back(i);
				NDdataPointsInShadowRegion->push_back((*NDdataPoints)[i]);
				break;
			}
		}

		PartitionedNDdataPoints[partition].push_back((*NDdataPoints)[i]);

		//update mapping for the point in the entire (global) dataset
		pointChunkLookupArr tmp;
		tmp.pointID=i;
		tmp.chunkID=partition;
		tmp.idxInChunk=PartitionedNDdataPoints[partition].size()-1;
		pointChunkMapping->push_back(tmp);

	
	}

	// double tends3=omp_get_wtime();
	// printf("\nTime S3 (original): %f", tends3-tstarts3);
	*/

	
	
	//add points to shadow region
	for (unsigned int i=0; i<NDdataPoints->size(); i++)
	{
		
		unsigned int binidx=((*NDdataPoints)[i][0]-minArr[0])/epsilon;
		//add points to the shadow region if it falls in the region
		//These are cells in two cells on each border on the left and right of
		//each partition, except the left and right-most partitions
		auto it = std::lower_bound(shadowBins.begin(), shadowBins.end(),binidx);
		if(!(it == shadowBins.end() || *it != binidx))
		{
			pointsIDsInShadowRegion->push_back(i);
			NDdataPointsInShadowRegion->push_back((*NDdataPoints)[i]);
		}
	}
	
	

	//Step1: for each point, compute its partition -- can be done in parallel
	std::vector<unsigned int> mapPointToPartition; 
	mapPointToPartition.resize(NDdataPoints->size());

	
	//Some threads may get lots of inner loop iterations, need to address load imbalance
	#pragma omp parallel for num_threads(NTHREADS) schedule(guided)
	for (unsigned int i=0; i<NDdataPoints->size(); i++)
	{
			unsigned int binidx=((*NDdataPoints)[i][0]-minArr[0])/epsilon;
			for (unsigned int j=0; j<CHUNKS+1; j++){
				if (binidx>=binBounaries[j] && binidx<binBounaries[j+1]){
					mapPointToPartition[i]=j;
					break;
				}		
			}

	}

	//step2: each thread is responsible for a different partition and adding the value to the correct partition
	//Therefore, we don't need critical sections for parallel updates

	//Num threads is either the chunks or the num threads, whichever is lower
	const unsigned int NTHREADSPARTITION=min(NTHREADS,CHUNKS);

	#pragma omp parallel for num_threads(NTHREADSPARTITION) 
	for (unsigned int k=0; k<CHUNKS; k++)
	{
		for (unsigned int i=0; i<NDdataPoints->size(); i++)
		{
			if (mapPointToPartition[i]==k)
			{
			
			unsigned int partition=k;	


			PartitionedNDdataPoints[partition].push_back((*NDdataPoints)[i]);
			//update mapping for the point in the entire (global) dataset
			pointChunkLookupArr tmp;
			tmp.pointID=i;
			tmp.chunkID=partition;
			tmp.idxInChunk=PartitionedNDdataPoints[partition].size()-1;
			(*pointChunkMapping)[i]=tmp;
			}
		}
		
	}

	



} //end function


void populateNDGridIndexAndLookupArrayParallel(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, 
	struct gridCellLookup ** gridCellLookupArr, struct grid ** index, unsigned int * indexLookupArr,  DTYPE* minArr, unsigned int * nCells, uint64_t totalCells, unsigned int * nNonEmptyCells)
{

	/////////////////////////////////
	//Populate grid lookup array
	//and corresponding indicies in the lookup array
	/////////////////////////////////
	printf("\n\n*****************************\nPopulating Grid Index and lookup array:\n*****************************\n");
	// printf("\nSize of dataset: %lu", NDdataPoints->size());


	///////////////////////////////
	//First, we need to figure out how many non-empty cells there will be
	//For memory allocation
	//Need to do a scan of the dataset and calculate this
	//Also need to keep track of the list of unique linear grid cell IDs for inserting into the grid
	///////////////////////////////
	
	//instead of using a set, do data deduplication manually by sorting
	std::vector<uint64_t> allGridCellLinearIds;

	std::vector<uint64_t>uniqueGridCellLinearIdsVect; //for random access

	printf("\ndatasize: %lu", (*NDdataPoints)[0].size());fflush(stdout);
	

	for (int i=0; i<(*NDdataPoints)[0].size(); i++){
		unsigned int tmpNDCellIdx[NUMINDEXEDDIM];
		for (int j=0; j<NUMINDEXEDDIM; j++){
			tmpNDCellIdx[j]=(((*NDdataPoints)[j][i]-minArr[j])/epsilon);
		}
		uint64_t linearID=getLinearID_nDimensions(tmpNDCellIdx, nCells, NUMINDEXEDDIM);

		allGridCellLinearIds.push_back(linearID);
	}

	printf("\nSorting linear ids");	cout.flush();
	//call parallel sort, but is in my custom function because nvcc conflicts with parallel mode extensions
	sortLinearIds(&allGridCellLinearIds);
	
	
	uniqueGridCellLinearIdsVect.push_back(allGridCellLinearIds[0]);
	for (int i=1; i<allGridCellLinearIds.size(); i++)
	{
		if (allGridCellLinearIds[i]!=allGridCellLinearIds[i-1])
		{
			uniqueGridCellLinearIdsVect.push_back(allGridCellLinearIds[i]);
		}
	}
	



	
	

	//The size of the vectors can be large, so we only use 4 (not NTHREADS)
	const int NTHREADSGridElemIds=4;
	printf("\nSize tmp vectors for threads: %f (GiB)", (uniqueGridCellLinearIdsVect.size()*NTHREADSGridElemIds*sizeof(std::vector<uint64_t>))/(1024*1024*1024.0));cout.flush();

	//temp vectors so that arrays can concurrently write without mutual exclusion
	//memory allocation can take quite a long time so we parallelize it
	
	std::vector<uint64_t> ** gridElemIDsTmpThreads;	
	gridElemIDsTmpThreads=new std::vector<uint64_t>*[NTHREADSGridElemIds]; 		
	#pragma omp parallel for num_threads(NTHREADSGridElemIds)
	for (int i=0; i<NTHREADSGridElemIds; i++)
	{
		gridElemIDsTmpThreads[i]=new std::vector<uint64_t>[uniqueGridCellLinearIdsVect.size()];
	}

	


	//threads will store the ids in here
	std::vector<uint64_t> * gridElemIDs;
	gridElemIDs = new std::vector<uint64_t>[uniqueGridCellLinearIdsVect.size()];

	
	/*
	

	std::set<unsigned int> NDArrMaskParallel[NTHREADS][NUMINDEXEDDIM];

	#pragma omp parallel num_threads(NTHREADS)   shared(NDArrMaskParallel)
	{
		int tid=omp_get_thread_num();
		#pragma omp for schedule(guided)
		for (int i=0; i<(*NDdataPoints)[0].size(); i++){
			unsigned int tmpNDCellID[NUMINDEXEDDIM];
			for (int j=0; j<NUMINDEXEDDIM; j++){
				tmpNDCellID[j]=(((*NDdataPoints)[j][i]-minArr[j])/epsilon);

				//add value to the ND array mask
				NDArrMaskParallel[tid][j].insert(tmpNDCellID[j]);
				}	
			}
	}	

	//now sequentially we insert into the NDArrMask from each set produced by each thread
	for (int i=0; i<NTHREADS; i++)
	{
			for (int j=0; j<NUMINDEXEDDIM; j++){
				std::set<unsigned int>::iterator it = NDArrMaskParallel[i][j].begin();
				for (auto it=NDArrMaskParallel[i][j].begin(); it != NDArrMaskParallel[i][j].end(); ++it)
				{
						NDArrMask[j].insert(*it);
				}
			}

	}

	*/
	
	

	
	


	
	#pragma omp parallel num_threads(NTHREADSGridElemIds)  
	{
		int tid=omp_get_thread_num();
		#pragma omp for schedule(guided)
		for (int i=0; i<(*NDdataPoints)[0].size(); i++){
			
			
			unsigned int tmpNDCellID[NUMINDEXEDDIM];
			for (int j=0; j<NUMINDEXEDDIM; j++){
				tmpNDCellID[j]=(((*NDdataPoints)[j][i]-minArr[j])/epsilon);			
			}

			//get the linear id of the cell
			uint64_t linearID=getLinearID_nDimensions(tmpNDCellID, nCells, NUMINDEXEDDIM);
			//printf("\nlinear id: %d",linearID);
			if (linearID > totalCells){

				printf("\n\nERROR Linear ID is: %lu, total cells is only: %lu\n\n", linearID, totalCells);
			}

			//find the index in gridElemIds that corresponds to this grid cell linear id
			
			vector<uint64_t>::iterator lower=std::lower_bound(uniqueGridCellLinearIdsVect.begin(), uniqueGridCellLinearIdsVect.end(),linearID);
			uint64_t gridIdx=lower - uniqueGridCellLinearIdsVect.begin();
			
			//original
			// gridElemIDs[gridIdx].push_back(i);

			gridElemIDsTmpThreads[tid][gridIdx].push_back(i);
			
		}
	}

	


	//copy the grid elem ids from the temp vectors 
	//ORIGINAL
	// for (int i=0; i<NTHREADS; i++)
	// {
	// 	for (int j=0; j<uniqueGridCellLinearIdsVect.size(); j++)
	// 	{
	// 		for (int k=0; k<gridElemIDsTmpThreads[i][j].size(); k++)
	// 		{
	// 				gridElemIDs[j].push_back(gridElemIDsTmpThreads[i][j][k]);				
	// 		}	
	// 	}
	// }

	for (int i=0; i<NTHREADSGridElemIds; i++)
	{
		#pragma omp parallel for shared(i) schedule(guided) num_threads(4)
		for (int j=0; j<uniqueGridCellLinearIdsVect.size(); j++)
		{
			std::copy(gridElemIDsTmpThreads[i][j].begin(), gridElemIDsTmpThreads[i][j].end(), std::back_inserter(gridElemIDs[j]));
		}
	}



	

	





	
	///////////////////////////////
	//Here we fill a temporary index with points, and then copy the non-empty cells to the actual index
	///////////////////////////////
	printf("\nSize of tmp index: %f (GiB)",((sizeof(grid)*uniqueGridCellLinearIdsVect.size()))/(1024*1024*1024.0));cout.flush();
	struct grid * tmpIndex=new grid[uniqueGridCellLinearIdsVect.size()];

	int cnt=0;

	

	//populate temp index and lookup array

	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++)
	{
			tmpIndex[i].indexmin=cnt;
			for (int j=0; j<gridElemIDs[i].size(); j++)
			{
				if (j>(((*NDdataPoints)[0].size()-1)))
				{
					printf("\n\n***ERROR Value of a data point is larger than the dataset! %d\n\n", j);
					return;
				}
				indexLookupArr[cnt]=gridElemIDs[i][j]; 
				cnt++;
			}
			tmpIndex[i].indexmax=cnt-1;
	}

	// printf("\nExiting grid populate method early!");
	// return;

	printf("\nFull cells: %d (%f, fraction full)",(unsigned int)uniqueGridCellLinearIdsVect.size(), uniqueGridCellLinearIdsVect.size()*1.0/double(totalCells));
	printf("\nEmpty cells: %ld (%f, fraction empty)",totalCells-(unsigned int)uniqueGridCellLinearIdsVect.size(), (totalCells-uniqueGridCellLinearIdsVect.size()*1.0)/double(totalCells));
	
	*nNonEmptyCells=uniqueGridCellLinearIdsVect.size();


	printf("\nSize of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f", (double)sizeof(struct grid)*(totalCells)/(1024.0*1024.0*1024.0));
	printf("\nSize of compressed index to be sent to GPU (GiB) , excluding the data and grid lookup arr: %f", (double)sizeof(struct grid)*(uniqueGridCellLinearIdsVect.size()*1.0)/(1024.0*1024.0*1024.0));


	

	//////////////
	
	/////////////////////////////////////////
	//copy the tmp index into the actual index that only has the non-empty cells


	//allocate memory for the index that will be sent to the GPU
	*index=new grid[uniqueGridCellLinearIdsVect.size()];
	*gridCellLookupArr= new struct gridCellLookup[uniqueGridCellLinearIdsVect.size()];

	

	// cmpStruct theStruct(*NDdataPoints);
	

	
	for (int i=0; i<uniqueGridCellLinearIdsVect.size(); i++){
			(*index)[i].indexmin=tmpIndex[i].indexmin;
			(*index)[i].indexmax=tmpIndex[i].indexmax;
			(*gridCellLookupArr)[i].idx=i;
			(*gridCellLookupArr)[i].gridLinearID=uniqueGridCellLinearIdsVect[i];
	}
	
	

	printf("\nWhen copying from entire index to compressed index: number of non-empty cells: %lu",uniqueGridCellLinearIdsVect.size());
		
	
	
	for (int i=0; i<NTHREADSGridElemIds; i++)
	{
		delete [] gridElemIDsTmpThreads[i];
	}
	delete [] gridElemIDsTmpThreads; 

	delete [] gridElemIDs;

	delete [] tmpIndex;
		


} //end function populate grid index and lookup array



//reorders the input data by variance of each dimension
void ReorderByDimension(std::vector<std::vector <DTYPE> > *NDdataPoints)
{
	
	double tstart_sort=omp_get_wtime();
	DTYPE sums[GPUNUMDIM];
	DTYPE average[GPUNUMDIM];
	struct dim_reorder_sort dim_variance[GPUNUMDIM];
	for (int i=0; i< GPUNUMDIM; i++){
		sums[i]=0;
		average[i]=0;
	}

	DTYPE greatest_variance=0;
	int greatest_variance_dim=0;

	
	int sample=100;
	DTYPE inv_sample=1.0/(sample*1.0);
	printf("\nCalculating variance based on on the following fraction of pts: %f",inv_sample);
	double tvariancestart=omp_get_wtime();
		//calculate the variance in each dimension	
		for (int i=0; i<GPUNUMDIM; i++)
		{
			//first calculate the average in the dimension:
			//only use every nth point (sample variable)
			for (int j=0; j<(*NDdataPoints)[0].size(); j+=sample)
			{
			sums[i]+=(*NDdataPoints)[i][j];
			}


			average[i]=(sums[i])/((*NDdataPoints)[0].size()*inv_sample);
			// printf("\nAverage in dim: %d, %f",i,average[i]);

			//Next calculate the std. deviation
			sums[i]=0; //reuse this for other sums
			for (int j=0; j<(*NDdataPoints)[0].size(); j+=sample)
			{
			sums[i]+=(((*NDdataPoints)[i][j])-average[i])*(((*NDdataPoints)[i][j])-average[i]);
			}
			
			dim_variance[i].variance=sums[i]/((*NDdataPoints)[0].size()*inv_sample);
			dim_variance[i].dim=i;
			
			// printf("\nDim:%d, variance: %f",dim_variance[i].dim,dim_variance[i].variance);

			if(greatest_variance<dim_variance[i].variance)
			{
				greatest_variance=dim_variance[i].variance;
				greatest_variance_dim=i;
			}
		}


	// double tvarianceend=omp_get_wtime();
	// printf("\nTime to compute variance only: %f",tvarianceend - tvariancestart);
	//sort based on variance in dimension:

	// double tstartsortreorder=omp_get_wtime();
	std::sort(dim_variance,dim_variance+GPUNUMDIM,compareByDimVariance); 	

	for (int i=0; i<GPUNUMDIM; i++)
	{
		printf("\nReodering dimension by: dim: %d, variance: %f",dim_variance[i].dim,dim_variance[i].variance);
	}

	printf("\nDimension with greatest variance: %d",greatest_variance_dim);

	//copy the database
	// double * tmp_database= (double *)malloc(sizeof(double)*(*NDdataPoints).size()*(GPUNUMDIM));  
	// std::copy(database, database+((*DBSIZE)*(GPUNUMDIM)),tmp_database);
	std::vector<std::vector <DTYPE> > tmp_database;

	//copy data into temp database
	tmp_database=(*NDdataPoints);

	
	
	#pragma omp parallel for num_threads(5) shared(NDdataPoints, tmp_database)
	for (int j=0; j<GPUNUMDIM; j++){

		int originDim=dim_variance[j].dim;	
		for (int i=0; i<(*NDdataPoints)[0].size(); i++)
		{	
			(*NDdataPoints)[j][i]=tmp_database[originDim][i];
		}
	}

	double tend_sort=omp_get_wtime();
	// double tendsortreorder=omp_get_wtime();
	// printf("\nTime to sort/reorder only: %f",tendsortreorder-tstartsortreorder);
	double timecomponent=tend_sort - tstart_sort;
	printf("\nTime to reorder cols by variance (this gets added to the time because its an optimization): %f",timecomponent);
	
}


