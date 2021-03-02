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

//precompute direct neighbors with the GPU:
#include <cuda_runtime.h>
#include <cuda.h>
#include "structs.h"
#include <stdio.h>
#include "kernel.h"
#include <math.h>
#include "GPU.h"
#include <algorithm>
#include "omp.h"
#include <queue>
#include <unistd.h>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)


//for warming up GPU:
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>


//elements for the result set
//FOR A SINGLE KERNEL INVOCATION
//NOT FOR THE BATCHED ONE
#define BUFFERELEM 300000000 //400000000-original (when removing the data from the device before putting it back for the sort)

//FOR THE BATCHED EXECUTION:
//#define BATCHTOTALELEM 1200000000 //THE TOTAL SIZE ALLOCATED ON THE HOST
//THE NUMBER OF BATCHES AND THE SIZE OF THE BUFFER FOR EACH KERNEL EXECUTION ARE NOT RELATED TO THE TOTAL NUMBER
//OF ELEMENTS (ABOVE).
// #define NUMBATCHES 20
// #define BATCHBUFFERELEM 100000000 //THE SMALLER SIZE ALLOCATED ON THE DEVICE FOR EACH KERNEL EXECUTION 






using namespace std;


//sort ascending
bool compareByPointValue(const key_val_sort &a, const key_val_sort &b)
{
    return a.value_at_dim < b.value_at_dim;
}

//We use the query points to estimate the number of batches because if we use densebox
//we want to only sample the query points that will be searched for
//otherwise, we will sample the points eliminated by densebox, which will have far more neighbors on average
unsigned long long callGPUBatchEst(unsigned int * DBSIZE, unsigned int * dev_queryPts, unsigned int NqueryPts, unsigned int NrefPts, bool refPointsPopulated, DTYPE* dev_database, DTYPE* dev_epsilon, struct grid * dev_grid, 
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr, 
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * retNumBatches, unsigned int * retGPUBufferSize)
{



	//CUDA error code:
	cudaError_t errCode;

	printf("\n\n***********************************\nEstimating Batches:");
	cout<<"\n** BATCH ESTIMATOR: Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();



//////////////////////////////////////////////////////////
	//ESTIMATE THE BUFFER SIZE AND NUMBER OF BATCHES ETC BY COUNTING THE NUMBER OF RESULTS
	//TAKE A SAMPLE OF THE DATA POINTS, NOT ALL OF THEM
	//Use sampleRate for this
	/////////////////////////////////////////////////////////

	
	//We use a dynamic sampling rate. As the number of data points increases, fewer need to be sampled.
	//For small datasets, need to make sure we sample enough points
	//Sample 1% for <=2 million points
	//Scale from here


	//Parameters for the batch size estimation.
	double sampleRate=0.01; //sample 1% of the points in the dataset sampleRate=0.01. 
					
	//Decrease fraction of points sampled when dataset is >2M points
	//We fix sampling 40000 points regardless of the dataset size when there are at least 2M points
	if (*DBSIZE>(2000000))
	{
		sampleRate=40000.0/(*DBSIZE);
		printf("\nDynamic sample rate: %f", sampleRate);
	}


	int offsetRate=1.0/sampleRate;
	printf("\nOffset: %d", offsetRate);



	/////////////////
	//N GPU threads
	////////////////

	
	

	unsigned int * dev_N_batchEst; 
	

	unsigned int * N_batchEst; 
	N_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	// *N_batchEst=*DBSIZE*sampleRate;
	// *N_batchEst=NqueryPts*sampleRate;
	*N_batchEst=NqueryPts/offsetRate;
	


	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_N_batchEst Got error with code " << errCode << endl; 
	}	

	//copy N to device 
	//N IS THE NUMBER OF THREADS
	errCode=cudaMemcpy( dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: N batchEST Got error with code " << errCode << endl; 
	}


	/////////////
	//count the result set size 
	////////////

	unsigned int * dev_cnt_batchEst; 
	

	unsigned int * cnt_batchEst; 
	cnt_batchEst=(unsigned int*)malloc(sizeof(unsigned int));
	*cnt_batchEst=0;


	//allocate on the device
	errCode=cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}	

	//copy cnt to device 
	errCode=cudaMemcpy( dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt_batchEst Got error with code " << errCode << endl; 
	}


	
	//////////////////
	//SAMPLE OFFSET - TO SAMPLE THE DATA TO ESTIMATE THE TOTAL NUMBER OF KEY VALUE PAIRS
	/////////////////

	//offset into the database when batching the results
	unsigned int * sampleOffset; 
	sampleOffset=(unsigned int*)malloc(sizeof(unsigned int));
	*sampleOffset=offsetRate;


	unsigned int * dev_sampleOffset; 
	

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: sample offset Got error with code " << errCode << endl; 
	}

	//copy offset to device 
	errCode=cudaMemcpy( dev_sampleOffset, sampleOffset, sizeof(unsigned int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_sampleOffset Got error with code " << errCode << endl; 
	}


	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	//debug values
	unsigned int * dev_debug1; 
	

	unsigned int * dev_debug2; 
	
	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 alloc -- error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 alloc -- error with code " << errCode << endl; 
	}		

	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 copy to device -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 copy to device -- with code " << errCode << endl; 
	}




	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////	



	const int TOTALBLOCKSBATCHEST=ceil((*N_batchEst)/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKSBATCHEST);

	// __global__ void kernelNDGridIndexBatchEstimator(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	// unsigned int * sampleOffset, double * database, double *epsilon, struct grid * index, unsigned int * indexLookupArr, 
	// struct gridCellLookup * gridCellLookupArr, double * minArr, unsigned int * nCells, unsigned int * cnt, 
	// unsigned int * nNonEmptyCells)

	kernelNDGridIndexBatchEstimator<<< TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_debug1, dev_debug2, dev_N_batchEst, 
		dev_sampleOffset, dev_database, dev_queryPts, dev_epsilon, dev_grid, dev_indexLookupArr, 
		dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_cnt_batchEst, dev_nNonEmptyCells);
		cout<<"\n** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: "<<cudaGetLastError();
		// find the size of the number of results
		errCode=cudaMemcpy( cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if(errCode != cudaSuccess) {
		cout << "\nError: getting cnt for batch estimate from GPU Got error with code " << errCode << endl; 
		}
		else
		{
			printf("\nGPU: result set size for estimating the number of batches (sampled): %u",*cnt_batchEst);
		}

	uint64_t estimatedNeighbors=(uint64_t)*cnt_batchEst*(uint64_t)offsetRate;		
	
	//if no reference points, use the 1-epsilon number of neighbors epsilon
	if (refPointsPopulated==false)
	{	
	printf("\nFrom gpu cnt: %d, offset rate: %d, estimated neighbors: %lu", *cnt_batchEst,offsetRate, estimatedNeighbors);
	}
	//if reference points need to use the 1.5xepsilon in the calculation
	else
	{
		//volume ratio of 1 eps points to 1.5 eps ref points
		double volRatio=2.25;

		//avg neighbors per point from 1-epsilon searches
		double avgNeighborsPerPoint=(estimatedNeighbors*1.0)/(NqueryPts*1.0);
		//total neighbors for regular 1-epsilon data points
		double numNeighborsDataPnts=avgNeighborsPerPoint*((NqueryPts*1.0)-(NrefPts*1.0));
		//total neighbors for regular 1.5-epsilon reference points
		double numNeighborsRefPnts=avgNeighborsPerPoint*(NrefPts*1.0)*volRatio;
		estimatedNeighbors=(uint64_t)(numNeighborsDataPnts+numNeighborsRefPnts);
		printf("\nFrom gpu cnt: %d, offset rate: %d, estimated neighbors: %lu (Ref points-on, including ref. point geometry in estimate)", *cnt_batchEst,offsetRate, estimatedNeighbors);
	}



	// double fractionQueryPts=NqueryPts*1.0/(*DBSIZE)*1.0;
	// estimatedNeighbors=estimatedNeighbors*(fractionQueryPts);	
	// printf("\nEstimated neighbors considering the fraction (%f) of query points: %ld", fractionQueryPts, estimatedNeighbors);
	//initial
	


	
	unsigned int GPUBufferSize=GPUBUFFERSIZE;

	double alpha=0.25; //overestimation factor
	//Need to overestimate more because of the reference points that have eps=1.5
	//The ratio of an eps=1.5 to eps=1.0 search radius in area: 2.25

	
	uint64_t estimatedTotalSizeWithAlpha=estimatedNeighbors*(1.0+alpha*1.0);
	printf("\nEstimated total result set size: %lu", estimatedNeighbors);
	printf("\nEstimated total result set size (with Alpha %f): %lu", alpha,estimatedTotalSizeWithAlpha);	
	

	unsigned int numBatches=ceil((estimatedTotalSizeWithAlpha*1.0)/((uint64_t)GPUBufferSize*1.0));

	if (estimatedNeighbors<(GPUBufferSize*GPUSTREAMS))
	{
		
		if (numBatches<GPUSTREAMS)
		{
			numBatches=GPUSTREAMS;
		}
		printf("\nSmall buffer size, but still set %d batches and %u elems",GPUSTREAMS, GPUBufferSize);
	}	

	// if (estimatedNeighbors<(GPUBufferSize*GPUSTREAMS))
	// {
	// 	printf("\nSmall buffer size, increasing alpha to: %f",alpha*3.0);
	// 	GPUBufferSize=estimatedNeighbors*(1.0+(alpha*2.0))/(GPUSTREAMS);		//we do 2*alpha for small datasets because the
	// 																	//sampling will be worse for small datasets
	// 																	//but we fix the 3 streams still (thats why divide by 3).			
	// }

	

	printf("\nNumber of batches: %d, buffer size: %d", numBatches, GPUBufferSize);

	*retNumBatches=numBatches;
	*retGPUBufferSize=GPUBufferSize;
		

	printf("\nEnd Batch Estimator\n***********************************\n");




	cudaFree(dev_cnt_batchEst);	
	cudaFree(dev_N_batchEst);
	cudaFree(dev_sampleOffset);







return estimatedTotalSizeWithAlpha;

}









//modified from: makeDistanceTableGPUGridIndexBatchesAlternateTest

//refPointOffset- the number of "real" data points, not the reference points
void distanceTableNDGridBatches(std::vector<std::vector<DTYPE> > * NDdataPoints, std::vector<unsigned int> *queryVect, DTYPE* epsilon, struct grid * index, 
	struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, DTYPE* minArr, unsigned int * nCells, 
	unsigned int * indexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, 
	uint64_t * totalNeighbors,  CTYPE* workCounts, 
	unsigned int refPointOffset, int * pointIDKey[NUMGPU][GPUSTREAMS], int * pointInDistValue[NUMGPU][GPUSTREAMS], bool refPointsPopulated, int gpuid)
{

	// cudaSetDevice(gpuid);	
	// gpuid=0; //test
	cudaSetDevice(gpuid);	


	double tKernelResultsStart=omp_get_wtime();

	//CUDA error code:
	cudaError_t errCode;


	cout<<"\n** Sometimes the GPU will error on a previous execution and you won't know. \nLast error start of function: "<<cudaGetLastError();


	
	



	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * DBSIZE;
	DBSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*DBSIZE=(*NDdataPoints)[0].size();
	
	printf("\nIn main GPU method: DBSIZE is: %u",*DBSIZE);cout.flush();

	
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM));  
	DTYPE* dev_database;  
	//dev_database= (DTYPE*)malloc(sizeof(DTYPE)*(*DBSIZE)*(GPUNUMDIM));  
	
		
	
	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	
	


	//copy the database from the ND vector to the array:
	// for (int i=0; i<(*DBSIZE); i++){
	// 	std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*(GPUNUMDIM)));
	// }

	// unsigned int cntelem=0;

	#pragma omp parallel for num_threads(3)
	for (unsigned int i=0; i<(*DBSIZE); i++){
		for (unsigned int j=0; j<GPUNUMDIM; j++)
		{
			unsigned int idx=(i*GPUNUMDIM)+j;
			database[idx]=(*NDdataPoints)[j][i];
			
		}
	}



		//copy database to the device
	errCode=cudaMemcpy(dev_database, database, sizeof(DTYPE)*(GPUNUMDIM)*(*DBSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database2 Got error with code " << errCode << endl; 
	}




	//printf("\n size of database: %d",N);



	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	//////////////////////////////////
	//Copy the query points to the GPU
	/////////////////////////////////
	unsigned int * QUERYSIZE;
	QUERYSIZE=(unsigned int*)malloc(sizeof(unsigned int));
	*QUERYSIZE=queryVect->size();


	printf("\nIn GPU function num. query points: %u",*QUERYSIZE);

	unsigned int * queryPts= (unsigned int *)malloc(sizeof(unsigned int )*(*QUERYSIZE));  
	// unsigned int * dev_queryPts= (unsigned int *)malloc(sizeof(unsigned int )*(*QUERYSIZE));  
	unsigned int * dev_queryPts;

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_queryPts, sizeof(unsigned int)*(*QUERYSIZE));		
	if(errCode != cudaSuccess) {
	cout << "\nError: queryPts alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy the query point ids from the vector to the array:
	std::copy((*queryVect).begin(), (*queryVect).end(), queryPts);
	


	//copy database to the device
	errCode=cudaMemcpy(dev_queryPts, queryPts, sizeof(unsigned int)*(*QUERYSIZE), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: queryPts Got error with code " << errCode << endl; 
	}


	// for (int i=0; i<*QUERYSIZE; i++)
	// {
	// 	printf("\nQuery point: %u",queryPts[i]);
	// }


	//number of query points
	unsigned int * dev_sizequerypts;
	// dev_sizequerypts=(unsigned int*)malloc(sizeof(unsigned int));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_sizequerypts, sizeof(unsigned int));		
	if(errCode != cudaSuccess) {
	cout << "\nError: size query pts alloc -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy the number of query points to the devid
	errCode=cudaMemcpy(dev_sizequerypts, QUERYSIZE, sizeof(unsigned int), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: size query pts got error with code " << errCode << endl; 
	}


	//////////////////////////////////
	//END Copy the query points to the GPU
	/////////////////////////////////

	

	struct grid * dev_grid;
	// dev_grid=(struct grid*)malloc(sizeof(struct grid)*(*nNonEmptyCells));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_grid, sizeof(struct grid)*(*nNonEmptyCells));	
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index -- error with code " << errCode << endl; cout.flush(); 
	}


	//copy grid index to the device:
	errCode=cudaMemcpy(dev_grid, index, sizeof(struct grid)*(*nNonEmptyCells), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: grid index copy to device -- error with code " << errCode << endl; 
	}	

	printf("\nSize of index sent to GPU (MiB): %f", (DTYPE)sizeof(struct grid)*(*nNonEmptyCells)/(1024.0*1024.0));


	///////////////////////////////////
	//END COPY THE INDEX TO THE GPU
	///////////////////////////////////


	///////////////////////////////////
	//COPY THE LOOKUP ARRAY TO THE DATA ELEMS TO THE GPU
	///////////////////////////////////

	

	


	
	//The lookup array is only for the "real" data points and not the reference points
	//Need to pass in the number of "real" data points


	unsigned int * dev_indexLookupArr;
	// dev_indexLookupArr=(unsigned int*)malloc(sizeof(unsigned int)*(refPointOffset));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_indexLookupArr, sizeof(unsigned int)*(refPointOffset));
	if(errCode != cudaSuccess) {
	cout << "\nError: lookup array allocation -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_indexLookupArr, indexLookupArr, sizeof(unsigned int)*(refPointOffset), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: copy lookup array to device -- error with code " << errCode << endl; 
	}	


	
	

	///////////////////////////////////
	//END COPY THE LOOKUP ARRAY TO THE DATA ELEMS TO THE GPU
	///////////////////////////////////



	///////////////////////////////////
	//COPY THE GRID CELL LOOKUP ARRAY 
	///////////////////////////////////

	
	
						
	struct gridCellLookup * dev_gridCellLookupArr;
	// dev_gridCellLookupArr=(struct gridCellLookup*)malloc(sizeof(struct gridCellLookup)*(*nNonEmptyCells));

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_gridCellLookupArr, sizeof(struct gridCellLookup)*(*nNonEmptyCells));
	if(errCode != cudaSuccess) {
	cout << "\nError: copy grid cell lookup array allocation -- error with code " << errCode << endl; cout.flush(); 
	}

	//copy lookup array to the device:
	errCode=cudaMemcpy(dev_gridCellLookupArr, gridCellLookupArr, sizeof(struct gridCellLookup)*(*nNonEmptyCells), cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: copy grid cell lookup array to device -- error with code " << errCode << endl; 
	}	

	

	///////////////////////////////////
	//END COPY THE GRID CELL LOOKUP ARRAY 
	///////////////////////////////////





	
	///////////////////////////////////
	//COPY GRID DIMENSIONS TO THE GPU
	//THIS INCLUDES THE NUMBER OF CELLS IN EACH DIMENSION, 
	//AND THE STARTING POINT OF THE GRID IN THE DIMENSIONS 
	///////////////////////////////////

	//minimum boundary of the grid:
	DTYPE* dev_minArr;
	// dev_minArr=(DTYPE*)malloc(sizeof(DTYPE)*(NUMINDEXEDDIM));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_minArr, sizeof(DTYPE)*(NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc minArr -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_minArr, minArr, sizeof(DTYPE)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: Copy minArr to device -- error with code " << errCode << endl; 
	}	


	//number of cells in each dimension
	unsigned int * dev_nCells;
	// dev_nCells=(unsigned int*)malloc(sizeof(unsigned int)*(NUMINDEXEDDIM));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_nCells, sizeof(unsigned int)*(NUMINDEXEDDIM));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc nCells -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_nCells, nCells, sizeof(unsigned int)*(NUMINDEXEDDIM), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: Copy nCells to device -- error with code " << errCode << endl; 
	}	


	///////////////////////////////////
	//END COPY GRID DIMENSIONS TO THE GPU
	///////////////////////////////////




	///////////////////////////////////
	//COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////

	//total size of the result set as it's batched
	//this isnt sent to the GPU
	// unsigned int * totalResultSetCnt;
	// totalResultSetCnt=(unsigned int*)malloc(sizeof(unsigned int));
	// *totalResultSetCnt=0;

	//count values - for an individual kernel launch
	//need different count values for each stream
	unsigned int * cnt;
	cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*cnt=0;

	unsigned int * dev_cnt; 
	dev_cnt=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_cnt, sizeof(unsigned int)*GPUSTREAMS);	
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc cnt -- error with code " << errCode << endl; 
	}

	///////////////////////////////////
	//END COUNT VALUES -- RESULT SET SIZE FOR EACH KERNEL INVOCATION
	///////////////////////////////////
	
	

	///////////////////////////////////
	//EPSILON
	///////////////////////////////////
	DTYPE* dev_epsilon;
	// dev_epsilon=(DTYPE*)malloc(sizeof( DTYPE));
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc epsilon -- error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon copy to device -- error with code " << errCode << endl; 
	}		

	///////////////////////////////////
	//END EPSILON
	///////////////////////////////////


	///////////////////////////////////
	//reference point offset (the actual size of the database)
	///////////////////////////////////
	unsigned int* dev_refPointOffset;
	
	

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_refPointOffset, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc ref point offset -- error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_refPointOffset, &refPointOffset, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: ref point offset copy to device -- error with code " << errCode << endl; 
	}		

	///////////////////////////////////
	//END reference point offset 
	///////////////////////////////////


	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS
	///////////////////////////////////
	unsigned int * dev_nNonEmptyCells;
	// dev_nNonEmptyCells=(unsigned int*)malloc(sizeof( unsigned int ));
	


	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_nNonEmptyCells, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc nNonEmptyCells -- error with code " << errCode << endl; 
	}

	//copy to device
	errCode=cudaMemcpy( dev_nNonEmptyCells, nNonEmptyCells, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: nNonEmptyCells copy to device -- error with code " << errCode << endl; 
	}		

	///////////////////////////////////
	//NUMBER OF NON-EMPTY CELLS
	///////////////////////////////////
	



	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////

	//THE NUMBER OF THREADS THAT ARE LAUNCHED IN A SINGLE KERNEL INVOCATION
	//CAN BE FEWER THAN THE NUMBER OF ELEMENTS IN THE DATABASE IF MORE THAN 1 BATCH
	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_N; 
	// dev_N=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc dev_N -- error with code " << errCode << endl; 
	}	

	////////////////////////////////////
	//NUMBER OF THREADS PER GPU STREAM
	////////////////////////////////////


	////////////////////////////////////
	//OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER 
	////////////////////////////////////
	unsigned int * batchOffset; 
	batchOffset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);
	
	unsigned int * dev_offset; 
	// dev_offset=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_offset, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc offset -- error with code " << errCode << endl; 
	}

	//Batch number to calculate the point to process (in conjunction with the offset)
	//offset into the database when batching the results
	unsigned int * batchNumber; 
	batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	unsigned int * dev_batchNumber; 
	// dev_batchNumber=(unsigned int*)malloc(sizeof(unsigned int)*GPUSTREAMS);

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int)*GPUSTREAMS);
	if(errCode != cudaSuccess) {
	cout << "\nError: Alloc batch number -- error with code " << errCode << endl; 
	}

	////////////////////////////////////
	//END OFFSET INTO THE DATABASE FOR BATCHING THE RESULTS
	//BATCH NUMBER
	////////////////////////////////////

				
	
	

	unsigned long long estimatedNeighbors=0;	
	unsigned int numBatches=0;
	unsigned int GPUBufferSize=0;

	double tstartbatchest=omp_get_wtime();
	estimatedNeighbors=callGPUBatchEst(DBSIZE, dev_queryPts, queryVect->size(), *nNonEmptyCells, refPointsPopulated, dev_database, dev_epsilon, dev_grid, dev_indexLookupArr,dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, &numBatches, &GPUBufferSize);	
	double tendbatchest=omp_get_wtime();
	printf("\nTime to estimate batches: %f",tendbatchest - tstartbatchest);
	printf("\nIn Calling fn: Estimated neighbors: %llu, num. batches: %d, GPU Buffer size: %d",estimatedNeighbors, numBatches,GPUBufferSize);
	
	






	//WE CALCULATE THE BUFFER SIZES AND NUMBER OF BATCHES

	// unsigned int GPUBufferSize=100000000;
	// double alpha=0.05; //overestimation factor

	// unsigned long long estimatedTotalSize=(unsigned long long)(estimatedNeighbors)*(unsigned long long)offsetRate;
	// unsigned long long estimatedTotalSizeWithAlpha=(unsigned long long)(estimatedNeighbors)*(unsigned long long)offsetRate*(1.0+(alpha));
	// printf("\nEstimated total result set size: %llu", estimatedTotalSize);
	// printf("\nEstimated total result set size (with Alpha %f): %llu", alpha,estimatedTotalSizeWithAlpha);
	

	//to accomodate small datasets, we need smaller buffers because the pinned memory malloc is expensive
	// if (estimatedNeighbors<(GPUBufferSize*GPUSTREAMS))
	// {
	// 	GPUBufferSize=estimatedNeighbors*(1.0+(alpha*2.0))/(GPUSTREAMS);		//we do 2*alpha for small datasets because the
	// 																	//sampling will be worse for small datasets
	// 																	//but we fix the 3 streams still (thats why divide by 3).			
	// }

	// unsigned int numBatches=ceil(((1.0+alpha)*estimatedNeighbors*1.0)/(GPUBufferSize*1.0));
	// printf("\nNumber of batches: %d, buffer size: %d", numBatches, GPUBufferSize);


		

	// printf("\nEnd Batch Estimator\n***********************************\n");


	


	/////////////////////////////////////////////////////////	
	//END BATCH ESTIMATOR	
	/////////////////////////////////////////////////////////


	

	////////////////////////////////////
	//TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	//debug values
	unsigned int * dev_debug1; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0; 

	

	unsigned int * dev_debug2; 
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;

	unsigned int * debug1; 
	debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug1=0;

	// //TEST
	// *debug1=*DBSIZE;

	unsigned int * debug2; 
	debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*debug2=0;



	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 alloc -- error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 alloc -- error with code " << errCode << endl; 
	}		

	//copy debug to device
	errCode=cudaMemcpy( dev_debug1, debug1, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug1 copy to device -- error with code " << errCode << endl; 
	}

	errCode=cudaMemcpy( dev_debug2, debug2, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_debug2 copy to device -- with code " << errCode << endl; 
	}


	////////////////////////////////////
	//END TWO DEBUG VALUES SENT TO THE GPU FOR GOOD MEASURE
	////////////////////////////////////			

	

	///////////////////
	//ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////

	//THE NUMBER OF POINTERS IS EQUAL TO THE NUMBER OF BATCHES
	for (int i=0; i<numBatches; i++){
		int *ptr=NULL;
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr=ptr;
		tmpStruct.sizeOfDataArr=0;
		
		pointersToNeighbors->push_back(tmpStruct);
	}

	///////////////////
	//END ALLOCATE POINTERS TO INTEGER ARRAYS FOR THE VALUES FOR THE NEIGHBORTABLES
	///////////////////



	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET USING THE BATCH ESTIMATOR
	///////////////////////////////////
	

	//NEED BUFFERS ON THE GPU AND THE HOST FOR THE NUMBER OF CONCURRENT STREAMS	
	//GPU BUFFER ON THE DEVICE
	//BUFFER ON THE HOST WITH PINNED MEMORY FOR FAST MEMCPY
	//BUFFER ON THE HOST TO DUMP THE RESULTS OF BATCHES SO THAT GPU THREADS CAN CONTINUE
	//EXECUTING STREAMS ON THE HOST



	//GPU MEMORY ALLOCATION: key/value pairs

	int * dev_pointIDKey[GPUSTREAMS]; //key
	int * dev_pointInDistValue[GPUSTREAMS]; //value

	

	for (int i=0; i<GPUSTREAMS; i++)
	{
		errCode=cudaMalloc((void **)&dev_pointIDKey[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}

		errCode=cudaMalloc((void **)&dev_pointInDistValue[i], sizeof(int)*GPUBufferSize);
		if(errCode != cudaSuccess) {
		cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
		}


	}


	
	/*
	//WE NOW ALLOC ONCE IN MAIN

	//HOST RESULT ALLOCATION FOR THE GPU TO COPY THE DATA INTO A PINNED MEMORY ALLOCATION
	//ON THE HOST
	//pinned result set memory for the host
	//the number of elements are recorded for that batch in resultElemCountPerBatch
	//NEED PINNED MEMORY ALSO BECAUSE YOU NEED IT TO USE STREAMS IN THRUST FOR THE MEMCOPY OF THE SORTED RESULTS	
	//can't do async copies without pinned memory

	//PINNED MEMORY TO COPY FROM THE GPU	
	int * pointIDKey[GPUSTREAMS]; //key
	int * pointInDistValue[GPUSTREAMS]; //value
	

	double tstartpinnedresults=omp_get_wtime();
	
	for (int i=0; i<GPUSTREAMS; i++)
	{
	cudaMallocHost((void **) &pointIDKey[i], sizeof(int)*GPUBufferSize);
	cudaMallocHost((void **) &pointInDistValue[i], sizeof(int)*GPUBufferSize);
	}

	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results: %f", tendpinnedresults - tstartpinnedresults);
	


	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*2*GPUBufferSize*GPUSTREAMS)/(1024*1024*1024));

	
	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	*/










	



	/////////////////////////////////
	//SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////

	omp_set_num_threads(GPUSTREAMS);
	

	/////////////////////////////////
	//END SET OPENMP ENVIRONMENT VARIABLES
	////////////////////////////////
	
	

	/////////////////////////////////
	//CREATE STREAMS
	////////////////////////////////

	//set device	
	// cudaSetDevice(gpuid);

	cudaStream_t stream[GPUSTREAMS];
	
	for (int i=0; i<GPUSTREAMS; i++){
	cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}	

	/////////////////////////////////
	//END CREATE STREAMS
	////////////////////////////////
	
	

	///////////////////////////////////
	//LAUNCH KERNEL IN BATCHES
	///////////////////////////////////
		
	//since we use the strided scheme, some of the batch sizes
	//are off by 1 of each other, a first group of batches will
	//have 1 extra data point to process, and we calculate which batch numbers will 
	//have that.  The batchSize is the lower value (+1 is added to the first ones)

	CTYPE* dev_workCounts;
	cudaMalloc((void **)&dev_workCounts, sizeof(CTYPE)*2);

	unsigned int batchSize=(*QUERYSIZE)/numBatches;
	unsigned int batchesThatHaveOneMore=(*QUERYSIZE)-(batchSize*numBatches); //batch number 0- < this value have one more
	printf("\nBatches that have one more GPU thread: %u batchSize(N): %u, \n",batchesThatHaveOneMore,batchSize);



	uint64_t totalResultsLoop=0;


		
		//FOR LOOP OVER THE NUMBER OF BATCHES STARTS HERE
		//i=0...numBatches
		#pragma omp parallel for schedule(static,1) reduction(+:totalResultsLoop) num_threads(GPUSTREAMS)
		for (int i=0; i<numBatches; i++)
		{	
			
			cudaSetDevice(gpuid);	
			int tid=omp_get_thread_num();
			

			
			printf("\ntid: %d, starting iteration: %d",tid,i);
			printf("\ntid: %d, starting iteration: %d, gpuid: %d",tid, i,gpuid);
			//N NOW BECOMES THE NUMBER OF POINTS TO PROCESS PER BATCH
			//AS ONE GPU THREAD PROCESSES A SINGLE POINT
			



			
			if (i<batchesThatHaveOneMore)
			{
				N[tid]=batchSize+1;	
				printf("\nN (GPU threads): %d, tid: %d",N[tid], tid);
			}
			else
			{
				N[tid]=batchSize;	
				printf("\nN (1 less): %d tid: %d",N[tid], tid);
			}

			//set relevant parameters for the batched execution that get reset
			
			//copy N to device 
			//N IS THE NUMBER OF THREADS
			errCode=cudaMemcpyAsync( &dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: N Got error with code " << errCode << endl; 
			}

			//the batched result set size (reset to 0):
			cnt[tid]=0;
			errCode=cudaMemcpyAsync( &dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
			}

			//the offset for batching, which keeps track of where to start processing at each batch
			batchOffset[tid]=numBatches; //for the strided
			errCode=cudaMemcpyAsync( &dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_offset memcpy Got error with code " << errCode << endl; 
			}

			//the batch number for batching with strided
			batchNumber[tid]=i;
			errCode=cudaMemcpyAsync( &dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: dev_batchNumber memcpy Got error with code " << errCode << endl; 
			}

			const int TOTALBLOCKS=ceil((1.0*(N[tid]))/(1.0*BLOCKSIZE));	
			printf("\ntotal blocks: %d",TOTALBLOCKS);

			//execute kernel	
			//0 is shared memory pool
			kernelNDGridIndexGlobal<<< TOTALBLOCKS, BLOCKSIZE, 0, stream[tid]>>>(dev_debug1, dev_debug2, &dev_N[tid], dev_queryPts,
		&dev_offset[tid], &dev_batchNumber[tid], dev_database, dev_epsilon, dev_grid, dev_indexLookupArr, 
		dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells, dev_pointIDKey[tid], dev_pointInDistValue[tid], dev_workCounts, dev_refPointOffset);

			// errCode=cudaDeviceSynchronize();
			// cout <<"\n\nError from device synchronize: "<<errCode;

			cout <<"\n\nKERNEL LAUNCH RETURN: "<<cudaGetLastError()<<endl<<endl;
			if ( cudaSuccess != cudaGetLastError() ){
		    	cout <<"\n\nERROR IN KERNEL LAUNCH. ERROR: "<<cudaSuccess<<endl<<endl;
		    }

		    

		   
			// find the size of the number of results
			errCode=cudaMemcpyAsync( &cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid] );
			if(errCode != cudaSuccess) {
			cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
			}
			else{
				printf("\nGPU: result set size within epsilon (GPU grid): %d",cnt[tid]);cout.flush();
			}


			
			


			////////////////////////////////////
			//SORT THE TABLE DATA ON THE GPU
			//THERE IS NO ORDERING BETWEEN EACH POINT AND THE ONES THAT IT'S WITHIN THE DISTANCE OF
			////////////////////////////////////

			/////////////////////////////
			//ONE PROBLEM WITH NOT TRANSFERING THE RESULT OFF OF THE DEVICE IS THAT
			//YOU CAN'T RESIZE THE RESULTS TO BE THE SIZE OF *CNT
			//SO THEN YOU HAVE POTENTIALLY LOTS OF WASTED SPACE
			/////////////////////////////

			//sort by key with the data already on the device:
			//wrap raw pointer with a device_ptr to use with Thrust functions
			thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
			thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

			//XXXXXXXXXXXXXXXX
			//THRUST USING STREAMS REQUIRES THRUST V1.8 
			//XXXXXXXXXXXXXXXX
			
			
			try{
			thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);


			}
			catch(std::bad_alloc &e)
			  {
			    std::cerr << "Ran out of memory while sorting, " << GPUBufferSize << std::endl;
			    exit(-1);
			  }
			


	  		//thrust with streams into individual buffers for each batch
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[gpuid][tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
			cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[gpuid][tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid]*sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);	

			//need to make sure the data is copied before constructing portion of the neighbor table
			cudaStreamSynchronize(stream[tid]);
			
			

			
			double tableconstuctstart=omp_get_wtime();
			//set the number of neighbors in the pointer struct:
			(*pointersToNeighbors)[i].sizeOfDataArr=cnt[tid];    
			(*pointersToNeighbors)[i].dataPtr=new int[cnt[tid]]; 

			
			constructNeighborTableKeyValueWithPtrs(pointIDKey[gpuid][tid], pointInDistValue[gpuid][tid], neighborTable, (*pointersToNeighbors)[i].dataPtr, &cnt[tid]);
			
			
			
			// cout <<"\nIn make neighbortable. Data array ptr: "<<(*pointersToNeighbors)[i].dataPtr<<" , size of data array: "<<(*pointersToNeighbors)[i].sizeOfDataArr;cout.flush();

			double tableconstuctend=omp_get_wtime();	
			
			printf("\nTable construct time: %f", tableconstuctend - tableconstuctstart);

			
			//add the batched result set size to the total count
			totalResultsLoop+=cnt[tid];

			printf("\nRunning total of total size of result array, tid: %d: %lu", tid, totalResultsLoop);
			
			
		

			

		} //END LOOP OVER THE GPU BATCHES
	
	printf("\nTOTAL RESULT SET SIZE ON HOST:  %lu", totalResultsLoop);
	*totalNeighbors=totalResultsLoop;


	double tKernelResultsEnd=omp_get_wtime();
	
	printf("\nTime to launch kernel and execute everything (get results etc.) except freeing memory: %f",tKernelResultsEnd-tKernelResultsStart);


	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////



	///////////////////////////////////	
	//OPTIONAL DEBUG VALUES
	///////////////////////////////////
	
	// double tStartdebug=omp_get_wtime();

	// errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	// }
	// else
	// {
	// 	printf("\nDebug1 value: %u",*debug1);
	// }

	// errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	// if(errCode != cudaSuccess) {
	// cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	// }
	// else
	// {
	// 	printf("\nDebug2 value: %u",*debug2);
	// }	

	// double tEnddebug=omp_get_wtime();
	// printf("\nTime to retrieve debug values: %f", tEnddebug - tStartdebug);
	

	///////////////////////////////////	
	//END OPTIONAL DEBUG VALUES
	///////////////////////////////////
	

	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
	

	double tFreeStart=omp_get_wtime();

	for (int i=0; i<GPUSTREAMS; i++){
		errCode=cudaStreamDestroy(stream[i]);
		if(errCode != cudaSuccess) {
		cout << "\nError: destroying stream" << errCode << endl; 
		}
	}
	
	

	//free host-side memory
	free(DBSIZE);
	free(database);
	free(QUERYSIZE);
	free(queryPts);
	free(N);
	free(batchOffset);
	free(batchNumber);
	free(debug1);
	free(debug2);



	//free the data on the device
	errCode=cudaFree(dev_database);
	errCode=cudaFree(dev_debug1);
	errCode=cudaFree(dev_debug2);
	errCode=cudaFree(dev_epsilon);
	errCode=cudaFree(dev_grid);
	errCode=cudaFree(dev_gridCellLookupArr);
	errCode=cudaFree(dev_indexLookupArr);
	errCode=cudaFree(dev_minArr);
	errCode=cudaFree(dev_nCells);
	errCode=cudaFree(dev_queryPts);
	errCode=cudaFree(dev_nNonEmptyCells);
	errCode=cudaFree(dev_N);
	errCode=cudaFree(dev_cnt);
	errCode=cudaFree(dev_offset);
	errCode=cudaFree(dev_batchNumber);

	
	//free data related to the individual streams for each batch
	for (int i=0; i<GPUSTREAMS; i++){
		//free the data on the device
		errCode=cudaFree(dev_pointIDKey[i]);
		errCode=cudaFree(dev_pointInDistValue[i]);
		//free on the host
		//moved to another section, as we allocate in main
		// cudaFreeHost(pointIDKey[i]);
		// cudaFreeHost(pointInDistValue[i]);
	}

	double tFreeEnd=omp_get_wtime();

	printf("\nTime freeing memory: %f", tFreeEnd - tFreeStart);
	
	cout<<"\n** last error at end of fn batches (could be from freeing memory): "<<cudaGetLastError();

}









void warmUpGPU(){
// initialize all ten integers of a device_vector to 1 
thrust::device_vector<int> D(10, 1); 
// set the first seven elements of a vector to 9 
thrust::fill(D.begin(), D.begin() + 7, 9); 
// initialize a host_vector with the first five elements of D 
thrust::host_vector<int> H(D.begin(), D.begin() + 5); 
// set the elements of H to 0, 1, 2, 3, ... 
thrust::sequence(H.begin(), H.end()); // copy all of H back to the beginning of D 
thrust::copy(H.begin(), H.end(), D.begin()); 
// print D 
for(int i = 0; i < D.size(); i++) 
std::cout << " D[" << i << "] = " << D[i]; 


return;
}




void allocPinnedmemory(int * pointIDKey[NUMGPU][GPUSTREAMS], int * pointInDistValue[NUMGPU][GPUSTREAMS])
{




	double tstartpinnedresults=omp_get_wtime();

	for (int i=0; i<NUMGPU; i++)
	{	
		for (int j=0; j<GPUSTREAMS; j++)
		{
		cudaMallocHost((void **) &pointIDKey[i][j], sizeof(int)*GPUBUFFERSIZE);
		cudaMallocHost((void **) &pointInDistValue[i][j], sizeof(int)*GPUBUFFERSIZE);
		}
	}	
	double tendpinnedresults=omp_get_wtime();
	printf("\nTime to allocate pinned memory for results: %f", tendpinnedresults - tstartpinnedresults);
	


	printf("\nmemory requested for results ON GPU (GiB): %f",(double)(sizeof(int)*2.0*GPUBUFFERSIZE*GPUSTREAMS*NUMGPU)/(1024*1024*1024));
	printf("\nmemory requested for results in MAIN MEMORY (GiB): %f",(double)(sizeof(int)*2.0*GPUBUFFERSIZE*GPUSTREAMS*NUMGPU)/(1024*1024*1024));

}


void deAllocPinnedMemory(int * pointIDKey[NUMGPU][GPUSTREAMS], int * pointInDistValue[NUMGPU][GPUSTREAMS])
{
	
	for (int i=0; i<NUMGPU; i++)
	{	
		for (int j=0; j<GPUSTREAMS; j++)
		{
			cudaFreeHost(pointIDKey[i][j]);
			cudaFreeHost(pointInDistValue[i][j]);
		}
	}	

	// cudaFreeHost(pointIDKey);
	// cudaFreeHost(pointInDistValue);

}


void constructNeighborTableKeyValueWithPtrs(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt)
{

	
	
	//copy the value data:
	
	std::copy(pointInDistValue, pointInDistValue+(*cnt), pointersToNeighbors);
	


	//Step 1: find all of the unique keys and their positions in the key array
	unsigned int numUniqueKeys=0;

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++){
		if (pointIDKey[i-1]!=pointIDKey[i]){
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}

	
	//insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. 
	for (int i=0; i<uniqueKeyData.size()-1; i++) {
		int keyElem=uniqueKeyData[i].key;
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].indexmin=uniqueKeyData[i].position;
		neighborTable[keyElem].indexmax=uniqueKeyData[i+1].position-1;
	
		//update the pointer to the data array for the values
		neighborTable[keyElem].dataPtr=pointersToNeighbors;	
	}



}






void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt)
{
	
	//newer multithreaded way:
	//Step 1: find all of the unique keys and their positions in the key array
	
	//double tstart=omp_get_wtime();

	unsigned int numUniqueKeys=0;
	unsigned int count=0;

	

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key=pointIDKey[0];
	tmp.position=0;
	uniqueKeyData.push_back(tmp);



	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (int i=1; i<(*cnt)+1; i++)
	{
		if (pointIDKey[i-1]!=pointIDKey[i])
		{
			numUniqueKeys++;
			tmp.key=pointIDKey[i];
			tmp.position=i;
			uniqueKeyData.push_back(tmp);
		}
	}



	//Step 2: In parallel, insert into the neighbor table the values based on the positions of 
	//the unique keys obtained above. Since multiple threads access this function, we don't want to 
	//do too many memory operations while GPU memory transfers are occurring, or else we decrease the speed that we 
	//get data off of the GPU
	omp_set_nested(1);
	#pragma omp parallel for reduction(+:count) num_threads(2) schedule(static,1)
	for (int i=0; i<uniqueKeyData.size()-1; i++) 
	{
		int keyElem=uniqueKeyData[i].key;
		int valStart=uniqueKeyData[i].position;
		int valEnd=uniqueKeyData[i+1].position-1;
		int size=valEnd-valStart+1;
		
		//seg fault from here: is it neighbortable mem alloc?
		neighborTable[keyElem].pointID=keyElem;
		neighborTable[keyElem].neighbors.insert(neighborTable[keyElem].neighbors.begin(),&pointInDistValue[valStart],&pointInDistValue[valStart+size]);
		
		//printf("\nval: start:%d, end: %d", valStart,valEnd);
		//printf("\ni: %d, keyElem: %d, position start: %d, position end: %d, size: %d", i,keyElem,valStart, valEnd,size);	


		count+=size;

	}
	

}





//Uses a brute force kernel to calculate the direct neighbors of the points in the database
//void makeDistanceTableGPUBruteForce(std::vector<struct dataElem> * dataPoints, double * epsilon, struct table * neighborTable, int * totalNeighbors)
void makeDistanceTableGPUBruteForce(std::vector<std::vector <DTYPE> > * NDdataPoints, DTYPE* epsilon, struct table * neighborTable, unsigned long long int * totalNeighbors)
{
	//CUDA error code:
	cudaError_t errCode;


	///////////////////////////////////
	//COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	unsigned int * N;
	N=(unsigned int*)malloc(sizeof(unsigned int));
	*N=NDdataPoints->size();
	
	printf("\nIn main GPU method: Number of data points, (N), is: %u ",*N);cout.flush();



	
	//the database will just be a 1-D array, we access elemenets based on NDIM
	DTYPE* database= (DTYPE*)malloc(sizeof(DTYPE)*(*N)*GPUNUMDIM);  
	DTYPE* dev_database= (DTYPE*)malloc(sizeof(DTYPE)*(*N)*GPUNUMDIM);  
	

	//allocate memory on device:
	errCode=cudaMalloc( (void**)&dev_database, sizeof(DTYPE)*GPUNUMDIM*(*N));		
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; cout.flush(); 
	}


	



	//copy the database from the ND vector to the array:
	for (int i=0; i<*N; i++){
		std::copy((*NDdataPoints)[i].begin(), (*NDdataPoints)[i].end(), database+(i*GPUNUMDIM));
	}
	//test 
	// printf("\n\n");
	// int tmpcnt=0;
	// for (int i=0; i<NDdataPoints->size(); i++)
	// {
	// 	for (int j=0; j<(*NDdataPoints)[i].size(); j++)
	// 	{
	// 		database[tmpcnt]=(*NDdataPoints)[i][j];
	// 		tmpcnt++;
	// 	}
	// }
	// for (int i=0; i<(*N)*GPUNUMDIM; i++){
	// 	printf("%f,",database[i]);
	// }	



	
	//copy database to the device:
	errCode=cudaMemcpy(dev_database, database, sizeof(DTYPE)*(*N)*GPUNUMDIM, cudaMemcpyHostToDevice);	
	if(errCode != cudaSuccess) {
	cout << "\nError: database Got error with code " << errCode << endl; 
	}	




	///////////////////////////////////
	//END COPY THE DATABASE TO THE GPU
	///////////////////////////////////


	


	///////////////////////////////////
	//ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////
	//NON-PINNED MEMORY FOR SINGLE KERNEL INVOCATION (NO BATCHING)


	//CHANGING THE RESULTS TO KEY VALUE PAIR SORT, WHICH IS TWO ARRAYS
	//KEY IS THE POINT ID
	//THE VALUE IS THE POINT ID WITHIN THE DISTANCE OF KEY

	int * dev_pointIDKey; //key
	int * dev_pointInDistValue; //value

	
	


	errCode=cudaMalloc((void **)&dev_pointIDKey, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}

	errCode=cudaMalloc((void **)&dev_pointInDistValue, sizeof(int)*BUFFERELEM);
	if(errCode != cudaSuccess) {
	cout << "CUDA: Got error with code " << errCode << endl; //2 means not enough memory
	}


	printf("\nmemory requested for results (GiB): %f",(double)(sizeof(int)*2*BUFFERELEM)/(1024*1024*1024));

	
	double tstartalloc=omp_get_wtime();

	//host result allocation:
	//pinned result set memory for the host
	// cudaMallocHost((void **) &pointIDKey, sizeof(int)*BUFFERELEM);
	// cudaMallocHost((void **) &pointInDistValue, sizeof(int)*BUFFERELEM);



	//PAGED MEMORY ALLOCATION FOR SMALL RESULT SET WITH SINGLE KERNEL EXECUTION?
	int * pointIDKey=(int*)malloc(sizeof(int)*BUFFERELEM);
	int * pointInDistValue=(int*)malloc(sizeof(int)*BUFFERELEM);

	double tendalloc=omp_get_wtime();


	//printf("\nTime to allocate pinned memory on the host: %f", tendalloc - tstartalloc);
	printf("\nTime to allocate (non-pinned) memory on the host: %f", tendalloc - tstartalloc);

	

	///////////////////////////////////
	//END ALLOCATE MEMORY FOR THE RESULT SET
	///////////////////////////////////















	///////////////////////////////////
	//SET OTHER KERNEL PARAMETERS
	///////////////////////////////////

	
	
	//count values
	unsigned long long int * cnt;
	cnt=(unsigned long long int*)malloc(sizeof(unsigned long long int));
	*cnt=0;

	unsigned long long int * dev_cnt; 
	dev_cnt=(unsigned long long int*)malloc(sizeof(unsigned long long int));
	*dev_cnt=0;

	//allocate on the device
	errCode=cudaMalloc((unsigned long long int**)&dev_cnt, sizeof(unsigned long long int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: cnt Got error with code " << errCode << endl; 
	}


	errCode=cudaMemcpy( dev_cnt, cnt, sizeof(unsigned long long int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_cnt memcpy Got error with code " << errCode << endl; 
	}




	
	
	//Epsilon
	DTYPE* dev_epsilon;
	dev_epsilon=(DTYPE*)malloc(sizeof( DTYPE));

	//Allocate on the device
	errCode=cudaMalloc((void**)&dev_epsilon, sizeof(DTYPE));
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}


		
	//size of the database:
	unsigned int * dev_N; 
	dev_N=(unsigned int*)malloc(sizeof( unsigned int ));

	//allocate on the device
	errCode=cudaMalloc((void**)&dev_N, sizeof(unsigned int));
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}	



	//debug values
	unsigned int * dev_debug1; 
	unsigned int * dev_debug2; 
	dev_debug1=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug1=0;
	dev_debug2=(unsigned int *)malloc(sizeof(unsigned int ));
	*dev_debug2=0;




	//allocate on the device
	errCode=cudaMalloc( (unsigned int **)&dev_debug1, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug1 Got error with code " << errCode << endl; 
	}		
	errCode=cudaMalloc( (unsigned int **)&dev_debug2, sizeof(unsigned int ) );
	if(errCode != cudaSuccess) {
	cout << "\nError: debug2 Got error with code " << errCode << endl; 
	}		


	//copy N, epsilon to the device
	//epsilon
	errCode=cudaMemcpy( dev_epsilon, epsilon, sizeof(DTYPE), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: epsilon Got error with code " << errCode << endl; 
	}		

	//N (DATASET SIZE)
	errCode=cudaMemcpy( dev_N, N, sizeof(unsigned int), cudaMemcpyHostToDevice );
	if(errCode != cudaSuccess) {
	cout << "\nError: N Got error with code " << errCode << endl; 
	}		

	



	///////////////////////////////////
	//END SET OTHER KERNEL PARAMETERS
	///////////////////////////////////


	


	///////////////////////////////////
	//LAUNCH KERNEL
	///////////////////////////////////

	const int TOTALBLOCKS=ceil((1.0*(*N))/(1.0*BLOCKSIZE));	
	printf("\ntotal blocks: %d",TOTALBLOCKS);


	//execute kernel	

	
	double tkernel_start=omp_get_wtime();
	kernelBruteForce<<< TOTALBLOCKS, BLOCKSIZE >>>(dev_N, dev_debug1, dev_debug2, dev_epsilon, dev_cnt, dev_database, dev_pointIDKey, dev_pointInDistValue);
	if ( cudaSuccess != cudaGetLastError() ){
    	printf( "Error in kernel launch!\n" );
    }


    cudaDeviceSynchronize();
    double tkernel_end=omp_get_wtime();
    printf("\nTime for kernel only: %f", tkernel_end - tkernel_start);
    ///////////////////////////////////
	//END LAUNCH KERNEL
	///////////////////////////////////
    


    ///////////////////////////////////
	//GET RESULT SET
	///////////////////////////////////

	//first find the size of the number of results
	errCode=cudaMemcpy( cnt, dev_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	if(errCode != cudaSuccess) {
	cout << "\nError: getting cnt from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nGPU: result set size on within epsilon: %llu",*cnt);
	}



	*totalNeighbors=(*cnt);


	//get debug information (optional)
	unsigned int * debug1;
	debug1=(unsigned int*)malloc(sizeof(unsigned int));
	*debug1=0;
	unsigned int * debug2;
	debug2=(unsigned int*)malloc(sizeof(unsigned int));
	*debug2=0;

	errCode=cudaMemcpy(debug1, dev_debug1, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug1 value: %u",*debug1);
	}

	errCode=cudaMemcpy(debug2, dev_debug2, sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
	if(errCode != cudaSuccess) {
	cout << "\nError: getting debug1 from GPU Got error with code " << errCode << endl; 
	}
	else
	{
		printf("\nDebug2 value: %u",*debug2);
	}	

	
	///////////////////////////////////
	//END GET RESULT SET
	///////////////////////////////////


	///////////////////////////////////
	//FREE MEMORY FROM THE GPU
	///////////////////////////////////
    //free:
	cudaFree(dev_database);
	cudaFree(dev_debug1);
	cudaFree(dev_debug2);
	cudaFree(dev_cnt);
	cudaFree(dev_epsilon);
	////////////////////////////////////


}

