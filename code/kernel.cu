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

#include "kernel.h"
#include "structs.h"
#include <math.h>	
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

#include "params.h"



__device__ void evaluateCellEstimate(unsigned int* nCells, unsigned int* indexes, struct gridCellLookup * gridCellLookupArr, unsigned int* nNonEmptyCells, DTYPE* database, DTYPE *epsilon, struct grid * index, unsigned int * indexLookupArr, DTYPE* point, unsigned int* cnt, bool differentCell);

__device__ void evaluateCell(unsigned int* nCells, unsigned int* indexes, struct gridCellLookup * gridCellLookupArr, unsigned int* nNonEmptyCells, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, DTYPE* point, unsigned int* cnt,int* pointIDKey, int* pointInDistVal, int pointIdx, bool differentCell, unsigned int* nDCellIDs, CTYPE* workCounts);

__device__ void swap(unsigned int* a, unsigned int* b) {
	unsigned int temp = *a;
	*a = *b;
	*b= temp;
}

__device__ void sortCell(unsigned int* list, DTYPE* database, int length, int tid){
	bool odd=false;
	for(int i=0; i<length; i++) {
		for(int j=(tid*2)+(int)odd; j<length-1; j+=32) {
			if(database[list[j]*GPUNUMDIM] > database[list[j+1]*GPUNUMDIM]) {
				swap(&list[j], &list[j+1]);
			}
		}
		odd = !odd;
	}
}

__device__ void seqSortCell(unsigned int* list, DTYPE* database, int length){
	int min;
	int minIdx;

	for(int i=0; i<length-1; i++ ) {
		min = database[list[i]*GPUNUMDIM];
		minIdx=i;
		for(int j=i; j<length; i++) {
			if(database[list[j]*GPUNUMDIM] < min) {
				min = database[list[j]*GPUNUMDIM];
				minIdx = j;
			}
		}
		swap(&list[i], &list[minIdx]);
	}
}


__global__ void kernelSortPointsInCells(DTYPE* database, struct grid * index, unsigned int* indexLookupArr, unsigned int nNonEmptyCells) {
        int tid = threadIdx.x + (blockIdx.x*BLOCKSIZE);
        int warpId = tid/32;
        int totalWarps = (gridDim.x*BLOCKSIZE)/32;

	int sortDim=0;
	if(GPUNUMDIM > NUMINDEXEDDIM)
		sortDim = NUMINDEXEDDIM;
	

        for(int i=warpId; i<nNonEmptyCells; i+=totalWarps) {
		if(index[i].indexmin < index[i].indexmax) {
  	              sortCell(indexLookupArr+index[i].indexmin, database+sortDim, (index[i].indexmax-index[i].indexmin)+1, threadIdx.x%32);
		}
        }

}


/////////////////////////////////////////
//THE RESULTS GET GENERATED AS KEY/VALUE PAIRS IN TWO ARRAYS
//KEY- THE POINT ID BEING SEARCHED
//VALUE- A POINT ID WITHIN EPSILON OF THE KEY POINT THAT WAS SEARCHED
//THE RESULTS ARE SORTED IN SITU ON THE DEVICE BY THRUST AFTER THE KERNEL FINISHES
/////////////////////////////////////////



__device__ uint64_t getLinearID_nDimensionsGPU(unsigned int * indexes, unsigned int * dimLen, unsigned int nDimensions) {

    uint64_t offset = 0;
	uint64_t multiplier = 1;
	for (int i = 0; i<nDimensions; i++){
  	offset += (uint64_t)indexes[i] * multiplier;
  	multiplier *= dimLen[i];
	}

	return offset;
}






//This version is the same as the batch estimator
//One query point per GPU thread


// unsigned int *debug1, unsigned int *debug2 – ignore, debug values
// unsigned int *N – total GPU threads for the kernel  	
// unsigned int * queryPts -- the Query Points to be searched on the GPU  
// unsigned int * offset -  This is to offset into every nth data point, e.g., every 100th point calculates its neighbors 
// unsigned int *batchNum - The batch number being executed, used to calculate the point being processed
// DTYPE* database – The points in the database as 1 array
// DTYPE* epsilon – distance threshold
// struct grid * index – each non-empty grid cell is one of these, stores the indices into indexLookupArray that coincide with the data points in the database that are in the cell
// unsigned int * indexLookupArr – array of the size of database, has the indices of the datapoints in the database stored contiguously for each grid cell. each grid index cell references this 	
// struct gridCellLookup * gridCellLookupArr, - lookup array to the grid cells, needed to find if a grid cell exists (this is binary searched). Maps the location of the non-empty grid cells in grid * index to their linearized (1-D) array
// DTYPE* minArr – The minimum “edge” of the grid in each dimension
// unsigned int * nCells –The total number of cells in each dimension (if all were indexed), can compute the spatial extent, with minArr[0]+nCells[0]*epsilon, in the 1st dimension
// unsigned int * cnt – the result set size 	
// unsigned int * nNonEmptyCells – the number of non-empty cells in total, this is the size of the gridCellLookupArr
// int * pointIDKey, int * pointInDistVal - result set to be sorted as key/value pairs
//unsigned int * refPointBeginId -- the id that begins the reference points all ids < the value are normal data points (from the queryPts Array)
__global__ void kernelNDGridIndexGlobal(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  unsigned int * queryPts,  
	unsigned int * offset, unsigned int *batchNum, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells, int * pointIDKey, int * pointInDistVal, CTYPE* workCounts, unsigned int * refPointBeginId)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}




//original
// unsigned int pointIdx=tid*(*offset)+(*batchNum);
//considering the query point array
unsigned int pointIdx=queryPts[tid*(*offset)+(*batchNum)];
//The offset into the database, taking into consideration the length of each dimension


//original
// unsigned int pointOffset=tid*(GPUNUMDIM)*(*offset)+(*batchNum)*(GPUNUMDIM);
unsigned int pointOffset=pointIdx*(GPUNUMDIM);


//1.5 epsilon for the reference points
DTYPE eps=*epsilon;
if (pointIdx>=(*refPointBeginId))
{

	eps=*epsilon*1.5;
	// printf("gpu pnt: %u, %f\n",pointIdx,eps);
	// unsigned int idx=atomicAdd(debug1,int(1));
}


//make a local copy of the point
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[pointOffset+i];	
}

	//calculate the coords of the Cell for the point
	//and the min/max ranges in each dimension
	unsigned int nDCellIDs[NUMINDEXEDDIM];
	unsigned int nDMinCellIDs[NUMINDEXEDDIM];
	unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
	for (int i=0; i<NUMINDEXEDDIM; i++){
		nDCellIDs[i]=(point[i]-minArr[i])/(*epsilon);
		nDMinCellIDs[i]=max(0,nDCellIDs[i]-1); //boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)

	}




	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];


	// for (loopRng[0]=rangeFilteredCellIdsMin[0]; loopRng[0]<=rangeFilteredCellIdsMax[0]; loopRng[0]++)
	// for (loopRng[1]=rangeFilteredCellIdsMin[1]; loopRng[1]<=rangeFilteredCellIdsMax[1]; loopRng[1]++)
	for (loopRng[0]=nDMinCellIDs[0]; loopRng[0]<=nDMaxCellIDs[0]; loopRng[0]++)
	for (loopRng[1]=nDMinCellIDs[1]; loopRng[1]<=nDMaxCellIDs[1]; loopRng[1]++)
	{ //beginning of loop body
	
	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	}		
		//original
		// evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs, workCounts);
		//with 1.5 eps for detecting merges if the point is a reference point
		evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, &eps, index, indexLookupArr, point, cnt, pointIDKey, pointInDistVal, pointIdx, false, nDCellIDs, workCounts);
	} //end loop body


}

__forceinline__ __device__ void evalPoint(unsigned int* indexLookupArr, int k, DTYPE* database, DTYPE* epsilon, DTYPE* point, unsigned int* cnt, int* pointIDKey, int* pointInDistVal, int pointIdx) 
{

	DTYPE runningTotalDist=0;
	unsigned int dataIdx=indexLookupArr[k];
        for (int l=0; l<GPUNUMDIM; l++){
          runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
        }

        if (sqrt(runningTotalDist)<=(*epsilon)){
          unsigned int idx=atomicAdd(cnt,int(1));
          pointIDKey[idx]=pointIdx;
          pointInDistVal[idx]=dataIdx;
	}

}

__device__ void evaluateCell(unsigned int* nCells, unsigned int* indexes, struct gridCellLookup * gridCellLookupArr, unsigned int* nNonEmptyCells, DTYPE* database, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, DTYPE* point, unsigned int* cnt, int* pointIDKey, int* pointInDistVal, int pointIdx, bool differentCell, unsigned int* nDCellIDs, CTYPE* workCounts) {


        uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
        //compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
        //a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

        struct gridCellLookup tmp;
        tmp.gridLinearID=calcLinearID;
        //find if the cell is non-empty
        if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr+ (*nNonEmptyCells), gridCellLookup(tmp))){


                //compute the neighbors for the adjacent non-empty cell
                struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
                unsigned int GridIndex=resultBinSearch->idx;
	




	for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
		evalPoint(indexLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
        }

	}//end if binary search

}



//Kernel brute forces to generate the neighbor table for each point in the database
__global__ void kernelBruteForce(unsigned int *N, unsigned int *debug1, unsigned int *debug2, DTYPE* epsilon, unsigned long long int * cnt, DTYPE* database, int * pointIDKey, int * pointInDistVal) {

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}


int dataOffset=tid*GPUNUMDIM;
DTYPE runningDist=0;
//compare my point to every other point
for (int i=0; i<(*N); i++)
{
	runningDist=0;
	for (int j=0; j<GPUNUMDIM; j++){
		runningDist+=(database[(i*GPUNUMDIM)+j]-database[dataOffset+j])*(database[(i*GPUNUMDIM)+j]-database[dataOffset+j]);
	}

	//if within epsilon:
	if ((sqrt(runningDist))<=(*epsilon)){
		atomicAdd(cnt, (unsigned long long int)1);
	}
}


return;
}





//Need to use the query points to get a good estimate of the total result set size

__global__ void kernelNDGridIndexBatchEstimator(unsigned int *debug1, unsigned int *debug2, unsigned int *N,  
	unsigned int * sampleOffset, DTYPE* database, unsigned int *queryPts, DTYPE* epsilon, struct grid * index, unsigned int * indexLookupArr, 
	struct gridCellLookup * gridCellLookupArr, DTYPE* minArr, unsigned int * nCells, unsigned int * cnt, 
	unsigned int * nNonEmptyCells)
{

unsigned int tid=threadIdx.x+ (blockIdx.x*BLOCKSIZE); 

if (tid>=*N){
	return;
}

//Added this because of the offset, we may go beyond the end of the array
// if (tid*(*sampleOffset)>=*N)
// {
// 	return;
// }


//original
// unsigned int pointID=tid*(*sampleOffset)*(GPUNUMDIM);
//considering the query point array
unsigned int pointID=queryPts[tid*(*sampleOffset)];



//The offset into the database, taking into consideration the length of each dimension
unsigned int pointOffset=pointID*(GPUNUMDIM);





//make a local copy of the point
DTYPE point[GPUNUMDIM];
for (int i=0; i<GPUNUMDIM; i++){
	point[i]=database[pointOffset+i];	
}

//calculate the coords of the Cell for the point
//and the min/max ranges in each dimension
unsigned int nDCellIDs[NUMINDEXEDDIM];
unsigned int nDMinCellIDs[NUMINDEXEDDIM];
unsigned int nDMaxCellIDs[NUMINDEXEDDIM];
for (int i=0; i<NUMINDEXEDDIM; i++){
	nDCellIDs[i]=(point[i]-minArr[i])/(*epsilon);
	nDMinCellIDs[i]=max(0,nDCellIDs[i]-1); //boundary conditions (don't go beyond cell 0)
	nDMaxCellIDs[i]=min(nCells[i]-1,nDCellIDs[i]+1); //boundary conditions (don't go beyond the maximum number of cells)

}




	unsigned int indexes[NUMINDEXEDDIM];
	unsigned int loopRng[NUMINDEXEDDIM];

	for (loopRng[0]=nDMinCellIDs[0]; loopRng[0]<=nDMaxCellIDs[0]; loopRng[0]++)
	for (loopRng[1]=nDMinCellIDs[1]; loopRng[1]<=nDMaxCellIDs[1]; loopRng[1]++)
	{ //beginning of loop body

	for (int x=0; x<NUMINDEXEDDIM; x++){
	indexes[x]=loopRng[x];	
	}
	

	
	uint64_t calcLinearID=getLinearID_nDimensionsGPU(indexes, nCells, NUMINDEXEDDIM);
	//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says 
	//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

	struct gridCellLookup tmp;
	tmp.gridLinearID=calcLinearID;
	if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr+ (*nNonEmptyCells), gridCellLookup(tmp))){

				
		struct gridCellLookup * resultBinSearch=thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr+(*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex=resultBinSearch->idx;

		for (int k=index[GridIndex].indexmin; k<=index[GridIndex].indexmax; k++){
				DTYPE runningTotalDist=0;
				unsigned int dataIdx=indexLookupArr[k];

				for (int l=0; l<GPUNUMDIM; l++){
				runningTotalDist+=(database[dataIdx*GPUNUMDIM+l]-point[l])*(database[dataIdx*GPUNUMDIM+l]-point[l]);
				}

				if (sqrt(runningTotalDist)<=(*epsilon)){
					unsigned int idx=atomicAdd(cnt,int(1));

				}
			}
	}

	
	//printf("\nLinear id: %d",calcLinearID);
	} //end loop body

}
