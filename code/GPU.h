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

#include "structs.h"
#include "params.h"
#include <omp.h>

void allocPinnedmemory(int * pointIDKey[NUMGPU][GPUSTREAMS], int * pointInDistValue[NUMGPU][GPUSTREAMS]);
void deAllocPinnedMemory(int * pointIDKey[NUMGPU][GPUSTREAMS], int * pointInDistValue[NUMGPU][GPUSTREAMS]);


void makeDistanceTableGPUBruteForce(std::vector<std::vector <DTYPE> > * NDdataPoints, DTYPE* epsilon, struct table * neighborTable, unsigned long long int * totalNeighbors);

void distanceTableNDGridBatches(std::vector<std::vector<DTYPE> > * NDdataPoints, std::vector<unsigned int> *queryVect, DTYPE* epsilon, struct grid * index, 
	struct gridCellLookup * gridCellLookupArr, unsigned int * nNonEmptyCells, DTYPE* minArr, unsigned int * nCells, 
	unsigned int * indexLookupArr, struct neighborTableLookup * neighborTable, std::vector<struct neighborDataPtrs> * pointersToNeighbors, 
	uint64_t * totalNeighbors, CTYPE* workCounts, unsigned int refPointOffset, int * pointIDKey[NUMGPU][GPUSTREAMS], int * pointInDistValue[NUMGPU][GPUSTREAMS], bool DenseBoxEnabled,int gpuid);



unsigned long long callGPUBatchEst(unsigned int * DBSIZE, unsigned int * dev_queryPts, unsigned int NqueryPts, 
	unsigned int NrefPts, bool refPointsPopulated, DTYPE* dev_database, DTYPE* dev_epsilon, struct grid * dev_grid, 
	unsigned int * dev_indexLookupArr, struct gridCellLookup * dev_gridCellLookupArr, DTYPE* dev_minArr, 
	unsigned int * dev_nCells, unsigned int * dev_nNonEmptyCells, unsigned int * dev_gridCellNDMask, 
	unsigned int * dev_gridCellNDMaskOffsets, unsigned int * dev_nNDMaskElems, unsigned int * retNumBatches, unsigned int * retGPUBufferSize);

void constructNeighborTableKeyValueWithPtrs(int * pointIDKey, int * pointInDistValue, struct neighborTableLookup * neighborTable, int * pointersToNeighbors, unsigned int * cnt);

void warmUpGPU();

//for the brute force version without batches
void constructNeighborTableKeyValue(int * pointIDKey, int * pointInDistValue, struct table * neighborTable, unsigned int * cnt);



