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

#ifndef STRUCTS_H
#define STRUCTS_H
#include <vector>
#include <stdio.h>
#include <iostream>


//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "params.h"

struct key_val_sort
{
		unsigned int pid; //point id
		DTYPE value_at_dim;
};


struct cellCoords
	{
		unsigned int dim1;
		unsigned int dim2;
	}; 


struct dim_reorder_sort
{
		unsigned int dim; //point dimension
		DTYPE variance; //variance of the points in this dimension
};



struct keyData{
		int key;
		int position;
};


//need to pass in the neighbortable thats an array of the dataset size.
//carry around a pointer to the array that has the points within epsilon though
struct neighborTableLookup
{
	int pointID;
	int indexmin;
	int indexmax;
	int * dataPtr;
};



//a struct that points to the arrays of individual data points within epsilon
//and the size of each of these arrays (needed to construct a subsequent neighbor table)
//will be used inside a vector.
struct neighborDataPtrs{
	int * dataPtr;
	int sizeOfDataArr;
};

//struct for computing the denseboxes
struct keyValDenseBoxStruct{
		uint64_t linearID;
		unsigned int pointID;
		//compare function for linearID
		bool operator<(const keyValDenseBoxStruct & other) const
		{
		   return linearID < other.linearID;
		}
	};

//maps dense box linear ids to their points
struct DenseBoxPointIDStruct{
	uint64_t linearID;
	std::vector<unsigned int> pointIDs;
	//compare function for linearID
	bool operator<(const DenseBoxPointIDStruct & other) const
	{
	   return linearID < other.linearID;
	}
};	

struct pointChunkLookupArr 
{
	unsigned int pointID;
	unsigned int chunkID;
	unsigned int idxInChunk;
};


//the neighbortable.  The index is the point ID, each contains a vector
//only for the GPU Brute force implementation
struct table{
int pointID;
std::vector<int> neighbors;
};

//index lookup table for the GPU. Contains the indices for each point in an array
//where the array stores the direct neighbours of all of the points
struct gpulookuptable{
int indexmin;
int indexmax;
};

struct grid{	
int indexmin; //Contains the indices for each point in an array where the array stores the ids of the points in the grid
int indexmax;
};

//key/value pair for the gridCellLookup -- maps the location in an array of non-empty cells
struct gridCellLookup{	
unsigned int idx; //idx in the "grid" struct array
uint64_t gridLinearID; //The linear ID of the grid cell
//compare function for linearID
  bool operator<(const gridCellLookup & other) const
  {
    return gridLinearID < other.gridLinearID;
  }
};





// struct compareThrust
// {
//   __host__ __device__
//   bool operator()(structresults const& lhs, structresults const& rhs)
//   {
//     if (lhs.pointID != rhs.pointID)
//     {
//         return (lhs.pointID < rhs.pointID);
//     }
//         return (lhs.pointInDist < rhs.pointInDist);
//   }
// };


#endif
