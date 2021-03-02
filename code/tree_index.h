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

#include <vector>
#include "params.h"


//MBBS
struct Rect
{
	Rect()  {}
	  DTYPE Point[GPUNUMDIM];//point
	  DTYPE MBB_min[GPUNUMDIM]; //MBB min
	  DTYPE MBB_max[GPUNUMDIM]; //MBB max
	  int pid; //point id

  	void CreateMBB(){
		for (int i=0; i<GPUNUMDIM; i++){
			MBB_min[i]=Point[i];
			MBB_max[i]=Point[i];
		}	
	}
};


//neighbortable CPU -- indexmin and indexmax point to a single vector
struct neighborTableLookupCPU
{
	int pointID;
	int indexmin;
	int indexmax;
};

void createEntryMBBs(std::vector<std::vector <DTYPE> > *NDdataPoints, Rect * dataRects);
bool DBSCANmySearchCallbackSequential(int id, void* arg);
double RtreeSearch(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, unsigned long int * numNeighbors);
void generateQueryMBB(std::vector<std::vector<DTYPE> > *NDdataPoints, unsigned int idx, DTYPE epsilon, DTYPE * MBB_min, DTYPE * MBB_max);
unsigned int filterCandidatesAddToTable(std::vector<std::vector<DTYPE> > *NDdataPoints, unsigned int idx, DTYPE epsilon, std::vector<unsigned int> * neighborList, neighborTableLookupCPU * neighborTable, std::vector<unsigned int > * neighborTableVect);

