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

#ifndef STRUCTSDBSCAN_H
#define STRUCTSDBSCAN_H
#include <vector>
#include <stdio.h>
#include <iostream>
#include "params.h"

//struct that outlines the parameters for performance evaluation experiments
//multiple instances of DBScan, each one described per struct 
struct experiment{
	double epsilon;
	int minpts;
};

	//keeps an ordering of the experiments/variants, whether it should be clustered from scratch, and if it has started
	struct schedInfo{
		bool clusterScratch; //if it should be clustered from scratch 0-no 1-yes
		bool status; //whether it has been started 0-not started, 1-started
	};




struct cachedNeighborList{
	std::vector<int> * individualDirectNeighbours; //an array of vectors of ints that store the direct neighbours
	int datasetID; //an ID of the dataset
	int numDataPoints; //the number of data points in the dataset
	double distance; //the distance parameter used
	bool status; //whether the cache is used
};


struct dataElem{
	DTYPE x; //latitude
	DTYPE y; //longitude
	
};

//used to order the clusters that will get reused
struct pointsSqAreaStruct{
		int clusterID;
		double metric;
	};



//used to order the clusters that will get reused
struct densityStruct{
		int clusterID;
		double density;
	};


//Used for the index of point objects - multiple points per MBB
//they make the MBBs that are inserted into the tree


//Used for the index of point objects 
//they make the MBBs that are inserted into the tree

//2D MBBs

struct Rect
{
	Rect()  {}

	  DTYPE P1[2];//point
	  DTYPE MBB_min[2]; //MBB min
	  DTYPE MBB_max[2]; //MBB max
	  int pid; //point id

  	void CreateMBB(){
		MBB_min[0]=P1[0];
		MBB_max[0]=P1[0];
		MBB_min[1]=P1[1];
		MBB_max[1]=P1[1];
	} //end of function CreateMBB
};


//MBB for querying the R-tree
struct QueryRect
{
  QueryRect()  {}
  
  //two points defining the MBB: assume the two points can be given in any order:
  DTYPE P1[2];
  DTYPE P2[2];

  //MBB min and max points
  DTYPE MBB_min[2]; //MBB min
  DTYPE MBB_max[2]; //MBB max
  

  	void CreateMBB()
 	{
		
		for (int i=0; i<2; i++)
		{	
			if (P1[i]<P2[i])
			{
				MBB_min[i]=P1[i];
				MBB_max[i]=P2[i];
			}
			else
			{
				MBB_max[i]=P1[i];
				MBB_min[i]=P2[i];
			}
		}
		
	} //end of function CreateMBB


};

#endif
