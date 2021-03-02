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
#include "RTree.h"


//initialized in globals.cpp

//NOT USED, JUST FOR TESTING THE R-TREE BEFORE SENT TO DBSCAN
extern std::vector <int> fidVect[1];


//pointer to the data points -- gets initialized, memory allocation in main
extern struct dataElem * dataPoints;

//temporary list of neighbors from the R-tree for the sequential version
extern std::vector<int> neighbourList;


//temporary list of neighbors from the R-tree for the parallel
//extern std::vector<int> neighbourListParallel[NSEARCHTHREADS];
extern std::vector<int> neighbourListParallel[];




//temporary global for debugging
extern double timeSpentInCallback[];

