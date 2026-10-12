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

#include "RTree.h"
#include "globalsDBSCAN.h"
#include <omp.h>
#include <stdlib.h>
#include "tree_functions.h"

//NOT USED - JUST FOR TESTING THE INDEX BEFORE ITS USED FOR DBSCAN
// bool MySearchCallback(int id, void* arg) 
// 	{
// 		//for later:
// 	  //int t_id=omp_get_thread_num();
	
// 	  int t_id=0;
// 	  fidVect[t_id].push_back(id);
// 	  //printf("\nfound in tree: %d", id, fidVect[0].);
// 	  return true; // keep going
// 	}





//function for the R-tree search.
//it's different than the other one in "tree_functions.cpp", because we want to store the searched IDs in 
//a different location inside the DBScanobject

//used for the Sequential version
bool DBSCANmySearchCallbackSequential(int id, void* arg) {
  neighbourList.push_back(id);
  return true; // keep going
}







//NOT USED, IT'S A TEST FUNCTION
// bool TESTCALLBACK(int id, void* arg) {
//   //char * a=(char*)arg;
//   int index=atoi((char*)arg);
//   printf("\n index: %d",index);
//   return true; // keep going
// }






//function for the R-tree search.
//it's different than the other one in "tree_functions.cpp", because we want to store the searched IDs in 
//a different location inside the DBScanobject

//used for the Parallel version
//has an array of vectors for each thread
//USED IN THE SEQUENTIAL VERSION, BUT WHEN WE MAKE MULTIPLE DBSCAN OBJECTS
//STILL NEED TO STORE THE NEIGHBOURLIST PER THREAD
bool DBSCANmySearchCallbackParallel(int id, void* arg) {
  
  
  int index=atoi((char*)arg);
  
  neighbourListParallel[index].push_back(id);
  return true; // keep going
}



//backup version, going to attempt to pass in index so I dont keep calling the functions below
// bool DBSCANmySearchCallbackParallelBACKUP(int id, void* arg) {
//   //to deal with compiler errors regarding the index variable
//   #if SEARCHMODE==0 || SEARCHMODE==1 || SEARCHMODE==2 || SEARCHMODE==3 || SEARCHMODE==4 || SEARCHMODE==5 || SEARCHMODE==6
//   int index=0;
//   #endif

//   #if SEARCHMODE==2 || SEARCHMODE==3
//   int tid=omp_get_thread_num();
//   int tancestor=omp_get_ancestor_thread_num(1);
//   index=(tancestor*(NSEARCHTHREADS/NINSTANCETHREADS))+tid;
//   #endif


//   #if SEARCHMODE==1 || SEARCHMODE==4
//   index=omp_get_thread_num();
//   #endif
  
//   neighbourListParallel[index].push_back(id);
//   return true; // keep going
// }

















