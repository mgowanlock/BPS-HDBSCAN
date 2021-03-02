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
#include "tree_index.h"
#include "RTree.h" 
#include <omp.h>



//temporary list of neighbors from the R-tree for the sequential version
std::vector<unsigned int> neighborList;

//create MBBs for R-tree
void createEntryMBBs(std::vector<std::vector <DTYPE> > *NDdataPoints, struct Rect * dataRects){
	for (int i=0; i<(*NDdataPoints).size(); i++){
		
		for (int j=0; j<GPUNUMDIM; j++){
			dataRects[i].Point[j]=(*NDdataPoints)[i][j];
			dataRects[i].pid=i;
		}
		dataRects[i].CreateMBB();
	}
}




//used for the Sequential version
bool DBSCANmySearchCallbackSequential(int id, void* arg) {
  neighborList.push_back(id);
  return true; // keep going
}






double RtreeSearch(std::vector<std::vector <DTYPE> > *NDdataPoints, DTYPE epsilon, unsigned long int * numNeighbors)
{
	double runningTotalTime=0;
	unsigned long int totalNeighbors=0;
	printf("\nCalling R-tree sequential");
	
	//search and filter times
	//slows down execution, so only used for demonstration (disable for actual runs)
	#if SEARCHFILTTIME==1
	double timecomponent[2]={0,0};
	#endif

	//fpr the CPU implementation:
	//We use 1 vector to store all neighbors
	//This is to not disadvantage the CPU version over the GPU version
	//Not going to use a vector for each point
	//This can easily be expanded to use 1 vector per CPU thread
	//if we do a multithreaded implementation in the future
	

	//run experiments:
	
	

	RTree<int,DTYPE, GPUNUMDIM, float> tree; 

	//Create MBBs
	//struct to store MBB data
	//this is for 1 point per MBB
	printf("\nMemory requested for R-Tree entry rects (MiB): %f", (sizeof(struct Rect)*(*NDdataPoints).size())/(1024.0*1024.0));
	//entries in the tree
	Rect * dataRects;
	dataRects= new Rect[(*NDdataPoints).size()];
	createEntryMBBs(NDdataPoints, dataRects);


	double sequentialTime=0;

	//insert data into the R-tree
	for (int i=0; i<(*NDdataPoints).size(); i++){
		tree.Insert(dataRects[i].MBB_min,dataRects[i].MBB_max, dataRects[i].pid);	
	}


		neighborTableLookupCPU * neighborTable= new neighborTableLookupCPU[NDdataPoints->size()];
		std::vector<unsigned int > neighborTableVect;

		
		totalNeighbors=0;

		#if SEARCHFILTTIME==1
		timecomponent[0]=0;
		timecomponent[1]=0;
		#endif

		//do this so we don't get weird differences in response times over the trials based on memory allocation of vectors
		neighborList.shrink_to_fit();
		neighborTableVect.clear();
		neighborTableVect.shrink_to_fit();
		neighborTableVect.reserve(NDdataPoints->size()); //neighbortable will be at least the number of data points
															

		double tstart=omp_get_wtime();
		
		for (int i=0; i<NDdataPoints->size(); i++){

			DTYPE queryMBB_min[GPUNUMDIM]; //query MBB min
	  		DTYPE queryMBB_max[GPUNUMDIM]; //query MBB max
	  	
	  		generateQueryMBB(NDdataPoints, i, epsilon, queryMBB_min, queryMBB_max);


	  		#if SEARCHFILTTIME==1
	  		double ttreesearchstart=omp_get_wtime();
	  		#endif
			tree.Search(queryMBB_min, queryMBB_max, DBSCANmySearchCallbackSequential, NULL);
			
			#if SEARCHFILTTIME==1
			double ttreesearchend=omp_get_wtime();		
			timecomponent[0]+=ttreesearchend - ttreesearchstart;
			#endif


			#if SEARCHFILTTIME==1
			double filterstart=omp_get_wtime();
			#endif

			totalNeighbors+=filterCandidatesAddToTable(NDdataPoints, i, epsilon, &neighborList, neighborTable, &neighborTableVect);
			
			#if SEARCHFILTTIME==1
			double filterend=omp_get_wtime();		
			timecomponent[1]+=filterend - filterstart;
			#endif

			
			neighborList.clear();


			// if(i%1000==0)	
			// printf("\nFraction complete: %f",i*1.0/NDdataPoints->size());
		
		double tend=omp_get_wtime();
		
		

		sequentialTime=tend-tstart;
		*numNeighbors=neighborTableVect.size();
		printf("\nTotal neighbors: %lu",totalNeighbors);
		printf("\nsize of neighbortable vect: %zu", neighborTableVect.size());
		printf("\ntotal time to run sequential alg: %f", sequentialTime);
		
		#if SEARCHFILTTIME==1
		printf("\ntotal time searching tree: %f, total time filtering: %f, combined: %f",timecomponent[0],timecomponent[1],timecomponent[0]+timecomponent[1]);
		#endif

		runningTotalTime+=sequentialTime;
	}

	

	

	

	


	//TESTING: Print NeighborTable:
	/*
	for (int i=0; i<NDdataPoints->size(); i++)
	{
		//sort to compare against GPU implementation:
		std::sort(neighborTableVect.begin()+neighborTable[i].indexmin,neighborTableVect.begin()+neighborTable[i].indexmax+1);
		printf("\npoint id: %d, neighbors: ",i);
		for (int j=neighborTable[i].indexmin; j<=neighborTable[i].indexmax; j++){
			printf("%d,",neighborTableVect[j]);
		}
		
	}
	*/


	return sequentialTime;

}





void generateQueryMBB(std::vector<std::vector<DTYPE> > *NDdataPoints, unsigned int idx, DTYPE epsilon, DTYPE * MBB_min, DTYPE * MBB_max)
{
	for (int i=0; i<GPUNUMDIM; i++){
	MBB_min[i]=(*NDdataPoints)[idx][i]-epsilon;
	MBB_max[i]=(*NDdataPoints)[idx][i]+epsilon;
	}


}


unsigned int filterCandidatesAddToTable(std::vector<std::vector<DTYPE> > *NDdataPoints, unsigned int idx, DTYPE epsilon, std::vector<unsigned int> * neighborList, neighborTableLookupCPU * neighborTable, std::vector<unsigned int > * neighborTableVect)
{
	neighborTable[idx].pointID=idx;
	neighborTable[idx].indexmin=neighborTableVect->size();
	


	unsigned int neighborsWithinEps=0;
	
	DTYPE runningDistance=0;
	for (int i=0; i<neighborList->size(); i++){
		runningDistance=0;
		unsigned int candIdx=(*neighborList)[i];

		for (int j=0; j<GPUNUMDIM; j++){
			runningDistance+=((*NDdataPoints)[idx][j]-(*NDdataPoints)[candIdx][j])*((*NDdataPoints)[idx][j]-(*NDdataPoints)[candIdx][j]);
		}
		
		if (sqrt(runningDistance)<=epsilon){
			neighborsWithinEps++;
			neighborTableVect->push_back(candIdx);
		}
	}

	
	neighborTable[idx].indexmax=neighborTable[idx].indexmin+neighborsWithinEps-1;

	// if (idx==0 || idx==1)
	// 	printf("\ntable min/max: %d, %d",neighborTable[idx].indexmin,neighborTable[idx].indexmax);

	return neighborsWithinEps;
}












