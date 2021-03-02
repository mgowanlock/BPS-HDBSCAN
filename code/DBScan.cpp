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
#include "structsDBSCAN.h"
#include "tree_functions.h"
#include "params.h"
#include <fstream>
#include <vector>
#include <set>

#include "DBScan.h"
#include <omp.h>
#include <iostream>
#include <unistd.h>
#include <set>



// #define NSEARCHTHREADS 1


//stores the temporary neighbour list in SEQUENTIAL DBScan. Made global, but only accessed by the DBScan class
//this is because of issues using the callback method for the R-tree inside in the DBScan class
std::vector<int> neighbourList;

//stores the temporary neighbour list in SEQUENTIAL DBScan, but that needs memory space for multiple vectors for 
//different threads. Made global, but only accessed by the DBScan class
//this is because of issues using the callback method for the R-tree inside in the DBScan class
//pass in the number of threads
//used in the function:  algDBScanParallel (although it isn't actually parallel)
std::vector<int> neighbourListParallel[NSEARCHTHREADS];



	//constructor for the pure sequential implementation:
	//constructor
	DBScan::DBScan(std::vector<struct dataElem> *ptrData, DTYPE epsilon, int minimumPts, 
		RTree<int,DTYPE,2,float> *indexPtr)
	{
		distance=epsilon;
		minPts=minimumPts;

		//pointer to the data
		dataPoints=ptrData;
		
		//pointer to the R-tree index
		tree=indexPtr;

		//pointer to the MPB lookup array
		//ptrMPBlookup=ptr_MPB_lookup;
		
		//initialize vector that keeps track of the points that have been visited
		initializeVisitedPoints((*ptrData).size());

		//initialize the vector that keeps track of the cluster assignment for the points
		initializeClusterIDs((*ptrData).size());


		//reserve initial space for the neighbourList:

		for (int i=0; i<NSEARCHTHREADS; i++)
		{
		neighbourListParallel[i].reserve(10000);
		}

		


		//the number of clusters
		clusterCnt=0;


	


	}


	//before testing the problem with memory allocation
	void DBScan::algDBScan()
	{
		printf("\ntotal data points: %zu", (*dataPoints).size());


		clusterCnt=0;

		//neighborSet is the current set of points being searched that belongs to a cluster
		std::vector<int>neighborSet;

		for (unsigned int i=0; i<(*dataPoints).size(); i++){
			
			if ((i%1000000)==0)
			printf("\nDBSCAN iteration %u/%lu", i,(*dataPoints).size());cout.flush();	

			//see if the point has been visited, if so, go onto the next loop iteration
			if (visited[i]==true){
				continue;
			}

			//clear the vector of neighbors
			neighborSet.clear();

			//mark the point as visited:
			visited[i]=true;
			
			//get the neighbors of the data point
			dataElem tmpDataPoint=(*dataPoints)[i];
			


			getNeighbours(&tmpDataPoint,distance, &setIDsInDist);

			
			//if the number of neighbors is less than the number required for a cluster,
			//then it is noise.  The noise will be cluster 0.
			if (setIDsInDist.size()<minPts)
			{
				clusterIDs[i]=0;
			}
				
			
			//if there's enough points to make a cluster
			
			else
			{
				
				clusterCnt++;
				//make a new cluster with the correct cluster ID 
				clusterIDs[i]=clusterCnt;	


				//printf("\n***1size of neighbor set: %d", neighborSet.size());

				//assign the neighbor ids to the neighborSet, which may be expanded when searching
				//through the neighbors
				neighborSet=setIDsInDist;
				//copyVect(&neighborSet,&neighbourList);
				
				
				//expand the cluster
				 	
				 while (neighborSet.size()!=0){
				 	//examine the point that's the last one in the neighbor set
				 	int pntID=neighborSet.back(); 
				 	
				 	//if this point has been visited before
				 	if (visited[pntID]==true){
				 	//remove the value from the list of neighbors to check 	
				 	neighborSet.pop_back();
					continue;
					}

					//mark the point as visited:
					visited[pntID]=true;

					//get the neighbors of the data point
					dataElem tmpDataPoint=(*dataPoints)[pntID];

					getNeighbours(&tmpDataPoint,distance, &setIDsInDist);
					//getNeighboursBruteForce(&tmpDataPoint,distance);

					//if the number of neighbors is greater than the number required to form a cluster
					if (setIDsInDist.size()>=minPts)
					{
						unsigned long int sizeNeighbors=neighborSet.size();

						neighborSet.resize(sizeNeighbors+setIDsInDist.size());
						for (unsigned int j=0; j<setIDsInDist.size(); j++)
						{
							neighborSet[sizeNeighbors+j]=setIDsInDist[j];
						}

						//assign the neighbor ids to the neighborSet
						// copyVect(&neighborSet,&setIDsInDist);		
					}
					//if the point has part not been assigned to a cluster yet
					
					//AFTER FIXING THE BORDER POINTS TO BE PART OF A CLUSTER
					if (clusterIDs[pntID]==0){
						clusterIDs[pntID]=clusterCnt;							
					}
					
					


				 } //end of while loop
			} //end of else
			

						
			//testTotalNeighbors+=neighbourList.size();

			//now have the vector of ids within the distance
			//setIDsInDist
			
			

		} //end of main for loop

		//increment the total cluster count by 1 because cluster 0 is for the noise data points
		clusterCnt++;

		


		// printf("\n***printing cluster array:");
		// for (int i=0; i<clusterIDs.size(); i++)
		// {

		// 	printf("\ni, ID: %d, %d",i,clusterIDs[i]);
		// }
		// printf("\n***end of printing cluster array:");



		printf("\ntotal clusters, including cluster 0, which is the noise points: %d", clusterCnt);
		


	}





//TESTING using a set to store the epsilon neighborhood
void DBScan::algDBScanWithSet()
	{
		printf("\ntotal data points: %zu", (*dataPoints).size());

		// printf("\n***************\nData in DBSCAN ref.:\n");
		// for (int i=0; i<(*dataPoints).size(); i++)
		// {
		// 	printf("i: %d, %f, %f\n",i, (*dataPoints)[i].x,(*dataPoints)[i].y);
		// }


		clusterCnt=0;

		//neighborSet is the current set of points being searched that belongs to a cluster
		std::set<int>neighborSet;
		//iterator for the neighborSet
		std::set<int>::iterator it;


		//Changed std::vector<int>neighborSet to std set due to large numbers of neighbors
		//Need only unique elements, but didn't want the performance degradation of the set

		for (int i=0; i<(*dataPoints).size(); i++){
				

			//see if the point has been visited, if so, go onto the next loop iteration
			if (visited[i]==true){
				continue;
			}

			//clear the vector of neighbors
			neighborSet.clear();

			//mark the point as visited:
			visited[i]=true;
			
			//get the neighbors of the data point
			dataElem tmpDataPoint=(*dataPoints)[i];
			


			getNeighbours(&tmpDataPoint,distance, &setIDsInDist);

			
			//if the number of neighbors is less than the number required for a cluster,
			//then it is noise.  The noise will be cluster 0.
			if (setIDsInDist.size()<minPts)
			{
				clusterIDs[i]=0;
			}
				
			
			//if there's enough points to make a cluster
			
			else
			{
				
				clusterCnt++;
				//make a new cluster with the correct cluster ID 
				clusterIDs[i]=clusterCnt;	


				//printf("\n***1size of neighbor set: %d", neighborSet.size());

				//assign the neighbor ids to the neighborSet, which may be expanded when searching
				//through the neighbors
				
				//original
				// neighborSet=setIDsInDist;

				//add neighbors to neighborset
				for (unsigned int j=0; j<setIDsInDist.size(); j++)
				{
					neighborSet.insert(setIDsInDist[j]);
				}
				
				
				
				//expand the cluster
				 	
				 while (neighborSet.size()!=0){
				 	//examine the point that's the last one in the neighbor set
				 	
				 	it=neighborSet.begin();

				 	int pntID=*it;
				 	//original
				 	// int pntID=neighborSet.back(); 
				 	
				 	//if this point has been visited before
				 	if (visited[pntID]==true){
				 	//remove the value from the list of neighbors to check 	
				 	//original
				 	// neighborSet.pop_back();
				 	neighborSet.erase(it);	
					continue;
					}

					//mark the point as visited:
					visited[pntID]=true;

					//get the neighbors of the data point
					dataElem tmpDataPoint=(*dataPoints)[pntID];

					getNeighbours(&tmpDataPoint,distance, &setIDsInDist);
					//getNeighboursBruteForce(&tmpDataPoint,distance);

					//if the number of neighbors is greater than the number required to form a cluster
					if (setIDsInDist.size()>=minPts)
					{
						//assign the neighbor ids to the neighborSet
						//original
						// copyVect(&neighborSet,&setIDsInDist);		
						for (unsigned int j=0; j<setIDsInDist.size(); j++)
						{
							neighborSet.insert(setIDsInDist[j]);	
						}
					}
					//if the point has part not been assigned to a cluster yet
					
					//AFTER FIXING THE BORDER POINTS TO BE PART OF A CLUSTER
					if (clusterIDs[pntID]==0){
						clusterIDs[pntID]=clusterCnt;							
					}
					
					


				 } //end of while loop
			} //end of else
			

						
			//testTotalNeighbors+=neighbourList.size();

			//now have the vector of ids within the distance
			//setIDsInDist
			
			

		} //end of main for loop

		//increment the total cluster count by 1 because cluster 0 is for the noise data points
		clusterCnt++;

		


		// printf("\n***printing cluster array:");
		// for (int i=0; i<clusterIDs.size(); i++)
		// {

		// 	printf("\ni, ID: %d, %d",i,clusterIDs[i]);
		// }
		// printf("\n***end of printing cluster array:");



		printf("\ntotal clusters, including cluster 0, which is the noise points: %d", clusterCnt);
		


	}



int DBScan::getDBScanNumClusters()
{
	return clusterCnt;
}



//thismethod appends an MBB by epsilon
void DBScan::appendMBBByEpsilon(DTYPE * MBB_min, DTYPE * MBB_max, DTYPE eps)
{
	MBB_min[0]=MBB_min[0]-eps;
	MBB_max[0]=MBB_max[0]+eps;
	MBB_min[1]=MBB_min[1]-eps;
	MBB_max[1]=MBB_max[1]+eps;

}



	void DBScan::printSet(std::vector<int> * set)
	{
		for (int i=0; i<(*set).size(); i++)
		{
			printf("\nset ID: %d",(*set)[i]);	
		}
	}

	//appends the elements from the source vector to the end of the dest vector
	void DBScan::copyVect(std::vector<int> * dest, std::vector<int> * source)
	{

		// printf("\ncpyvect size of dest: %d", dest->size());
		// printf("\ncpyvect size of source: %d", source->size());
		for (int i=0; i<(*source).size(); i++)
		{
			(*dest).push_back((*source)[i]);	
		}
	}



	//R-tree version of the algorithm, thus making the algorithm O(nlogn), where 
	//n is the number of points
	//setIDsInDistPtr is a pointer to the buffer used to store the ids that are actually within the distance,
	//not the candidates from the R-tree search
	void DBScan::getNeighbours(struct dataElem * point, DTYPE distance, std::vector<int> * setIDsInDistPtr){
		
		//first, clear the temporary list of neighbours-- these are the ones from the R-tree
		//not the final ones that have been filtered
		//should maintain the memory allocation, so the vector doesn't need to grow again
		neighbourList.clear();
		//setIDsInDistPtr->clear();

		//construct an MBB for the point
		DTYPE MBB_min[2];
		DTYPE MBB_max[2];
		DTYPE MBB_min2[2]; //used only if 2 MBBs are required around the longitudinal equator 
		DTYPE MBB_max2[2]; //(between 0 and 360 degrees)



		
		generateMBBNormal(point, distance, MBB_min, MBB_max);
			
			(*tree).Search(MBB_min,MBB_max, DBSCANmySearchCallbackSequential, NULL);
			
			
		//after the candidate set has been found, then we filter the points to find those that are actually 
		//within the distance		
		filterCandidates(point, &neighbourList, distance, setIDsInDistPtr);		


		
		

		
		


		
	}

	//R-tree version of the algorithm, thus making the algorithm O(nlogn), where 
	//n is the number of points
	//this is for the parallel version, because we need different vectors to hold the candidates
	//for different threads
	//setIDsInDistPtr is the vector of ids of the points that are actually withiun the distance, not the candidates
	//from the R-tree
	void DBScan::getNeighboursParallel(struct dataElem * point, DTYPE distance, std::vector<int> * setIDsInDistPtr){
		
		//first, clear the temporary list of neighbours-- these are the ones from the R-tree
		//not the final ones that have been filtered
		//should maintain the memory allocation, so the vector doesn't need to grow again
		
		int tid=omp_get_thread_num();
		neighbourListParallel[tid].clear();
		
		


		//construct an MBB for the point
		DTYPE MBB_min[2];
		DTYPE MBB_max[2];
		DTYPE MBB_min2[2]; //used only if 2 MBBs are required around the longitudinal equator 
		DTYPE MBB_max2[2]; //(between 0 and 360 degrees)


		

		generateMBBNormal(point, distance, MBB_min, MBB_max);
		
			double tstart=omp_get_wtime();
			(*tree).Search(MBB_min,MBB_max, DBSCANmySearchCallbackParallel, NULL);
			double tend=omp_get_wtime();
			// totalTimeSearchingTree[tid]+=tend-tstart;
		//after the candidate set has been found, then we filter the points to find those that are actually 
		//within the distance		
		
		filterCandidates(point, &neighbourListParallel[tid], distance, setIDsInDistPtr);		
		

		

		
		

		
	}



	//This function takes the candidate set from the R-tree and filters them to find those that are
	//actually within the threshold distance, epsilon
	//takes as input the point, the candidateSet pointers, and the distance
	int DBScan::filterCandidates(struct dataElem * point, std::vector<int> * candidateSet, DTYPE distance, std::vector<int> * setIDsInDistPtr){

		//first, clear the vector.  It should maintain its memory allocation so it can only grow.
		setIDsInDistPtr->clear();

		//printf("\n candidate set size: %d", (*candidateSet).size());

		for (int i=0; i<(*candidateSet).size();i++)
		{
			//the ID of a candidate point
			int candID=(*candidateSet)[i];

			//make a temp copy of the candidate data point
			dataElem candPoint=(*dataPoints)[candID];

			//calculate the distance between the point and the candidate point
			//if it is within the threshold distance, then add it to the vector of IDs that are within
			//the threshold distance
			if (EuclidianDistance(point,&candPoint)<=distance)
			{
				setIDsInDistPtr->push_back(candID);
			}
		}

	}







//THIS IS FOR THE VERSION WITH MULTIPLE POINTS PER MBB
	int DBScan::filterCandidatesMPB(struct dataElem * point, std::vector<int> * candidateSet, DTYPE distance, std::vector<int> * setIDsInDistPtr){

		//first, clear the vector.  It should maintain its memory allocation so it can only grow.
		setIDsInDistPtr->clear();
		
		//#pragma omp parallel for num_threads(4) 
		
		for (int i=0; i<(*candidateSet).size(); i++)
		{

					int tid=omp_get_thread_num();
					//printf("\ntid: %d", tid);


					//DECOMPOSE THE CANDIDATE SET, WHICH CONTAINS THE RESULT OF THE OVERLAPPING MBBS
					//WITH THE MULTIPLE POINT BOXES (MULTIPLE POINTS PER MBB)
					//the ID of a candidate point
					int MBBID=(*candidateSet)[i];

															
					for (int j=0; j<(*ptrMPBlookup)[MBBID].size(); j++)
					{
					int candID=(*ptrMPBlookup)[MBBID][j];
					
					//make a temp copy of the candidate data point
					dataElem candPoint=(*dataPoints)[candID];

					//calculate the distance between the point and the candidate point
					//if it is within the threshold distance, then add it to the vector of IDs that are within
					//the threshold distance
					if (EuclidianDistance(point,&candPoint)<=distance)
					{
						//#pragma omp critical
						setIDsInDistPtr->push_back(candID);
					}
					
				}//end of inner for loop
		
		} //end of outer for loop

	}



	double degrees_to_radian(double deg)
	{
    return deg * M_PI / 180.0;
	}



	//The 2D EuclidianDistance
	double DBScan::EuclidianDistance(struct dataElem * point1, struct dataElem * point2)
	{
		return(sqrt(((point1->x-point2->x)*(point1->x-point2->x))+((point1->y-point2->y)*(point1->y-point2->y))));
	}


	//initialize all of the points to initially not be visited
	void DBScan::initializeVisitedPoints(int size)
	{
		for (int i=0; i<size; i++){
			visited.push_back(false);
		}
	}

	void DBScan::initializeClusterIDs(int size)
	{
		for (int i=0; i<size; i++){
			clusterIDs.push_back(0);
		}
	}

	//generate a query MBB around the point to search for the values
	//this supports the other two generate MBB methods 
	void DBScan::generateMBB(struct dataElem * point, DTYPE distance, DTYPE * MBB_min, DTYPE * MBB_max)
	{
		//the MBB is made up by the same time and electron content values, but with the distance added or 
		//subtracted from the MBB

		
		MBB_min[0]=(point->x)-distance;
		MBB_min[1]=(point->y)-distance;
		
		MBB_max[0]=(point->x)+distance;
		MBB_max[1]=(point->y)+distance;

	}

	//generate a query MBB around the point to search for the values
	//returns true if it was able to generate the query using a single MBB
	//returns false if it needs two MBBs because the query wraps around the longitude of 360 degrees
	bool DBScan::generateMBBNormal(struct dataElem * point, DTYPE distance, DTYPE * MBB_min, DTYPE * MBB_max)
	{
		generateMBB(point, distance, MBB_min, MBB_max);
		
		return true;
	}

	


	//output the clusters
	//takes as input the filename to output the data to
	//the resulting file is a python script
	void DBScan::outputClusters(char * fname)
	{

		ofstream output(fname,ios::out);
		printf("\nOutputting cluster data to: %s",fname);

		output<<"import matplotlib.pyplot as plt";
		output<<"\nimport random";
		output<<"\nimport numpy as np";
		output<<endl;
		output<<endl;

		char alphabet[26] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
		int colorcnt=0;

		int pntcnt=0;
		int ptsincluster=0;
		int totalpointsincluster[clusterCnt];

		//printf("\n**************Outputting clusters: ");
		for (int i=0; i<clusterCnt;i++)
		{
			ptsincluster=0;
			output<<"\n"<<alphabet[i%26]<<i<<"=[";
			for (int j=0; j<clusterIDs.size(); j++)
			{

				if (clusterIDs[j]==i)
				{	
					//total number of points
					pntcnt++;
					//the number of points in the individual cluster
					ptsincluster++;
					
					//output as latitude vs. longitude (y val vs x val)
					//printf("[%f,%f],", (*dataPoints)[j].y, (*dataPoints)[j].x);
					output<<"["<<(*dataPoints)[j].y<<","<<(*dataPoints)[j].x<<"],";
				}
			}

			totalpointsincluster[i]=ptsincluster;

			//printf("]",alphabet[i%26]);
			output<<"]";
			
		}

		//printf("\n**************Outputting plot string: ");
		output<<endl<<endl;	
		//output the string that plots the data
		for (int i=0; i<clusterCnt;i++)
		{

			//printf("\n%c%d=[",alphabet[i%26],i);
			//printf("\nplt.plot(*zip(*%c%d), marker='o', color=np.random.rand(3,1), linestyle='None')",alphabet[i%26],i);
			output<<"\n"<<"plt.plot(*zip(*"<<alphabet[i%26]<<i<<"), marker='o', color=np.random.rand(3,1), linestyle='None')";
		}

		//printf("\ntotal points output: %d", pntcnt);

		output<<"\nplt.axis([0, 360, 0, 180])";
		output<<"\nplt.show()";

		output<<"\n\n\n#Total points output for plot: "<<pntcnt;
		output<<"\n#Total clusters for plot, including cluster 0, which is the noise point cluster: "<<clusterCnt;

		//output the number of points assigned to each cluster
		//printf("\nOutput of the cluster number and the number of points in the cluster: ");
		
		output<<"\n#Output of the cluster number and the number of points in the cluster:";
		
		for (int i=0; i<clusterCnt; i++)
		{
			//printf("\ncluster:%d, points:%d ",i,totalpointsincluster[i]);
			output<<"\n#Cluster: "<<i<<", points: "<<totalpointsincluster[i];
		}

		output<<"\n#Total points: "<<pntcnt;

		//printf("\n#Total points: %d",pntcnt);


		output.close();

		//make a unique set of clusters:
		//std::set<int>clusterSet;

		//insert the list of cluster IDs
		
		// for (int i=0; i<clusterIDs.size(); i++)
		// {
		// 	for (int j=0; j<(*dataPoints).size(); j++)
		// 	{
		// 		if ()
		// 	}

		// }	
	}


	//this function takes as input an array that is a series of points and their assigned cluster
	//it compares the cluster assignment found in the array to the cluster assignment found in this object
	//note that the data size must be equal 
	//REPLACED BY NEW METHOD
	// void DBScan::calcErrorPointsInCluster(std::vector<int> * inputClusterArray)
	// {
	// 	//make sure the arrays are of the same length
	// 	if ((*inputClusterArray).size()!=clusterIDs.size())
	// 	{
	// 		printf("\n*********\nError: cannot compare the two point sets because they aren't of the same size!\n***********");
	// 		return;
	// 	}

	// 	unsigned int numEqual=0;
	// 	unsigned int numNotEqual=0;

	// 	for (int i=0; i<clusterIDs.size(); i++)
	// 	{
	// 		if(clusterIDs[i]==(*inputClusterArray)[i]){
	// 			numEqual++;
	// 		}
	// 		else
	// 		{
	// 			numNotEqual++;
	// 		}


	// 	}

	// 	printf("\nSimilarity between the clusters: %f, incorrectly classified: %f", (numEqual*1.0/clusterIDs.size()),(numNotEqual*1.0/clusterIDs.size()));

	// } //end of function	


	//takes as input two arrays that have the assigned clusters corresponding to the datapoints
	void DBScan::DetermineErrorTwoClusterResults(std::vector<int> * c1, std::vector<int> * c2)
	{
			
		printf("\nIn method to compare two clusters similarity metric");cout.flush();

		printf("\nsize of array of datapoints1: %zu, size of array of datapoints2: %zu",c1->size(),c2->size());cout.flush();
	
		//the sizes must be equal, or the metric will not work. Thesize should be the number of datapoints in the dataset
		if (c1->size()!=c2->size())
		{
			printf("\n**********\nERROR WHEN TESTING THE SIMILARITY/ERROR OF TWO CLUSTERING RESULTS. THE NUMBER OF POINTS IN EACH ARRAY ARE NOT EQUAL\n\n");
			return;
		}

		const int sizeData=int(c1->size());
		printf("\nsize of data in var: %d",sizeData);
			
		//score array that will be averaged to get the final score
		//keeps track of the score for each point
		//double scoreArr[sizeData];
		double * scoreArr;
		scoreArr=new double[sizeData];

		//visited array so that as we iterate over the points we get rid of those that have already been filtered as noise
		//bool visitedArr[sizeData];
		bool * visitedArr;
		visitedArr=new bool[sizeData];

		//initialize:
		for (int i=0; i<sizeData;i++)
		{
			scoreArr[i]=0;
			visitedArr[i]=false;
		}
		
		//int max1=*std::max_element(c1->begin(),c1->end());
		//printf("\nmax1: %d",max1);cout.flush();

		
		//find the number of clusters from each clustering result
		int max1=0;
		
		

		
		for (int i=0; i<sizeData;i++)
		{
			
			//the max cluster id in the first one
			//printf("\nmax1: %d",max1);cout.flush();
			if (((*c1)[i])>max1)
			{
				max1=(*c1)[i];
			}
			
		}
		
		int max2=0;
		for (int i=0; i<c2->size();i++)
		{
			//the max cluster id in the first one
			if (((*c2)[i])>max2)
			{
			max2=(*c2)[i];
			}
		}	
		
		
		
		
		//want to include the final cluster
		max1++;
		max2++;

		printf("\nnum clusters in list1, list2: %d,%d",max1,max2);cout.flush();
		
		
		//create 2 arrays of the clusters and their associated ids
		//the number of clusters may not be equal, but the total number of points is equal
		std::vector<int > clusterArr1[max1];
		std::vector<int > clusterArr2[max2];
		

		for (int i=0; i<sizeData;i++)
		{
			int clusterid1=(*c1)[i];
			int clusterid2=(*c2)[i];
		
			
			clusterArr1[clusterid1].push_back(i);
		 	clusterArr2[clusterid2].push_back(i);

		}


		//sort the array of clusters for cluster set 1
		for (int i=0; i<max1; i++)
		{
			std::sort(clusterArr1[i].begin(),clusterArr1[i].end());
		}

		//sort the array of clusters for cluster set 2
		for (int i=0; i<max2; i++)
		{
			std::sort(clusterArr2[i].begin(),clusterArr2[i].end());
		}


		//first we score the points that are mismatched with regards to noise points
		//i.e., if one point is assigned as noise, and another is not noise
		//then the point gets a quality score of 0
		//if they are both noise, they each point gets a score of 1

		int cntNoiseError=0;
		int cntNoiseEqual=0;

		

		for (int i=0; i<sizeData;i++)
		{
			
			if (((*c1)[i]==0)&&((*c2)[i]!=0))
			{
				cntNoiseError++;
				scoreArr[i]=0;
				visitedArr[i]=true;

			}

			if (((*c1)[i]!=0)&&((*c2)[i]==0))
			{
				cntNoiseError++;
				scoreArr[i]=0;	
				visitedArr[i]=true;
			}

			//both are noise points:
			if (((*c1)[i]==0)&&((*c2)[i]==0))
			{
				cntNoiseEqual++;
				scoreArr[i]=1;
				visitedArr[i]=true;
			}
		}
		
		printf("\nmismatched noise points: %d, agreement noise points: %d", cntNoiseError, cntNoiseEqual);
		
		

		//for all of the points that are in clusers (not noise) we now calculate the 
		//score using the Second Object Quality Function Section 8.2 in "DBDC: Density Based Distributed Clustering"
		//http://www.dbs.informatik.uni-muenchen.de/Publikationen/Papers/EDBT_04_DBDC.pdf

		//upon entering the loop
		
		//omp_set_num_threads(4);
		//printf("\nRUNNING THE VALIDATION IN PARALLEL WITH THREADS!!");
		//#pragma omp parallel for 
		
		//we need to count the number of elements in the intersection of two sets, and the union of two sets.
		//we dont need the actual values themselves, so we can avoid taking the actual union and intersection
		//which is more computationally expensive
		
		double totalScoreMiscluster=0;
		//upon entering the loop
		
		omp_set_num_threads(16);
		printf("\nRUNNING THE VALIDATION IN PARALLEL WITH THREADS!!");
		#pragma omp parallel for 
		for (int i=0; i<sizeData;i++)
		{
			//if point was already noise, we dealt with the point already
			if (visitedArr[i]==true)
			{
				continue;
			}


			//get the two cluster ids for the point from the two experiments
			int clusterid1=(*c1)[i];
			int clusterid2=(*c2)[i];

			std::vector<int>ids_in_cluster1=clusterArr1[clusterid1];
			std::vector<int>ids_in_cluster2=clusterArr2[clusterid2];
			
			//find the number of points that intersect:
			//the sets have already been presorted.
			//for each element in set 1, we see if its found in set 2 doing a binary search and count the number of times this occurs
			int cntIntersection=0;
			int cntUnion=0;

			for (int j=0; j<ids_in_cluster1.size();j++)
			{
				int findElem=ids_in_cluster1[j];
				if(std::binary_search(ids_in_cluster2.begin(), ids_in_cluster2.end(), findElem))
				{
					cntIntersection++;
				}
			}

			//need to pre-allocate memory for the unionSet vector, which is at worst case the sum of the size of both clusters
			int preallocateUnion=ids_in_cluster1.size()+ids_in_cluster2.size();
			std::vector<int> unionSet(preallocateUnion);
			std::vector<int>::iterator it2;

			//get the union of the two clusters and store in the unionSet vector
			it2=std::set_union(ids_in_cluster1.begin(), ids_in_cluster1.end(), ids_in_cluster2.begin(), ids_in_cluster2.end(), unionSet.begin());
			unionSet.resize(it2-unionSet.begin());

			scoreArr[i]=(cntIntersection*1.0)/(unionSet.size()*1.0);

			//for testing:
			if (scoreArr[i]!=1.0){
			totalScoreMiscluster+=1.0-scoreArr[i];
			}
			//printf("\nscore for point: %d, %f",i,scoreArr[i]);



			/*
			//need to pre-allocate memory for the intersection vector, which at worst case is the value of the minimum of the number of ids in each cluster being compared
			int preallocateIntersection=min(ids_in_cluster1.size(),ids_in_cluster2.size());
			//printf("\n*****\nsize of intersection allocated: %d",preallocateIntersection);
			std::vector<int> intersectionSet(preallocateIntersection);
			std::vector<int>::iterator it1;
			//need to pre-allocate memory for the unionSet vector, which is at worst case the sum of the size of both clusters
			int preallocateUnion=ids_in_cluster1.size()+ids_in_cluster2.size();
			//printf("\nsize of union allocated: %d",preallocateUnion);
			std::vector<int> unionSet(preallocateUnion);
			std::vector<int>::iterator it2;
			// for (int j=0; j<ids_in_cluster2.size();j++)
			// {
			// 	printf("\nid: %d",ids_in_cluster2[j]);
			// }	


			
			
			
			//get the intersection of the two clusters and store in the intersection vector
			it1=std::set_intersection (ids_in_cluster1.begin(), ids_in_cluster1.end(), ids_in_cluster2.begin(), ids_in_cluster2.end(), intersectionSet.begin());
			intersectionSet.resize(it1-intersectionSet.begin());
			//printf("\nsize of intersection after intersection taken: %d",intersectionSet.size());
			
			

			
			

			

			//get the union of the two clusters and store in the unionSet vector
			it2=std::set_union(ids_in_cluster1.begin(), ids_in_cluster1.end(), ids_in_cluster2.begin(), ids_in_cluster2.end(), unionSet.begin());
			unionSet.resize(it2-unionSet.begin());
			//printf("\nsize of union after union taken: %zu", unionSet.size());
			
			//the score is the (size of the intersection set)/(size of the union set)
			
			scoreArr[i]=(intersectionSet.size()*1.0)/(unionSet.size()*1.0);
			*/				

			


		}	

		printf("\nfraction lost due to mismatches between clusters: %f", totalScoreMiscluster/(sizeData*1.0));

		//final score:
		double sum=0;
		for (int i=0; i<sizeData;i++)
		{
			sum+=scoreArr[i];

		}

		printf("\nFinal Error metric: %f",(1.0*sum)/(sizeData*1.0));
	
	

	}//end of method




	//takes as input two arrays that have the assigned clusters corresponding to the datapoints
	//initial version that works
	void DBScan::DetermineErrorTwoClusterResultsBACKUP(std::vector<int> * c1, std::vector<int> * c2)
	{
			
		printf("\nIn method to compare two clusters similarity metric");cout.flush();

		printf("\nsize of array of datapoints1: %zu, size of array of datapoints2: %zu",c1->size(),c2->size());cout.flush();
	
		//the sizes must be equal, or the metric will not work. Thesize should be the number of datapoints in the dataset
		if (c1->size()!=c2->size())
		{
			printf("\n**********\nERROR WHEN TESTING THE SIMILARITY/ERROR OF TWO CLUSTERING RESULTS. THE NUMBER OF POINTS IN EACH ARRAY ARE NOT EQUAL\n\n");
			return;
		}

		const int sizeData=int(c1->size());
		printf("\nsize of data in var: %d",sizeData);
			
		//score array that will be averaged to get the final score
		//keeps track of the score for each point
		//double scoreArr[sizeData];
		double * scoreArr;
		scoreArr=new double[sizeData];

		//visited array so that as we iterate over the points we get rid of those that have already been filtered as noise
		//bool visitedArr[sizeData];
		bool * visitedArr;
		visitedArr=new bool[sizeData];

		//initialize:
		for (int i=0; i<sizeData;i++)
		{
			scoreArr[i]=0;
			visitedArr[i]=false;
		}
		
		//int max1=*std::max_element(c1->begin(),c1->end());
		//printf("\nmax1: %d",max1);cout.flush();

		
		//find the number of clusters from each clustering result
		int max1=0;
		
		

		
		for (int i=0; i<sizeData;i++)
		{
			
			//the max cluster id in the first one
			//printf("\nmax1: %d",max1);cout.flush();
			if (((*c1)[i])>max1)
			{
				max1=(*c1)[i];
			}
			
		}
		
		int max2=0;
		for (int i=0; i<c2->size();i++)
		{
			//the max cluster id in the first one
			if (((*c2)[i])>max2)
			{
			max2=(*c2)[i];
			}
		}	
		
		
		
		
		//want to include the final cluster
		max1++;
		max2++;

		printf("\nnum clusters in list1, list2: %d,%d",max1,max2);cout.flush();
		
		
		//create 2 arrays of the clusters and their associated ids
		//the number of clusters may not be equal, but the total number of points is equal
		std::vector<int > clusterArr1[max1];
		std::vector<int > clusterArr2[max2];
		

		for (int i=0; i<sizeData;i++)
		{
			int clusterid1=(*c1)[i];
			int clusterid2=(*c2)[i];
		
			
			clusterArr1[clusterid1].push_back(i);
		 	clusterArr2[clusterid2].push_back(i);

		}


		//sort the array of clusters for cluster set 1
		for (int i=0; i<max1; i++)
		{
			std::sort(clusterArr1[i].begin(),clusterArr1[i].end());
		}

		//sort the array of clusters for cluster set 2
		for (int i=0; i<max2; i++)
		{
			std::sort(clusterArr2[i].begin(),clusterArr2[i].end());
		}


		//first we score the points that are mismatched with regards to noise points
		//i.e., if one point is assigned as noise, and another is not noise
		//then the point gets a quality score of 0
		//if they are both noise, they each point gets a score of 1

		int cntNoiseError=0;
		int cntNoiseEqual=0;

		int cntNoiseC1=0;
		int cntNoiseC2=0;		

		for (int i=0; i<sizeData;i++)
		{
			
			if (((*c1)[i]==0)&&((*c2)[i]!=0))
			{
				cntNoiseError++;
				scoreArr[i]=0;
				visitedArr[i]=true;
				cntNoiseC1++;

			}

			if (((*c1)[i]!=0)&&((*c2)[i]==0))
			{
				cntNoiseError++;
				scoreArr[i]=0;	
				visitedArr[i]=true;
				cntNoiseC2++;
			}

			//both are noise points:
			if (((*c1)[i]==0)&&((*c2)[i]==0))
			{
				cntNoiseEqual++;
				scoreArr[i]=1;
				visitedArr[i]=true;
				cntNoiseC1++;
				cntNoiseC2++;
			}
		}
		
		
		printf("\nNum noise points C1: %d, C2: %d", cntNoiseC1, cntNoiseC2);cout.flush();
		printf("\nmismatched noise points: %d, agreement noise points: %d", cntNoiseError, cntNoiseEqual);cout.flush();
		
		

		//for all of the points that are in clusers (not noise) we now calculate the 
		//score using the Second Object Quality Function Section 8.2 in "DBDC: Density Based Distributed Clustering"
		//http://www.dbs.informatik.uni-muenchen.de/Publikationen/Papers/EDBT_04_DBDC.pdf

		//upon entering the loop
		
		omp_set_num_threads(4);
		printf("\nRUNNING THE VALIDATION IN PARALLEL WITH THREADS!!");
		#pragma omp parallel for 
		for (int i=0; i<sizeData;i++)
		{
			//if point was already noise, we dealt with the point already
			if (visitedArr[i]==true)
			{
				continue;
			}


			//get the two cluster ids for the point from the two experiments
			int clusterid1=(*c1)[i];
			int clusterid2=(*c2)[i];

			std::vector<int>ids_in_cluster1=clusterArr1[clusterid1];
			std::vector<int>ids_in_cluster2=clusterArr2[clusterid2];
			//need to pre-allocate memory for the intersection vector, which at worst case is the value of the minimum of the number of ids in each cluster being compared
			int preallocateIntersection=min(ids_in_cluster1.size(),ids_in_cluster2.size());
			//printf("\n*****\nsize of intersection allocated: %d",preallocateIntersection);
			std::vector<int> intersectionSet(preallocateIntersection);
			std::vector<int>::iterator it1;
			//need to pre-allocate memory for the unionSet vector, which is at worst case the sum of the size of both clusters
			int preallocateUnion=ids_in_cluster1.size()+ids_in_cluster2.size();
			//printf("\nsize of union allocated: %d",preallocateUnion);
			std::vector<int> unionSet(preallocateUnion);
			std::vector<int>::iterator it2;
			// for (int j=0; j<ids_in_cluster2.size();j++)
			// {
			// 	printf("\nid: %d",ids_in_cluster2[j]);
			// }	


			
			
			
			//get the intersection of the two clusters and store in the intersection vector
			it1=std::set_intersection (ids_in_cluster1.begin(), ids_in_cluster1.end(), ids_in_cluster2.begin(), ids_in_cluster2.end(), intersectionSet.begin());
			intersectionSet.resize(it1-intersectionSet.begin());
			//printf("\nsize of intersection after intersection taken: %d",intersectionSet.size());
			
			

			
			

			

			//get the union of the two clusters and store in the unionSet vector
			it2=std::set_union(ids_in_cluster1.begin(), ids_in_cluster1.end(), ids_in_cluster2.begin(), ids_in_cluster2.end(), unionSet.begin());
			unionSet.resize(it2-unionSet.begin());
			//printf("\nsize of union after union taken: %zu", unionSet.size());

			//the score is the (size of the intersection set)/(size of the union set)
			scoreArr[i]=(intersectionSet.size()*1.0)/(unionSet.size()*1.0);
			

			//printf("\nscore for point: %d, %f",i,scoreArr[i]);


		}	

		//final score:
		double sum=0;
		for (int i=0; i<sizeData;i++)
		{
			sum+=scoreArr[i];

		}

		printf("\nFinal Error metric: %f",(1.0*sum)/(sizeData*1.0));
	
	

	}//end of method



