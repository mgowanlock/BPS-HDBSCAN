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

#include <set>
#include <vector>
#include <stdio.h>
#include "structs.h"
#include "cluster.h"
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "RTree.h"
#include "structsDBSCAN.h"
#include "DBScan.h"
#include <iterator>



//DBSCAN that uses the neighbortable from the Hybrid DBSCAN Paper
//DBSCAN the points in the neighbortable that could not be added to any other cluster
//queryVect- the points that need to be DBSCANNED
//clusterOffset- the clusterId to begin enumerating at (not equal to the total number of clusters already clustered)
void dbscanTableLeftovers(struct neighborTableLookup * neighborTable, std::vector<unsigned int> * queryVect, std::vector<int> *clusterIDs, unsigned int clusterOffset, int minPts, int sizeDB)
{	

	//need to start the cluster label id beyond the ids used for densebox
	unsigned int clusterCnt=clusterOffset;

	/*
	int cntnoise=0;
	int cntunassigned=0;
	
	for (int i=0; i<queryVect->size(); i++)
	{
		int idx=(*queryVect)[i];
		if ((*clusterIDs)[idx]==0)
		cntnoise++;

		if ((*clusterIDs)[idx]==-1)
			cntunassigned++;

	}


		printf("\n[BEGINNING OF LEFTOVERS] Num noise (check): %d",cntnoise);
		printf("\n[BEGINNING OF LEFTOVERS] Num unassigned (check): %d",cntunassigned);

	*/

	
	int cntclustersdbscanleftovers=0;

	//neighborSet is the current set of points being searched that belongs to a cluster
	std::vector<int>neighborSet;

	//vector that keeps track of the points that have beeen visited
	//initialize all points to not being visited
	std::vector<bool>visited(clusterIDs->size(),false);



	//vector that keeps track of the assignment of the points to a cluster
 	//cluster 0 means a noise point

	// std::vector<int>clusterIDs(queryVect->size(),0); //initialize all points to be in cluster 0
	


	for (int i=0; i<queryVect->size(); i++){
			
		unsigned int idx=(*queryVect)[i];	

		//see if the point has been visited, if so, go onto the next loop iteration
		if (visited[idx]==true){
			continue;
		}

		//clear the vector of neighbors
		neighborSet.clear();

		//mark the point as visited:
		visited[idx]=true;
		
		
		
		//get the neighbors of the data point
		//dataElem tmpDataPoint=(*dataPoints)[i];
		//getNeighbours(&tmpDataPoint,distance, &setIDsInDist);

		
		//if the number of neighbors is less than the number required for a cluster,
		//then it is noise.  The noise will be cluster 0.
		if (((neighborTable[idx].indexmax-neighborTable[idx].indexmin)+1)<minPts)
		{
			(*clusterIDs)[idx]=0;
		}
				
		
		//if there are enough points to make a cluster
		
		else
		{
				
			clusterCnt++;
			// printf("\ncluster cnt: %d",clusterCnt);
			cntclustersdbscanleftovers++;
			//make a new cluster with the correct cluster ID 
			(*clusterIDs)[idx]=clusterCnt;	


			//printf("\n***1size of neighbor set: %d", neighborSet.size());

			//assign the neighbor ids to the neighborSet, which may be expanded when searching
			//through the neighbors
			unsigned int sizeInsert=neighborTable[idx].indexmax-neighborTable[idx].indexmin+1;
			neighborSet.resize(sizeInsert);
			// neighborSet.insert(neighborSet.end(),&neighborTable[idx].dataPtr[neighborTable[idx].indexmin], &neighborTable[idx].dataPtr[neighborTable[idx].indexmin]+(sizeInsert));

			// #pragma omp parallel for num_threads(2)
			for (unsigned int j=neighborTable[idx].indexmin; j<=neighborTable[idx].indexmax; j++)
			{
				// unsigned int idx_neighborset=j-neighborTable[idx].indexmin;
				neighborSet[j-neighborTable[idx].indexmin]=neighborTable[idx].dataPtr[j];
			}
			
			

			
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

				//if the number of neighbors is greater than the number required to form a cluster
				
				unsigned long int sizeInsert=(neighborTable[pntID].indexmax-neighborTable[pntID].indexmin)+1;
				if (sizeInsert>=minPts)
				{
					unsigned long int sizeNeighborTable=neighborSet.size();
					neighborSet.resize(sizeNeighborTable+sizeInsert);
					
					// #pragma omp parallel for num_threads(2)	
					for (int j=neighborTable[pntID].indexmin; j<=neighborTable[pntID].indexmax;j++)
					{
						neighborSet[sizeNeighborTable+(j-neighborTable[pntID].indexmin)]=neighborTable[pntID].dataPtr[j];
					}	
				}


				//if the point has part not been assigned to a cluster yet
				if ((*clusterIDs)[pntID]==-1){
					(*clusterIDs)[pntID]=clusterCnt;							
				}
				
				


			 } //end of while loop

		} //end of else
		

					
		//testTotalNeighbors+=neighbourList.size();

		//now have the vector of ids within the distance
		//setIDsInDist
		
		

	} //end of main for loop

	//increment the total cluster count by 1 because cluster 0 is for the noise data points
	clusterCnt++;

	


	// printf("\n***printing cluster array from GPU version with neighbor table:");
	// for (int i=0; i<clusterIDs.size(); i++)
	// {

	// 	printf("\n%d, %d",i,clusterIDs[i]);
	// }

	// printf("\n***end of printing cluster array:");



	printf("\nDBSCAN leftovers: total clusters %d", cntclustersdbscanleftovers);
	

	/*
	cntnoise=0;
	cntunassigned=0;
	for (int i=0; i<queryVect->size(); i++)
	{
		int idx=(*queryVect)[i];
		if ((*clusterIDs)[idx]==0)
		cntnoise++;

		if ((*clusterIDs)[idx]==-1)
			cntunassigned++;

	}

		printf("\n[END OF LEFTOVERS] Num noise (check): %d",cntnoise);
		printf("\n[END OF LEFTOVERS] Num unassigned (check): %d",cntunassigned);

	*/



}











//DBSCAN that uses the neighbortable from the Hybrid DBSCAN Paper
//DBSCAN the points in the neighbortable that could not be added to any other cluster
//queryVect- the points that need to be DBSCANNED
//clusterOffset- the clusterId to begin enumerating at (not equal to the total number of clusters already clustered)
void dbscanTableLeftoversOriginal(struct neighborTableLookup * neighborTable, std::vector<unsigned int> * queryVect, std::vector<int> *clusterIDs, unsigned int clusterOffset, int minPts, int sizeDB)
{	

	//need to start the cluster label id beyond the ids used for densebox
	unsigned int clusterCnt=clusterOffset;

	int cntnoise=0;
	int cntunassigned=0;
	for (int i=0; i<queryVect->size(); i++)
	{
		int idx=(*queryVect)[i];
		if ((*clusterIDs)[idx]==0)
		cntnoise++;

		if ((*clusterIDs)[idx]==-1)
			cntunassigned++;

	}

		printf("\n[BEGINNING OF LEFTOVERS] Num noise (check): %d",cntnoise);
		printf("\n[BEGINNING OF LEFTOVERS] Num unassigned (check): %d",cntunassigned);



	
	int cntclustersdbscanleftovers=0;

	//neighborSet is the current set of points being searched that belongs to a cluster
	std::vector<int>neighborSet;

	//vector that keeps track of the points that have beeen visited
	//initialize all points to not being visited
	std::vector<bool>visited(clusterIDs->size(),false);



	//vector that keeps track of the assignment of the points to a cluster
 	//cluster 0 means a noise point

	// std::vector<int>clusterIDs(queryVect->size(),0); //initialize all points to be in cluster 0
	


	for (int i=0; i<queryVect->size(); i++){
			
		unsigned int idx=(*queryVect)[i];	

		//see if the point has been visited, if so, go onto the next loop iteration
		if (visited[idx]==true){
			continue;
		}

		//clear the vector of neighbors
		neighborSet.clear();

		//mark the point as visited:
		visited[idx]=true;
		
		
		
		//get the neighbors of the data point
		//dataElem tmpDataPoint=(*dataPoints)[i];
		//getNeighbours(&tmpDataPoint,distance, &setIDsInDist);

		
		//if the number of neighbors is less than the number required for a cluster,
		//then it is noise.  The noise will be cluster 0.

		//This should be impossible, because we've already checked that each point has at least minpts points
		if (((neighborTable[idx].indexmax-neighborTable[idx].indexmin)+1)<minPts)
		{
			(*clusterIDs)[idx]=0;
		}
				
		
		//if there are enough points to make a cluster
		
		else
		{
				
			clusterCnt++;
			// printf("\ncluster cnt: %d",clusterCnt);
			cntclustersdbscanleftovers++;
			//make a new cluster with the correct cluster ID 
			(*clusterIDs)[idx]=clusterCnt;	


			//printf("\n***1size of neighbor set: %d", neighborSet.size());

			//assign the neighbor ids to the neighborSet, which may be expanded when searching
			//through the neighbors
			int sizeInsert=neighborTable[idx].indexmax-neighborTable[idx].indexmin+1;
			neighborSet.insert(neighborSet.end(),&neighborTable[idx].dataPtr[neighborTable[idx].indexmin], &neighborTable[idx].dataPtr[neighborTable[idx].indexmin]+(sizeInsert));

			//printf("\n***size of neighbor set: %d", neighborSet.size());
			

			
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
				//dataElem tmpDataPoint=(*dataPoints)[pntID];

				//getNeighbours(&tmpDataPoint,distance, &setIDsInDist);
				//getNeighboursBruteForce(&tmpDataPoint,distance);

				//if the number of neighbors is greater than the number required to form a cluster
				
				
				if (((neighborTable[pntID].indexmax-neighborTable[pntID].indexmin)+1)>=minPts)
				{
					//assign the neighbor ids to the neighborSet
					//copyVect(&neighborSet,&setIDsInDist);	
					
					//XXXXX
					//CHANGE THIS LATER TO INSERTS
					//XXXXX
					for (int j=neighborTable[pntID].indexmin; j<=neighborTable[pntID].indexmax;j++)
					{
						neighborSet.push_back(neighborTable[pntID].dataPtr[j]);
					}	
				}
				//if the point has part not been assigned to a cluster yet
				
				
				if ((*clusterIDs)[pntID]==-1){
					(*clusterIDs)[pntID]=clusterCnt;							
				}
				
				


			 } //end of while loop

		} //end of else
		

					
		//testTotalNeighbors+=neighbourList.size();

		//now have the vector of ids within the distance
		//setIDsInDist
		
		

	} //end of main for loop

	//increment the total cluster count by 1 because cluster 0 is for the noise data points
	clusterCnt++;

	


	// printf("\n***printing cluster array from GPU version with neighbor table:");
	// for (int i=0; i<clusterIDs.size(); i++)
	// {

	// 	printf("\n%d, %d",i,clusterIDs[i]);
	// }

	// printf("\n***end of printing cluster array:");



	printf("\nDBSCAN leftovers: total clusters %d", cntclustersdbscanleftovers);
	


	cntnoise=0;
	cntunassigned=0;
	for (int i=0; i<queryVect->size(); i++)
	{
		int idx=(*queryVect)[i];
		if ((*clusterIDs)[idx]==0)
		cntnoise++;

		if ((*clusterIDs)[idx]==-1)
			cntunassigned++;

	}

		printf("\n[END OF LEFTOVERS] Num noise (check): %d",cntnoise);
		printf("\n[END OF LEFTOVERS] Num unassigned (check): %d",cntunassigned);

	



}








//From variantdbscan code

//"OLD" before we store the results and then look them up later to skip computing the scores

void DetermineErrorTwoClusterResultsBeforeShortCircuit(std::vector<int> * c1, std::vector<int> * c2)
	{

		// printf("\ntesting importing the cluster ids from other implementation instead of the R-tree");

		// c2->clear();
		// char fname[400]="/home/mgowanlock/VDBSCAN_No_Wrap_Around_Or_Indirection_Clean/output_clusters_sequential_VDBSCAN.txt";
		// importClusterIDsFromFile(c2,fname);



			
		printf("\nIn method to compare two clusters using the similarity metric");

		printf("\nsize of array of datapoints1: %zu, size of array of datapoints2: %zu",c1->size(),c2->size());
	
		//the sizes must be equal, or the metric will not work. Thesize should be the number of datapoints in the dataset
		if (c1->size()!=c2->size())
		{
			printf("\n**********\nERROR WHEN TESTING THE SIMILARITY/ERROR OF TWO CLUSTERING RESULTS. THE NUMBER OF POINTS IN EACH ARRAY ARE NOT EQUAL\n\n");
			printf("\nSize clusters1: %lu, size clusters2: %lu",c1->size(),c2->size());
			return;
		}




		std::vector<int> denseBoxClusters(c2->begin(), c2->end());
		
		

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
		
		
		//For the densebox clusters, need to find 
		//1)the number of clusters
		//2)remap the cluster ids to 0,1,2,... for doing the validation

		std::set<int> setOfClusterIds;

		for (int i=0; i<denseBoxClusters.size(); i++)
		{
		setOfClusterIds.insert(denseBoxClusters[i]);
		}

		int max2=setOfClusterIds.size();
		printf("\nSize of densebox unique clusters: %lu",setOfClusterIds.size());

		std::vector<int> setOfClusterIdsVect;
		
		std::copy(setOfClusterIds.begin(), setOfClusterIds.end(),std::back_inserter(setOfClusterIdsVect));

		// printf("\nCluster ids densebox: \n");
		unsigned int cntunassigned=0;
		for (int i=0; i<denseBoxClusters.size();i++)
		{
			if (denseBoxClusters[i]==-1)
			{
			// printf("i: %d, %d\n",i,setOfClusterIdsVect[i]);
			cntunassigned++;
			}
		}

		printf("\ndensebox unassigned: %u",cntunassigned);

		// printf("\nData points are unassigned\nThe problem then is that the mapping is off from the original cluster ids to the remapping of cluster ids, since cluster -1 is counted!\n!");

		//remap so clusters go from 0, 1, 2, 3,...
		for (int i=0; i<denseBoxClusters.size(); i++)
		{
			int clusterid=denseBoxClusters[i];
			auto it = std::lower_bound(setOfClusterIdsVect.begin(), setOfClusterIdsVect.end(), clusterid);
			if(!(it == setOfClusterIdsVect.end() || *it != clusterid))
			{

				uint64_t ind = std::distance(setOfClusterIdsVect.begin(), it);	    		
				// printf("\nMapping cluster: %d to %lu",denseBoxClusters[i],ind);
				denseBoxClusters[i]=ind;

			}	
		}
		
		
		
		//want to include the final cluster
		max1++;
		max2++;

		printf("\nnum clusters in list1 (R-tree), list2 (densebox): %d,%d",max1,max2);
		
		
		//create 2 arrays of the clusters and the points assigned to each
		//the number of clusters may not be equal, but the total number of points is equal
		std::vector<int > clusterArr1[max1];
		std::vector<int > clusterArr2[max2];
		

		for (int i=0; i<sizeData;i++)
		{
			int clusterid1=(*c1)[i];
			int clusterid2=denseBoxClusters[i];
		
			
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
			
			if (((*c1)[i]==0)&&(denseBoxClusters[i]!=0))
			{
				cntNoiseError++;
				scoreArr[i]=0;
				visitedArr[i]=true;
				cntNoiseC1++;



			}

			if (((*c1)[i]!=0)&&(denseBoxClusters[i]==0))
			{
				cntNoiseError++;
				scoreArr[i]=0;	
				visitedArr[i]=true;
				cntNoiseC2++;
			}

			//both are noise points:
			if (((*c1)[i]==0)&&(denseBoxClusters[i]==0))
			{
				cntNoiseEqual++;
				scoreArr[i]=1;
				visitedArr[i]=true;
				cntNoiseC1++;
				cntNoiseC2++;
			}
		}
		

		

		printf("\nNum noise points: C1: %d, densebox: %d", cntNoiseC1,cntNoiseC2);
		printf("\nmismatched noise points: %d, agreement noise points: %d", cntNoiseError, cntNoiseEqual);
		
		

		// printf("\n Returning early");
		// return;

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
		
		omp_set_num_threads(32);
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
			int clusterid2=denseBoxClusters[i];

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
			#pragma omp critical
			{
				if (scoreArr[i]!=1.0){
				totalScoreMiscluster+=1.0-scoreArr[i];
				}
			}
			


		}	

		printf("\nfraction lost due to mismatches between clusters: %f", totalScoreMiscluster/(sizeData*1.0));

		//final score:
		double sum=0;
		for (int i=0; i<sizeData;i++)
		{
			sum+=scoreArr[i];

		}

		printf("\nFinal Error metric: %f",(1.0*sum)/(sizeData*1.0));


		// for (int i=0; i<sizeData;i++)
		// {
		// 	printf("\nPoint: %d, cluster id [densebox]: %d, cluster id (R-tree): %d, score: %f",i,denseBoxClusters[i], (*c1)[i], scoreArr[i]);
		// }
	
	

	}


	//From variantdbscan code
	//modifying to store scores when the score has been computed already
	//can skip the insersection/union test when it has already been computed
void DetermineErrorTwoClusterResults(std::vector<int> * c1, std::vector<int> * c2)
	{

		//use this to short circuit computing the score if it has already been computed (since it's very expensive to compute)
		struct scoreLookup{
			int clusterA;
			int clusterB;
			double score;
		};

		std::vector<struct scoreLookup> scoreLookupVect;


		// printf("\ntesting importing the cluster ids from other implementation instead of the R-tree");

		// c2->clear();
		// char fname[400]="/home/mgowanlock/VDBSCAN_No_Wrap_Around_Or_Indirection_Clean/output_clusters_sequential_VDBSCAN.txt";
		// importClusterIDsFromFile(c2,fname);



			
		printf("\nIn method to compare two clusters using the similarity metric");

		printf("\nsize of array of datapoints1: %zu, size of array of datapoints2: %zu",c1->size(),c2->size());
	
		//the sizes must be equal, or the metric will not work. Thesize should be the number of datapoints in the dataset
		if (c1->size()!=c2->size())
		{
			printf("\n**********\nERROR WHEN TESTING THE SIMILARITY/ERROR OF TWO CLUSTERING RESULTS. THE NUMBER OF POINTS IN EACH ARRAY ARE NOT EQUAL\n\n");
			printf("\nSize clusters1: %lu, size clusters2: %lu",c1->size(),c2->size());
			return;
		}




		std::vector<int> denseBoxClusters(c2->begin(), c2->end());
		
		

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
		
		
		//For the densebox clusters, need to find 
		//1)the number of clusters
		//2)remap the cluster ids to 1,2,... for doing the validation


		std::set<int> setOfClusterIds;
		//Need to add the noise cluster to the set in the event that there 
		//are no noise points, that the mapping does not map a real cluster to 0
		setOfClusterIds.insert(0);
		for (int i=0; i<denseBoxClusters.size(); i++)
		{
			setOfClusterIds.insert(denseBoxClusters[i]);
		}

		int max2=setOfClusterIds.size();
		printf("\nSize of densebox unique clusters: %lu",setOfClusterIds.size());

		std::vector<int> setOfClusterIdsVect;
		
		std::copy(setOfClusterIds.begin(), setOfClusterIds.end(),std::back_inserter(setOfClusterIdsVect));

		// printf("\nCluster ids densebox: \n");
		unsigned int cntunassigned=0;
		for (int i=0; i<denseBoxClusters.size();i++)
		{
			if (denseBoxClusters[i]==-1)
			{
			// printf("i: %d, %d\n",i,setOfClusterIdsVect[i]);
			cntunassigned++;
			}
		}

		printf("\ndensebox unassigned: %u",cntunassigned);

		

		// printf("\nData points are unassigned\nThe problem then is that the mapping is off from the original cluster ids to the remapping of cluster ids, since cluster -1 is counted!\n!");

		//remap so clusters go from 1, 2, 3,...
		for (int i=0; i<denseBoxClusters.size(); i++)
		{
			int clusterid=denseBoxClusters[i];
			auto it = std::lower_bound(setOfClusterIdsVect.begin(), setOfClusterIdsVect.end(), clusterid);
			if(!(it == setOfClusterIdsVect.end() || *it != clusterid))
			{

				uint64_t ind = std::distance(setOfClusterIdsVect.begin(), it);	    		
				// printf("\nMapping cluster: %d to %lu",denseBoxClusters[i],ind);
				denseBoxClusters[i]=ind; //add the 1 offset so we don't assign to noise?

			}	
		}
		
		
		//want to include the final cluster
		max1++;
		max2++;

		printf("\nnum clusters in list1 (R-tree), list2 (densebox): %d,%d",max1,max2);
		
		


		//create 2 arrays of the clusters and the points assigned to each
		//the number of clusters may not be equal, but the total number of points is equal
		std::vector<std::vector <int> > clusterArr1(max1);
		std::vector<std::vector <int> > clusterArr2(max2);
		



		for (int i=0; i<sizeData;i++)
		{
			int clusterid1=(*c1)[i];
			int clusterid2=denseBoxClusters[i];
		
			
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
			
			if (((*c1)[i]==0)&&(denseBoxClusters[i]!=0))
			{
				cntNoiseError++;
				scoreArr[i]=0;
				visitedArr[i]=true;
				cntNoiseC1++;



			}

			if (((*c1)[i]!=0)&&(denseBoxClusters[i]==0))
			{
				cntNoiseError++;
				scoreArr[i]=0;	
				visitedArr[i]=true;
				cntNoiseC2++;
			}

			//both are noise points:
			if (((*c1)[i]==0)&&(denseBoxClusters[i]==0))
			{
				cntNoiseEqual++;
				scoreArr[i]=1;
				visitedArr[i]=true;
				cntNoiseC1++;
				cntNoiseC2++;
			}
		}
		

		

		printf("\nNum noise points: C1: %d, densebox: %d", cntNoiseC1,cntNoiseC2);
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
		unsigned int cntShortCircuit=0;
		//upon entering the loop
		


		omp_set_num_threads(32);
		printf("\nRUNNING THE VALIDATION IN PARALLEL WITH THREADS!!");
		#pragma omp parallel for reduction(+:cntShortCircuit) schedule(dynamic) shared(scoreArr, visitedArr, scoreLookupVect)
		for (int i=0; i<sizeData;i++)
		{
			//if point was already noise, we dealt with the point already
			if (visitedArr[i]==true)
			{
				continue;
			}




			//get the two cluster ids for the point from the two experiments
			int clusterid1=(*c1)[i];
			int clusterid2=denseBoxClusters[i];

			bool shortcircuitflag=0;

			#pragma omp critical
			{
				//see if we can short circuit the calculation if the score has already been computed for the two clusters
				//just scan because the maximum size would be the number of data points
				for (int j=0; j<scoreLookupVect.size(); j++)
				{
					//test both combinations
					if (clusterid1==scoreLookupVect[j].clusterA && clusterid2==scoreLookupVect[j].clusterB)
					{
						scoreArr[i]=scoreLookupVect[j].score;
						cntShortCircuit++;
						shortcircuitflag=1;
						break;
					}
					else if (clusterid1==scoreLookupVect[j].clusterB && clusterid2==scoreLookupVect[j].clusterA)
					{
						scoreArr[i]=scoreLookupVect[j].score;
						cntShortCircuit++;
						shortcircuitflag=1;
						break;
						
					}
				}
			}

			if (shortcircuitflag==1)
			{
				continue;
			}


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


			#pragma omp critical
			{
				scoreLookup tmpScore;
				tmpScore.clusterA=clusterid1; 
				tmpScore.clusterB=clusterid2; 
				tmpScore.score=scoreArr[i];
				scoreLookupVect.push_back(tmpScore);
			}


			//for testing:
			#pragma omp critical
			{
				if (scoreArr[i]!=1.0){
				totalScoreMiscluster+=1.0-scoreArr[i];
				}
			}
			


		}	

		printf("\nfraction lost due to mismatches between clusters: %f", totalScoreMiscluster/(sizeData*1.0));

		//final score:
		double sum=0;
		for (int i=0; i<sizeData;i++)
		{
			sum+=scoreArr[i];

		}

		printf("\nFinal Error metric: %f",(1.0*sum)/(sizeData*1.0));

		printf("\nNumber of calculations (intersection/union tests) skipped: %u of %u (%f)", cntShortCircuit, sizeData, (1.0*cntShortCircuit)/(1.0*sizeData));


		// for (int i=0; i<sizeData;i++)
		// {
		// 	printf("\nPoint: %d, cluster id [densebox]: %d, cluster id (R-tree): %d, score: %f",i,denseBoxClusters[i], (*c1)[i], scoreArr[i]);
		// }
	
	

	}


void importDatasetDBSCAN(std::vector<dataElem> *dataPoints, std::vector<std::vector <DTYPE> > *NDdataPoints)
{

	struct dataElem tmpStruct;


	for (int i=0; i<(*NDdataPoints)[0].size(); i++)
	{
		tmpStruct.x=(*NDdataPoints)[0][i];
		tmpStruct.y=(*NDdataPoints)[1][i];
		dataPoints->push_back(tmpStruct);
	}




}

void importClusterIDsFromFile(std::vector<int> *clusterIDs, char * fname)
{

	FILE *fileInput = fopen(fname,"r");

		char in_line[400];
		char data[10];

	while(fgets(in_line, 400, fileInput)!=NULL)
	{
	//fgets(in_line, 400, fileInput);

	sscanf(in_line,"%[^,]",data);

	clusterIDs->push_back(atof(data));

	}


}

//create MBBs for R-tree
void createEntryMBBs(std::vector<dataElem> *dataPoints, Rect * dataRects)
{
	for (int i=0; i<(*dataPoints).size(); i++){
		dataRects[i].P1[0]=(*dataPoints)[i].x;
		dataRects[i].P1[1]=(*dataPoints)[i].y;
		// dataRects[i].P1[2]=(*dataPoints)[i].val;
		// dataRects[i].P1[3]=(*dataPoints)[i].time;
		dataRects[i].pid=i;
		dataRects[i].CreateMBB();
	}

}	




void compareClusteringOutput(double epsilon, unsigned int minPts, std::vector<int> * clusterIDs, std::vector<std::vector <DTYPE> > *NDdataPoints)
{

	
	printf("\n\n********************\nGOING TO RUN SEQUENTIAL ALG. TO MEASURE SIMILARITY BETWEEN CLUSTER RESULTS\n\n");

	std::vector<struct dataElem> dataPoints;
	importDatasetDBSCAN(&dataPoints, NDdataPoints);

	RTree<int,DTYPE,2,float> tree;
	

	//Create MBBs
	//struct to store MBB data
	//this is for 1 point per MBB
	


	// printf("\nSize of data rects: %f (GiB)", ((sizeof(Rect)*dataPoints.size()*1.0)/(1024*1024*1024.0)));cout.flush();

	// printf("\n***********\nProblem is that we run out of memory when building the tree. Maybe insert one MBB at a time?\n******\n");

	//original - create all MBBs
	/*
	Rect * dataRectsSequential;
	dataRectsSequential= new Rect[dataPoints.size()];
	createEntryMBBs(&dataPoints, dataRectsSequential);
	//insert data into the R-tree
	for (int i=0; i<dataPoints.size(); i++){
		//Insert:
		tree.Insert(dataRectsSequential[i].MBB_min,dataRectsSequential[i].MBB_max, dataRectsSequential[i].pid);	
	}
	*/

	//NEW
	//Create one MBB and then insert (don't precompute all MBBs)
	//insert data into the R-tree
	for (int i=0; i<dataPoints.size(); i++){
		//Insert:
		Rect dataRectsSequential;
		dataRectsSequential.P1[0]=dataPoints[i].x;
		dataRectsSequential.P1[1]=dataPoints[i].y;
		dataRectsSequential.pid=i;
		dataRectsSequential.CreateMBB();
		tree.Insert(dataRectsSequential.MBB_min,dataRectsSequential.MBB_max, dataRectsSequential.pid);	
	}
	
	
		
	
		
	DBScan * clusterSequential=new DBScan(&dataPoints, epsilon, minPts, &tree);

	printf("\nBefore calling DBScan sequential FOR VALIDATION!!");cout.flush();
		
	
	//run experiments:
	
	double tstartseq=omp_get_wtime();
	
		clusterSequential->algDBScan();
	
	double tendseq=omp_get_wtime();
	printf("\nTime for sequential implementation (just for validation part, not for actual timing): %f",tendseq-tstartseq);cout.flush();
	

	
	// return 0;	

	//validation
	//compare the clusters produced by the cluster reuse method vs the sequential implementation
	
	// printf("\nReturning before comparing output");
	// return;

	double tstartvalidation=omp_get_wtime();

	DetermineErrorTwoClusterResults(&clusterSequential->clusterIDs,clusterIDs);	

	// DetermineErrorTwoClusterResultsBeforeShortCircuit(&clusterSequential->clusterIDs,clusterIDs);	

	double tendvalidation=omp_get_wtime();

	printf("\nTime to validate results: %f",tendvalidation - tstartvalidation);


	delete clusterSequential;
	
	// delete [] dataRectsSequential;
	
	
}


