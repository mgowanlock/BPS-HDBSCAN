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

#include <stdio.h>
#include <vector>
#include <fstream>
#include <istream>
#include <iostream>
#include <string>
#include <string.h>
#include <sstream>
#include <cstdlib>
// #include "prototypes.h"
#include <algorithm>
#include <omp.h>
// #include "globals.h"

#include "params.h"

bool sortNDComp(const std::vector<DTYPE>& a, const std::vector<DTYPE>& b)
{
    for (int i=0; i<GPUNUMDIM; i++){
      if (int(a[i])<int(b[i])){
      return 1;
      }
      else if(int(a[i])>int(b[i])){
      return 0;
      }  
    }

    return 0;

}



void importNDDataset(std::vector<std::vector <DTYPE> > *dataPoints, char * fname)
{


  double tstart=omp_get_wtime();

	std::vector<DTYPE>tmpAllData;
	std::ifstream in(fname);
	int cnttmp=0;
  
	for (std::string f; getline(in, f, ',');){
	DTYPE i;
		 std::stringstream ss(f);
	    while (ss >> i)
	    {
	        tmpAllData.push_back(i);
	        if (ss.peek() == ',')
	            ss.ignore();
	    }
  		
  }	

    in.close();

    double tend=omp_get_wtime();
    printf("\nTime to import data: %f, Size (GiB): %f, MiB/s: %f", tend - tstart, (tmpAllData.size()*sizeof(DTYPE))/(1024*1024*1024.0), (tmpAllData.size()*sizeof(DTYPE)/(1024*1024.0))/(tend-tstart));


  	unsigned int cnt=0;
  	const unsigned int totalPoints=(unsigned int)tmpAllData.size()/GPUNUMDIM;
  	printf("\nData import: Total size of all data (1-D) vect (number of points * GPUNUMDIM): %zu",tmpAllData.size());
  	printf("\nData import: Total data points: %d",totalPoints);
  	

    for (int i=0; i<GPUNUMDIM; i++)
    {
      (*dataPoints)[i].resize(totalPoints);
    }

  	for (int i=0; i<totalPoints; i++){
  		for (int j=0; j<GPUNUMDIM; j++){
  			(*dataPoints)[j][i]=tmpAllData[cnt];
  			cnt++;
  		}
  	}


}


void sortInNDBins(std::vector<std::vector <DTYPE> > *dataPoints){
  
  std::sort(dataPoints->begin(),dataPoints->end(),sortNDComp);
  
}


//used so we don't need to import the data every time for testing
void generateSyntheticData(std::vector<std::vector <DTYPE> > *dataPoints, unsigned int datasetsize)
{

  double tstart=omp_get_wtime();
  /*
  //seed RNG
  srand(12347);

  dataPoints->resize(datasetsize);

  std::vector <DTYPE> tmp;

  #pragma omp parallel for private(tmp) num_threads(NTHREADS) shared(dataPoints,datasetsize)
  for (unsigned int i=0; i<datasetsize; i++)
  {

    tmp.clear();
    //x between [0,360]
    tmp.push_back(360.0*((double)(rand()) / RAND_MAX));
    //y between [-90,90]
    tmp.push_back((180.0*((double)(rand()) / RAND_MAX))-90);
  
    (*dataPoints)[i]=tmp;
  
  }

  */


  //seed RNG
  srand(12347);

  DTYPE * X=new DTYPE[datasetsize];
  DTYPE * Y=new DTYPE[datasetsize];

  std::vector <DTYPE> tmp;

  #pragma omp parallel for private(tmp) num_threads(NTHREADS) shared(X,Y,datasetsize)
  for (unsigned int i=0; i<datasetsize; i++)
  {

    
    //x between [0,360]
    X[i]=(360.0*((double)(rand()) / RAND_MAX));
    //y between [-90,90]
    Y[i]=((180.0*((double)(rand()) / RAND_MAX))-90);
  
    
  
  }


  double tend=omp_get_wtime();
  printf("\nTime to generate synthetic data: %f", tend-tstart);fflush(stdout);

}



