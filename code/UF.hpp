// The MIT License (MIT)

// Copyright (c) [2015] [Kartik Kukreja]

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

//from https://github.com/kartikkukreja/blog-codes/blob/master/src/Union%20Find%20(Disjoint%20Set)%20Data%20Structure.cpp

//Modified by Mike Gowanlock

class UF    {
    public:
    int *id, cnt, *sz;

	// Create an empty union find data structure with N isolated sets.
    UF(int N)   {
        cnt = N;
	id = new int[N];
	sz = new int[N];
        for(int i=0; i<N; i++)	{
            id[i] = i;
	    sz[i] = 1;
	}
    }
    ~UF()	{
	delete [] id;
	delete [] sz;
    }
	// Return the id of component corresponding to object p.
    int find(int p)	{
        int root = p;
        while (root != id[root])
            root = id[root];
        while (p != root) {
            int newp = id[p];
            id[p] = root;
            p = newp;
        }
        return root;
    }
	// Replace sets containing x and y with their union.
    void merge(int x, int y)	{
        int i = find(x);
        int j = find(y);
        if (i == j) return;
		
		// make smaller root point to larger one
        if   (sz[i] < sz[j])	{ 
		id[i] = j; 
		sz[j] += sz[i]; 
	} else	{ 
		id[j] = i; 
		sz[i] += sz[j]; 
	}
        cnt--;
    }
	// Are objects x and y in the same set?
    bool connected(int x, int y)    {
        return find(x) == find(y);
    }
	// Return the number of disjoint sets.
    int count() {
        return cnt;
    }

    //MG: output all of the disjoint sets
    //step1: make sure they are connected by scanning over all values
    //step2: given the size of the disjoint sets (cnt), copy the values with the index as appropriate
    void assembleSetList(std::vector<std::vector<unsigned int> >* inputVect, unsigned int N)
    {

    //first, scan to connect all of the components that have not yet been connected (since it occurs on the fly)
        //initialize first value of each vector to N (so that we know these sets have nothing in them)
        //N-nothing in this set

    for (unsigned int i=0; i<N; i++){
    //scan to connect
    connected(i,0);
    
    //initialize
    std::vector<unsigned int> tmp;
    tmp.push_back(N);
    inputVect->push_back(tmp);
    }

    //scan the list and assign the ids to the master
    for (unsigned int i=0; i<N; i++){
        //if the vector has just been initialized only (all values are initialized to N), then we clear the array so that we know this array has 
        //dense boxes assigned to it
        
        int representative=id[i];
        if ((*inputVect)[representative].size()==1 &&(*inputVect)[representative][0]==N)  
        {
         (*inputVect)[representative].clear();   
        }
        
        (*inputVect)[representative].push_back(i);
    }

    } 

};