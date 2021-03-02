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

#include "structs.h"
#include <math.h>

int bin_x(double x);
int bin_y(double x);

//going to sort the data.
//however, the points are sorted based on a binned granularity of 1 degree (can modify this later) 
//this is so that for a given x value, the y values will be ordered together

//compare function for std sort for the vector
//Sorts in the order: x, y, TEC, time
bool compareDataElemStructFunc(const dataElem &elem1, const dataElem &elem2)
{
   //compare based on the x-coodinate
   if ( bin_x(elem1.x) < bin_x(elem2.x))
      {
      return true;
      }
   else if (bin_x(elem1.x) > bin_x(elem2.x))
      {
      return false;
     }
      //if the x-coordinates are equal, compare on the y-coordinate
   else if ( bin_y(elem1.y) < bin_y(elem2.y))
         {
      return true;
         }
   else if (bin_y(elem1.y) > bin_y(elem2.y))
         {
      return false;
         }
         else{
      return false;
         }
}


//calculate the bin for a point, that's in the range of x:0-180 (latitude) 
//1 degree bins
int bin_x(double x)
{
int num_bins=180;

//set x to be a positive value (add 250)
double total_width=180;
return (ceil((x/total_width)*num_bins));
}

//calculate the bin for a point, that's in the range of y:0-360 (longitude)
//1 degree bins
int bin_y(double x)
{
int num_bins=360;

//set x to be a positive value (add 250)
double total_width=360;
return (ceil((x/total_width)*num_bins));
}