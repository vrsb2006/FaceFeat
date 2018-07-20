/* MIT License

Copyright (c) 2018 Vijay Bhardwaj

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

#pragma once
// opencv include directories
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>

// other useful include directories
#include<fstream>
#include<tuple>
#include<Windows.h>

using namespace cv;
using namespace std;

//#define write_video

extern int num_used_pts;
extern float err;

// Get camera matrix
cv::Mat get_camera_matrix(float focal_length, cv::Point2d center);

// Get 3d model points
std::vector<cv::Point3d> get_3D_model_points(void);

// Calculating HOG features for given shape
void featHOG(cv::Mat &img_inp, cv::Mat &shp, cv::Mat &out, int winsize);

// Reset a shape onto bounding box
cv::Mat resetshape(const cv::Mat &shp, cv::Rect box);

// Bounding box from a given shape
cv::Rect get_bbox(cv::Mat &shp);

// Read model files
void readModel(std::istream &i, Mat &m);
 
// Results on live video (or camera)
void live();

// Result on images
void test_images(std::string path);
