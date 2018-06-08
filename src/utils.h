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
#include<omp.h>

// value of pi and epsilon
#define PI 3.14159265
//#define write_video

using namespace cv;
using namespace std;

extern int num_used_pts;
extern float err;

struct img_data {
	cv::Mat img;
	cv::Rect bbox;
	cv::Mat shape, shape_org;
	std::vector<cv::Mat> init_shape;
	std::vector<cv::Mat> shape_diff;
};

extern std::vector<img_data> Tr_Data, Data;  

cv::Mat ProjectShape(const cv::Mat &shape, const cv::Rect &bounding_box);

cv::Mat ReProjectShape(const cv::Mat &shape, const cv::Rect &bounding_box);

void featHOG(cv::Mat &img_inp, cv::Mat &shp, cv::Mat &out, int winsize, cv::Rect bbox);

cv::Mat resetshape(const cv::Mat &shp, cv::Rect box);

cv::Rect get_bbox(cv::Mat &shp);

Mat correctGamma(Mat& img, float gamma);

void live();

void test_images(std::string path);
