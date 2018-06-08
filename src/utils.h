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