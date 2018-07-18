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

#include "utils.h"

String face_cascade_name = "haarcascade_frontalface_alt2.xml";
cv::CascadeClassifier face_cascade;
cv::Mat data_mean_shape;

int num_used_pts = 68;
float err = 0;

std::vector<img_data> Tr_Data, Data;

void featHOG(cv::Mat &img_inp, cv::Mat &shp, cv::Mat &out, int winsize) {

	// Add border for descritpor calculation at borders
	int border = (int)(winsize / 2);
	cv::Mat img;
	img = img_inp.clone();
	copyMakeBorder(img, img, border, border, border, border, BORDER_REPLICATE);
		
	cv::Mat descriptor = cv::Mat::zeros(1, 128 * shp.cols, CV_32FC1);	
	for (int i = 0; i < shp.cols; ++i) {		
		cv::Rect ROI;
		cv::Mat patch;
		vector<float> descriptorsValues;
		vector<Point> locations;
		int iter = 128 * i;

		// If a shape point is out of image, return descriptor containing all zeros
		if (shp.at<float>(0, i) < 0 || shp.at<float>(0, i) >= img_inp.cols ||
			shp.at<float>(1, i) < 0 || shp.at<float>(1, i) >= img_inp.rows) {
			for (int j = 0; j < 128; ++j) {
				descriptor.at<float>(0, iter + j) = 0;
			}
			continue;
		}

		ROI.x = shp.at<float>(0, i); ROI.y = shp.at<float>(1, i);
		ROI.width = 32; ROI.height = 32;
		patch = img(ROI).clone();

		HOGDescriptor d(cv::Size(32, 32), Size(16, 16), Size(16, 16), Size(8, 8), 8, 0, -1, 0, 0.2, 0);
		d.compute(patch, descriptorsValues, Size(0, 0), Size(0, 0), locations);

		for (int j = 0; j < descriptorsValues.size(); ++j) {
			descriptor.at<float>(0, iter + j) = descriptorsValues[j];
		}
	}
	out = descriptor;
}

cv::Mat resetshape(const cv::Mat &shp, cv::Rect box) {
	cv::Mat out;
	out = shp.clone();

	for (size_t i = 0; i < shp.rows; ++i) {
		out.at<float>(i, 0) = (out.at<float>(i, 0))*(box.width);
		out.at<float>(i, 1) = (out.at<float>(i, 1))*(box.height);

		out.at<float>(i, 0) = out.at<float>(i, 0) + (box.x);
		out.at<float>(i, 1) = out.at<float>(i, 1) + (box.y) + (box.height) / 5.0;
	}

	return out;
}

void readModel(std::istream &i, Mat &m) {
	int t, r, c;
	i.read((char*)&t, sizeof(int));
	i.read((char*)&c, sizeof(int));
	i.read((char*)&r, sizeof(int));
	m.create(r, c, t);
	i.read((char*)m.data, m.total() * m.elemSize());
}

void live() {
	// load the trained detection model
	std::vector<cv::Mat> Ws;
	cv::Mat mean_shape;
	std::ifstream in("ws.data", std::ios_base::binary);
	for (int i = 1; i <= 5; ++i) {
		Mat temp;
		readModel(in, temp);
		Ws.push_back(temp);
	}
	readModel(in, mean_shape);
	in.close();

	// detect the face
	if (!face_cascade.load(face_cascade_name)) { cout << "error loading the cascade classifier" << endl; }

	// live
	cv::VideoCapture camera;
	camera.open("vid.avi");
	//camera.open(0);             // Uncomment for webcamera
	Mat I, img;
	cv::Mat init_shape_org = cv::Mat::zeros(num_used_pts, 2, CV_32FC1);		
	cv::Mat one_mat = cv::Mat::ones(1, 1, CV_32FC1);

	LARGE_INTEGER start1, end1, freq1;
	QueryPerformanceFrequency(&freq1);

	int frate1;	
	float scale;
	cv::Mat features, feat;
	int width, height;
	while (1) {
		// Grab a frame from camera stream
		camera >> I;
		if (I.empty()) {
			break;
		}

		// Convert to gray image
		cv::cvtColor(I, img, CV_BGR2GRAY);
		width = img.cols; height = img.rows;
		
		// For measuring time 
		QueryPerformanceCounter(&start1);

		// Detect faces
		std::vector<cv::Rect> faces;
		face_cascade.detectMultiScale(img, faces, 1.2, 2, 0, cv::Size(100, 100));		
		
		if (!faces.empty()) {
			// Resize the image such that face size is 300 x 300			
			scale = 300 / (float)faces[0].width;
			cv::resize(img, img, cv::Size((int)(scale*width), (int)(scale*height)));

			// reset the mean shape onto the bounding box returned by the face detector
			cv::Mat init_shape = cv::Mat(68, 2, CV_32FC1);			
			init_shape = resetshape(mean_shape, faces[0]);

			// Project shape between 0 and 1
			for (int j = 0; j < init_shape.rows; j++) {
				init_shape.at<float>(j, 0) = (init_shape.at<float>(j, 0) - (faces[0].x)) / (faces[0].width);
				init_shape.at<float>(j, 1) = (init_shape.at<float>(j, 1) - (faces[0].y)) / (faces[0].height);
			}

			// Reset projected shape to the mean of previously known distribution
			for (int j = 0; j < init_shape.rows; ++j) {
				init_shape_org.at<float>(j, 0) = (init_shape.at<float>(j, 0) + 0.00792179) * faces[0].width + faces[0].x;
				init_shape_org.at<float>(j, 1) = (init_shape.at<float>(j, 1) - 0.0369028) * faces[0].height + faces[0].y;
			}

			// Resize the shape such that face size is 300 x 300				
			init_shape_org *= scale;

			//Main loop for calculations
			for (int i = 0; i < 5; ++i) {				
				cv::Mat init_temp = init_shape_org.t();

				// Calculate HOG features
				featHOG(img, init_temp, features, 32);

				// Concatenate with a column of ones (accounts for intercept in regression framework)
				cv::hconcat(one_mat, features, feat);

				// Multiply by learned model
				cv::Mat deltashapes_bar = feat*Ws[i];				

				// Update the shape
				init_shape_org += deltashapes_bar.reshape(0, num_used_pts);				
			}

			// Resize shape to original resolution
			init_shape_org /= scale;

			QueryPerformanceCounter(&end1);
			frate1 = (end1.QuadPart - start1.QuadPart) * 1000 / freq1.QuadPart;
			cout << "Program is running at " << (1000 / frate1) << " fps" << endl;

			// Plot the facial features
			for (size_t j = 0; j < init_shape_org.rows; ++j) {
				cv::circle(I, cv::Point(init_shape_org.at<float>(j, 0), init_shape_org.at<float>(j, 1)), 2, cv::Scalar(10, 255, 10), -1, 8, 0);
			}
		}
		cv::flip(I, I, 1);
		imshow("Image_c", I);
		char c = cv::waitKey(1);
		if ((char)c == 27) { break; }
	}
}


