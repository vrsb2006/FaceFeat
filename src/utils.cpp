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

cv::Mat ProjectShape(const cv::Mat &shape, const cv::Rect &bounding_box) {
	cv::Mat temp(shape.rows, 2, CV_32FC1);
	for (int j = 0; j < shape.rows; j++) {
		temp.at<float>(j, 0) = (shape.at<float>(j, 0) - (bounding_box.x + bounding_box.width / 2.0)) / (bounding_box.width / 2.0);		
		temp.at<float>(j, 1) = (shape.at<float>(j, 1) - (bounding_box.y + bounding_box.height / 2.0)) / (bounding_box.height / 2.0);
	}
	return temp;
}

cv::Mat ReProjectShape(const cv::Mat &shape, const cv::Rect &bounding_box) {
	cv::Mat temp(shape.rows, 2, CV_32FC1);
	for (int j = 0; j < shape.rows; j++) {
		temp.at<float>(j, 0) = (shape.at<float>(j, 0) * bounding_box.width / 2.0) + (bounding_box.x + bounding_box.width / 2.0);
		temp.at<float>(j, 1) = (shape.at<float>(j, 1) * bounding_box.height / 2.0) + (bounding_box.y + bounding_box.height / 2.0);
	}
	return temp;
}

void featHOG(cv::Mat &img_inp, cv::Mat &shp, cv::Mat &out, int winsize, cv::Rect bbox) {

	// resize image such that face is approx 300 x 300
	float scale = 300 / (float)bbox.width;
	cv::Mat img_temp;
	cv::resize(img_inp, img_temp, cv::Size((int)(scale*img_inp.cols), (int)(scale*img_inp.rows)));

	// resize shape
	shp *= scale;
	
	// to avoid error due to descriptor calculations at borders
	int border = (int)(winsize / 2);
	cv::Mat img;
	img = img_temp.clone();

	copyMakeBorder(img, img, border, border, border, border, BORDER_REPLICATE);

	cv::Mat descriptor = cv::Mat::zeros(1, 128 * shp.cols, CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < shp.cols; ++i) {		
		cv::Rect ROI;
		cv::Mat patch;
		vector<float> descriptorsValues;
		vector<Point> locations;
		int iter = 128 * i;

		if (shp.at<float>(0, i) < 0 || shp.at<float>(0, i) >= img_temp.cols ||
			shp.at<float>(1, i) < 0 || shp.at<float>(1, i) >= img_temp.rows) {
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

cv::Rect get_bbox(cv::Mat &shp) {
	float min0 = 10000, min1 = 10000, max0 = 0, max1 = 0;

	for (size_t i = 0; i < shp.rows; ++i) {
		if (shp.at<float>(i, 0) <= min0) {
			min0 = shp.at<float>(i, 0);
		}
		if (shp.at<float>(i, 0) >= max0) {
			max0 = shp.at<float>(i, 0);
		}
		if (shp.at<float>(i, 1) <= min1) {
			min1 = shp.at<float>(i, 1);
		}
		if (shp.at<float>(i, 1) >= max1) {
			max1 = shp.at<float>(i, 1);
		}
	}

	cv::Rect out;
	out.x = min0; out.y = min1;
	out.width = max0 - min0 + 1;
	out.height = max1 - min1 + 1;
	return out;
}

Mat correctGamma(Mat& img, float gamma) {
	float inverse_gamma = 1.0 / gamma;

	Mat lut_matrix(1, 256, CV_8UC1);
	uchar * ptr = lut_matrix.ptr();
	for (int i = 0; i < 256; i++)
		ptr[i] = (int)(pow((float)i / 255.0, inverse_gamma) * 255.0);

	Mat result;
	LUT(img, lut_matrix, result);

	return result;
}

void live() {
	// load the trained detection model
	std::vector<cv::Mat> Ws;
	for (int i = 1; i <= 5; ++i) {
		char name1[10], name2[10];
		sprintf(name1, "Ws%d.yml", i);
		cv::FileStorage Fs(name1, FileStorage::READ);
		sprintf(name2, "Ws%d", i);
		cv::Mat temp;
		Fs[name2] >> temp;
		Ws.push_back(temp);
		temp.release();
		Fs.release();
	}
	
	// load mean shape
	cv::Mat mean_shape;
	cv::FileStorage Fs("mean_shape_org.yml", FileStorage::READ);
	Fs["mean_shape"] >> mean_shape;
	Fs.release();
	mean_shape.convertTo(mean_shape, CV_32FC1);

	// detect the face
	if (!face_cascade.load(face_cascade_name)) { cout << "error loading the cascade classifier" << endl; }

	// live
	cv::VideoCapture camera;
	camera.open("vid.avi");	
	Mat I, img;
	cv::Mat init_shape_org = cv::Mat::zeros(num_used_pts, 2, CV_32FC1);		cv::Mat one_mat = cv::Mat::ones(1, 1, CV_32FC1);

	
#ifdef write_video
	VideoWriter outputVideo;                                       	
	outputVideo.open("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(780, 580), true);

	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video for write: " << endl;
		return;
	}
#endif
	bool write = false;
	while (1) {
		camera >> I;				
		if (I.empty()) {
			break;
		}
		//I = correctGamma(I, 1.2);
		cv::flip(I, I, 1);				
		cv::cvtColor(I, img, CV_BGR2GRAY);

		std::vector<cv::Rect> faces;
		face_cascade.detectMultiScale(img, faces, 1.2, 2, 0, cv::Size(100, 100));

		if (!faces.empty()) {
			// reset the mean shape onto the bounding box returned by the face detector
			cv::Mat init_shape = cv::Mat(68, 2, CV_32FC1);
			cv::Mat mn_shape = mean_shape.clone();
			init_shape = resetshape(mn_shape, faces[0]);

			int count = 0;
			for (int j = 0; j < init_shape.rows; ++j) {
				init_shape_org.at<float>(j, 0) = init_shape.at<float>(j, 0);// - 0.240297;
				init_shape_org.at<float>(j, 1) = init_shape.at<float>(j, 1);// - 1.97803;					
			}

			//Main loop for calculations
			for (int i = 0; i < 5; ++i) {
				cv::Mat features, feat;
				cv::Mat init_temp = init_shape_org.t();
				featHOG(img, init_temp, features, 32, faces[0]);
				cv::hconcat(one_mat, features, feat);

				cv::Mat deltashapes_bar = feat*Ws[i];
				cv::Mat deltashapes_bar_xy = deltashapes_bar.reshape(0, num_used_pts);

				init_shape_org = ProjectShape(init_shape_org, faces[0]) + deltashapes_bar_xy;
				init_shape_org = ReProjectShape(init_shape_org, faces[0]);
			}
			//cv::rectangle(I, faces[0], cv::Scalar(0, 255, 0), 1, 8, 0);
			for (size_t j = 0; j < init_shape_org.rows; ++j) {
				cv::circle(I, cv::Point(init_shape_org.at<float>(j, 0), init_shape_org.at<float>(j, 1)), 2, cv::Scalar(0, 255, 0), -1, 8, 0);				
			}
		}
#ifdef write_video
		if (write) {
			outputVideo << I;
		}
#endif		
		imshow("Image", I);
		char c = cv::waitKey(5);
		if ((char)c == 27) { break; }
		if ((char)c == 's') { write = true; }
	}
#ifdef write_video
	outputVideo.release();
#endif
}

void test_images(std::string path) {

	// Read images
	ifstream fin;
	fin.open(path);
	std::string input_name;
	cv::Mat img_temp;
	std::vector<cv::Mat> images;
	while (getline(fin, input_name)) {
		img_temp = imread(input_name);
		images.push_back(img_temp);
	}
	fin.close();

	// load the trained detection model
	std::vector<cv::Mat> Ws;
	for (int i = 1; i <= 5; ++i) {
		char name1[10], name2[10];
		sprintf(name1, "Ws%d.yml", i);
		cv::FileStorage Fs(name1, FileStorage::READ);
		sprintf(name2, "Ws%d", i);
		cv::Mat temp;
		Fs[name2] >> temp;
		Ws.push_back(temp);
		temp.release();
		Fs.release();
	}

	// load mean shape
	cv::Mat mean_shape;
	cv::FileStorage Fs("mean_shape_org.yml", FileStorage::READ);
	Fs["mean_shape"] >> mean_shape;
	Fs.release();
	mean_shape.convertTo(mean_shape, CV_32FC1);

	// detect the face
	if (!face_cascade.load(face_cascade_name)) { cout << "error loading the cascade classifier" << endl; }

	// test on images	
	Mat I, img;
	cv::Mat init_shape_org = cv::Mat::zeros(num_used_pts, 2, CV_32FC1);		cv::Mat one_mat = cv::Mat::ones(1, 1, CV_32FC1);
	for (size_t iter = 0; iter < images.size();++iter) {
		cout << iter << endl;
		
		I = images[iter].clone();
		//I = correctGamma(I, 1.2);
		cv::flip(I, I, 1);
		cv::cvtColor(I, img, CV_BGR2GRAY);

		std::vector<cv::Rect> faces;
		face_cascade.detectMultiScale(img, faces, 1.2, 2, 0, cv::Size(50, 50));

		if (!faces.empty()) {
			// reset the mean shape onto the bounding box returned by the face detector
			cv::Mat init_shape = cv::Mat(68, 2, CV_32FC1);
			cv::Mat mn_shape = mean_shape.clone();
			init_shape = resetshape(mn_shape, faces[0]);

			int count = 0;
			for (int j = 0; j < init_shape.rows; ++j) {
				init_shape_org.at<float>(j, 0) = init_shape.at<float>(j, 0);// - 0.240297;
				init_shape_org.at<float>(j, 1) = init_shape.at<float>(j, 1);// - 1.97803;					
			}

			//Main loop for calculations
			for (int i = 0; i < 5; ++i) {
				cv::Mat features, feat;
				cv::Mat init_temp = init_shape_org.t();
				featHOG(img, init_temp, features, 32, faces[0]);
				cv::hconcat(one_mat, features, feat);

				cv::Mat deltashapes_bar = feat*Ws[i];
				cv::Mat deltashapes_bar_xy = deltashapes_bar.reshape(0, num_used_pts);

				init_shape_org = ProjectShape(init_shape_org, faces[0]) + deltashapes_bar_xy;
				init_shape_org = ReProjectShape(init_shape_org, faces[0]);
			}
			cv::rectangle(I, faces[0], cv::Scalar(0, 255, 0), 1, 8, 0);
			for (size_t j = 0; j < init_shape_org.rows; ++j) {
				cv::circle(I, cv::Point(init_shape_org.at<float>(j, 0), init_shape_org.at<float>(j, 1)), 2, cv::Scalar(0, 255, 0), -1, 8, 0);				
			}
		}
		//imwrite("output/" + std::to_string(iter) + ".jpg", I);
		imshow("Image", I);		
		char c = cv::waitKey(0);
		if ((char)c == 27) { break; }
	}
}
