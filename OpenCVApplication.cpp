// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <fstream>
#include <random>
#include <ctime>
using namespace std;
using namespace cv;


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void leastMeanSquare() {
	char filename[100] = "PointsLeastSquares/points3.txt";
	ifstream f(filename);

	int n;
	f >> n;
	Point2f points[1000];
	float xmin = INT32_MAX, ymin = INT32_MAX, xmax = INT32_MIN, ymax = INT32_MIN;
	for (int i = 0; i < n; i++) {
		f >> points[i].x >> points[i].y;
		if (points[i].x < xmin)
			xmin = points[i].x;
		if (points[i].x > xmax)
			xmax = points[i].x;
		if (points[i].y < ymin)
			ymin = points[i].y;
		if (points[i].y > ymax)
			ymax = points[i].y;
	}

	Mat img(500, 500, CV_8UC3, CV_RGB(255, 255, 255));
	for (int i = 0; i < n; i++)
		drawMarker(img, cv::Point(points[i].x, points[i].y), CV_RGB(0, 0, 0), MARKER_CROSS, 4, 1, 8);

	float teta0 = 0, teta1 = 0, sumxy = 0, sumx = 0, sumy = 0, sumx2 = 0, sum = 0;
	for (int i = 0; i < n; i++)
	{
		sumxy += points[i].x*points[i].y;
		sumx += points[i].x;
		sumy += points[i].y;
		sumx2 += points[i].x*points[i].x;
		sum += points[i].y*points[i].y - points[i].x*points[i].x;
	}

	// model 1
	teta1 = (n*sumxy - sumx * sumy) / (n*sumx2 - sumx * sumx);
	teta0 = (sumy - teta1 * sumx) / n;
	//cout << "teta0: " << teta0 << ' ' << "teta1: " << teta1 << '\n';
	float fx = teta0 + teta1 * img.cols;
	line(img, Point(0, teta0), Point(img.cols, fx), CV_RGB(255, 0, 0), 3, 8, 0);

	// model 2
	float beta = -1 * (atan2(2 * sumxy - (2 * (sumx*sumy) / n), sumx * sumx / n - sumy * sumy / n + sum)) / 2;
	float p = (cos(beta)*sumx + sin(beta)*sumy) / n;
	//cout << beta << ' ' << p;
	Point one(0, p / sin(beta));
	Point two(img.cols, ((p - img.cols * cos(beta)) / sin(beta)));
	line(img, one, two, CV_RGB(0, 0, 255), 1, 8, 0);

	imshow(filename, img);
	waitKey(0);
	f.close();
}



void RANSACLineFitting() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	
	vector<Point> points;
	float t = 10, p = 0.99, q = 0.3, s = 2, N;
	
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (src.at<uchar>(i, j) == 0)
				points.push_back(Point(j, i));

	N = log(1 - p) / log(1 - pow(q, s));
	for (int tr = 0; tr <= 5; tr++)
	{
		int largest = 0;
		Point bestp1, bestp2;
		for (int i = 0; i < ceil(N); i++)
		{
			int first = rand() % points.size();
			int second = rand() % points.size();
			while (first == second)
				second = rand() % points.size();
			Point point1 = points.at(first);
			Point point2 = points.at(second);
			float a = point1.y - point2.y;
			float b = point2.x - point1.x;
			float c = point1.x*point2.y - point2.x * point1.y;

			int setSize = 0;
			for (int j = 0; j < points.size(); j++)
			{
				Point current = points.at(j);
				float dist = fabs(a*current.x + b * current.y + c) / sqrt(a*a + b * b);
				if (dist < t)
					setSize++;
			}
			if (setSize > largest)
			{
				largest = setSize;
				bestp1 = point1;
				bestp2 = point2;
			}
			//if Si > T break
			if (largest > q * points.size())
				break;
		}

		line(src, bestp1, bestp2, 1, 1, 8, 0);
		imshow("Points and line", src);
		waitKey(0);
	}
}

struct houghLine {
	int ro, teta, votes;
	bool operator < (const houghLine& o) const {
		return votes > o.votes;
	}
};

void houghLineDetection() {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat cpy = imread(fname, CV_LOAD_IMAGE_COLOR);

	int D = sqrt(pow(src.rows, 2) + pow(src.cols, 2));
	Mat hough = cv::Mat::zeros(360, D + 1, CV_32SC1);
	
	int maxHough = 0, dTeta = 1;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {
				for (int t = 0; t < 360; t += dTeta) {
					int ro = i * cos(t * PI/180) + j * sin(t * PI/180);
					if (ro > 0) {
						hough.at<int>(t, ro)++;
						if (hough.at<int>(t, ro) > maxHough)
							maxHough = hough.at<int>(t, ro);
					}
				}
			}
		}

	// select peaks
	vector<houghLine> lines;
	int window = 3;
	for (int t = 0; t < hough.rows; t++)
		for (int ro = window; ro < hough.cols - window; ro++) {

			// check if it is local maxima
			int val = hough.at<int>(t, ro);
			int max = 0;
			for(int i= t - window; i <= t + window; i++)
				for (int j = ro - window; j <= ro + window; j++) {
						if (hough.at<int>((i + 360) % 360, j) > max)
							max = hough.at<int>((i + 360) % 360, j);
				}

			if (max == val)
				lines.push_back(houghLine{ ro, t, hough.at<int>(t, ro) });
		}

	cout << "no of lines: " << lines.size();
	std::sort(lines.begin(), lines.end());

	int k = 10;
	//vector<Point> cartesianPoints;
	for (int i = 0; i < k; i++) {
		int ro = lines.at(i).ro;
		int t = lines.at(i).teta;
		if ((t > 45 && t < 135) || (t > 225 && t < 315))
			//vertical
			line(cpy, Point(ro / cos(t * PI / 180), 0), Point(ro - cpy.rows*sin(t * PI / 180) / cos(t * PI / 180), cpy.rows), CV_RGB(0, 255, 0), 1, 8, 0);
		else
			line(cpy, Point(0, ro / sin(t * PI/180)), Point(cpy.cols, (ro - cpy.cols*cos(t * PI / 180)) / sin(t * PI / 180)), CV_RGB(0, 255, 0), 1, 8, 0);
	}
		
	Mat houghImg;
	cout << maxHough;
	hough.convertTo(houghImg, CV_8UC1, 255.f/maxHough);
	imshow("Hough space", houghImg);
	imshow("Hough lines", cpy);
	waitKey(0);
}

Mat chamferDistanceTransform(Mat src) {
	Mat dt = src.clone();

	int diag_weight = 3, horiz_weight = 2;
	//top-down
	for(int i = 1; i < dt.rows - 1; i++)
		for (int j = 1; j < dt.cols - 1; j++) 
		{
			int min = 255;
			for(int k = i - 1; k <= i; k++)
				for (int l = j - 1; l <= j + 1; l++) {
					int weight;
					if ((abs(k - i)== 1) && (abs(l - j) == 1))
						weight = diag_weight;
					else if (k == i && l == j)
						weight = 0;
					else
						weight = horiz_weight;

					if (!((l == j + 1) && (k == i))) {
						int val = dt.at<uchar>(k, l) + weight;
						if (val < min)
							min = val;
					}
				}
			dt.at<uchar>(i, j) = min;
		}

	//bottom-up
	for (int i = dt.rows - 2; i >= 1; i--)
		for (int j = dt.cols - 2; j >= 1; j--)
		{
			int min = 255;
			for (int k = i; k <= i + 1; k++)
				for (int l = j - 1; l <= j + 1; l++) {
					int weight;
					if ((abs(k - i) == 1) && (abs(l - j) == 1))
						weight = diag_weight;
					else if (k == i && l == j)
						weight = 0;
					else
						weight = horiz_weight;

					if (!((l == j - 1) && (k == i))) {
						int val = dt.at<uchar>(k, l) + weight;
						if (val < min)
							min = val;
					}
				}
			dt.at<uchar>(i, j) = min;
		}

	imshow("original", src);
	imshow("distance transform", dt);
	waitKey(0);
	return dt;
}

void dtPatternMatching() {
	char fname1[MAX_PATH];
	openFileDlg(fname1);
	Mat unknown = imread(fname1, CV_LOAD_IMAGE_GRAYSCALE);

	char fname2[MAX_PATH];
	openFileDlg(fname2);
	Mat tmpl = imread(fname2, CV_LOAD_IMAGE_GRAYSCALE);

	Mat unknown_dt = chamferDistanceTransform(unknown);

	int sum = 0, count = 0;
	for(int i = 0; i < tmpl.rows; i++)
		for (int j = 0; j < tmpl.cols; j++) {
			if (tmpl.at<uchar>(i, j) == 0) {
				count++;
				sum += unknown_dt.at<uchar>(i, j);
			}
		}

	cout << "score: " << (float) sum / count;
	imshow("unknown", unknown);
	imshow("template", tmpl);
	waitKey(0);
}

void statisticalDataAnalysis() {
	int p = 400;
	int N = 361;
	Mat features(p, N, CV_8UC1);

	char folder[256] = "images_faces";
	char fname[256];
	//build matrix with all features
	for (int k = 1; k <= p; k++) {
		sprintf(fname, "%s/face%05d.bmp", folder, k);
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		for(int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
				features.at<uchar>(k-1, (i * 19 + j)) = img.at<uchar>(i, j);
	}
	//compute mean values for each image
	float mean_values[361];
	for (int i = 0; i < N; i++) {
		int sum = 0;
		for (int k = 0; k < p; k++)
			sum += features.at<uchar>(k, i);
		mean_values[i] = (float)sum / p;
		//cout << mean_values[i] << "\n";
	}

	Mat cov(N, N, CV_32FC1, float(0));
	//compute covariance matrix values
	for (int i = 0; i < N; i++) {
		for (int j = 0; j <= i; j++) {
			float sum = 0.0;
			for (int k = 0; k < p; k++)
				sum += ((float)features.at<uchar>(k, i) - mean_values[i]) * ((float)features.at<uchar>(k, j) - mean_values[j]);
			cov.at<float>(i, j) = sum / p;
			cov.at<float>(j, i) = cov.at<float>(i, j);
		}
	}
	//compute correlation matrix
	Mat cor(N, N, CV_32FC1, float(0));
	for(int i = 0; i< N; i++)
		for (int j = 0; j < N; j++) {
			cor.at<float>(i, j) = cov.at<float>(i, j) / (sqrt(cov.at<float>(i, i)) * sqrt(cov.at<float>(j, j)));
		}

	//write to csv
	FILE* cov_out = fopen("cov.csv", "w");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			fprintf(cov_out, "%.2f, ", cov.at<float>(i, j));
		fprintf(cov_out, "\n");
	}

	FILE* cor_out = fopen("cor.csv", "w");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			fprintf(cor_out, "%.2f, ", cor.at<float>(i, j));
		fprintf(cor_out, "\n");
	}
	fclose(cov_out);
	fclose(cor_out);

	//correlation
	Mat chart(256, 256, CV_8UC1);
	int x1 = 5, y1 = 4, x2 = 5, y2 = 14;
	for (int i = 0; i < p; i++) {
		int i1 = features.at<uchar>(i, x1 * 19 + y1);
		int i2 = features.at<uchar>(i, x2 * 19 + y2);
		chart.at<uchar>(i1, i2) = 0;
	}
	cout << "correlation coef: " << cor.at<float>(x1 * 19 + y1, x2 * 19 + y2);
	imshow("chart", chart);
	waitKey(0);
}

struct PointData {
	int x1;
	int x2;
	int label;
};

int distPointData(PointData p1, PointData p2) {
	return pow((p1.x1 - p2.x1), 2) + pow((p1.x2 - p2.x2), 2);
}

void kMeansClustering(int k) {
	char fname1[MAX_PATH];
	openFileDlg(fname1);
	Mat src = imread(fname1, CV_LOAD_IMAGE_GRAYSCALE);

	vector<PointData> x;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0)
				x.push_back(PointData{ i, j, -1 });
		}
	}
	int n = x.size();

	//initialize clusters
	default_random_engine gen;
	gen.seed(time(NULL));
	uniform_int_distribution<int> distribution(0, n-1);
	vector<PointData> centers;
	for (int i = 0; i < k; i++) {
		int randint = distribution(gen);
		x[randint].label = i;
		centers.push_back(x[randint]);
	}

	bool stop = false;
	while (stop == false) {
		//assignment
		for (int i = 0; i < n; i++) {
			int minDist = INT_MAX;
			//find closest cluster center
			for (int j = 0; j < k; j++) {
				int dist = distPointData(x[i], centers[j]);
				if (dist < minDist) {
					minDist = dist;
					x[i].label = centers[j].label;
				}
			}
		}

		//update centers
		for (int i = 0; i < k; i++) {
			int x1Sum = 0;
			int x2Sum = 0;
			int noPointsWithLabel = 0;
			for (int j = 0; j < n; j++) {
				if (x[j].label == i) {
					x1Sum += x[j].x1;
					x2Sum += x[j].x2;
					noPointsWithLabel++;
				}
			}
			if (centers[i].x1 == (x1Sum / noPointsWithLabel) && centers[i].x2 == (x2Sum / noPointsWithLabel))
				stop = true;
			else {
				centers[i].x1 = x1Sum / noPointsWithLabel;
				centers[i].x2 = x2Sum / noPointsWithLabel;
			}
		}
	}

	Mat img(src.rows, src.cols, CV_8UC3, Scalar(255, 255, 255));
	Vec3b* colors = new Vec3b[k];
	srand(time(NULL));
	for (int i = 0; i < k; i++) {
		unsigned char r = rand() % 256;
		unsigned char g = rand() % 256;
		unsigned char b = rand() % 256;
		colors[i] = { r, g, b };
	}
		

	for (int i = 0; i < n; i++) {
		PointData point = x[i];
		img.at<Vec3b>(point.x1, point.x2) = colors[point.label];
	}

	//voronoi
	Mat voronoi(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < voronoi.rows; i++)
		for (int j = 0; j < voronoi.cols; j++) {
			int min = INT_MAX;
			int label;
			for (int t = 0; t < k; t++) {
				int dist = distPointData(PointData{ i, j, 0 }, centers[t]);
				if (dist < min) {
					min = dist;
					label = t;
				}
			}
			voronoi.at<Vec3b>(i, j) = colors[label];
		}

	imshow("original", src);
	imshow("clusters", img);
	imshow("voronoi", voronoi);
	waitKey(0);
}


void principalComponent() {
	char* filename = "data_PCA/pca2d.txt";
	ifstream f(filename);
	int n, d;
	f >> n >> d;
	Mat X(n, d, CV_64FC1);

	//read data
	vector<double> meanVector(d);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < d; j++) {
			f >> X.at<double>(i, j);
			meanVector.at(j) += X.at<double>(i, j);
			//cout << X.at<double>(i, j) << " ";
		}
		//cout << "\n";
	}

	//compute mean vector
	for (int i = 0; i < d; i++)
		meanVector.at(i) /= n;

	//subtract mean
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			X.at<double>(i, j) -= meanVector.at(j);

	Mat C = X.t() * X / (n - 1);
	Mat Lambda, Q;
	eigen(C, Lambda, Q);
	Q = Q.t();
	//print eigenvalues
	cout << "Eigenvalues: \n";
	for (int i = 0; i < d; i++)
		cout << Lambda.at<double>(i, 0)<< " ";

	int k = 2;
	Rect roi(0, 0, k, d);
	Mat Qk(Q, roi);
	Mat Xpca = X * Qk;
	Mat Xk = Xpca * Qk.t();

	//compute mean absolute difference
	double MAD = 0.0;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			MAD += fabs(X.at<double>(i, j) - Xk.at<double>(i, j));
	MAD /= n*d;
	cout << "\nMean absolute difference: " << MAD << endl;

	//compute min and max along columns of coeff matrix
	double min = 99999, max = -99999, xmin = 9999, xmax = -9999, ymin = 9999, ymax = -9999;
	for(int i = 0; i < n; i ++)
		for (int j = 0; j < 2; j++) {
			if (Xpca.at<double>(i, 0) > xmax) xmax = Xpca.at<double>(i, 0);
			if (Xpca.at<double>(i, 0) < xmin) xmin = Xpca.at<double>(i, 0);
			if (Xpca.at<double>(i, 1) < ymin) ymin = Xpca.at<double>(i, 1);
			if (Xpca.at<double>(i, 1) > ymax) ymax = Xpca.at<double>(i, 1);
		}

	Mat plot((int)(ymax - ymin + 1), (int)(xmax - xmin + 1), CV_8UC1, Scalar(255));
	for (int i = 0; i < n; i++)
			plot.at<uchar>((int)(Xpca.at<double>(i,1) - ymin), (int)(Xpca.at<double>(i, 0) - xmin)) = 0;

	imshow("plot", plot);
	waitKey(0);
	f.close();
	getc(0);
}

Mat calcHist(Mat img, int nr_bins) {
	Mat hist_vec(1, nr_bins * 3, CV_32S, Scalar(0));
	int el_per_bin = 256 / nr_bins;

	for(int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			Vec3b value = img.at<Vec3b>(i, j);
			hist_vec.at<int>(value[0] / el_per_bin)++;
			hist_vec.at<int>(nr_bins + (value[1] / el_per_bin))++;
			hist_vec.at<int>(2 * nr_bins + (value[2] / el_per_bin))++;
		}
	return hist_vec;
}

float calcDist(Mat x, Mat y) {
	float dist = 0.0;
	for (int i = 0; i < x.cols; i++)
		dist += pow(x.at<int>(0, i) - y.at<int>(0, i), 2);
	return sqrt(dist);
}

struct Neighbour {
	int index;
	float distance;

	bool operator < (const Neighbour& n) const
	{
		return (distance < n.distance);
	}
};

bool compareNeighbours(const Neighbour& n1, const Neighbour& n2) {
	return n1.distance < n2.distance;
}

void kNearestClassifier(int no_of_bins, int k) {
	const int nrclasses = 6;
	char classes[nrclasses][10] = { "beach", "city", "desert", "forest", "landscape", "snow" };

	int feature_dim = no_of_bins * 3;
	Mat X(0, feature_dim, CV_32S);
	vector<int> Y;

	char* fname = (char*)malloc(256 * sizeof(char));
	for (int c = 0; c < nrclasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "images_KNN/train/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0)
				break;

			//add the histogram vec to global hist matrix
			X.push_back(calcHist(img, no_of_bins));
			Y.push_back(c);
		}
		//cout << "class " << c<< ": " << X.rows << " " << Y.size();
	}
	cout << "Finished computing feature vectors";

	//load test image
	Mat confusionMat(nrclasses, nrclasses, CV_32S, Scalar(0));
	for (int c = 0; c < nrclasses; c++) {
		int fileNr = 0;
		while (1) {
			sprintf(fname, "images_KNN/test/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
			if (img.cols == 0)
				break;

			Mat hist_vec = calcHist(img, no_of_bins);
			vector<Neighbour> neighbours;
			// compute distances
			for (int i = 0; i < X.rows; i++)
				neighbours.push_back(Neighbour{ i,  calcDist(hist_vec, X.row(i))});

			// sort neighbour vector
			sort(neighbours.begin(), neighbours.end());
			int classVotes[6] = {0, 0, 0, 0, 0, 0};
			for (int i = 0; i < k; i++)
				classVotes[Y.at(neighbours.at(i).index)]++;
			// predict class
			int max = classVotes[0], predictedClass = 0;
			for(int i = 1; i< nrclasses; i++)
				if (classVotes[i] > max) {
					max = classVotes[i];
					predictedClass = i;
				}
			 
			confusionMat.at<int>(c, predictedClass)++;
			//free(classVotes);
		}
	}
	
	// compute confusion matrix
	float accuracy;
	int mainDiagCount = 0;
	int nr_el = 0;
	cout << "Confusion matrix:\n";
	for (int i = 0; i < nrclasses; i++) {
		for (int j = 0; j < nrclasses; j++) {
			if(i == j)
				mainDiagCount += confusionMat.at<int>(i, i);
			nr_el += confusionMat.at<int>(i, j);
			cout << confusionMat.at<int>(i, j) << " ";
		}
		cout << endl;
	}
	cout << "\nAccuracy: " << (float)(mainDiagCount) / nr_el;

	free(fname);
	char a;
	scanf("%c", &a);
	scanf("%c", &a);
}

int classifyBayes(Mat img, Mat likelihood, Mat priors) {
	Mat copy;
	img.copyTo(copy);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img.at<uchar>(i, j) > 128)
				copy.at<uchar>(i, j) = 255;
			else
				copy.at<uchar>(i, j) = 0;

	int predicted_class = -1;
	const int no_classes = 10;
	double probs[no_classes];
	double max_prob = INT_MIN;
	for (int c = 0; c < no_classes; c++) {
		double likelihood_product = 0.0;
		for (int i = 0; i < copy.rows; i++)
			for (int j = 0; j < copy.cols; j++) {
				double p = likelihood.at<double>(c, i * copy.cols + j);
				if(copy.at<uchar>(i, j) == 255)
					likelihood_product += log(p);
				else
					likelihood_product += log(1 - p);
			}
		double posterior = likelihood_product + log(priors.at<double>(c, 0));
		probs[c] = posterior;
		if (posterior > max_prob) {
			max_prob = posterior;
			predicted_class = c;
		}
	}
	// print probabilities
	for (int i = 0; i < no_classes; i++)
		printf("Prob. for class %d is %f\n", i, probs[i]);

	return predicted_class;
}

void trainNaiveBayes() {
	char fname[256];
	int load_limit = 100;

	int d = 28 * 28;
	int n = 0;
	int no_classes = 10;
	Mat fX(load_limit * no_classes, d, CV_8UC1);
	Mat Y(load_limit * no_classes, 1, CV_8UC1);
	Mat priors(no_classes, 1, CV_64FC1);
	Mat likelihood(no_classes, d, CV_64FC1);
	for (int i = 0; i < no_classes; i++)
		for (int j = 0; j < d; j++)
			likelihood.at<double>(i, j) = 0.0;

	for (int c = 0; c < no_classes; c++) {
		int index = 0;
		// load first 100 images
		while (index < load_limit) {
			sprintf(fname, "C:/Users/zenbookx/Documents/Facultate/An IV/PRS/OpenCVApplication-VS2015_OCV31_basic/images_Bayes/train/%d/%06d.png", c, index);
			Mat img = imread(fname, 0);
			if (img.cols == 0)
				break;

			// binarize image
			for(int i = 0; i<img.rows; i++)
				for (int j = 0; j < img.cols; j++) {
					if (img.at<uchar>(i, j) > 128) {
						fX.at<uchar>(n, i*img.cols + j) = 255;
						likelihood.at<double>(c, i*img.cols + j)++;
					} else
						fX.at<uchar>(n, i*img.cols + j) = 0;
				}
			Y.at<uchar>(n, 0) = c;
			n++;
			index++;
		}
		priors.at<double>(c, 0) = index /(double) (load_limit * no_classes);
		for (int i = 0; i < d; i++) {
			double p = likelihood.at<double>(c, i) / index;
			if (p == 0.0)
				likelihood.at<double>(c, i) = pow(10, -5);
			else
				likelihood.at<double>(c, i) = p;
		}
	}
	cout << "Loaded images\n";

	/*for (int i = 0; i < 10; i++)
		printf("%lf ", priors.at<double>(i, 0));*/

	char fname1[256];
	openFileDlg(fname1);
	Mat src = imread(fname1, CV_LOAD_IMAGE_GRAYSCALE);
	int predicted_class = classifyBayes(src, likelihood, priors);
	cout << "Predicted class: " << predicted_class << endl;

	imshow("original", src);
	waitKey(0);
}

void batchPerceptron() {
	char fname1[256];
	openFileDlg(fname1);
	Mat src = imread(fname1, CV_LOAD_IMAGE_COLOR);

	Mat X(0, 3, CV_32FC1);
	Mat Y(0, 1, CV_32FC1);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			Vec3b colour = src.at<Vec3b>(i, j);
			if (colour != Vec3b(255, 255, 255)) {
				float coord[3] = { 1, j, i };
				X.push_back(Mat(1, 3, CV_32FC1, coord));
				if (colour == Vec3b(255, 0, 0))
					// red
					Y.push_back(-1.0f);
				else if (colour == Vec3b(0, 0, 255))
					// blue
					Y.push_back(1.0f);
			}
		}

	float etta = pow(10, -4);
	float e_limit = pow(10, -5);
	Mat w(1, 3, CV_32FC1, { 0.1, 0.1, 0.1 });
	int max_iter = pow(10, 5);
	int n = Y.rows;
	Mat gradL(1, 3, CV_32FC1, Scalar(0));
	float e;

	for (int iter = 0; iter < max_iter; iter++) {
		e = 0;
		for (int i = 0; i < n; i++) {
			float zi = 0;
			for (int j = 0; j < 3; j++)
				zi += w.at<float>(j) * X.at<float>(i, j);
			if (zi * Y.at<float>(i, 0) <= 0.0) {
				Mat temp = Y.at<float>(i, 0) * X.row(i);
				gradL = gradL - temp;
				e++;
			}
		}
		e /= n;
		gradL = gradL / n;
		if (e < e_limit)
			break;
		for(int i = 0; i<3; i++)
			w.at<float>(i) = w.at<float>(i) - etta*gradL.at<float>(i);
	}
	
	// draw line
	int y2 = (int)w.at<float>(0) + w.at<float>(1) * X.at<float>(n-1, 0) + w.at<float>(2) * X.at<float>(n-1, 1);
	line(src, Point2i(0, (int)(-w.at<float>(0) / w.at<float>(2))), Point2i(src.cols, (int)( - (w.at<float>(0) + w.at<float>(1) * src.cols) / w.at<float>(2))), 0, 1, 8, 0);

	imshow("boundary", src);
	waitKey(0);
}

#define MAXT 100

struct weaklearner {
	int feature_i;
	int threshold;
	int class_label;
	float error;

	int classify(Mat x) {
		if (x.at<int>(feature_i) < threshold)
			return class_label;
		else
			return -1 * class_label;
	}
};

struct classifier {
	int T;
	float alphas[MAXT];
	weaklearner hs[MAXT];
	
	int classify(Mat x) {
		float s = 0.0;

		for (int t = 0; t < T; t++) {
			s += alphas[t] * hs[t].classify(x.row(0));
		}
		if (s < 0.0)
			return -1;
		else
			return 1;
	}
};

weaklearner findWeakLearner(Mat img, Mat X, vector<int> y, vector<float> w) {
	weaklearner best_h = weaklearner();
	float best_err = 9999.9;
	int class_label[2] = { -1, 1 };

	for (int j = 0; j < X.cols; j++) {
		int size;
		j == 0 ? size = img.rows : size = img.cols;

		for (int th = 0; th < size; th++) {
			for (int c = 0; c < 2; c++) {
				float e = 0.0;
				for (int i = 0; i < X.rows; i++) {
					int zi = 0;
					if (X.at<int>(i, j) < th)
						zi = class_label[c];
					else
						zi = -1 * class_label[c];

					if (zi*y[i] < 0)
						e += w[i];
				}
				if (e < best_err) {
					best_err = e;
					best_h = weaklearner{ j, th, class_label[c], e };
				}
			}
		}
	}
	return best_h;
}

void adaBoost() {
	char fname1[256];
	openFileDlg(fname1);
	Mat src = imread(fname1, CV_LOAD_IMAGE_COLOR);

	Mat X(0, 2, CV_32S);
	vector<int> Y;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3b colour = src.at<Vec3b>(i, j);
			if (colour != Vec3b(255, 255, 255)) {
				int coord[2] = { j, i };
				X.push_back(Mat(1, 2, CV_32S, coord));
				if (colour == Vec3b(255, 0, 0))
					// red
					Y.push_back(-1);
				else if (colour == Vec3b(0, 0, 255))
					// blue
					Y.push_back(1);
			}
		}
	}

	int n = X.rows;
	vector<float> w(n, 1.0 / n);
	int T = 10;
	classifier c = classifier();
	c.T = T;
	for (int t = 0; t < T; t++) {
		weaklearner ht = findWeakLearner(src, X, Y, w);
		float et = ht.error;
		float alpha_t = 0.5 * log((1 - et) / et);
		float s = 0.0;
		for (int i = 0; i < n; i++) {
			w[i] = w[i] * exp(-1 * alpha_t * Y[i] * ht.classify(X.row(i)));
			s += w[i];
		}
		//normalize weights
		for (int i = 0; i < n; i++)
			w[i] = w[i] / s;

		c.alphas[t] = alpha_t;
		c.hs[t] = ht;
	}

	Mat copy;
	src.copyTo(copy);
	for(int i = 0; i< copy.rows; i++)
		for (int j = 0; j < copy.cols; j++) {
			Vec3b colour = src.at<Vec3b>(i, j);
			if (colour == Vec3b(255, 255, 255)) {
				int coords[2] = { j, i };
				int sign = c.classify(Mat(1, 2, CV_32S, coords));
				if (sign == 1)
					copy.at<Vec3b>(i, j) = Vec3b(166, 253, 251);
				else if(sign == -1)
					copy.at<Vec3b>(i, j) = Vec3b(253, 246, 166);
			}
		}

	imshow("src", src);
	imshow("coloured", copy);
	waitKey(0);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - L1 - Least Mean Square\n");
		printf(" 11 - L2 - RANSAC Line Fitting\n");
		printf(" 12 - L3 - Hough Line Detection\n");
		printf(" 13 - L4 - Distance transform\n");
		printf(" 14 - L4 - Pattern matching\n");
		printf(" 15 - L5 - Statistical Data Analysis\n");
		printf(" 16 - L6 - K Means Clustering\n");
		printf(" 17 - L7 - Principal component analysis\n");
		printf(" 18 - L8 - KNN Classifier\n");
		printf(" 19 - L9 - Naive Bayes Classifier\n");
		printf(" 20 - L10 - Batch perceptron\n");
		printf(" 21 - L11 - AdaBoost\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				leastMeanSquare();
				break;
			case 11:
				RANSACLineFitting();
				break;
			case 12:
				houghLineDetection();
				break;
			case 13:
			{
				char fname[MAX_PATH];
				openFileDlg(fname);
				Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
				chamferDistanceTransform(src);
			}
				break;
			case 14:
				dtPatternMatching();
				break;
			case 15:
				statisticalDataAnalysis();
				break;
			case 16:
				int k;
				cin >> k;
				kMeansClustering(k);
				break;
			case 17:
				principalComponent();
				break;
			case 18:
				int m, kk;
				cout << "m = ";
				cin >> m;
				cout << "k = ";
				cin >> kk;
				kNearestClassifier(m, kk);
				break;
			case 19:
				trainNaiveBayes();
				break;
			case 20:
				batchPerceptron();
				break;
			case 21:
				adaBoost();
				break;
		}
	}
	while (op!=0);
	return 0;
}