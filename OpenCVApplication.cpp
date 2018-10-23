// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <fstream>
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

		// the fastest approach using the �diblook style�
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
	char filename[100] = "PointsLeastSquares/points2.txt";
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
	float t = 10, p = 0.99, q = 0.8, s = 2, N;
	
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
		printf(" 10 - Least Mean Square\n");
		printf(" 11 - RANSAC Line Fitting\n");
		printf(" 12 - Hough Line Detection\n");
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
		}
	}
	while (op!=0);
	return 0;
}