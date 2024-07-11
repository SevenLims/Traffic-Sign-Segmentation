#include	<opencv2/opencv.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<opencv2/imgcodecs.hpp>
#include	<opencv2/imgproc.hpp>
#include	<iostream>
#include	<stdlib.h>
#include	<string>
#include	<filesystem>
#include	"Supp.h"

using namespace cv;
using namespace std;

Mat Red_HSV(Mat srcI) 
{
	Mat lowerRed_range;
	Mat upperRed_range;
	Mat redMask;

	cvtColor(srcI, srcI, COLOR_BGR2HSV);
	inRange(srcI, Scalar(0, 90, 72), Scalar(7, 255, 255), lowerRed_range);
	inRange(srcI, Scalar(173, 80, 72), Scalar(179, 255, 255), upperRed_range);
	addWeighted(lowerRed_range, 1.0, upperRed_range, 1.0, 0.0, redMask);

	return redMask;
}

Mat Blue_HSV(Mat srcI) 
{
	Mat blueMask;

	cvtColor(srcI, srcI, COLOR_BGR2HSV);
	Scalar minBlue(100, 135, 50), maxBlue(130, 255, 255);
	inRange(srcI, minBlue, maxBlue, blueMask);

	return blueMask;
}

Mat Yellow_HSV(Mat srcI) 
{
	Mat yellowMask;
	Mat orange_yellow_range;
	Mat green_yellow_range;

	cvtColor(srcI, srcI, COLOR_BGR2HSV);
	inRange(srcI, Scalar(8, 70, 40), Scalar(32.9, 255, 255), orange_yellow_range);
	inRange(srcI, Scalar(33, 150, 100), Scalar(38, 250, 200), green_yellow_range);
	addWeighted(orange_yellow_range, 1.0, green_yellow_range, 0.003, 0.0, yellowMask);

	return yellowMask;
}

vector<Point> contoursConvexHull(vector<vector<Point> > contours)
{
	vector<Point> output;
	vector<Point> pts;
	int index = 0, max = contours[0].size();

	for (size_t i = 0; i < contours.size(); i++)
		for (size_t j = 0; j < contours[i].size(); j++)
			pts.push_back(contours[i][j]);
	convexHull(pts, output);
	return output;
}

int main(int argc, char** argv) 
{
	string		windowName, select = "NaN";
	bool		con;
	Mat			mask, canvasColor, canvasGray, srcI, srcMask, srcMask2, srcOri;
	char		str[256];
	Point2i		center;
	vector<Scalar>	colors;
	int const	MAXfPt = 200;
	int			t1, t2, t3, t4, thresh = 100;
	RNG			rng(0);
	String		imgPattern;
	vector<string>	imageNames;



	// get MAXfPt random but brighter colours for drawing later
	for (int i = 0; i < MAXfPt; i++) 
	{
		for (;;) 
		{
			t1 = rng.uniform(0, 255); // blue
			t2 = rng.uniform(0, 255); // green
			t3 = rng.uniform(0, 255); // red
			t4 = t1 + t2 + t3;
			// get random colours that are not too dim
			if (t4 > 255) break;
		}
		colors.push_back(Scalar(t1, t2, t3));
	}

	//user input to select traffic sign colour
	do 
	{
		cout << " Please enter [R]ed or [B]lue, [Y]ellow : ";
		cin >> select;
		con = false;
		for (int i = 0; i < select.length(); i++)
			select[i] = toupper(select[i]);

		if (select == "R")
			imgPattern = "Inputs/Traffic signs/Red Colour/*.png";

		else if (select == "B")
			imgPattern = "Inputs/Traffic signs/Blue Colour/*.png";

		else if (select == "Y")
			imgPattern = "Inputs/Traffic signs/Yellow Colour/*.png";

		else 
		{
			cout << "Please enter R or B or Y only. Try it again." << endl;
			con = true;
		}
	} while (con);


	// Collect all image names satisfying the image name pattern
	cv::glob(imgPattern, imageNames, true);
	for (size_t i = 0; i < imageNames.size(); ++i) 
	{
		srcI = imread(imageNames[i]);

		if (srcI.empty()) 
		{
			cout << "cannot open image for reading" << endl;
			return -1;
		}

		resize(srcI, srcI, Size(200, 200));

		// Open 2 large windows to diaplay the results. One gives the detail. Other give only the results
		int const	noOfImagePerCol = 1, noOfImagePerRow = 6;
		Mat			detailResultWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
		createWindowPartition(srcI, detailResultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

		putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[1], "HSV filter", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[2], "Contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[3], "Biggest convex contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[4], "Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend[5], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

		int const	noOfImagePerCol2 = 1, noOfImagePerRow2 = 2;
		Mat			resultWin, win2[noOfImagePerRow2 * noOfImagePerCol2], legend2[noOfImagePerRow2 * noOfImagePerCol2];
		createWindowPartition(srcI, resultWin, win2, legend2, noOfImagePerCol2, noOfImagePerRow2);

		putText(legend2[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
		putText(legend2[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

		srcI.copyTo(win[0]);
		srcI.copyTo(win2[0]);

		// create canvases for drawing
		// canvasColor: use different color to draw each contour in win[2]
		// canvaxGray: used to check area inside a contour
		canvasColor.create(srcI.rows, srcI.cols, CV_8UC3);
		canvasGray.create(srcI.rows, srcI.cols, CV_8U);
		canvasColor = Scalar(0, 0, 0);

		//  image preprocessing
		//GaussianBlur(srcI, srcI, Size(5, 5), 0);
		//blur(srcI, srcMask, Size(5, 5));
		srcI.copyTo(srcMask);
		bilateralFilter(srcI, srcMask, 5, 100, 100);

		//create mask with HSV thresholding for each colour
		if (select == "R") 
		{
			mask = Red_HSV(srcMask);

			dilate(mask, mask, Mat());
		}

		else if (select == "B") 
		{
			mask = Blue_HSV(srcMask);

			dilate(mask, mask, Mat());
		}

		else if (select == "Y") 
		{
			srcMask.copyTo(srcMask2);

			//for removing noise purposes
			bilateralFilter(srcMask, srcMask2, 8, 220, 220);

			mask = Yellow_HSV(srcMask2);

			dilate(mask, mask, Mat());
		}


		cvtColor(mask, win[1], COLOR_GRAY2BGR); // show result of colour

		// get contours of the selected colour regions
		vector<vector<Point> >	contours, biggest_contour;
		findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		int			index, max = 0;

		for (int i = 0; i < contours.size(); i++)  // We could have more than one sign in image
		{
			canvasGray = 0;
			double newArea = contourArea(contours.at(i));
			if (max < newArea) // Find the largest contour as sign boundary
			{ 
				max = newArea;
				index = i;
			}

			drawContours(canvasColor, contours, i, colors[i]); // draw boundaries
			drawContours(canvasGray, contours, i, 255);

			// The code below compute the center of the region
			Moments M = moments(canvasGray);
			center.x = M.m10 / M.m00;
			center.y = M.m01 / M.m00;

			// If found center is not inside the contour, the result will return inaccurate
			floodFill(canvasGray, center, 255); // fill inside sign boundary
			if (countNonZero(canvasGray) > 30) // Check if sign too small
			{ 
				sprintf_s(str, "Mask %d (area > 30)", i);
				imshow(str, canvasGray);
			}
		}
		canvasColor.copyTo(win[2]);

		//  use convexhull to join points from biggest contour to include possible parts
		Mat ctr;
		ctr.create(srcI.rows, srcI.cols, CV_8U);
		ctr = 0;

		biggest_contour.push_back(contours[index]);

		vector<Point> ConvexHullPoints = contoursConvexHull(biggest_contour);
		polylines(ctr, ConvexHullPoints, true, Scalar(255, 255, 255), 1); //ctr is the contour using convexhull

		cvtColor(ctr, win[3], COLOR_GRAY2BGR); //contour drawn using convexHull

		Moments M = moments(ctr);
		center.x = M.m10 / M.m00;
		center.y = M.m01 / M.m00;

		// generate mask of the sign
		floodFill(ctr, center, 255); // fill inside sign boundary
		cvtColor(ctr, ctr, COLOR_GRAY2BGR);
		ctr.copyTo(win[4]);

		// use the mask to segment the colour portion from image
		canvasColor = ctr & srcI;
		canvasColor.copyTo(win[5]);
		canvasColor.copyTo(win2[1]);

		windowName = "Segmentation of " + imageNames[i] + " (detail)";
		imshow(windowName, detailResultWin);
		imshow("Traffic sign segmentation", resultWin);

		waitKey();
		destroyAllWindows();
	}
	return 0;

}