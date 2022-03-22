#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

using namespace cv;
using namespace std;

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
int hmin = 0, hmax = 180, smin = 0, smax = 255, vmin = 0, vmax = 46;//指定颜色二值化
int g_nStructElementSize = 2;//结构化元素
int g_nGaussianBlurValue = 7;//高斯滤波核参数

/*
事先标定好的相机的参数
fx 0 cx
0 fy cy
0 0  1
*/
//Mat cameraMatrixL = (Mat_<double>(3, 3) << 682.55880, 0, 384.13666,
//	0, 682.24569, 311.19558,
//	0, 0, 1);
float leftIntrinsic[3][3] = { 480.0957, 0, 345.0930,
	0, 481.6971, 233.2975,
	0, 0, 1 };
//对应matlab里的左相机标定矩阵


//Mat distCoeffL = (Mat_<double>(5, 1) << -0.51614, 0.36098, 0.00523, -0.00225, 0.00000);
float leftDistortion[1][5] = { 0.090126, -0.053687, 0.00000,0.00000,0.00000 };
//对应Matlab所得左i相机畸变参数

float leftRotation[3][3] = { 1,0,0,  0,1,0,  0,0,1 };
//左相机旋转矩阵

float leftTranslation[1][3] = { 0,0,0 };
//左相机平移向量

float rightIntrinsic[3][3] = { 484.030505, 0, 317.495928,
	0, 485.558955, 227.351146,
	0, 0, 1 };
//对应matlab里的右相机标定矩阵

float rightDistortion[1][5] = { 0.0454926, 0.1311704, 0.00000, -0.00000, 0.00000 };
//对应Matlab所得右相机畸变参数

float rightTranslation[1][3] = { -60.190623, 0.082353, -0.489894 };//T平移向量
															 //对应Matlab所得T参数
//Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量，对应matlab om参数
float rightRotation[3][3] = { 1.000, -0.00004736077, 0.0061330,
	0.00005339782, 1.000000, -0.000984187,
	-0.00613301, 0.000984496, 1.000000 };//R 旋转矩阵

Point2f xyz2uv(Point3f worldPoint, float intrinsic[3][3], float translation[1][3], float rotation[3][3]);
Point3f uv2xyz(Point2f uvLeft, Point2f uvRight);
void img2center(Mat img,Point2f &center1,Point2f &center2);

/*****主函数*****/
int main()
{
	VideoCapture capture(0); //设置摄像头参数 不要随意修改
	capture.set(CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH*2);//宽度
	capture.set(CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);//高度
	capture.set(CAP_PROP_FPS, 30);//帧率 帧/秒
	cv::waitKey(1000);

	Mat img, img_left, img_right;

	while (true) {
		capture.read(img);//获取视频帧
		if (img.empty()) {
			break;
		}
		//imshow("原图", img);

		//Mat leftImage, rightImage;
		img_left = img(Rect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT));//split left image
		img_right = img(Rect(IMAGE_WIDTH, 0, IMAGE_WIDTH, IMAGE_HEIGHT));//split right image

		//Point2f left_cen = img2center(img_left);
		//Point2f right_cen = img2center(img_right);
		Point2f left_cen1, right_cen1, left_cen2, right_cen2;
		img2center(img_left, left_cen1, left_cen2);
		img2center(img_right, right_cen1, right_cen2);

		//合并
		Mat img_both;
		img_both.create(IMAGE_HEIGHT, IMAGE_WIDTH * 2, img_left.type());
		Mat r1 = img_both(Rect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT));
		img_left.copyTo(r1);
		Mat r2 = img_both(Rect(IMAGE_WIDTH, 0, IMAGE_WIDTH, IMAGE_HEIGHT));
		img_right.copyTo(r2);

		imshow("合并", img_both);

		Point2f leftPoint1, leftPoint2, rightPoint1, rightPoint2;
		if (left_cen1.x < left_cen2.x)
		{
			leftPoint1 = left_cen1;
			leftPoint2 = left_cen2;
		}
		else
		{
			leftPoint1 = left_cen2;
			leftPoint2 = left_cen1;
		}
		if (right_cen1.x < right_cen2.x)
		{
			rightPoint1 = right_cen1;
			rightPoint2 = right_cen2;
		}
		else
		{
			rightPoint1 = right_cen2;
			rightPoint2 = right_cen1;
		}

		float angle = 0;
		angle = atan((leftPoint2.y - leftPoint1.y) / (leftPoint2.x - leftPoint1.x)) / 3.1415927 * 360;

		Point3f worldPoint1, worldPoint2;
		worldPoint1 = uv2xyz(leftPoint1, rightPoint1);
		worldPoint2 = uv2xyz(leftPoint2, rightPoint2);
		std::cout << "左空间坐标为:" << worldPoint1 <<"  "<<"右空间坐标为:"<< worldPoint2 <<"  "<<"角度为:"<<angle<< endl;

		int c = cv::waitKey(10);
		if (c == 27) {//退出
			break;
		}
		
	}
	
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

Point3f uv2xyz(Point2f uvLeft, Point2f uvRight)
{
	//  [u1]      |X|                     [u2]      |X|  
	//Z*|v1| = Ml*|Y|                   Z*|v2| = Mr*|Y|  
	//  [ 1]      |Z|                     [ 1]      |Z|  
	//            |1|                               |1|  
	Mat mLeftRotation = Mat(3, 3, CV_32F, leftRotation);
	Mat mLeftTranslation = Mat(3, 1, CV_32F, leftTranslation);
	Mat mLeftRT = Mat(3, 4, CV_32F);//左相机M矩阵  
	hconcat(mLeftRotation, mLeftTranslation, mLeftRT);
	Mat mLeftIntrinsic = Mat(3, 3, CV_32F, leftIntrinsic);
	Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout<<"左相机M矩阵 = "<<endl<<mLeftM<<endl;  

	Mat mRightRotation = Mat(3, 3, CV_32F, rightRotation);
	Mat mRightTranslation = Mat(3, 1, CV_32F, rightTranslation);
	Mat mRightRT = Mat(3, 4, CV_32F);//右相机M矩阵  
	hconcat(mRightRotation, mRightTranslation, mRightRT);
	Mat mRightIntrinsic = Mat(3, 3, CV_32F, rightIntrinsic);
	Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"右相机M矩阵 = "<<endl<<mRightM<<endl;  

	//最小二乘法A矩阵  
	Mat A = Mat(4, 3, CV_32F);
	A.at<float>(0, 0) = uvLeft.x * mLeftM.at<float>(2, 0) - mLeftM.at<float>(0, 0);
	A.at<float>(0, 1) = uvLeft.x * mLeftM.at<float>(2, 1) - mLeftM.at<float>(0, 1);
	A.at<float>(0, 2) = uvLeft.x * mLeftM.at<float>(2, 2) - mLeftM.at<float>(0, 2);

	A.at<float>(1, 0) = uvLeft.y * mLeftM.at<float>(2, 0) - mLeftM.at<float>(1, 0);
	A.at<float>(1, 1) = uvLeft.y * mLeftM.at<float>(2, 1) - mLeftM.at<float>(1, 1);
	A.at<float>(1, 2) = uvLeft.y * mLeftM.at<float>(2, 2) - mLeftM.at<float>(1, 2);

	A.at<float>(2, 0) = uvRight.x * mRightM.at<float>(2, 0) - mRightM.at<float>(0, 0);
	A.at<float>(2, 1) = uvRight.x * mRightM.at<float>(2, 1) - mRightM.at<float>(0, 1);
	A.at<float>(2, 2) = uvRight.x * mRightM.at<float>(2, 2) - mRightM.at<float>(0, 2);

	A.at<float>(3, 0) = uvRight.y * mRightM.at<float>(2, 0) - mRightM.at<float>(1, 0);
	A.at<float>(3, 1) = uvRight.y * mRightM.at<float>(2, 1) - mRightM.at<float>(1, 1);
	A.at<float>(3, 2) = uvRight.y * mRightM.at<float>(2, 2) - mRightM.at<float>(1, 2);

	//最小二乘法B矩阵  
	Mat B = Mat(4, 1, CV_32F);
	B.at<float>(0, 0) = mLeftM.at<float>(0, 3) - uvLeft.x * mLeftM.at<float>(2, 3);
	B.at<float>(1, 0) = mLeftM.at<float>(1, 3) - uvLeft.y * mLeftM.at<float>(2, 3);
	B.at<float>(2, 0) = mRightM.at<float>(0, 3) - uvRight.x * mRightM.at<float>(2, 3);
	B.at<float>(3, 0) = mRightM.at<float>(1, 3) - uvRight.y * mRightM.at<float>(2, 3);

	Mat XYZ = Mat(3, 1, CV_32F);
	//采用SVD最小二乘法求解XYZ  
	solve(A, B, XYZ, DECOMP_SVD);

	//cout<<"空间坐标为 = "<<endl<<XYZ<<endl;  

	//世界坐标系中坐标  
	Point3f world;
	world.x = XYZ.at<float>(0, 0);
	world.y = XYZ.at<float>(1, 0);
	world.z = XYZ.at<float>(2, 0);

	return world;
}

//************************************  
// Description: 将世界坐标系中的点投影到左右相机成像坐标系中  
// Method:    xyz2uv  
// FullName:  xyz2uv  
// Access:    public   
// Parameter: Point3f worldPoint  
// Parameter: float intrinsic[3][3]  
// Parameter: float translation[1][3]  
// Parameter: float rotation[3][3]  
// Returns:   cv::Point2f  
// Author:    小白  
// Date:      2017/01/10  
// History:  
//************************************  
Point2f xyz2uv(Point3f worldPoint, float intrinsic[3][3], float translation[1][3], float rotation[3][3])
{
	//    [fx s x0]                         [Xc]        [Xw]        [u]   1     [Xc]  
	//K = |0 fy y0|       TEMP = [R T]      |Yc| = TEMP*|Yw|        | | = —*K *|Yc|  
	//    [ 0 0 1 ]                         [Zc]        |Zw|        [v]   Zc    [Zc]  
	//                                                  [1 ]  
	Point3f c;
	c.x = rotation[0][0] * worldPoint.x + rotation[0][1] * worldPoint.y + rotation[0][2] * worldPoint.z + translation[0][0] * 1;
	c.y = rotation[1][0] * worldPoint.x + rotation[1][1] * worldPoint.y + rotation[1][2] * worldPoint.z + translation[0][1] * 1;
	c.z = rotation[2][0] * worldPoint.x + rotation[2][1] * worldPoint.y + rotation[2][2] * worldPoint.z + translation[0][2] * 1;

	Point2f uv;
	uv.x = (intrinsic[0][0] * c.x + intrinsic[0][1] * c.y + intrinsic[0][2] * c.z) / c.z;
	uv.y = (intrinsic[1][0] * c.x + intrinsic[1][1] * c.y + intrinsic[1][2] * c.z) / c.z;

	return uv;
}

void img2center(Mat img, Point2f &center1, Point2f &center2)
{
	imshow("原图", img);
	Mat imghsv;
	cvtColor(img, imghsv, COLOR_BGR2HSV);//RGB to HSV
	imshow("hsv", imghsv);
	//二值化
	Mat mask;
	inRange(imghsv, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), mask);//filter red color
	imshow("mask", mask);
	//腐蚀
	//#if 0
	Mat out2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));
	//输入图；输出图；结构元素，越大侵蚀效果越明显
	erode(mask, out2, element); //erode
	imshow("腐蚀", out2);

	Mat gaussian;
	GaussianBlur(out2, gaussian, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);//模糊化
	imshow("高斯滤波", gaussian);

	vector<vector<Point> > contours;//经过处理后，每一组Point点集就是一个轮廓。
	vector<Vec4i> hierarchy;//与上元素一一对应，分别表示第i个轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号。
	Mat imgcontours;
	Point2f center_temp1, center_temp2;
	float radius1 = 0, radius2 = 0;

	//检测轮廓
	//第一个参数单通道图像矩阵，可以是灰度图，但更常用的是二值图像，一般是经过Canny、拉普拉斯等边缘检测算子处理过的二值图像；
	//第四个是检索模式
	findContours(gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	
	//找到最大的圈
	double maxarea = 0;
	int maxareaidx = -1;
	for (int index = contours.size() - 1; index >= 0; index--)// 找到maxarea返回轮廓索引
	{
		double tmparea = fabs(contourArea(contours[index]));
		if (tmparea > maxarea)
		{
			maxarea = tmparea;
			maxareaidx = index;
		}
	}
	//找到第二大的圈
	double secondarea = 0;
	int secondidx = -1;
	for (int index = contours.size() - 1; index >= 0; index--)// 找到maxarea返回轮廓索引
	{
		double tmparea = fabs(contourArea(contours[index]));
		if (tmparea > secondarea && index != maxareaidx)
		{
			secondarea = tmparea;
			secondidx = index;
		}
	}


	//得到包含二维点集的最小圆
	//输入的二维点集；输出的圆形的中心坐标；输出的最小圆半径

	//for (int i = 0; i <= maxareaidx; i++) {
	if (maxareaidx == -1);
	
		//return Point2f(0, 0);
	else {
		//输出第一个圆
		minEnclosingCircle(contours[maxareaidx], center_temp1, radius1);//using index ssearching the min circle
																 //cout << contours[maxareaidx] << " " << maxareaidx << endl;
																 //if (radius > 35){
		//cout << radius << "  ";
		circle(img, static_cast<Point>(center_temp1), (int)radius1, Scalar(255, 0, 0), 3);//using contour index to drawing circle
		center1 = center_temp1;
	}
	if (secondidx == -1);
	else {
		//输出第二个圆
		minEnclosingCircle(contours[secondidx], center_temp2, radius2);//using index ssearching the min circle
																 //cout << contours[maxareaidx] << " " << maxareaidx << endl;
																 //if (radius > 35){
																 //cout << radius << "  ";
		circle(img, static_cast<Point>(center_temp2), (int)radius2, Scalar(255, 0, 0), 3);//using contour index to drawing circle
		center2 = center_temp2;
	}															   //cout << center << endl;
		//return center;
		//}

																			   //circle(img, static_cast<Point>(center), 5, Scalar(255, 0, 0), -3);
																			   //}
	//imshow("轮廓", img);
}
