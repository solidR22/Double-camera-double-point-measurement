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
int hmin = 0, hmax = 180, smin = 0, smax = 255, vmin = 0, vmax = 46;//ָ����ɫ��ֵ��
int g_nStructElementSize = 2;//�ṹ��Ԫ��
int g_nGaussianBlurValue = 7;//��˹�˲��˲���

/*
���ȱ궨�õ�����Ĳ���
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
//��Ӧmatlab���������궨����


//Mat distCoeffL = (Mat_<double>(5, 1) << -0.51614, 0.36098, 0.00523, -0.00225, 0.00000);
float leftDistortion[1][5] = { 0.090126, -0.053687, 0.00000,0.00000,0.00000 };
//��ӦMatlab������i����������

float leftRotation[3][3] = { 1,0,0,  0,1,0,  0,0,1 };
//�������ת����

float leftTranslation[1][3] = { 0,0,0 };
//�����ƽ������

float rightIntrinsic[3][3] = { 484.030505, 0, 317.495928,
	0, 485.558955, 227.351146,
	0, 0, 1 };
//��Ӧmatlab���������궨����

float rightDistortion[1][5] = { 0.0454926, 0.1311704, 0.00000, -0.00000, 0.00000 };
//��ӦMatlab����������������

float rightTranslation[1][3] = { -60.190623, 0.082353, -0.489894 };//Tƽ������
															 //��ӦMatlab����T����
//Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec��ת��������Ӧmatlab om����
float rightRotation[3][3] = { 1.000, -0.00004736077, 0.0061330,
	0.00005339782, 1.000000, -0.000984187,
	-0.00613301, 0.000984496, 1.000000 };//R ��ת����

Point2f xyz2uv(Point3f worldPoint, float intrinsic[3][3], float translation[1][3], float rotation[3][3]);
Point3f uv2xyz(Point2f uvLeft, Point2f uvRight);
void img2center(Mat img,Point2f &center1,Point2f &center2);

/*****������*****/
int main()
{
	VideoCapture capture(0); //��������ͷ���� ��Ҫ�����޸�
	capture.set(CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH*2);//���
	capture.set(CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);//�߶�
	capture.set(CAP_PROP_FPS, 30);//֡�� ֡/��
	cv::waitKey(1000);

	Mat img, img_left, img_right;

	while (true) {
		capture.read(img);//��ȡ��Ƶ֡
		if (img.empty()) {
			break;
		}
		//imshow("ԭͼ", img);

		//Mat leftImage, rightImage;
		img_left = img(Rect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT));//split left image
		img_right = img(Rect(IMAGE_WIDTH, 0, IMAGE_WIDTH, IMAGE_HEIGHT));//split right image

		//Point2f left_cen = img2center(img_left);
		//Point2f right_cen = img2center(img_right);
		Point2f left_cen1, right_cen1, left_cen2, right_cen2;
		img2center(img_left, left_cen1, left_cen2);
		img2center(img_right, right_cen1, right_cen2);

		//�ϲ�
		Mat img_both;
		img_both.create(IMAGE_HEIGHT, IMAGE_WIDTH * 2, img_left.type());
		Mat r1 = img_both(Rect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT));
		img_left.copyTo(r1);
		Mat r2 = img_both(Rect(IMAGE_WIDTH, 0, IMAGE_WIDTH, IMAGE_HEIGHT));
		img_right.copyTo(r2);

		imshow("�ϲ�", img_both);

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
		std::cout << "��ռ�����Ϊ:" << worldPoint1 <<"  "<<"�ҿռ�����Ϊ:"<< worldPoint2 <<"  "<<"�Ƕ�Ϊ:"<<angle<< endl;

		int c = cv::waitKey(10);
		if (c == 27) {//�˳�
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
	Mat mLeftRT = Mat(3, 4, CV_32F);//�����M����  
	hconcat(mLeftRotation, mLeftTranslation, mLeftRT);
	Mat mLeftIntrinsic = Mat(3, 3, CV_32F, leftIntrinsic);
	Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout<<"�����M���� = "<<endl<<mLeftM<<endl;  

	Mat mRightRotation = Mat(3, 3, CV_32F, rightRotation);
	Mat mRightTranslation = Mat(3, 1, CV_32F, rightTranslation);
	Mat mRightRT = Mat(3, 4, CV_32F);//�����M����  
	hconcat(mRightRotation, mRightTranslation, mRightRT);
	Mat mRightIntrinsic = Mat(3, 3, CV_32F, rightIntrinsic);
	Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"�����M���� = "<<endl<<mRightM<<endl;  

	//��С���˷�A����  
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

	//��С���˷�B����  
	Mat B = Mat(4, 1, CV_32F);
	B.at<float>(0, 0) = mLeftM.at<float>(0, 3) - uvLeft.x * mLeftM.at<float>(2, 3);
	B.at<float>(1, 0) = mLeftM.at<float>(1, 3) - uvLeft.y * mLeftM.at<float>(2, 3);
	B.at<float>(2, 0) = mRightM.at<float>(0, 3) - uvRight.x * mRightM.at<float>(2, 3);
	B.at<float>(3, 0) = mRightM.at<float>(1, 3) - uvRight.y * mRightM.at<float>(2, 3);

	Mat XYZ = Mat(3, 1, CV_32F);
	//����SVD��С���˷����XYZ  
	solve(A, B, XYZ, DECOMP_SVD);

	//cout<<"�ռ�����Ϊ = "<<endl<<XYZ<<endl;  

	//��������ϵ������  
	Point3f world;
	world.x = XYZ.at<float>(0, 0);
	world.y = XYZ.at<float>(1, 0);
	world.z = XYZ.at<float>(2, 0);

	return world;
}

//************************************  
// Description: ����������ϵ�еĵ�ͶӰ�����������������ϵ��  
// Method:    xyz2uv  
// FullName:  xyz2uv  
// Access:    public   
// Parameter: Point3f worldPoint  
// Parameter: float intrinsic[3][3]  
// Parameter: float translation[1][3]  
// Parameter: float rotation[3][3]  
// Returns:   cv::Point2f  
// Author:    С��  
// Date:      2017/01/10  
// History:  
//************************************  
Point2f xyz2uv(Point3f worldPoint, float intrinsic[3][3], float translation[1][3], float rotation[3][3])
{
	//    [fx s x0]                         [Xc]        [Xw]        [u]   1     [Xc]  
	//K = |0 fy y0|       TEMP = [R T]      |Yc| = TEMP*|Yw|        | | = ��*K *|Yc|  
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
	imshow("ԭͼ", img);
	Mat imghsv;
	cvtColor(img, imghsv, COLOR_BGR2HSV);//RGB to HSV
	imshow("hsv", imghsv);
	//��ֵ��
	Mat mask;
	inRange(imghsv, Scalar(hmin, smin, vmin), Scalar(hmax, smax, vmax), mask);//filter red color
	imshow("mask", mask);
	//��ʴ
	//#if 0
	Mat out2;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1), Point(g_nStructElementSize, g_nStructElementSize));
	//����ͼ�����ͼ���ṹԪ�أ�Խ����ʴЧ��Խ����
	erode(mask, out2, element); //erode
	imshow("��ʴ", out2);

	Mat gaussian;
	GaussianBlur(out2, gaussian, Size(g_nGaussianBlurValue * 2 + 1, g_nGaussianBlurValue * 2 + 1), 0, 0);//ģ����
	imshow("��˹�˲�", gaussian);

	vector<vector<Point> > contours;//���������ÿһ��Point�㼯����һ��������
	vector<Vec4i> hierarchy;//����Ԫ��һһ��Ӧ���ֱ��ʾ��i�������ĺ�һ��������ǰһ������������������Ƕ������������š�
	Mat imgcontours;
	Point2f center_temp1, center_temp2;
	float radius1 = 0, radius2 = 0;

	//�������
	//��һ��������ͨ��ͼ����󣬿����ǻҶ�ͼ���������õ��Ƕ�ֵͼ��һ���Ǿ���Canny��������˹�ȱ�Ե������Ӵ�����Ķ�ֵͼ��
	//���ĸ��Ǽ���ģʽ
	findContours(gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	
	//�ҵ�����Ȧ
	double maxarea = 0;
	int maxareaidx = -1;
	for (int index = contours.size() - 1; index >= 0; index--)// �ҵ�maxarea������������
	{
		double tmparea = fabs(contourArea(contours[index]));
		if (tmparea > maxarea)
		{
			maxarea = tmparea;
			maxareaidx = index;
		}
	}
	//�ҵ��ڶ����Ȧ
	double secondarea = 0;
	int secondidx = -1;
	for (int index = contours.size() - 1; index >= 0; index--)// �ҵ�maxarea������������
	{
		double tmparea = fabs(contourArea(contours[index]));
		if (tmparea > secondarea && index != maxareaidx)
		{
			secondarea = tmparea;
			secondidx = index;
		}
	}


	//�õ�������ά�㼯����СԲ
	//����Ķ�ά�㼯�������Բ�ε��������ꣻ�������СԲ�뾶

	//for (int i = 0; i <= maxareaidx; i++) {
	if (maxareaidx == -1);
	
		//return Point2f(0, 0);
	else {
		//�����һ��Բ
		minEnclosingCircle(contours[maxareaidx], center_temp1, radius1);//using index ssearching the min circle
																 //cout << contours[maxareaidx] << " " << maxareaidx << endl;
																 //if (radius > 35){
		//cout << radius << "  ";
		circle(img, static_cast<Point>(center_temp1), (int)radius1, Scalar(255, 0, 0), 3);//using contour index to drawing circle
		center1 = center_temp1;
	}
	if (secondidx == -1);
	else {
		//����ڶ���Բ
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
	//imshow("����", img);
}
