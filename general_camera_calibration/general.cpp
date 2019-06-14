//opencv2.4.9 vs2012
#include <opencv2\opencv.hpp>
#include <fstream>
using namespace std;
using namespace cv;

int main()
{
	double time0 = static_cast<double>(getTickCount());
	ofstream fout("images\\caliberation_result.txt");  /**    ���涨�������ļ�     **/

											   /************************************************************************
											   ��ȡÿһ��ͼ�񣬴�����ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��
											   *************************************************************************/
	cout << "��ʼ��ȡ�ǵ㡭����������" << endl;
	int image_count = 15;                    /****    ͼ������     ****/
	Size image_size;                         /****     ͼ��ĳߴ�      ****/
	Size board_size = Size(10, 7);            /****    �������ÿ�С��еĽǵ���       ****/
	vector<Point2f> corners;                  /****    ����ÿ��ͼ���ϼ�⵽�Ľǵ�       ****/

	vector<Point2f> corners1;                  /****    ����ÿ��ͼ���ϼ�⵽�Ľǵ�       ****/


	vector<vector<Point2f>>  corners_Seq;    /****  �����⵽�����нǵ�       ****/
	vector<Mat>  image_Seq;


	int count = 0;
	for (int i = 0; i != image_count; i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		string imageFileName;
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += ".bmp";
		Mat image = imread("images\\" + imageFileName);
		image_size = image.size();
		//image_size = Size(image.cols , image.rows);
		/* ��ȡ�ǵ� */
		Mat imageGray;
		cvtColor(image, imageGray, CV_RGB2GRAY);
	/*	bool patternfound = findChessboardCorners(image, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
			CALIB_CB_FAST_CHECK);*/

		bool patternfound = findChessboardCorners(image, board_size, corners, 0);
		if (!patternfound)
		{
			cout << "can not find chessboard corners!\n";
			continue;
			exit(1);
		}
		else
		{
			/* �����ؾ�ȷ�� */
			cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			/* ���Ƽ�⵽�Ľǵ㲢���� */
			Mat imageTemp = image.clone();
			for (int j = 0; j < corners.size(); j++)
			{
				circle(imageTemp, corners[j], 10, Scalar(0, 0, 255), 2, 8, 0);
			}
			string imageFileName;
			std::stringstream StrStm;
			StrStm << i + 1;
			StrStm >> imageFileName;
			imageFileName += "_corner.jpg";
			imwrite("images\\" + imageFileName, imageTemp);
			cout << "Frame corner#" << i + 1 << "...end" << endl;

			count = count + corners.size();
			corners_Seq.push_back(corners);
		}
		image_Seq.push_back(image);
	}
	cout << "�ǵ���ȡ��ɣ�\n";
	/************************************************************************
	���������
	*************************************************************************/
	cout << "��ʼ���ꡭ����������" << endl;
	Size square_size = Size(2500, 2500);                                      /**** ʵ�ʲ����õ��Ķ������ÿ�����̸�Ĵ�С   ****/
	vector<vector<Point3f>>  object_Points;                                      /****  ���涨����Ͻǵ����ά����   ****/


	Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0));          /*****   ������ȡ�����нǵ�   *****/
	vector<int>  point_counts;                                          /*****    ÿ��ͼ���нǵ������    ****/
	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));                /*****    ������ڲ�������    ****/
	Mat distortion_coeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));            /* �������4������ϵ����k1,k2,p1,p2 */
	vector<cv::Mat> rotation_vectors;                                      /* ÿ��ͼ�����ת���� */
	vector<cv::Mat> translation_vectors;                                  /* ÿ��ͼ���ƽ������ */

																		  /* ��ʼ��������Ͻǵ����ά���� */
	for (int t = 0;t<image_count;t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0;i<board_size.height;i++)
		{
			for (int j = 0;j<board_size.width;j++)
			{
				/* ���趨��������������ϵ��z=0��ƽ���� */
				Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);
	}

	/* ��ʼ��ÿ��ͼ���еĽǵ��������������Ǽ���ÿ��ͼ���ж����Կ��������Ķ���� */
	for (int i = 0; i< image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	/* ��ʼ���� */
	calibrateCamera(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, 0);
	cout << "������ɣ�\n";

	/************************************************************************
	�Զ�������������
	*************************************************************************/
	cout << "��ʼ���۶�����������������" << endl;
	double total_err = 0.0;                   /* ����ͼ���ƽ�������ܺ� */
	double err = 0.0;                        /* ÿ��ͼ���ƽ����� */
	vector<Point2f>  image_points2;             /****   �������¼���õ���ͶӰ��    ****/

	cout << "ÿ��ͼ��Ķ�����" << endl;
	cout << "ÿ��ͼ��Ķ�����" << endl << endl;
	for (int i = 0; i<image_count; i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];
		/****    ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ��     ****/
		projectPoints(tempPointSet, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs, image_points2);
		/* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
		vector<Point2f> tempImagePoint = corners_Seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (size_t i = 0; i != tempImagePoint.size(); i++)
		{
			image_points2Mat.at<Vec2f>(0, i) = Vec2f(image_points2[i].x, image_points2[i].y);
			tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		cout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
	}
	cout << "����ƽ����" << total_err / image_count << "����" << endl;
	fout << "����ƽ����" << total_err / image_count << "����" << endl << endl;
	cout << "������ɣ�" << endl;

	/************************************************************************
	���涨����
	*************************************************************************/
	cout << "��ʼ���涨����������������" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* ����ÿ��ͼ�����ת���� */

	fout << "����ڲ�������" << endl;
	fout << intrinsic_matrix << endl;
	fout << "����ϵ����\n";
	fout << distortion_coeffs << endl;
	for (int i = 0; i<image_count; i++)
	{
		fout << "��" << i + 1 << "��ͼ�����ת������" << endl;
		fout << rotation_vectors[i] << endl;

		/* ����ת����ת��Ϊ���Ӧ����ת���� */
		Rodrigues(rotation_vectors[i], rotation_matrix);
		fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		fout << rotation_matrix << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		fout << translation_vectors[i] << endl;
	}
	cout << "��ɱ���" << endl;
	fout << endl;


	/************************************************************************
	��ʾ������
	*************************************************************************/
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "�������ͼ��" << endl;
	for (int i = 0; i != image_count; i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
		initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, image_size, 1, image_size, 0), image_size, CV_32FC1, mapx, mapy);
		Mat t = image_Seq[i].clone();
		cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR);
		string imageFileName;
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += "_d.jpg";
		imwrite("images\\" + imageFileName, t);
	}
	cout << "�������" << endl;

	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << "�궨��ʱ:" << time0 << "��" << endl;

	/************************************************************************
	����һ��ͼƬ
	*************************************************************************/
	double time1 = static_cast<double>(getTickCount());
	if (1)
	{
		cout << "TestImage ..." << endl;
		Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
		Mat testImage = imread("images\\FINAL.jpg", 1);


		//image_size.height = MAX(image_size.height,image_size.width);
		//image_size.width = MAX(image_size.height,image_size.width);
		initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, image_size, 1, image_size, 0), image_size, CV_32FC1, mapx, mapy);
		Mat t = testImage.clone();
		cv::remap(testImage, t, mapx, mapy, INTER_LINEAR);

		imwrite("images\\TestOutput.jpg", t);

		cout << "�������" << endl;
	}
	time1 = ((double)getTickCount() - time1) / getTickFrequency();
	cout << "У����ʱ:" << time1 << "��" << endl;

	getchar();
	return 0;
}