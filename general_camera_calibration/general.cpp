//opencv2.4.9 vs2012
#include <opencv2\opencv.hpp>
#include <fstream>
using namespace std;
using namespace cv;

int main()
{
	double time0 = static_cast<double>(getTickCount());
	ofstream fout("images\\caliberation_result.txt");  /**    保存定标结果的文件     **/

											   /************************************************************************
											   读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
											   *************************************************************************/
	cout << "开始提取角点………………" << endl;
	int image_count = 15;                    /****    图像数量     ****/
	Size image_size;                         /****     图像的尺寸      ****/
	Size board_size = Size(10, 7);            /****    定标板上每行、列的角点数       ****/
	vector<Point2f> corners;                  /****    缓存每幅图像上检测到的角点       ****/

	vector<Point2f> corners1;                  /****    缓存每幅图像上检测到的角点       ****/


	vector<vector<Point2f>>  corners_Seq;    /****  保存检测到的所有角点       ****/
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
		/* 提取角点 */
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
			/* 亚像素精确化 */
			cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			/* 绘制检测到的角点并保存 */
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
	cout << "角点提取完成！\n";
	/************************************************************************
	摄像机定标
	*************************************************************************/
	cout << "开始定标………………" << endl;
	Size square_size = Size(2500, 2500);                                      /**** 实际测量得到的定标板上每个棋盘格的大小   ****/
	vector<vector<Point3f>>  object_Points;                                      /****  保存定标板上角点的三维坐标   ****/


	Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0));          /*****   保存提取的所有角点   *****/
	vector<int>  point_counts;                                          /*****    每幅图像中角点的数量    ****/
	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));                /*****    摄像机内参数矩阵    ****/
	Mat distortion_coeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));            /* 摄像机的4个畸变系数：k1,k2,p1,p2 */
	vector<cv::Mat> rotation_vectors;                                      /* 每幅图像的旋转向量 */
	vector<cv::Mat> translation_vectors;                                  /* 每幅图像的平移向量 */

																		  /* 初始化定标板上角点的三维坐标 */
	for (int t = 0;t<image_count;t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0;i<board_size.height;i++)
		{
			for (int j = 0;j<board_size.width;j++)
			{
				/* 假设定标板放在世界坐标系中z=0的平面上 */
				Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);
	}

	/* 初始化每幅图像中的角点数量，这里我们假设每幅图像中都可以看到完整的定标板 */
	for (int i = 0; i< image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	/* 开始定标 */
	calibrateCamera(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, 0);
	cout << "定标完成！\n";

	/************************************************************************
	对定标结果进行评价
	*************************************************************************/
	cout << "开始评价定标结果………………" << endl;
	double total_err = 0.0;                   /* 所有图像的平均误差的总和 */
	double err = 0.0;                        /* 每幅图像的平均误差 */
	vector<Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/

	cout << "每幅图像的定标误差：" << endl;
	cout << "每幅图像的定标误差：" << endl << endl;
	for (int i = 0; i<image_count; i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];
		/****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
		projectPoints(tempPointSet, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs, image_points2);
		/* 计算新的投影点和旧的投影点之间的误差*/
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
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << total_err / image_count << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
	cout << "评价完成！" << endl;

	/************************************************************************
	保存定标结果
	*************************************************************************/
	cout << "开始保存定标结果………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */

	fout << "相机内参数矩阵：" << endl;
	fout << intrinsic_matrix << endl;
	fout << "畸变系数：\n";
	fout << distortion_coeffs << endl;
	for (int i = 0; i<image_count; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << rotation_vectors[i] << endl;

		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(rotation_vectors[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << translation_vectors[i] << endl;
	}
	cout << "完成保存" << endl;
	fout << endl;


	/************************************************************************
	显示定标结果
	*************************************************************************/
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "保存矫正图像" << endl;
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
	cout << "保存结束" << endl;

	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << "标定用时:" << time0 << "秒" << endl;

	/************************************************************************
	测试一张图片
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

		cout << "保存结束" << endl;
	}
	time1 = ((double)getTickCount() - time1) / getTickFrequency();
	cout << "校正用时:" << time1 << "秒" << endl;

	getchar();
	return 0;
}