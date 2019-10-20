#include "visual_odometry.h"

#include <string>
#include <fstream>
#include<stdexcept>
#include "Kinect.h"
#include "Head.h"
#include <math.h>
#include <limits>
#include <strsafe.h>

const int kMinNumFeature = 150;
static const DWORD c_AppRunTime = 5 * 60;//程序运行时间(s)，设置5*60表示运行5分钟后程序自动关闭
static const float c_HandSize = 30.0f;

VisualOdometry::VisualOdometry()

{
	focal_ = 1051.7826;
	pp_ = cv::Point2d(955.7709, 535.5136);

	frame_stage_ = STAGE_FIRST_FRAME;


	m_pKinectSensor = NULL;
	m_pCoordinateMapper = NULL;
	m_pMultiSourceFrameReader = NULL;
	m_pOutputRGBX = NULL;
	m_pBackgroundRGBX = NULL;
	m_pColorRGBX = NULL;
	m_pCameraCoordinates = NULL;
	// create heap storage for composite image pixel data in RGBX format
	m_pOutputRGBX = new RGBQUAD[cColorWidth * cColorHeight];

	// create heap storage for background image pixel data in RGBX format
	m_pBackgroundRGBX = new RGBQUAD[cColorWidth * cColorHeight];

	// create heap storage for color pixel data in RGBX format
	m_pColorRGBX = new RGBQUAD[cColorWidth * cColorHeight];

	// create heap storage for the coorinate mapping from color to depth
	m_pCameraCoordinates = new CameraSpacePoint[cColorWidth * cColorHeight];

	m_pDepthRGBX = new RGBQUAD[cDepthWidth * cDepthHeight];// create heap storage for color pixel data in RGBX format

	//初始化OpenCV数组
	m_Depth.create(cDepthHeight, cDepthWidth, CV_16UC1);
	m_Color.create(cColorHeight, cColorWidth, CV_8UC4);

	////初始化与g2o有关的变量
	//linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
	//block_solver = new g2o::BlockSolver_6_3(linearSolver);
	//algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);
	//optimizer.setAlgorithm(algorithm);
	//// 不要输出调试信息
	//optimizer.setVerbose(false);
	//// 向optimizer增加第一个顶点
	//g2o::VertexSE3* v = new g2o::VertexSE3();
	//v->setId(1);
	//v->setEstimate(Eigen::Isometry3d::Identity()); //估计为单位矩阵
	//v->setFixed(true); //第一个顶点固定，不用优化
	//optimizer.addVertex(v);
}
VisualOdometry::~VisualOdometry()
{
	SAFE_RELEASE_VEC(m_pOutputRGBX);
	SAFE_RELEASE_VEC(m_pBackgroundRGBX);
	SAFE_RELEASE_VEC(m_pColorRGBX);
	SAFE_RELEASE_VEC(m_pCameraCoordinates);
	SAFE_RELEASE_VEC(m_pDepthRGBX);

	// done with frame reader
	SafeRelease(m_pMultiSourceFrameReader);

	// done with coordinate mapper
	SafeRelease(m_pCoordinateMapper);

	// close the Kinect Sensor
	if (m_pKinectSensor)
	{
		m_pKinectSensor->Close();
	}

	SafeRelease(m_pKinectSensor);
}
void VisualOdometry::addImage(const cv::Mat& img, int frame_id)
{

	if (img.empty() )//|| img.type() != CV_8UC1
		throw std::runtime_error("Frame: provided image is not grayscale");

	new_frame_ = img;
	//new_frame_ = sharpenImage1(new_frame_);
    bool res = true;
	if (frame_stage_ == STAGE_DEFAULT_FRAME)
		res = processFrame(frame_id);
	else if (frame_stage_ == STAGE_SECOND_FRAME)
		res = processSecondFrame(frame_id);
	else if (frame_stage_ == STAGE_FIRST_FRAME)
		res = processFirstFrame();
	last_frame_ = new_frame_;

}
bool VisualOdometry::processFirstFrame()
{
	
	if (ifORB)
	{
		featureDetectionORB(new_frame_, kp_ref_, orb_descriptor_ref_, px_ref_);
	}
	if (iffast)
	{
		featureDetection(new_frame_, px_ref_);
	}
	if (ifdetectline)
	{
		featureDetectionLines(new_frame_, px_ref_);
	}
    
	if (px_ref_.size() > 0)
	{
		px_ref_camera = get3dposition(px_ref_);
		////调试
		//cv::Mat rvec, t, solvePnP_Inliers;
		//double camera_matrix_data[3][3] =
		//{
		//	{ focal_, 0, pp_.x },
		//	{ 0, focal_, pp_.y },
		//	{ 0, 0, 1 }
		//};
		//cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
		//std::vector<cv::Point2f> pixl_cur_local = px_ref_;
		//vector<cv::Point3f> pts3f = PointCamera2Point3f(px_ref_camera, pixl_cur_local);
		//cv::solvePnPRansac(pts3f, pixl_cur_local, cameraMatrix, cv::Mat(), rvec, t, false, 1000, 8, 0.999999, solvePnP_Inliers, SOLVEPNP_EPNP);
		//std::cout << "rvec=" << rvec << ";" << endl << endl;
		//std::cout << "tvec=" << t << ";" << endl << endl;
		////调试结束
	}
	
    std::cout<<"img1_feature_size="<<px_ref_.size()<<std::endl;
	frame_stage_ = STAGE_SECOND_FRAME;
	frame_count_ = 0;
	return true;
}
bool VisualOdometry::processSecondFrame(int frame_id)
{
	std::vector<CameraSpacePoint> kp_cur_camera;
	double count, count_no_zero;
	if (ifORB)
	{
		featureDetectionORB(new_frame_, kp_cur_, orb_descriptor_cur_, px_cur_);//这里px_cur_的维度和kp_cur_的维度一致
		kp_cur_camera = get3dposition(px_cur_);//得到所有特征点的三维坐标，用于在计算玩运动后传递给px_ref_camera
		matches = ORBmatch(orb_descriptor_ref_, orb_descriptor_cur_, kp_ref_, kp_cur_, px_ref_, px_cur_, px_ref_camera);
		cv::drawMatches(last_frame_, kp_ref_, new_frame_, kp_cur_, matches, match_img);
		cv::resize(match_img, match_img, cv::Size(match_img.cols / 2, match_img.rows / 2));
		cv::imshow("匹配结果", match_img);
	}
	if (iffast)
	{
		featureTracking(last_frame_, new_frame_, px_ref_, px_cur_, disparities_, px_ref_camera); //
		px_cur_camera = get3dposition(px_cur_);
		SaperateFeaturesIntoTwoGroup(px_ref_camera,
			px_cur_camera, px_ref_camera_rot,
			px_ref_camera_trans, px_cur_camera_rot,
			px_cur_camera_trans, px_ref_, px_cur_, px_ref_rot, px_ref_trans, px_cur_rot, px_cur_trans);
	}
	if (ifdetectline)
	{
		featureTracking(last_frame_, new_frame_, px_ref_, px_cur_, disparities_, px_ref_camera); //
		px_cur_camera = get3dposition(px_cur_);//得到所有特征点的三维坐标，用于在计算玩运动后传递给px_ref_camera
		SaperateFeaturesIntoTwoGroup(px_ref_camera,
			px_cur_camera, px_ref_camera_rot,
			px_ref_camera_trans, px_cur_camera_rot,
			px_cur_camera_trans, px_ref_, px_cur_, px_ref_rot, px_ref_trans, px_cur_rot, px_cur_trans);

	}


	cv::Mat E, R, t, R1, mask, mask_trans,solvePnP_Inliers;
	if (ifsolvePnP)
	{
		//用Iterative的方法solvePnP，这里先求出迭代初始值rvec和t
		E = cv::findEssentialMat(px_cur_, px_ref_, focal_, pp_, cv::RANSAC, 0.999, 1.0, mask);
		cv::recoverPose(E, px_cur_, px_ref_, R, t, focal_, pp_, mask);
		R = rectifyR(R);
		//t = getTrans(px_ref_camera, px_cur_camera, R, t, mask);
		cv::Mat rvec;
		cv::Rodrigues(R, rvec);
		//用solvePnP求解R和t
		double camera_matrix_data[3][3] =
		{
			{ focal_, 0, pp_.x },
			{ 0, focal_, pp_.y },
			{ 0, 0, 1 }
		};
		double distortion_data[4][1] =
		{
			{ 0.022711 },
			{ -0.019329 },
			{ 0. },
			{ 0. }
			/*		{ -0.00132974 },
			{ -0.0162668 },
			{ 0.00362266 },
			{ 0.00106375 }*/
		};
		cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
		std::vector<cv::Point2f> pixl_cur_local = px_cur_;
		vector<cv::Point3f> pts3f = PointCamera2Point3f(px_ref_camera, pixl_cur_local);
		cv::solvePnPRansac(pts3f, pixl_cur_local, cameraMatrix, cv::Mat(), rvec, t, false, 200, 2, 0.99999, solvePnP_Inliers, SOLVEPNP_UPNP);
		std::cout << "rvec=" << rvec << ";" << endl << endl;
		std::cout << "tvec=" << t << ";" << endl << endl;
		cv::Rodrigues(rvec, R);
		//t = R1*t;
		//std::cout << "tvec=" << t << ";" << endl << endl;
	}
	if (iffindEssential)
	{
		if (px_ref_.size() <= 50)
		{
			E = cv::findEssentialMat(px_cur_, px_ref_, focal_, pp_, cv::LMEDS, 0.999, 1, mask);//prob = 0.999
		}
		else E = cv::findEssentialMat(px_cur_, px_ref_, focal_, pp_, cv::RANSAC, 0.999, 1, mask);//prob = 0.999
		count_no_zero = countNonZero(mask);
		if (count_no_zero / px_cur_.size() < 0.6)
		{
			E = cv::findEssentialMat(px_cur_, px_ref_, focal_, pp_, cv::LMEDS, 0.999, 1, mask);
		}
		//std::cout << "E=" << E << ";" << endl << endl;
		count = cv::recoverPose(E, px_cur_, px_ref_, R, t, focal_, pp_, mask);           //得到的t只是运动方向向量,返回位于相机前方的点的数量！！！！！！！！！！！！！！！！
		//std::cout << "通过手性检测点的数量=" << count << ";"  << endl;
		//std::cout << "R1=" << R << ";" << endl << endl;
		R = rectifyR(R);
		t = rectifyT(t);
		if (px_cur_trans.size() >= 50)
		{
			E = cv::findEssentialMat(px_cur_trans, px_ref_trans, focal_, pp_, cv::RANSAC, 0.999, 1, mask_trans);//prob = 0.999
		    count=cv::recoverPose(E, px_cur_trans, px_ref_trans, R, t, focal_, pp_, mask_trans);
			t = rectifyT(t);
			//scale = getRalativeScale(px_ref_camera, px_cur_camera);
			t = getTrans(px_ref_camera_trans, px_cur_camera_trans, R, t, mask_trans);
			//std::cout << "t=" << t << ";"  << endl;

		}
		else t = getTrans(px_ref_camera, px_cur_camera, R, t, mask);
	/*if (px_cur_.size()>5)
	{
		//*******************************************************
		E = cv::findEssentialMat(px_cur_, px_ref_, focal_, pp_, cv::RANSAC, 0.999, 1.0, mask);
		cv::recoverPose(E, px_cur_, px_ref_, R, t, focal_, pp_, mask);
		R = rectifyR(R);
		t = rectifyT(t);
		t = getTrans(px_ref_camera, px_cur_camera, R, t, mask);
		std::cout << "R=" << R << ";" << endl << endl;
	}*/
	}
		////***********************************************************
		////优化
		////添加顶点
		//g2o::VertexSE3 *v = new g2o::VertexSE3();
		//static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct("Cauchy");
		//v->setId(frame_id);
		//v->setEstimate(Eigen::Isometry3d::Identity());//估计为单位矩阵
		//optimizer.addVertex(v);
		////添加边
		//g2o::EdgeSE3* edge = new g2o::EdgeSE3();
		//// 连接此边的两个顶点id
		//edge->vertices()[0] = optimizer.vertex(frame_id-1);
		//edge->vertices()[1] = optimizer.vertex(frame_id);
		//edge->setRobustKernel(robustKernel);
		//// 信息矩阵
		//Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6, 6 >::Identity();
		//information(0, 0) = information(1, 1) = information(2, 2) = 100;
		//information(3, 3) = information(4, 4) = information(5, 5) = 100;
		//edge->setInformation(information);
		//// 边的估计即是pnp求解之结果
		//Eigen::Isometry3d T = Eigen::Isometry3d::Identity();// (这个T要替换成运动估计结果R和t组合成的4乘4的矩阵)
		//Eigen::Matrix3d r=Eigen::Matrix3d(R.rows,R.cols);
		//cv::cv2eigen(R, r);
		//


		//
		//Eigen::AngleAxisd angle(r);
		////Eigen::Translation<double, 3> trans(t.at<double>(0, 0), t.at<double>(0, 1), t.at<double>(0, 2));
		//T = angle;
		//cout << "T=" << T.matrix() << endl;
		//T(0, 3) = t.at<double>(0, 0);
		//T(1, 3) = t.at<double>(1, 0);
		//T(2, 3) = t.at<double>(2, 0);
		//cout << "T=" << T.matrix() << endl;
		//edge->setMeasurement(T);//T要不要取逆？
		//// 将此边加入图中
		//optimizer.addEdge(edge);
		//***********************************************************

		camera_space_points_.push_back(px_cur_camera);
		pxes_.push_back(px_cur_);
		frames_.push_back(new_frame_);



	cur_R_ = R.clone();
	cur_t_ = t.clone();
	std::cout << "t=" << t << ";" << endl << endl;
	std::cout << "R1=" << R << ";" << endl << endl;
    frame_stage_ = STAGE_DEFAULT_FRAME;//
	px_ref_rot.clear();
	px_ref_trans.clear();
	px_ref_camera_rot.clear(); 
	px_ref_camera_trans.clear();
	px_cur_rot.clear();
	px_cur_trans.clear();
	px_cur_camera_rot.clear();
	px_cur_camera_trans.clear();
	
	
	
	if (ifORB)
	{
		orb_descriptor_ref_ = orb_descriptor_cur_.clone();
		px_ref_camera = kp_cur_camera;
		kp_ref_ = kp_cur_;
	}
	if (ifdetectline)
	{
		//featureDetectionLines(new_frame_, px_cur_);//这里px_cur_的维度和kp_cur_的维度一致,每帧检测追踪
		//px_cur_camera = get3dposition(px_cur_);
		px_ref_camera = px_cur_camera;
		px_ref_ = px_cur_;
	}
	if (iffast)
	{
		px_ref_camera = px_cur_camera;
		px_ref_ = px_cur_;
	}
	return true;
}
bool VisualOdometry::processFrame(int frame_id)
{
	cv::Mat E, R, t, R1, t1, mask, mask_trans, img, sharp_img, solvePnP_Inliers;
	img = new_frame_;
	double count, count_no_zero;
	std::vector<CameraSpacePoint> kp_cur_camera;

	//px_cur_camera = get3dposition(px_ref_);//????????????
	//sharp_img = sharpenImage1(new_frame_);
	if (ifORB)
	{

		featureDetectionORB(new_frame_, kp_cur_, orb_descriptor_cur_, px_cur_);//这里px_cur_的维度和kp_cur_的维度一致
		kp_cur_camera = get3dposition(px_cur_);//得到所有特征点的三维坐标，用于在计算玩运动后传递给px_ref_camera
		matches = ORBmatch(orb_descriptor_ref_, orb_descriptor_cur_, kp_ref_, kp_cur_, px_ref_, px_cur_, px_ref_camera);
		cv::drawMatches(last_frame_, kp_ref_, new_frame_, kp_cur_, matches, match_img);
		cv::resize(match_img, match_img, cv::Size(match_img.cols / 2, match_img.rows / 2));
		cv::imshow("匹配结果", match_img);
	}
	//--------------------------------------------------------------------------------------------------------------------------------------
	if (ifdetectline)
	{
		featureTracking(last_frame_, new_frame_, px_ref_, px_cur_, disparities_, px_ref_camera); //
		px_cur_camera = get3dposition(px_cur_);//得到所有特征点的三维坐标，用于在计算玩运动后传递给px_ref_camera
		SaperateFeaturesIntoTwoGroup(px_ref_camera,
			px_cur_camera, px_ref_camera_rot,
			px_ref_camera_trans, px_cur_camera_rot,
			px_cur_camera_trans, px_ref_, px_cur_, px_ref_rot, px_ref_trans, px_cur_rot, px_cur_trans);
	}
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	if (iffast)
	{
		featureTracking(last_frame_, new_frame_, px_ref_, px_cur_, disparities_, px_ref_camera); //
		px_cur_camera = get3dposition(px_cur_);
		SaperateFeaturesIntoTwoGroup(px_ref_camera,
		px_cur_camera, px_ref_camera_rot,
		px_ref_camera_trans, px_cur_camera_rot,
		px_cur_camera_trans, px_ref_, px_cur_, px_ref_rot, px_ref_trans, px_cur_rot, px_cur_trans);
	}

	if (ifsolvePnP)
	{
		//用Iterative的方法solvePnP，这里先求出迭代初始值rvec和t
		E = cv::findEssentialMat(px_cur_, px_ref_, focal_, pp_, cv::RANSAC, 0.999, 1, mask);//prob = 0.999
		cv::recoverPose(E, px_cur_, px_ref_, R, t, focal_, pp_, mask);
		R = rectifyR(R);
		//t = getTrans(px_ref_camera, px_cur_camera, R, t, mask_trans);
		cv::Mat rvec;
		cv::Rodrigues(R, rvec);
		//用solvePnP求解R,t
		double camera_matrix_data[3][3] =
		{
			{ focal_, 0, pp_.x },
			{ 0, focal_, pp_.y },
			{ 0, 0, 1 }
		};
		double distortion_data[4][1] =
		{
			{ -0.00670442 },
			{ -0.0155172  },
			{ 0.00322575 },
			{ -0.00291871 }
		};

		cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
		cv::Mat distCoeffs(4, 1, CV_64FC1, distortion_data);  // vector of distortion coefficients
		std::vector<cv::Point2f> pixl_cur_local = px_cur_;
		vector<cv::Point3f> pts3f = PointCamera2Point3f(px_ref_camera, pixl_cur_local);
		cv::solvePnPRansac(pts3f, pixl_cur_local, cameraMatrix, cv::Mat(), rvec, t, false, 200, 2, 0.99999, solvePnP_Inliers, SOLVEPNP_UPNP);
		std::cout << "rvec=" << rvec << ";" << endl << endl;
		cv::Rodrigues(rvec, R1);
		//std::cout << "t0=" << t << ";" << endl << endl;
		//t = R1*t;
	}
	if (iffindEssential)
	{
		if (px_ref_.size() >= 50)
		{
			//*****************************************************************/*
			if (px_ref_.size() <= 110)
			{
				E = cv::findEssentialMat(px_cur_, px_ref_, focal_, pp_, cv::LMEDS, 0.999, 1, mask);//prob = 0.999
			}
			else E = cv::findEssentialMat(px_cur_, px_ref_, focal_, pp_, cv::RANSAC, 0.999, 1, mask);//prob = 0.999
			count_no_zero = countNonZero(mask);
			if (count_no_zero / px_cur_.size() < 0.6)
			{
				E = cv::findEssentialMat(px_cur_, px_ref_, focal_, pp_, cv::LMEDS, 0.999, 1, mask);
			}
			//std::cout << "E=" << E << ";" << endl << endl;
			count = cv::recoverPose(E, px_cur_, px_ref_, R, t, focal_, pp_, mask);           //得到的t只是运动方向向量,返回位于相机前方的点的数量！！！！！！！！！！！！！！！！
			//std::cout << "通过手性检测点的数量=" << count << ";"  << endl;
			//std::cout << "R1=" << R << ";" << endl << endl;
			R = rectifyR(R);
			t = rectifyT(t);
			if (px_cur_trans.size() >= 8)
			{
				E = cv::findEssentialMat(px_cur_trans, px_ref_trans, focal_, pp_, cv::RANSAC, 0.999, 1, mask_trans);//prob = 0.999
				count = cv::recoverPose(E, px_cur_trans, px_ref_trans, R, t, focal_, pp_, mask_trans);
				t = rectifyT(t);
				//scale = getRalativeScale(px_ref_camera, px_cur_camera);
				t = getTrans(px_ref_camera_trans, px_cur_camera_trans, R, t, mask_trans);
				//std::cout << "t=" << t << ";"  << endl;

			}
			else t = getTrans(px_ref_camera, px_cur_camera, R, t, mask);
		}


	}

			////*****************************************************************/*
			////优化
			//g2o::VertexSE3 *v = new g2o::VertexSE3();
			//static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct("Cauchy");
			//v->setId(frame_id);
			//v->setEstimate(Eigen::Isometry3d::Identity());
			//optimizer.addVertex(v);
			////添加边
			//g2o::EdgeSE3* edge = new g2o::EdgeSE3();
			//// 连接此边的两个顶点id
			//edge->vertices()[0] = optimizer.vertex(frame_id-1);
			//edge->vertices()[1] = optimizer.vertex(frame_id);
			//edge->setRobustKernel(robustKernel);
			//// 信息矩阵
			//Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6, 6 >::Identity();
			//information(0, 0) = information(1, 1) = information(2, 2) = 100;
			//information(3, 3) = information(4, 4) = information(5, 5) = 100;
			//edge->setInformation(information);
			//// 边的估计即是pnp求解之结果
			//Eigen::Isometry3d T = Eigen::Isometry3d::Identity();// (这个T要替换成运动估计结果R和t组合成的4乘4的矩阵)
			//Eigen::Matrix3d r;
			//cv::cv2eigen(R, r);




			//Eigen::AngleAxisd angle(r);
			////Eigen::Translation<double, 3> trans(t.at<double>(0, 0), t.at<double>(0, 1), t.at<double>(0, 2));
			//T = angle;
			//T(0, 3) = t.at<double>(0, 0);
			//T(1, 3) = t.at<double>(1, 0);
			//T(2, 3) = t.at<double>(2, 0);
			//edge->setMeasurement(T);//T要不要取逆？
			//// 将此边加入图中
			//optimizer.addEdge(edge);

			cur_t_ = cur_t_ + cur_R_*t;//
		cur_R_ = R*cur_R_;
		std::cout << "R=" << R << ";" << endl << endl;
		std::cout << "t=" << t << ";" << endl << endl;
		//featureDetection(new_frame_, px_cur_);
		//px_cur_camera = get3dposition(px_cur_);

		transes_.push_back(t);
		rotations_.push_back(R);
		frames_.push_back(new_frame_);
		camera_space_points_.push_back(px_cur_camera);
		pxes_.push_back(px_cur_);
		frame_count_++;
	
	//featureDetection(new_frame_, px_cur_);
	//px_cur_camera = get3dposition(px_cur_);
	
	if (px_ref_.size() < kMinNumFeature)
	{
		std::cout << "redirection" << std::endl;

		transes_.clear();
		rotations_.clear();
		camera_space_points_.clear();
		pxes_.clear();
		frames_.clear();
		frame_count_ = 0;

		//new_frame_ = sharpenImage1(new_frame_);
		if (ifORB)
		{
			featureDetectionORB(new_frame_, kp_cur_, orb_descriptor_cur_, px_cur_);
		}
		if (iffast)
		{
			featureDetection(new_frame_, px_cur_);
		}
		if (ifdetectline)
		{
			featureDetectionLines(new_frame_, px_cur_);
		}

		px_cur_camera = get3dposition(px_cur_);

		camera_space_points_.push_back(px_cur_camera);
		pxes_.push_back(px_cur_);
		frames_.push_back(new_frame_);
	}
    //******************************************************************
	if (frame_count_>=2)
	{
		//bundleAdjustment(frames_, transes_, rotations_, camera_space_points_, pxes_);
	}

	for (int i = 0; i < px_cur_trans.size(); i++)
	{
		//以随机的颜色绘制出角点
		circle(img, px_cur_trans[i], 3, Scalar(145, 175,
			143), -1, 8, 0);
	}
	for (int i = 0; i < px_cur_rot.size(); i++)
	{
		//以随机的颜色绘制出角点
		rectangle(img, px_cur_rot[i], px_cur_rot[i] + cv::Point2f(2, 2), Scalar(145, 175,
			143), CV_FILLED);
	}
	imshow("RGB camera", img);
	px_ref_rot.clear();
	px_ref_trans.clear();
	px_ref_camera_rot.clear();
	px_ref_camera_trans.clear();
	px_cur_rot.clear();
	px_cur_trans.clear();
	px_cur_camera_rot.clear();
	px_cur_camera_trans.clear();
	//new_frame_ = sharp_img;
	if (ifORB)
	{
		px_ref_camera = kp_cur_camera;
		kp_ref_ = kp_cur_;
		orb_descriptor_ref_ = orb_descriptor_cur_.clone();
	}
	if (ifdetectline)
	{
		//featureDetectionLines(new_frame_, px_cur_);//这里px_cur_的维度和kp_cur_的维度一致，每帧检测追踪
		px_ref_camera = get3dposition(px_cur_);//得到所有特征点的三维坐标，用于在计算玩运动后传递给px_ref_camera
	}
	if (iffast)
	{
		px_ref_camera = px_cur_camera;
	}
	px_ref_ = px_cur_;

    std::cout<<px_ref_.size()<<std::endl;
	return true;
}
std::vector<CameraSpacePoint> VisualOdometry::get3dposition(std::vector<Point2f> points)
{
	std::vector<CameraSpacePoint> points_camera;
	CameraSpacePoint point_cur_camera;
	for (int index = 0; index < points.size();index++)
	{
		cv::Point2f point = points[index];
		int px_x = point.x;
		int px_y = point.y;
		if (px_y>=1080||px_y<0||px_x<0||px_x>=1920)
		{
			point_cur_camera.X = -INFINITY;
			point_cur_camera.Y = -INFINITY;
			point_cur_camera.Z = -INFINITY;
		}
		else point_cur_camera = *(m_pCameraCoordinates + (px_y)*cColorWidth + px_x);
		points_camera.push_back(point_cur_camera);
	}
	//以下是调试代码，检查彩色坐标系和相机坐标系的投影是否正确------------------------------------
	//CameraSpacePoint* points_camera_test = new CameraSpacePoint [points_camera.size()];
	//CameraSpacePoint* ptr_camera = points_camera_test;
	//int i = 0;
	//for (int index = 0; index < points_camera.size(); index++)
	//{
	//	cv::Point2f point;
	//	point_cur_camera = points_camera[index];
	//	if (point_cur_camera.X == -INFINITY)
	//		continue;
	//	*points_camera_test=point_cur_camera;
	//	points_camera_test++;
	//	i++;
	//}
	//ColorSpacePoint *ptr_color = new ColorSpacePoint[points_camera.size()];
	//HRESULT hr = m_pCoordinateMapper->MapCameraPointsToColorSpace(i, ptr_camera, i, ptr_color);
	//---------------------------------------------------------------------------------------------------
	//检查结果：坐标变换基本上正确，存在一两个像素的误差，不确定这一两个像素误差是否影响最终结果
	return points_camera;
}
void VisualOdometry::RemoveInvalidData(std::vector<CameraSpacePoint>& camera_space_points1,
	std::vector<CameraSpacePoint>& camera_space_points2)
{
	CameraSpacePoint p1_;
	CameraSpacePoint p2_;
	std::vector<CameraSpacePoint>::iterator iter1_ = camera_space_points1.begin();
	std::vector<CameraSpacePoint>::iterator iter2_ = camera_space_points2.begin();
	for (; iter1_ != camera_space_points1.end(); )
	{
		p1_ = *iter1_;
		p2_ = *iter2_;
		float p1_x_ = p1_.X; float p1_y_ = p1_.Y; float p1_z_ = p1_.Z;
		float p2_x_ = p2_.X; float p2_y_ = p2_.Y; float p2_z_ = p2_.Z;
		if (isinf(p1_x_) || isinf(p2_x_))				
		{
			iter1_ = camera_space_points1.erase(iter1_);
			iter2_ = camera_space_points2.erase(iter2_);
			continue;
		}
		iter1_++; iter2_++;
	}
}
	
	
float VisualOdometry::getRalativeScale(std::vector<CameraSpacePoint> camera_space_points1, 
										std::vector<CameraSpacePoint> camera_space_points2)
{
	std::vector<float> scales;
	if (camera_space_points1.size() == 0 || camera_space_points1.size() != camera_space_points2.size())
	{
		return 1.0;
	}
	
	for (int index = 0; index < camera_space_points1.size(); ++index)
	{
		for (int index1 = 0; index1 < camera_space_points2.size(); ++index1)
		{
			CameraSpacePoint point1_ref = camera_space_points1[index];
			CameraSpacePoint point1_cur = camera_space_points2[index];
			CameraSpacePoint point2_ref = camera_space_points1[index1];
			CameraSpacePoint point2_cur = camera_space_points2[index1];
			if (isinf(point1_ref.X) || isinf(point2_ref.X) || isinf(point1_cur.X) || isinf(point2_cur.X))
			{
				continue;
			}
			float denominator = sqrt((point1_ref.X - point2_ref.X)*(point1_ref.X - point2_ref.X) + (point1_ref.Y - point2_ref.Y)*(point1_ref.Y - point2_ref.Y) + (point1_ref.Z - point2_ref.Z)*(point1_ref.Z - point2_ref.Z));
			if (denominator<=0)
			{
				continue;
			}
			float nominator = sqrt((point1_cur.X - point2_cur.X)*(point1_cur.X - point2_cur.X) + (point1_cur.Y - point2_cur.Y)*(point1_cur.Y - point2_cur.Y) + (point1_cur.Z - point2_cur.Z)*(point1_cur.Z - point2_cur.Z));
			float scale = nominator / denominator;
			if (!isinf(scale) && !isnan(scale))
			{
				scales.push_back(scale);
			}
		}
	}
	sort(scales.begin(),scales.end());
	return (scales[int(scales.size()/2)]);

}
cv::Mat VisualOdometry::getTrans(std::vector<CameraSpacePoint> camera_space_points1,
	std::vector<CameraSpacePoint> camera_space_points2, cv::Mat R, cv::Mat t, cv::Mat mask)
{
	cv::Mat X1=t.clone(), X2=t.clone();
	std::ofstream out("scale.txt");
	double scale = 0.;
	double scale1 = 0.;
	cv::Mat T;
	T = 0;
	double average_tx=0., average_ty=0., average_tz=0.;
	int i = 0;
	std::vector<double> tx;
	std::vector<double> ty;
	std::vector<double> tz;
	if (camera_space_points1.size() == 0 || camera_space_points1.size() != camera_space_points2.size())
	{
		return T;
	}
	for (int index = 0; index < camera_space_points1.size(); ++index)
	{

		CameraSpacePoint point1_ref = camera_space_points1[index];
		CameraSpacePoint point1_cur = camera_space_points2[index];
		bool mask_flag = mask.at<bool>(index);
		if (isinf(point1_ref.Z) || isinf(point1_cur.Z) || mask_flag == 0)
		{
			continue;
		}
		X1.at<double>(0) = point1_ref.X;
		X1.at<double>(1) = point1_ref.Y;
		X1.at<double>(2) = point1_ref.Z;
		X2.at<double>(0) = point1_cur.X;
		X2.at<double>(1) = point1_cur.Y;
		X2.at<double>(2) = point1_cur.Z;
		T = X2 - R*X1;
		//i++;
		if (abs(T.at<double>(0)) < 1 && abs(T.at<double>(1)) < 1 && abs(T.at<double>(2)) < 1)
		//if (true)
		{
			double test_prob1 = T.at<double>(0);
			double test_prob2 = T.at<double>(1);
			double test_prob3 = T.at<double>(2);
			tx.push_back(T.at<double>(0));
			ty.push_back(T.at<double>(1));
			tz.push_back(T.at<double>(2));
			average_tx = average_tx + T.at<double>(0);
			average_ty = average_ty + T.at<double>(1);
			average_tz = average_tz + T.at<double>(2);
		}
		else 
		{
			continue;
	    }

	}
	if (tx.size() > 0)
	{
		sort(tx.begin(), tx.end());
		sort(ty.begin(), ty.end());
		sort(tz.begin(), tz.end());
		T.at<double>(0) = tx[int(tx.size() / 2)];
		T.at<double>(1) = ty[int(ty.size() / 2)];
		T.at<double>(2) = tz[int(tz.size() / 2)];
		double x1 = T.at<double>(0);
		double y1 = T.at<double>(1);
		double z1 = T.at<double>(2);
		scale1 = sqrt(x1*x1 + y1*y1 + z1*z1);
		//average_tx = average_tx / tx.size();
		//average_ty = average_ty / ty.size();
		//average_tz = average_tz / tz.size();
		T.at<double>(0) = average_tx / tx.size();
		T.at<double>(1) = average_ty / ty.size();
		T.at<double>(2) = average_tz / tz.size();
		double x = T.at<double>(0) ;
		double y = T.at<double>(1) ;
		double z = T.at<double>(2) ;
		scale = sqrt(x*x + y*y + z*z);
		if (scale<0.005||scale>1)
		//if (scale<0.005 )
		{
			scale = 0.;
		}
	}
	
	out << scale<< std::endl;
	out.close();
	return  scale*t;
	//t = t / i;
	//return T;
}

void VisualOdometry::SaperateFeaturesIntoTwoGroup(std::vector<CameraSpacePoint> camera_space_points_ref,
	std::vector<CameraSpacePoint> camera_space_points_cur, std::vector<CameraSpacePoint>& camera_space_points_ref_rot,
	std::vector<CameraSpacePoint>& camera_space_points_ref_trans, std::vector<CameraSpacePoint>& camera_space_points_cur_rot,
	std::vector<CameraSpacePoint>& camera_space_points_cur_trans, std::vector<cv::Point2f> px_ref, std::vector<cv::Point2f> px_cur,
	std::vector<cv::Point2f>& px_ref_rot, std::vector<cv::Point2f>& px_ref_trans,
	std::vector<cv::Point2f>& px_cur_rot, std::vector<cv::Point2f>& px_cur_trans)
{
	int flag = 1;
	int n = camera_space_points_ref.size();
	CameraSpacePoint P1;
	CameraSpacePoint P2;
	cv::Point2f p1;
	cv::Point2f p2;
	float z_average = 0.;
	int i = 0;
	switch (flag)
	{
	case 0:
		//计算当前帧的平均z


		for (int index = 0; index < n; ++index)
		{
			P1 = camera_space_points_ref[index];
			P2 = camera_space_points_cur[index];
			if (isinf(P1.X) || isinf(P2.X))
			{
				continue;
			}
			z_average = z_average + P2.Z;
			i++;
		}
		z_average = z_average / i;
		//判断参考帧中相机坐标系下的特征点的z值是否大于平均z，若大于，则入栈到rot中，否则入栈到trans中
		for (int index = 0; index < n; ++index)
		{
			P1 = camera_space_points_ref[index];
			P2 = camera_space_points_cur[index];
			p1 = px_ref[index];
			p2 = px_cur[index];
			//if (isinf(P1.X) || isinf(P2.X))
			//{
			//	camera_space_points_ref_rot.push_back(P1);
			//	camera_space_points_cur_rot.push_back(P2);
			//	px_ref_rot.push_back(p1);
			//	px_cur_rot.push_back(p2);
			//	continue;
			//}
			if (abs(P2.Z) <= z_average)
			{
				camera_space_points_ref_trans.push_back(P1);
				camera_space_points_cur_trans.push_back(P2);
				px_ref_trans.push_back(p1);
				px_cur_trans.push_back(p2);

			}
			else
			{
				camera_space_points_ref_rot.push_back(P1);
				camera_space_points_cur_rot.push_back(P2);
				px_ref_rot.push_back(p1);
				px_cur_rot.push_back(p2);
			}
		}
		break;
	case 1:
		for (int index = 0; index < n; ++index)
		{
			P1 = camera_space_points_ref[index];
			P2 = camera_space_points_cur[index];
			p1 = px_ref[index];
			p2 = px_cur[index];
			if (isinf(P1.X) || isinf(P2.X))
			{
				camera_space_points_ref_rot.push_back(P1);
				camera_space_points_cur_rot.push_back(P2);
				px_ref_rot.push_back(p1);
				px_cur_rot.push_back(p2);
			}
			else
			{
				camera_space_points_ref_trans.push_back(P1);
				camera_space_points_cur_trans.push_back(P2);
				px_ref_trans.push_back(p1);
				px_cur_trans.push_back(p2);
			}
		}
		break;
	}
	



}

vector<cv::Point3f> VisualOdometry::PointCamera2Point3f(std::vector<CameraSpacePoint> points_camera, std::vector<cv::Point2f>& pixl_ref_local)
{
	std::vector<cv::Point3f> pts_3f;
	std::vector<cv::Point2f> pix_ref = pixl_ref_local;
	cv::Point3f p3f;
	pixl_ref_local.clear();
	for (size_t i = 0; i < points_camera.size(); i++)
	{
		CameraSpacePoint p = points_camera[i];
		cv::Point2f Pixel = pix_ref[i];
		if (isinf(p.X))
			continue;
		p3f.x = p.X;
		p3f.y = -p.Y;
		p3f.z = p.Z;
		pts_3f.push_back(p3f);
		pixl_ref_local.push_back(Pixel);
	}
	return pts_3f;
}

cv::Mat VisualOdometry::sharpenImage1(cv::Mat input)
{
	cv::Mat output;
	cv::Mat show;
	//创建并初始化滤波模板
	//cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	//kernel.at<float>(1, 1) = 5.0;
	//kernel.at<float>(0, 1) = -1.0;
	//kernel.at<float>(1, 0) = -1.0;
	//kernel.at<float>(1, 2) = -1.0;
	//kernel.at<float>(2, 1) = -1.0;
	//////GaussianBlur(input, input, Size(3, 3), 0, 0);
	//////Laplacian(input, output, CV_16S, 3);
	//////convertScaleAbs(output, output);
	//////imshow("高斯滤波", output);
	Mat kernel(3, 3, CV_32F, Scalar(-1));  // 分配像素置  
	kernel.at<float>(1, 1) = 8.9;
	output.create(input.size(), input.type());
	//对图像进行滤波
	cv::filter2D(input, output, input.depth(), kernel);
	show = output;
	resize(show, show, cv::Size(show.cols / 2, show.rows / 2));
	imshow("锐化", show);
	return output;
}


double VisualOdometry::excludeOutliers(std::vector<double> dispariaties)
{
	double num, num1;
	int count=0, count1=0;
	sort(dispariaties.begin(),dispariaties.end());
	std::vector<double>::iterator it = dispariaties.begin();
	num = *it;
	num1 = num;
	for (; it != dispariaties.end();it++)
	{
		if (num - *it<=20)
			count++;
		else
		{
			if (count>count1)
			{
				count1 = count;
				num1 = num;
				num = *it;
			}
		}

	}
	return num1;
}

cv::Mat VisualOdometry::rectifyR(cv::Mat R)
{
	cv::Mat I = cv::Mat(3, 3, CV_64F);
	cv::setIdentity(I,1.);
	float R00 = R.at<double>(0, 0);
	float R10 = R.at<double>(1, 0);
	float R20 = R.at<double>(2, 0);
	bool flag = abs(R00 + 0.3333333) < 0.0001 && abs(R10 + 0.6666666 < 0.0001) && abs(R20 - 0.6666666 < 0.0001);
	bool flag2 = abs(R20) - 0.03 < 0;
	if (flag || flag2)
	{
		return I;
	}
	return R;
}

cv::Mat VisualOdometry::rectifyT(cv::Mat t)
{
	cv::Mat I = cv::Mat::zeros(3,1,CV_64F);
	float t00 = t.at<double>(0, 0);
	float t10 = t.at<double>(1, 0);
	float t20 = t.at<double>(2, 0);
	bool flag = (abs(t00) - 0.57735 < 0.01 && abs(t10) - 0.57735 < 0.01 && abs(t20) - 0.57735 < 0.01 );
	if (flag)
	{

		return I;
	}
	return t;
}

void VisualOdometry::bundleAdjustment(std::vector<cv::Mat>& frames_, std::vector<cv::Mat>& transes_, std::vector<cv::Mat>& rotations_,
	std::vector<std::vector<CameraSpacePoint>>& camera_space_points_, std::vector<std::vector<cv::Point2f>>& pxes_)
{
	cv::Mat R, t;
	//frame_count_ = 0;
	cv::Mat f_ref = frames_[0];
	cv::Mat f_cur = frames_[2];
	cv::Mat t0 = transes_[0];
	cv::Mat r0 = rotations_[0];
	cv::Mat t1 = transes_[1];
	cv::Mat r1 = rotations_[1];
	std::vector<cv::Point2f> pixel_ref_ = pxes_[0];
	std::vector<cv::Point2f> pixel_cur_;
	std::vector<CameraSpacePoint> point_camera_ref = camera_space_points_[0];
	featureTracking(f_ref, f_cur, pixel_ref_, pixel_cur_, disparities_, point_camera_ref);
	std::vector<CameraSpacePoint> point_camera_cur = get3dposition(pixel_cur_);
	cv::Mat E = cv::findEssentialMat(pixel_cur_, pixel_ref_, focal_, pp_, cv::RANSAC, 0.999, 1.0);
	cv::recoverPose(E, pixel_cur_, pixel_ref_, R, t, focal_, pp_);
	//t = getTrans(point_camera_ref, point_camera_cur, R, t, mask);
	cv::Mat er = R-r1*r0;
	cv::Mat et = t-(t0+r0*t1);
	std::cout << "er=" << er << ";" << endl << endl;
	std::cout << "et=" << et << ";" << endl << endl;
	transes_.erase(transes_.begin());
	rotations_.erase(rotations_.begin());
	camera_space_points_.erase(camera_space_points_.begin());
	pxes_.erase(pxes_.begin());
	frames_.erase(frames_.begin());
}

int VisualOdometry::eightPointAugorithm(const CvMat* _m1, const CvMat* _m2, CvMat* _fmatrix)
{
	double a[9 * 9], w[9], v[9 * 9];
	CvMat W = cvMat(1, 9, CV_64F, w);
	CvMat V = cvMat(9, 9, CV_64F, v);
	CvMat A = cvMat(9, 9, CV_64F, a);
	CvMat U, F0, TF;
	CvPoint2D64f m0c = { 0, 0 }, m1c = { 0, 0 };
	double t, scale0 = 0, scale1 = 0;
	const CvPoint2D64f* m1 = (const CvPoint2D64f*)_m1->data.ptr;
	const CvPoint2D64f* m2 = (const CvPoint2D64f*)_m2->data.ptr;
	double* fmatrix = _fmatrix->data.db;
	int i, j, k, count = _m1->cols*_m1->rows;
	// compute centers and average distances for each of the two point sets
	for (i = 0; i < count; i++)
	{
		double x = m1[i].x, y = m1[i].y;
		m0c.x += x; m0c.y += y;
		x = m2[i].x, y = m2[i].y;
		m1c.x += x; m1c.y += y;
	}
	// calculate the normalizing transformations for each of the point sets:
	// after the transformation each set will have the mass center at the coordinate origin
	// and the average distance from the origin will be ~sqrt(2).
	t = 1. / count;
	m0c.x *= t; m0c.y *= t;
	m1c.x *= t; m1c.y *= t;
	for (i = 0; i < count; i++)
	{
		double x = m1[i].x - m0c.x, y = m1[i].y - m0c.y;
		scale0 += sqrt(x*x + y*y);
		x = fabs(m2[i].x - m1c.x), y = fabs(m2[i].y - m1c.y);
		scale1 += sqrt(x*x + y*y);
	}
	scale0 *= t;
	scale1 *= t;
	if (scale0 < FLT_EPSILON || scale1 < FLT_EPSILON)
		return 0;
	scale0 = sqrt(2.) / scale0;
	scale1 = sqrt(2.) / scale1;

	cvZero(&A);
	// form a linear system Ax=0: for each selected pair of points m1 & m2,
	// the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0
	// to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0. 
	for (i = 0; i < count; i++)
	{
		double x0 = (m1[i].x - m0c.x)*scale0;
		double y0 = (m1[i].y - m0c.y)*scale0;
		double x1 = (m2[i].x - m1c.x)*scale1;
		double y1 = (m2[i].y - m1c.y)*scale1;
		double r[9] = { x1*x0, x1*y0, x1, y1*x0, y1*y0, y1, x0, y0, 1 };
		for (j = 0; j < 9; j++)
			for (k = 0; k < 9; k++)
				a[j * 9 + k] += r[j] * r[k];
	}
	cvSVD(&A, &W, 0, &V, CV_SVD_MODIFY_A + CV_SVD_V_T);
	for (i = 0; i < 8; i++)
	{
		if (fabs(w[i]) < DBL_EPSILON)
			break;
	}
	if (i < 7)
		return 0;
	F0 = cvMat(3, 3, CV_64F, v + 9 * 8); // take the last column of v as a solution of Af = 0
	// make F0 singular (of rank 2) by decomposing it with SVD,
	// zeroing the last diagonal element of W and then composing the matrices back.
	// use v as a temporary storage for different 3x3 matrices
	W = U = V = TF = F0;
	W.data.db = v;
	U.data.db = v + 9;
	V.data.db = v + 18;
	TF.data.db = v + 27;
	cvSVD(&F0, &W, &U, &V, CV_SVD_MODIFY_A + CV_SVD_U_T + CV_SVD_V_T);
	W.data.db[8] = 0.;
	// F0 <- U*diag([W(1), W(2), 0])*V'
	cvGEMM(&U, &W, 1., 0, 0., &TF, CV_GEMM_A_T);
	cvGEMM(&TF, &V, 1., 0, 0., &F0, 0/*CV_GEMM_B_T*/);
	// apply the transformation that is inverse
	// to what we used to normalize the point coordinates
	{
		double tt0[] = { scale0, 0, -scale0*m0c.x, 0, scale0, -scale0*m0c.y, 0, 0, 1 };
		double tt1[] = { scale1, 0, -scale1*m1c.x, 0, scale1, -scale1*m1c.y, 0, 0, 1 };
		CvMat T0, T1;
		T0 = T1 = F0;
		T0.data.db = tt0;
		T1.data.db = tt1;
		// F0 <- T1'*F0*T0
		cvGEMM(&T1, &F0, 1., 0, 0., &TF, CV_GEMM_A_T);
		F0.data.db = fmatrix;
		cvGEMM(&TF, &T0, 1., 0, 0., &F0, 0);
		// make F(3,3) = 1
		if (fabs(F0.data.db[8]) > FLT_EPSILON)
			cvScale(&F0, &F0, 1. / F0.data.db[8]);
	}
	return 1;
}

void VisualOdometry::featureDetection(cv::Mat input_img, std::vector<cv::Point2f>& px_vec)	
{
	std::vector<cv::KeyPoint> keypoints;
	int fast_threshold = 20;
	bool non_max_suppression = true;
	cv::FAST(input_img, keypoints, fast_threshold, non_max_suppression);
	cv::KeyPoint::convert(keypoints, px_vec);
}
void VisualOdometry::featureDetectionLines(cv::Mat image, std::vector<cv::Point2f> &px_vec)
{
	cv::Mat dstimg;
	std::vector<cv::Point2f> keypoints;
	cv::Canny(image, dstimg, 50, 200, 3);
	std::vector<cv::Vec4f> lines;
	cv::HoughLinesP(dstimg, lines, 1, CV_PI / 180, 80, 50, 10);
	for (int i = 0; i < lines.size(); i++)
	{
		cv::Vec4f l = lines[i];
		keypoints.push_back(cv::Point(l[0], l[1]));
		//keypoints.push_back(cv::Point((l[0] + l[2]) / 2, (l[1] + l[3]) / 2));
		keypoints.push_back(cv::Point(l[2], l[3]));
	}
	px_vec = keypoints;
	/*std::vector<cv::KeyPoint> keypoints;
	int fast_threshold = 20;
	bool non_max_suppression = true;
	cv::FAST(image, keypoints, fast_threshold, non_max_suppression);
	cv::KeyPoint::convert(keypoints, px_vec);*/
}

void VisualOdometry::featureDetectionORB(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor, std::vector<cv::Point2f>& px)
{
	px.clear();
	cv::Ptr<cv::ORB> orb = cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
	orb->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
	cv::KeyPoint::convert(keypoints, px);
}

std::vector< cv::DMatch > VisualOdometry::ORBmatch(cv::Mat descriptor1, cv::Mat descriptor2, std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2,
	std::vector<cv::Point2f>& px_ref, std::vector<cv::Point2f>& px_cur, std::vector<CameraSpacePoint>& px_ref_camera)
{
	px_ref.clear();
	px_cur.clear();
	std::vector<CameraSpacePoint> points3D;
	cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	double knn_match_ratio = 0.8;
	std::vector< std::vector<cv::DMatch> > matches_knn;
	matcher->knnMatch(descriptor1, descriptor2, matches_knn, 2);
	std::vector< cv::DMatch > matches;
	for (size_t i = 0; i < matches_knn.size(); i++)
	{
		if (matches_knn[i][0].distance < knn_match_ratio  *
			matches_knn[i][1].distance)
			matches.push_back(matches_knn[i][0]);
	}
	if (matches.size() <= 20) //匹配点太少
		return matches;
	for (auto m : matches)
	{
		px_ref.push_back(keypoints1[m.queryIdx].pt);
		px_cur.push_back(keypoints2[m.trainIdx].pt);
		points3D.push_back(px_ref_camera[m.queryIdx]);
	}
	px_ref_camera.clear();
	px_ref_camera = points3D;
	return matches;
}

void VisualOdometry::featureTracking(cv::Mat image_ref, cv::Mat image_cur,
	std::vector<cv::Point2f>& px_ref, std::vector<cv::Point2f>& px_cur, std::vector<double>& disparities, std::vector<CameraSpacePoint>& px_ref_camera)
{
	const double klt_win_size = 21.0;
	const int klt_max_iter = 30;
	const double klt_eps = 0.0001;
	std::vector<uchar> status;
	std::vector<float> error;
	std::vector<float> min_eig_vec;
	cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, klt_max_iter, klt_eps);
	cv::calcOpticalFlowPyrLK(image_ref, image_cur,
		px_ref, px_cur,
		status, error,
		cv::Size2i(klt_win_size, klt_win_size),
		4, termcrit, 0, 1e-4);
	
	std::vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
	std::vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
	std::vector<CameraSpacePoint>::iterator px_ref_camera_it = px_ref_camera.begin();
	disparities.clear(); disparities.reserve(px_cur.size());

	//根据追踪误差滤波，即剔除误差过大的追踪点
	for (size_t i = 0; px_ref_it != px_ref.end(); ++i)
	{
		(*px_cur_it).x = round((*px_cur_it).x);
		(*px_cur_it).y = round((*px_cur_it).y);
		int breakpoint = (*px_cur_it).y;
		if (!status[i])
		{
			px_ref_it = px_ref.erase(px_ref_it);
			px_ref_camera_it = px_ref_camera.erase(px_ref_camera_it);
			px_cur_it = px_cur.erase(px_cur_it);
			continue;
		}
		disparities.push_back(norm(cv::Point2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y)));
		++px_ref_it;
		++px_cur_it;
	}
	//double optical_vel = excludeOutliers(disparities);
	

}


//----------------------------------------------------------------------------------------------------------------------
HRESULT	VisualOdometry::InitKinect()
{
	HRESULT hr;
	
	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	if (m_pKinectSensor)
	{
		// Initialize the Kinect and get coordinate mapper and the frame reader
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
		}

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->OpenMultiSourceFrameReader(
				FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color, &m_pMultiSourceFrameReader);

			//			hr = m_pKinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes::FrameSourceTypes_Color ,	&m_pMultiSourceFrameReader);
		}
	}

	if (!m_pKinectSensor || FAILED(hr))
	{
		return E_FAIL;
	}

	return hr;
}
HRESULT VisualOdometry::Update()
{
	HRESULT hr=1;
	if (!m_pMultiSourceFrameReader)
	{
		return hr;
	}

	IMultiSourceFrame* pMultiSourceFrame = NULL;
	IDepthFrame* pDepthFrame = NULL;
	IColorFrame* pColorFrame = NULL;


	hr = m_pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);
	if (!pMultiSourceFrame)
	{
		printf("trying to get multi_source_frame\n");
	}
	if (SUCCEEDED(hr))//深度信息
	{
		IDepthFrameReference* pDepthFrameReference = NULL;
		hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
		}
		SafeRelease(pDepthFrameReference);
	}

	if (SUCCEEDED(hr))//彩色信息
	{
		IColorFrameReference* pColorFrameReference = NULL;
		hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pColorFrameReference->AcquireFrame(&pColorFrame);
		}
		SafeRelease(pColorFrameReference);
	}


	if (SUCCEEDED(hr))
	{
		INT64 nDepthTime = 0;
		IFrameDescription* pDepthFrameDescription = NULL;
		int nDepthWidth = 512;
		int nDepthHeight = 424;
		UINT nDepthBufferSize = 0;
		UINT16 *pDepthBuffer = NULL;
		USHORT nDepthMinReliableDistance = 500;
		USHORT nDepthMaxDistance = USHRT_MAX;

		IFrameDescription* pColorFrameDescription = NULL;
		int nColorWidth = 1920;
		int nColorHeight = 1080;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		UINT nColorBufferSize = 0;
		RGBQUAD *pColorBuffer = NULL;


		// get depth frame data
		hr = pDepthFrame->get_RelativeTime(&nDepthTime);

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameDescription->get_Width(&nDepthWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameDescription->get_Height(&nDepthHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pDepthFrame->get_DepthMinReliableDistance(&nDepthMinReliableDistance);
		}

		if (SUCCEEDED(hr))
		{
			nDepthMaxDistance = USHRT_MAX;
		}

		//if (SUCCEEDED(hr))
		//{
		//	hr = pDepthFrame->AccessUnderlyingBuffer(&nDepthBufferSize, &pDepthBuffer);
		//}

		//m_Depth = Mat(nDepthHeight, nDepthWidth, CV_16UC1, pDepthBuffer).clone();///////////////

		pDepthBuffer = m_Depth.ptr<ushort>(0);
		// get color frame data
		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameDescription->get_Width(&nColorWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameDescription->get_Height(&nColorHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
		}

		//if (SUCCEEDED(hr))
		//{
		//	if (imageFormat == ColorImageFormat_Bgra)
		//	{
		//		hr = pColorFrame->AccessRawUnderlyingBuffer(&nColorBufferSize, reinterpret_cast<BYTE**>(&pColorBuffer));
		//	}
		//	else if (m_pColorRGBX)
		//	{
		//		pColorBuffer = m_pColorRGBX;
		//		nColorBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
		//		hr = pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, reinterpret_cast<BYTE*>(pColorBuffer), ColorImageFormat_Bgra);
		//	}
		//	else
		//	{
		//		hr = E_FAIL;
		//	}
		//}

		//m_Color = Mat(nColorHeight, nColorWidth, CV_8UC4, pColorBuffer);///////////////
		pColorBuffer = m_Color.ptr<RGBQUAD>(0);









	
		ProcessFrame( pDepthBuffer, nDepthWidth, nDepthHeight, nDepthMinReliableDistance, nDepthMaxDistance,
			pColorBuffer, nColorWidth, nColorHeight);
		

		SafeRelease(pDepthFrameDescription);
		SafeRelease(pColorFrameDescription);
	}

	SafeRelease(pDepthFrame);
	SafeRelease(pColorFrame);
	SafeRelease(pMultiSourceFrame);
	return hr;
}
void VisualOdometry::ProcessFrame(
	const UINT16* pDepthBuffer, int nDepthWidth, int nDepthHeight, USHORT nMinDepth, USHORT nMaxDepth,
	const RGBQUAD* pColorBuffer, int nColorWidth, int nColorHeight)
{
	LARGE_INTEGER qpcNow = { 0 };
	WCHAR szStatusMessage[64];

	// Make sure we've received valid data
	if (m_pCoordinateMapper && m_pCameraCoordinates && 
		pDepthBuffer && pColorBuffer&&
		m_pDepthRGBX)
	{
		HRESULT hr = m_pCoordinateMapper->MapColorFrameToCameraSpace(nDepthWidth * nDepthHeight,
			(UINT16*)pDepthBuffer, nColorWidth * nColorHeight, m_pCameraCoordinates);
		if (FAILED(hr))
		{
			return;
		}

	}//确保参数都准确

	//imshow("color", m_Color);
	//resize(m_Color, m_Color, Size(cColorWidth / 2, cColorHeight / 2));
	//cvtColor(m_Color, m_Color, CV_BGR2GRAY);
	//imshow("Color", m_Color);										//显示彩色图像
										//显示深度图像






	//std::vector<CameraSpacePoint> camerasps;
	//std::vector<ColorSpacePoint> colorsps;
	//CameraSpacePoint* cameraSpacePoints = new CameraSpacePoint[cColorWidth * cColorHeight];
	//HRESULT hr=m_pCoordinateMapper->MapColorFrameToCameraSpace(cDepthWidth * cDepthHeight, pDepthBuffer, cColorWidth * cColorHeight, cameraSpacePoints);
	//if (SUCCEEDED(hr))
	//{
	//	for (ColorSpacePoint colorsp : colorsps)
	//	{
	//		int colorX = static_cast<int>(colorsp.X + 0.5f);
	//		int colorY = static_cast<int>(colorsp.Y + 0.5f);
	//		long colorIndex = (long)(colorY * cColorWidth + colorX);
	//		CameraSpacePoint csp = cameraSpacePoints[colorIndex];
	//		camerasps.push_back(CameraSpacePoint{ -csp.X, -csp.Y, csp.Z });
	//	}
	//}
	//isnan(p1_x_) || isnan(p2_x_) ||    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;















	RGBQUAD* pRGBX = m_pDepthRGBX;
	// end pixel is start + width*height - 1
	const UINT16* pBufferEnd = pDepthBuffer + (nDepthWidth * nDepthHeight);
	while (pDepthBuffer < pBufferEnd)
	{
		USHORT depth = *pDepthBuffer;
		BYTE intensity = static_cast<BYTE>((depth >= nMinDepth) && (depth <= nMaxDepth) ? (depth % 256) : 0);
		pRGBX->rgbRed = intensity;
		pRGBX->rgbGreen = intensity;
		pRGBX->rgbBlue = intensity;
		++pRGBX;
		++pDepthBuffer;
	}

	// Draw the data nDepthHeight OpenCV
	Mat DepthImage(nDepthHeight, nDepthWidth, CV_8UC4, m_pDepthRGBX);
	Mat show = DepthImage.clone();
	imshow("DepthImage", show);



















	waitKey(1);
}