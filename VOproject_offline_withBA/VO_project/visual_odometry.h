#ifndef VISUAL_ODOMETRY_H_
#define VISUAL_ODOMETRY_H_
//#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <Eigen/Eigen>
#include <Kinect.h>// Kinect Header files
//#include <boost/concept_check.hpp>
//// for g2o
//#include <g2o/core/sparse_optimizer.h>
//#include <g2o/core/block_solver.h>
//#include <g2o/core/robust_kernel.h>
//#include <g2o/core/robust_kernel_impl.h>
//#include <g2o/core/optimization_algorithm_levenberg.h>
////#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
//#include <g2o/solvers/eigen/linear_solver_eigen.h>
//#include <g2o/types/slam3d/se3quat.h>
//#include <g2o/types/sba/types_six_dof_expmap.h>
//#include <g2o/types/slam3d/types_slam3d.h> //顶点类型
//#include <g2o/core/factory.h>
//#include <g2o/core/optimization_algorithm_factory.h>
//#include <g2o/core/optimization_algorithm_gauss_newton.h>
//#include <g2o/solvers/csparse/linear_solver_csparse.h>
//#include <g2o/core/robust_kernel_factory.h>




using namespace cv;
// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}

class VisualOdometry
{
public:

	enum FrameStage {
        STAGE_FIRST_FRAME,//
        STAGE_SECOND_FRAME,//
        STAGE_DEFAULT_FRAME//
	};

	VisualOdometry();
	virtual ~VisualOdometry();
	///
	HRESULT					InitKinect();//初始化Kinect
	HRESULT					Update();//更新数据
	void                    ProcessFrame(
		const UINT16* pDepthBuffer, int nDepthHeight, int nDepthWidth, USHORT nMinDepth, USHORT nMaxDepth,
		const RGBQUAD* pColorBuffer, int nColorWidth, int nColorHeight);
    ///
	void addImage(const cv::Mat& img, int frame_id);
	void bundleAdjustment(std::vector<cv::Mat>& frames_, std::vector<cv::Mat>& transes_, std::vector<cv::Mat>& rotations_,
		std::vector<std::vector<CameraSpacePoint>>& camera_space_points_, std::vector<std::vector<cv::Point2f>>& pxes_ref_);
    ///
	cv::Mat getCurrentR() { return cur_R_; }
    ///
	cv::Mat getCurrentT() { return cur_t_; }

	cv::Mat getm_Color() { return m_Color; }

	bool ifORB = false;
	bool ifsolvePnP = false;
	bool ifdetectline = true;
	bool iffast = false;
	bool iffindEssential = true;

	std::vector<CameraSpacePoint> get3dposition(std::vector<Point2f> points);
	void RemoveInvalidData(std::vector<CameraSpacePoint>& camera_space_points1,
							std::vector<CameraSpacePoint>& camera_space_points2);

	float getRalativeScale(std::vector<CameraSpacePoint> camera_space_points1,
							std::vector<CameraSpacePoint> camera_space_points2);

	cv::Mat getTrans(std::vector<CameraSpacePoint> camera_space_points1,
		std::vector<CameraSpacePoint> camera_space_points2, cv::Mat R, cv::Mat t, cv::Mat mask);
	//------------------------------------------------------------------------------------
	IKinectSensor*          m_pKinectSensor;// Current Kinect
	// Frame reader
	IMultiSourceFrameReader*m_pMultiSourceFrameReader;

	ICoordinateMapper*      m_pCoordinateMapper;
	CameraSpacePoint*        m_pCameraCoordinates;
	RGBQUAD*                m_pOutputRGBX;
	RGBQUAD*                m_pColorRGBX;
	RGBQUAD*                m_pBackgroundRGBX;
	RGBQUAD*                m_pDepthRGBX;

	Mat						m_Depth;
	static const int        cDepthWidth = 512;
	static const int        cDepthHeight = 424;
	static const int        cColorWidth = 1920;
	static const int        cColorHeight = 1080;
	Mat						m_Color;

	std::vector<cv::Point2f> px_ref_;      
	std::vector<cv::Point2f> px_cur_;    
	std::vector<cv::KeyPoint> kp_ref_;
	std::vector<cv::KeyPoint> kp_cur_;
	cv::Mat orb_descriptor_ref_;
	cv::Mat orb_descriptor_cur_;

	std::vector<cv::Point2f> px_ref_rot;      
	std::vector<cv::Point2f> px_ref_trans;      
	std::vector<cv::Point2f> px_cur_rot;
	std::vector<cv::Point2f> px_cur_trans;

	std::vector<CameraSpacePoint> px_ref_camera;
	std::vector<CameraSpacePoint> px_cur_camera;

	std::vector<CameraSpacePoint> px_ref_camera_rot;
	std::vector<CameraSpacePoint> px_ref_camera_trans;
	std::vector<CameraSpacePoint> px_cur_camera_rot;
	std::vector<CameraSpacePoint> px_cur_camera_trans;

	//----------------------------------------------------------------------------------------
	////BA用到的变量
	//g2o::SparseOptimizer     optimizer;
	////  在这里不能使用 Cholmod 求解器了，还是因为没有 Cholmod 模块，但是 Eigen 的肯定能使
	//g2o::BlockSolver_6_3::LinearSolverType*  linearSolver;	//这里本来是g2o::LinearSolverCSparse，但是我没装这个包
	//g2o::BlockSolver_6_3*  block_solver;		// 6*3  的参数
	//g2o::OptimizationAlgorithmLevenberg*  algorithm;		// L-M  下降

//private:



	//-----------------------------------------------------------------------------------------
protected:



	std::vector< cv::DMatch > matches;
	cv::Mat match_img;

    ///
	virtual bool processFirstFrame();
    ///
	virtual bool processSecondFrame(int frame_id);
    ///
	virtual bool processFrame(int frame_id);
    ///
	
	cv::Mat rectifyR(cv::Mat R);
	cv::Mat rectifyT(cv::Mat t);
	int eightPointAugorithm(const CvMat* _m1, const CvMat* _m2, CvMat* _fmatrix);

	std::vector<cv::Point3f> PointCamera2Point3f(std::vector<CameraSpacePoint> points_camera,
		std::vector<cv::Point2f>& pixl_ref_local);

	cv::Mat sharpenImage1(cv::Mat input);

	void SaperateFeaturesIntoTwoGroup(std::vector<CameraSpacePoint> camera_space_points_ref,
		std::vector<CameraSpacePoint> camera_space_points_cur, std::vector<CameraSpacePoint>& camera_space_points_ref_rot,
		std::vector<CameraSpacePoint>& camera_space_points_ref_trans, std::vector<CameraSpacePoint>& camera_space_points_cur_rot,
		std::vector<CameraSpacePoint>& camera_space_points_cur_trans, std::vector<cv::Point2f> px_ref_, std::vector<cv::Point2f> px_cur_,
		std::vector<cv::Point2f>& px_ref_rot, std::vector<cv::Point2f>& px_ref_trans,
		std::vector<cv::Point2f>& px_cur_rot, std::vector<cv::Point2f>& px_cur_trans);

	double excludeOutliers(std::vector<double> dispariaties);
    ///
	void featureDetection(cv::Mat image, std::vector<cv::Point2f> &px_vec);
    ///
	void featureDetectionLines(cv::Mat image, std::vector<cv::Point2f> &px_vec);

	void featureDetectionORB(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor, std::vector<cv::Point2f>& px);

	std::vector< cv::DMatch > ORBmatch(cv::Mat descriptor1, cv::Mat descriptor2, std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2,
		std::vector<cv::Point2f>& px_ref, std::vector<cv::Point2f>& px_cur, std::vector<CameraSpacePoint>& px_ref_camera);
	
	void featureTracking(cv::Mat image_ref, cv::Mat image_cur,
		std::vector<cv::Point2f>& px_ref, std::vector<cv::Point2f>& px_cur, std::vector<double>& disparities
		,std::vector<CameraSpacePoint>& px_ref_camera);
	

    FrameStage frame_stage_;                 //
   // PinholeCamera *cam_;                     //
    cv::Mat new_frame_;                      //
    cv::Mat last_frame_;                     //

    cv::Mat cur_R_;//
    cv::Mat cur_t_;//
	float scale;

	std::vector<cv::Mat> transes_;
	std::vector<cv::Mat> rotations_;
	std::vector<std::vector<CameraSpacePoint>> camera_space_points_;
	std::vector<cv::Mat> frames_;
	std::vector<std::vector<cv::Point2f>> pxes_;

	int frame_count_;

    std::vector<double> disparities_;      //

    double focal_;//
    cv::Point2d pp_; //
	//double camera_matrix_data[3][3];
	//cv::Mat cameraMatrix;

};

#endif // VISUAL_ODOMETRY_H_
