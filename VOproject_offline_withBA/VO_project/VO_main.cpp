
#include "stdafx.h"
#include "Tools.h"
#include <windows.h>
#include "Kinect.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include "visual_odometry.h"
#include <Mmsystem.h>//需要 Winmm.lib库的支持 ----timeGetTime()

int main()
{
	//PinholeCamera *cam = new PinholeCamera(1241.0, 376.0,
	//	718.8560, 718.8560, 607.1928, 185.2157);
	VisualOdometry vo;

	//std::ofstream out("position.txt");
	HRESULT hr;
	std::ofstream out("position.txt");
	char text[100];
	int font_face = cv::FONT_HERSHEY_PLAIN;
	double font_scale = 1;
	int thickness = 1;
	cv::Point text_org(10, 50);
	cv::namedWindow("RGB camera", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
	cv::Mat traj = cv::Mat::zeros(700, 1200, CV_8UC3);//
	double x = 0.0, y = 0.0, z = 0.0;
	double a = 0.0, b = 0.0, c = 0.0;
	//used for draw the coner with random color
	RNG g_rng(12345);
	int r = 3;
	//cv::Mat img_resized;
	
	
	printf_s("start analysis Kinect for Windows V2 body data....");
	hr = vo.InitKinect();
	if (FAILED(hr))
	{
		return 0;
	}
	int img_id = 0;
	int draw_x_pre = 220;
	int draw_y_pre = 300;
	while (img_id<=82)
	{

		cv::Mat I = cv::Mat::zeros(3, 1, CV_64F);
		cv::setIdentity(I, 1.);
		
		//if (FAILED(hr))
		//{
		//	continue;
		//}

		


		//cout << img_id << endl;
		std::stringstream ss,ssd;
		ss << "D:/Visual Odometry/视觉里程计1/odometry2/机电楼数据/彩色图试验场11/"
			<< std::setw(6) << std::setfill('0') << img_id << ".jpg";
		ssd << "D:/Visual Odometry/视觉里程计1/odometry2/机电楼数据/深度图试验场11/"
			<< std::setw(6) << std::setfill('0') << img_id << ".xml";
		vo.m_Color=(cv::imread(ss.str().c_str(), 0));
		assert(!vo.m_Color.empty());
		FileStorage fs(ssd.str().c_str(), FileStorage::READ);
		fs["depth"] >> vo.m_Depth;
		//vo.m_Depth=(cv::imread(ssd.str().c_str(), 0));
		assert(!vo.m_Depth.empty());
		//imshow("Depth", vo.m_Depth);
		cv::Mat img = vo.m_Color;
		//imshow("BGR", vo.m_Color);
		while (1)
		{
			hr = vo.Update();
			if (SUCCEEDED(hr))
				break;
		}
		
		

		vo.addImage(img, img_id);
		cv::Mat cur_t = vo.getCurrentT();
		cv::Mat cur_R = vo.getCurrentR();
		if (cur_R.rows != 0)
		{
			I = cur_R*I;
		}
		if (I.rows != 0)
		{
			a = 50*I.at<double>(0);
			b = 50*I.at<double>(1);
			c = 50*I.at<double>(2);
		}
		int draw_X = int(a) + 900;
		int draw_Y = int(c) + 150;
		const double PI = 3.1415926;
		cv::Point pEnd = cv::Point(draw_X, draw_Y);
		cv::Point pStart = cv::Point(900, 150);
		cv::line(traj, pStart, pEnd, Scalar(0, 255, 0));
		Point arrow;
		double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
		arrow.x = pEnd.x + 10 * cos(angle + PI * 30 / 180);
		arrow.y = pEnd.y + 10 * sin(angle + PI * 30 / 180);
		cv::line(traj, pEnd, arrow, Scalar(0, 255, 0));
		arrow.x = pEnd.x + 10 * cos(angle - PI * 30 / 180);
		arrow.y = pEnd.y + 10 * sin(angle - PI * 30 / 180);
		cv::line(traj, pEnd, arrow, Scalar(0, 255, 0));

		//std::cout << "cur_R=" << cur_R << ";" << endl << endl;
		if (cur_t.rows != 0)
		{
			x = cur_t.at<double>(0);
			y = cur_t.at<double>(1);
			z = cur_t.at<double>(2);
		}
		cout << x << " " << y << " " << z << std::endl;
		out << x << " " << y << " " << z << std::endl;
		int draw_x = 50 * x + 220;
		int draw_y = 50 * z + 300;
		//cv::circle(traj, cv::Point(draw_x, draw_y), 1, Scalar(255, 0, 0), 1);
		cv::line(traj, cv::Point(draw_x, draw_y), cv::Point(draw_x_pre, draw_y_pre), Scalar(255, 0, 0), 2);
		cv::rectangle(traj, cv::Point(10, 30), cv::Point(580, 60), Scalar(0, 0, 0), CV_FILLED);
		sprintf_s(text, "the coordinate: x = %02fm y = %02fm z = %02fm", x, y, z);
		cv::putText(traj, text, text_org, font_face, font_scale, cv::Scalar::all(255), thickness, 8);
		draw_x_pre = draw_x;
		draw_y_pre = draw_y;
		////绘制检测到的角点

		//for (int i = 0; i < vo.px_cur_trans.size(); i++)
		//{
		//	//以随机的颜色绘制出角点
		//	circle(img, vo.px_cur_trans[i], r, Scalar(145, 175,
		//		143), -1, 8, 0);
		//}
		//for (int i = 0; i < vo.px_cur_rot.size(); i++)
		//{
		//	//以随机的颜色绘制出角点
		//	rectangle(img, vo.px_cur_rot[i],vo.px_cur_rot[i]+cv::Point2f (2,2),Scalar(145, 175,
		//		143), CV_FILLED);
		//}


		//cv::imshow("RGB camera", img);
		cv::imshow("Trajectory", traj);
		//cv::resize(img, img_resized, cv::Size(vo.cColorWidth / 2, vo.cColorHeight / 2));
		//cv::imshow("img_resized", img_resized);

		if (cv::waitKey(1) >= 0)
		{
			break;
		}
		img_id++;
	}
	//// 优化所有边
	//cout << "optimizing pose graph, vertices: " << vo.optimizer.vertices().size() << endl;
	//vo.optimizer.save("F:/study work/Visual Odometry dev/VO_practice_2016.4/VO_project离线处理withBA_录视频/VO_project/result_before.g2o");
	//vo.optimizer.initializeOptimization();
	//vo.optimizer.optimize(100); //可以指定优化步数
	//vo.optimizer.save("F:/study work/Visual Odometry dev/VO_practice_2016.4/VO_project离线处理withBA_录视频/VO_project/result_after.g2o");
	//cout << "Optimization done." << endl;
	//vo.optimizer.clear();
	//delete cam;
	out.close();
    endstop:
	system("pause");
	return 0;
}




