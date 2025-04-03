#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include "common.h"
#include "result_verify.h"


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;
using namespace cv;

void getColor(int &result_r, int &result_g, int &result_b, float cur_depth);

float max_depth = 60;
float min_depth = 3;

cv::Mat src_img;

int threshold_lidar;  // number of cloud point on the photo
string input_bag_path, input_photo_path, output_path, intrinsic_path, extrinsic_path;



// set the color by distance to the cloud
void getColor(int &result_r, int &result_g, int &result_b, float cur_depth) {
    float scale = (max_depth - min_depth)/10;
    if (cur_depth < min_depth) {
        result_r = 0;
        result_g = 0;
        result_b = 0xff;
    }
    else if (cur_depth < min_depth + scale) {
        result_r = 0;
        result_g = int((cur_depth - min_depth) / scale * 255) & 0xff;
        result_b = 0xff;
    }
    else if (cur_depth < min_depth + scale*2) {
        result_r = 0;
        result_g = 0xff;
        result_b = (0xff - int((cur_depth - min_depth - scale) / scale * 255)) & 0xff;
    }
    else if (cur_depth < min_depth + scale*4) {
        result_r = int((cur_depth - min_depth - scale*2) / scale * 255) & 0xff;
        result_g = 0xff;
        result_b = 0;
    }
    else if (cur_depth < min_depth + scale*7) {
        result_r = 0xff;
        result_g = (0xff - int((cur_depth - min_depth - scale*4) / scale * 255)) & 0xff;
        result_b = 0;
    }
    else if (cur_depth < min_depth + scale*10) {
        result_r = 0xff;
        result_g = 0;
        result_b = int((cur_depth - min_depth - scale*7) / scale * 255) & 0xff;
    }
    else {
        result_r = 0xff;
        result_g = 0;
        result_b = 0xff;
    }

}


int main(int argc, char **argv) {
    ros::init(argc, argv, "projectCloud");

    src_img = cv::imread("../data/test.jpg");

    vector<float> intrinsic{
        3856.78,0,945.335,
        0,3829.08,580.422,
        0,0,1
    };
    vector<float> distortion{
        -0.37972,0.130612,-0.00458177,-0.00441604,0
    };
    vector<float> extrinsic{
        -0.0296049,-0.999515,-0.00968518,0.219986,
        0.0132464,0.00929621,-0.999868,-0.024571,
        0.999473,-0.0297295,0.0129648,0.00271523,
        0,0,0,1
    };

	// set intrinsic parameters of the camera
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = intrinsic[0];
    cameraMatrix.at<double>(0, 2) = intrinsic[2];
    cameraMatrix.at<double>(1, 1) = intrinsic[4];
    cameraMatrix.at<double>(1, 2) = intrinsic[5];

	// set radial distortion and tangential distortion
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = distortion[0];
    distCoeffs.at<double>(1, 0) = distortion[1];
    distCoeffs.at<double>(2, 0) = distortion[2];
    distCoeffs.at<double>(3, 0) = distortion[3];
    distCoeffs.at<double>(4, 0) = distortion[4];


    ROS_INFO("Start to project the lidar cloud");
    float x, y, z;
    float theoryUV[2] = {0, 0};
    int myCount = 0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/test.pcd", *cloud) == -1) {
        PCL_ERROR("无法读取 PCD 文件\n");
        return -1;
    }

    for (size_t i = 0; i < cloud->points.size(); ++i) {

        x = cloud->points[i].x;
        y = cloud->points[i].y ;
        z = cloud->points[i].z ;

        getTheoreticalUV(theoryUV, intrinsic, extrinsic, x, y, z);
        int u = floor(theoryUV[0] + 0.5);
        int v = floor(theoryUV[1] + 0.5);
        int r,g,b;
        getColor(r, g, b, x);

        Point p(u, v);
        circle(src_img, p, 1, Scalar(b, g, r), -1);
        ++myCount;
        if (myCount > 1000000) {
            break;
        }

    }


    ROS_INFO("Show and save the picture, tap any key to close the photo");

    // cv::Mat view, rview, map1, map2;
    // cv::Size imageSize = src_img.size();
    // cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
    // cv::remap(src_img, src_img, map1, map2, cv::INTER_LINEAR);  // correct the distortion
    // cv::namedWindow("source", cv::WINDOW_KEEPRATIO);
    
    cv::imshow("source", src_img);
    cv::waitKey(0);
    cv::imwrite("../data/result.jpg", src_img);
    return 0;
}