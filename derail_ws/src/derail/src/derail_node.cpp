#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/Quaternion.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <deque>
#include <cmath>

std::deque<double> roll_window;
std::deque<double> pitch_window;

int window_size = 10;
double roll_variance_threshold = 0.01;  
double pitch_variance_threshold = 0.01;


double computeVariance(const std::deque<double>& data)
{
    if (data.size() < 2) return 0.0;
    
    double mean = 0.0;
    for (double val : data) {
        mean += val;
    }
    mean /= data.size();

    double variance = 0.0;
    for (double val : data) {
        variance += (val - mean) * (val - mean);
    }
    variance /= data.size();
    
    return variance;
}


ros::Publisher roll_variance_pub;
ros::Publisher pitch_variance_pub;
ros::Publisher trigger_pub;

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    geometry_msgs::Quaternion quat_msg = msg->pose.pose.orientation;
    
    tf2::Quaternion quat;
    tf2::fromMsg(quat_msg, quat);

    tf2::Matrix3x3 m(quat);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);


    if (roll_window.size() >= window_size) {
        roll_window.pop_front();  
        pitch_window.pop_front();
    }
    roll_window.push_back(roll);
    pitch_window.push_back(pitch);

    double roll_variance = computeVariance(roll_window);
    double pitch_variance = computeVariance(pitch_window);


    ROS_INFO("roll_variance = %lf", roll_variance);
    ROS_INFO("pitch_variance = %lf", pitch_variance);

    std_msgs::Float64 roll_variance_msg;
    std_msgs::Float64 pitch_variance_msg;
    roll_variance_msg.data = roll_variance;
    pitch_variance_msg.data = pitch_variance;

    roll_variance_pub.publish(roll_variance_msg);
    pitch_variance_pub.publish(pitch_variance_msg);

    if (roll_variance > roll_variance_threshold || pitch_variance > pitch_variance_threshold) {
        ROS_WARN("Roll or Pitch has sudden change: Roll variance = %f, Pitch variance = %f", roll_variance, pitch_variance);
        std_msgs::Int32 trigger_msg;
        trigger_msg.data = 1;  
        trigger_pub.publish(trigger_msg);
    }
    

}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "derail_node");
    ros::NodeHandle nh;


    nh.param("window_size", window_size, 10);  
    nh.param("roll_variance_threshold", roll_variance_threshold, 0.01);  
    nh.param("pitch_variance_threshold", pitch_variance_threshold, 0.01); 

    ros::Subscriber sub = nh.subscribe("/aft_mapped_to_init", 10, odomCallback);

    roll_variance_pub = nh.advertise<std_msgs::Float64>("/roll_variance", 10);
    pitch_variance_pub = nh.advertise<std_msgs::Float64>("/pitch_variance", 10);
    trigger_pub = nh.advertise<std_msgs::Int32>("/out_rail", 10);

    ros::spin();
    return 0;
}
