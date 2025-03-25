#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


void HapLidarTopicCallbackPointCloud2(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::fromROSMsg(*msg, *cloud);

    ROS_INFO("Received point cloud with %zu points", cloud->size());

    for (const auto& point : cloud->points)
    {
        float x = point.x;
        float y = point.y;
        float z = point.z;
        float intensity = point.intensity;

        ROS_INFO("Point: x=%.3f, y=%.3f, z=%.3f, intensity=%.3f", x, y, z, intensity);
    }


}






void markerArrayCallback(const visualization_msgs::MarkerArray::ConstPtr& msg) {
    
    for (const auto& marker : msg->markers) {
        if (!marker.points.empty()) {
            ROS_INFO("Marker ID: %d, Type: %d, Points size: %lu", 
                     marker.id, marker.type, marker.points.size());

            for (size_t i = 0; i < marker.points.size(); ++i) {
                const geometry_msgs::Point& p = marker.points[i];
                ROS_INFO("  Point %lu: (%.2f, %.2f, %.2f)", 
                         i, p.x, p.y, p.z);
            }
        } else {
            ROS_WARN("Marker ID: %d has no points!", marker.id);
        }
    }
}


int main(int argc, char** argv) {
    ros::init(argc, argv, "marker_array_subscriber");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe<visualization_msgs::MarkerArray>(
        "/detect_box3d", 1, markerArrayCallback);

    // ros::Subscriber pc_sub = nh.subscribe("/livox/lidar", 1, HapLidarTopicCallbackPointCloud2);


    ROS_INFO("Subscriber node is ready...");
    ros::spin();  

    return 0;
}