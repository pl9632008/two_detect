#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

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
        "/detect_box3d", 10, markerArrayCallback);

    ROS_INFO("Subscriber node is ready...");
    ros::spin();  

    return 0;
}