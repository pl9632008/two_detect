
#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
// #include <opencv2/xfeatures2d/nonfree.hpp>
#include <atomic>
#include <fstream>
#include <cmath>
#include <experimental/filesystem>
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "ini.h"
using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

struct RailInfo{
    int height;
    int width;
    int y_limit;
    float confidence;
    int near_turn_distance;
    int back_near_turn_distance;

    int near_left_limit = 120;
    int near_right_limit = 1800;

    int back_near_left_limit = 120;
    int back_near_right_limit = 1800;

    std::string matrix_config_path;
    std::string yolo_engine_path;
    std::string tinyvit_engine_path;
};

struct MarkInfo{
 
    int p_left_top_x;
    int p_left_top_y;

    int p_left_bottom_x;
    int p_left_bottom_y;

    int p_right_top_x;
    int p_right_top_y;

    int p_right_bottom_x;
    int p_right_bottom_y;

    int offset_bottom;
    int offset_top;
    int bottom_diff;
    int top_diff;

};

struct Configuration {
    std::string section;
    std::map<std::string, std::string> data;
};

struct TestMask{
    int ans;
    cv::Mat mask;

};

struct point_set
{
    int xmin;      // left
    int xmax;      // right
    int ymin;      // top
    int ymax;      // bottom
    int label;     // 标签类别 0:轨道 1:人 2:车 3:电动车自行车
    int type;      // 预警类别 1:不在轨道上 2：在轨道上 3: 非人，在轨道上
    int proximity; // 用于远近摄像头判断 0:近 1：远
    float confidence; //yolo置信度

};

class infer_rail
{
public:
    infer_rail();
    ~infer_rail();

    std::shared_ptr<yolo::Infer> YOLO;
    void init(RailInfo & rail_info, MarkInfo & mark_info_near, MarkInfo & mark_info_far, MarkInfo & back_mark_info_near, MarkInfo & back_mark_info_far);
    static yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }
    std::vector<point_set> infer_out(cv::Mat & image1, cv::Mat & image2);
    float point_distance(float left, float top, float right, float bottom);
    bool point_distanceV2(yolo::Box & test_obj);
    float point_distanceV3(yolo::Box & test_obj);
    float point_distanceV4(yolo::Box & test_obj);


      //img_scene:near, img_object:far
    // cv::Mat featureMatching(cv::Mat & img_scene, cv::Mat & img_object);
    // std::vector<point_set> featureTransform(std::vector<point_set> & box_set);
    cv::Mat preprocessNearRail(yolo::Box & obj );
    int preprocessFarRail(yolo::BoxArray & far_rail_objs, cv::Mat & current_near_rail_mask);
    TestMask findTotalMaskContours(cv::Mat & total_mask, cv::Mat & current_near_rail_mask );
    int findEdge(cv::Mat & rail_mask, const int & far_flag);
    std::vector<std::vector<cv::Point>> splitRailPoints(std::vector<cv::Point> & pts, const int & len);
    std::vector<bool> findAbnormalCluster( std::vector<std::vector<cv::Point>> & splits, const int & flag, const int & far_flag);
    int findMaxPosition(std::vector<bool>& temp);
    void fitLines(const int & flag);
    cv::Mat findOriginalMask(yolo::Box & obj );
    int nearRailNew(point_set & target,cv::Mat & rail_mask);
    std::vector<Configuration> readConfig(std::string & configPath);
    void setMatrix(std::vector<Configuration> & configurations );
    void setRailInfo(RailInfo & rail_info);
    void setMarkInfoNear(MarkInfo & mark_info_near);
    void setMarkInfoFar(MarkInfo & mark_info_far);
    void setMarkInfoNearBack(MarkInfo & back_mark_info_near);
    void setMarkInfoFarBack(MarkInfo & back_mark_info_far);
    bool setYolo();
    void setMatrix(ini::iniReader &config);
    std::vector<std::string> listFiles(const std::string& directory,const std::string & ext);

    Logger logger_;
    void loadEngine(const std::string& path );
    cv::Mat preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh); 
    int personClassification(cv::Mat & img);
    bool use_tinyvit_ = false;
    float * person_cls_in_ = new float[1*3*224*224]{};
    float * person_cls_out_ = new float[1*2]{};
    // std::unique_ptr<IRuntime> runtime_person_;
    // std::unique_ptr<ICudaEngine> engine_person_;
    // std::unique_ptr<IExecutionContext> context_person_;

    IRuntime* runtime_person_;
    ICudaEngine* engine_person_;
    IExecutionContext* context_person_;


    const int BATCH_SIZE_ = 1;
    const int PERSON_INPUT_W_ = 224;
    const int PERSON_INPUT_H_ = 224;
    const int CHANNELS_ = 3;
    const int CLASSES_ = 2; // 0: 非人  1：人
    const char * INPUT_NAMES_ = "input";
    const char * OUTPUT_NAMES_ = "output";

    int near_turn_distance_ = 700;
    int back_near_turn_distance_ = 600;
    bool image_empty_ = false;

    std::deque<int> que_;
    bool before_front_flag_ = true;


    std::atomic<bool> front_direction_ {true};
    std::string matrix_config_path_ ;
    std::string yolo_engine_path_ ;
    std::string tinyvit_engine_path_;
    int near_left_limit_ ;
    int near_right_limit_ ;
    int back_near_left_limit_;
    int back_near_right_limit_;

    int height_;
    int width_;
    int y_limit_;
    float confidence_;
    int lidar_distance_near_;//近处雷达
    int lidar_distance_far_;//远处雷达
    bool far_is_valid_;//远处雷达是否有效
    bool near_has_rail_ = false;

    cv::Mat H_ = {cv::Mat::zeros(3,3, CV_64FC1)};
    cv::Mat H_front_ = {cv::Mat::zeros(3,3, CV_64FC1)};
    cv::Mat H_back_  = {cv::Mat::zeros(3,3, CV_64FC1)};

    cv::Vec4f cal_line1_;
    cv::Vec4f cal_line2_;
    cv::Vec4f cal_line3_;
    cv::Vec4f cal_line4_;
    cv::Vec4f cal_line1_far_;
    cv::Vec4f cal_line2_far_;
    cv::Vec4f cal_line3_far_;
    cv::Vec4f cal_line4_far_;


 //近
    cv::Point p_left_top_ = {cv::Point(1082,362)};
    cv::Point p_left_bottom_ = {cv::Point(804,1068)};
    cv::Point p_right_top_ = {cv::Point(1114,370)};
    cv::Point p_right_bottom_= {cv::Point(1467,1050)};
    int offset_bottom_ = 60;
    int offset_top_ = 10;
    int bottom_diff_ = 662;
    int top_diff_ = 32;
    cv::Point p1_;
    cv::Point p2_;
    cv::Vec4f line1_;
    cv::Vec4f line2_;
    cv::Vec4f line3_;
    cv::Vec4f line4_;

//远
    cv::Point p_left_top_far_ = {cv::Point(1268,34)};
    cv::Point p_left_bottom_far_ = {cv::Point(826,1064)};
    cv::Point p_right_top_far_ = {cv::Point(1312,34)};
    cv::Point p_right_bottom_far_= {cv::Point(1875,1064)};
    int offset_bottom_far_ = 90;
    int offset_top_far_ = 20;
    int bottom_diff_far_ = 1049;
    int top_diff_far_ = 44;
    cv::Point p1_far_;
    cv::Point p2_far_;
    cv::Point p1_origin_far_;
    cv::Point p2_origin_far_;
    cv::Mat img_rail_mask_far_;
    cv::Vec4f line1_far_;
    cv::Vec4f line2_far_;
    cv::Vec4f line3_far_;
    cv::Vec4f line4_far_;




 //Back近
    cv::Point back_p_left_top_ = {cv::Point(1082,362)};
    cv::Point back_p_left_bottom_ = {cv::Point(804,1068)};
    cv::Point back_p_right_top_ = {cv::Point(1114,370)};
    cv::Point back_p_right_bottom_= {cv::Point(1467,1050)};
    int back_offset_bottom_ = 60;
    int back_offset_top_ = 10;
    int back_bottom_diff_ = 662;
    int back_top_diff_ = 32;
    cv::Vec4f back_line1_;
    cv::Vec4f back_line2_;
    cv::Vec4f back_line3_;
    cv::Vec4f back_line4_;

//Back远
    cv::Point back_p_left_top_far_ = {cv::Point(1268,34)};
    cv::Point back_p_left_bottom_far_ = {cv::Point(826,1064)};
    cv::Point back_p_right_top_far_ = {cv::Point(1312,34)};
    cv::Point back_p_right_bottom_far_= {cv::Point(1875,1064)};
    int back_offset_bottom_far_ =90;
    int back_offset_top_far_ = 20;
    int back_bottom_diff_far_ = 1049;
    int back_top_diff_far_ = 44;
    cv::Vec4f back_line1_far_;
    cv::Vec4f back_line2_far_;
    cv::Vec4f back_line3_far_;
    cv::Vec4f back_line4_far_;


//外部可视化
    cv::Mat img1_rail_mask_;
    cv::Mat img2_rail_mask_;

};

