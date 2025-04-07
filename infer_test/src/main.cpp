#include "rail.h"
 
cv::Mat img_near;
std::mutex mtx_near;

void getFrameNear(cv::VideoCapture &cap_near){

    while(1){

        {
            std::lock_guard<std::mutex> lock(mtx_near);
            cap_near>>img_near;
        }

        if (img_near.empty()) {
  
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 500 毫秒
    }

}


cv::Mat img_far;
std::mutex mtx_far;

void getFrameFar(cv::VideoCapture &cap_far){

    while(1){

        {
            std::lock_guard<std::mutex> lock(mtx_far);
            cap_far>>img_far;
        }

        if (img_far.empty()) {
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 500 毫秒
    }

}




int main(int argc, char* argv[]) {

    infer_rail infer;

    std::string config_path = "../CameraIP.ini";
    auto configurations = infer.readConfig(config_path);

    RailInfo rail_info;
    MarkInfo mark_info_near;
    MarkInfo mark_info_far;

    MarkInfo back_mark_info_near;
    MarkInfo back_mark_info_far;

    std::string front_near_path;
    std::string front_far_path;
    std::string back_near_path;
    std::string back_far_path;



    for(auto config : configurations){

        if(config.section == "[ConfAI]"){
            
            rail_info.width = std::stod(config.data["width"]);
            rail_info.height = std::stod(config.data["height"]);
            rail_info.y_limit = std::stod(config.data["y_limit"]);
            rail_info.confidence = std::stod(config.data["confidence"]);
            rail_info.matrix_config_path = config.data["matrix_config_path"];
            rail_info.yolo_engine_path = config.data["yolo_engine_path"];
            rail_info.tinyvit_engine_path = config.data["tinyvit_engine_path"];
            rail_info.near_turn_distance = std::stoi(config.data["near_turn_distance"]);
            rail_info.back_near_turn_distance = std::stoi(config.data["back_near_turn_distance"]);
            rail_info.near_left_limit = std::stoi(config.data["near_left_limit"]);
            rail_info.near_right_limit = std::stoi(config.data["near_right_limit"]);
            rail_info.back_near_left_limit = std::stoi(config.data["back_near_left_limit"]);
            rail_info.back_near_right_limit = std::stoi(config.data["back_near_right_limit"]);

          
            mark_info_near.p_left_top_x = std::stod(config.data["p_left_top_x"]);
            mark_info_near.p_left_top_y = std::stod(config.data["p_left_top_y"]);
            mark_info_near.p_left_bottom_x = std::stod(config.data["p_left_bottom_x"]); 
            mark_info_near.p_left_bottom_y = std::stod(config.data["p_left_bottom_y"]);
            mark_info_near.p_right_top_x = std::stod(config.data["p_right_top_x"]);
            mark_info_near.p_right_top_y = std::stod(config.data["p_right_top_y"]);
            mark_info_near.p_right_bottom_x = std::stod(config.data["p_right_bottom_x"]);
            mark_info_near.p_right_bottom_y = std::stod(config.data["p_right_bottom_y"]);
            mark_info_near.offset_bottom = std::stod(config.data["offset_bottom"]);
            mark_info_near.offset_top = std::stod(config.data["offset_top"]);
            mark_info_near.bottom_diff = std::stod(config.data["bottom_diff"]);
            mark_info_near.top_diff = std::stod(config.data["top_diff"]);

            mark_info_far.p_left_top_x = std::stod(config.data["p_left_top_far_x"]);
            mark_info_far.p_left_top_y = std::stod(config.data["p_left_top_far_y"]);
            mark_info_far.p_left_bottom_x = std::stod(config.data["p_left_bottom_far_x"]); 
            mark_info_far.p_left_bottom_y = std::stod(config.data["p_left_bottom_far_y"]);
            mark_info_far.p_right_top_x = std::stod(config.data["p_right_top_far_x"]);
            mark_info_far.p_right_top_y = std::stod(config.data["p_right_top_far_y"]);
            mark_info_far.p_right_bottom_x = std::stod(config.data["p_right_bottom_far_x"]);
            mark_info_far.p_right_bottom_y = std::stod(config.data["p_right_bottom_far_y"]);
            mark_info_far.offset_bottom = std::stod(config.data["offset_bottom_far"]);
            mark_info_far.offset_top = std::stod(config.data["offset_top_far"]);
            mark_info_far.bottom_diff = std::stod(config.data["bottom_diff_far"]);
            mark_info_far.top_diff = std::stod(config.data["top_diff_far"]);


            back_mark_info_near.p_left_top_x = std::stod(config.data["back_p_left_top_x"]);
            back_mark_info_near.p_left_top_y = std::stod(config.data["back_p_left_top_y"]);
            back_mark_info_near.p_left_bottom_x = std::stod(config.data["back_p_left_bottom_x"]); 
            back_mark_info_near.p_left_bottom_y = std::stod(config.data["back_p_left_bottom_y"]);
            back_mark_info_near.p_right_top_x = std::stod(config.data["back_p_right_top_x"]);
            back_mark_info_near.p_right_top_y = std::stod(config.data["back_p_right_top_y"]);
            back_mark_info_near.p_right_bottom_x = std::stod(config.data["back_p_right_bottom_x"]);
            back_mark_info_near.p_right_bottom_y = std::stod(config.data["back_p_right_bottom_y"]);
            back_mark_info_near.offset_bottom = std::stod(config.data["back_offset_bottom"]);
            back_mark_info_near.offset_top = std::stod(config.data["back_offset_top"]);
            back_mark_info_near.bottom_diff = std::stod(config.data["back_bottom_diff"]);
            back_mark_info_near.top_diff = std::stod(config.data["back_top_diff"]);

            back_mark_info_far.p_left_top_x = std::stod(config.data["back_p_left_top_far_x"]);
            back_mark_info_far.p_left_top_y = std::stod(config.data["back_p_left_top_far_y"]);
            back_mark_info_far.p_left_bottom_x = std::stod(config.data["back_p_left_bottom_far_x"]); 
            back_mark_info_far.p_left_bottom_y = std::stod(config.data["back_p_left_bottom_far_y"]);
            back_mark_info_far.p_right_top_x = std::stod(config.data["back_p_right_top_far_x"]);
            back_mark_info_far.p_right_top_y = std::stod(config.data["back_p_right_top_far_y"]);
            back_mark_info_far.p_right_bottom_x = std::stod(config.data["back_p_right_bottom_far_x"]);
            back_mark_info_far.p_right_bottom_y = std::stod(config.data["back_p_right_bottom_far_y"]);
            back_mark_info_far.offset_bottom = std::stod(config.data["back_offset_bottom_far"]);
            back_mark_info_far.offset_top = std::stod(config.data["back_offset_top_far"]);
            back_mark_info_far.bottom_diff = std::stod(config.data["back_bottom_diff_far"]);
            back_mark_info_far.top_diff = std::stod(config.data["back_top_diff_far"]);

        }

        else if(config.section == "[Camera]"){
            front_near_path = config.data["rtspAddr"];

        }else if(config.section == "[Camera_Far]"){
            front_far_path = config.data["rtspAddr"];

        }else if(config.section == "[Camera_Back]"){
            back_near_path = config.data["rtspAddr"];

        }else if(config.section == "[Camera_Back_Far]"){
            back_far_path = config.data["rtspAddr"];
        }
        
    }

    std::string str(argv[1]);
    if(str=="qian"){
        infer.front_direction_ = true;
    }else if(str=="hou"){
        infer.front_direction_ = false;
        front_near_path = back_near_path;
        front_far_path = back_far_path;
    }




    infer.init(rail_info, mark_info_near, mark_info_far, back_mark_info_near, back_mark_info_far);

    cv::VideoCapture cap_near(front_near_path); // 替换为你的视频文件路径
    cv::VideoCapture cap_far(front_far_path); // 替换为你的视频文件路径
    
    int width = cap_far.get(cv::CAP_PROP_FRAME_WIDTH);             //帧宽度
    int height = cap_far.get(cv::CAP_PROP_FRAME_HEIGHT);           //帧高度
    int totalFrames = cap_far.get(cv::CAP_PROP_FRAME_COUNT);       //总帧数
    int frameRate = cap_far.get(cv::CAP_PROP_FPS);                 //帧率 x frames/s
    int fcc = cap_far.get(cv::CAP_PROP_FOURCC);

    std::cout << "视频宽度： " << width << std::endl;
    std::cout << "视频高度： " << height << std::endl;
    std::cout << "视频总帧数： " << totalFrames << std::endl;
    std::cout << "帧率： " << frameRate << std::endl;


    cv::Size frameSize(width, height); // Frame size (width x height)

    cv::VideoWriter video_near;
    cv::VideoWriter video_far;

    // video_near.open("../output_near.avi", fcc, frameRate, frameSize);
    // video_far.open("../output_far.avi", fcc, frameRate, frameSize);


    if(!cap_far.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    std::thread thread_near(getFrameNear,std::ref(cap_near));
    std::thread thread_far(getFrameFar,std::ref(cap_far));
    thread_near.detach();
    thread_far.detach();


    std::this_thread::sleep_for(std::chrono::milliseconds(3000));  // 500 毫秒


    
    while(1) {
        cv::Mat frame_far;
        cv::Mat frame_near;
        // cap_near>>frame_near;
        // cap_far>>frame_far;

        {
            std::lock_guard<std::mutex> lock(mtx_near);
            frame_near = img_near.clone();
        }

        {
            std::lock_guard<std::mutex> lock(mtx_far);
            frame_far = img_far.clone();
        }
        

        if(frame_far.empty() || frame_near.empty()){

            std::this_thread::sleep_for(std::chrono::milliseconds(5));  // 500 毫秒
 
            continue;
            // break;
        }
        
        cv::Size targetSize(1920, 1080); // wh

        if(frame_near.rows != 1080 || frame_near.cols != 1920 ) {
            cv::resize(frame_near, frame_near, targetSize);
        }    
        
        
        if(frame_far.rows != 1080 || frame_far.cols != 1920 ){
            cv::resize(frame_far, frame_far,targetSize);
        }
        
        // std::cout<<"rows = " << frame_near.rows<<std::endl;
        // std::cout<<"cols = " << frame_near.cols<<std::endl;


        std::vector<point_set> po = infer.infer_out(frame_near, frame_far);
        
        cv::Mat img1_rail_mk = infer.img1_rail_mask_;
        cv::Mat img2_rail_mk = infer.img2_rail_mask_;

        if(!img1_rail_mk.empty()){

            cv::imwrite("../frame_near_railmask.jpg",img1_rail_mk);
        }

        cv::Scalar color1 = cv::Scalar(56,0,255);
        cv::Scalar color2 = cv::Scalar(0,255,56);

        for(int row = 0 ; row < frame_near.rows ; row++){
            for(int col = 0 ; col < frame_near.cols ; col++){
                if( !img1_rail_mk.empty() && img1_rail_mk.at<uint8_t>(row,col) == 255){
                    frame_near.at<cv::Vec3b>(row,col)[0] = cv::saturate_cast<uint8_t>( 0.5 * frame_near.at<cv::Vec3b>(row,col)[0] + 0.5 * color1[0] );
                    frame_near.at<cv::Vec3b>(row,col)[1] = cv::saturate_cast<uint8_t>( 0.5 * frame_near.at<cv::Vec3b>(row,col)[1] + 0.5 * color1[1] );
                    frame_near.at<cv::Vec3b>(row,col)[2] = cv::saturate_cast<uint8_t>( 0.5 * frame_near.at<cv::Vec3b>(row,col)[2] + 0.5 * color1[2] );
                }
                if( !img2_rail_mk.empty() && img2_rail_mk.at<uint8_t>(row,col) == 255){

                    frame_near.at<cv::Vec3b>(row,col)[0] = cv::saturate_cast<uint8_t>( 0.5 * frame_near.at<cv::Vec3b>(row,col)[0] + 0.5 * color2[0] );
                    frame_near.at<cv::Vec3b>(row,col)[1] = cv::saturate_cast<uint8_t>( 0.5 * frame_near.at<cv::Vec3b>(row,col)[1] + 0.5 * color2[1] );
                    frame_near.at<cv::Vec3b>(row,col)[2] = cv::saturate_cast<uint8_t>( 0.5 * frame_near.at<cv::Vec3b>(row,col)[2] + 0.5 * color2[2] );
                }
            }
        }
        
        int offset_bottom ;
        int offset_top;
        cv::Point left_top;
        cv::Point left_bottom;
        cv::Point right_top;
        cv::Point right_bottom;

        int offset_bottom_far;
        int offset_top_far;
        cv::Point left_top_far;
        cv::Point left_bottom_far;
        cv::Point right_top_far;
        cv::Point right_bottom_far;

        if(infer.front_direction_){

            offset_bottom = infer.offset_bottom_;
            offset_top = infer.offset_top_;
            left_top = infer.p_left_top_;
            left_bottom = infer.p_left_bottom_;
            right_top = infer.p_right_top_;
            right_bottom = infer.p_right_bottom_;

            offset_bottom_far = infer.offset_bottom_far_;
            offset_top_far = infer.offset_top_far_;
            left_top_far = infer.p_left_top_far_;
            left_bottom_far = infer.p_left_bottom_far_;
            right_top_far = infer.p_right_top_far_;
            right_bottom_far = infer.p_right_bottom_far_;

        }else{

            offset_bottom = infer.back_offset_bottom_;
            offset_top = infer.back_offset_top_;
            left_top = infer.back_p_left_top_;
            left_bottom = infer.back_p_left_bottom_;
            right_top = infer.back_p_right_top_;
            right_bottom = infer.back_p_right_bottom_;

            offset_bottom_far = infer.back_offset_bottom_far_;
            offset_top_far = infer.back_offset_top_far_;
            left_top_far = infer.back_p_left_top_far_;
            left_bottom_far = infer.back_p_left_bottom_far_;
            right_top_far = infer.back_p_right_top_far_;
            right_bottom_far = infer.back_p_right_bottom_far_;
            
        }

        cv::line(frame_near, cv::Point(left_top.x - offset_top, left_top.y), cv::Point(left_bottom.x - offset_bottom, left_bottom.y), cv::Scalar(255,0,0),3);
        cv::line(frame_near, cv::Point(left_top.x             , left_top.y), cv::Point(left_bottom.x                , left_bottom.y), cv::Scalar(0,0,255),3);
        cv::line(frame_near, cv::Point(left_top.x + offset_top, left_top.y), cv::Point(left_bottom.x + offset_bottom, left_bottom.y), cv::Scalar(255,0,0),3);
        cv::line(frame_near, cv::Point(right_top.x - offset_top, right_top.y), cv::Point(right_bottom.x - offset_bottom, right_bottom.y), cv::Scalar(255,0,0),3);
        cv::line(frame_near, cv::Point(right_top.x             , right_top.y), cv::Point(right_bottom.x                , right_bottom.y), cv::Scalar(0,0,255),3);
        cv::line(frame_near, cv::Point(right_top.x + offset_top, right_top.y), cv::Point(right_bottom.x + offset_bottom, right_bottom.y), cv::Scalar(255,0,0),3);

        cv::line(frame_far, cv::Point(left_top_far.x - offset_top_far, left_top_far.y), cv::Point(left_bottom_far.x - offset_bottom_far, left_bottom_far.y), cv::Scalar(255,0,0),3);
        cv::line(frame_far, cv::Point(left_top_far.x                 , left_top_far.y), cv::Point(left_bottom_far.x                    , left_bottom_far.y), cv::Scalar(0,0,255),3);
        cv::line(frame_far, cv::Point(left_top_far.x + offset_top_far, left_top_far.y), cv::Point(left_bottom_far.x + offset_bottom_far, left_bottom_far.y), cv::Scalar(255,0,0),3);
        cv::line(frame_far, cv::Point(right_top_far.x - offset_top_far, right_top_far.y), cv::Point(right_bottom_far.x - offset_bottom_far, right_bottom_far.y), cv::Scalar(255,0,0),3);
        cv::line(frame_far, cv::Point(right_top_far.x                 , right_top_far.y), cv::Point(right_bottom_far.x                    , right_bottom_far.y), cv::Scalar(0,0,255),3);
        cv::line(frame_far, cv::Point(right_top_far.x + offset_top_far, right_top_far.y), cv::Point(right_bottom_far.x + offset_bottom_far, right_bottom_far.y), cv::Scalar(255,0,0),3);
        
        //远处本身雷达距离
        // int y_origin = std::max(infer.p1_origin_far_.y, infer.p2_origin_far_.y);
        // cv::line(frame_far,cv::Point(0,y_origin),cv::Point(1919,y_origin),cv::Scalar(240,32,160),3);

        if(infer.far_is_valid_){
            cv::line(frame_far,cv::Point(0,infer.lidar_distance_far_),cv::Point(1919,infer.lidar_distance_far_),cv::Scalar(240,32,160),3);
        }else{
            cv::line(frame_far,cv::Point(0, 1079),cv::Point(1919, 1079),cv::Scalar(240,32,160),3);
        }

        cv::line(frame_far,cv::Point(frame_far.cols/2, 0), cv::Point(frame_far.cols/2, 1079), cv::Scalar(0,0,0),3);

        cv::Mat img_rail_mk_far = infer.img_rail_mask_far_;
        if(infer.far_is_valid_){
            for(int row = 0 ; row < frame_far.rows ; row++){
                for(int col = 0 ; col < frame_far.cols ; col++){
                    if( !img_rail_mk_far.empty() && img_rail_mk_far.at<uint8_t>(row,col) == 255){
                        frame_far.at<cv::Vec3b>(row,col)[0] = cv::saturate_cast<uint8_t>( 0.5 * frame_far.at<cv::Vec3b>(row,col)[0] + 0.5 * color2[0] );
                        frame_far.at<cv::Vec3b>(row,col)[1] = cv::saturate_cast<uint8_t>( 0.5 * frame_far.at<cv::Vec3b>(row,col)[1] + 0.5 * color2[1] );
                        frame_far.at<cv::Vec3b>(row,col)[2] = cv::saturate_cast<uint8_t>( 0.5 * frame_far.at<cv::Vec3b>(row,col)[2] + 0.5 * color2[2] );
                    }
                }
            }
        }else{
            std::string test_str = "Far is invalid! Drop purple line distance!!!";
            cv::putText(frame_far,test_str,cv::Point(0,frame_far.rows/2), 3, 3,cv::Scalar(0, 0,255),3);
        }
        

        int cnt = 0;
        std::vector<std::string> names{"NO","person"};
        std::vector<std::string> labels{"rail","person","car", "vehicle"};

        for(auto i : po){
            if(i.proximity == 0 ){
                if(i.type == 1){
                    cv::rectangle(frame_near,cv::Point(i.xmin,i.ymin),cv::Point(i.xmax,i.ymax),cv::Scalar(255,0,0),3);
                    cv::putText(frame_near,labels[i.label],cv::Point(i.xmin,i.ymin), 2, 2,cv::Scalar(255,0,0),2);
                    // cv::imwrite("/home/nvidia/wzw/20240807/PDS_LRIALS_v2.0/two_dect/infer_test/result/type_1_"+std::to_string(cnt)+".jpg",frame_near);
                    // cnt++;

                }else if(i.type==2){
                    cv::rectangle(frame_near,cv::Point(i.xmin,i.ymin),cv::Point(i.xmax,i.ymax),cv::Scalar(0,0,255),3);
                    cv::putText(frame_near,labels[i.label],cv::Point(i.xmin,i.ymin), 2, 2,cv::Scalar(0,0,255),2);
                    // cv::imwrite("/home/nvidia/wzw/20240807/PDS_LRIALS_v2.0/two_dect/infer_test/result/type_2_"+std::to_string(cnt)+".jpg",frame_near);
                    // cnt++;

                }else if(i.type==3){
                    cv::rectangle(frame_near,cv::Point(i.xmin,i.ymin),cv::Point(i.xmax,i.ymax),cv::Scalar(0,255,0),3);
                    cv::putText(frame_near,names[0],cv::Point(i.xmin,i.ymin), 2, 2,cv::Scalar(0,255,0),2);
                    // cv::imwrite("/home/nvidia/wzw/20240807/PDS_LRIALS_v2.0/two_dect/infer_test/result/"+std::to_string(cnt)+".jpg",frame_near);
                    // cnt++;
                }

            }else if(i.proximity==1){

                //远处
                if(i.type == 1){
                    cv::rectangle(frame_far,cv::Point(i.xmin,i.ymin),cv::Point(i.xmax,i.ymax),cv::Scalar(255,0,0),3);
                    cv::putText(frame_far,labels[i.label],cv::Point(i.xmin,i.ymin), 2, 2,cv::Scalar(255,0,0),2);
                }else if(i.type==2){
                    cv::rectangle(frame_far,cv::Point(i.xmin,i.ymin),cv::Point(i.xmax,i.ymax),cv::Scalar(0,0,255),3);
                    cv::putText(frame_far,labels[i.label],cv::Point(i.xmin,i.ymin), 2, 2,cv::Scalar(0,0,255),2);
                }else if(i.type==3){
                    cv::rectangle(frame_far,cv::Point(i.xmin,i.ymin),cv::Point(i.xmax,i.ymax),cv::Scalar(0,255,0),3);
                    cv::putText(frame_far,names[0],cv::Point(i.xmin,i.ymin), 2, 2,cv::Scalar(0,255,0),2);
                }
            }else if(i.proximity == 2 ){
               //远处投影到近处的box，绿色，不管type 
                cv::rectangle(frame_near,cv::Point(i.xmin,i.ymin),cv::Point(i.xmax,i.ymax),cv::Scalar(0,255,0),3);
            }
        }

        // 投影，近处本身雷达距离和远处投影过来的雷达距离    
        // int y = std::max(infer.p1_.y, infer.p2_.y);
        // int y_far = std::max(infer.p1_far_.y, infer.p2_far_.y);
        // cv::line(frame_near,cv::Point(0,y),cv::Point(1919,y),cv::Scalar(0,0,0),3);
        // cv::line(frame_near,cv::Point(0,y_far),cv::Point(1919,y_far),cv::Scalar(240,32,160),3);

        cv::line(frame_near,cv::Point(0,infer.lidar_distance_near_),cv::Point(1919,infer.lidar_distance_near_),cv::Scalar(0,0,0),3);
        cv::line(frame_near,cv::Point(frame_near.cols/2, 0), cv::Point(frame_near.cols/2, 1079), cv::Scalar(0,0,0),3);

     	cv::namedWindow("frame_near",cv::WINDOW_NORMAL);
        cv::imwrite("../frame_near.jpg",frame_near);
        cv::imshow("frame_near",frame_near);


        cv::namedWindow("frame_far",cv::WINDOW_NORMAL);
        cv::imshow("frame_far",frame_far);
        cv::imwrite("../frame_far.jpg",frame_far);

        
        // 按 'q' 退出
        if (cv::waitKey(30) == 'q') break;

        // cv::destroyAllWindows();

        // video_near<<frame_near;
        // video_far<<frame_far;

    }
    cap_near.release();
    cap_far.release();

    video_near.release();
    video_far.release();

    cv::destroyAllWindows();
    return 0;
}



// int main(){

//     infer_rail infer;

//     infer.loadEngine("/home/nvidia/tinyvit.engine");

//     std::string directory = "/home/nvidia/wzw/20240807/PDS_LRIALS_v2.0/two_dect/infer_test/images/";

//     auto images = infer.listFiles(directory,".jpg");

//     std::vector<std::string> names{"NO","person"};
//     int cnt = 0;

//     for(auto i: images){
//         cv::Mat img = cv::imread(i);
//         std::cout<<i<<std::endl;
//         int result = infer.personClassification(img);

//         int x = img.cols/2;
//         int y = img.rows/2;

//         cv::putText(img,names[result],cv::Point(0,y),2,2,cv::Scalar(0,0,255),2);
//         cv::imwrite("/home/nvidia/wzw/20240807/PDS_LRIALS_v2.0/two_dect/infer_test/result/"+std::to_string(cnt)+".jpg",img);
//         cnt++;
//     }

// }