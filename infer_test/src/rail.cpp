#include "rail.h"

infer_rail::infer_rail()
{
  
}

infer_rail::~infer_rail(){
    delete[] person_cls_in_;
    delete[] person_cls_out_;
}


void infer_rail::loadEngine(const std::string& path ) {
    size_t size{ 0 };
    char* trtModelStream{ nullptr };
    std::ifstream file(path, std::ios::binary);

    if (file.good()) {
        file.seekg(0, std::ios::end);
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }

    // std::unique_ptr<IRuntime> runtime(createInferRuntime(logger_));
    // std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(trtModelStream,size));
    // std::unique_ptr<IExecutionContext>context(engine->createExecutionContext());

    // runtime_person_ = std::move(runtime);
    // engine_person_  = std::move(engine);
    // context_person_ = std::move(context);

    runtime_person_ = createInferRuntime(logger_);
    engine_person_= runtime_person_->deserializeCudaEngine(trtModelStream,size);
    context_person_ = engine_person_->createExecutionContext();

 
    delete[] trtModelStream;
}


cv::Mat infer_rail::preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(127, 127, 127));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    padw = (input_w - w) / 2;
    padh = (input_h - h) / 2;
    return out;
}


std::vector<std::string> infer_rail::listFiles(const std::string& directory,const std::string & ext) {

    std::vector<std::string> total_names;

    std::experimental::filesystem::path p(directory);
    for(auto & entry : std::experimental::filesystem::directory_iterator(p)){
        if(entry.path().extension().string() == ext){
            total_names.push_back(entry.path().string());
        }
    }
    return total_names;
}



int infer_rail::personClassification(cv::Mat & img){


    int padw,padh;
    cv::Mat img_pad = preprocessImg(img, PERSON_INPUT_W_, PERSON_INPUT_H_, padw, padh);

    int input_index = engine_person_->getBindingIndex(INPUT_NAMES_);
    int output_index = engine_person_->getBindingIndex(OUTPUT_NAMES_);

    void* buffers[2];
    cudaMalloc(&buffers[input_index], BATCH_SIZE_ * CHANNELS_ * PERSON_INPUT_W_ * PERSON_INPUT_H_ * sizeof(float));
    cudaMalloc(&buffers[output_index], BATCH_SIZE_ * CLASSES_ * sizeof(float));

    for (int i = 0; i < PERSON_INPUT_W_ * PERSON_INPUT_H_ ; i++) {
        person_cls_in_[i] = img_pad.at<cv::Vec3b>(i)[2] / 1.0;
        person_cls_in_[i + PERSON_INPUT_W_ * PERSON_INPUT_H_ ] = img_pad.at<cv::Vec3b>(i)[1] / 1.0;
        person_cls_in_[i + 2 * PERSON_INPUT_W_ * PERSON_INPUT_H_ ] = img_pad.at<cv::Vec3b>(i)[0] / 1.0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[input_index], person_cls_in_, BATCH_SIZE_ * CHANNELS_ * PERSON_INPUT_W_ * PERSON_INPUT_H_  * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_person_->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(person_cls_out_, buffers[output_index], BATCH_SIZE_ * CLASSES_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output_index]);

    int index = std::max_element( person_cls_out_, person_cls_out_ + CLASSES_ ) - person_cls_out_;
    // std::cout<<"not person score :" <<person_cls_out_[0]<<"  person score : "<<person_cls_out_[1]<<std::endl;

    return index;
}




void infer_rail::init(RailInfo & rail_info, MarkInfo & mark_info_near, MarkInfo & mark_info_far, MarkInfo & back_mark_info_near, MarkInfo & back_mark_info_far){

  setRailInfo(rail_info);
  setMarkInfoNear(mark_info_near);
  setMarkInfoFar(mark_info_far);
  setMarkInfoNearBack(back_mark_info_near);
  setMarkInfoFarBack(back_mark_info_far);

  bool succeed = setYolo();
  if(!succeed) return;
  std::cout<<"load yolo successfully!"<<std::endl;
  
  if(std::experimental::filesystem::exists(tinyvit_engine_path_)){
      use_tinyvit_ = true;
      std::cout<<"begin to load engine"<<std::endl;
      loadEngine(tinyvit_engine_path_);
      std::cout<<"load tinyvit successfully!"<<std::endl;
  }
  
  ini::iniReader config;
  bool ret = config.ReadConfig(matrix_config_path_);
  if(!ret){
    std::cout<<"read Matrix failed!"<<std::endl;
  };

  setMatrix(config);

  std::cout<<"set Matrix successfully!"<<std::endl;

  fitLines(0);
  fitLines(1);
  fitLines(2);
  fitLines(3);
  std::cout<<"fit Lines successfully!"<<std::endl;


}

void infer_rail::setRailInfo(RailInfo & rail_info){

  width_                   =  rail_info.width;
  height_                  =  rail_info.height;
  y_limit_                 =  rail_info.y_limit;
  matrix_config_path_      =  rail_info.matrix_config_path;
  yolo_engine_path_        =  rail_info.yolo_engine_path;
  tinyvit_engine_path_     =  rail_info.tinyvit_engine_path;
  confidence_              =  rail_info.confidence;
  near_turn_distance_      =  rail_info.near_turn_distance;
  back_near_turn_distance_ =  rail_info.back_near_turn_distance;
  near_left_limit_         =  rail_info.near_left_limit;
  near_right_limit_        =  rail_info.near_right_limit;
  back_near_left_limit_    =  rail_info.back_near_left_limit;
  back_near_right_limit_   =  rail_info.back_near_right_limit;


}

void infer_rail::setMarkInfoNear(MarkInfo & mark_info_near){
    
    p_left_top_    =  cv::Point(mark_info_near.p_left_top_x    ,  mark_info_near.p_left_top_y    );
    p_left_bottom_ =  cv::Point(mark_info_near.p_left_bottom_x ,  mark_info_near.p_left_bottom_y );
  
    p_right_top_   =  cv::Point(mark_info_near.p_right_top_x    , mark_info_near.p_right_top_y   );
    p_right_bottom_=  cv::Point(mark_info_near.p_right_bottom_x , mark_info_near.p_right_bottom_y);
    
    offset_bottom_ =  mark_info_near.offset_bottom;
    offset_top_    =  mark_info_near.offset_top;
    bottom_diff_   =  mark_info_near.bottom_diff;
    top_diff_      =  mark_info_near.top_diff;

}


void infer_rail::setMarkInfoFar(MarkInfo & mark_info_far){
    
    p_left_top_far_    =  cv::Point(mark_info_far.p_left_top_x    ,  mark_info_far.p_left_top_y    );
    p_left_bottom_far_ =  cv::Point(mark_info_far.p_left_bottom_x ,  mark_info_far.p_left_bottom_y );
  
    p_right_top_far_   =  cv::Point(mark_info_far.p_right_top_x    , mark_info_far.p_right_top_y   );
    p_right_bottom_far_=  cv::Point(mark_info_far.p_right_bottom_x , mark_info_far.p_right_bottom_y);
    
    offset_bottom_far_ =  mark_info_far.offset_bottom;
    offset_top_far_    =  mark_info_far.offset_top;
    bottom_diff_far_   =  mark_info_far.bottom_diff;
    top_diff_far_      =  mark_info_far.top_diff;

}


void infer_rail::setMarkInfoNearBack(MarkInfo & back_mark_info_near){
    
    back_p_left_top_    =  cv::Point(back_mark_info_near.p_left_top_x    ,  back_mark_info_near.p_left_top_y    );
    back_p_left_bottom_ =  cv::Point(back_mark_info_near.p_left_bottom_x ,  back_mark_info_near.p_left_bottom_y );
  
    back_p_right_top_   =  cv::Point(back_mark_info_near.p_right_top_x    , back_mark_info_near.p_right_top_y   );
    back_p_right_bottom_=  cv::Point(back_mark_info_near.p_right_bottom_x , back_mark_info_near.p_right_bottom_y);
    
    back_offset_bottom_ =  back_mark_info_near.offset_bottom;
    back_offset_top_    =  back_mark_info_near.offset_top;
    back_bottom_diff_   =  back_mark_info_near.bottom_diff;
    back_top_diff_      =  back_mark_info_near.top_diff;

}


void infer_rail::setMarkInfoFarBack(MarkInfo & back_mark_info_far){
    
    back_p_left_top_far_    =  cv::Point(back_mark_info_far.p_left_top_x    ,  back_mark_info_far.p_left_top_y    );
    back_p_left_bottom_far_ =  cv::Point(back_mark_info_far.p_left_bottom_x ,  back_mark_info_far.p_left_bottom_y );
  
    back_p_right_top_far_   =  cv::Point(back_mark_info_far.p_right_top_x    , back_mark_info_far.p_right_top_y   );
    back_p_right_bottom_far_=  cv::Point(back_mark_info_far.p_right_bottom_x , back_mark_info_far.p_right_bottom_y);
    
    back_offset_bottom_far_ =  back_mark_info_far.offset_bottom;
    back_offset_top_far_    =  back_mark_info_far.offset_top;
    back_bottom_diff_far_   =  back_mark_info_far.bottom_diff;
    back_top_diff_far_      =  back_mark_info_far.top_diff;

}




bool infer_rail::setYolo(){

  YOLO = yolo::load(yolo_engine_path_, yolo::Type::V8Seg, confidence_);

  if (YOLO == nullptr)
  {
    std::cout << "初始化模型失败" << std::endl;
    return false;
  }
  return true;


}



float infer_rail::point_distance(float left, float top, float right, float bottom)
{
  if (top < y_limit_ && bottom > y_limit_)
  {
    
    float dis = fabs(((left + right) / 2) - width_ /2.0 );
    
    return dis;
  }
  return -1;
}



bool infer_rail::point_distanceV2(yolo::Box & test_obj)
{

  cv::Mat test_far_rail_msk = findOriginalMask(test_obj);
  int rows = test_far_rail_msk.rows;
  int cols = test_far_rail_msk.cols;

  int bottom_left = -1;
  int bottom_right = -1;

  bool find_left = false;
  bool find_right = false;
  for(int row = rows - 2; row > 2; row--){
    for(int col = 2 ; col < cols-2; col++){        
      if(test_far_rail_msk.at<uint8_t>(row,col) == 255){
          bottom_left = col;
          find_left = true;
          break;
      }
    }
    if(find_left) break;

  }

  for(int row = rows - 2; row > 2; row--){
    for(int col = cols-2 ; col > 2; col--){        
      if(test_far_rail_msk.at<uint8_t>(row,col) == 255){
          bottom_right = col;
          find_right = true;
          break;
      }
    }
    if(find_right) break;
  }

  float edge_bottom_left = -1;
  float edge_bottom_right = -1;
  if(front_direction_){
    edge_bottom_left = p_left_bottom_far_.x - offset_bottom_far_ - bottom_diff_far_;
    edge_bottom_right = p_right_bottom_far_.x + offset_bottom_far_ + bottom_diff_far_;
  }else{
    edge_bottom_left = back_p_left_bottom_far_.x - back_offset_bottom_far_ - back_bottom_diff_far_;
    edge_bottom_right = back_p_right_bottom_far_.x + back_offset_bottom_far_ + back_bottom_diff_far_;

  }

  // std::cout<<"(edge_bottom_left < bottom_left) && (bottom_right < edge_bottom_right)"<<std::endl;
  // std::cout<<edge_bottom_left<<" "<< bottom_left<<" "<<bottom_right<<" "<<edge_bottom_right<<std::endl;

  if( (edge_bottom_left < bottom_left) && (bottom_right < edge_bottom_right)){

      return true;
  }


  return false;
}




float infer_rail::point_distanceV3(yolo::Box & test_obj)
{
  float top = test_obj.top;
  float bottom = test_obj.bottom;
  float left = test_obj.left;
  float right = test_obj.right;

  int row_bottom_left = -1;
  int row_bottom_right = -1;

 
  
  cv::Mat test_far_rail_msk = findOriginalMask(test_obj);
  int rows = test_far_rail_msk.rows;
  int cols = test_far_rail_msk.cols;

  int bottom_left = -1;
  int bottom_right = -1;

  bool find_left = false;
  bool find_right = false;
  for(int row = rows - 2; row > 2; row--){
    for(int col = 2 ; col < cols-2; col++){        
      if(test_far_rail_msk.at<uint8_t>(row,col) == 255){
          bottom_left = col;
          find_left = true;
          row_bottom_left = row;

          break;
      }
    }
    if(find_left) break;

  }
  for(int row = rows - 2; row > 2; row--){
    for(int col = cols-2 ; col > 2; col--){        
      if(test_far_rail_msk.at<uint8_t>(row,col) == 255){
          bottom_right = col;
          find_right = true;
          row_bottom_right = row;

          break;
      }
    }
    if(find_right) break;
  }

float dis = -1;
//轨道最底部左右掩码存在， 并且在人为指定的限界内
if(front_direction_ == true && find_left && find_right &&  near_left_limit_ < bottom_left  &&  bottom_right < near_right_limit_ && row_bottom_left > y_limit_ && row_bottom_right > y_limit_){
    float mid = (bottom_left + bottom_right ) /2.0;
    dis = fabs(mid - width_ /2.0 );
}

if(front_direction_ == false  && find_left && find_right && back_near_left_limit_ < bottom_left  &&  bottom_right < back_near_right_limit_ && row_bottom_left > y_limit_ && row_bottom_right > y_limit_ ){
    float mid = (bottom_left + bottom_right ) /2.0;
    dis = fabs(mid- width_ /2.0 );
}

return dis;

}




float infer_rail::point_distanceV4(yolo::Box & test_obj)
{
  float top = test_obj.top;
  float bottom = test_obj.bottom;
  float left = test_obj.left;
  float right = test_obj.right;

  
  int row_bottom_left = -1;
  int row_bottom_right = -1;

    
  cv::Mat test_far_rail_msk = findOriginalMask(test_obj);
  int rows = test_far_rail_msk.rows;
  int cols = test_far_rail_msk.cols;

  int bottom_left = -1;
  int bottom_right = -1;

  bool find_left = false;
  bool find_right = false;
  for(int row = rows - 2; row > 2; row--){
    for(int col = 2 ; col < cols-2; col++){        
      if(test_far_rail_msk.at<uint8_t>(row,col) == 255){
          bottom_left = col;
          find_left = true;
          row_bottom_left = row;
          break;
      }
    }
    if(find_left) break;

  }
  for(int row = rows - 2; row > 2; row--){
    for(int col = cols-2 ; col > 2; col--){        
      if(test_far_rail_msk.at<uint8_t>(row,col) == 255){
          bottom_right = col;
          find_right = true;
          row_bottom_right = row;
          break;
      }
    }
    if(find_right) break;
  }

  // float dis = -1;
  // //轨道最底部左右掩码存在， 并且在人为指定的限界内
  // if(find_left && find_right){
  //     float mid = (bottom_left + bottom_right ) /2.0;
  //     dis = fabs(mid - width_ /2.0 );

  // }


  float dis = -1;
  float mid = (bottom_left + bottom_right ) /2.0;
  if(front_direction_ == true && find_left && find_right && row_bottom_left > y_limit_ && row_bottom_right > y_limit_ ){

        float mark_mid = (p_right_bottom_far_.x - p_left_bottom_far_.x)/2.0;
        dis = fabs(mid - mark_mid);

  }else if(front_direction_ == false && find_left && find_right && row_bottom_left > y_limit_ && row_bottom_right > y_limit_){

        float mark_mid = (back_p_right_bottom_far_.x - back_p_left_bottom_far_.x)/2.0;
        dis = fabs(mid - mark_mid);
  }


  return dis;
  

}







cv::Mat findLargestContour(const cv::Mat &mask)
{
  cv::Mat result(mask.size(), CV_8UC1, cv::Scalar(0));

  std::vector<std::vector<cv::Point>> contours;


  cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  double maxArea = 0;
  int maxAreaIdx = -1;
  for (int i = 0; i < contours.size(); ++i)
  {
    double area = cv::contourArea(contours[i]);
    if (area > maxArea)
    {
      maxArea = area;
      maxAreaIdx = i;
    }
  }

  if (maxAreaIdx >= 0)
  {
    cv::drawContours(result, contours, maxAreaIdx, cv::Scalar(255), cv::FILLED);
  }


  return result;
}

std::vector<point_set> infer_rail::infer_out(cv::Mat & image1, cv::Mat & image2){

  if(front_direction_){

      H_ = H_front_.clone();
      cal_line1_ = line1_;
      cal_line2_ = line2_;
      cal_line3_ = line3_;
      cal_line4_ = line4_;
      cal_line1_far_ = line1_far_;
      cal_line2_far_ = line2_far_;
      cal_line3_far_ = line3_far_;
      cal_line4_far_ = line4_far_;


  }else{

      H_ = H_back_.clone();
      cal_line1_ = back_line1_;
      cal_line2_ = back_line2_;
      cal_line3_ = back_line3_;
      cal_line4_ = back_line4_;
      cal_line1_far_ = back_line1_far_;
      cal_line2_far_ = back_line2_far_;
      cal_line3_far_ = back_line3_far_;
      cal_line4_far_ = back_line4_far_;


  }
  
  std::vector<point_set> box_set;

  if(image1.empty() || image2.empty()){

    image_empty_ = true;
    return box_set;
  }
  image_empty_ = false;

  cv::Mat current_near_rail_mask;
  bool flag = false;
  bool far_has_rail = false;
  int lidar_near = height_ - 1;
  int lidar_far = height_ - 1;
  
  std::vector<cv::Mat> images_v8{image1, image2};
  std::vector<yolo::Image> yoloimages(images_v8.size());
  std::transform(images_v8.begin(), images_v8.end(), yoloimages.begin(), cvimg);
  auto batched_result = YOLO->forwards(yoloimages);

  for (int ib = 0; ib < batched_result.size(); ++ib)
  {
    auto objs = batched_result[ib];
    float min_dis = width_;
    int rail_correct = -1; // 正确的轨道序号

    float far_min_dis = width_;
    int far_rail_correct = -1;

    std::vector<point_set> target_set; // 非轨道

    for (int i = 0; i < objs.size(); ++i){

      yolo::Box obj = objs[i];

      if (obj.class_label == 0) {
        if (ib == 0){

          // float dis = point_distance(obj.left, obj.top, obj.right, obj.bottom);
          float dis = point_distanceV3(obj);
          if (dis > 0 && dis < min_dis){
            min_dis = dis;
            rail_correct = i;
          }
        }
  
      }else if (obj.class_label == 1 || obj.class_label == 2 || obj.class_label == 3) // 人 汽车 电动车
      {
        if (ib == 0) // 近处摄像头
        {
   
          target_set.push_back({int(obj.left), int(obj.right), int(obj.top), int(obj.bottom), obj.class_label, 0, 0, obj.confidence}); // 记录位置，用于第二次循环判断人车是否在轨道上
                                                                                                          
        }
        else
        {
   
          target_set.push_back({int(obj.left), int(obj.right), int(obj.top), int(obj.bottom), obj.class_label, 0, 1, obj.confidence}); // 记录位置，用于第二次循环判断人车是否在轨道上
        }
      }
    }


    if(ib == 0 && rail_correct != -1){

        current_near_rail_mask = preprocessNearRail(objs[rail_correct]);      
        flag = true;

        lidar_near = findEdge(current_near_rail_mask, 0);
        
        near_has_rail_ = true;

        
        //外部可视化
        img1_rail_mask_ = current_near_rail_mask;



    }
    
    if(ib == 0 && rail_correct == -1){

        near_has_rail_ = false;

        flag = false;

        for(auto & target : target_set){
          target.type = 1;
          box_set.push_back(target);
        }

        
        continue;

    }


    if(ib == 1 && !objs.empty() && flag ){

      for (int test_idx = 0; test_idx < objs.size(); ++test_idx){
          yolo::Box test_obj = objs[test_idx];
          if (test_obj.class_label == 0) {
            // float far_dis = point_distance(test_obj.left, test_obj.top, test_obj.right, test_obj.bottom);
            float far_dis = point_distanceV4(test_obj);

            if (far_dis > 0 && far_dis < far_min_dis){
                far_min_dis = far_dis;
                far_rail_correct = test_idx;
            }
          }
      }

      if(far_rail_correct != -1){

        bool res_bool = point_distanceV2(objs[far_rail_correct]);
        rail_correct = res_bool? far_rail_correct : -1;

      }

      // std::cout<<"res_bool = "<<res_bool<<"  far_rail_correct = "<<far_rail_correct<<" rail_correct = "<<rail_correct<<std::endl;
      // rail_correct = preprocessFarRail(objs, current_near_rail_mask);
      // rail_correct = far_rail_correct;


      if(rail_correct != -1){

        far_has_rail = true;
        cv::Mat current_far_rail_mask = findOriginalMask(objs[rail_correct]);

        lidar_far = findEdge(current_far_rail_mask,1);

         //外部可视化
        img_rail_mask_far_ = current_far_rail_mask;


      }

    }


    if(ib == 1 && rail_correct == -1){


      for(auto & target : target_set){
        target.type = 1;
        box_set.push_back(target);
      }
      continue;

    }


      for (int tk = 0; tk < target_set.size(); tk++)
      {
        point_set target = target_set[tk];
        cv::Mat box_mask = findOriginalMask(objs[rail_correct]);
  
        int type = nearRailNew(target, box_mask);

        //if type==2
        //target.label==1

        if(use_tinyvit_ == true && target.label==1 && type==2) //判定为人 且在轨道上
        {

          int rect_width = target.xmax - target.xmin;
          int rect_height = target.ymax - target.ymin;
          if(rect_width >0 && rect_height > 0){
            
              cv::Rect rect(target.xmin, target.ymin, rect_width, rect_height);

              int cls_ans = 1;
              if(target.proximity == 0)//近 
              {    
                cv::Mat person_roi(image1, rect);
                cls_ans = personClassification(person_roi);

              }else if(target.proximity == 1)//远
              {
                cv::Mat person_roi(image2, rect);
                cls_ans = personClassification(person_roi);
              }

              if(cls_ans == 0)//分类结果非人，type转成3
              {
                  type = 3;
              }
          }
        }
        target.type = type;
        if (target.type != 0){
          box_set.push_back(target);
        }
      }

  }


  lidar_distance_near_ = lidar_near;
  lidar_distance_far_ = lidar_far;
  far_is_valid_ = far_has_rail;



  //换向 清0
  if(before_front_flag_ != front_direction_){
      que_.clear();
  }
  before_front_flag_ = front_direction_;

  que_.push_back(lidar_distance_near_);

  if(que_.size()>5){
    que_.pop_front();
  }

  if(que_.size()==5){

    auto max_iter = std::max_element(que_.begin(), que_.end());
    lidar_distance_near_ = *max_iter;
    
  }


  // if(que_.size()>3){
  //   que_.pop_front();
  // }

  // if(que_.size()==3){
  //   int diff1 = std::abs(que_[1] - que_[0] );
  //   int diff2 = std::abs(que_[1] - que_[2] );
  //   if(que_[0]> que_[1] && que_[1] < que_[2] && diff1 > 100 && diff2 > 100){

  //       int max_dist = std::max(que_[0], que_[2]);
  //       lidar_distance_near_ = max_dist;
  //   }else{

  //       // pass 
  //       lidar_distance_near_ = que_[1];
  //   }

  // }





  if(front_direction_){
      if(lidar_distance_near_ > near_turn_distance_){
          lidar_distance_far_ = height_ - 1;
          far_is_valid_ = false;
        }

  }else{

     if(lidar_distance_near_ > back_near_turn_distance_){

          lidar_distance_far_ = height_ - 1;
          far_is_valid_ = false;
        }

  }



  
//  std::vector<point_set> res = featureTransform(box_set);

  return box_set;
}




//  //img_object:far, img_scene:near
// cv::Mat infer_rail::featureMatching(cv::Mat & img_scene, cv::Mat & img_object){

//     int minHessian = 500;
//     cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
//     std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
//     cv::Mat descriptors_object, descriptors_scene;
//     detector->detectAndCompute( img_object, cv::noArray(), keypoints_object, descriptors_object );
//     detector->detectAndCompute( img_scene, cv::noArray(), keypoints_scene, descriptors_scene );
//     //-- Step 2: Matching descriptor vectors with a FLANN based matcher
//     // Since SURF is a floating-point descriptor NORM_L2 is used
//     cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
//     std::vector< std::vector<cv::DMatch> > knn_matches;
//     matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
//     //-- Filter matches using the Lowe's ratio test
//     const float ratio_thresh = 0.75f;
//     std::vector<cv::DMatch> good_matches;
//     for (size_t i = 0; i < knn_matches.size(); i++)
//     {
//         if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
//         {
//             good_matches.push_back(knn_matches[i][0]);
//         }
//     }


//     std::vector<cv::Point2f> obj;
//     std::vector<cv::Point2f> scene;
//     for( size_t i = 0; i < good_matches.size(); i++ )
//     {
//         //-- Get the keypoints from the good matches
//         obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
//         scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
//     }
//     cv::Mat H = findHomography( obj, scene, cv::RANSAC );
//     return H;

// }




// std::vector<point_set> infer_rail::featureTransform(std::vector<point_set> & box_set){
  
//   std::vector<point_set> perspective_box;
//   std::vector<point_set> cal_set;
//   std::vector<point_set> res;

//   if(box_set.empty()){
//     return res;
//   }

//   for(auto & item : box_set){
//         if(item.proximity == 0) {
//             cal_set.push_back(item);
//             continue;
//         }

//         int item_width = item.xmax - item.xmin;
//         int item_height = item.ymax - item.ymin;
                    
//         std::vector<cv::Point2f> obj_corners(4);
//         obj_corners[0] = cv::Point2f(item.xmin, item.ymin);
//         obj_corners[1] = cv::Point2f(item.xmin + item_width, item.ymin );
//         obj_corners[2] = cv::Point2f(item.xmin + item_width, item.ymin + item_height);
//         obj_corners[3] = cv::Point2f(item.xmin, item.ymin + item_height);
//         std::vector<cv::Point2f> scene_corners(4);
        
//         perspectiveTransform( obj_corners, scene_corners, H_);

//         cv::Rect result_box  = cv::boundingRect(scene_corners);

//         cv::Point tl = result_box.tl(); // 左上角点
//         cv::Point br = result_box.br(); // 右下角点

//         point_set perspective_new;
//         perspective_new.xmin = tl.x;
//         perspective_new.ymin = tl.y;
//         perspective_new.xmax = br.x;
//         perspective_new.ymax = br.y;
//         perspective_new.label = item.label;
//         perspective_new.type = item.type;
//         perspective_new.proximity = 0;
//         perspective_new.confidence = item.confidence;

//         perspective_box.push_back(perspective_new);

//   }


//   for(auto & per_item : perspective_box){
//           cal_set.push_back(per_item);
//   }


//       std::vector<cv::Rect> bboxes;
//       std::vector<float>scores;
//       std::vector<int>indices;
//       std::vector<int>label_idxs;
//       std::vector<int> nms_types;
//       std::vector<int> nms_proximities;

//       for(auto & item : cal_set){
 
//           cv::Point pt1(item.xmin, item.ymin);
//           cv::Point pt2(item.xmax, item.ymax);
//           cv::Rect rect(pt1,pt2);
//           bboxes.push_back(rect);
//           scores.push_back(item.confidence);
//           label_idxs.push_back(item.label);
//           nms_types.push_back(item.type);
//           nms_proximities.push_back(item.proximity);

//       }
      
//       cv::dnn::NMSBoxes(bboxes,scores,0.5,0.5,indices);

//       for(auto & idx : indices){
//           int num = idx;
//           cv::Rect rec = bboxes[num];
//           float score = scores[num];
//           int label = label_idxs[num];
//           int type = nms_types[num];      
//           int proximity = nms_proximities[num];
//           cv::Point tl = rec.tl(); // 左上角点
//           cv::Point br = rec.br(); // 右下角点
          
//           point_set temp;
//           temp.xmin = tl.x;
//           temp.ymin = tl.y;
//           temp.xmax = br.x;
//           temp.ymax = br.y;
//           temp.label = label;
//           temp.type = type;
//           temp.proximity = proximity;
//           temp.confidence = score;
//           res.push_back(temp);
//       }


//       for(auto & item : box_set ){

//           if(item.proximity == 1){

//             res.push_back(item);
//           }

//       }

//     return res;


// }


cv::Mat infer_rail::preprocessNearRail(yolo::Box & obj){
    
    cv::Mat total_mask = findOriginalMask(obj);

    return total_mask;

}


TestMask infer_rail::findTotalMaskContours(cv::Mat & total_mask, cv::Mat & current_near_rail_mask ){


      for(int row = 0 ; row < total_mask.rows; row ++){
        for(int col = 0 ; col < total_mask.cols; col++){
            if(row == 0 || row == total_mask.rows - 1){
              total_mask.at<uint8_t>(row,col) = 0;

            }
            if(col ==0 || col == total_mask.cols-1){
              total_mask.at<uint8_t>(row,col) = 0;
            }
        }

      }

      std::vector<std::vector<cv::Point>> contours;

      cv::findContours(total_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      int maxArea = 0;
      int maxIndex = -1;
      for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIndex = i;
            }
      }

      std::vector<cv::Point2f> obj_corners{};
      std::vector<cv::Point2f> scene_corners{};

      for(auto  p : contours[maxIndex]){

          obj_corners.push_back(cv::Point2f( p.x, p.y ));

      }
            
      perspectiveTransform( obj_corners, scene_corners, H_);

      std::vector<cv::Point> scene_corners_int{};

      for(auto p : scene_corners){

          scene_corners_int.push_back( cv::Point( int(p.x), int(p.y) ) );

      }

      auto newEnd = std::unique(scene_corners_int.begin(), scene_corners_int.end(),
                   [&]( const cv::Point& p1, const cv::Point& p2) {return p1.x == p2.x && p1.y == p2.y;});
      
      scene_corners_int.erase(newEnd, scene_corners_int.end());

      std::vector<std::vector<cv::Point>> result_contours{scene_corners_int};

      cv::Mat result_mask =cv::Mat::zeros(total_mask.rows, total_mask.cols, CV_8UC1);

      cv::drawContours(result_mask, result_contours, 0, cv::Scalar(255), cv::FILLED);



      cv::Mat ans = cv::Mat::zeros(total_mask.rows, total_mask.cols, CV_8UC1);
      cv::bitwise_and(current_near_rail_mask, result_mask, ans);

      int nonzeroCount = cv::countNonZero(ans) ;

      TestMask tm;
      tm.ans = nonzeroCount;
      
      //外部可视化
      tm.mask = result_mask;
      
      return tm;

}



int infer_rail::preprocessFarRail(yolo::BoxArray & far_rail_objs, cv::Mat & current_near_rail_mask){

      int correct = -1;
      int maxcnt = 0;
      
      for(int i = 0 ; i < far_rail_objs.size(); i++){

          auto & obj = far_rail_objs[i];
          if(obj.class_label != 0 ){
            continue;
          }


          cv::Mat total_mask = findOriginalMask(obj);

          auto test_mask = findTotalMaskContours(total_mask, current_near_rail_mask);
    
          if (test_mask.ans > maxcnt)
          {
            maxcnt = test_mask.ans;
            correct = i;
            
            //外部可视化
            img2_rail_mask_ = test_mask.mask;

          }
    
      }
      return correct;

}



int infer_rail::findEdge(cv::Mat & rail_mask, const int & far_flag){
  
  std::vector<cv::Point> left_points{};
  std::vector<cv::Point> right_points{};


  for(int row = rail_mask.rows -10 ; row > 0; row --){
    for(int col = 0; col < rail_mask.cols; col++){
      if(rail_mask.at<uint8_t>(row,col) == 255){
        left_points.push_back(cv::Point(col,row));
        break;
      }
    }
  }


  for(int row = rail_mask.rows -10 ; row > 0; row --){
    for(int col = rail_mask.cols; col > 0; col--){
      if(rail_mask.at<uint8_t>(row,col) == 255){
        right_points.push_back(cv::Point(col,row));
        break;
      }
    }
  }

    std::vector<std::vector<cv::Point>> left_split = splitRailPoints(left_points,1);
    std::vector<std::vector<cv::Point>> right_split = splitRailPoints(right_points,1);

    std::vector<bool> left_result = findAbnormalCluster(left_split,0, far_flag);
    std::vector<bool> right_result = findAbnormalCluster(right_split,1,far_flag);

    int max_left_pos = findMaxPosition(left_result);
    int max_right_pos = findMaxPosition(right_result);
    

    std::vector<cv::Point> left_result_cluster = left_split[max_left_pos];
    std::vector<cv::Point> right_result_cluster = right_split[max_right_pos];

    cv::Point left_result_point = left_result_cluster[0];
    cv::Point right_result_point = right_result_cluster[0];

    int lidar_distance = height_ - 1;

    lidar_distance = std::max(left_result_point.y, right_result_point.y);

    // 远处lidar_distance投影到近处, 外部可视化
    // if(far_flag == 0){

    //     lidar_distance = std::max(left_result_point.y, right_result_point.y);
        
    //     p1_ = left_result_point;
    //     p2_ = right_result_point;

    // }else {

    //     std::vector<cv::Point2f> obj_corners{left_result_point, right_result_point};
    //     std::vector<cv::Point2f> scene_corners;
        
    //     perspectiveTransform( obj_corners, scene_corners, H_);

    //     lidar_distance = std::max(scene_corners[0].y, scene_corners[1].y);

    //     p1_origin_far_ = left_result_point;
    //     p2_origin_far_ = right_result_point;
    //     p1_far_ = scene_corners[0];
    //     p2_far_ = scene_corners[1];

    // }
    
    return lidar_distance; 

}

 std::vector<std::vector<cv::Point>> infer_rail::splitRailPoints(std::vector<cv::Point> & pts, const int & len){

    std::vector<std::vector<cv::Point>> result_points{};

    for(int i = 0 ; i < pts.size() -len  + 1; i += len){
      
      std::vector<cv::Point> temp;
      for(int j = 0 ; j < len ; j++){
        temp.push_back(pts[i+j]);

      }
      result_points.push_back(temp);


    }
    return result_points;

}

std::vector<bool> infer_rail::findAbnormalCluster( std::vector<std::vector<cv::Point>> & splits, const int & flag, const int & far_flag){
      
      std::vector<bool> result;
      
      float k = 0;
      int x0 = 0;
      int y0 = 0;

      float k2 = 0;
      int x02 = 0;
      int y02 = 0;

      if(far_flag == 0){

          if(flag == 0){

              k = cal_line1_[1]/cal_line1_[0];
              x0 = cal_line1_[2];
              y0 = cal_line1_[3];

              k2 = cal_line2_[1]/cal_line2_[0];
              x02 = cal_line2_[2];
              y02 = cal_line2_[3];
          }else if(flag == 1){
              k = cal_line3_[1]/cal_line3_[0];
              x0 = cal_line3_[2];
              y0 = cal_line3_[3];

              k2 = cal_line4_[1]/cal_line4_[0];
              x02 = cal_line4_[2];
              y02 = cal_line4_[3];

          }

      }else if(far_flag==1){

          if(flag == 0){
              k = cal_line1_far_[1]/cal_line1_far_[0];
              x0 = cal_line1_far_[2];
              y0 = cal_line1_far_[3];

              k2 = cal_line2_far_[1]/cal_line2_far_[0];
              x02 = cal_line2_far_[2];
              y02 = cal_line2_far_[3];
          }else if(flag == 1){
              k = cal_line3_far_[1]/cal_line3_far_[0];
              x0 = cal_line3_far_[2];
              y0 = cal_line3_far_[3];

              k2 = cal_line4_far_[1]/cal_line4_far_[0];
              x02 = cal_line4_far_[2];
              y02 = cal_line4_far_[3];

          }

      }


      for(auto item : splits ){

        int cnt = 0;
        for(auto i : item){

          float x = ( i.y - y0)/k + x0;
          float x2 = (i.y - y02)/k2 + x02;

          if( x<=i.x && i.x<=x2){
              cnt++;

          }  
        }

        if(cnt>0){
          result.push_back(true);
        }else{
          result.push_back(false);
        }

      }
      return result;


}


int infer_rail::findMaxPosition(std::vector<bool> & temp){

    int longestStart = -1;
    int longestLength = 0;
    int currentStart = -1;
    int currentLength = 0;

    for (int i = 0; i <= temp.size(); ++i) {  // 注意这里是小于等于temp.size()
        if (i < temp.size() && temp[i] == 0) {  // 检查是否越界
            if (currentStart == -1) {
                currentStart = i;
            }
            currentLength++;
        } else {
            if (currentLength > longestLength) {
                longestStart = currentStart;
                longestLength = currentLength;
            }
            currentStart = -1;
            currentLength = 0;
        }
    }

    if (longestStart == -1) {  // 如果数组全为1
        longestStart = temp.size() - 1;
    }

    return longestStart;

}


void infer_rail::fitLines(const int & flag){

      std::vector<cv::Point>fit_points1;
      std::vector<cv::Point>fit_points2;
      std::vector<cv::Point>fit_points3;
      std::vector<cv::Point>fit_points4;

      if(flag == 0){
        
        fit_points1.push_back(cv::Point(p_left_top_.x     -  offset_top_,     p_left_top_.y)) ;
        fit_points1.push_back(cv::Point(p_left_bottom_.x  -  offset_bottom_,  p_left_bottom_.y));

        fit_points2.push_back(cv::Point(p_left_top_.x    +  offset_top_,       p_left_top_.y));
        fit_points2.push_back(cv::Point(p_left_bottom_.x +  offset_bottom_,  p_left_bottom_.y));

        fit_points3.push_back(cv::Point(p_right_top_.x     -  offset_top_,    p_right_top_.y)) ;
        fit_points3.push_back(cv::Point(p_right_bottom_.x  -  offset_bottom_, p_right_bottom_.y));
        
        fit_points4.push_back(cv::Point(p_right_top_.x    +  offset_top_,    p_right_top_.y));
        fit_points4.push_back(cv::Point(p_right_bottom_.x +  offset_bottom_, p_right_bottom_.y));


        cv::fitLine(fit_points1, line1_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points2, line2_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points3, line3_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points4, line4_, cv::DIST_L2, 0, 1e-2, 1e-2);

      }else if(flag == 1){

        fit_points1.push_back(cv::Point(p_left_top_far_.x     -  offset_top_far_,     p_left_top_far_.y)) ;
        fit_points1.push_back(cv::Point(p_left_bottom_far_.x  -  offset_bottom_far_,  p_left_bottom_far_.y));

        fit_points2.push_back(cv::Point(p_left_top_far_.x    +  offset_top_far_,     p_left_top_far_.y));
        fit_points2.push_back(cv::Point(p_left_bottom_far_.x +  offset_bottom_far_,  p_left_bottom_far_.y));

        fit_points3.push_back(cv::Point(p_right_top_far_.x     -  offset_top_far_,    p_right_top_far_.y)) ;
        fit_points3.push_back(cv::Point(p_right_bottom_far_.x  -  offset_bottom_far_, p_right_bottom_far_.y));
        
        fit_points4.push_back(cv::Point(p_right_top_far_.x    +  offset_top_far_,    p_right_top_far_.y));
        fit_points4.push_back(cv::Point(p_right_bottom_far_.x +  offset_bottom_far_, p_right_bottom_far_.y));

        cv::fitLine(fit_points1, line1_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points2, line2_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points3, line3_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points4, line4_far_, cv::DIST_L2, 0, 1e-2, 1e-2);

      }else if(flag == 2){
        
        fit_points1.push_back(cv::Point(back_p_left_top_.x     -  back_offset_top_,     back_p_left_top_.y)) ;
        fit_points1.push_back(cv::Point(back_p_left_bottom_.x  -  back_offset_bottom_,  back_p_left_bottom_.y));

        fit_points2.push_back(cv::Point(back_p_left_top_.x    +  back_offset_top_,       back_p_left_top_.y));
        fit_points2.push_back(cv::Point(back_p_left_bottom_.x +  back_offset_bottom_,    back_p_left_bottom_.y));

        fit_points3.push_back(cv::Point(back_p_right_top_.x     -  back_offset_top_,    back_p_right_top_.y)) ;
        fit_points3.push_back(cv::Point(back_p_right_bottom_.x  -  back_offset_bottom_, back_p_right_bottom_.y));
        
        fit_points4.push_back(cv::Point(back_p_right_top_.x    +  back_offset_top_,    back_p_right_top_.y));
        fit_points4.push_back(cv::Point(back_p_right_bottom_.x +  back_offset_bottom_, back_p_right_bottom_.y));


        cv::fitLine(fit_points1, back_line1_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points2, back_line2_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points3, back_line3_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points4, back_line4_, cv::DIST_L2, 0, 1e-2, 1e-2);

      }else if(flag==3){

        fit_points1.push_back(cv::Point(back_p_left_top_far_.x     -  back_offset_top_far_,     back_p_left_top_far_.y)) ;
        fit_points1.push_back(cv::Point(back_p_left_bottom_far_.x  -  back_offset_bottom_far_,  back_p_left_bottom_far_.y));

        fit_points2.push_back(cv::Point(back_p_left_top_far_.x    +  back_offset_top_far_,     back_p_left_top_far_.y));
        fit_points2.push_back(cv::Point(back_p_left_bottom_far_.x +  back_offset_bottom_far_,  back_p_left_bottom_far_.y));

        fit_points3.push_back(cv::Point(back_p_right_top_far_.x     -  back_offset_top_far_,    back_p_right_top_far_.y)) ;
        fit_points3.push_back(cv::Point(back_p_right_bottom_far_.x  -  back_offset_bottom_far_, back_p_right_bottom_far_.y));
        
        fit_points4.push_back(cv::Point(back_p_right_top_far_.x    +  back_offset_top_far_,    back_p_right_top_far_.y));
        fit_points4.push_back(cv::Point(back_p_right_bottom_far_.x +  back_offset_bottom_far_, back_p_right_bottom_far_.y));

        cv::fitLine(fit_points1, back_line1_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points2, back_line2_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points3, back_line3_far_, cv::DIST_L2, 0, 1e-2, 1e-2);
        cv::fitLine(fit_points4, back_line4_far_, cv::DIST_L2, 0, 1e-2, 1e-2);

      }


}


cv::Mat infer_rail::findOriginalMask(yolo::Box & obj ){

  float scale_x = 640 / (float)width_;
  float scale_y = 640 / (float)height_;
  float scale = std::min(scale_x, scale_y);
  float ox = -scale * width_ * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
  float oy = -scale * height_ * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
  cv::Mat M = (cv::Mat_<float>(2, 3) << scale, 0, ox, 0, scale, oy);

  cv::Mat IM;
  cv::invertAffineTransform(M, IM);

  cv::Mat mask_map = cv::Mat::zeros(cv::Size(160, 160), CV_8UC1);
  cv::Mat small_mask(obj.seg->height, obj.seg->width, CV_8UC1, obj.seg->data);
  
  cv::Rect roi(obj.seg->left, obj.seg->top, obj.seg->width, obj.seg->height);
  

    if (roi.x < 0) {
        roi.width += roi.x;  
        roi.x = 0;           
    }
    if (roi.y < 0) {
        roi.height += roi.y; 
        roi.y = 0;            
    }
    if (roi.x + roi.width > mask_map.cols) {
        roi.width = mask_map.cols - roi.x;  
    }
    if (roi.y + roi.height > mask_map.rows) {
        roi.height = mask_map.rows - roi.y; 
    }
  

  cv::resize(small_mask,small_mask,mask_map(roi).size());
  small_mask.copyTo(mask_map(roi));

  cv::resize(mask_map, mask_map, cv::Size(640, 640)); 
  cv::threshold(mask_map, mask_map, 128, 1, cv::THRESH_BINARY);

  cv::Mat mask_resized;
  cv::warpAffine(mask_map, mask_resized, IM, cv::Size(width_, height_), cv::INTER_LINEAR);

  auto res = findLargestContour(mask_resized);


  return res;

}

int infer_rail::nearRailNew(point_set & target, cv::Mat & rail_mask){
    
    if (target.xmin < 0){
        target.xmin = 0;
    }
    if (target.xmax > rail_mask.cols -1 ){
        target.xmax = rail_mask.cols -1;
    }
    if (target.ymin < 0){
        target.ymin = 0;
    }
    if (target.ymax > rail_mask.rows -1 ){
        target.ymax = rail_mask.rows -1;
    }

    // cv::Rect rect(cv::Point(target.xmin, target.ymin), cv::Point(target.xmax, target.ymax));


////////////////////////////
    int height = 0.15 * ( target.ymax - target.ymin);
    int width = 0.15 * (target.xmax - target.xmin);
    int xmin = target.xmin - width;
    int ymin = target.ymin - height;
    int xmax = target.xmax + width;
    int ymax = target.ymax + height;
 
 
    if (xmin < 0){
        xmin = 0;
    }
    if (xmax > rail_mask.cols -1 ){
        xmax = rail_mask.cols -1;
     }
     if (ymin < 0){
         ymin = 0;
     }
     if (ymax > rail_mask.rows -1 ){
         ymax = rail_mask.rows -1;
     }

    cv::Rect rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));

///////////////////////////


    cv::Mat roi = rail_mask(rect);

    int nonzeroCount = cv::countNonZero(roi);

    if(nonzeroCount > 0){
        return 2;
    }
    else{
        return 1;
    }

   return 0;


}

std::vector<Configuration> infer_rail::readConfig(std::string & configPath){
    
    std::vector<Configuration> configurations;

    std::ifstream configFile;
    configFile.open(configPath);
    if (!configFile.is_open()) {
        std::cerr << "Error opening config file." << std::endl;
        return configurations;
    }

   
    Configuration currentConfig;
    std::string line;
    while (std::getline(configFile, line)) {
        if (line.empty()) continue; // 跳过空行

        if (line[0] == '[') { // 新的大类开始
            if (!currentConfig.data.empty()) { // 如果之前的大类有配置数据，存储起来
                configurations.push_back(currentConfig);
                currentConfig.data.clear();
            }
            // 保存大类名

            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            currentConfig.section = line;

        } else if (line[0] != '%') { // 不是注释行
            // Split line into key and value
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, '=')) {
                if (std::getline(iss, value)) {
                    // Remove leading and trailing whitespace from key and value
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);

                  
                    key.erase(std::remove(key.begin(), key.end(), '\r'), key.end());
                    value.erase(std::remove(value.begin(), value.end(), '\r'), value.end());
                    currentConfig.data[key] = value;
                                
                }
            }
        }
    }
    // Store the last configuration
    if (!currentConfig.data.empty()) {
        configurations.push_back(currentConfig);
    }



    return configurations;

}

void infer_rail::setMatrix(std::vector<Configuration> & configurations ){

  double M00, M01, M02 = 0;
  double M10, M11, M12 = 0;
  double M20, M21, M22 = 0;

  for(auto config : configurations){
    M00 = std::stod(config.data["M00"]);
    M01 = std::stod(config.data["M01"]);
    M02 = std::stod(config.data["M02"]);
    M10 = std::stod(config.data["M10"]);
    M11 = std::stod(config.data["M11"]);
    M12 = std::stod(config.data["M12"]);
    M20 = std::stod(config.data["M20"]);
    M21 = std::stod(config.data["M21"]);
    M22 = std::stod(config.data["M22"]);
    if(config.section == "[Matrix_Front]"){

        H_front_.at<double>(0, 0) = M00;
        H_front_.at<double>(0, 1) = M01;
        H_front_.at<double>(0, 2) = M02;
        H_front_.at<double>(1, 0) = M10;
        H_front_.at<double>(1, 1) = M11;
        H_front_.at<double>(1, 2) = M12;
        H_front_.at<double>(2, 0) = M20;
        H_front_.at<double>(2, 1) = M21;
        H_front_.at<double>(2, 2) = M22;

    }else if(config.section == "[Matrix_Back]"){

        H_back_.at<double>(0, 0) = M00;
        H_back_.at<double>(0, 1) = M01;
        H_back_.at<double>(0, 2) = M02;
        H_back_.at<double>(1, 0) = M10;
        H_back_.at<double>(1, 1) = M11;
        H_back_.at<double>(1, 2) = M12;
        H_back_.at<double>(2, 0) = M20;
        H_back_.at<double>(2, 1) = M21;
        H_back_.at<double>(2, 2) = M22;

    }

  }
  std::cout<<"H_front_ :"<<std::endl;
  std::cout<<H_front_<<std::endl;
  std::cout<<"H_back_ :"<<std::endl;
  std::cout<<H_back_<<std::endl;
}


void infer_rail::setMatrix(ini::iniReader &config){
    double M00, M01, M02 = 0;
    double M10, M11, M12 = 0;
    double M20, M21, M22 = 0;

    M00 = config.ReadFloat("Matrix_Front", "M00", 0);
    M01 = config.ReadFloat("Matrix_Front", "M01", 0);
    M02 = config.ReadFloat("Matrix_Front", "M02", 0);
    M10 = config.ReadFloat("Matrix_Front", "M10", 0);
    M11 = config.ReadFloat("Matrix_Front", "M11", 0);
    M12 = config.ReadFloat("Matrix_Front", "M12", 0);
    M20 = config.ReadFloat("Matrix_Front", "M20", 0);
    M21 = config.ReadFloat("Matrix_Front", "M21", 0);
    M22 = config.ReadFloat("Matrix_Front", "M22", 0);

    std::vector<double> temp_front{M00,M01,M02,
                                  M10,M11,M12,
                                  M20,M21,M22};

    cv::Mat temp_mat_front(3, 3, CV_64FC1, temp_front.data());
    H_front_ = temp_mat_front.clone();

    M00 = config.ReadFloat("Matrix_Back", "M00", 0);
    M01 = config.ReadFloat("Matrix_Back", "M01", 0);
    M02 = config.ReadFloat("Matrix_Back", "M02", 0);
    M10 = config.ReadFloat("Matrix_Back", "M10", 0);
    M11 = config.ReadFloat("Matrix_Back", "M11", 0);
    M12 = config.ReadFloat("Matrix_Back", "M12", 0);
    M20 = config.ReadFloat("Matrix_Back", "M20", 0);
    M21 = config.ReadFloat("Matrix_Back", "M21", 0);
    M22 = config.ReadFloat("Matrix_Back", "M22", 0);

    std::vector<double> temp_back{M00,M01,M02,
                                  M10,M11,M12,
                                  M20,M21,M22};
                                  
    cv::Mat temp_mat_back(3, 3, CV_64FC1, temp_back.data());
    H_back_ = temp_mat_back.clone();

    std::cout<<"H_front_ :"<<std::endl;
    std::cout<<H_front_<<std::endl;
    std::cout<<"H_back_ :"<<std::endl;
    std::cout<<H_back_<<std::endl;
}
