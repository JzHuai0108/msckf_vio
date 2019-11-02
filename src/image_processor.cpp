/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */
#include <msckf_vio/image_processor.h>

#include <msckf_vio/CameraMeasurement.h>
#include <msckf_vio/TrackingInfo.h>
#include <msckf_vio/utils.h>

using namespace std;
using namespace cv;

namespace msckf_vio {
ImageProcessor::ImageProcessor(ros::NodeHandle& n) :
  nh(n),
  //img_transport(n),
  stereo_sub(10) {
  return;
}

ImageProcessor::~ImageProcessor() {
  destroyAllWindows();
  return;
}

bool ImageProcessor::loadParameters() {
  // Camera calibration parameters
  nh.param<string>("cam0/distortion_model",
      feature_tracker_.cam0_distortion_model, string("radtan"));
  nh.param<string>("cam1/distortion_model",
      feature_tracker_.cam1_distortion_model, string("radtan"));

  vector<int> cam0_resolution_temp(2);
  nh.getParam("cam0/resolution", cam0_resolution_temp);
  feature_tracker_.cam0_resolution[0] = cam0_resolution_temp[0];
  feature_tracker_.cam0_resolution[1] = cam0_resolution_temp[1];

  vector<int> cam1_resolution_temp(2);
  nh.getParam("cam1/resolution", cam1_resolution_temp);
  feature_tracker_.cam1_resolution[0] = cam1_resolution_temp[0];
  feature_tracker_.cam1_resolution[1] = cam1_resolution_temp[1];

  vector<double> cam0_intrinsics_temp(4);
  nh.getParam("cam0/intrinsics", cam0_intrinsics_temp);
  feature_tracker_.cam0_intrinsics[0] = cam0_intrinsics_temp[0];
  feature_tracker_.cam0_intrinsics[1] = cam0_intrinsics_temp[1];
  feature_tracker_.cam0_intrinsics[2] = cam0_intrinsics_temp[2];
  feature_tracker_.cam0_intrinsics[3] = cam0_intrinsics_temp[3];

  vector<double> cam1_intrinsics_temp(4);
  nh.getParam("cam1/intrinsics", cam1_intrinsics_temp);
  feature_tracker_.cam1_intrinsics[0] = cam1_intrinsics_temp[0];
  feature_tracker_.cam1_intrinsics[1] = cam1_intrinsics_temp[1];
  feature_tracker_.cam1_intrinsics[2] = cam1_intrinsics_temp[2];
  feature_tracker_.cam1_intrinsics[3] = cam1_intrinsics_temp[3];

  vector<double> cam0_distortion_coeffs_temp(4);
  nh.getParam("cam0/distortion_coeffs",
      cam0_distortion_coeffs_temp);
  feature_tracker_.cam0_distortion_coeffs[0] = cam0_distortion_coeffs_temp[0];
  feature_tracker_.cam0_distortion_coeffs[1] = cam0_distortion_coeffs_temp[1];
  feature_tracker_.cam0_distortion_coeffs[2] = cam0_distortion_coeffs_temp[2];
  feature_tracker_.cam0_distortion_coeffs[3] = cam0_distortion_coeffs_temp[3];

  vector<double> cam1_distortion_coeffs_temp(4);
  nh.getParam("cam1/distortion_coeffs",
      cam1_distortion_coeffs_temp);
  feature_tracker_.cam1_distortion_coeffs[0] = cam1_distortion_coeffs_temp[0];
  feature_tracker_.cam1_distortion_coeffs[1] = cam1_distortion_coeffs_temp[1];
  feature_tracker_.cam1_distortion_coeffs[2] = cam1_distortion_coeffs_temp[2];
  feature_tracker_.cam1_distortion_coeffs[3] = cam1_distortion_coeffs_temp[3];

  cv::Mat     T_imu_cam0 = utils::getTransformCV(nh, "cam0/T_cam_imu");
  cv::Matx33d R_imu_cam0(T_imu_cam0(cv::Rect(0,0,3,3)));
  cv::Vec3d   t_imu_cam0 = T_imu_cam0(cv::Rect(3,0,1,3));
  feature_tracker_.R_cam0_imu = R_imu_cam0.t();
  feature_tracker_.t_cam0_imu = -R_imu_cam0.t() * t_imu_cam0;

  cv::Mat T_cam0_cam1 = utils::getTransformCV(nh, "cam1/T_cn_cnm1");
  cv::Mat T_imu_cam1 = T_cam0_cam1 * T_imu_cam0;
  cv::Matx33d R_imu_cam1(T_imu_cam1(cv::Rect(0,0,3,3)));
  cv::Vec3d   t_imu_cam1 = T_imu_cam1(cv::Rect(3,0,1,3));
  feature_tracker_.R_cam1_imu = R_imu_cam1.t();
  feature_tracker_.t_cam1_imu = -R_imu_cam1.t() * t_imu_cam1;

  // Processor parameters
  nh.param<bool>("monocular", feature_tracker_.processor_config.monocular, false);
  nh.param<int>("grid_row", feature_tracker_.processor_config.grid_row, 4);
  nh.param<int>("grid_col", feature_tracker_.processor_config.grid_col, 4);
  nh.param<int>("grid_min_feature_num",
      feature_tracker_.processor_config.grid_min_feature_num, 2);
  nh.param<int>("grid_max_feature_num",
      feature_tracker_.processor_config.grid_max_feature_num, 4);
  nh.param<int>("pyramid_levels",
      feature_tracker_.processor_config.pyramid_levels, 3);
  nh.param<int>("patch_size",
      feature_tracker_.processor_config.patch_size, 31);
  nh.param<int>("fast_threshold",
      feature_tracker_.processor_config.fast_threshold, 20);
  nh.param<int>("max_iteration",
      feature_tracker_.processor_config.max_iteration, 30);
  nh.param<double>("track_precision",
      feature_tracker_.processor_config.track_precision, 0.01);
  nh.param<double>("ransac_threshold",
      feature_tracker_.processor_config.ransac_threshold, 3);
  nh.param<double>("stereo_threshold",
      feature_tracker_.processor_config.stereo_threshold, 3);

  ROS_INFO("===========================================");
  ROS_INFO("cam0_resolution: %d, %d",
      feature_tracker_.cam0_resolution[0], feature_tracker_.cam0_resolution[1]);
  ROS_INFO("cam0_intrinscs: %f, %f, %f, %f",
      feature_tracker_.cam0_intrinsics[0], feature_tracker_.cam0_intrinsics[1],
      feature_tracker_.cam0_intrinsics[2], feature_tracker_.cam0_intrinsics[3]);
  ROS_INFO("cam0_distortion_model: %s",
      feature_tracker_.cam0_distortion_model.c_str());
  ROS_INFO("cam0_distortion_coefficients: %f, %f, %f, %f",
      feature_tracker_.cam0_distortion_coeffs[0], feature_tracker_.cam0_distortion_coeffs[1],
      feature_tracker_.cam0_distortion_coeffs[2], feature_tracker_.cam0_distortion_coeffs[3]);

  ROS_INFO("cam1_resolution: %d, %d",
      feature_tracker_.cam1_resolution[0], feature_tracker_.cam1_resolution[1]);
  ROS_INFO("cam1_intrinscs: %f, %f, %f, %f",
      feature_tracker_.cam1_intrinsics[0], feature_tracker_.cam1_intrinsics[1],
      feature_tracker_.cam1_intrinsics[2], feature_tracker_.cam1_intrinsics[3]);
  ROS_INFO("cam1_distortion_model: %s",
      feature_tracker_.cam1_distortion_model.c_str());
  ROS_INFO("cam1_distortion_coefficients: %f, %f, %f, %f",
      feature_tracker_.cam1_distortion_coeffs[0], feature_tracker_.cam1_distortion_coeffs[1],
      feature_tracker_.cam1_distortion_coeffs[2], feature_tracker_.cam1_distortion_coeffs[3]);

  cout << R_imu_cam0 << endl;
  cout << t_imu_cam0.t() << endl;

  ROS_INFO("monocular: %d",
      feature_tracker_.processor_config.monocular);
  ROS_INFO("grid_row: %d",
      feature_tracker_.processor_config.grid_row);
  ROS_INFO("grid_col: %d",
      feature_tracker_.processor_config.grid_col);
  ROS_INFO("grid_min_feature_num: %d",
      feature_tracker_.processor_config.grid_min_feature_num);
  ROS_INFO("grid_max_feature_num: %d",
      feature_tracker_.processor_config.grid_max_feature_num);
  ROS_INFO("pyramid_levels: %d",
      feature_tracker_.processor_config.pyramid_levels);
  ROS_INFO("patch_size: %d",
      feature_tracker_.processor_config.patch_size);
  ROS_INFO("fast_threshold: %d",
      feature_tracker_.processor_config.fast_threshold);
  ROS_INFO("max_iteration: %d",
      feature_tracker_.processor_config.max_iteration);
  ROS_INFO("track_precision: %f",
      feature_tracker_.processor_config.track_precision);
  ROS_INFO("ransac_threshold: %f",
      feature_tracker_.processor_config.ransac_threshold);
  ROS_INFO("stereo_threshold: %f",
      feature_tracker_.processor_config.stereo_threshold);
  ROS_INFO("===========================================");
  return true;
}

bool ImageProcessor::createRosIO() {
  feature_pub = nh.advertise<CameraMeasurement>(
      "features", 3);
  tracking_info_pub = nh.advertise<TrackingInfo>(
      "tracking_info", 1);
  image_transport::ImageTransport it(nh);
  debug_stereo_pub = it.advertise("debug_stereo_image", 1);

  cam0_img_sub.subscribe(nh, "cam0_image", 10);
  cam1_img_sub.subscribe(nh, "cam1_image", 10);
  stereo_sub.connectInput(cam0_img_sub, cam1_img_sub);
  stereo_sub.registerCallback(&ImageProcessor::stereoCallback, this);
  imu_sub = nh.subscribe("imu", 50,
      &ImageProcessor::imuCallback, this);

  return true;
}

bool ImageProcessor::initialize() {
  if (!loadParameters()) return false;
  ROS_INFO("Finish loading ROS parameters...");

  feature_tracker_.initialize();

  if (!createRosIO()) return false;
  ROS_INFO("Finish creating ROS IO...");

  return true;
}

void ImageProcessor::stereoCallback(
    const sensor_msgs::ImageConstPtr& cam0_img,
    const sensor_msgs::ImageConstPtr& cam1_img) {

  //cout << "==================================" << endl;

  // Get the current image.
  cv_bridge::CvImageConstPtr cam0_curr_img_ptr = cv_bridge::toCvShare(cam0_img,
      sensor_msgs::image_encodings::MONO8);
  cv_bridge::CvImageConstPtr cam1_curr_img_ptr = cv_bridge::toCvShare(cam1_img,
      sensor_msgs::image_encodings::MONO8);

  feature_tracker_.stereoCallback(
      cam0_curr_img_ptr->image, cam1_curr_img_ptr->image,
      cam0_curr_img_ptr->header);

  // Draw results.
  ros::Time start_time_draw = ros::Time::now();
  drawFeatures();


  // Publish features in the current image.
  ros::Time start_time = ros::Time::now();
  publish();
  //ROS_INFO("Publishing: %f",
  //    (ros::Time::now()-start_time).toSec());
  
  feature_tracker_.prepareForNextFrame();
  
  return;
}

void ImageProcessor::imuCallback(
    const sensor_msgs::ImuConstPtr& msg) {
  // Wait for the first image to be set.
  feature_tracker_.imuCallback(msg);
  return;
}

void ImageProcessor::drawFeatures() {
  if(debug_stereo_pub.getNumSubscribers() > 0) {
    cv::Mat out_img;
    if (feature_tracker_.processor_config.monocular) {
      out_img = feature_tracker_.drawFeaturesMono();
    } else {
      out_img = feature_tracker_.drawFeaturesStereo();
    }
    cv_bridge::CvImage debug_image(feature_tracker_.cam0_curr_img_header_, "bgr8", out_img);
    debug_stereo_pub.publish(debug_image.toImageMsg());
  }
  return;
}

void ImageProcessor::publish() {
  vector<feature_tracker::FeatureIDType> curr_ids(0);  
  feature_tracker_.getCurrentFeatureIds(&curr_ids);

  vector<Point2f> curr_cam0_points_undistorted(0);
  vector<Point2f> curr_cam1_points_undistorted(0);
  feature_tracker_.getCurrentFeaturesUndistorted(
      &curr_cam0_points_undistorted, 
      &curr_cam1_points_undistorted);

  // Publish features.
  CameraMeasurementPtr feature_msg_ptr(new CameraMeasurement);
  feature_msg_ptr->header.stamp = feature_tracker_.cam0_curr_img_header_.stamp;

  for (int i = 0; i < curr_ids.size(); ++i) {
    feature_msg_ptr->features.push_back(FeatureMeasurement());
    feature_msg_ptr->features[i].id = curr_ids[i];
    feature_msg_ptr->features[i].u0 = curr_cam0_points_undistorted[i].x;
    feature_msg_ptr->features[i].v0 = curr_cam0_points_undistorted[i].y;
  }
  if (curr_cam1_points_undistorted.size()) {
    for (int i = 0; i < curr_ids.size(); ++i) {
        feature_msg_ptr->features[i].u1 = curr_cam1_points_undistorted[i].x;
        feature_msg_ptr->features[i].v1 = curr_cam1_points_undistorted[i].y;
    }
  }

  feature_pub.publish(feature_msg_ptr);

  // Publish tracking info.
  TrackingInfoPtr tracking_info_msg_ptr(new TrackingInfo());
  tracking_info_msg_ptr->header.stamp = feature_tracker_.cam0_curr_img_header_.stamp;
  tracking_info_msg_ptr->before_tracking = feature_tracker_.before_tracking;
  tracking_info_msg_ptr->after_tracking = feature_tracker_.after_tracking;
  tracking_info_msg_ptr->after_matching = feature_tracker_.after_matching;
  tracking_info_msg_ptr->after_ransac = feature_tracker_.after_ransac;
  tracking_info_pub.publish(tracking_info_msg_ptr);

  return;
}
} // end namespace msckf_vio
