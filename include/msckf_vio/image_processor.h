/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_IMAGE_PROCESSOR_H
#define MSCKF_VIO_IMAGE_PROCESSOR_H

#include <vector>
#include <map>
// #include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <msckf_vio/feature_tracker.h>

namespace msckf_vio {

class ImageProcessor {
public:
  // Constructor
  ImageProcessor(ros::NodeHandle& n);
  // Disable copy and assign constructors.
  ImageProcessor(const ImageProcessor&) = delete;
  ImageProcessor operator=(const ImageProcessor&) = delete;

  // Destructor
  ~ImageProcessor();

  // Initialize the object.
  bool initialize();

  typedef std::shared_ptr<ImageProcessor> Ptr;
  typedef std::shared_ptr<const ImageProcessor> ConstPtr;

private:
  /*
   * @brief loadParameters
   *    Load parameters from the parameter server.
   */
  bool loadParameters();

  /*
   * @brief createRosIO
   *    Create ros publisher and subscirbers.
   */
  bool createRosIO();

  /*
   * @brief publish
   *    Publish the features on the current image including
   *    both the tracked and newly detected ones.
   */
  void publish();

    /*
   * @brief imuCallback
   *    Callback function for the imu message.
   * @param msg IMU msg.
   */
  void imuCallback(const sensor_msgs::ImuConstPtr& msg);

  /*
   * @brief stereoCallback
   *    Callback function for the stereo images.
   * @param cam0_img left image.
   * @param cam1_img right image.
   */
  void stereoCallback(
      const sensor_msgs::ImageConstPtr& cam0_img,
      const sensor_msgs::ImageConstPtr& cam1_img);

  /*
   * @brief drawFeatures
   *    Draw tracked and newly detected features on the
   *    monocular/stereo images.
   */
  void drawFeatures();

  FeatureTracker feature_tracker_;

  // Ros node handle
  ros::NodeHandle nh;

  // Subscribers and publishers.
  message_filters::Subscriber<
    sensor_msgs::Image> cam0_img_sub;
  message_filters::Subscriber<
    sensor_msgs::Image> cam1_img_sub;
  message_filters::TimeSynchronizer<
    sensor_msgs::Image, sensor_msgs::Image> stereo_sub;
  ros::Subscriber imu_sub;
  ros::Publisher feature_pub;
  ros::Publisher tracking_info_pub;
  image_transport::Publisher debug_stereo_pub;
};

typedef ImageProcessor::Ptr ImageProcessorPtr;
typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;

} // end namespace msckf_vio

#endif
