#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

//  Paths of the images
extern const std::string image_1;
extern const std::string image_2;

extern const int image_not_found;
extern const int default_val;

// color bounds 
extern const cv::Scalar lower_red1;
extern const cv::Scalar upper_red1;
extern const cv::Scalar lower_red2;
extern const cv::Scalar upper_red2;
extern const cv::Scalar lower_blue;
extern const cv::Scalar upper_blue;
extern const cv::Scalar lower_yellow;
extern const cv::Scalar upper_yellow;
extern const cv::Scalar lower_white;
extern const cv::Scalar upper_white;
extern const cv::Scalar lower_black1;
extern const cv::Scalar upper_black1;
extern const cv::Scalar lower_black2;
extern const cv::Scalar upper_black2;

//  vectors for names and pair of bounds for colors
extern const std::vector<std::string> color_names;
extern const std::vector<std::pair<cv::Scalar, cv::Scalar>> color_bounds;

//  Constants for geometrical filter and barycenter of the boxes of the red cones
extern const double main_color_bound;
extern const double secondary_color_bound;
extern const int point_for_orb;
extern const int left_red_cone_selected;
extern const int right_red_cone_selected;

//  Indexes of colors
constexpr const int white_index = 0;
constexpr const int light_black_index = 1;
constexpr const int dark_black_index = 2;
constexpr const int red_index = 3;
constexpr const int blue_index = 4;
constexpr const int yellow_index = 5;

//  Kernels for image processing
extern const cv::Mat kernel_9;
extern const cv::Mat kernel_7;
extern const cv::Mat kernel_5;
extern const cv::Mat kernel_3;

// invalid point for quick point initialization
extern const cv::Point invalid_point;

//  Essential matrix for level 5
extern cv::Mat K;

#endif 
