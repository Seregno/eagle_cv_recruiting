#ifndef PROCESSING_H
#define PROCESSING_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "constants.h"

// ─────────────────────────────────────────────
//  Environment and Debug Utilities
// ─────────────────────────────────────────────
void printEnvironmentInfo();
void printImg(cv::Mat to_print, std::string msg);

// ─────────────────────────────────────────────
//  Image Processing
// ─────────────────────────────────────────────
void processConeMask(cv::Mat& mask, 
                     const cv::Mat& kernel_dilate, 
                     const cv::Mat& kernel_morph_open, 
                     const cv::Mat& kernel_morph_close);

// ─────────────────────────────────────────────
//  Geometry & Math Utilities
// ─────────────────────────────────────────────
double point_distance(const cv::Point& a, const cv::Point& b);
cv::Point median_point(const cv::Point& a, const cv::Point& b);

// ─────────────────────────────────────────────
//  Cones and Circuit Utilities
// ─────────────────────────────────────────────
void assign_red_cone(cv::Point& red_cone, const cv::Point& new_cone, const int selected_cone);

cv::Point get_circuit_point(const cv::Point& current_cone, 
                            std::vector<cv::Point>& other_side_cones);

void sortCircuitPoints(const cv::Point& starting_point, 
                       std::vector<cv::Point>& circuit_points);

void drawCircuit(cv::Mat& image, const std::vector<cv::Point>& circuit_points);

// ─────────────────────────────────────────────
//  Pose Estimation
// ─────────────────────────────────────────────
void pose_estimation();

#endif // PROCESSING_H
