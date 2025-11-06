#include "processing.h"

void printEnvironmentInfo() {
    std::cout << "------------------------------------\n";
    std::cout << "   Runtime Environment Information   \n";
    std::cout << "------------------------------------\n";
    std::cout << "OpenCV version : " << CV_VERSION << "\n";

    #if defined(__clang__)
            std::cout << "Compiler        : Clang " 
                << __clang_major__ << "." << __clang_minor__ << "\n";
    #elif defined(__GNUC__)
        std::cout << "Compiler        : GCC " 
                << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n";
    #elif defined(_MSC_VER)
        std::cout << "Compiler        : MSVC " << _MSC_VER << "\n";
    #else
        std::cout << "Compiler        : Unknown\n";
    #endif

        std::cout << "C++ standard    : ";
    #if __cplusplus == 199711L
        std::cout << "C++98\n";
    #elif __cplusplus == 201103L
        std::cout << "C++11\n";
    #elif __cplusplus == 201402L
        std::cout << "C++14\n";
    #elif __cplusplus == 201703L
        std::cout << "C++17\n";
    #elif __cplusplus == 202002L
        std::cout << "C++20\n";
    #elif __cplusplus > 202002L
        std::cout << "C++23 or newer\n";
    #else
        std::cout << "Unknown (" << __cplusplus << ")\n";
    #endif
        std::cout << "------------------------------------\n\n";
}

void printImg(cv::Mat to_print, std::string msg)
{
    cv::imshow(msg, to_print);
    cv::waitKey(default_val);
}

void processConeMask(cv::Mat& mask, const cv::Mat& kernel_dilate, const cv::Mat& kernel_morph_open, const cv::Mat& kernel_morph_close)
{
    /*
        Classic pipeline for the detection of coloure object of a different shape from the other artifacts in an image

        First we dilate the colors joining closer areas
        Then we fill the black holes inside white spaces
        Last, we cut some extra edges and noise
    */
    cv::dilate(mask, mask, kernel_dilate);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_morph_close);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel_morph_open);
}


double point_distance(const cv::Point& a, const cv::Point& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

cv::Point median_point(const cv::Point& a, const cv::Point& b) {
    return cv::Point((a.x + b.x) / 2, (a.y + b.y) / 2);
}

void assign_red_cone(cv::Point& red_cone, const cv::Point& new_cone, const int selected_cone) {
    if( selected_cone == left_red_cone_selected && ( (red_cone.x == invalid_point.x && red_cone.y == invalid_point.y) || (red_cone.x > new_cone.x) ) ) // unassigned cone
    {
        red_cone.x = new_cone.x;
        red_cone.y = new_cone.y;
    }
    else if( selected_cone == right_red_cone_selected && ( (red_cone.x == invalid_point.x && red_cone.y == invalid_point.y) || (red_cone.x < new_cone.x) ) )
    {
        red_cone.x = new_cone.x;
        red_cone.y = new_cone.y;
    }   
}

cv::Point get_circuit_point(const cv::Point& current_cone, std::vector <cv::Point>& other_side_cones)
{
    if (other_side_cones.empty())
    {
        return current_cone;
    }
    cv::Point closest_cone = other_side_cones[0];
    double best_len = point_distance(current_cone, other_side_cones[0]);
    double current_len;
    for(size_t i = 1; i < other_side_cones.size(); i++)
    {
        current_len = point_distance(current_cone, other_side_cones[i]);
        if (current_len < best_len)
        {
            closest_cone = other_side_cones[i];
            best_len = current_len;
        }
    }
    return median_point(current_cone, closest_cone);
}

// Sort the points of the circuit considering first the ones with a bigger x value, eventually sorting considering  y value

void sortCircuitPoints(const cv::Point& starting_point,std::vector<cv::Point>& circuit_points) {
    std::sort(circuit_points.begin(), circuit_points.end(),
              [&starting_point](const cv::Point& a, const cv::Point& b) {
                  return point_distance(a,starting_point) > point_distance(b, starting_point);          
              });
}

void drawCircuit(cv::Mat& image, const std::vector<cv::Point>& circuit_points) {
    // Drawing every point as a circle
    for (const auto& pt : circuit_points) {
        cv::circle(image, pt, 5, cv::Scalar(0, 0, 255), -1); // rosso, riempito
    }

    // Joining points with lines
    for (size_t i = 1; i < circuit_points.size(); ++i) {
        cv::line(image, circuit_points[i-1], circuit_points[i], cv::Scalar(255, 0, 0), 2); // blu, spessore 2
    }
}

void pose_estimation()
{
    // Opening the two clean images
    cv::Mat frame_1_clean = cv::imread(image_1);
    cv::Mat frame_2_clean = cv::imread(image_2);
    
    // Checking if the two frames have been read correctly
    if (frame_2_clean.empty()) {
        std::cerr << "Frame 1 not found!\n";
        return;
    }

    if (frame_2_clean.empty()) {
        std::cerr << "Frame 2 not found!\n";
        return;
    }

    // Instantiating orb
    cv::Ptr<cv::ORB> orb = cv::ORB::create(point_for_orb);

    // Instantiating data structures to memorize the features
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    // Detecting features from the frames
    orb->detectAndCompute(frame_1_clean, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(frame_2_clean, cv::noArray(), keypoints2, descriptors2);

    // Instantiating a Brute Force Matcher in order to get the similarity between the most significant points of the two frames
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Filtering the matches

    double max_dist = 0; double min_dist = 100;
    for (auto& m : matches) {
        double dist = m.distance;
        min_dist = std::min(min_dist, dist);
        max_dist = std::max(max_dist, dist);
    }
    std::vector<cv::DMatch> good_matches;
    for (auto& m : matches) {
        if (m.distance <= std::max(2 * min_dist, 30.0))
            good_matches.push_back(m);
    }

    // Converting keypoints into 2D points

    std::vector<cv::Point2f> pts1, pts2;
    for (auto& m : good_matches) {
        pts1.push_back(keypoints1[m.queryIdx].pt);
        pts2.push_back(keypoints2[m.trainIdx].pt);
    }

    // Calculating the pose matrix using the essential matrix

    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0);

    // Getting the pose (R, t)
    cv::Mat R, t;
    int inliers = cv::recoverPose(E, pts1, pts2, K, R, t);

    // Printing the results:

    std::cout << "Rotation:\n" << R << std::endl;
    std::cout << "Traslation:\n" << t << std::endl;
}