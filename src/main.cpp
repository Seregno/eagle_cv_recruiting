#include "utils/processing.h"
#include <algorithm>

using namespace std;

int main() {

    printEnvironmentInfo();

    // Level 1: load and display the image 

    cv::Mat frame_1 = cv::imread(image_1);
    if (frame_1.empty()) {
        std::cerr << "Image not found!\n";
        return image_not_found;
    }

    printImg(frame_1,"Frame 1 clear");
    
    // Level 2: Cone detection

    // Pre processing of the image:

    // Changing color mode from bgr to hsv in order to remove color brightness and saturation  

    cv::Mat hsv_frame_1; // Hsv version of the original frame
    cv::cvtColor(frame_1, hsv_frame_1, cv::COLOR_BGR2HSV); 

    std::vector<cv::Mat> frames_cones_color(color_bounds.size()); // color detector for each color of the cones
    std::vector<std::vector<std::vector<cv::Point>>> colors_contours(color_bounds.size()); // contours of each color

    cv::Point leftmost_red_cone = cv::Point(-1,-1); // coordinates of the baricenter of the leftmost red cone
    cv::Point rightmost_red_cone = cv::Point(-1,-1); // coordinates of the baricenter of the rightmost red cone

    int bounding_area = 15; // minimum area for a contour to be recognized as a potential cone

    // vectors that will containt the point of the baricenters of the blue and yellow cones
    std::vector <cv::Point> blue_cones;
    std::vector <cv::Point> yellow_cones;

    // vector that will contain the points which the trach edges move along
    std::vector <cv::Point> circuit_points;

    // Pipeline application for each image of the cones color
    for(size_t i = 0; i < color_bounds.size(); i++)
    {
        const std::pair<cv::Scalar,cv::Scalar> bounds = color_bounds[i]; //bounds of the colors of each cones

        // Extract the current color
        cv::inRange(hsv_frame_1, bounds.first, bounds.second, frames_cones_color[i]); 

        // Tune the current mask with another one in order to combine different colors to create the best profile for the cones
        if( i > dark_black_index )
        {
                // Boundings to be initialized to classify the contours of the current color in order to detect a cone rather than a random black/white surface
                cv::Scalar main_color_lower_bound;
                cv::Scalar main_color_upper_bound;
                cv::Scalar secondary_color_lower_bound;
                cv::Scalar secondary_color_upper_bound;
            switch (i)
            {
                case dark_black_index:
                    cv::bitwise_or(frames_cones_color[i], frames_cones_color[1], frames_cones_color[i]);
                break;

                case red_index:
                    cv::bitwise_or(frames_cones_color[i], frames_cones_color[white_index], frames_cones_color[i]);
                    cv::GaussianBlur(frames_cones_color[i], frames_cones_color[i], cv::Size(5,5), 0);
                    processConeMask(frames_cones_color[i], kernel_5, kernel_5, kernel_3); // 5  5 3

                    // Setting bounding for cones size and colors
                    bounding_area = 150;
                    main_color_lower_bound = color_bounds[red_index].first;
                    main_color_upper_bound = color_bounds[red_index].second;
                    secondary_color_lower_bound = color_bounds[white_index].first;
                    secondary_color_upper_bound = color_bounds[white_index].second;
                break;

                case blue_index:
                    cv::bitwise_or(frames_cones_color[i], frames_cones_color[white_index], frames_cones_color[i]);
                    cv::GaussianBlur(frames_cones_color[i], frames_cones_color[i], cv::Size(5,5), 0);
                    processConeMask(frames_cones_color[i], kernel_5, kernel_9, kernel_3); // standard is 5, 5, 3

                    // Setting bounding for cones size and colors
                    bounding_area = 10;
                    main_color_lower_bound = color_bounds[blue_index].first;
                    main_color_upper_bound = color_bounds[blue_index].second;
                    secondary_color_lower_bound = color_bounds[white_index].first;
                    secondary_color_upper_bound = color_bounds[white_index].second;  
                break;

                case yellow_index:
                    cv::bitwise_or(frames_cones_color[i], frames_cones_color[dark_black_index], frames_cones_color[i]);
                    //cv::medianBlur(frames_cones_color[i], frames_cones_color[i], 3);
                    processConeMask(frames_cones_color[i], kernel_5, kernel_5, kernel_3); // standard is 5, 5, 3

                    // Setting bounding for cones size and colors
                    bounding_area = 10;
                    main_color_lower_bound = color_bounds[yellow_index].first;
                    main_color_upper_bound = color_bounds[yellow_index].second;
                    secondary_color_lower_bound = color_bounds[dark_black_index].first;
                    secondary_color_upper_bound = color_bounds[dark_black_index].second;
                break;

                default:
                    cout <<"No color tuning needed" <<endl;
                break;
            }

            std::vector<cv::Vec4i> hierarchy;

            // Find the contours of the white areas
            cv::findContours(frames_cones_color[i], colors_contours[i], hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : colors_contours[i]) {

                // Initializing variables for the approximation of the contour
                std::vector<cv::Point> approx;
                double epsilon = 0.02 * cv::arcLength(contour, true); 
                cv::approxPolyDP(contour, approx, epsilon, true);

                // Check if the shape is similar to a triangle
                if (approx.size() >= 3 && approx.size() <= 9) {

                    // Initializing geometrical information
                    double area = cv::contourArea(contour);
                    cv::Rect box = cv::boundingRect(contour);
                    double aspectRatio = (double)box.width / box.height;

                    if(area > bounding_area && aspectRatio < 1 )
                    { 
                        cv::Mat main_color_mask, secondary_color_mask;
                        cv::Mat roi_hsv = hsv_frame_1(box); 

                        cv::inRange(roi_hsv, main_color_lower_bound, main_color_upper_bound, main_color_mask);
                        cv::inRange(roi_hsv, secondary_color_lower_bound, secondary_color_upper_bound, secondary_color_mask);

                        int contour_area = (box.width * box.height);

                        // Calculating how many of the pixel with the same color as the cones are in each boc
                        double main_color_ratio = (double)cv::countNonZero(main_color_mask) / contour_area;
                        double secondary_color_ratio = (double)cv::countNonZero(secondary_color_mask) / contour_area;

                        // Level 3: classifying each cones with a different borders color based on the ones of the found one
                        
                        /*
                        cout <<"Color: " <<color_names[i] <<endl;
                        cout <<"Main color ratio " <<main_color_ratio <<" vs secondary color ratio " <<secondary_color_ratio <<endl; 
                        */
                        // Draw only those boxes with the colors of the cones and a relatively small amount of white
                        //  This is to prevent openCV from detecting the white stripe of a red or blue cone as a separate item

                        if ( main_color_ratio > main_color_bound && secondary_color_ratio < secondary_color_bound ) {
                            
                            cv::rectangle(frame_1, box, bounds.first, 2);
                            cv::Point new_cone = cv::Point( box.x + (box.width / 2), box.y + (box.height / 2) ); // creating the coordinates of the new cones corresponding to the ones of the center of the already drawn box
                            
                            // Storing and classifying the new cone in order to use its coordinates to detect the circuit in the next level
                            switch (i)
                            {
                                case red_index: // checking if the new red cone is the left most or right most
                                    assign_red_cone(leftmost_red_cone, new_cone ,left_red_cone_selected);
                                    assign_red_cone(rightmost_red_cone, new_cone ,right_red_cone_selected);
                                break;
                                
                                case blue_index:
                                    blue_cones.push_back(new_cone);
                                break;
                                
                                case yellow_index:
                                    yellow_cones.push_back(new_cone);
                                break;
                                
                                default:
                                    cout <<"Unknown item detected" <<endl;
                                break;
                            }
                        }
                    }
                }
            }   
        }
    }

    printImg(frame_1,"Cones detected");

    // Level 4: detect circuit

    // Detect the point of the circuit by calculating the median point between the current blue/yellow cone with the closest one on the other side

    cv::Point starting_point = median_point(leftmost_red_cone, rightmost_red_cone);
    circuit_points.push_back(starting_point);
    for(size_t i = 0; i < blue_cones.size(); i++)
    {
        cv::Point new_circuit_point = get_circuit_point(blue_cones[i], yellow_cones);
        circuit_points.push_back(new_circuit_point);
    }

    for(size_t i = 0; i < yellow_cones.size(); i++)
    {
        cv::Point new_circuit_point = get_circuit_point(yellow_cones[i], blue_cones);
        circuit_points.push_back(new_circuit_point);
    }
    
    // Sort the points of the circuit based on how close they are to a starting point

    if(circuit_points.size() > 0)
    {
        sortCircuitPoints(starting_point, circuit_points);
        drawCircuit(frame_1,circuit_points);
        printImg(frame_1, "Circuit detected");
    }

    // Level 5: Odometry and Pose estimation

    pose_estimation();

    return default_val;
<<<<<<< HEAD
}
=======
}

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
>>>>>>> 9180210d31f7b741764275fb59d383bcafd748f1
