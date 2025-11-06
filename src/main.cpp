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
}
