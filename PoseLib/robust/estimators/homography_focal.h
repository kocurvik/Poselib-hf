//
// Created by kocur on 12-Sep-24.
//

#ifndef POSELIB_HOMOGRAPHY_FOCAL_H
#define POSELIB_HOMOGRAPHY_FOCAL_H

#include <vector>
#include <Eigen/Core>
#include "PoseLib/types.h"

namespace poselib {

void get_homographies(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, int iterations,
                      double inlier_threshold, double distance_threshold, std::vector<Eigen::Matrix3d> *Hs);

std::vector<double> estimate_focal_homography(std::vector<Point2D> x1_1, std::vector<Point2D> x2_1,
                                              std::vector<Point2D> x1_2, std::vector<Point2D> x3_2,
                                              const Point2D& pp, int iterations, double inlier_threshold,
                                              double distance_threshold);
} // namespace poselib


#endif // POSELIB_HOMOGRAPHY_FOCAL_H
