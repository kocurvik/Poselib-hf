//
// Created by kocur on 17-Sep-24.
//

#ifndef POSELIB_PLANE_PARALLAX_SHARED_FOCAL_H
#define POSELIB_PLANE_PARALLAX_SHARED_FOCAL_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Core>
namespace poselib {

int plane_parallax_5pt_shared_focal(const Eigen::Matrix3d &H, const Point2D &x1, const Point2D &x2,
                                    ImagePairVector *out_image_pairs);

} //namespace poselib

#endif // POSELIB_PLANE_PARALLAX_SHARED_FOCAL_H
