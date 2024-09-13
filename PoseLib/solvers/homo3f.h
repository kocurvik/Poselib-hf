//
// Created by kocur on 05-Sep-24.
//

#ifndef POSELIB_HOMO3F_H
#define POSELIB_HOMO3F_H

namespace poselib {
std::vector<double> solver_homo3f(Eigen::Matrix3d &H12, Eigen::Matrix3d &H13);
}

#endif // POSELIB_HOMO3F_H
