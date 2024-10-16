//
// Created by kocur on 15-Oct-24.
//

#ifndef POSELIB_HOMO3F_UTILS_H
#define POSELIB_HOMO3F_UTILS_H

#include <Eigen/Core>

namespace poselib {
double coeff0(double b0, double b1, double b2, double b3, double b4, double b5, double d0, double d1, double d2,
              double d3, double d4, double d5);

Eigen::VectorXd equal_coeff(double a0, double a1, double a2, double a3, double a4,
                            double a5, double b0, double b1, double b2, double b3,
                            double b4, double b5, double c0, double c1, double c2,
                            double c3, double c4, double c5, double d0, double d1,
                            double d2, double d3, double d4, double d5);

Eigen::VectorXd coeff_sequal(double a0, double a1, double a2, double a3, double a4,
                             double a5, double b0, double b1, double b2, double b3,
                             double b4, double b5, double c0, double c1, double c2,
                             double c3, double c4, double c5, double d0, double d1,
                             double d2, double d3, double d4, double d5);
}

#endif // POSELIB_HOMO3F_UTILS_H
