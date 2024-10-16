//
// Created by kocur on 15-Oct-24.
//

#ifndef POSELIB_COEFF_H
#define POSELIB_COEFF_H

#include "PoseLib/solvers/var/coeffs_var_eq1.cc"
#include "PoseLib/solvers/var/coeffs_var_eq2.cc"
#include "PoseLib/solvers/var/coeffs_var_eq3.cc"
#include "PoseLib/solvers/var/coeffs_var_eq4.cc"
#include "PoseLib/solvers/var/coeffs_var_eq5.cc"
#include "PoseLib/solvers/var/coeffs_var_eq6.cc"
#include "PoseLib/solvers/var/coeffs_var_eq7.cc"

Eigen::MatrixXd var_eq1(double a0, double a1, double a2, double a3, double a4,
                        double a5, double b0, double b1, double b2, double b3,
                        double b4, double b5, double c0, double c1, double c2,
                        double c3, double c4, double c5, double d0, double d1,
                        double d2, double d3, double d4, double d5);


Eigen::MatrixXd var_eq2(double a0, double a1, double a2, double a3, double a4,
                        double a5, double b0, double b1, double b2, double b3,
                        double b4, double b5, double c0, double c1, double c2,
                        double c3, double c4, double c5, double d0, double d1,
                        double d2, double d3, double d4, double d5);

Eigen::MatrixXd var_eq3(double a0, double a1, double a2, double a3, double a4,
                 double a5, double b0, double b1, double b2, double b3,
                 double b4, double b5, double c0, double c1, double c2,
                 double c3, double c4, double c5, double d0, double d1,
                 double d2, double d3, double d4, double d5);

Eigen::MatrixXd var_eq4(double a0, double a1, double a2, double a3, double a4,
                 double a5, double b0, double b1, double b2, double b3,
                 double b4, double b5, double c0, double c1, double c2,
                 double c3, double c4, double c5, double d0, double d1,
                 double d2, double d3, double d4, double d5);

Eigen::MatrixXd var_eq5(double a0, double a1, double a2, double a3, double a4,
                 double a5, double b0, double b1, double b2, double b3,
                 double b4, double b5, double c0, double c1, double c2,
                 double c3, double c4, double c5, double d0, double d1,
                 double d2, double d3, double d4, double d5);

Eigen::MatrixXd var_eq6(double a0, double a1, double a2, double a3, double a4,
                 double a5, double b0, double b1, double b2, double b3,
                 double b4, double b5, double c0, double c1, double c2,
                 double c3, double c4, double c5, double d0, double d1,
                 double d2, double d3, double d4, double d5);

Eigen::MatrixXd var_eq7(double a0, double a1, double a2, double a3, double a4,
                 double a5, double b0, double b1, double b2, double b3,
                 double b4, double b5, double c0, double c1, double c2,
                 double c3, double c4, double c5, double d0, double d1,
                 double d2, double d3, double d4, double d5);


#endif // POSELIB_COEFF_H
