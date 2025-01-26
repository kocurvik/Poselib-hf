//
// Created by kocur on 17-Sep-24.
//
#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/types.h"

#include <Eigen/Core>
#include <Eigen/src/Eigenvalues/EigenSolver.h>
#include <iostream>

using namespace Eigen;
namespace poselib {
MatrixXcd solver_4plus1(const VectorXd &data) {
    // Action =  y
    // Quotient ring basis (V) = 1,x,x*y,y,y^2,
    // Available monomials (RR*V) = x*y^2,y^3,1,x,x*y,y,y^2,

    const double *d = data.data();
    VectorXd coeffs(108);
    coeffs[0] =
        -std::pow(d[5], 2) * d[6] * std::pow(d[11], 2) * d[12] + 2 * d[3] * d[5] * d[8] * std::pow(d[11], 2) * d[12] +
        2 * d[2] * d[5] * d[6] * d[11] * std::pow(d[12], 2) - 2 * d[2] * d[3] * d[8] * d[11] * std::pow(d[12], 2) -
        2 * d[0] * d[5] * d[8] * d[11] * std::pow(d[12], 2) - std::pow(d[2], 2) * d[6] * std::pow(d[12], 3) +
        2 * d[0] * d[2] * d[8] * std::pow(d[12], 3);
    coeffs[1] =
        -std::pow(d[5], 2) * d[6] * d[10] * std::pow(d[11], 2) + 2 * d[3] * d[5] * d[8] * d[10] * std::pow(d[11], 2) -
        2 * std::pow(d[5], 2) * d[6] * d[9] * d[11] * d[12] + 4 * d[3] * d[5] * d[8] * d[9] * d[11] * d[12] +
        4 * d[2] * d[5] * d[6] * d[10] * d[11] * d[12] - 4 * d[2] * d[3] * d[8] * d[10] * d[11] * d[12] -
        4 * d[0] * d[5] * d[8] * d[10] * d[11] * d[12] + 2 * d[2] * d[5] * d[6] * d[9] * std::pow(d[12], 2) -
        2 * d[2] * d[3] * d[8] * d[9] * std::pow(d[12], 2) - 2 * d[0] * d[5] * d[8] * d[9] * std::pow(d[12], 2) -
        3 * std::pow(d[2], 2) * d[6] * d[10] * std::pow(d[12], 2) +
        6 * d[0] * d[2] * d[8] * d[10] * std::pow(d[12], 2) - d[3] * std::pow(d[5], 2) * std::pow(d[11], 2) +
        2 * d[0] * std::pow(d[5], 2) * d[11] * d[12] + std::pow(d[2], 2) * d[3] * std::pow(d[12], 2) -
        2 * d[0] * d[2] * d[5] * std::pow(d[12], 2);
    coeffs[2] =
        std::pow(d[3], 2) * d[6] * std::pow(d[11], 2) * d[12] - std::pow(d[4], 2) * d[6] * std::pow(d[11], 2) * d[12] +
        2 * d[3] * d[4] * d[7] * std::pow(d[11], 2) * d[12] + d[6] * std::pow(d[8], 2) * std::pow(d[11], 2) * d[12] -
        2 * d[0] * d[3] * d[6] * d[11] * std::pow(d[12], 2) + 2 * d[1] * d[4] * d[6] * d[11] * std::pow(d[12], 2) -
        2 * d[1] * d[3] * d[7] * d[11] * std::pow(d[12], 2) - 2 * d[0] * d[4] * d[7] * d[11] * std::pow(d[12], 2) +
        std::pow(d[0], 2) * d[6] * std::pow(d[12], 3) - std::pow(d[1], 2) * d[6] * std::pow(d[12], 3) +
        2 * d[0] * d[1] * d[7] * std::pow(d[12], 3) + d[6] * std::pow(d[8], 2) * std::pow(d[12], 3);
    coeffs[3] =
        -2 * std::pow(d[5], 2) * d[6] * d[9] * d[10] * d[11] + 4 * d[3] * d[5] * d[8] * d[9] * d[10] * d[11] +
        2 * d[2] * d[5] * d[6] * std::pow(d[10], 2) * d[11] - 2 * d[2] * d[3] * d[8] * std::pow(d[10], 2) * d[11] -
        2 * d[0] * d[5] * d[8] * std::pow(d[10], 2) * d[11] - std::pow(d[5], 2) * d[6] * std::pow(d[9], 2) * d[12] +
        2 * d[3] * d[5] * d[8] * std::pow(d[9], 2) * d[12] + 4 * d[2] * d[5] * d[6] * d[9] * d[10] * d[12] -
        4 * d[2] * d[3] * d[8] * d[9] * d[10] * d[12] - 4 * d[0] * d[5] * d[8] * d[9] * d[10] * d[12] -
        3 * std::pow(d[2], 2) * d[6] * std::pow(d[10], 2) * d[12] +
        6 * d[0] * d[2] * d[8] * std::pow(d[10], 2) * d[12] - 2 * d[3] * std::pow(d[5], 2) * d[9] * d[11] +
        2 * d[0] * std::pow(d[5], 2) * d[10] * d[11] + 2 * d[0] * std::pow(d[5], 2) * d[9] * d[12] +
        2 * std::pow(d[2], 2) * d[3] * d[10] * d[12] - 4 * d[0] * d[2] * d[5] * d[10] * d[12];
    coeffs[4] =
        std::pow(d[3], 2) * d[6] * d[10] * std::pow(d[11], 2) - std::pow(d[4], 2) * d[6] * d[10] * std::pow(d[11], 2) +
        2 * d[3] * d[4] * d[7] * d[10] * std::pow(d[11], 2) + d[6] * std::pow(d[8], 2) * d[10] * std::pow(d[11], 2) +
        2 * std::pow(d[3], 2) * d[6] * d[9] * d[11] * d[12] - 2 * std::pow(d[4], 2) * d[6] * d[9] * d[11] * d[12] +
        4 * d[3] * d[4] * d[7] * d[9] * d[11] * d[12] + 2 * d[6] * std::pow(d[8], 2) * d[9] * d[11] * d[12] -
        4 * d[0] * d[3] * d[6] * d[10] * d[11] * d[12] + 4 * d[1] * d[4] * d[6] * d[10] * d[11] * d[12] -
        4 * d[1] * d[3] * d[7] * d[10] * d[11] * d[12] - 4 * d[0] * d[4] * d[7] * d[10] * d[11] * d[12] -
        2 * d[0] * d[3] * d[6] * d[9] * std::pow(d[12], 2) + 2 * d[1] * d[4] * d[6] * d[9] * std::pow(d[12], 2) -
        2 * d[1] * d[3] * d[7] * d[9] * std::pow(d[12], 2) - 2 * d[0] * d[4] * d[7] * d[9] * std::pow(d[12], 2) +
        3 * std::pow(d[0], 2) * d[6] * d[10] * std::pow(d[12], 2) -
        3 * std::pow(d[1], 2) * d[6] * d[10] * std::pow(d[12], 2) +
        6 * d[0] * d[1] * d[7] * d[10] * std::pow(d[12], 2) +
        3 * d[6] * std::pow(d[8], 2) * d[10] * std::pow(d[12], 2) - std::pow(d[3], 3) * std::pow(d[11], 2) -
        d[3] * std::pow(d[4], 2) * std::pow(d[11], 2) - 2 * d[5] * d[6] * d[8] * std::pow(d[11], 2) +
        d[3] * std::pow(d[8], 2) * std::pow(d[11], 2) + 2 * d[0] * std::pow(d[3], 2) * d[11] * d[12] +
        2 * d[0] * std::pow(d[4], 2) * d[11] * d[12] - 2 * d[0] * std::pow(d[8], 2) * d[11] * d[12] -
        std::pow(d[0], 2) * d[3] * std::pow(d[12], 2) + std::pow(d[1], 2) * d[3] * std::pow(d[12], 2) -
        2 * d[0] * d[1] * d[4] * std::pow(d[12], 2) - 2 * d[5] * d[6] * d[8] * std::pow(d[12], 2) -
        d[3] * std::pow(d[8], 2) * std::pow(d[12], 2);
    coeffs[5] = std::pow(d[6], 3) * std::pow(d[11], 2) * d[12] + d[6] * std::pow(d[7], 2) * std::pow(d[11], 2) * d[12] +
                std::pow(d[6], 3) * std::pow(d[12], 3) + d[6] * std::pow(d[7], 2) * std::pow(d[12], 3);
    coeffs[6] =
        -std::pow(d[5], 2) * d[6] * std::pow(d[9], 2) * d[10] + 2 * d[3] * d[5] * d[8] * std::pow(d[9], 2) * d[10] +
        2 * d[2] * d[5] * d[6] * d[9] * std::pow(d[10], 2) - 2 * d[2] * d[3] * d[8] * d[9] * std::pow(d[10], 2) -
        2 * d[0] * d[5] * d[8] * d[9] * std::pow(d[10], 2) - std::pow(d[2], 2) * d[6] * std::pow(d[10], 3) +
        2 * d[0] * d[2] * d[8] * std::pow(d[10], 3) - d[3] * std::pow(d[5], 2) * std::pow(d[9], 2) +
        2 * d[0] * std::pow(d[5], 2) * d[9] * d[10] + std::pow(d[2], 2) * d[3] * std::pow(d[10], 2) -
        2 * d[0] * d[2] * d[5] * std::pow(d[10], 2);
    coeffs[7] =
        2 * std::pow(d[3], 2) * d[6] * d[9] * d[10] * d[11] - 2 * std::pow(d[4], 2) * d[6] * d[9] * d[10] * d[11] +
        4 * d[3] * d[4] * d[7] * d[9] * d[10] * d[11] + 2 * d[6] * std::pow(d[8], 2) * d[9] * d[10] * d[11] -
        2 * d[0] * d[3] * d[6] * std::pow(d[10], 2) * d[11] + 2 * d[1] * d[4] * d[6] * std::pow(d[10], 2) * d[11] -
        2 * d[1] * d[3] * d[7] * std::pow(d[10], 2) * d[11] - 2 * d[0] * d[4] * d[7] * std::pow(d[10], 2) * d[11] +
        std::pow(d[3], 2) * d[6] * std::pow(d[9], 2) * d[12] - std::pow(d[4], 2) * d[6] * std::pow(d[9], 2) * d[12] +
        2 * d[3] * d[4] * d[7] * std::pow(d[9], 2) * d[12] + d[6] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[12] -
        4 * d[0] * d[3] * d[6] * d[9] * d[10] * d[12] + 4 * d[1] * d[4] * d[6] * d[9] * d[10] * d[12] -
        4 * d[1] * d[3] * d[7] * d[9] * d[10] * d[12] - 4 * d[0] * d[4] * d[7] * d[9] * d[10] * d[12] +
        3 * std::pow(d[0], 2) * d[6] * std::pow(d[10], 2) * d[12] -
        3 * std::pow(d[1], 2) * d[6] * std::pow(d[10], 2) * d[12] +
        6 * d[0] * d[1] * d[7] * std::pow(d[10], 2) * d[12] +
        3 * d[6] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[12] - 2 * std::pow(d[3], 3) * d[9] * d[11] -
        2 * d[3] * std::pow(d[4], 2) * d[9] * d[11] - 4 * d[5] * d[6] * d[8] * d[9] * d[11] +
        2 * d[3] * std::pow(d[8], 2) * d[9] * d[11] + 2 * d[0] * std::pow(d[3], 2) * d[10] * d[11] +
        2 * d[0] * std::pow(d[4], 2) * d[10] * d[11] - 2 * d[0] * std::pow(d[8], 2) * d[10] * d[11] +
        2 * d[0] * std::pow(d[3], 2) * d[9] * d[12] + 2 * d[0] * std::pow(d[4], 2) * d[9] * d[12] -
        2 * d[0] * std::pow(d[8], 2) * d[9] * d[12] - 2 * std::pow(d[0], 2) * d[3] * d[10] * d[12] +
        2 * std::pow(d[1], 2) * d[3] * d[10] * d[12] - 4 * d[0] * d[1] * d[4] * d[10] * d[12] -
        4 * d[5] * d[6] * d[8] * d[10] * d[12] - 2 * d[3] * std::pow(d[8], 2) * d[10] * d[12] +
        2 * d[2] * d[5] * d[6] * d[11] - 2 * d[2] * d[3] * d[8] * d[11] + 2 * d[0] * d[5] * d[8] * d[11] -
        std::pow(d[2], 2) * d[6] * d[12] + std::pow(d[5], 2) * d[6] * d[12] + 2 * d[0] * d[2] * d[8] * d[12] +
        2 * d[3] * d[5] * d[8] * d[12];
    coeffs[8] = std::pow(d[6], 3) * d[10] * std::pow(d[11], 2) + d[6] * std::pow(d[7], 2) * d[10] * std::pow(d[11], 2) +
                2 * std::pow(d[6], 3) * d[9] * d[11] * d[12] + 2 * d[6] * std::pow(d[7], 2) * d[9] * d[11] * d[12] +
                3 * std::pow(d[6], 3) * d[10] * std::pow(d[12], 2) +
                3 * d[6] * std::pow(d[7], 2) * d[10] * std::pow(d[12], 2) -
                d[3] * std::pow(d[6], 2) * std::pow(d[11], 2) - 2 * d[4] * d[6] * d[7] * std::pow(d[11], 2) +
                d[3] * std::pow(d[7], 2) * std::pow(d[11], 2) - 2 * d[0] * std::pow(d[6], 2) * d[11] * d[12] -
                2 * d[0] * std::pow(d[7], 2) * d[11] * d[12] - 3 * d[3] * std::pow(d[6], 2) * std::pow(d[12], 2) -
                2 * d[4] * d[6] * d[7] * std::pow(d[12], 2) - d[3] * std::pow(d[7], 2) * std::pow(d[12], 2);
    coeffs[9] =
        std::pow(d[3], 2) * d[6] * std::pow(d[9], 2) * d[10] - std::pow(d[4], 2) * d[6] * std::pow(d[9], 2) * d[10] +
        2 * d[3] * d[4] * d[7] * std::pow(d[9], 2) * d[10] + d[6] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[10] -
        2 * d[0] * d[3] * d[6] * d[9] * std::pow(d[10], 2) + 2 * d[1] * d[4] * d[6] * d[9] * std::pow(d[10], 2) -
        2 * d[1] * d[3] * d[7] * d[9] * std::pow(d[10], 2) - 2 * d[0] * d[4] * d[7] * d[9] * std::pow(d[10], 2) +
        std::pow(d[0], 2) * d[6] * std::pow(d[10], 3) - std::pow(d[1], 2) * d[6] * std::pow(d[10], 3) +
        2 * d[0] * d[1] * d[7] * std::pow(d[10], 3) + d[6] * std::pow(d[8], 2) * std::pow(d[10], 3) -
        std::pow(d[3], 3) * std::pow(d[9], 2) - d[3] * std::pow(d[4], 2) * std::pow(d[9], 2) -
        2 * d[5] * d[6] * d[8] * std::pow(d[9], 2) + d[3] * std::pow(d[8], 2) * std::pow(d[9], 2) +
        2 * d[0] * std::pow(d[3], 2) * d[9] * d[10] + 2 * d[0] * std::pow(d[4], 2) * d[9] * d[10] -
        2 * d[0] * std::pow(d[8], 2) * d[9] * d[10] - std::pow(d[0], 2) * d[3] * std::pow(d[10], 2) +
        std::pow(d[1], 2) * d[3] * std::pow(d[10], 2) - 2 * d[0] * d[1] * d[4] * std::pow(d[10], 2) -
        2 * d[5] * d[6] * d[8] * std::pow(d[10], 2) - d[3] * std::pow(d[8], 2) * std::pow(d[10], 2) +
        2 * d[2] * d[5] * d[6] * d[9] - 2 * d[2] * d[3] * d[8] * d[9] + 2 * d[0] * d[5] * d[8] * d[9] -
        std::pow(d[2], 2) * d[6] * d[10] + std::pow(d[5], 2) * d[6] * d[10] + 2 * d[0] * d[2] * d[8] * d[10] +
        2 * d[3] * d[5] * d[8] * d[10] + std::pow(d[2], 2) * d[3] - 2 * d[0] * d[2] * d[5] - d[3] * std::pow(d[5], 2);
    coeffs[10] = 2 * std::pow(d[6], 3) * d[9] * d[10] * d[11] + 2 * d[6] * std::pow(d[7], 2) * d[9] * d[10] * d[11] +
                 std::pow(d[6], 3) * std::pow(d[9], 2) * d[12] + d[6] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[12] +
                 3 * std::pow(d[6], 3) * std::pow(d[10], 2) * d[12] +
                 3 * d[6] * std::pow(d[7], 2) * std::pow(d[10], 2) * d[12] -
                 2 * d[3] * std::pow(d[6], 2) * d[9] * d[11] - 4 * d[4] * d[6] * d[7] * d[9] * d[11] +
                 2 * d[3] * std::pow(d[7], 2) * d[9] * d[11] - 2 * d[0] * std::pow(d[6], 2) * d[10] * d[11] -
                 2 * d[0] * std::pow(d[7], 2) * d[10] * d[11] - 2 * d[0] * std::pow(d[6], 2) * d[9] * d[12] -
                 2 * d[0] * std::pow(d[7], 2) * d[9] * d[12] - 6 * d[3] * std::pow(d[6], 2) * d[10] * d[12] -
                 4 * d[4] * d[6] * d[7] * d[10] * d[12] - 2 * d[3] * std::pow(d[7], 2) * d[10] * d[12] +
                 2 * d[0] * d[3] * d[6] * d[11] + 2 * d[1] * d[4] * d[6] * d[11] - 2 * d[1] * d[3] * d[7] * d[11] +
                 2 * d[0] * d[4] * d[7] * d[11] + std::pow(d[0], 2) * d[6] * d[12] - std::pow(d[1], 2) * d[6] * d[12] +
                 3 * std::pow(d[3], 2) * d[6] * d[12] + std::pow(d[4], 2) * d[6] * d[12] +
                 2 * d[0] * d[1] * d[7] * d[12] + 2 * d[3] * d[4] * d[7] * d[12];
    coeffs[11] = std::pow(d[6], 3) * std::pow(d[9], 2) * d[10] + d[6] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[10] +
                 std::pow(d[6], 3) * std::pow(d[10], 3) + d[6] * std::pow(d[7], 2) * std::pow(d[10], 3) -
                 d[3] * std::pow(d[6], 2) * std::pow(d[9], 2) - 2 * d[4] * d[6] * d[7] * std::pow(d[9], 2) +
                 d[3] * std::pow(d[7], 2) * std::pow(d[9], 2) - 2 * d[0] * std::pow(d[6], 2) * d[9] * d[10] -
                 2 * d[0] * std::pow(d[7], 2) * d[9] * d[10] - 3 * d[3] * std::pow(d[6], 2) * std::pow(d[10], 2) -
                 2 * d[4] * d[6] * d[7] * std::pow(d[10], 2) - d[3] * std::pow(d[7], 2) * std::pow(d[10], 2) +
                 2 * d[0] * d[3] * d[6] * d[9] + 2 * d[1] * d[4] * d[6] * d[9] - 2 * d[1] * d[3] * d[7] * d[9] +
                 2 * d[0] * d[4] * d[7] * d[9] + std::pow(d[0], 2) * d[6] * d[10] - std::pow(d[1], 2) * d[6] * d[10] +
                 3 * std::pow(d[3], 2) * d[6] * d[10] + std::pow(d[4], 2) * d[6] * d[10] +
                 2 * d[0] * d[1] * d[7] * d[10] + 2 * d[3] * d[4] * d[7] * d[10] - std::pow(d[0], 2) * d[3] +
                 std::pow(d[1], 2) * d[3] - std::pow(d[3], 3) - 2 * d[0] * d[1] * d[4] - d[3] * std::pow(d[4], 2);
    coeffs[12] =
        std::pow(d[5], 2) * d[6] * std::pow(d[11], 3) - 2 * d[3] * d[5] * d[8] * std::pow(d[11], 3) -
        2 * d[2] * d[5] * d[6] * std::pow(d[11], 2) * d[12] + 2 * d[2] * d[3] * d[8] * std::pow(d[11], 2) * d[12] +
        2 * d[0] * d[5] * d[8] * std::pow(d[11], 2) * d[12] + std::pow(d[2], 2) * d[6] * d[11] * std::pow(d[12], 2) -
        2 * d[0] * d[2] * d[8] * d[11] * std::pow(d[12], 2);
    coeffs[13] =
        3 * std::pow(d[5], 2) * d[6] * d[9] * std::pow(d[11], 2) - 6 * d[3] * d[5] * d[8] * d[9] * std::pow(d[11], 2) -
        2 * d[2] * d[5] * d[6] * d[10] * std::pow(d[11], 2) + 2 * d[2] * d[3] * d[8] * d[10] * std::pow(d[11], 2) +
        2 * d[0] * d[5] * d[8] * d[10] * std::pow(d[11], 2) - 4 * d[2] * d[5] * d[6] * d[9] * d[11] * d[12] +
        4 * d[2] * d[3] * d[8] * d[9] * d[11] * d[12] + 4 * d[0] * d[5] * d[8] * d[9] * d[11] * d[12] +
        2 * std::pow(d[2], 2) * d[6] * d[10] * d[11] * d[12] - 4 * d[0] * d[2] * d[8] * d[10] * d[11] * d[12] +
        std::pow(d[2], 2) * d[6] * d[9] * std::pow(d[12], 2) - 2 * d[0] * d[2] * d[8] * d[9] * std::pow(d[12], 2) +
        2 * d[2] * d[3] * d[5] * std::pow(d[11], 2) - d[0] * std::pow(d[5], 2) * std::pow(d[11], 2) -
        2 * std::pow(d[2], 2) * d[3] * d[11] * d[12] + d[0] * std::pow(d[2], 2) * std::pow(d[12], 2);
    coeffs[14] =
        -std::pow(d[3], 2) * d[6] * std::pow(d[11], 3) + std::pow(d[4], 2) * d[6] * std::pow(d[11], 3) -
        2 * d[3] * d[4] * d[7] * std::pow(d[11], 3) - d[6] * std::pow(d[8], 2) * std::pow(d[11], 3) +
        2 * d[0] * d[3] * d[6] * std::pow(d[11], 2) * d[12] - 2 * d[1] * d[4] * d[6] * std::pow(d[11], 2) * d[12] +
        2 * d[1] * d[3] * d[7] * std::pow(d[11], 2) * d[12] + 2 * d[0] * d[4] * d[7] * std::pow(d[11], 2) * d[12] -
        std::pow(d[0], 2) * d[6] * d[11] * std::pow(d[12], 2) + std::pow(d[1], 2) * d[6] * d[11] * std::pow(d[12], 2) -
        2 * d[0] * d[1] * d[7] * d[11] * std::pow(d[12], 2) - d[6] * std::pow(d[8], 2) * d[11] * std::pow(d[12], 2);
    coeffs[15] =
        3 * std::pow(d[5], 2) * d[6] * std::pow(d[9], 2) * d[11] - 6 * d[3] * d[5] * d[8] * std::pow(d[9], 2) * d[11] -
        4 * d[2] * d[5] * d[6] * d[9] * d[10] * d[11] + 4 * d[2] * d[3] * d[8] * d[9] * d[10] * d[11] +
        4 * d[0] * d[5] * d[8] * d[9] * d[10] * d[11] + std::pow(d[2], 2) * d[6] * std::pow(d[10], 2) * d[11] -
        2 * d[0] * d[2] * d[8] * std::pow(d[10], 2) * d[11] - 2 * d[2] * d[5] * d[6] * std::pow(d[9], 2) * d[12] +
        2 * d[2] * d[3] * d[8] * std::pow(d[9], 2) * d[12] + 2 * d[0] * d[5] * d[8] * std::pow(d[9], 2) * d[12] +
        2 * std::pow(d[2], 2) * d[6] * d[9] * d[10] * d[12] - 4 * d[0] * d[2] * d[8] * d[9] * d[10] * d[12] +
        4 * d[2] * d[3] * d[5] * d[9] * d[11] - 2 * d[0] * std::pow(d[5], 2) * d[9] * d[11] -
        2 * std::pow(d[2], 2) * d[3] * d[10] * d[11] - 2 * std::pow(d[2], 2) * d[3] * d[9] * d[12] +
        2 * d[0] * std::pow(d[2], 2) * d[10] * d[12];
    coeffs[16] =
        -3 * std::pow(d[3], 2) * d[6] * d[9] * std::pow(d[11], 2) +
        3 * std::pow(d[4], 2) * d[6] * d[9] * std::pow(d[11], 2) - 6 * d[3] * d[4] * d[7] * d[9] * std::pow(d[11], 2) -
        3 * d[6] * std::pow(d[8], 2) * d[9] * std::pow(d[11], 2) + 2 * d[0] * d[3] * d[6] * d[10] * std::pow(d[11], 2) -
        2 * d[1] * d[4] * d[6] * d[10] * std::pow(d[11], 2) + 2 * d[1] * d[3] * d[7] * d[10] * std::pow(d[11], 2) +
        2 * d[0] * d[4] * d[7] * d[10] * std::pow(d[11], 2) + 4 * d[0] * d[3] * d[6] * d[9] * d[11] * d[12] -
        4 * d[1] * d[4] * d[6] * d[9] * d[11] * d[12] + 4 * d[1] * d[3] * d[7] * d[9] * d[11] * d[12] +
        4 * d[0] * d[4] * d[7] * d[9] * d[11] * d[12] - 2 * std::pow(d[0], 2) * d[6] * d[10] * d[11] * d[12] +
        2 * std::pow(d[1], 2) * d[6] * d[10] * d[11] * d[12] - 4 * d[0] * d[1] * d[7] * d[10] * d[11] * d[12] -
        2 * d[6] * std::pow(d[8], 2) * d[10] * d[11] * d[12] - std::pow(d[0], 2) * d[6] * d[9] * std::pow(d[12], 2) +
        std::pow(d[1], 2) * d[6] * d[9] * std::pow(d[12], 2) - 2 * d[0] * d[1] * d[7] * d[9] * std::pow(d[12], 2) -
        d[6] * std::pow(d[8], 2) * d[9] * std::pow(d[12], 2) + d[0] * std::pow(d[3], 2) * std::pow(d[11], 2) +
        2 * d[1] * d[3] * d[4] * std::pow(d[11], 2) - d[0] * std::pow(d[4], 2) * std::pow(d[11], 2) +
        2 * d[2] * d[6] * d[8] * std::pow(d[11], 2) + d[0] * std::pow(d[8], 2) * std::pow(d[11], 2) -
        2 * std::pow(d[0], 2) * d[3] * d[11] * d[12] - 2 * std::pow(d[1], 2) * d[3] * d[11] * d[12] +
        2 * d[3] * std::pow(d[8], 2) * d[11] * d[12] + std::pow(d[0], 3) * std::pow(d[12], 2) +
        d[0] * std::pow(d[1], 2) * std::pow(d[12], 2) + 2 * d[2] * d[6] * d[8] * std::pow(d[12], 2) -
        d[0] * std::pow(d[8], 2) * std::pow(d[12], 2);
    coeffs[17] = -std::pow(d[6], 3) * std::pow(d[11], 3) - d[6] * std::pow(d[7], 2) * std::pow(d[11], 3) -
                 std::pow(d[6], 3) * d[11] * std::pow(d[12], 2) - d[6] * std::pow(d[7], 2) * d[11] * std::pow(d[12], 2);
    coeffs[18] =
        std::pow(d[5], 2) * d[6] * std::pow(d[9], 3) - 2 * d[3] * d[5] * d[8] * std::pow(d[9], 3) -
        2 * d[2] * d[5] * d[6] * std::pow(d[9], 2) * d[10] + 2 * d[2] * d[3] * d[8] * std::pow(d[9], 2) * d[10] +
        2 * d[0] * d[5] * d[8] * std::pow(d[9], 2) * d[10] + std::pow(d[2], 2) * d[6] * d[9] * std::pow(d[10], 2) -
        2 * d[0] * d[2] * d[8] * d[9] * std::pow(d[10], 2) + 2 * d[2] * d[3] * d[5] * std::pow(d[9], 2) -
        d[0] * std::pow(d[5], 2) * std::pow(d[9], 2) - 2 * std::pow(d[2], 2) * d[3] * d[9] * d[10] +
        d[0] * std::pow(d[2], 2) * std::pow(d[10], 2);
    coeffs[19] =
        -3 * std::pow(d[3], 2) * d[6] * std::pow(d[9], 2) * d[11] +
        3 * std::pow(d[4], 2) * d[6] * std::pow(d[9], 2) * d[11] - 6 * d[3] * d[4] * d[7] * std::pow(d[9], 2) * d[11] -
        3 * d[6] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[11] + 4 * d[0] * d[3] * d[6] * d[9] * d[10] * d[11] -
        4 * d[1] * d[4] * d[6] * d[9] * d[10] * d[11] + 4 * d[1] * d[3] * d[7] * d[9] * d[10] * d[11] +
        4 * d[0] * d[4] * d[7] * d[9] * d[10] * d[11] - std::pow(d[0], 2) * d[6] * std::pow(d[10], 2) * d[11] +
        std::pow(d[1], 2) * d[6] * std::pow(d[10], 2) * d[11] - 2 * d[0] * d[1] * d[7] * std::pow(d[10], 2) * d[11] -
        d[6] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[11] + 2 * d[0] * d[3] * d[6] * std::pow(d[9], 2) * d[12] -
        2 * d[1] * d[4] * d[6] * std::pow(d[9], 2) * d[12] + 2 * d[1] * d[3] * d[7] * std::pow(d[9], 2) * d[12] +
        2 * d[0] * d[4] * d[7] * std::pow(d[9], 2) * d[12] - 2 * std::pow(d[0], 2) * d[6] * d[9] * d[10] * d[12] +
        2 * std::pow(d[1], 2) * d[6] * d[9] * d[10] * d[12] - 4 * d[0] * d[1] * d[7] * d[9] * d[10] * d[12] -
        2 * d[6] * std::pow(d[8], 2) * d[9] * d[10] * d[12] + 2 * d[0] * std::pow(d[3], 2) * d[9] * d[11] +
        4 * d[1] * d[3] * d[4] * d[9] * d[11] - 2 * d[0] * std::pow(d[4], 2) * d[9] * d[11] +
        4 * d[2] * d[6] * d[8] * d[9] * d[11] + 2 * d[0] * std::pow(d[8], 2) * d[9] * d[11] -
        2 * std::pow(d[0], 2) * d[3] * d[10] * d[11] - 2 * std::pow(d[1], 2) * d[3] * d[10] * d[11] +
        2 * d[3] * std::pow(d[8], 2) * d[10] * d[11] - 2 * std::pow(d[0], 2) * d[3] * d[9] * d[12] -
        2 * std::pow(d[1], 2) * d[3] * d[9] * d[12] + 2 * d[3] * std::pow(d[8], 2) * d[9] * d[12] +
        2 * std::pow(d[0], 3) * d[10] * d[12] + 2 * d[0] * std::pow(d[1], 2) * d[10] * d[12] +
        4 * d[2] * d[6] * d[8] * d[10] * d[12] - 2 * d[0] * std::pow(d[8], 2) * d[10] * d[12] -
        std::pow(d[2], 2) * d[6] * d[11] + std::pow(d[5], 2) * d[6] * d[11] - 2 * d[0] * d[2] * d[8] * d[11] -
        2 * d[3] * d[5] * d[8] * d[11] - 2 * d[2] * d[5] * d[6] * d[12] - 2 * d[2] * d[3] * d[8] * d[12] +
        2 * d[0] * d[5] * d[8] * d[12];
    coeffs[20] = -3 * std::pow(d[6], 3) * d[9] * std::pow(d[11], 2) -
                 3 * d[6] * std::pow(d[7], 2) * d[9] * std::pow(d[11], 2) -
                 2 * std::pow(d[6], 3) * d[10] * d[11] * d[12] - 2 * d[6] * std::pow(d[7], 2) * d[10] * d[11] * d[12] -
                 std::pow(d[6], 3) * d[9] * std::pow(d[12], 2) - d[6] * std::pow(d[7], 2) * d[9] * std::pow(d[12], 2) +
                 3 * d[0] * std::pow(d[6], 2) * std::pow(d[11], 2) + 2 * d[1] * d[6] * d[7] * std::pow(d[11], 2) +
                 d[0] * std::pow(d[7], 2) * std::pow(d[11], 2) + 2 * d[3] * std::pow(d[6], 2) * d[11] * d[12] +
                 2 * d[3] * std::pow(d[7], 2) * d[11] * d[12] + d[0] * std::pow(d[6], 2) * std::pow(d[12], 2) +
                 2 * d[1] * d[6] * d[7] * std::pow(d[12], 2) - d[0] * std::pow(d[7], 2) * std::pow(d[12], 2);
    coeffs[21] =
        -std::pow(d[3], 2) * d[6] * std::pow(d[9], 3) + std::pow(d[4], 2) * d[6] * std::pow(d[9], 3) -
        2 * d[3] * d[4] * d[7] * std::pow(d[9], 3) - d[6] * std::pow(d[8], 2) * std::pow(d[9], 3) +
        2 * d[0] * d[3] * d[6] * std::pow(d[9], 2) * d[10] - 2 * d[1] * d[4] * d[6] * std::pow(d[9], 2) * d[10] +
        2 * d[1] * d[3] * d[7] * std::pow(d[9], 2) * d[10] + 2 * d[0] * d[4] * d[7] * std::pow(d[9], 2) * d[10] -
        std::pow(d[0], 2) * d[6] * d[9] * std::pow(d[10], 2) + std::pow(d[1], 2) * d[6] * d[9] * std::pow(d[10], 2) -
        2 * d[0] * d[1] * d[7] * d[9] * std::pow(d[10], 2) - d[6] * std::pow(d[8], 2) * d[9] * std::pow(d[10], 2) +
        d[0] * std::pow(d[3], 2) * std::pow(d[9], 2) + 2 * d[1] * d[3] * d[4] * std::pow(d[9], 2) -
        d[0] * std::pow(d[4], 2) * std::pow(d[9], 2) + 2 * d[2] * d[6] * d[8] * std::pow(d[9], 2) +
        d[0] * std::pow(d[8], 2) * std::pow(d[9], 2) - 2 * std::pow(d[0], 2) * d[3] * d[9] * d[10] -
        2 * std::pow(d[1], 2) * d[3] * d[9] * d[10] + 2 * d[3] * std::pow(d[8], 2) * d[9] * d[10] +
        std::pow(d[0], 3) * std::pow(d[10], 2) + d[0] * std::pow(d[1], 2) * std::pow(d[10], 2) +
        2 * d[2] * d[6] * d[8] * std::pow(d[10], 2) - d[0] * std::pow(d[8], 2) * std::pow(d[10], 2) -
        std::pow(d[2], 2) * d[6] * d[9] + std::pow(d[5], 2) * d[6] * d[9] - 2 * d[0] * d[2] * d[8] * d[9] -
        2 * d[3] * d[5] * d[8] * d[9] - 2 * d[2] * d[5] * d[6] * d[10] - 2 * d[2] * d[3] * d[8] * d[10] +
        2 * d[0] * d[5] * d[8] * d[10] + d[0] * std::pow(d[2], 2) + 2 * d[2] * d[3] * d[5] - d[0] * std::pow(d[5], 2);
    coeffs[22] =
        -3 * std::pow(d[6], 3) * std::pow(d[9], 2) * d[11] - 3 * d[6] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[11] -
        std::pow(d[6], 3) * std::pow(d[10], 2) * d[11] - d[6] * std::pow(d[7], 2) * std::pow(d[10], 2) * d[11] -
        2 * std::pow(d[6], 3) * d[9] * d[10] * d[12] - 2 * d[6] * std::pow(d[7], 2) * d[9] * d[10] * d[12] +
        6 * d[0] * std::pow(d[6], 2) * d[9] * d[11] + 4 * d[1] * d[6] * d[7] * d[9] * d[11] +
        2 * d[0] * std::pow(d[7], 2) * d[9] * d[11] + 2 * d[3] * std::pow(d[6], 2) * d[10] * d[11] +
        2 * d[3] * std::pow(d[7], 2) * d[10] * d[11] + 2 * d[3] * std::pow(d[6], 2) * d[9] * d[12] +
        2 * d[3] * std::pow(d[7], 2) * d[9] * d[12] + 2 * d[0] * std::pow(d[6], 2) * d[10] * d[12] +
        4 * d[1] * d[6] * d[7] * d[10] * d[12] - 2 * d[0] * std::pow(d[7], 2) * d[10] * d[12] -
        3 * std::pow(d[0], 2) * d[6] * d[11] - std::pow(d[1], 2) * d[6] * d[11] - std::pow(d[3], 2) * d[6] * d[11] +
        std::pow(d[4], 2) * d[6] * d[11] - 2 * d[0] * d[1] * d[7] * d[11] - 2 * d[3] * d[4] * d[7] * d[11] -
        2 * d[0] * d[3] * d[6] * d[12] - 2 * d[1] * d[4] * d[6] * d[12] - 2 * d[1] * d[3] * d[7] * d[12] +
        2 * d[0] * d[4] * d[7] * d[12];
    coeffs[23] = -std::pow(d[6], 3) * std::pow(d[9], 3) - d[6] * std::pow(d[7], 2) * std::pow(d[9], 3) -
                 std::pow(d[6], 3) * d[9] * std::pow(d[10], 2) - d[6] * std::pow(d[7], 2) * d[9] * std::pow(d[10], 2) +
                 3 * d[0] * std::pow(d[6], 2) * std::pow(d[9], 2) + 2 * d[1] * d[6] * d[7] * std::pow(d[9], 2) +
                 d[0] * std::pow(d[7], 2) * std::pow(d[9], 2) + 2 * d[3] * std::pow(d[6], 2) * d[9] * d[10] +
                 2 * d[3] * std::pow(d[7], 2) * d[9] * d[10] + d[0] * std::pow(d[6], 2) * std::pow(d[10], 2) +
                 2 * d[1] * d[6] * d[7] * std::pow(d[10], 2) - d[0] * std::pow(d[7], 2) * std::pow(d[10], 2) -
                 3 * std::pow(d[0], 2) * d[6] * d[9] - std::pow(d[1], 2) * d[6] * d[9] -
                 std::pow(d[3], 2) * d[6] * d[9] + std::pow(d[4], 2) * d[6] * d[9] - 2 * d[0] * d[1] * d[7] * d[9] -
                 2 * d[3] * d[4] * d[7] * d[9] - 2 * d[0] * d[3] * d[6] * d[10] - 2 * d[1] * d[4] * d[6] * d[10] -
                 2 * d[1] * d[3] * d[7] * d[10] + 2 * d[0] * d[4] * d[7] * d[10] + std::pow(d[0], 3) +
                 d[0] * std::pow(d[1], 2) + d[0] * std::pow(d[3], 2) + 2 * d[1] * d[3] * d[4] -
                 d[0] * std::pow(d[4], 2);
    coeffs[24] = d[3] * std::pow(d[5], 2) * std::pow(d[11], 3) - 2 * d[2] * d[3] * d[5] * std::pow(d[11], 2) * d[12] -
                 d[0] * std::pow(d[5], 2) * std::pow(d[11], 2) * d[12] +
                 std::pow(d[2], 2) * d[3] * d[11] * std::pow(d[12], 2) +
                 2 * d[0] * d[2] * d[5] * d[11] * std::pow(d[12], 2) - d[0] * std::pow(d[2], 2) * std::pow(d[12], 3);
    coeffs[25] =
        3 * d[3] * std::pow(d[5], 2) * d[9] * std::pow(d[11], 2) - 2 * d[2] * d[3] * d[5] * d[10] * std::pow(d[11], 2) -
        d[0] * std::pow(d[5], 2) * d[10] * std::pow(d[11], 2) - 4 * d[2] * d[3] * d[5] * d[9] * d[11] * d[12] -
        2 * d[0] * std::pow(d[5], 2) * d[9] * d[11] * d[12] + 2 * std::pow(d[2], 2) * d[3] * d[10] * d[11] * d[12] +
        4 * d[0] * d[2] * d[5] * d[10] * d[11] * d[12] + std::pow(d[2], 2) * d[3] * d[9] * std::pow(d[12], 2) +
        2 * d[0] * d[2] * d[5] * d[9] * std::pow(d[12], 2) - 3 * d[0] * std::pow(d[2], 2) * d[10] * std::pow(d[12], 2);
    coeffs[26] =
        std::pow(d[3], 3) * std::pow(d[11], 3) + d[3] * std::pow(d[4], 2) * std::pow(d[11], 3) +
        2 * d[5] * d[6] * d[8] * std::pow(d[11], 3) - d[3] * std::pow(d[8], 2) * std::pow(d[11], 3) -
        3 * d[0] * std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] -
        2 * d[1] * d[3] * d[4] * std::pow(d[11], 2) * d[12] - d[0] * std::pow(d[4], 2) * std::pow(d[11], 2) * d[12] -
        2 * d[2] * d[6] * d[8] * std::pow(d[11], 2) * d[12] + d[0] * std::pow(d[8], 2) * std::pow(d[11], 2) * d[12] +
        3 * std::pow(d[0], 2) * d[3] * d[11] * std::pow(d[12], 2) +
        std::pow(d[1], 2) * d[3] * d[11] * std::pow(d[12], 2) + 2 * d[0] * d[1] * d[4] * d[11] * std::pow(d[12], 2) +
        2 * d[5] * d[6] * d[8] * d[11] * std::pow(d[12], 2) - d[3] * std::pow(d[8], 2) * d[11] * std::pow(d[12], 2) -
        std::pow(d[0], 3) * std::pow(d[12], 3) - d[0] * std::pow(d[1], 2) * std::pow(d[12], 3) -
        2 * d[2] * d[6] * d[8] * std::pow(d[12], 3) + d[0] * std::pow(d[8], 2) * std::pow(d[12], 3);
    coeffs[27] =
        3 * d[3] * std::pow(d[5], 2) * std::pow(d[9], 2) * d[11] - 4 * d[2] * d[3] * d[5] * d[9] * d[10] * d[11] -
        2 * d[0] * std::pow(d[5], 2) * d[9] * d[10] * d[11] + std::pow(d[2], 2) * d[3] * std::pow(d[10], 2) * d[11] +
        2 * d[0] * d[2] * d[5] * std::pow(d[10], 2) * d[11] - 2 * d[2] * d[3] * d[5] * std::pow(d[9], 2) * d[12] -
        d[0] * std::pow(d[5], 2) * std::pow(d[9], 2) * d[12] + 2 * std::pow(d[2], 2) * d[3] * d[9] * d[10] * d[12] +
        4 * d[0] * d[2] * d[5] * d[9] * d[10] * d[12] - 3 * d[0] * std::pow(d[2], 2) * std::pow(d[10], 2) * d[12];
    coeffs[28] =
        3 * std::pow(d[3], 3) * d[9] * std::pow(d[11], 2) + 3 * d[3] * std::pow(d[4], 2) * d[9] * std::pow(d[11], 2) +
        6 * d[5] * d[6] * d[8] * d[9] * std::pow(d[11], 2) - 3 * d[3] * std::pow(d[8], 2) * d[9] * std::pow(d[11], 2) -
        3 * d[0] * std::pow(d[3], 2) * d[10] * std::pow(d[11], 2) -
        2 * d[1] * d[3] * d[4] * d[10] * std::pow(d[11], 2) - d[0] * std::pow(d[4], 2) * d[10] * std::pow(d[11], 2) -
        2 * d[2] * d[6] * d[8] * d[10] * std::pow(d[11], 2) + d[0] * std::pow(d[8], 2) * d[10] * std::pow(d[11], 2) -
        6 * d[0] * std::pow(d[3], 2) * d[9] * d[11] * d[12] - 4 * d[1] * d[3] * d[4] * d[9] * d[11] * d[12] -
        2 * d[0] * std::pow(d[4], 2) * d[9] * d[11] * d[12] - 4 * d[2] * d[6] * d[8] * d[9] * d[11] * d[12] +
        2 * d[0] * std::pow(d[8], 2) * d[9] * d[11] * d[12] + 6 * std::pow(d[0], 2) * d[3] * d[10] * d[11] * d[12] +
        2 * std::pow(d[1], 2) * d[3] * d[10] * d[11] * d[12] + 4 * d[0] * d[1] * d[4] * d[10] * d[11] * d[12] +
        4 * d[5] * d[6] * d[8] * d[10] * d[11] * d[12] - 2 * d[3] * std::pow(d[8], 2) * d[10] * d[11] * d[12] +
        3 * std::pow(d[0], 2) * d[3] * d[9] * std::pow(d[12], 2) +
        std::pow(d[1], 2) * d[3] * d[9] * std::pow(d[12], 2) + 2 * d[0] * d[1] * d[4] * d[9] * std::pow(d[12], 2) +
        2 * d[5] * d[6] * d[8] * d[9] * std::pow(d[12], 2) - d[3] * std::pow(d[8], 2) * d[9] * std::pow(d[12], 2) -
        3 * std::pow(d[0], 3) * d[10] * std::pow(d[12], 2) - 3 * d[0] * std::pow(d[1], 2) * d[10] * std::pow(d[12], 2) -
        6 * d[2] * d[6] * d[8] * d[10] * std::pow(d[12], 2) +
        3 * d[0] * std::pow(d[8], 2) * d[10] * std::pow(d[12], 2) - 2 * d[2] * d[5] * d[6] * std::pow(d[11], 2) +
        2 * d[2] * d[3] * d[8] * std::pow(d[11], 2) - 2 * d[0] * d[5] * d[8] * std::pow(d[11], 2) +
        2 * std::pow(d[2], 2) * d[6] * d[11] * d[12] - 2 * std::pow(d[5], 2) * d[6] * d[11] * d[12] +
        2 * d[2] * d[5] * d[6] * std::pow(d[12], 2) + 2 * d[2] * d[3] * d[8] * std::pow(d[12], 2) -
        2 * d[0] * d[5] * d[8] * std::pow(d[12], 2);
    coeffs[29] =
        d[3] * std::pow(d[6], 2) * std::pow(d[11], 3) + 2 * d[4] * d[6] * d[7] * std::pow(d[11], 3) -
        d[3] * std::pow(d[7], 2) * std::pow(d[11], 3) - d[0] * std::pow(d[6], 2) * std::pow(d[11], 2) * d[12] -
        2 * d[1] * d[6] * d[7] * std::pow(d[11], 2) * d[12] + d[0] * std::pow(d[7], 2) * std::pow(d[11], 2) * d[12] +
        d[3] * std::pow(d[6], 2) * d[11] * std::pow(d[12], 2) + 2 * d[4] * d[6] * d[7] * d[11] * std::pow(d[12], 2) -
        d[3] * std::pow(d[7], 2) * d[11] * std::pow(d[12], 2) - d[0] * std::pow(d[6], 2) * std::pow(d[12], 3) -
        2 * d[1] * d[6] * d[7] * std::pow(d[12], 3) + d[0] * std::pow(d[7], 2) * std::pow(d[12], 3);
    coeffs[30] = d[3] * std::pow(d[5], 2) * std::pow(d[9], 3) - 2 * d[2] * d[3] * d[5] * std::pow(d[9], 2) * d[10] -
                 d[0] * std::pow(d[5], 2) * std::pow(d[9], 2) * d[10] +
                 std::pow(d[2], 2) * d[3] * d[9] * std::pow(d[10], 2) +
                 2 * d[0] * d[2] * d[5] * d[9] * std::pow(d[10], 2) - d[0] * std::pow(d[2], 2) * std::pow(d[10], 3);
    coeffs[31] =
        3 * std::pow(d[3], 3) * std::pow(d[9], 2) * d[11] + 3 * d[3] * std::pow(d[4], 2) * std::pow(d[9], 2) * d[11] +
        6 * d[5] * d[6] * d[8] * std::pow(d[9], 2) * d[11] - 3 * d[3] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[11] -
        6 * d[0] * std::pow(d[3], 2) * d[9] * d[10] * d[11] - 4 * d[1] * d[3] * d[4] * d[9] * d[10] * d[11] -
        2 * d[0] * std::pow(d[4], 2) * d[9] * d[10] * d[11] - 4 * d[2] * d[6] * d[8] * d[9] * d[10] * d[11] +
        2 * d[0] * std::pow(d[8], 2) * d[9] * d[10] * d[11] +
        3 * std::pow(d[0], 2) * d[3] * std::pow(d[10], 2) * d[11] +
        std::pow(d[1], 2) * d[3] * std::pow(d[10], 2) * d[11] + 2 * d[0] * d[1] * d[4] * std::pow(d[10], 2) * d[11] +
        2 * d[5] * d[6] * d[8] * std::pow(d[10], 2) * d[11] - d[3] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[11] -
        3 * d[0] * std::pow(d[3], 2) * std::pow(d[9], 2) * d[12] - 2 * d[1] * d[3] * d[4] * std::pow(d[9], 2) * d[12] -
        d[0] * std::pow(d[4], 2) * std::pow(d[9], 2) * d[12] - 2 * d[2] * d[6] * d[8] * std::pow(d[9], 2) * d[12] +
        d[0] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[12] + 6 * std::pow(d[0], 2) * d[3] * d[9] * d[10] * d[12] +
        2 * std::pow(d[1], 2) * d[3] * d[9] * d[10] * d[12] + 4 * d[0] * d[1] * d[4] * d[9] * d[10] * d[12] +
        4 * d[5] * d[6] * d[8] * d[9] * d[10] * d[12] - 2 * d[3] * std::pow(d[8], 2) * d[9] * d[10] * d[12] -
        3 * std::pow(d[0], 3) * std::pow(d[10], 2) * d[12] - 3 * d[0] * std::pow(d[1], 2) * std::pow(d[10], 2) * d[12] -
        6 * d[2] * d[6] * d[8] * std::pow(d[10], 2) * d[12] +
        3 * d[0] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[12] - 4 * d[2] * d[5] * d[6] * d[9] * d[11] +
        4 * d[2] * d[3] * d[8] * d[9] * d[11] - 4 * d[0] * d[5] * d[8] * d[9] * d[11] +
        2 * std::pow(d[2], 2) * d[6] * d[10] * d[11] - 2 * std::pow(d[5], 2) * d[6] * d[10] * d[11] +
        2 * std::pow(d[2], 2) * d[6] * d[9] * d[12] - 2 * std::pow(d[5], 2) * d[6] * d[9] * d[12] +
        4 * d[2] * d[5] * d[6] * d[10] * d[12] + 4 * d[2] * d[3] * d[8] * d[10] * d[12] -
        4 * d[0] * d[5] * d[8] * d[10] * d[12] - std::pow(d[2], 2) * d[3] * d[11] + 2 * d[0] * d[2] * d[5] * d[11] +
        d[3] * std::pow(d[5], 2) * d[11] - d[0] * std::pow(d[2], 2) * d[12] - 2 * d[2] * d[3] * d[5] * d[12] +
        d[0] * std::pow(d[5], 2) * d[12];
    coeffs[32] =
        3 * d[3] * std::pow(d[6], 2) * d[9] * std::pow(d[11], 2) + 6 * d[4] * d[6] * d[7] * d[9] * std::pow(d[11], 2) -
        3 * d[3] * std::pow(d[7], 2) * d[9] * std::pow(d[11], 2) -
        d[0] * std::pow(d[6], 2) * d[10] * std::pow(d[11], 2) - 2 * d[1] * d[6] * d[7] * d[10] * std::pow(d[11], 2) +
        d[0] * std::pow(d[7], 2) * d[10] * std::pow(d[11], 2) - 2 * d[0] * std::pow(d[6], 2) * d[9] * d[11] * d[12] -
        4 * d[1] * d[6] * d[7] * d[9] * d[11] * d[12] + 2 * d[0] * std::pow(d[7], 2) * d[9] * d[11] * d[12] +
        2 * d[3] * std::pow(d[6], 2) * d[10] * d[11] * d[12] + 4 * d[4] * d[6] * d[7] * d[10] * d[11] * d[12] -
        2 * d[3] * std::pow(d[7], 2) * d[10] * d[11] * d[12] + d[3] * std::pow(d[6], 2) * d[9] * std::pow(d[12], 2) +
        2 * d[4] * d[6] * d[7] * d[9] * std::pow(d[12], 2) - d[3] * std::pow(d[7], 2) * d[9] * std::pow(d[12], 2) -
        3 * d[0] * std::pow(d[6], 2) * d[10] * std::pow(d[12], 2) -
        6 * d[1] * d[6] * d[7] * d[10] * std::pow(d[12], 2) +
        3 * d[0] * std::pow(d[7], 2) * d[10] * std::pow(d[12], 2) - 2 * d[0] * d[3] * d[6] * std::pow(d[11], 2) -
        2 * d[1] * d[4] * d[6] * std::pow(d[11], 2) + 2 * d[1] * d[3] * d[7] * std::pow(d[11], 2) -
        2 * d[0] * d[4] * d[7] * std::pow(d[11], 2) + 2 * std::pow(d[0], 2) * d[6] * d[11] * d[12] +
        2 * std::pow(d[1], 2) * d[6] * d[11] * d[12] - 2 * std::pow(d[3], 2) * d[6] * d[11] * d[12] -
        2 * std::pow(d[4], 2) * d[6] * d[11] * d[12] + 2 * d[0] * d[3] * d[6] * std::pow(d[12], 2) +
        2 * d[1] * d[4] * d[6] * std::pow(d[12], 2) + 2 * d[1] * d[3] * d[7] * std::pow(d[12], 2) -
        2 * d[0] * d[4] * d[7] * std::pow(d[12], 2);
    coeffs[33] =
        std::pow(d[3], 3) * std::pow(d[9], 3) + d[3] * std::pow(d[4], 2) * std::pow(d[9], 3) +
        2 * d[5] * d[6] * d[8] * std::pow(d[9], 3) - d[3] * std::pow(d[8], 2) * std::pow(d[9], 3) -
        3 * d[0] * std::pow(d[3], 2) * std::pow(d[9], 2) * d[10] - 2 * d[1] * d[3] * d[4] * std::pow(d[9], 2) * d[10] -
        d[0] * std::pow(d[4], 2) * std::pow(d[9], 2) * d[10] - 2 * d[2] * d[6] * d[8] * std::pow(d[9], 2) * d[10] +
        d[0] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[10] +
        3 * std::pow(d[0], 2) * d[3] * d[9] * std::pow(d[10], 2) +
        std::pow(d[1], 2) * d[3] * d[9] * std::pow(d[10], 2) + 2 * d[0] * d[1] * d[4] * d[9] * std::pow(d[10], 2) +
        2 * d[5] * d[6] * d[8] * d[9] * std::pow(d[10], 2) - d[3] * std::pow(d[8], 2) * d[9] * std::pow(d[10], 2) -
        std::pow(d[0], 3) * std::pow(d[10], 3) - d[0] * std::pow(d[1], 2) * std::pow(d[10], 3) -
        2 * d[2] * d[6] * d[8] * std::pow(d[10], 3) + d[0] * std::pow(d[8], 2) * std::pow(d[10], 3) -
        2 * d[2] * d[5] * d[6] * std::pow(d[9], 2) + 2 * d[2] * d[3] * d[8] * std::pow(d[9], 2) -
        2 * d[0] * d[5] * d[8] * std::pow(d[9], 2) + 2 * std::pow(d[2], 2) * d[6] * d[9] * d[10] -
        2 * std::pow(d[5], 2) * d[6] * d[9] * d[10] + 2 * d[2] * d[5] * d[6] * std::pow(d[10], 2) +
        2 * d[2] * d[3] * d[8] * std::pow(d[10], 2) - 2 * d[0] * d[5] * d[8] * std::pow(d[10], 2) -
        std::pow(d[2], 2) * d[3] * d[9] + 2 * d[0] * d[2] * d[5] * d[9] + d[3] * std::pow(d[5], 2) * d[9] -
        d[0] * std::pow(d[2], 2) * d[10] - 2 * d[2] * d[3] * d[5] * d[10] + d[0] * std::pow(d[5], 2) * d[10];
    coeffs[34] =
        3 * d[3] * std::pow(d[6], 2) * std::pow(d[9], 2) * d[11] + 6 * d[4] * d[6] * d[7] * std::pow(d[9], 2) * d[11] -
        3 * d[3] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[11] - 2 * d[0] * std::pow(d[6], 2) * d[9] * d[10] * d[11] -
        4 * d[1] * d[6] * d[7] * d[9] * d[10] * d[11] + 2 * d[0] * std::pow(d[7], 2) * d[9] * d[10] * d[11] +
        d[3] * std::pow(d[6], 2) * std::pow(d[10], 2) * d[11] + 2 * d[4] * d[6] * d[7] * std::pow(d[10], 2) * d[11] -
        d[3] * std::pow(d[7], 2) * std::pow(d[10], 2) * d[11] - d[0] * std::pow(d[6], 2) * std::pow(d[9], 2) * d[12] -
        2 * d[1] * d[6] * d[7] * std::pow(d[9], 2) * d[12] + d[0] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[12] +
        2 * d[3] * std::pow(d[6], 2) * d[9] * d[10] * d[12] + 4 * d[4] * d[6] * d[7] * d[9] * d[10] * d[12] -
        2 * d[3] * std::pow(d[7], 2) * d[9] * d[10] * d[12] -
        3 * d[0] * std::pow(d[6], 2) * std::pow(d[10], 2) * d[12] -
        6 * d[1] * d[6] * d[7] * std::pow(d[10], 2) * d[12] +
        3 * d[0] * std::pow(d[7], 2) * std::pow(d[10], 2) * d[12] - 4 * d[0] * d[3] * d[6] * d[9] * d[11] -
        4 * d[1] * d[4] * d[6] * d[9] * d[11] + 4 * d[1] * d[3] * d[7] * d[9] * d[11] -
        4 * d[0] * d[4] * d[7] * d[9] * d[11] + 2 * std::pow(d[0], 2) * d[6] * d[10] * d[11] +
        2 * std::pow(d[1], 2) * d[6] * d[10] * d[11] - 2 * std::pow(d[3], 2) * d[6] * d[10] * d[11] -
        2 * std::pow(d[4], 2) * d[6] * d[10] * d[11] + 2 * std::pow(d[0], 2) * d[6] * d[9] * d[12] +
        2 * std::pow(d[1], 2) * d[6] * d[9] * d[12] - 2 * std::pow(d[3], 2) * d[6] * d[9] * d[12] -
        2 * std::pow(d[4], 2) * d[6] * d[9] * d[12] + 4 * d[0] * d[3] * d[6] * d[10] * d[12] +
        4 * d[1] * d[4] * d[6] * d[10] * d[12] + 4 * d[1] * d[3] * d[7] * d[10] * d[12] -
        4 * d[0] * d[4] * d[7] * d[10] * d[12] + std::pow(d[0], 2) * d[3] * d[11] - std::pow(d[1], 2) * d[3] * d[11] +
        std::pow(d[3], 3) * d[11] + 2 * d[0] * d[1] * d[4] * d[11] + d[3] * std::pow(d[4], 2) * d[11] -
        std::pow(d[0], 3) * d[12] - d[0] * std::pow(d[1], 2) * d[12] - d[0] * std::pow(d[3], 2) * d[12] -
        2 * d[1] * d[3] * d[4] * d[12] + d[0] * std::pow(d[4], 2) * d[12];
    coeffs[35] =
        d[3] * std::pow(d[6], 2) * std::pow(d[9], 3) + 2 * d[4] * d[6] * d[7] * std::pow(d[9], 3) -
        d[3] * std::pow(d[7], 2) * std::pow(d[9], 3) - d[0] * std::pow(d[6], 2) * std::pow(d[9], 2) * d[10] -
        2 * d[1] * d[6] * d[7] * std::pow(d[9], 2) * d[10] + d[0] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[10] +
        d[3] * std::pow(d[6], 2) * d[9] * std::pow(d[10], 2) + 2 * d[4] * d[6] * d[7] * d[9] * std::pow(d[10], 2) -
        d[3] * std::pow(d[7], 2) * d[9] * std::pow(d[10], 2) - d[0] * std::pow(d[6], 2) * std::pow(d[10], 3) -
        2 * d[1] * d[6] * d[7] * std::pow(d[10], 3) + d[0] * std::pow(d[7], 2) * std::pow(d[10], 3) -
        2 * d[0] * d[3] * d[6] * std::pow(d[9], 2) - 2 * d[1] * d[4] * d[6] * std::pow(d[9], 2) +
        2 * d[1] * d[3] * d[7] * std::pow(d[9], 2) - 2 * d[0] * d[4] * d[7] * std::pow(d[9], 2) +
        2 * std::pow(d[0], 2) * d[6] * d[9] * d[10] + 2 * std::pow(d[1], 2) * d[6] * d[9] * d[10] -
        2 * std::pow(d[3], 2) * d[6] * d[9] * d[10] - 2 * std::pow(d[4], 2) * d[6] * d[9] * d[10] +
        2 * d[0] * d[3] * d[6] * std::pow(d[10], 2) + 2 * d[1] * d[4] * d[6] * std::pow(d[10], 2) +
        2 * d[1] * d[3] * d[7] * std::pow(d[10], 2) - 2 * d[0] * d[4] * d[7] * std::pow(d[10], 2) +
        std::pow(d[0], 2) * d[3] * d[9] - std::pow(d[1], 2) * d[3] * d[9] + std::pow(d[3], 3) * d[9] +
        2 * d[0] * d[1] * d[4] * d[9] + d[3] * std::pow(d[4], 2) * d[9] - std::pow(d[0], 3) * d[10] -
        d[0] * std::pow(d[1], 2) * d[10] - d[0] * std::pow(d[3], 2) * d[10] - 2 * d[1] * d[3] * d[4] * d[10] +
        d[0] * std::pow(d[4], 2) * d[10];
    coeffs[36] =
        -std::pow(d[5], 2) * d[7] * std::pow(d[11], 2) * d[12] + 2 * d[4] * d[5] * d[8] * std::pow(d[11], 2) * d[12] +
        2 * d[2] * d[5] * d[7] * d[11] * std::pow(d[12], 2) - 2 * d[2] * d[4] * d[8] * d[11] * std::pow(d[12], 2) -
        2 * d[1] * d[5] * d[8] * d[11] * std::pow(d[12], 2) - std::pow(d[2], 2) * d[7] * std::pow(d[12], 3) +
        2 * d[1] * d[2] * d[8] * std::pow(d[12], 3);
    coeffs[37] =
        -std::pow(d[5], 2) * d[7] * d[10] * std::pow(d[11], 2) + 2 * d[4] * d[5] * d[8] * d[10] * std::pow(d[11], 2) -
        2 * std::pow(d[5], 2) * d[7] * d[9] * d[11] * d[12] + 4 * d[4] * d[5] * d[8] * d[9] * d[11] * d[12] +
        4 * d[2] * d[5] * d[7] * d[10] * d[11] * d[12] - 4 * d[2] * d[4] * d[8] * d[10] * d[11] * d[12] -
        4 * d[1] * d[5] * d[8] * d[10] * d[11] * d[12] + 2 * d[2] * d[5] * d[7] * d[9] * std::pow(d[12], 2) -
        2 * d[2] * d[4] * d[8] * d[9] * std::pow(d[12], 2) - 2 * d[1] * d[5] * d[8] * d[9] * std::pow(d[12], 2) -
        3 * std::pow(d[2], 2) * d[7] * d[10] * std::pow(d[12], 2) +
        6 * d[1] * d[2] * d[8] * d[10] * std::pow(d[12], 2) - d[4] * std::pow(d[5], 2) * std::pow(d[11], 2) +
        2 * d[1] * std::pow(d[5], 2) * d[11] * d[12] + std::pow(d[2], 2) * d[4] * std::pow(d[12], 2) -
        2 * d[1] * d[2] * d[5] * std::pow(d[12], 2);
    coeffs[38] =
        2 * d[3] * d[4] * d[6] * std::pow(d[11], 2) * d[12] - std::pow(d[3], 2) * d[7] * std::pow(d[11], 2) * d[12] +
        std::pow(d[4], 2) * d[7] * std::pow(d[11], 2) * d[12] + d[7] * std::pow(d[8], 2) * std::pow(d[11], 2) * d[12] -
        2 * d[1] * d[3] * d[6] * d[11] * std::pow(d[12], 2) - 2 * d[0] * d[4] * d[6] * d[11] * std::pow(d[12], 2) +
        2 * d[0] * d[3] * d[7] * d[11] * std::pow(d[12], 2) - 2 * d[1] * d[4] * d[7] * d[11] * std::pow(d[12], 2) +
        2 * d[0] * d[1] * d[6] * std::pow(d[12], 3) - std::pow(d[0], 2) * d[7] * std::pow(d[12], 3) +
        std::pow(d[1], 2) * d[7] * std::pow(d[12], 3) + d[7] * std::pow(d[8], 2) * std::pow(d[12], 3);
    coeffs[39] =
        -2 * std::pow(d[5], 2) * d[7] * d[9] * d[10] * d[11] + 4 * d[4] * d[5] * d[8] * d[9] * d[10] * d[11] +
        2 * d[2] * d[5] * d[7] * std::pow(d[10], 2) * d[11] - 2 * d[2] * d[4] * d[8] * std::pow(d[10], 2) * d[11] -
        2 * d[1] * d[5] * d[8] * std::pow(d[10], 2) * d[11] - std::pow(d[5], 2) * d[7] * std::pow(d[9], 2) * d[12] +
        2 * d[4] * d[5] * d[8] * std::pow(d[9], 2) * d[12] + 4 * d[2] * d[5] * d[7] * d[9] * d[10] * d[12] -
        4 * d[2] * d[4] * d[8] * d[9] * d[10] * d[12] - 4 * d[1] * d[5] * d[8] * d[9] * d[10] * d[12] -
        3 * std::pow(d[2], 2) * d[7] * std::pow(d[10], 2) * d[12] +
        6 * d[1] * d[2] * d[8] * std::pow(d[10], 2) * d[12] - 2 * d[4] * std::pow(d[5], 2) * d[9] * d[11] +
        2 * d[1] * std::pow(d[5], 2) * d[10] * d[11] + 2 * d[1] * std::pow(d[5], 2) * d[9] * d[12] +
        2 * std::pow(d[2], 2) * d[4] * d[10] * d[12] - 4 * d[1] * d[2] * d[5] * d[10] * d[12];
    coeffs[40] =
        2 * d[3] * d[4] * d[6] * d[10] * std::pow(d[11], 2) - std::pow(d[3], 2) * d[7] * d[10] * std::pow(d[11], 2) +
        std::pow(d[4], 2) * d[7] * d[10] * std::pow(d[11], 2) + d[7] * std::pow(d[8], 2) * d[10] * std::pow(d[11], 2) +
        4 * d[3] * d[4] * d[6] * d[9] * d[11] * d[12] - 2 * std::pow(d[3], 2) * d[7] * d[9] * d[11] * d[12] +
        2 * std::pow(d[4], 2) * d[7] * d[9] * d[11] * d[12] + 2 * d[7] * std::pow(d[8], 2) * d[9] * d[11] * d[12] -
        4 * d[1] * d[3] * d[6] * d[10] * d[11] * d[12] - 4 * d[0] * d[4] * d[6] * d[10] * d[11] * d[12] +
        4 * d[0] * d[3] * d[7] * d[10] * d[11] * d[12] - 4 * d[1] * d[4] * d[7] * d[10] * d[11] * d[12] -
        2 * d[1] * d[3] * d[6] * d[9] * std::pow(d[12], 2) - 2 * d[0] * d[4] * d[6] * d[9] * std::pow(d[12], 2) +
        2 * d[0] * d[3] * d[7] * d[9] * std::pow(d[12], 2) - 2 * d[1] * d[4] * d[7] * d[9] * std::pow(d[12], 2) +
        6 * d[0] * d[1] * d[6] * d[10] * std::pow(d[12], 2) -
        3 * std::pow(d[0], 2) * d[7] * d[10] * std::pow(d[12], 2) +
        3 * std::pow(d[1], 2) * d[7] * d[10] * std::pow(d[12], 2) +
        3 * d[7] * std::pow(d[8], 2) * d[10] * std::pow(d[12], 2) - std::pow(d[3], 2) * d[4] * std::pow(d[11], 2) -
        std::pow(d[4], 3) * std::pow(d[11], 2) - 2 * d[5] * d[7] * d[8] * std::pow(d[11], 2) +
        d[4] * std::pow(d[8], 2) * std::pow(d[11], 2) + 2 * d[1] * std::pow(d[3], 2) * d[11] * d[12] +
        2 * d[1] * std::pow(d[4], 2) * d[11] * d[12] - 2 * d[1] * std::pow(d[8], 2) * d[11] * d[12] -
        2 * d[0] * d[1] * d[3] * std::pow(d[12], 2) + std::pow(d[0], 2) * d[4] * std::pow(d[12], 2) -
        std::pow(d[1], 2) * d[4] * std::pow(d[12], 2) - 2 * d[5] * d[7] * d[8] * std::pow(d[12], 2) -
        d[4] * std::pow(d[8], 2) * std::pow(d[12], 2);
    coeffs[41] = std::pow(d[6], 2) * d[7] * std::pow(d[11], 2) * d[12] +
                 std::pow(d[7], 3) * std::pow(d[11], 2) * d[12] + std::pow(d[6], 2) * d[7] * std::pow(d[12], 3) +
                 std::pow(d[7], 3) * std::pow(d[12], 3);
    coeffs[42] =
        -std::pow(d[5], 2) * d[7] * std::pow(d[9], 2) * d[10] + 2 * d[4] * d[5] * d[8] * std::pow(d[9], 2) * d[10] +
        2 * d[2] * d[5] * d[7] * d[9] * std::pow(d[10], 2) - 2 * d[2] * d[4] * d[8] * d[9] * std::pow(d[10], 2) -
        2 * d[1] * d[5] * d[8] * d[9] * std::pow(d[10], 2) - std::pow(d[2], 2) * d[7] * std::pow(d[10], 3) +
        2 * d[1] * d[2] * d[8] * std::pow(d[10], 3) - d[4] * std::pow(d[5], 2) * std::pow(d[9], 2) +
        2 * d[1] * std::pow(d[5], 2) * d[9] * d[10] + std::pow(d[2], 2) * d[4] * std::pow(d[10], 2) -
        2 * d[1] * d[2] * d[5] * std::pow(d[10], 2);
    coeffs[43] =
        4 * d[3] * d[4] * d[6] * d[9] * d[10] * d[11] - 2 * std::pow(d[3], 2) * d[7] * d[9] * d[10] * d[11] +
        2 * std::pow(d[4], 2) * d[7] * d[9] * d[10] * d[11] + 2 * d[7] * std::pow(d[8], 2) * d[9] * d[10] * d[11] -
        2 * d[1] * d[3] * d[6] * std::pow(d[10], 2) * d[11] - 2 * d[0] * d[4] * d[6] * std::pow(d[10], 2) * d[11] +
        2 * d[0] * d[3] * d[7] * std::pow(d[10], 2) * d[11] - 2 * d[1] * d[4] * d[7] * std::pow(d[10], 2) * d[11] +
        2 * d[3] * d[4] * d[6] * std::pow(d[9], 2) * d[12] - std::pow(d[3], 2) * d[7] * std::pow(d[9], 2) * d[12] +
        std::pow(d[4], 2) * d[7] * std::pow(d[9], 2) * d[12] + d[7] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[12] -
        4 * d[1] * d[3] * d[6] * d[9] * d[10] * d[12] - 4 * d[0] * d[4] * d[6] * d[9] * d[10] * d[12] +
        4 * d[0] * d[3] * d[7] * d[9] * d[10] * d[12] - 4 * d[1] * d[4] * d[7] * d[9] * d[10] * d[12] +
        6 * d[0] * d[1] * d[6] * std::pow(d[10], 2) * d[12] -
        3 * std::pow(d[0], 2) * d[7] * std::pow(d[10], 2) * d[12] +
        3 * std::pow(d[1], 2) * d[7] * std::pow(d[10], 2) * d[12] +
        3 * d[7] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[12] - 2 * std::pow(d[3], 2) * d[4] * d[9] * d[11] -
        2 * std::pow(d[4], 3) * d[9] * d[11] - 4 * d[5] * d[7] * d[8] * d[9] * d[11] +
        2 * d[4] * std::pow(d[8], 2) * d[9] * d[11] + 2 * d[1] * std::pow(d[3], 2) * d[10] * d[11] +
        2 * d[1] * std::pow(d[4], 2) * d[10] * d[11] - 2 * d[1] * std::pow(d[8], 2) * d[10] * d[11] +
        2 * d[1] * std::pow(d[3], 2) * d[9] * d[12] + 2 * d[1] * std::pow(d[4], 2) * d[9] * d[12] -
        2 * d[1] * std::pow(d[8], 2) * d[9] * d[12] - 4 * d[0] * d[1] * d[3] * d[10] * d[12] +
        2 * std::pow(d[0], 2) * d[4] * d[10] * d[12] - 2 * std::pow(d[1], 2) * d[4] * d[10] * d[12] -
        4 * d[5] * d[7] * d[8] * d[10] * d[12] - 2 * d[4] * std::pow(d[8], 2) * d[10] * d[12] +
        2 * d[2] * d[5] * d[7] * d[11] - 2 * d[2] * d[4] * d[8] * d[11] + 2 * d[1] * d[5] * d[8] * d[11] -
        std::pow(d[2], 2) * d[7] * d[12] + std::pow(d[5], 2) * d[7] * d[12] + 2 * d[1] * d[2] * d[8] * d[12] +
        2 * d[4] * d[5] * d[8] * d[12];
    coeffs[44] =
        std::pow(d[6], 2) * d[7] * d[10] * std::pow(d[11], 2) + std::pow(d[7], 3) * d[10] * std::pow(d[11], 2) +
        2 * std::pow(d[6], 2) * d[7] * d[9] * d[11] * d[12] + 2 * std::pow(d[7], 3) * d[9] * d[11] * d[12] +
        3 * std::pow(d[6], 2) * d[7] * d[10] * std::pow(d[12], 2) + 3 * std::pow(d[7], 3) * d[10] * std::pow(d[12], 2) +
        d[4] * std::pow(d[6], 2) * std::pow(d[11], 2) - 2 * d[3] * d[6] * d[7] * std::pow(d[11], 2) -
        d[4] * std::pow(d[7], 2) * std::pow(d[11], 2) - 2 * d[1] * std::pow(d[6], 2) * d[11] * d[12] -
        2 * d[1] * std::pow(d[7], 2) * d[11] * d[12] - d[4] * std::pow(d[6], 2) * std::pow(d[12], 2) -
        2 * d[3] * d[6] * d[7] * std::pow(d[12], 2) - 3 * d[4] * std::pow(d[7], 2) * std::pow(d[12], 2);
    coeffs[45] =
        2 * d[3] * d[4] * d[6] * std::pow(d[9], 2) * d[10] - std::pow(d[3], 2) * d[7] * std::pow(d[9], 2) * d[10] +
        std::pow(d[4], 2) * d[7] * std::pow(d[9], 2) * d[10] + d[7] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[10] -
        2 * d[1] * d[3] * d[6] * d[9] * std::pow(d[10], 2) - 2 * d[0] * d[4] * d[6] * d[9] * std::pow(d[10], 2) +
        2 * d[0] * d[3] * d[7] * d[9] * std::pow(d[10], 2) - 2 * d[1] * d[4] * d[7] * d[9] * std::pow(d[10], 2) +
        2 * d[0] * d[1] * d[6] * std::pow(d[10], 3) - std::pow(d[0], 2) * d[7] * std::pow(d[10], 3) +
        std::pow(d[1], 2) * d[7] * std::pow(d[10], 3) + d[7] * std::pow(d[8], 2) * std::pow(d[10], 3) -
        std::pow(d[3], 2) * d[4] * std::pow(d[9], 2) - std::pow(d[4], 3) * std::pow(d[9], 2) -
        2 * d[5] * d[7] * d[8] * std::pow(d[9], 2) + d[4] * std::pow(d[8], 2) * std::pow(d[9], 2) +
        2 * d[1] * std::pow(d[3], 2) * d[9] * d[10] + 2 * d[1] * std::pow(d[4], 2) * d[9] * d[10] -
        2 * d[1] * std::pow(d[8], 2) * d[9] * d[10] - 2 * d[0] * d[1] * d[3] * std::pow(d[10], 2) +
        std::pow(d[0], 2) * d[4] * std::pow(d[10], 2) - std::pow(d[1], 2) * d[4] * std::pow(d[10], 2) -
        2 * d[5] * d[7] * d[8] * std::pow(d[10], 2) - d[4] * std::pow(d[8], 2) * std::pow(d[10], 2) +
        2 * d[2] * d[5] * d[7] * d[9] - 2 * d[2] * d[4] * d[8] * d[9] + 2 * d[1] * d[5] * d[8] * d[9] -
        std::pow(d[2], 2) * d[7] * d[10] + std::pow(d[5], 2) * d[7] * d[10] + 2 * d[1] * d[2] * d[8] * d[10] +
        2 * d[4] * d[5] * d[8] * d[10] + std::pow(d[2], 2) * d[4] - 2 * d[1] * d[2] * d[5] - d[4] * std::pow(d[5], 2);
    coeffs[46] = 2 * std::pow(d[6], 2) * d[7] * d[9] * d[10] * d[11] + 2 * std::pow(d[7], 3) * d[9] * d[10] * d[11] +
                 std::pow(d[6], 2) * d[7] * std::pow(d[9], 2) * d[12] + std::pow(d[7], 3) * std::pow(d[9], 2) * d[12] +
                 3 * std::pow(d[6], 2) * d[7] * std::pow(d[10], 2) * d[12] +
                 3 * std::pow(d[7], 3) * std::pow(d[10], 2) * d[12] + 2 * d[4] * std::pow(d[6], 2) * d[9] * d[11] -
                 4 * d[3] * d[6] * d[7] * d[9] * d[11] - 2 * d[4] * std::pow(d[7], 2) * d[9] * d[11] -
                 2 * d[1] * std::pow(d[6], 2) * d[10] * d[11] - 2 * d[1] * std::pow(d[7], 2) * d[10] * d[11] -
                 2 * d[1] * std::pow(d[6], 2) * d[9] * d[12] - 2 * d[1] * std::pow(d[7], 2) * d[9] * d[12] -
                 2 * d[4] * std::pow(d[6], 2) * d[10] * d[12] - 4 * d[3] * d[6] * d[7] * d[10] * d[12] -
                 6 * d[4] * std::pow(d[7], 2) * d[10] * d[12] + 2 * d[1] * d[3] * d[6] * d[11] -
                 2 * d[0] * d[4] * d[6] * d[11] + 2 * d[0] * d[3] * d[7] * d[11] + 2 * d[1] * d[4] * d[7] * d[11] +
                 2 * d[0] * d[1] * d[6] * d[12] + 2 * d[3] * d[4] * d[6] * d[12] - std::pow(d[0], 2) * d[7] * d[12] +
                 std::pow(d[1], 2) * d[7] * d[12] + std::pow(d[3], 2) * d[7] * d[12] +
                 3 * std::pow(d[4], 2) * d[7] * d[12];
    coeffs[47] = std::pow(d[6], 2) * d[7] * std::pow(d[9], 2) * d[10] + std::pow(d[7], 3) * std::pow(d[9], 2) * d[10] +
                 std::pow(d[6], 2) * d[7] * std::pow(d[10], 3) + std::pow(d[7], 3) * std::pow(d[10], 3) +
                 d[4] * std::pow(d[6], 2) * std::pow(d[9], 2) - 2 * d[3] * d[6] * d[7] * std::pow(d[9], 2) -
                 d[4] * std::pow(d[7], 2) * std::pow(d[9], 2) - 2 * d[1] * std::pow(d[6], 2) * d[9] * d[10] -
                 2 * d[1] * std::pow(d[7], 2) * d[9] * d[10] - d[4] * std::pow(d[6], 2) * std::pow(d[10], 2) -
                 2 * d[3] * d[6] * d[7] * std::pow(d[10], 2) - 3 * d[4] * std::pow(d[7], 2) * std::pow(d[10], 2) +
                 2 * d[1] * d[3] * d[6] * d[9] - 2 * d[0] * d[4] * d[6] * d[9] + 2 * d[0] * d[3] * d[7] * d[9] +
                 2 * d[1] * d[4] * d[7] * d[9] + 2 * d[0] * d[1] * d[6] * d[10] + 2 * d[3] * d[4] * d[6] * d[10] -
                 std::pow(d[0], 2) * d[7] * d[10] + std::pow(d[1], 2) * d[7] * d[10] +
                 std::pow(d[3], 2) * d[7] * d[10] + 3 * std::pow(d[4], 2) * d[7] * d[10] - 2 * d[0] * d[1] * d[3] +
                 std::pow(d[0], 2) * d[4] - std::pow(d[1], 2) * d[4] - std::pow(d[3], 2) * d[4] - std::pow(d[4], 3);
    coeffs[48] =
        std::pow(d[5], 2) * d[7] * std::pow(d[11], 3) - 2 * d[4] * d[5] * d[8] * std::pow(d[11], 3) -
        2 * d[2] * d[5] * d[7] * std::pow(d[11], 2) * d[12] + 2 * d[2] * d[4] * d[8] * std::pow(d[11], 2) * d[12] +
        2 * d[1] * d[5] * d[8] * std::pow(d[11], 2) * d[12] + std::pow(d[2], 2) * d[7] * d[11] * std::pow(d[12], 2) -
        2 * d[1] * d[2] * d[8] * d[11] * std::pow(d[12], 2);
    coeffs[49] =
        3 * std::pow(d[5], 2) * d[7] * d[9] * std::pow(d[11], 2) - 6 * d[4] * d[5] * d[8] * d[9] * std::pow(d[11], 2) -
        2 * d[2] * d[5] * d[7] * d[10] * std::pow(d[11], 2) + 2 * d[2] * d[4] * d[8] * d[10] * std::pow(d[11], 2) +
        2 * d[1] * d[5] * d[8] * d[10] * std::pow(d[11], 2) - 4 * d[2] * d[5] * d[7] * d[9] * d[11] * d[12] +
        4 * d[2] * d[4] * d[8] * d[9] * d[11] * d[12] + 4 * d[1] * d[5] * d[8] * d[9] * d[11] * d[12] +
        2 * std::pow(d[2], 2) * d[7] * d[10] * d[11] * d[12] - 4 * d[1] * d[2] * d[8] * d[10] * d[11] * d[12] +
        std::pow(d[2], 2) * d[7] * d[9] * std::pow(d[12], 2) - 2 * d[1] * d[2] * d[8] * d[9] * std::pow(d[12], 2) +
        2 * d[2] * d[4] * d[5] * std::pow(d[11], 2) - d[1] * std::pow(d[5], 2) * std::pow(d[11], 2) -
        2 * std::pow(d[2], 2) * d[4] * d[11] * d[12] + d[1] * std::pow(d[2], 2) * std::pow(d[12], 2);
    coeffs[50] =
        -2 * d[3] * d[4] * d[6] * std::pow(d[11], 3) + std::pow(d[3], 2) * d[7] * std::pow(d[11], 3) -
        std::pow(d[4], 2) * d[7] * std::pow(d[11], 3) - d[7] * std::pow(d[8], 2) * std::pow(d[11], 3) +
        2 * d[1] * d[3] * d[6] * std::pow(d[11], 2) * d[12] + 2 * d[0] * d[4] * d[6] * std::pow(d[11], 2) * d[12] -
        2 * d[0] * d[3] * d[7] * std::pow(d[11], 2) * d[12] + 2 * d[1] * d[4] * d[7] * std::pow(d[11], 2) * d[12] -
        2 * d[0] * d[1] * d[6] * d[11] * std::pow(d[12], 2) + std::pow(d[0], 2) * d[7] * d[11] * std::pow(d[12], 2) -
        std::pow(d[1], 2) * d[7] * d[11] * std::pow(d[12], 2) - d[7] * std::pow(d[8], 2) * d[11] * std::pow(d[12], 2);
    coeffs[51] =
        3 * std::pow(d[5], 2) * d[7] * std::pow(d[9], 2) * d[11] - 6 * d[4] * d[5] * d[8] * std::pow(d[9], 2) * d[11] -
        4 * d[2] * d[5] * d[7] * d[9] * d[10] * d[11] + 4 * d[2] * d[4] * d[8] * d[9] * d[10] * d[11] +
        4 * d[1] * d[5] * d[8] * d[9] * d[10] * d[11] + std::pow(d[2], 2) * d[7] * std::pow(d[10], 2) * d[11] -
        2 * d[1] * d[2] * d[8] * std::pow(d[10], 2) * d[11] - 2 * d[2] * d[5] * d[7] * std::pow(d[9], 2) * d[12] +
        2 * d[2] * d[4] * d[8] * std::pow(d[9], 2) * d[12] + 2 * d[1] * d[5] * d[8] * std::pow(d[9], 2) * d[12] +
        2 * std::pow(d[2], 2) * d[7] * d[9] * d[10] * d[12] - 4 * d[1] * d[2] * d[8] * d[9] * d[10] * d[12] +
        4 * d[2] * d[4] * d[5] * d[9] * d[11] - 2 * d[1] * std::pow(d[5], 2) * d[9] * d[11] -
        2 * std::pow(d[2], 2) * d[4] * d[10] * d[11] - 2 * std::pow(d[2], 2) * d[4] * d[9] * d[12] +
        2 * d[1] * std::pow(d[2], 2) * d[10] * d[12];
    coeffs[52] =
        -6 * d[3] * d[4] * d[6] * d[9] * std::pow(d[11], 2) + 3 * std::pow(d[3], 2) * d[7] * d[9] * std::pow(d[11], 2) -
        3 * std::pow(d[4], 2) * d[7] * d[9] * std::pow(d[11], 2) -
        3 * d[7] * std::pow(d[8], 2) * d[9] * std::pow(d[11], 2) + 2 * d[1] * d[3] * d[6] * d[10] * std::pow(d[11], 2) +
        2 * d[0] * d[4] * d[6] * d[10] * std::pow(d[11], 2) - 2 * d[0] * d[3] * d[7] * d[10] * std::pow(d[11], 2) +
        2 * d[1] * d[4] * d[7] * d[10] * std::pow(d[11], 2) + 4 * d[1] * d[3] * d[6] * d[9] * d[11] * d[12] +
        4 * d[0] * d[4] * d[6] * d[9] * d[11] * d[12] - 4 * d[0] * d[3] * d[7] * d[9] * d[11] * d[12] +
        4 * d[1] * d[4] * d[7] * d[9] * d[11] * d[12] - 4 * d[0] * d[1] * d[6] * d[10] * d[11] * d[12] +
        2 * std::pow(d[0], 2) * d[7] * d[10] * d[11] * d[12] - 2 * std::pow(d[1], 2) * d[7] * d[10] * d[11] * d[12] -
        2 * d[7] * std::pow(d[8], 2) * d[10] * d[11] * d[12] - 2 * d[0] * d[1] * d[6] * d[9] * std::pow(d[12], 2) +
        std::pow(d[0], 2) * d[7] * d[9] * std::pow(d[12], 2) - std::pow(d[1], 2) * d[7] * d[9] * std::pow(d[12], 2) -
        d[7] * std::pow(d[8], 2) * d[9] * std::pow(d[12], 2) - d[1] * std::pow(d[3], 2) * std::pow(d[11], 2) +
        2 * d[0] * d[3] * d[4] * std::pow(d[11], 2) + d[1] * std::pow(d[4], 2) * std::pow(d[11], 2) +
        2 * d[2] * d[7] * d[8] * std::pow(d[11], 2) + d[1] * std::pow(d[8], 2) * std::pow(d[11], 2) -
        2 * std::pow(d[0], 2) * d[4] * d[11] * d[12] - 2 * std::pow(d[1], 2) * d[4] * d[11] * d[12] +
        2 * d[4] * std::pow(d[8], 2) * d[11] * d[12] + std::pow(d[0], 2) * d[1] * std::pow(d[12], 2) +
        std::pow(d[1], 3) * std::pow(d[12], 2) + 2 * d[2] * d[7] * d[8] * std::pow(d[12], 2) -
        d[1] * std::pow(d[8], 2) * std::pow(d[12], 2);
    coeffs[53] = -std::pow(d[6], 2) * d[7] * std::pow(d[11], 3) - std::pow(d[7], 3) * std::pow(d[11], 3) -
                 std::pow(d[6], 2) * d[7] * d[11] * std::pow(d[12], 2) - std::pow(d[7], 3) * d[11] * std::pow(d[12], 2);
    coeffs[54] =
        std::pow(d[5], 2) * d[7] * std::pow(d[9], 3) - 2 * d[4] * d[5] * d[8] * std::pow(d[9], 3) -
        2 * d[2] * d[5] * d[7] * std::pow(d[9], 2) * d[10] + 2 * d[2] * d[4] * d[8] * std::pow(d[9], 2) * d[10] +
        2 * d[1] * d[5] * d[8] * std::pow(d[9], 2) * d[10] + std::pow(d[2], 2) * d[7] * d[9] * std::pow(d[10], 2) -
        2 * d[1] * d[2] * d[8] * d[9] * std::pow(d[10], 2) + 2 * d[2] * d[4] * d[5] * std::pow(d[9], 2) -
        d[1] * std::pow(d[5], 2) * std::pow(d[9], 2) - 2 * std::pow(d[2], 2) * d[4] * d[9] * d[10] +
        d[1] * std::pow(d[2], 2) * std::pow(d[10], 2);
    coeffs[55] =
        -6 * d[3] * d[4] * d[6] * std::pow(d[9], 2) * d[11] + 3 * std::pow(d[3], 2) * d[7] * std::pow(d[9], 2) * d[11] -
        3 * std::pow(d[4], 2) * d[7] * std::pow(d[9], 2) * d[11] -
        3 * d[7] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[11] + 4 * d[1] * d[3] * d[6] * d[9] * d[10] * d[11] +
        4 * d[0] * d[4] * d[6] * d[9] * d[10] * d[11] - 4 * d[0] * d[3] * d[7] * d[9] * d[10] * d[11] +
        4 * d[1] * d[4] * d[7] * d[9] * d[10] * d[11] - 2 * d[0] * d[1] * d[6] * std::pow(d[10], 2) * d[11] +
        std::pow(d[0], 2) * d[7] * std::pow(d[10], 2) * d[11] - std::pow(d[1], 2) * d[7] * std::pow(d[10], 2) * d[11] -
        d[7] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[11] + 2 * d[1] * d[3] * d[6] * std::pow(d[9], 2) * d[12] +
        2 * d[0] * d[4] * d[6] * std::pow(d[9], 2) * d[12] - 2 * d[0] * d[3] * d[7] * std::pow(d[9], 2) * d[12] +
        2 * d[1] * d[4] * d[7] * std::pow(d[9], 2) * d[12] - 4 * d[0] * d[1] * d[6] * d[9] * d[10] * d[12] +
        2 * std::pow(d[0], 2) * d[7] * d[9] * d[10] * d[12] - 2 * std::pow(d[1], 2) * d[7] * d[9] * d[10] * d[12] -
        2 * d[7] * std::pow(d[8], 2) * d[9] * d[10] * d[12] - 2 * d[1] * std::pow(d[3], 2) * d[9] * d[11] +
        4 * d[0] * d[3] * d[4] * d[9] * d[11] + 2 * d[1] * std::pow(d[4], 2) * d[9] * d[11] +
        4 * d[2] * d[7] * d[8] * d[9] * d[11] + 2 * d[1] * std::pow(d[8], 2) * d[9] * d[11] -
        2 * std::pow(d[0], 2) * d[4] * d[10] * d[11] - 2 * std::pow(d[1], 2) * d[4] * d[10] * d[11] +
        2 * d[4] * std::pow(d[8], 2) * d[10] * d[11] - 2 * std::pow(d[0], 2) * d[4] * d[9] * d[12] -
        2 * std::pow(d[1], 2) * d[4] * d[9] * d[12] + 2 * d[4] * std::pow(d[8], 2) * d[9] * d[12] +
        2 * std::pow(d[0], 2) * d[1] * d[10] * d[12] + 2 * std::pow(d[1], 3) * d[10] * d[12] +
        4 * d[2] * d[7] * d[8] * d[10] * d[12] - 2 * d[1] * std::pow(d[8], 2) * d[10] * d[12] -
        std::pow(d[2], 2) * d[7] * d[11] + std::pow(d[5], 2) * d[7] * d[11] - 2 * d[1] * d[2] * d[8] * d[11] -
        2 * d[4] * d[5] * d[8] * d[11] - 2 * d[2] * d[5] * d[7] * d[12] - 2 * d[2] * d[4] * d[8] * d[12] +
        2 * d[1] * d[5] * d[8] * d[12];
    coeffs[56] = -3 * std::pow(d[6], 2) * d[7] * d[9] * std::pow(d[11], 2) -
                 3 * std::pow(d[7], 3) * d[9] * std::pow(d[11], 2) -
                 2 * std::pow(d[6], 2) * d[7] * d[10] * d[11] * d[12] - 2 * std::pow(d[7], 3) * d[10] * d[11] * d[12] -
                 std::pow(d[6], 2) * d[7] * d[9] * std::pow(d[12], 2) - std::pow(d[7], 3) * d[9] * std::pow(d[12], 2) +
                 d[1] * std::pow(d[6], 2) * std::pow(d[11], 2) + 2 * d[0] * d[6] * d[7] * std::pow(d[11], 2) +
                 3 * d[1] * std::pow(d[7], 2) * std::pow(d[11], 2) + 2 * d[4] * std::pow(d[6], 2) * d[11] * d[12] +
                 2 * d[4] * std::pow(d[7], 2) * d[11] * d[12] - d[1] * std::pow(d[6], 2) * std::pow(d[12], 2) +
                 2 * d[0] * d[6] * d[7] * std::pow(d[12], 2) + d[1] * std::pow(d[7], 2) * std::pow(d[12], 2);
    coeffs[57] =
        -2 * d[3] * d[4] * d[6] * std::pow(d[9], 3) + std::pow(d[3], 2) * d[7] * std::pow(d[9], 3) -
        std::pow(d[4], 2) * d[7] * std::pow(d[9], 3) - d[7] * std::pow(d[8], 2) * std::pow(d[9], 3) +
        2 * d[1] * d[3] * d[6] * std::pow(d[9], 2) * d[10] + 2 * d[0] * d[4] * d[6] * std::pow(d[9], 2) * d[10] -
        2 * d[0] * d[3] * d[7] * std::pow(d[9], 2) * d[10] + 2 * d[1] * d[4] * d[7] * std::pow(d[9], 2) * d[10] -
        2 * d[0] * d[1] * d[6] * d[9] * std::pow(d[10], 2) + std::pow(d[0], 2) * d[7] * d[9] * std::pow(d[10], 2) -
        std::pow(d[1], 2) * d[7] * d[9] * std::pow(d[10], 2) - d[7] * std::pow(d[8], 2) * d[9] * std::pow(d[10], 2) -
        d[1] * std::pow(d[3], 2) * std::pow(d[9], 2) + 2 * d[0] * d[3] * d[4] * std::pow(d[9], 2) +
        d[1] * std::pow(d[4], 2) * std::pow(d[9], 2) + 2 * d[2] * d[7] * d[8] * std::pow(d[9], 2) +
        d[1] * std::pow(d[8], 2) * std::pow(d[9], 2) - 2 * std::pow(d[0], 2) * d[4] * d[9] * d[10] -
        2 * std::pow(d[1], 2) * d[4] * d[9] * d[10] + 2 * d[4] * std::pow(d[8], 2) * d[9] * d[10] +
        std::pow(d[0], 2) * d[1] * std::pow(d[10], 2) + std::pow(d[1], 3) * std::pow(d[10], 2) +
        2 * d[2] * d[7] * d[8] * std::pow(d[10], 2) - d[1] * std::pow(d[8], 2) * std::pow(d[10], 2) -
        std::pow(d[2], 2) * d[7] * d[9] + std::pow(d[5], 2) * d[7] * d[9] - 2 * d[1] * d[2] * d[8] * d[9] -
        2 * d[4] * d[5] * d[8] * d[9] - 2 * d[2] * d[5] * d[7] * d[10] - 2 * d[2] * d[4] * d[8] * d[10] +
        2 * d[1] * d[5] * d[8] * d[10] + d[1] * std::pow(d[2], 2) + 2 * d[2] * d[4] * d[5] - d[1] * std::pow(d[5], 2);
    coeffs[58] =
        -3 * std::pow(d[6], 2) * d[7] * std::pow(d[9], 2) * d[11] - 3 * std::pow(d[7], 3) * std::pow(d[9], 2) * d[11] -
        std::pow(d[6], 2) * d[7] * std::pow(d[10], 2) * d[11] - std::pow(d[7], 3) * std::pow(d[10], 2) * d[11] -
        2 * std::pow(d[6], 2) * d[7] * d[9] * d[10] * d[12] - 2 * std::pow(d[7], 3) * d[9] * d[10] * d[12] +
        2 * d[1] * std::pow(d[6], 2) * d[9] * d[11] + 4 * d[0] * d[6] * d[7] * d[9] * d[11] +
        6 * d[1] * std::pow(d[7], 2) * d[9] * d[11] + 2 * d[4] * std::pow(d[6], 2) * d[10] * d[11] +
        2 * d[4] * std::pow(d[7], 2) * d[10] * d[11] + 2 * d[4] * std::pow(d[6], 2) * d[9] * d[12] +
        2 * d[4] * std::pow(d[7], 2) * d[9] * d[12] - 2 * d[1] * std::pow(d[6], 2) * d[10] * d[12] +
        4 * d[0] * d[6] * d[7] * d[10] * d[12] + 2 * d[1] * std::pow(d[7], 2) * d[10] * d[12] -
        2 * d[0] * d[1] * d[6] * d[11] - 2 * d[3] * d[4] * d[6] * d[11] - std::pow(d[0], 2) * d[7] * d[11] -
        3 * std::pow(d[1], 2) * d[7] * d[11] + std::pow(d[3], 2) * d[7] * d[11] - std::pow(d[4], 2) * d[7] * d[11] +
        2 * d[1] * d[3] * d[6] * d[12] - 2 * d[0] * d[4] * d[6] * d[12] - 2 * d[0] * d[3] * d[7] * d[12] -
        2 * d[1] * d[4] * d[7] * d[12];
    coeffs[59] = -std::pow(d[6], 2) * d[7] * std::pow(d[9], 3) - std::pow(d[7], 3) * std::pow(d[9], 3) -
                 std::pow(d[6], 2) * d[7] * d[9] * std::pow(d[10], 2) - std::pow(d[7], 3) * d[9] * std::pow(d[10], 2) +
                 d[1] * std::pow(d[6], 2) * std::pow(d[9], 2) + 2 * d[0] * d[6] * d[7] * std::pow(d[9], 2) +
                 3 * d[1] * std::pow(d[7], 2) * std::pow(d[9], 2) + 2 * d[4] * std::pow(d[6], 2) * d[9] * d[10] +
                 2 * d[4] * std::pow(d[7], 2) * d[9] * d[10] - d[1] * std::pow(d[6], 2) * std::pow(d[10], 2) +
                 2 * d[0] * d[6] * d[7] * std::pow(d[10], 2) + d[1] * std::pow(d[7], 2) * std::pow(d[10], 2) -
                 2 * d[0] * d[1] * d[6] * d[9] - 2 * d[3] * d[4] * d[6] * d[9] - std::pow(d[0], 2) * d[7] * d[9] -
                 3 * std::pow(d[1], 2) * d[7] * d[9] + std::pow(d[3], 2) * d[7] * d[9] -
                 std::pow(d[4], 2) * d[7] * d[9] + 2 * d[1] * d[3] * d[6] * d[10] - 2 * d[0] * d[4] * d[6] * d[10] -
                 2 * d[0] * d[3] * d[7] * d[10] - 2 * d[1] * d[4] * d[7] * d[10] + std::pow(d[0], 2) * d[1] +
                 std::pow(d[1], 3) - d[1] * std::pow(d[3], 2) + 2 * d[0] * d[3] * d[4] + d[1] * std::pow(d[4], 2);
    coeffs[60] = d[4] * std::pow(d[5], 2) * std::pow(d[11], 3) - 2 * d[2] * d[4] * d[5] * std::pow(d[11], 2) * d[12] -
                 d[1] * std::pow(d[5], 2) * std::pow(d[11], 2) * d[12] +
                 std::pow(d[2], 2) * d[4] * d[11] * std::pow(d[12], 2) +
                 2 * d[1] * d[2] * d[5] * d[11] * std::pow(d[12], 2) - d[1] * std::pow(d[2], 2) * std::pow(d[12], 3);
    coeffs[61] =
        3 * d[4] * std::pow(d[5], 2) * d[9] * std::pow(d[11], 2) - 2 * d[2] * d[4] * d[5] * d[10] * std::pow(d[11], 2) -
        d[1] * std::pow(d[5], 2) * d[10] * std::pow(d[11], 2) - 4 * d[2] * d[4] * d[5] * d[9] * d[11] * d[12] -
        2 * d[1] * std::pow(d[5], 2) * d[9] * d[11] * d[12] + 2 * std::pow(d[2], 2) * d[4] * d[10] * d[11] * d[12] +
        4 * d[1] * d[2] * d[5] * d[10] * d[11] * d[12] + std::pow(d[2], 2) * d[4] * d[9] * std::pow(d[12], 2) +
        2 * d[1] * d[2] * d[5] * d[9] * std::pow(d[12], 2) - 3 * d[1] * std::pow(d[2], 2) * d[10] * std::pow(d[12], 2);
    coeffs[62] =
        std::pow(d[3], 2) * d[4] * std::pow(d[11], 3) + std::pow(d[4], 3) * std::pow(d[11], 3) +
        2 * d[5] * d[7] * d[8] * std::pow(d[11], 3) - d[4] * std::pow(d[8], 2) * std::pow(d[11], 3) -
        d[1] * std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] - 2 * d[0] * d[3] * d[4] * std::pow(d[11], 2) * d[12] -
        3 * d[1] * std::pow(d[4], 2) * std::pow(d[11], 2) * d[12] -
        2 * d[2] * d[7] * d[8] * std::pow(d[11], 2) * d[12] + d[1] * std::pow(d[8], 2) * std::pow(d[11], 2) * d[12] +
        2 * d[0] * d[1] * d[3] * d[11] * std::pow(d[12], 2) + std::pow(d[0], 2) * d[4] * d[11] * std::pow(d[12], 2) +
        3 * std::pow(d[1], 2) * d[4] * d[11] * std::pow(d[12], 2) +
        2 * d[5] * d[7] * d[8] * d[11] * std::pow(d[12], 2) - d[4] * std::pow(d[8], 2) * d[11] * std::pow(d[12], 2) -
        std::pow(d[0], 2) * d[1] * std::pow(d[12], 3) - std::pow(d[1], 3) * std::pow(d[12], 3) -
        2 * d[2] * d[7] * d[8] * std::pow(d[12], 3) + d[1] * std::pow(d[8], 2) * std::pow(d[12], 3);
    coeffs[63] =
        3 * d[4] * std::pow(d[5], 2) * std::pow(d[9], 2) * d[11] - 4 * d[2] * d[4] * d[5] * d[9] * d[10] * d[11] -
        2 * d[1] * std::pow(d[5], 2) * d[9] * d[10] * d[11] + std::pow(d[2], 2) * d[4] * std::pow(d[10], 2) * d[11] +
        2 * d[1] * d[2] * d[5] * std::pow(d[10], 2) * d[11] - 2 * d[2] * d[4] * d[5] * std::pow(d[9], 2) * d[12] -
        d[1] * std::pow(d[5], 2) * std::pow(d[9], 2) * d[12] + 2 * std::pow(d[2], 2) * d[4] * d[9] * d[10] * d[12] +
        4 * d[1] * d[2] * d[5] * d[9] * d[10] * d[12] - 3 * d[1] * std::pow(d[2], 2) * std::pow(d[10], 2) * d[12];
    coeffs[64] =
        3 * std::pow(d[3], 2) * d[4] * d[9] * std::pow(d[11], 2) + 3 * std::pow(d[4], 3) * d[9] * std::pow(d[11], 2) +
        6 * d[5] * d[7] * d[8] * d[9] * std::pow(d[11], 2) - 3 * d[4] * std::pow(d[8], 2) * d[9] * std::pow(d[11], 2) -
        d[1] * std::pow(d[3], 2) * d[10] * std::pow(d[11], 2) - 2 * d[0] * d[3] * d[4] * d[10] * std::pow(d[11], 2) -
        3 * d[1] * std::pow(d[4], 2) * d[10] * std::pow(d[11], 2) -
        2 * d[2] * d[7] * d[8] * d[10] * std::pow(d[11], 2) + d[1] * std::pow(d[8], 2) * d[10] * std::pow(d[11], 2) -
        2 * d[1] * std::pow(d[3], 2) * d[9] * d[11] * d[12] - 4 * d[0] * d[3] * d[4] * d[9] * d[11] * d[12] -
        6 * d[1] * std::pow(d[4], 2) * d[9] * d[11] * d[12] - 4 * d[2] * d[7] * d[8] * d[9] * d[11] * d[12] +
        2 * d[1] * std::pow(d[8], 2) * d[9] * d[11] * d[12] + 4 * d[0] * d[1] * d[3] * d[10] * d[11] * d[12] +
        2 * std::pow(d[0], 2) * d[4] * d[10] * d[11] * d[12] + 6 * std::pow(d[1], 2) * d[4] * d[10] * d[11] * d[12] +
        4 * d[5] * d[7] * d[8] * d[10] * d[11] * d[12] - 2 * d[4] * std::pow(d[8], 2) * d[10] * d[11] * d[12] +
        2 * d[0] * d[1] * d[3] * d[9] * std::pow(d[12], 2) + std::pow(d[0], 2) * d[4] * d[9] * std::pow(d[12], 2) +
        3 * std::pow(d[1], 2) * d[4] * d[9] * std::pow(d[12], 2) + 2 * d[5] * d[7] * d[8] * d[9] * std::pow(d[12], 2) -
        d[4] * std::pow(d[8], 2) * d[9] * std::pow(d[12], 2) -
        3 * std::pow(d[0], 2) * d[1] * d[10] * std::pow(d[12], 2) - 3 * std::pow(d[1], 3) * d[10] * std::pow(d[12], 2) -
        6 * d[2] * d[7] * d[8] * d[10] * std::pow(d[12], 2) +
        3 * d[1] * std::pow(d[8], 2) * d[10] * std::pow(d[12], 2) - 2 * d[2] * d[5] * d[7] * std::pow(d[11], 2) +
        2 * d[2] * d[4] * d[8] * std::pow(d[11], 2) - 2 * d[1] * d[5] * d[8] * std::pow(d[11], 2) +
        2 * std::pow(d[2], 2) * d[7] * d[11] * d[12] - 2 * std::pow(d[5], 2) * d[7] * d[11] * d[12] +
        2 * d[2] * d[5] * d[7] * std::pow(d[12], 2) + 2 * d[2] * d[4] * d[8] * std::pow(d[12], 2) -
        2 * d[1] * d[5] * d[8] * std::pow(d[12], 2);
    coeffs[65] =
        -d[4] * std::pow(d[6], 2) * std::pow(d[11], 3) + 2 * d[3] * d[6] * d[7] * std::pow(d[11], 3) +
        d[4] * std::pow(d[7], 2) * std::pow(d[11], 3) + d[1] * std::pow(d[6], 2) * std::pow(d[11], 2) * d[12] -
        2 * d[0] * d[6] * d[7] * std::pow(d[11], 2) * d[12] - d[1] * std::pow(d[7], 2) * std::pow(d[11], 2) * d[12] -
        d[4] * std::pow(d[6], 2) * d[11] * std::pow(d[12], 2) + 2 * d[3] * d[6] * d[7] * d[11] * std::pow(d[12], 2) +
        d[4] * std::pow(d[7], 2) * d[11] * std::pow(d[12], 2) + d[1] * std::pow(d[6], 2) * std::pow(d[12], 3) -
        2 * d[0] * d[6] * d[7] * std::pow(d[12], 3) - d[1] * std::pow(d[7], 2) * std::pow(d[12], 3);
    coeffs[66] = d[4] * std::pow(d[5], 2) * std::pow(d[9], 3) - 2 * d[2] * d[4] * d[5] * std::pow(d[9], 2) * d[10] -
                 d[1] * std::pow(d[5], 2) * std::pow(d[9], 2) * d[10] +
                 std::pow(d[2], 2) * d[4] * d[9] * std::pow(d[10], 2) +
                 2 * d[1] * d[2] * d[5] * d[9] * std::pow(d[10], 2) - d[1] * std::pow(d[2], 2) * std::pow(d[10], 3);
    coeffs[67] =
        3 * std::pow(d[3], 2) * d[4] * std::pow(d[9], 2) * d[11] + 3 * std::pow(d[4], 3) * std::pow(d[9], 2) * d[11] +
        6 * d[5] * d[7] * d[8] * std::pow(d[9], 2) * d[11] - 3 * d[4] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[11] -
        2 * d[1] * std::pow(d[3], 2) * d[9] * d[10] * d[11] - 4 * d[0] * d[3] * d[4] * d[9] * d[10] * d[11] -
        6 * d[1] * std::pow(d[4], 2) * d[9] * d[10] * d[11] - 4 * d[2] * d[7] * d[8] * d[9] * d[10] * d[11] +
        2 * d[1] * std::pow(d[8], 2) * d[9] * d[10] * d[11] + 2 * d[0] * d[1] * d[3] * std::pow(d[10], 2) * d[11] +
        std::pow(d[0], 2) * d[4] * std::pow(d[10], 2) * d[11] +
        3 * std::pow(d[1], 2) * d[4] * std::pow(d[10], 2) * d[11] +
        2 * d[5] * d[7] * d[8] * std::pow(d[10], 2) * d[11] - d[4] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[11] -
        d[1] * std::pow(d[3], 2) * std::pow(d[9], 2) * d[12] - 2 * d[0] * d[3] * d[4] * std::pow(d[9], 2) * d[12] -
        3 * d[1] * std::pow(d[4], 2) * std::pow(d[9], 2) * d[12] - 2 * d[2] * d[7] * d[8] * std::pow(d[9], 2) * d[12] +
        d[1] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[12] + 4 * d[0] * d[1] * d[3] * d[9] * d[10] * d[12] +
        2 * std::pow(d[0], 2) * d[4] * d[9] * d[10] * d[12] + 6 * std::pow(d[1], 2) * d[4] * d[9] * d[10] * d[12] +
        4 * d[5] * d[7] * d[8] * d[9] * d[10] * d[12] - 2 * d[4] * std::pow(d[8], 2) * d[9] * d[10] * d[12] -
        3 * std::pow(d[0], 2) * d[1] * std::pow(d[10], 2) * d[12] - 3 * std::pow(d[1], 3) * std::pow(d[10], 2) * d[12] -
        6 * d[2] * d[7] * d[8] * std::pow(d[10], 2) * d[12] +
        3 * d[1] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[12] - 4 * d[2] * d[5] * d[7] * d[9] * d[11] +
        4 * d[2] * d[4] * d[8] * d[9] * d[11] - 4 * d[1] * d[5] * d[8] * d[9] * d[11] +
        2 * std::pow(d[2], 2) * d[7] * d[10] * d[11] - 2 * std::pow(d[5], 2) * d[7] * d[10] * d[11] +
        2 * std::pow(d[2], 2) * d[7] * d[9] * d[12] - 2 * std::pow(d[5], 2) * d[7] * d[9] * d[12] +
        4 * d[2] * d[5] * d[7] * d[10] * d[12] + 4 * d[2] * d[4] * d[8] * d[10] * d[12] -
        4 * d[1] * d[5] * d[8] * d[10] * d[12] - std::pow(d[2], 2) * d[4] * d[11] + 2 * d[1] * d[2] * d[5] * d[11] +
        d[4] * std::pow(d[5], 2) * d[11] - d[1] * std::pow(d[2], 2) * d[12] - 2 * d[2] * d[4] * d[5] * d[12] +
        d[1] * std::pow(d[5], 2) * d[12];
    coeffs[68] =
        -3 * d[4] * std::pow(d[6], 2) * d[9] * std::pow(d[11], 2) + 6 * d[3] * d[6] * d[7] * d[9] * std::pow(d[11], 2) +
        3 * d[4] * std::pow(d[7], 2) * d[9] * std::pow(d[11], 2) +
        d[1] * std::pow(d[6], 2) * d[10] * std::pow(d[11], 2) - 2 * d[0] * d[6] * d[7] * d[10] * std::pow(d[11], 2) -
        d[1] * std::pow(d[7], 2) * d[10] * std::pow(d[11], 2) + 2 * d[1] * std::pow(d[6], 2) * d[9] * d[11] * d[12] -
        4 * d[0] * d[6] * d[7] * d[9] * d[11] * d[12] - 2 * d[1] * std::pow(d[7], 2) * d[9] * d[11] * d[12] -
        2 * d[4] * std::pow(d[6], 2) * d[10] * d[11] * d[12] + 4 * d[3] * d[6] * d[7] * d[10] * d[11] * d[12] +
        2 * d[4] * std::pow(d[7], 2) * d[10] * d[11] * d[12] - d[4] * std::pow(d[6], 2) * d[9] * std::pow(d[12], 2) +
        2 * d[3] * d[6] * d[7] * d[9] * std::pow(d[12], 2) + d[4] * std::pow(d[7], 2) * d[9] * std::pow(d[12], 2) +
        3 * d[1] * std::pow(d[6], 2) * d[10] * std::pow(d[12], 2) -
        6 * d[0] * d[6] * d[7] * d[10] * std::pow(d[12], 2) -
        3 * d[1] * std::pow(d[7], 2) * d[10] * std::pow(d[12], 2) - 2 * d[1] * d[3] * d[6] * std::pow(d[11], 2) +
        2 * d[0] * d[4] * d[6] * std::pow(d[11], 2) - 2 * d[0] * d[3] * d[7] * std::pow(d[11], 2) -
        2 * d[1] * d[4] * d[7] * std::pow(d[11], 2) + 2 * std::pow(d[0], 2) * d[7] * d[11] * d[12] +
        2 * std::pow(d[1], 2) * d[7] * d[11] * d[12] - 2 * std::pow(d[3], 2) * d[7] * d[11] * d[12] -
        2 * std::pow(d[4], 2) * d[7] * d[11] * d[12] - 2 * d[1] * d[3] * d[6] * std::pow(d[12], 2) +
        2 * d[0] * d[4] * d[6] * std::pow(d[12], 2) + 2 * d[0] * d[3] * d[7] * std::pow(d[12], 2) +
        2 * d[1] * d[4] * d[7] * std::pow(d[12], 2);
    coeffs[69] =
        std::pow(d[3], 2) * d[4] * std::pow(d[9], 3) + std::pow(d[4], 3) * std::pow(d[9], 3) +
        2 * d[5] * d[7] * d[8] * std::pow(d[9], 3) - d[4] * std::pow(d[8], 2) * std::pow(d[9], 3) -
        d[1] * std::pow(d[3], 2) * std::pow(d[9], 2) * d[10] - 2 * d[0] * d[3] * d[4] * std::pow(d[9], 2) * d[10] -
        3 * d[1] * std::pow(d[4], 2) * std::pow(d[9], 2) * d[10] - 2 * d[2] * d[7] * d[8] * std::pow(d[9], 2) * d[10] +
        d[1] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[10] + 2 * d[0] * d[1] * d[3] * d[9] * std::pow(d[10], 2) +
        std::pow(d[0], 2) * d[4] * d[9] * std::pow(d[10], 2) +
        3 * std::pow(d[1], 2) * d[4] * d[9] * std::pow(d[10], 2) + 2 * d[5] * d[7] * d[8] * d[9] * std::pow(d[10], 2) -
        d[4] * std::pow(d[8], 2) * d[9] * std::pow(d[10], 2) - std::pow(d[0], 2) * d[1] * std::pow(d[10], 3) -
        std::pow(d[1], 3) * std::pow(d[10], 3) - 2 * d[2] * d[7] * d[8] * std::pow(d[10], 3) +
        d[1] * std::pow(d[8], 2) * std::pow(d[10], 3) - 2 * d[2] * d[5] * d[7] * std::pow(d[9], 2) +
        2 * d[2] * d[4] * d[8] * std::pow(d[9], 2) - 2 * d[1] * d[5] * d[8] * std::pow(d[9], 2) +
        2 * std::pow(d[2], 2) * d[7] * d[9] * d[10] - 2 * std::pow(d[5], 2) * d[7] * d[9] * d[10] +
        2 * d[2] * d[5] * d[7] * std::pow(d[10], 2) + 2 * d[2] * d[4] * d[8] * std::pow(d[10], 2) -
        2 * d[1] * d[5] * d[8] * std::pow(d[10], 2) - std::pow(d[2], 2) * d[4] * d[9] + 2 * d[1] * d[2] * d[5] * d[9] +
        d[4] * std::pow(d[5], 2) * d[9] - d[1] * std::pow(d[2], 2) * d[10] - 2 * d[2] * d[4] * d[5] * d[10] +
        d[1] * std::pow(d[5], 2) * d[10];
    coeffs[70] =
        -3 * d[4] * std::pow(d[6], 2) * std::pow(d[9], 2) * d[11] + 6 * d[3] * d[6] * d[7] * std::pow(d[9], 2) * d[11] +
        3 * d[4] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[11] + 2 * d[1] * std::pow(d[6], 2) * d[9] * d[10] * d[11] -
        4 * d[0] * d[6] * d[7] * d[9] * d[10] * d[11] - 2 * d[1] * std::pow(d[7], 2) * d[9] * d[10] * d[11] -
        d[4] * std::pow(d[6], 2) * std::pow(d[10], 2) * d[11] + 2 * d[3] * d[6] * d[7] * std::pow(d[10], 2) * d[11] +
        d[4] * std::pow(d[7], 2) * std::pow(d[10], 2) * d[11] + d[1] * std::pow(d[6], 2) * std::pow(d[9], 2) * d[12] -
        2 * d[0] * d[6] * d[7] * std::pow(d[9], 2) * d[12] - d[1] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[12] -
        2 * d[4] * std::pow(d[6], 2) * d[9] * d[10] * d[12] + 4 * d[3] * d[6] * d[7] * d[9] * d[10] * d[12] +
        2 * d[4] * std::pow(d[7], 2) * d[9] * d[10] * d[12] +
        3 * d[1] * std::pow(d[6], 2) * std::pow(d[10], 2) * d[12] -
        6 * d[0] * d[6] * d[7] * std::pow(d[10], 2) * d[12] -
        3 * d[1] * std::pow(d[7], 2) * std::pow(d[10], 2) * d[12] - 4 * d[1] * d[3] * d[6] * d[9] * d[11] +
        4 * d[0] * d[4] * d[6] * d[9] * d[11] - 4 * d[0] * d[3] * d[7] * d[9] * d[11] -
        4 * d[1] * d[4] * d[7] * d[9] * d[11] + 2 * std::pow(d[0], 2) * d[7] * d[10] * d[11] +
        2 * std::pow(d[1], 2) * d[7] * d[10] * d[11] - 2 * std::pow(d[3], 2) * d[7] * d[10] * d[11] -
        2 * std::pow(d[4], 2) * d[7] * d[10] * d[11] + 2 * std::pow(d[0], 2) * d[7] * d[9] * d[12] +
        2 * std::pow(d[1], 2) * d[7] * d[9] * d[12] - 2 * std::pow(d[3], 2) * d[7] * d[9] * d[12] -
        2 * std::pow(d[4], 2) * d[7] * d[9] * d[12] - 4 * d[1] * d[3] * d[6] * d[10] * d[12] +
        4 * d[0] * d[4] * d[6] * d[10] * d[12] + 4 * d[0] * d[3] * d[7] * d[10] * d[12] +
        4 * d[1] * d[4] * d[7] * d[10] * d[12] + 2 * d[0] * d[1] * d[3] * d[11] - std::pow(d[0], 2) * d[4] * d[11] +
        std::pow(d[1], 2) * d[4] * d[11] + std::pow(d[3], 2) * d[4] * d[11] + std::pow(d[4], 3) * d[11] -
        std::pow(d[0], 2) * d[1] * d[12] - std::pow(d[1], 3) * d[12] + d[1] * std::pow(d[3], 2) * d[12] -
        2 * d[0] * d[3] * d[4] * d[12] - d[1] * std::pow(d[4], 2) * d[12];
    coeffs[71] =
        -d[4] * std::pow(d[6], 2) * std::pow(d[9], 3) + 2 * d[3] * d[6] * d[7] * std::pow(d[9], 3) +
        d[4] * std::pow(d[7], 2) * std::pow(d[9], 3) + d[1] * std::pow(d[6], 2) * std::pow(d[9], 2) * d[10] -
        2 * d[0] * d[6] * d[7] * std::pow(d[9], 2) * d[10] - d[1] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[10] -
        d[4] * std::pow(d[6], 2) * d[9] * std::pow(d[10], 2) + 2 * d[3] * d[6] * d[7] * d[9] * std::pow(d[10], 2) +
        d[4] * std::pow(d[7], 2) * d[9] * std::pow(d[10], 2) + d[1] * std::pow(d[6], 2) * std::pow(d[10], 3) -
        2 * d[0] * d[6] * d[7] * std::pow(d[10], 3) - d[1] * std::pow(d[7], 2) * std::pow(d[10], 3) -
        2 * d[1] * d[3] * d[6] * std::pow(d[9], 2) + 2 * d[0] * d[4] * d[6] * std::pow(d[9], 2) -
        2 * d[0] * d[3] * d[7] * std::pow(d[9], 2) - 2 * d[1] * d[4] * d[7] * std::pow(d[9], 2) +
        2 * std::pow(d[0], 2) * d[7] * d[9] * d[10] + 2 * std::pow(d[1], 2) * d[7] * d[9] * d[10] -
        2 * std::pow(d[3], 2) * d[7] * d[9] * d[10] - 2 * std::pow(d[4], 2) * d[7] * d[9] * d[10] -
        2 * d[1] * d[3] * d[6] * std::pow(d[10], 2) + 2 * d[0] * d[4] * d[6] * std::pow(d[10], 2) +
        2 * d[0] * d[3] * d[7] * std::pow(d[10], 2) + 2 * d[1] * d[4] * d[7] * std::pow(d[10], 2) +
        2 * d[0] * d[1] * d[3] * d[9] - std::pow(d[0], 2) * d[4] * d[9] + std::pow(d[1], 2) * d[4] * d[9] +
        std::pow(d[3], 2) * d[4] * d[9] + std::pow(d[4], 3) * d[9] - std::pow(d[0], 2) * d[1] * d[10] -
        std::pow(d[1], 3) * d[10] + d[1] * std::pow(d[3], 2) * d[10] - 2 * d[0] * d[3] * d[4] * d[10] -
        d[1] * std::pow(d[4], 2) * d[10];
    coeffs[72] = std::pow(d[5], 2) * d[8] * std::pow(d[11], 2) * d[12] -
                 2 * d[2] * d[5] * d[8] * d[11] * std::pow(d[12], 2) + std::pow(d[2], 2) * d[8] * std::pow(d[12], 3);
    coeffs[73] = std::pow(d[5], 2) * d[8] * d[10] * std::pow(d[11], 2) +
                 2 * std::pow(d[5], 2) * d[8] * d[9] * d[11] * d[12] - 4 * d[2] * d[5] * d[8] * d[10] * d[11] * d[12] -
                 2 * d[2] * d[5] * d[8] * d[9] * std::pow(d[12], 2) +
                 3 * std::pow(d[2], 2) * d[8] * d[10] * std::pow(d[12], 2) - std::pow(d[5], 3) * std::pow(d[11], 2) +
                 2 * d[2] * std::pow(d[5], 2) * d[11] * d[12] - std::pow(d[2], 2) * d[5] * std::pow(d[12], 2);
    coeffs[74] =
        2 * d[3] * d[5] * d[6] * std::pow(d[11], 2) * d[12] + 2 * d[4] * d[5] * d[7] * std::pow(d[11], 2) * d[12] -
        std::pow(d[3], 2) * d[8] * std::pow(d[11], 2) * d[12] - std::pow(d[4], 2) * d[8] * std::pow(d[11], 2) * d[12] +
        std::pow(d[8], 3) * std::pow(d[11], 2) * d[12] - 2 * d[2] * d[3] * d[6] * d[11] * std::pow(d[12], 2) -
        2 * d[0] * d[5] * d[6] * d[11] * std::pow(d[12], 2) - 2 * d[2] * d[4] * d[7] * d[11] * std::pow(d[12], 2) -
        2 * d[1] * d[5] * d[7] * d[11] * std::pow(d[12], 2) + 2 * d[0] * d[3] * d[8] * d[11] * std::pow(d[12], 2) +
        2 * d[1] * d[4] * d[8] * d[11] * std::pow(d[12], 2) + 2 * d[0] * d[2] * d[6] * std::pow(d[12], 3) +
        2 * d[1] * d[2] * d[7] * std::pow(d[12], 3) - std::pow(d[0], 2) * d[8] * std::pow(d[12], 3) -
        std::pow(d[1], 2) * d[8] * std::pow(d[12], 3) + std::pow(d[8], 3) * std::pow(d[12], 3);
    coeffs[75] = 2 * std::pow(d[5], 2) * d[8] * d[9] * d[10] * d[11] -
                 2 * d[2] * d[5] * d[8] * std::pow(d[10], 2) * d[11] +
                 std::pow(d[5], 2) * d[8] * std::pow(d[9], 2) * d[12] - 4 * d[2] * d[5] * d[8] * d[9] * d[10] * d[12] +
                 3 * std::pow(d[2], 2) * d[8] * std::pow(d[10], 2) * d[12] - 2 * std::pow(d[5], 3) * d[9] * d[11] +
                 2 * d[2] * std::pow(d[5], 2) * d[10] * d[11] + 2 * d[2] * std::pow(d[5], 2) * d[9] * d[12] -
                 2 * std::pow(d[2], 2) * d[5] * d[10] * d[12];
    coeffs[76] =
        2 * d[3] * d[5] * d[6] * d[10] * std::pow(d[11], 2) + 2 * d[4] * d[5] * d[7] * d[10] * std::pow(d[11], 2) -
        std::pow(d[3], 2) * d[8] * d[10] * std::pow(d[11], 2) - std::pow(d[4], 2) * d[8] * d[10] * std::pow(d[11], 2) +
        std::pow(d[8], 3) * d[10] * std::pow(d[11], 2) + 4 * d[3] * d[5] * d[6] * d[9] * d[11] * d[12] +
        4 * d[4] * d[5] * d[7] * d[9] * d[11] * d[12] - 2 * std::pow(d[3], 2) * d[8] * d[9] * d[11] * d[12] -
        2 * std::pow(d[4], 2) * d[8] * d[9] * d[11] * d[12] + 2 * std::pow(d[8], 3) * d[9] * d[11] * d[12] -
        4 * d[2] * d[3] * d[6] * d[10] * d[11] * d[12] - 4 * d[0] * d[5] * d[6] * d[10] * d[11] * d[12] -
        4 * d[2] * d[4] * d[7] * d[10] * d[11] * d[12] - 4 * d[1] * d[5] * d[7] * d[10] * d[11] * d[12] +
        4 * d[0] * d[3] * d[8] * d[10] * d[11] * d[12] + 4 * d[1] * d[4] * d[8] * d[10] * d[11] * d[12] -
        2 * d[2] * d[3] * d[6] * d[9] * std::pow(d[12], 2) - 2 * d[0] * d[5] * d[6] * d[9] * std::pow(d[12], 2) -
        2 * d[2] * d[4] * d[7] * d[9] * std::pow(d[12], 2) - 2 * d[1] * d[5] * d[7] * d[9] * std::pow(d[12], 2) +
        2 * d[0] * d[3] * d[8] * d[9] * std::pow(d[12], 2) + 2 * d[1] * d[4] * d[8] * d[9] * std::pow(d[12], 2) +
        6 * d[0] * d[2] * d[6] * d[10] * std::pow(d[12], 2) + 6 * d[1] * d[2] * d[7] * d[10] * std::pow(d[12], 2) -
        3 * std::pow(d[0], 2) * d[8] * d[10] * std::pow(d[12], 2) -
        3 * std::pow(d[1], 2) * d[8] * d[10] * std::pow(d[12], 2) + 3 * std::pow(d[8], 3) * d[10] * std::pow(d[12], 2) -
        std::pow(d[3], 2) * d[5] * std::pow(d[11], 2) - std::pow(d[4], 2) * d[5] * std::pow(d[11], 2) -
        d[5] * std::pow(d[8], 2) * std::pow(d[11], 2) + 2 * d[2] * std::pow(d[3], 2) * d[11] * d[12] +
        2 * d[2] * std::pow(d[4], 2) * d[11] * d[12] - 2 * d[2] * std::pow(d[8], 2) * d[11] * d[12] -
        2 * d[0] * d[2] * d[3] * std::pow(d[12], 2) - 2 * d[1] * d[2] * d[4] * std::pow(d[12], 2) +
        std::pow(d[0], 2) * d[5] * std::pow(d[12], 2) + std::pow(d[1], 2) * d[5] * std::pow(d[12], 2) -
        3 * d[5] * std::pow(d[8], 2) * std::pow(d[12], 2);
    coeffs[77] = std::pow(d[6], 2) * d[8] * std::pow(d[11], 2) * d[12] +
                 std::pow(d[7], 2) * d[8] * std::pow(d[11], 2) * d[12] + std::pow(d[6], 2) * d[8] * std::pow(d[12], 3) +
                 std::pow(d[7], 2) * d[8] * std::pow(d[12], 3);
    coeffs[78] = std::pow(d[5], 2) * d[8] * std::pow(d[9], 2) * d[10] -
                 2 * d[2] * d[5] * d[8] * d[9] * std::pow(d[10], 2) + std::pow(d[2], 2) * d[8] * std::pow(d[10], 3) -
                 std::pow(d[5], 3) * std::pow(d[9], 2) + 2 * d[2] * std::pow(d[5], 2) * d[9] * d[10] -
                 std::pow(d[2], 2) * d[5] * std::pow(d[10], 2);
    coeffs[79] =
        4 * d[3] * d[5] * d[6] * d[9] * d[10] * d[11] + 4 * d[4] * d[5] * d[7] * d[9] * d[10] * d[11] -
        2 * std::pow(d[3], 2) * d[8] * d[9] * d[10] * d[11] - 2 * std::pow(d[4], 2) * d[8] * d[9] * d[10] * d[11] +
        2 * std::pow(d[8], 3) * d[9] * d[10] * d[11] - 2 * d[2] * d[3] * d[6] * std::pow(d[10], 2) * d[11] -
        2 * d[0] * d[5] * d[6] * std::pow(d[10], 2) * d[11] - 2 * d[2] * d[4] * d[7] * std::pow(d[10], 2) * d[11] -
        2 * d[1] * d[5] * d[7] * std::pow(d[10], 2) * d[11] + 2 * d[0] * d[3] * d[8] * std::pow(d[10], 2) * d[11] +
        2 * d[1] * d[4] * d[8] * std::pow(d[10], 2) * d[11] + 2 * d[3] * d[5] * d[6] * std::pow(d[9], 2) * d[12] +
        2 * d[4] * d[5] * d[7] * std::pow(d[9], 2) * d[12] - std::pow(d[3], 2) * d[8] * std::pow(d[9], 2) * d[12] -
        std::pow(d[4], 2) * d[8] * std::pow(d[9], 2) * d[12] + std::pow(d[8], 3) * std::pow(d[9], 2) * d[12] -
        4 * d[2] * d[3] * d[6] * d[9] * d[10] * d[12] - 4 * d[0] * d[5] * d[6] * d[9] * d[10] * d[12] -
        4 * d[2] * d[4] * d[7] * d[9] * d[10] * d[12] - 4 * d[1] * d[5] * d[7] * d[9] * d[10] * d[12] +
        4 * d[0] * d[3] * d[8] * d[9] * d[10] * d[12] + 4 * d[1] * d[4] * d[8] * d[9] * d[10] * d[12] +
        6 * d[0] * d[2] * d[6] * std::pow(d[10], 2) * d[12] + 6 * d[1] * d[2] * d[7] * std::pow(d[10], 2) * d[12] -
        3 * std::pow(d[0], 2) * d[8] * std::pow(d[10], 2) * d[12] -
        3 * std::pow(d[1], 2) * d[8] * std::pow(d[10], 2) * d[12] + 3 * std::pow(d[8], 3) * std::pow(d[10], 2) * d[12] -
        2 * std::pow(d[3], 2) * d[5] * d[9] * d[11] - 2 * std::pow(d[4], 2) * d[5] * d[9] * d[11] -
        2 * d[5] * std::pow(d[8], 2) * d[9] * d[11] + 2 * d[2] * std::pow(d[3], 2) * d[10] * d[11] +
        2 * d[2] * std::pow(d[4], 2) * d[10] * d[11] - 2 * d[2] * std::pow(d[8], 2) * d[10] * d[11] +
        2 * d[2] * std::pow(d[3], 2) * d[9] * d[12] + 2 * d[2] * std::pow(d[4], 2) * d[9] * d[12] -
        2 * d[2] * std::pow(d[8], 2) * d[9] * d[12] - 4 * d[0] * d[2] * d[3] * d[10] * d[12] -
        4 * d[1] * d[2] * d[4] * d[10] * d[12] + 2 * std::pow(d[0], 2) * d[5] * d[10] * d[12] +
        2 * std::pow(d[1], 2) * d[5] * d[10] * d[12] - 6 * d[5] * std::pow(d[8], 2) * d[10] * d[12] +
        2 * d[2] * d[5] * d[8] * d[11] + std::pow(d[2], 2) * d[8] * d[12] + 3 * std::pow(d[5], 2) * d[8] * d[12];
    coeffs[80] =
        std::pow(d[6], 2) * d[8] * d[10] * std::pow(d[11], 2) + std::pow(d[7], 2) * d[8] * d[10] * std::pow(d[11], 2) +
        2 * std::pow(d[6], 2) * d[8] * d[9] * d[11] * d[12] + 2 * std::pow(d[7], 2) * d[8] * d[9] * d[11] * d[12] +
        3 * std::pow(d[6], 2) * d[8] * d[10] * std::pow(d[12], 2) +
        3 * std::pow(d[7], 2) * d[8] * d[10] * std::pow(d[12], 2) + d[5] * std::pow(d[6], 2) * std::pow(d[11], 2) +
        d[5] * std::pow(d[7], 2) * std::pow(d[11], 2) - 2 * d[3] * d[6] * d[8] * std::pow(d[11], 2) -
        2 * d[4] * d[7] * d[8] * std::pow(d[11], 2) - 2 * d[2] * std::pow(d[6], 2) * d[11] * d[12] -
        2 * d[2] * std::pow(d[7], 2) * d[11] * d[12] - d[5] * std::pow(d[6], 2) * std::pow(d[12], 2) -
        d[5] * std::pow(d[7], 2) * std::pow(d[12], 2) - 2 * d[3] * d[6] * d[8] * std::pow(d[12], 2) -
        2 * d[4] * d[7] * d[8] * std::pow(d[12], 2);
    coeffs[81] =
        2 * d[3] * d[5] * d[6] * std::pow(d[9], 2) * d[10] + 2 * d[4] * d[5] * d[7] * std::pow(d[9], 2) * d[10] -
        std::pow(d[3], 2) * d[8] * std::pow(d[9], 2) * d[10] - std::pow(d[4], 2) * d[8] * std::pow(d[9], 2) * d[10] +
        std::pow(d[8], 3) * std::pow(d[9], 2) * d[10] - 2 * d[2] * d[3] * d[6] * d[9] * std::pow(d[10], 2) -
        2 * d[0] * d[5] * d[6] * d[9] * std::pow(d[10], 2) - 2 * d[2] * d[4] * d[7] * d[9] * std::pow(d[10], 2) -
        2 * d[1] * d[5] * d[7] * d[9] * std::pow(d[10], 2) + 2 * d[0] * d[3] * d[8] * d[9] * std::pow(d[10], 2) +
        2 * d[1] * d[4] * d[8] * d[9] * std::pow(d[10], 2) + 2 * d[0] * d[2] * d[6] * std::pow(d[10], 3) +
        2 * d[1] * d[2] * d[7] * std::pow(d[10], 3) - std::pow(d[0], 2) * d[8] * std::pow(d[10], 3) -
        std::pow(d[1], 2) * d[8] * std::pow(d[10], 3) + std::pow(d[8], 3) * std::pow(d[10], 3) -
        std::pow(d[3], 2) * d[5] * std::pow(d[9], 2) - std::pow(d[4], 2) * d[5] * std::pow(d[9], 2) -
        d[5] * std::pow(d[8], 2) * std::pow(d[9], 2) + 2 * d[2] * std::pow(d[3], 2) * d[9] * d[10] +
        2 * d[2] * std::pow(d[4], 2) * d[9] * d[10] - 2 * d[2] * std::pow(d[8], 2) * d[9] * d[10] -
        2 * d[0] * d[2] * d[3] * std::pow(d[10], 2) - 2 * d[1] * d[2] * d[4] * std::pow(d[10], 2) +
        std::pow(d[0], 2) * d[5] * std::pow(d[10], 2) + std::pow(d[1], 2) * d[5] * std::pow(d[10], 2) -
        3 * d[5] * std::pow(d[8], 2) * std::pow(d[10], 2) + 2 * d[2] * d[5] * d[8] * d[9] +
        std::pow(d[2], 2) * d[8] * d[10] + 3 * std::pow(d[5], 2) * d[8] * d[10] - std::pow(d[2], 2) * d[5] -
        std::pow(d[5], 3);
    coeffs[82] =
        2 * std::pow(d[6], 2) * d[8] * d[9] * d[10] * d[11] + 2 * std::pow(d[7], 2) * d[8] * d[9] * d[10] * d[11] +
        std::pow(d[6], 2) * d[8] * std::pow(d[9], 2) * d[12] + std::pow(d[7], 2) * d[8] * std::pow(d[9], 2) * d[12] +
        3 * std::pow(d[6], 2) * d[8] * std::pow(d[10], 2) * d[12] +
        3 * std::pow(d[7], 2) * d[8] * std::pow(d[10], 2) * d[12] + 2 * d[5] * std::pow(d[6], 2) * d[9] * d[11] +
        2 * d[5] * std::pow(d[7], 2) * d[9] * d[11] - 4 * d[3] * d[6] * d[8] * d[9] * d[11] -
        4 * d[4] * d[7] * d[8] * d[9] * d[11] - 2 * d[2] * std::pow(d[6], 2) * d[10] * d[11] -
        2 * d[2] * std::pow(d[7], 2) * d[10] * d[11] - 2 * d[2] * std::pow(d[6], 2) * d[9] * d[12] -
        2 * d[2] * std::pow(d[7], 2) * d[9] * d[12] - 2 * d[5] * std::pow(d[6], 2) * d[10] * d[12] -
        2 * d[5] * std::pow(d[7], 2) * d[10] * d[12] - 4 * d[3] * d[6] * d[8] * d[10] * d[12] -
        4 * d[4] * d[7] * d[8] * d[10] * d[12] + 2 * d[2] * d[3] * d[6] * d[11] - 2 * d[0] * d[5] * d[6] * d[11] +
        2 * d[2] * d[4] * d[7] * d[11] - 2 * d[1] * d[5] * d[7] * d[11] + 2 * d[0] * d[3] * d[8] * d[11] +
        2 * d[1] * d[4] * d[8] * d[11] + 2 * d[0] * d[2] * d[6] * d[12] + 2 * d[3] * d[5] * d[6] * d[12] +
        2 * d[1] * d[2] * d[7] * d[12] + 2 * d[4] * d[5] * d[7] * d[12] - std::pow(d[0], 2) * d[8] * d[12] -
        std::pow(d[1], 2) * d[8] * d[12] + std::pow(d[3], 2) * d[8] * d[12] + std::pow(d[4], 2) * d[8] * d[12];
    coeffs[83] = std::pow(d[6], 2) * d[8] * std::pow(d[9], 2) * d[10] +
                 std::pow(d[7], 2) * d[8] * std::pow(d[9], 2) * d[10] + std::pow(d[6], 2) * d[8] * std::pow(d[10], 3) +
                 std::pow(d[7], 2) * d[8] * std::pow(d[10], 3) + d[5] * std::pow(d[6], 2) * std::pow(d[9], 2) +
                 d[5] * std::pow(d[7], 2) * std::pow(d[9], 2) - 2 * d[3] * d[6] * d[8] * std::pow(d[9], 2) -
                 2 * d[4] * d[7] * d[8] * std::pow(d[9], 2) - 2 * d[2] * std::pow(d[6], 2) * d[9] * d[10] -
                 2 * d[2] * std::pow(d[7], 2) * d[9] * d[10] - d[5] * std::pow(d[6], 2) * std::pow(d[10], 2) -
                 d[5] * std::pow(d[7], 2) * std::pow(d[10], 2) - 2 * d[3] * d[6] * d[8] * std::pow(d[10], 2) -
                 2 * d[4] * d[7] * d[8] * std::pow(d[10], 2) + 2 * d[2] * d[3] * d[6] * d[9] -
                 2 * d[0] * d[5] * d[6] * d[9] + 2 * d[2] * d[4] * d[7] * d[9] - 2 * d[1] * d[5] * d[7] * d[9] +
                 2 * d[0] * d[3] * d[8] * d[9] + 2 * d[1] * d[4] * d[8] * d[9] + 2 * d[0] * d[2] * d[6] * d[10] +
                 2 * d[3] * d[5] * d[6] * d[10] + 2 * d[1] * d[2] * d[7] * d[10] + 2 * d[4] * d[5] * d[7] * d[10] -
                 std::pow(d[0], 2) * d[8] * d[10] - std::pow(d[1], 2) * d[8] * d[10] +
                 std::pow(d[3], 2) * d[8] * d[10] + std::pow(d[4], 2) * d[8] * d[10] - 2 * d[0] * d[2] * d[3] -
                 2 * d[1] * d[2] * d[4] + std::pow(d[0], 2) * d[5] + std::pow(d[1], 2) * d[5] -
                 std::pow(d[3], 2) * d[5] - std::pow(d[4], 2) * d[5];
    coeffs[84] = -std::pow(d[5], 2) * d[8] * std::pow(d[11], 3) + 2 * d[2] * d[5] * d[8] * std::pow(d[11], 2) * d[12] -
                 std::pow(d[2], 2) * d[8] * d[11] * std::pow(d[12], 2);
    coeffs[85] = -3 * std::pow(d[5], 2) * d[8] * d[9] * std::pow(d[11], 2) +
                 2 * d[2] * d[5] * d[8] * d[10] * std::pow(d[11], 2) + 4 * d[2] * d[5] * d[8] * d[9] * d[11] * d[12] -
                 2 * std::pow(d[2], 2) * d[8] * d[10] * d[11] * d[12] -
                 std::pow(d[2], 2) * d[8] * d[9] * std::pow(d[12], 2) + d[2] * std::pow(d[5], 2) * std::pow(d[11], 2) -
                 2 * std::pow(d[2], 2) * d[5] * d[11] * d[12] + std::pow(d[2], 3) * std::pow(d[12], 2);
    coeffs[86] =
        -2 * d[3] * d[5] * d[6] * std::pow(d[11], 3) - 2 * d[4] * d[5] * d[7] * std::pow(d[11], 3) +
        std::pow(d[3], 2) * d[8] * std::pow(d[11], 3) + std::pow(d[4], 2) * d[8] * std::pow(d[11], 3) -
        std::pow(d[8], 3) * std::pow(d[11], 3) + 2 * d[2] * d[3] * d[6] * std::pow(d[11], 2) * d[12] +
        2 * d[0] * d[5] * d[6] * std::pow(d[11], 2) * d[12] + 2 * d[2] * d[4] * d[7] * std::pow(d[11], 2) * d[12] +
        2 * d[1] * d[5] * d[7] * std::pow(d[11], 2) * d[12] - 2 * d[0] * d[3] * d[8] * std::pow(d[11], 2) * d[12] -
        2 * d[1] * d[4] * d[8] * std::pow(d[11], 2) * d[12] - 2 * d[0] * d[2] * d[6] * d[11] * std::pow(d[12], 2) -
        2 * d[1] * d[2] * d[7] * d[11] * std::pow(d[12], 2) + std::pow(d[0], 2) * d[8] * d[11] * std::pow(d[12], 2) +
        std::pow(d[1], 2) * d[8] * d[11] * std::pow(d[12], 2) - std::pow(d[8], 3) * d[11] * std::pow(d[12], 2);
    coeffs[87] = -3 * std::pow(d[5], 2) * d[8] * std::pow(d[9], 2) * d[11] +
                 4 * d[2] * d[5] * d[8] * d[9] * d[10] * d[11] - std::pow(d[2], 2) * d[8] * std::pow(d[10], 2) * d[11] +
                 2 * d[2] * d[5] * d[8] * std::pow(d[9], 2) * d[12] -
                 2 * std::pow(d[2], 2) * d[8] * d[9] * d[10] * d[12] + 2 * d[2] * std::pow(d[5], 2) * d[9] * d[11] -
                 2 * std::pow(d[2], 2) * d[5] * d[10] * d[11] - 2 * std::pow(d[2], 2) * d[5] * d[9] * d[12] +
                 2 * std::pow(d[2], 3) * d[10] * d[12];
    coeffs[88] =
        -6 * d[3] * d[5] * d[6] * d[9] * std::pow(d[11], 2) - 6 * d[4] * d[5] * d[7] * d[9] * std::pow(d[11], 2) +
        3 * std::pow(d[3], 2) * d[8] * d[9] * std::pow(d[11], 2) +
        3 * std::pow(d[4], 2) * d[8] * d[9] * std::pow(d[11], 2) - 3 * std::pow(d[8], 3) * d[9] * std::pow(d[11], 2) +
        2 * d[2] * d[3] * d[6] * d[10] * std::pow(d[11], 2) + 2 * d[0] * d[5] * d[6] * d[10] * std::pow(d[11], 2) +
        2 * d[2] * d[4] * d[7] * d[10] * std::pow(d[11], 2) + 2 * d[1] * d[5] * d[7] * d[10] * std::pow(d[11], 2) -
        2 * d[0] * d[3] * d[8] * d[10] * std::pow(d[11], 2) - 2 * d[1] * d[4] * d[8] * d[10] * std::pow(d[11], 2) +
        4 * d[2] * d[3] * d[6] * d[9] * d[11] * d[12] + 4 * d[0] * d[5] * d[6] * d[9] * d[11] * d[12] +
        4 * d[2] * d[4] * d[7] * d[9] * d[11] * d[12] + 4 * d[1] * d[5] * d[7] * d[9] * d[11] * d[12] -
        4 * d[0] * d[3] * d[8] * d[9] * d[11] * d[12] - 4 * d[1] * d[4] * d[8] * d[9] * d[11] * d[12] -
        4 * d[0] * d[2] * d[6] * d[10] * d[11] * d[12] - 4 * d[1] * d[2] * d[7] * d[10] * d[11] * d[12] +
        2 * std::pow(d[0], 2) * d[8] * d[10] * d[11] * d[12] + 2 * std::pow(d[1], 2) * d[8] * d[10] * d[11] * d[12] -
        2 * std::pow(d[8], 3) * d[10] * d[11] * d[12] - 2 * d[0] * d[2] * d[6] * d[9] * std::pow(d[12], 2) -
        2 * d[1] * d[2] * d[7] * d[9] * std::pow(d[12], 2) + std::pow(d[0], 2) * d[8] * d[9] * std::pow(d[12], 2) +
        std::pow(d[1], 2) * d[8] * d[9] * std::pow(d[12], 2) - std::pow(d[8], 3) * d[9] * std::pow(d[12], 2) -
        d[2] * std::pow(d[3], 2) * std::pow(d[11], 2) - d[2] * std::pow(d[4], 2) * std::pow(d[11], 2) +
        2 * d[0] * d[3] * d[5] * std::pow(d[11], 2) + 2 * d[1] * d[4] * d[5] * std::pow(d[11], 2) +
        3 * d[2] * std::pow(d[8], 2) * std::pow(d[11], 2) - 2 * std::pow(d[0], 2) * d[5] * d[11] * d[12] -
        2 * std::pow(d[1], 2) * d[5] * d[11] * d[12] + 2 * d[5] * std::pow(d[8], 2) * d[11] * d[12] +
        std::pow(d[0], 2) * d[2] * std::pow(d[12], 2) + std::pow(d[1], 2) * d[2] * std::pow(d[12], 2) +
        d[2] * std::pow(d[8], 2) * std::pow(d[12], 2);
    coeffs[89] = -std::pow(d[6], 2) * d[8] * std::pow(d[11], 3) - std::pow(d[7], 2) * d[8] * std::pow(d[11], 3) -
                 std::pow(d[6], 2) * d[8] * d[11] * std::pow(d[12], 2) -
                 std::pow(d[7], 2) * d[8] * d[11] * std::pow(d[12], 2);
    coeffs[90] = -std::pow(d[5], 2) * d[8] * std::pow(d[9], 3) + 2 * d[2] * d[5] * d[8] * std::pow(d[9], 2) * d[10] -
                 std::pow(d[2], 2) * d[8] * d[9] * std::pow(d[10], 2) + d[2] * std::pow(d[5], 2) * std::pow(d[9], 2) -
                 2 * std::pow(d[2], 2) * d[5] * d[9] * d[10] + std::pow(d[2], 3) * std::pow(d[10], 2);
    coeffs[91] =
        -6 * d[3] * d[5] * d[6] * std::pow(d[9], 2) * d[11] - 6 * d[4] * d[5] * d[7] * std::pow(d[9], 2) * d[11] +
        3 * std::pow(d[3], 2) * d[8] * std::pow(d[9], 2) * d[11] +
        3 * std::pow(d[4], 2) * d[8] * std::pow(d[9], 2) * d[11] - 3 * std::pow(d[8], 3) * std::pow(d[9], 2) * d[11] +
        4 * d[2] * d[3] * d[6] * d[9] * d[10] * d[11] + 4 * d[0] * d[5] * d[6] * d[9] * d[10] * d[11] +
        4 * d[2] * d[4] * d[7] * d[9] * d[10] * d[11] + 4 * d[1] * d[5] * d[7] * d[9] * d[10] * d[11] -
        4 * d[0] * d[3] * d[8] * d[9] * d[10] * d[11] - 4 * d[1] * d[4] * d[8] * d[9] * d[10] * d[11] -
        2 * d[0] * d[2] * d[6] * std::pow(d[10], 2) * d[11] - 2 * d[1] * d[2] * d[7] * std::pow(d[10], 2) * d[11] +
        std::pow(d[0], 2) * d[8] * std::pow(d[10], 2) * d[11] + std::pow(d[1], 2) * d[8] * std::pow(d[10], 2) * d[11] -
        std::pow(d[8], 3) * std::pow(d[10], 2) * d[11] + 2 * d[2] * d[3] * d[6] * std::pow(d[9], 2) * d[12] +
        2 * d[0] * d[5] * d[6] * std::pow(d[9], 2) * d[12] + 2 * d[2] * d[4] * d[7] * std::pow(d[9], 2) * d[12] +
        2 * d[1] * d[5] * d[7] * std::pow(d[9], 2) * d[12] - 2 * d[0] * d[3] * d[8] * std::pow(d[9], 2) * d[12] -
        2 * d[1] * d[4] * d[8] * std::pow(d[9], 2) * d[12] - 4 * d[0] * d[2] * d[6] * d[9] * d[10] * d[12] -
        4 * d[1] * d[2] * d[7] * d[9] * d[10] * d[12] + 2 * std::pow(d[0], 2) * d[8] * d[9] * d[10] * d[12] +
        2 * std::pow(d[1], 2) * d[8] * d[9] * d[10] * d[12] - 2 * std::pow(d[8], 3) * d[9] * d[10] * d[12] -
        2 * d[2] * std::pow(d[3], 2) * d[9] * d[11] - 2 * d[2] * std::pow(d[4], 2) * d[9] * d[11] +
        4 * d[0] * d[3] * d[5] * d[9] * d[11] + 4 * d[1] * d[4] * d[5] * d[9] * d[11] +
        6 * d[2] * std::pow(d[8], 2) * d[9] * d[11] - 2 * std::pow(d[0], 2) * d[5] * d[10] * d[11] -
        2 * std::pow(d[1], 2) * d[5] * d[10] * d[11] + 2 * d[5] * std::pow(d[8], 2) * d[10] * d[11] -
        2 * std::pow(d[0], 2) * d[5] * d[9] * d[12] - 2 * std::pow(d[1], 2) * d[5] * d[9] * d[12] +
        2 * d[5] * std::pow(d[8], 2) * d[9] * d[12] + 2 * std::pow(d[0], 2) * d[2] * d[10] * d[12] +
        2 * std::pow(d[1], 2) * d[2] * d[10] * d[12] + 2 * d[2] * std::pow(d[8], 2) * d[10] * d[12] -
        3 * std::pow(d[2], 2) * d[8] * d[11] - std::pow(d[5], 2) * d[8] * d[11] - 2 * d[2] * d[5] * d[8] * d[12];
    coeffs[92] =
        -3 * std::pow(d[6], 2) * d[8] * d[9] * std::pow(d[11], 2) -
        3 * std::pow(d[7], 2) * d[8] * d[9] * std::pow(d[11], 2) -
        2 * std::pow(d[6], 2) * d[8] * d[10] * d[11] * d[12] - 2 * std::pow(d[7], 2) * d[8] * d[10] * d[11] * d[12] -
        std::pow(d[6], 2) * d[8] * d[9] * std::pow(d[12], 2) - std::pow(d[7], 2) * d[8] * d[9] * std::pow(d[12], 2) +
        d[2] * std::pow(d[6], 2) * std::pow(d[11], 2) + d[2] * std::pow(d[7], 2) * std::pow(d[11], 2) +
        2 * d[0] * d[6] * d[8] * std::pow(d[11], 2) + 2 * d[1] * d[7] * d[8] * std::pow(d[11], 2) +
        2 * d[5] * std::pow(d[6], 2) * d[11] * d[12] + 2 * d[5] * std::pow(d[7], 2) * d[11] * d[12] -
        d[2] * std::pow(d[6], 2) * std::pow(d[12], 2) - d[2] * std::pow(d[7], 2) * std::pow(d[12], 2) +
        2 * d[0] * d[6] * d[8] * std::pow(d[12], 2) + 2 * d[1] * d[7] * d[8] * std::pow(d[12], 2);
    coeffs[93] =
        -2 * d[3] * d[5] * d[6] * std::pow(d[9], 3) - 2 * d[4] * d[5] * d[7] * std::pow(d[9], 3) +
        std::pow(d[3], 2) * d[8] * std::pow(d[9], 3) + std::pow(d[4], 2) * d[8] * std::pow(d[9], 3) -
        std::pow(d[8], 3) * std::pow(d[9], 3) + 2 * d[2] * d[3] * d[6] * std::pow(d[9], 2) * d[10] +
        2 * d[0] * d[5] * d[6] * std::pow(d[9], 2) * d[10] + 2 * d[2] * d[4] * d[7] * std::pow(d[9], 2) * d[10] +
        2 * d[1] * d[5] * d[7] * std::pow(d[9], 2) * d[10] - 2 * d[0] * d[3] * d[8] * std::pow(d[9], 2) * d[10] -
        2 * d[1] * d[4] * d[8] * std::pow(d[9], 2) * d[10] - 2 * d[0] * d[2] * d[6] * d[9] * std::pow(d[10], 2) -
        2 * d[1] * d[2] * d[7] * d[9] * std::pow(d[10], 2) + std::pow(d[0], 2) * d[8] * d[9] * std::pow(d[10], 2) +
        std::pow(d[1], 2) * d[8] * d[9] * std::pow(d[10], 2) - std::pow(d[8], 3) * d[9] * std::pow(d[10], 2) -
        d[2] * std::pow(d[3], 2) * std::pow(d[9], 2) - d[2] * std::pow(d[4], 2) * std::pow(d[9], 2) +
        2 * d[0] * d[3] * d[5] * std::pow(d[9], 2) + 2 * d[1] * d[4] * d[5] * std::pow(d[9], 2) +
        3 * d[2] * std::pow(d[8], 2) * std::pow(d[9], 2) - 2 * std::pow(d[0], 2) * d[5] * d[9] * d[10] -
        2 * std::pow(d[1], 2) * d[5] * d[9] * d[10] + 2 * d[5] * std::pow(d[8], 2) * d[9] * d[10] +
        std::pow(d[0], 2) * d[2] * std::pow(d[10], 2) + std::pow(d[1], 2) * d[2] * std::pow(d[10], 2) +
        d[2] * std::pow(d[8], 2) * std::pow(d[10], 2) - 3 * std::pow(d[2], 2) * d[8] * d[9] -
        std::pow(d[5], 2) * d[8] * d[9] - 2 * d[2] * d[5] * d[8] * d[10] + std::pow(d[2], 3) + d[2] * std::pow(d[5], 2);
    coeffs[94] =
        -3 * std::pow(d[6], 2) * d[8] * std::pow(d[9], 2) * d[11] -
        3 * std::pow(d[7], 2) * d[8] * std::pow(d[9], 2) * d[11] -
        std::pow(d[6], 2) * d[8] * std::pow(d[10], 2) * d[11] - std::pow(d[7], 2) * d[8] * std::pow(d[10], 2) * d[11] -
        2 * std::pow(d[6], 2) * d[8] * d[9] * d[10] * d[12] - 2 * std::pow(d[7], 2) * d[8] * d[9] * d[10] * d[12] +
        2 * d[2] * std::pow(d[6], 2) * d[9] * d[11] + 2 * d[2] * std::pow(d[7], 2) * d[9] * d[11] +
        4 * d[0] * d[6] * d[8] * d[9] * d[11] + 4 * d[1] * d[7] * d[8] * d[9] * d[11] +
        2 * d[5] * std::pow(d[6], 2) * d[10] * d[11] + 2 * d[5] * std::pow(d[7], 2) * d[10] * d[11] +
        2 * d[5] * std::pow(d[6], 2) * d[9] * d[12] + 2 * d[5] * std::pow(d[7], 2) * d[9] * d[12] -
        2 * d[2] * std::pow(d[6], 2) * d[10] * d[12] - 2 * d[2] * std::pow(d[7], 2) * d[10] * d[12] +
        4 * d[0] * d[6] * d[8] * d[10] * d[12] + 4 * d[1] * d[7] * d[8] * d[10] * d[12] -
        2 * d[0] * d[2] * d[6] * d[11] - 2 * d[3] * d[5] * d[6] * d[11] - 2 * d[1] * d[2] * d[7] * d[11] -
        2 * d[4] * d[5] * d[7] * d[11] - std::pow(d[0], 2) * d[8] * d[11] - std::pow(d[1], 2) * d[8] * d[11] +
        std::pow(d[3], 2) * d[8] * d[11] + std::pow(d[4], 2) * d[8] * d[11] + 2 * d[2] * d[3] * d[6] * d[12] -
        2 * d[0] * d[5] * d[6] * d[12] + 2 * d[2] * d[4] * d[7] * d[12] - 2 * d[1] * d[5] * d[7] * d[12] -
        2 * d[0] * d[3] * d[8] * d[12] - 2 * d[1] * d[4] * d[8] * d[12];
    coeffs[95] = -std::pow(d[6], 2) * d[8] * std::pow(d[9], 3) - std::pow(d[7], 2) * d[8] * std::pow(d[9], 3) -
                 std::pow(d[6], 2) * d[8] * d[9] * std::pow(d[10], 2) -
                 std::pow(d[7], 2) * d[8] * d[9] * std::pow(d[10], 2) + d[2] * std::pow(d[6], 2) * std::pow(d[9], 2) +
                 d[2] * std::pow(d[7], 2) * std::pow(d[9], 2) + 2 * d[0] * d[6] * d[8] * std::pow(d[9], 2) +
                 2 * d[1] * d[7] * d[8] * std::pow(d[9], 2) + 2 * d[5] * std::pow(d[6], 2) * d[9] * d[10] +
                 2 * d[5] * std::pow(d[7], 2) * d[9] * d[10] - d[2] * std::pow(d[6], 2) * std::pow(d[10], 2) -
                 d[2] * std::pow(d[7], 2) * std::pow(d[10], 2) + 2 * d[0] * d[6] * d[8] * std::pow(d[10], 2) +
                 2 * d[1] * d[7] * d[8] * std::pow(d[10], 2) - 2 * d[0] * d[2] * d[6] * d[9] -
                 2 * d[3] * d[5] * d[6] * d[9] - 2 * d[1] * d[2] * d[7] * d[9] - 2 * d[4] * d[5] * d[7] * d[9] -
                 std::pow(d[0], 2) * d[8] * d[9] - std::pow(d[1], 2) * d[8] * d[9] + std::pow(d[3], 2) * d[8] * d[9] +
                 std::pow(d[4], 2) * d[8] * d[9] + 2 * d[2] * d[3] * d[6] * d[10] - 2 * d[0] * d[5] * d[6] * d[10] +
                 2 * d[2] * d[4] * d[7] * d[10] - 2 * d[1] * d[5] * d[7] * d[10] - 2 * d[0] * d[3] * d[8] * d[10] -
                 2 * d[1] * d[4] * d[8] * d[10] + std::pow(d[0], 2) * d[2] + std::pow(d[1], 2) * d[2] -
                 d[2] * std::pow(d[3], 2) - d[2] * std::pow(d[4], 2) + 2 * d[0] * d[3] * d[5] + 2 * d[1] * d[4] * d[5];
    coeffs[96] = std::pow(d[5], 3) * std::pow(d[11], 3) - 3 * d[2] * std::pow(d[5], 2) * std::pow(d[11], 2) * d[12] +
                 3 * std::pow(d[2], 2) * d[5] * d[11] * std::pow(d[12], 2) - std::pow(d[2], 3) * std::pow(d[12], 3);
    coeffs[97] =
        3 * std::pow(d[5], 3) * d[9] * std::pow(d[11], 2) - 3 * d[2] * std::pow(d[5], 2) * d[10] * std::pow(d[11], 2) -
        6 * d[2] * std::pow(d[5], 2) * d[9] * d[11] * d[12] + 6 * std::pow(d[2], 2) * d[5] * d[10] * d[11] * d[12] +
        3 * std::pow(d[2], 2) * d[5] * d[9] * std::pow(d[12], 2) - 3 * std::pow(d[2], 3) * d[10] * std::pow(d[12], 2);
    coeffs[98] =
        std::pow(d[3], 2) * d[5] * std::pow(d[11], 3) + std::pow(d[4], 2) * d[5] * std::pow(d[11], 3) +
        d[5] * std::pow(d[8], 2) * std::pow(d[11], 3) - d[2] * std::pow(d[3], 2) * std::pow(d[11], 2) * d[12] -
        d[2] * std::pow(d[4], 2) * std::pow(d[11], 2) * d[12] - 2 * d[0] * d[3] * d[5] * std::pow(d[11], 2) * d[12] -
        2 * d[1] * d[4] * d[5] * std::pow(d[11], 2) * d[12] - d[2] * std::pow(d[8], 2) * std::pow(d[11], 2) * d[12] +
        2 * d[0] * d[2] * d[3] * d[11] * std::pow(d[12], 2) + 2 * d[1] * d[2] * d[4] * d[11] * std::pow(d[12], 2) +
        std::pow(d[0], 2) * d[5] * d[11] * std::pow(d[12], 2) + std::pow(d[1], 2) * d[5] * d[11] * std::pow(d[12], 2) +
        d[5] * std::pow(d[8], 2) * d[11] * std::pow(d[12], 2) - std::pow(d[0], 2) * d[2] * std::pow(d[12], 3) -
        std::pow(d[1], 2) * d[2] * std::pow(d[12], 3) - d[2] * std::pow(d[8], 2) * std::pow(d[12], 3);
    coeffs[99] =
        3 * std::pow(d[5], 3) * std::pow(d[9], 2) * d[11] - 6 * d[2] * std::pow(d[5], 2) * d[9] * d[10] * d[11] +
        3 * std::pow(d[2], 2) * d[5] * std::pow(d[10], 2) * d[11] -
        3 * d[2] * std::pow(d[5], 2) * std::pow(d[9], 2) * d[12] + 6 * std::pow(d[2], 2) * d[5] * d[9] * d[10] * d[12] -
        3 * std::pow(d[2], 3) * std::pow(d[10], 2) * d[12];
    coeffs[100] =
        3 * std::pow(d[3], 2) * d[5] * d[9] * std::pow(d[11], 2) +
        3 * std::pow(d[4], 2) * d[5] * d[9] * std::pow(d[11], 2) +
        3 * d[5] * std::pow(d[8], 2) * d[9] * std::pow(d[11], 2) -
        d[2] * std::pow(d[3], 2) * d[10] * std::pow(d[11], 2) - d[2] * std::pow(d[4], 2) * d[10] * std::pow(d[11], 2) -
        2 * d[0] * d[3] * d[5] * d[10] * std::pow(d[11], 2) - 2 * d[1] * d[4] * d[5] * d[10] * std::pow(d[11], 2) -
        d[2] * std::pow(d[8], 2) * d[10] * std::pow(d[11], 2) - 2 * d[2] * std::pow(d[3], 2) * d[9] * d[11] * d[12] -
        2 * d[2] * std::pow(d[4], 2) * d[9] * d[11] * d[12] - 4 * d[0] * d[3] * d[5] * d[9] * d[11] * d[12] -
        4 * d[1] * d[4] * d[5] * d[9] * d[11] * d[12] - 2 * d[2] * std::pow(d[8], 2) * d[9] * d[11] * d[12] +
        4 * d[0] * d[2] * d[3] * d[10] * d[11] * d[12] + 4 * d[1] * d[2] * d[4] * d[10] * d[11] * d[12] +
        2 * std::pow(d[0], 2) * d[5] * d[10] * d[11] * d[12] + 2 * std::pow(d[1], 2) * d[5] * d[10] * d[11] * d[12] +
        2 * d[5] * std::pow(d[8], 2) * d[10] * d[11] * d[12] + 2 * d[0] * d[2] * d[3] * d[9] * std::pow(d[12], 2) +
        2 * d[1] * d[2] * d[4] * d[9] * std::pow(d[12], 2) + std::pow(d[0], 2) * d[5] * d[9] * std::pow(d[12], 2) +
        std::pow(d[1], 2) * d[5] * d[9] * std::pow(d[12], 2) + d[5] * std::pow(d[8], 2) * d[9] * std::pow(d[12], 2) -
        3 * std::pow(d[0], 2) * d[2] * d[10] * std::pow(d[12], 2) -
        3 * std::pow(d[1], 2) * d[2] * d[10] * std::pow(d[12], 2) -
        3 * d[2] * std::pow(d[8], 2) * d[10] * std::pow(d[12], 2) - 2 * d[2] * d[5] * d[8] * std::pow(d[11], 2) +
        2 * std::pow(d[2], 2) * d[8] * d[11] * d[12] - 2 * std::pow(d[5], 2) * d[8] * d[11] * d[12] +
        2 * d[2] * d[5] * d[8] * std::pow(d[12], 2);
    coeffs[101] =
        -d[5] * std::pow(d[6], 2) * std::pow(d[11], 3) - d[5] * std::pow(d[7], 2) * std::pow(d[11], 3) +
        2 * d[3] * d[6] * d[8] * std::pow(d[11], 3) + 2 * d[4] * d[7] * d[8] * std::pow(d[11], 3) +
        d[2] * std::pow(d[6], 2) * std::pow(d[11], 2) * d[12] + d[2] * std::pow(d[7], 2) * std::pow(d[11], 2) * d[12] -
        2 * d[0] * d[6] * d[8] * std::pow(d[11], 2) * d[12] - 2 * d[1] * d[7] * d[8] * std::pow(d[11], 2) * d[12] -
        d[5] * std::pow(d[6], 2) * d[11] * std::pow(d[12], 2) - d[5] * std::pow(d[7], 2) * d[11] * std::pow(d[12], 2) +
        2 * d[3] * d[6] * d[8] * d[11] * std::pow(d[12], 2) + 2 * d[4] * d[7] * d[8] * d[11] * std::pow(d[12], 2) +
        d[2] * std::pow(d[6], 2) * std::pow(d[12], 3) + d[2] * std::pow(d[7], 2) * std::pow(d[12], 3) -
        2 * d[0] * d[6] * d[8] * std::pow(d[12], 3) - 2 * d[1] * d[7] * d[8] * std::pow(d[12], 3);
    coeffs[102] = std::pow(d[5], 3) * std::pow(d[9], 3) - 3 * d[2] * std::pow(d[5], 2) * std::pow(d[9], 2) * d[10] +
                  3 * std::pow(d[2], 2) * d[5] * d[9] * std::pow(d[10], 2) - std::pow(d[2], 3) * std::pow(d[10], 3);
    coeffs[103] =
        3 * std::pow(d[3], 2) * d[5] * std::pow(d[9], 2) * d[11] +
        3 * std::pow(d[4], 2) * d[5] * std::pow(d[9], 2) * d[11] +
        3 * d[5] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[11] - 2 * d[2] * std::pow(d[3], 2) * d[9] * d[10] * d[11] -
        2 * d[2] * std::pow(d[4], 2) * d[9] * d[10] * d[11] - 4 * d[0] * d[3] * d[5] * d[9] * d[10] * d[11] -
        4 * d[1] * d[4] * d[5] * d[9] * d[10] * d[11] - 2 * d[2] * std::pow(d[8], 2) * d[9] * d[10] * d[11] +
        2 * d[0] * d[2] * d[3] * std::pow(d[10], 2) * d[11] + 2 * d[1] * d[2] * d[4] * std::pow(d[10], 2) * d[11] +
        std::pow(d[0], 2) * d[5] * std::pow(d[10], 2) * d[11] + std::pow(d[1], 2) * d[5] * std::pow(d[10], 2) * d[11] +
        d[5] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[11] - d[2] * std::pow(d[3], 2) * std::pow(d[9], 2) * d[12] -
        d[2] * std::pow(d[4], 2) * std::pow(d[9], 2) * d[12] - 2 * d[0] * d[3] * d[5] * std::pow(d[9], 2) * d[12] -
        2 * d[1] * d[4] * d[5] * std::pow(d[9], 2) * d[12] - d[2] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[12] +
        4 * d[0] * d[2] * d[3] * d[9] * d[10] * d[12] + 4 * d[1] * d[2] * d[4] * d[9] * d[10] * d[12] +
        2 * std::pow(d[0], 2) * d[5] * d[9] * d[10] * d[12] + 2 * std::pow(d[1], 2) * d[5] * d[9] * d[10] * d[12] +
        2 * d[5] * std::pow(d[8], 2) * d[9] * d[10] * d[12] -
        3 * std::pow(d[0], 2) * d[2] * std::pow(d[10], 2) * d[12] -
        3 * std::pow(d[1], 2) * d[2] * std::pow(d[10], 2) * d[12] -
        3 * d[2] * std::pow(d[8], 2) * std::pow(d[10], 2) * d[12] - 4 * d[2] * d[5] * d[8] * d[9] * d[11] +
        2 * std::pow(d[2], 2) * d[8] * d[10] * d[11] - 2 * std::pow(d[5], 2) * d[8] * d[10] * d[11] +
        2 * std::pow(d[2], 2) * d[8] * d[9] * d[12] - 2 * std::pow(d[5], 2) * d[8] * d[9] * d[12] +
        4 * d[2] * d[5] * d[8] * d[10] * d[12] + std::pow(d[2], 2) * d[5] * d[11] + std::pow(d[5], 3) * d[11] -
        std::pow(d[2], 3) * d[12] - d[2] * std::pow(d[5], 2) * d[12];
    coeffs[104] =
        -3 * d[5] * std::pow(d[6], 2) * d[9] * std::pow(d[11], 2) -
        3 * d[5] * std::pow(d[7], 2) * d[9] * std::pow(d[11], 2) + 6 * d[3] * d[6] * d[8] * d[9] * std::pow(d[11], 2) +
        6 * d[4] * d[7] * d[8] * d[9] * std::pow(d[11], 2) + d[2] * std::pow(d[6], 2) * d[10] * std::pow(d[11], 2) +
        d[2] * std::pow(d[7], 2) * d[10] * std::pow(d[11], 2) - 2 * d[0] * d[6] * d[8] * d[10] * std::pow(d[11], 2) -
        2 * d[1] * d[7] * d[8] * d[10] * std::pow(d[11], 2) + 2 * d[2] * std::pow(d[6], 2) * d[9] * d[11] * d[12] +
        2 * d[2] * std::pow(d[7], 2) * d[9] * d[11] * d[12] - 4 * d[0] * d[6] * d[8] * d[9] * d[11] * d[12] -
        4 * d[1] * d[7] * d[8] * d[9] * d[11] * d[12] - 2 * d[5] * std::pow(d[6], 2) * d[10] * d[11] * d[12] -
        2 * d[5] * std::pow(d[7], 2) * d[10] * d[11] * d[12] + 4 * d[3] * d[6] * d[8] * d[10] * d[11] * d[12] +
        4 * d[4] * d[7] * d[8] * d[10] * d[11] * d[12] - d[5] * std::pow(d[6], 2) * d[9] * std::pow(d[12], 2) -
        d[5] * std::pow(d[7], 2) * d[9] * std::pow(d[12], 2) + 2 * d[3] * d[6] * d[8] * d[9] * std::pow(d[12], 2) +
        2 * d[4] * d[7] * d[8] * d[9] * std::pow(d[12], 2) + 3 * d[2] * std::pow(d[6], 2) * d[10] * std::pow(d[12], 2) +
        3 * d[2] * std::pow(d[7], 2) * d[10] * std::pow(d[12], 2) -
        6 * d[0] * d[6] * d[8] * d[10] * std::pow(d[12], 2) - 6 * d[1] * d[7] * d[8] * d[10] * std::pow(d[12], 2) -
        2 * d[2] * d[3] * d[6] * std::pow(d[11], 2) + 2 * d[0] * d[5] * d[6] * std::pow(d[11], 2) -
        2 * d[2] * d[4] * d[7] * std::pow(d[11], 2) + 2 * d[1] * d[5] * d[7] * std::pow(d[11], 2) -
        2 * d[0] * d[3] * d[8] * std::pow(d[11], 2) - 2 * d[1] * d[4] * d[8] * std::pow(d[11], 2) +
        2 * std::pow(d[0], 2) * d[8] * d[11] * d[12] + 2 * std::pow(d[1], 2) * d[8] * d[11] * d[12] -
        2 * std::pow(d[3], 2) * d[8] * d[11] * d[12] - 2 * std::pow(d[4], 2) * d[8] * d[11] * d[12] -
        2 * d[2] * d[3] * d[6] * std::pow(d[12], 2) + 2 * d[0] * d[5] * d[6] * std::pow(d[12], 2) -
        2 * d[2] * d[4] * d[7] * std::pow(d[12], 2) + 2 * d[1] * d[5] * d[7] * std::pow(d[12], 2) +
        2 * d[0] * d[3] * d[8] * std::pow(d[12], 2) + 2 * d[1] * d[4] * d[8] * std::pow(d[12], 2);
    coeffs[105] =
        std::pow(d[3], 2) * d[5] * std::pow(d[9], 3) + std::pow(d[4], 2) * d[5] * std::pow(d[9], 3) +
        d[5] * std::pow(d[8], 2) * std::pow(d[9], 3) - d[2] * std::pow(d[3], 2) * std::pow(d[9], 2) * d[10] -
        d[2] * std::pow(d[4], 2) * std::pow(d[9], 2) * d[10] - 2 * d[0] * d[3] * d[5] * std::pow(d[9], 2) * d[10] -
        2 * d[1] * d[4] * d[5] * std::pow(d[9], 2) * d[10] - d[2] * std::pow(d[8], 2) * std::pow(d[9], 2) * d[10] +
        2 * d[0] * d[2] * d[3] * d[9] * std::pow(d[10], 2) + 2 * d[1] * d[2] * d[4] * d[9] * std::pow(d[10], 2) +
        std::pow(d[0], 2) * d[5] * d[9] * std::pow(d[10], 2) + std::pow(d[1], 2) * d[5] * d[9] * std::pow(d[10], 2) +
        d[5] * std::pow(d[8], 2) * d[9] * std::pow(d[10], 2) - std::pow(d[0], 2) * d[2] * std::pow(d[10], 3) -
        std::pow(d[1], 2) * d[2] * std::pow(d[10], 3) - d[2] * std::pow(d[8], 2) * std::pow(d[10], 3) -
        2 * d[2] * d[5] * d[8] * std::pow(d[9], 2) + 2 * std::pow(d[2], 2) * d[8] * d[9] * d[10] -
        2 * std::pow(d[5], 2) * d[8] * d[9] * d[10] + 2 * d[2] * d[5] * d[8] * std::pow(d[10], 2) +
        std::pow(d[2], 2) * d[5] * d[9] + std::pow(d[5], 3) * d[9] - std::pow(d[2], 3) * d[10] -
        d[2] * std::pow(d[5], 2) * d[10];
    coeffs[106] =
        -3 * d[5] * std::pow(d[6], 2) * std::pow(d[9], 2) * d[11] -
        3 * d[5] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[11] + 6 * d[3] * d[6] * d[8] * std::pow(d[9], 2) * d[11] +
        6 * d[4] * d[7] * d[8] * std::pow(d[9], 2) * d[11] + 2 * d[2] * std::pow(d[6], 2) * d[9] * d[10] * d[11] +
        2 * d[2] * std::pow(d[7], 2) * d[9] * d[10] * d[11] - 4 * d[0] * d[6] * d[8] * d[9] * d[10] * d[11] -
        4 * d[1] * d[7] * d[8] * d[9] * d[10] * d[11] - d[5] * std::pow(d[6], 2) * std::pow(d[10], 2) * d[11] -
        d[5] * std::pow(d[7], 2) * std::pow(d[10], 2) * d[11] + 2 * d[3] * d[6] * d[8] * std::pow(d[10], 2) * d[11] +
        2 * d[4] * d[7] * d[8] * std::pow(d[10], 2) * d[11] + d[2] * std::pow(d[6], 2) * std::pow(d[9], 2) * d[12] +
        d[2] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[12] - 2 * d[0] * d[6] * d[8] * std::pow(d[9], 2) * d[12] -
        2 * d[1] * d[7] * d[8] * std::pow(d[9], 2) * d[12] - 2 * d[5] * std::pow(d[6], 2) * d[9] * d[10] * d[12] -
        2 * d[5] * std::pow(d[7], 2) * d[9] * d[10] * d[12] + 4 * d[3] * d[6] * d[8] * d[9] * d[10] * d[12] +
        4 * d[4] * d[7] * d[8] * d[9] * d[10] * d[12] + 3 * d[2] * std::pow(d[6], 2) * std::pow(d[10], 2) * d[12] +
        3 * d[2] * std::pow(d[7], 2) * std::pow(d[10], 2) * d[12] -
        6 * d[0] * d[6] * d[8] * std::pow(d[10], 2) * d[12] - 6 * d[1] * d[7] * d[8] * std::pow(d[10], 2) * d[12] -
        4 * d[2] * d[3] * d[6] * d[9] * d[11] + 4 * d[0] * d[5] * d[6] * d[9] * d[11] -
        4 * d[2] * d[4] * d[7] * d[9] * d[11] + 4 * d[1] * d[5] * d[7] * d[9] * d[11] -
        4 * d[0] * d[3] * d[8] * d[9] * d[11] - 4 * d[1] * d[4] * d[8] * d[9] * d[11] +
        2 * std::pow(d[0], 2) * d[8] * d[10] * d[11] + 2 * std::pow(d[1], 2) * d[8] * d[10] * d[11] -
        2 * std::pow(d[3], 2) * d[8] * d[10] * d[11] - 2 * std::pow(d[4], 2) * d[8] * d[10] * d[11] +
        2 * std::pow(d[0], 2) * d[8] * d[9] * d[12] + 2 * std::pow(d[1], 2) * d[8] * d[9] * d[12] -
        2 * std::pow(d[3], 2) * d[8] * d[9] * d[12] - 2 * std::pow(d[4], 2) * d[8] * d[9] * d[12] -
        4 * d[2] * d[3] * d[6] * d[10] * d[12] + 4 * d[0] * d[5] * d[6] * d[10] * d[12] -
        4 * d[2] * d[4] * d[7] * d[10] * d[12] + 4 * d[1] * d[5] * d[7] * d[10] * d[12] +
        4 * d[0] * d[3] * d[8] * d[10] * d[12] + 4 * d[1] * d[4] * d[8] * d[10] * d[12] +
        2 * d[0] * d[2] * d[3] * d[11] + 2 * d[1] * d[2] * d[4] * d[11] - std::pow(d[0], 2) * d[5] * d[11] -
        std::pow(d[1], 2) * d[5] * d[11] + std::pow(d[3], 2) * d[5] * d[11] + std::pow(d[4], 2) * d[5] * d[11] -
        std::pow(d[0], 2) * d[2] * d[12] - std::pow(d[1], 2) * d[2] * d[12] + d[2] * std::pow(d[3], 2) * d[12] +
        d[2] * std::pow(d[4], 2) * d[12] - 2 * d[0] * d[3] * d[5] * d[12] - 2 * d[1] * d[4] * d[5] * d[12];
    coeffs[107] =
        -d[5] * std::pow(d[6], 2) * std::pow(d[9], 3) - d[5] * std::pow(d[7], 2) * std::pow(d[9], 3) +
        2 * d[3] * d[6] * d[8] * std::pow(d[9], 3) + 2 * d[4] * d[7] * d[8] * std::pow(d[9], 3) +
        d[2] * std::pow(d[6], 2) * std::pow(d[9], 2) * d[10] + d[2] * std::pow(d[7], 2) * std::pow(d[9], 2) * d[10] -
        2 * d[0] * d[6] * d[8] * std::pow(d[9], 2) * d[10] - 2 * d[1] * d[7] * d[8] * std::pow(d[9], 2) * d[10] -
        d[5] * std::pow(d[6], 2) * d[9] * std::pow(d[10], 2) - d[5] * std::pow(d[7], 2) * d[9] * std::pow(d[10], 2) +
        2 * d[3] * d[6] * d[8] * d[9] * std::pow(d[10], 2) + 2 * d[4] * d[7] * d[8] * d[9] * std::pow(d[10], 2) +
        d[2] * std::pow(d[6], 2) * std::pow(d[10], 3) + d[2] * std::pow(d[7], 2) * std::pow(d[10], 3) -
        2 * d[0] * d[6] * d[8] * std::pow(d[10], 3) - 2 * d[1] * d[7] * d[8] * std::pow(d[10], 3) -
        2 * d[2] * d[3] * d[6] * std::pow(d[9], 2) + 2 * d[0] * d[5] * d[6] * std::pow(d[9], 2) -
        2 * d[2] * d[4] * d[7] * std::pow(d[9], 2) + 2 * d[1] * d[5] * d[7] * std::pow(d[9], 2) -
        2 * d[0] * d[3] * d[8] * std::pow(d[9], 2) - 2 * d[1] * d[4] * d[8] * std::pow(d[9], 2) +
        2 * std::pow(d[0], 2) * d[8] * d[9] * d[10] + 2 * std::pow(d[1], 2) * d[8] * d[9] * d[10] -
        2 * std::pow(d[3], 2) * d[8] * d[9] * d[10] - 2 * std::pow(d[4], 2) * d[8] * d[9] * d[10] -
        2 * d[2] * d[3] * d[6] * std::pow(d[10], 2) + 2 * d[0] * d[5] * d[6] * std::pow(d[10], 2) -
        2 * d[2] * d[4] * d[7] * std::pow(d[10], 2) + 2 * d[1] * d[5] * d[7] * std::pow(d[10], 2) +
        2 * d[0] * d[3] * d[8] * std::pow(d[10], 2) + 2 * d[1] * d[4] * d[8] * std::pow(d[10], 2) +
        2 * d[0] * d[2] * d[3] * d[9] + 2 * d[1] * d[2] * d[4] * d[9] - std::pow(d[0], 2) * d[5] * d[9] -
        std::pow(d[1], 2) * d[5] * d[9] + std::pow(d[3], 2) * d[5] * d[9] + std::pow(d[4], 2) * d[5] * d[9] -
        std::pow(d[0], 2) * d[2] * d[10] - std::pow(d[1], 2) * d[2] * d[10] + d[2] * std::pow(d[3], 2) * d[10] +
        d[2] * std::pow(d[4], 2) * d[10] - 2 * d[0] * d[3] * d[5] * d[10] - 2 * d[1] * d[4] * d[5] * d[10];

    static const int coeffs_ind[] = {0,  12, 24, 36, 48, 72, 1,  13, 25, 37, 49, 73, 2,  14, 26, 38, 50,
                                     74, 3,  15, 27, 39, 51, 75, 4,  16, 28, 40, 52, 76, 5,  17, 29, 41,
                                     53, 77, 11, 23, 35, 47, 59, 83, 9,  21, 33, 45, 57, 81, 7,  19, 31,
                                     43, 55, 79, 10, 22, 34, 46, 58, 82, 8,  20, 32, 44, 56, 80};

    static const int C_ind[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65};

    MatrixXd C = MatrixXd::Zero(6, 11);
    for (int i = 0; i < 66; i++) {
        C(C_ind[i]) = coeffs(coeffs_ind[i]);
    }

    Matrix<double, 6, 6> C0 = C.leftCols(6);
    Matrix<double, 6, 5> C1 = C.rightCols(5);
    //    MatrixXd C12 = C0.fullPivLu().solve(C1);
    MatrixXd C12 = C0.partialPivLu().solve(C1);
    MatrixXd RR(7, 5);
    RR << -C12.bottomRows(2), MatrixXd::Identity(5, 5);

    static const int AM_ind[] = {5, 4, 0, 6, 1};
    MatrixXd AM(5, 5);
    for (int i = 0; i < 5; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    EigenSolver<MatrixXd> es(AM);
    ArrayXcd D = es.eigenvalues();
    ArrayXXcd V = es.eigenvectors();
    V = (V / V.row(0).replicate(5, 1)).eval();

    MatrixXcd sols(2, 5);
    sols.row(0) = V.row(1);
    sols.row(1) = D.transpose();
    return sols;
}

int plane_parallax_5pt_shared_focal(const Eigen::Matrix3d &H, const Point2D &x1, const Point2D &x2,
                                    ImagePairVector *out_image_pairs) {
    Point2D v = (H * x1.homogeneous()).hnormalized() - x2;
    VectorXd data(13);
    //    data << H(0, 0), H(1, 0), H(2, 0), H(0, 1), H(1, 1), H(2, 1), H(0, 2), H(1, 2), H(2, 2), x2(0), x2(1), v(0),
    //    v(1);
    data << H(0, 0), H(0, 1), H(0, 2), H(1, 0), H(1, 1), H(1, 2), H(2, 0), H(2, 1), H(2, 2), x2(0), x2(1), v(0), v(1);
    MatrixXcd sols = solver_4plus1(data);

    int n_sols = 0;

    out_image_pairs->reserve(out_image_pairs->size() + 5 * 4);

    for (int i = 0; i < sols.cols(); ++i) {
        if (std::abs(sols(0, i).imag()) > 1e-8)
            continue;

        if (sols(0, i).real() < 1e-8)
            continue;

        if (std::abs(sols(1, i).imag()) > 1e-8)
            continue;

        double focal = 1 / std::sqrt(sols(0, i).real());
        double s = sols(1, i).real();
        Vector3d e(x2(0) + s * v(0), x2(1) + s * v(1), 1.0);
        DiagonalMatrix<double, 3> K(focal, focal, 1.0);
        Matrix3d e_x;
        e_x << 0, -1.0, e(1), 1.0, 0, -e(0), -e(1), e(0), 0;
        Matrix3d E = K * e_x * H * K;
        DiagonalMatrix<double, 3> K_inv(1.0, 1.0, focal);

        CameraPoseVector poses;
        Camera calib = Camera("SIMPLE_PINHOLE", std::vector<double>{focal, 0.0, 0.0}, -1, -1);

        motion_from_essential(E, {x1.homogeneous().normalized()}, {x2.homogeneous().normalized()}, &poses);

        for (CameraPose const &pose : poses) {
            out_image_pairs->emplace_back(ImagePair(pose, calib, calib));
            n_sols++;
        }
    }
    return n_sols;
}
} // namespace poselib
