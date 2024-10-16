//
// Created by kocur on 05-Sep-24.
//

#include "PoseLib/misc/sturm.h"
#include "PoseLib/solvers/homo3f_utils.h"
#include "PoseLib/solvers/var/coeff.h"

#include <Eigen/Core>
#include <iostream>

namespace poselib {

std::vector<double> solver_homo3f(Eigen::Matrix3d &H12, Eigen::Matrix3d &H13) {
    // Compute coefficients
    double g0 = H12(0, 0), g1 = H12(0, 1), g2 = H12(0, 2), g3 = H12(1, 0), g4 = H12(1, 1), g5 = H12(1, 2),
           g6 = H12(2, 0), g7 = H12(2, 1), g8 = H12(2, 2);

    double a0 = g6 * g6, b0 = g0 * g0 + g3 * g3, a1 = g6 * g7, b1 = g0 * g1 + g3 * g4, a2 = g6 * g8,
           b2 = g0 * g2 + g3 * g5, a3 = g7 * g7, b3 = g1 * g1 + g4 * g4, a4 = g7 * g8, b4 = g1 * g2 + g4 * g5,
           a5 = g8 * g8, b5 = g2 * g2 + g5 * g5;

    double m0 = H13(0, 0), m1 = H13(0, 1), m2 = H13(0, 2), m3 = H13(1, 0), m4 = H13(1, 1), m5 = H13(1, 2),
           m6 = H13(2, 0), m7 = H13(2, 1), m8 = H13(2, 2);

    double c0 = m6 * m6, d0 = m0 * m0 + m3 * m3, c1 = m6 * m7, d1 = m0 * m1 + m3 * m4, c2 = m6 * m8,
           d2 = m0 * m2 + m3 * m5, c3 = m7 * m7, d3 = m1 * m1 + m4 * m4, c4 = m7 * m8, d4 = m1 * m2 + m4 * m5,
           c5 = m8 * m8, d5 = m2 * m2 + m5 * m5;

    Eigen::VectorXd coeffs = equal_coeff(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5,
                                                d0, d1, d2, d3, d4, d5);


//    polynomial::Polynomial<9> h(coeffs);
//
//    std::vector<double> roots;
//    h.realRootsSturm(0, 10, roots);
//
//    std::vector<double> sols;
//    sols.reserve(roots.size());
//    for (size_t i = 0; i < roots.size(); ++i) {
//        if (roots[i] > 0.0) {
//            sols.emplace_back(std::sqrt(roots[i]));
//        }
//    }

    double roots[9];
    double p[10];
    for (int i = 0; i < 10; ++i){
        p[9 - i] = coeffs[i];
    }
    int nroots = sturm::bisect_sturm<9>(p, roots, 1e-12);

    std::vector<double> sols;
    sols.reserve(nroots);
    for (int i = 0; i < nroots; ++i) {
        if (roots[i] > 0.0 and roots[i] < 100) {
            sols.emplace_back(std::sqrt(roots[i]));
        }
    }

    return sols;
}

std::vector<double> solver_homo_case2(Eigen::Matrix3d &H12, Eigen::Matrix3d &H13) {
    double g0 = H12(0, 0), g1 = H12(0, 1), g2 = H12(0, 2), g3 = H12(1, 0), g4 = H12(1, 1), g5 = H12(1, 2),
           g6 = H12(2, 0), g7 = H12(2, 1), g8 = H12(2, 2);

    double a0 = g6 * g6, b0 = g0 * g0 + g3 * g3, a1 = g6 * g7,
           b1 = g0 * g1 + g3 * g4, a2 = g6 * g8, b2 = g0 * g2 + g3 * g5,
           a3 = g7 * g7, b3 = g1 * g1 + g4 * g4, a4 = g7 * g8,
           b4 = g1 * g2 + g4 * g5, a5 = g8 * g8, b5 = g2 * g2 + g5 * g5;

    double m0 = H13(0, 0), m1 = H13(0, 1), m2 = H13(0, 2), m3 = H13(1, 0), m4 = H13(1, 1), m5 = H13(1, 2),
           m6 = H13(2, 0), m7 = H13(2, 1), m8 = H13(2, 2);

    double c0 = m6 * m6, d0 = m0 * m0 + m3 * m3, c1 = m6 * m7,
           d1 = m0 * m1 + m3 * m4, c2 = m6 * m8, d2 = m0 * m2 + m3 * m5,
           c3 = m7 * m7, d3 = m1 * m1 + m4 * m4, c4 = m7 * m8,
           d4 = m1 * m2 + m4 * m5, c5 = m8 * m8, d5 = m2 * m2 + m5 * m5;

    // std::cout << "g6" << g6 << "" << a0 << " " << b0 << " "<< a1 << " "<< b1 << " "<< a2<< " " << b2 <<std::endl;

    Eigen::VectorXd coeffs = coeff_sequal(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2,
                                          c3, c4, c5, d0, d1, d2, d3, d4, d5);

    double roots[6];
    double p[7];
    for (int i = 0; i < 7; ++i){
        p[6 - i] = coeffs[i];
    }
    int nroots = sturm::bisect_sturm<6>(p, roots, 1e-12);

    // std::cout<<"roots.size() " << roots.size() << std::endl;

    std::vector<double> sols;
    sols.reserve(nroots);
    for (int i = 0; i < nroots; ++i) {
        if (roots[i] > 0.0 and roots[i] < 100) {
            sols.emplace_back(std::sqrt(roots[i]));
        }
    }

    return sols;
}


Eigen::MatrixXd solver_homo_case3(Eigen::Matrix3d &H12, Eigen::Matrix3d &H13) {
    double g0 = H12(0, 0), g1 = H12(0, 1), g2 = H12(0, 2), g3 = H12(1, 0), g4 = H12(1, 1), g5 = H12(1, 2),
           g6 = H12(2, 0), g7 = H12(2, 1), g8 = H12(2, 2);

    double a0 = g6 * g6, b0 = g0 * g0 + g3 * g3, a1 = g6 * g7,
           b1 = g0 * g1 + g3 * g4, a2 = g6 * g8, b2 = g0 * g2 + g3 * g5,
           a3 = g7 * g7, b3 = g1 * g1 + g4 * g4, a4 = g7 * g8,
           b4 = g1 * g2 + g4 * g5, a5 = g8 * g8, b5 = g2 * g2 + g5 * g5;

    double m0 = H13(0, 0), m1 = H13(0, 1), m2 = H13(0, 2), m3 = H13(1, 0), m4 = H13(1, 1), m5 = H13(1, 2),
           m6 = H13(2, 0), m7 = H13(2, 1), m8 = H13(2, 2);

    double c0 = m6 * m6, d0 = m0 * m0 + m3 * m3, c1 = m6 * m7,
           d1 = m0 * m1 + m3 * m4, c2 = m6 * m8, d2 = m0 * m2 + m3 * m5,
           c3 = m7 * m7, d3 = m1 * m1 + m4 * m4, c4 = m7 * m8,
           d4 = m1 * m2 + m4 * m5, c5 = m8 * m8, d5 = m2 * m2 + m5 * m5;

    Eigen::MatrixXd eq1 = var_eq1(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2,
                                  c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq2 = var_eq2(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2,
                                  c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq3 = var_eq3(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2,
                                  c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq4 = var_eq4(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2,
                                  c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq5 = var_eq5(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2,
                                  c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq6 = var_eq6(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2,
                                  c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq7 = var_eq7(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2,
                                  c3, c4, c5, d0, d1, d2, d3, d4, d5);

    Eigen::MatrixXd C(7,28);
    C.row(0) = eq1;
    C.row(1) = eq2;
    C.row(2) = eq3;
    C.row(3) = eq4;
    C.row(4) = eq5;
    C.row(5) = eq6;
    C.row(6) = eq7;

    C.row(3) = C.row(3) + C.row(0);
    C.row(5) = C.row(5) - C.row(1);
    C.row(6) = C.row(6) + C.row(2);


    Eigen::MatrixXd C0(7,7);
    Eigen::MatrixXd C1(7,7);
    Eigen::MatrixXd C2(7,7);
    Eigen::MatrixXd C3(7,7);
    Eigen::MatrixXd C4(4,7);

    C0 = C.block<7,7>(0,0);
    C1 = C.block<7,7>(0,7);
    C2 = C.block<7,7>(0,14);
    C3 = C.block<7,7>(0,21);

    C4.row(0) = C.block<1,7>(0,21);
    C4.row(1) = C.block<1,7>(1,21);
    C4.row(2) = C.block<1,7>(2,21);
    C4.row(3) = C.block<1,7>(4,21);

    // MatrixXd M(7, 21);
    Eigen::MatrixXd M(7, 18);
    M << C4.transpose(), C2.transpose(), C1.transpose();
    M = (-C0.transpose().fullPivLu().solve(M)).eval();
    // M = (-C0.fullPivHouseholderQr().solve(M)).eval();

    Eigen::MatrixXd K(18,18);
    K.setZero();
    K(0,4) = 1.0; K(1,5) = 1.0; K(2,6) = 1.0; K(3,8) = 1.0; K(4,11) = 1.0; K(5,12) = 1.0; K(6,13) = 1.0;
    K(7,14) = 1.0; K(8,15) = 1.0; K(9,16) = 1.0; K(10,17) = 1.0;

    K.block<7, 18>(11, 0) = M;

    Eigen::EigenSolver<Eigen::Matrix<double, 18, 18>> es(K);
    const Eigen::VectorXcd& eigenvalues = es.eigenvalues();
    Eigen::MatrixXd sols = Eigen::MatrixXd::Zero(2, 18);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(7, 7);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(6, 1);

    int k = 0;
    double f;

    for (int i = 0; i < 18; i++) // find real eigenvalues
    {
        if (abs(eigenvalues(i).imag()) > 0.001 || std::isnan(eigenvalues(i).real()) || eigenvalues(i).real() < 0.0) {
            continue;
        }
        f = eigenvalues(i).real();
        U = f * f * f* C0 + f*f*C1 + f*C2 + C3;
        V = -U.block<6, 6>(0, 1).partialPivLu().solve(U.block<6, 1>(0, 0));
        if (  (V(0,0)>0) &&  (fabs(V(1,0)-V(0,0)*V(0,0)) < 0.01*V(1,0)))
        {
            sols(0,k) = std::sqrt(f); // 1st row - f2, 2nd row - 1/f1, 3rd row - k.
            sols(1,k) = std::sqrt(V(0,0));
            k++;
        }
    }

    sols.conservativeResize(2, k);

    return sols;
}

} // namespace poselib