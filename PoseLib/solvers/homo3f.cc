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

    Eigen::VectorXd coeffs =
        equal_coeff(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);

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
    for (int i = 0; i < 10; ++i) {
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

    double a0 = g6 * g6, b0 = g0 * g0 + g3 * g3, a1 = g6 * g7, b1 = g0 * g1 + g3 * g4, a2 = g6 * g8,
           b2 = g0 * g2 + g3 * g5, a3 = g7 * g7, b3 = g1 * g1 + g4 * g4, a4 = g7 * g8, b4 = g1 * g2 + g4 * g5,
           a5 = g8 * g8, b5 = g2 * g2 + g5 * g5;

    double m0 = H13(0, 0), m1 = H13(0, 1), m2 = H13(0, 2), m3 = H13(1, 0), m4 = H13(1, 1), m5 = H13(1, 2),
           m6 = H13(2, 0), m7 = H13(2, 1), m8 = H13(2, 2);

    double c0 = m6 * m6, d0 = m0 * m0 + m3 * m3, c1 = m6 * m7, d1 = m0 * m1 + m3 * m4, c2 = m6 * m8,
           d2 = m0 * m2 + m3 * m5, c3 = m7 * m7, d3 = m1 * m1 + m4 * m4, c4 = m7 * m8, d4 = m1 * m2 + m4 * m5,
           c5 = m8 * m8, d5 = m2 * m2 + m5 * m5;

    // std::cout << "g6" << g6 << "" << a0 << " " << b0 << " "<< a1 << " "<< b1 << " "<< a2<< " " << b2 <<std::endl;

    Eigen::VectorXd coeffs =
        coeff_sequal(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);

    double roots[6];
    double p[7];
    for (int i = 0; i < 7; ++i) {
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

    double a0 = g6 * g6, b0 = g0 * g0 + g3 * g3, a1 = g6 * g7, b1 = g0 * g1 + g3 * g4, a2 = g6 * g8,
           b2 = g0 * g2 + g3 * g5, a3 = g7 * g7, b3 = g1 * g1 + g4 * g4, a4 = g7 * g8, b4 = g1 * g2 + g4 * g5,
           a5 = g8 * g8, b5 = g2 * g2 + g5 * g5;

    double m0 = H13(0, 0), m1 = H13(0, 1), m2 = H13(0, 2), m3 = H13(1, 0), m4 = H13(1, 1), m5 = H13(1, 2),
           m6 = H13(2, 0), m7 = H13(2, 1), m8 = H13(2, 2);

    double c0 = m6 * m6, d0 = m0 * m0 + m3 * m3, c1 = m6 * m7, d1 = m0 * m1 + m3 * m4, c2 = m6 * m8,
           d2 = m0 * m2 + m3 * m5, c3 = m7 * m7, d3 = m1 * m1 + m4 * m4, c4 = m7 * m8, d4 = m1 * m2 + m4 * m5,
           c5 = m8 * m8, d5 = m2 * m2 + m5 * m5;

    Eigen::MatrixXd eq1 =
        var_eq1(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq2 =
        var_eq2(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq3 =
        var_eq3(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq4 =
        var_eq4(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq5 =
        var_eq5(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq6 =
        var_eq6(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq7 =
        var_eq7(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);

    Eigen::MatrixXd C(7, 28);
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

    Eigen::MatrixXd C0(7, 7);
    Eigen::MatrixXd C1(7, 7);
    Eigen::MatrixXd C2(7, 7);
    Eigen::MatrixXd C3(7, 7);
    Eigen::MatrixXd C4(4, 7);

    C0 = C.block<7, 7>(0, 0);
    C1 = C.block<7, 7>(0, 7);
    C2 = C.block<7, 7>(0, 14);
    C3 = C.block<7, 7>(0, 21);

    C4.row(0) = C.block<1, 7>(0, 21);
    C4.row(1) = C.block<1, 7>(1, 21);
    C4.row(2) = C.block<1, 7>(2, 21);
    C4.row(3) = C.block<1, 7>(4, 21);

    // Eigen::MatrixXd M(7, 21);
    Eigen::MatrixXd M(7, 18);
    M << C4.transpose(), C2.transpose(), C1.transpose();
    M = (-C0.transpose().fullPivLu().solve(M)).eval();
    // M = (-C0.fullPivHouseholderQr().solve(M)).eval();

    Eigen::MatrixXd K(18, 18);
    K.setZero();
    K(0, 4) = 1.0;
    K(1, 5) = 1.0;
    K(2, 6) = 1.0;
    K(3, 8) = 1.0;
    K(4, 11) = 1.0;
    K(5, 12) = 1.0;
    K(6, 13) = 1.0;
    K(7, 14) = 1.0;
    K(8, 15) = 1.0;
    K(9, 16) = 1.0;
    K(10, 17) = 1.0;

    K.block<7, 18>(11, 0) = M;

    Eigen::EigenSolver<Eigen::Matrix<double, 18, 18>> es(K);
    const Eigen::VectorXcd &eigenvalues = es.eigenvalues();
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
        U = f * f * f * C0 + f * f * C1 + f * C2 + C3;
        V = -U.block<6, 6>(0, 1).partialPivLu().solve(U.block<6, 1>(0, 0));
        if ((V(0, 0) > 0) && (fabs(V(1, 0) - V(0, 0) * V(0, 0)) < 0.01 * V(1, 0))) {
            sols(0, k) = std::sqrt(f); // 1st row - f2, 2nd row - 1/f1, 3rd row - k.
            sols(1, k) = std::sqrt(V(0, 0));
            k++;
        }
    }

    sols.conservativeResize(2, k);

    return sols;
}

Eigen::MatrixXd solver_homo_case4(Eigen::Matrix3d &H12, Eigen::Matrix3d &H13) {

    // using namespace Eigen;
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

    // std::cout << "g6" << g6 << "" << a0 << " " << b0 << " "<< a1 << " "<< b1 << " "<< a2<< " " << b2 <<std::endl;

    Eigen::MatrixXd eq1 =
        svar_eq1(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq2 =
        svar_eq2(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq3 =
        svar_eq3(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);
    Eigen::MatrixXd eq4 =
        svar_eq4(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5, c0, c1, c2, c3, c4, c5, d0, d1, d2, d3, d4, d5);

    Eigen::MatrixXd C(4, 16);
    C.row(0) = eq1;
    C.row(1) = eq2;
    C.row(2) = eq3;
    C.row(3) = eq4;

    Eigen::MatrixXd C0(4, 4);
    Eigen::MatrixXd C1(4, 4);
    Eigen::MatrixXd C2(4, 4);
    Eigen::MatrixXd C3(4, 4);

    C0 = C.block<4, 4>(0, 0);
    C1 = C.block<4, 4>(0, 4);
    C2 = C.block<4, 4>(0, 8);
    C3 = C.block<4, 4>(0, 12);

    Eigen::MatrixXd M(4, 12);
    M << C0, C1, C2;
    M = (-C3.partialPivLu().solve(M)).eval();
    // M = (-C0.fullPivHouseholderQr().solve(M)).eval();

    Eigen::MatrixXd K(12, 12);
    K.setZero();
    K(0, 4) = 1.0;
    K(1, 5) = 1.0;
    K(2, 6) = 1.0;
    K(3, 7) = 1.0;
    K(4, 8) = 1.0;
    K(5, 9) = 1.0;
    K(6, 10) = 1.0;
    K(7, 11) = 1.0;

    K.block<4, 12>(8, 0) = M;

    // std::cout << K <<std::endl;
    // Eigen::MatrixXd K2(12,12);
    // K2 = K.cast <double> ();
    Eigen::EigenSolver<Eigen::Matrix<double, 12, 12>> es(K);
    const Eigen::VectorXcd &eigenvalues = es.eigenvalues();
    Eigen::MatrixXd sols = Eigen::MatrixXd::Zero(1, 12);
    int k = 0;
    for (int i = 0; i < 12; i++) // find real eigenvalues
    {
        if (abs(eigenvalues(i).imag()) > 0.001 || std::isnan(eigenvalues(i).real()) || eigenvalues(i).real() < 0.0) {
            continue;
        }
        sols(0, k) = eigenvalues(i).real();
        k++;
    }

    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(4, 4);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(3, 1);

    Eigen::MatrixXd res = Eigen::MatrixXd::Zero(2, k);
    int m = 0;
    double f;

    for (int i = 0; i < k; i++) {
        f = sols(0, i);
        U = f * f * f * C3 + f * f * C2 + f * C1 + C0;
        V = -U.block<3, 3>(0, 1).partialPivLu().solve(U.block<3, 1>(0, 0));
        if ((V(0, 0) > 0) && (fabs(V(1, 0) - V(0, 0) * V(0, 0)) < 0.01 * V(1, 0))) {
            res(0, m) = std::sqrt(sols(0, i));
            res(1, m) = std::sqrt(V(0, 0));
            m++;
        }
    }

    res.conservativeResize(2, m);

    return res;
}

void removeRowAndColumn(Eigen::MatrixXd &matrix, int index)
{
    int n = matrix.rows();
    matrix.block(index, 0, n - index - 1, n) = matrix.block(index + 1, 0, n - index - 1, n);
    matrix.block(0, index, n, n - index - 1) = matrix.block(0, index + 1, n, n - index - 1);
    matrix.conservativeResize(n - 1, n - 1);
}

std::vector<double> solver_homo3f_baseline(Eigen::Matrix3d &H1, Eigen::Matrix3d &H2) {
    H1 /= H1(2, 2);
    H2 /= H2(2, 2);
    double g1 = H1(0, 0), g2 = H1(0, 1), g3 = H1(0, 2), g4 = H1(1, 0), g5 = H1(1, 1), g6 = H1(1, 2), g7 = H1(2, 0), g8 = H1(2, 1);
    double h1 = H2(0, 0), h2 = H2(0, 1), h3 = H2(0, 2), h4 = H2(1, 0), h5 = H2(1, 1), h6 = H2(1, 2), h7 = H2(2, 0), h8 = H2(2, 1);

    // Initialize matrices
    Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(22, 22);
    Eigen::MatrixXd C1 = Eigen::MatrixXd::Zero(22, 22);
    Eigen::MatrixXd C2 = Eigen::MatrixXd::Zero(22, 22);
    Eigen::MatrixXd C3 = Eigen::MatrixXd::Zero(22, 22);
    Eigen::MatrixXd C4 = Eigen::MatrixXd::Zero(22, 22);

    // Fill C0
    C0(0, 8) = -g7 * g7 - g8 * g8;
    C0(1, 14) = -g7 * g7 - g8 * g8;
    C0(2, 9) = -g7 * g7 - g8 * g8;
    C0(3, 19) = -g7 * g7 - g8 * g8;
    C0(4, 10) = -g7 * g7 - g8 * g8;
    C0(5, 11) = -g7 * g7 - g8 * g8;
    C0(6, 15) = -g7 * g7 - g8 * g8;
    C0(7, 16) = -g7 * g7 - g8 * g8;
    C0(8, 20) = -g7 * g7 - g8 * g8;

    //
    C0(9, 8) = -h7 * h7 - h8 * h8;
    C0(10, 14) = -h7 * h7 - h8 * h8;
    C0(11, 9) = -h7 * h7 - h8 * h8;
    C0(12, 19) = -h7 * h7 - h8 * h8;
    C0(13, 10) = -h7 * h7 - h8 * h8;
    C0(14, 11) = -h7 * h7 - h8 * h8;
    C0(15, 15) = -h7 * h7 - h8 * h8;
    C0(16, 16) = -h7 * h7 - h8 * h8;
    C0(17, 20) = -h7 * h7 - h8 * h8;

    //
    C0(18, 2) = h7 * h7 + h8 * h8;
    C0(18, 4) = h7 * h7 + h8 * h8;
    C0(18, 13) = -h7 * h7 - h8 * h8;
    C0(18, 15) = h7 * h7 + h8 * h8;

    C0(19, 9) = h7 * h7 + h8 * h8;
    C0(19, 11) = h7 * h7 + h8 * h8;
    C0(19, 18) = -h7 * h7 - h8 * h8;
    C0(19, 20) = h7 * h7 + h8 * h8;
    C0(20, 3) = h7 * h7 + h8 * h8;
    C0(20, 5) = h7 * h7 + h8 * h8;
    C0(20, 14) = -h7 * h7 - h8 * h8;
    C0(20, 16) = h7 * h7 + h8 * h8;
    C0(21, 4) = h7 * h7 + h8 * h8;
    C0(21, 6) = h7 * h7 + h8 * h8;
    C0(21, 15) = -h7 * h7 - h8 * h8;
    C0(21, 17) = h7 * h7 + h8 * h8;

    // Fill C1
    C1(0, 1) = g1 * g7 + g2 * g8;
    C1(0, 3) = g1 * g7 + g2 * g8;
    C1(0, 7) = g4 * g7 + g5 * g8;
    C1(0, 9) = -g4 * g7 - g5 * g8;

    //  2
    C1(1, 8) = g1 * g7 + g2 * g8;
    C1(1, 10) = g1 * g7 + g2 * g8;
    C1(1, 13) = g4 * g7 + g5 * g8;
    C1(1, 15) = -g4 * g7 - g5 * g8;

    //  3
    C1(2, 2) = g1 * g7 + g2 * g8;
    C1(2, 4) = g1 * g7 + g2 * g8;
    C1(2, 8) = g4 * g7 + g5 * g8;
    C1(2, 10) = -g4 * g7 - g5 * g8;

    //  4
    C1(3, 14) = g1 * g7 + g2 * g8;
    C1(3, 16) = g1 * g7 + g2 * g8;
    C1(3, 18) = g4 * g7 + g5 * g8;
    C1(3, 20) = -g4 * g7 - g5 * g8;

    //  5
    C1(4, 3) = g1 * g7 + g2 * g8;
    C1(4, 5) = g1 * g7 + g2 * g8;
    C1(4, 9) = g4 * g7 + g5 * g8;
    C1(4, 11) = -g4 * g7 - g5 * g8;

    //  6
    C1(5, 4) = g1 * g7 + g2 * g8;
    C1(5, 6) = g1 * g7 + g2 * g8;
    C1(5, 10) = g4 * g7 + g5 * g8;
    C1(5, 12) = -g4 * g7 - g5 * g8;

    //  7
    C1(6, 9) = g1 * g7 + g2 * g8;
    C1(6, 11) = g1 * g7 + g2 * g8;
    C1(6, 14) = g4 * g7 + g5 * g8;
    C1(6, 16) = -g4 * g7 - g5 * g8;

    //  8
    C1(7, 10) = g1 * g7 + g2 * g8;
    C1(7, 12) = g1 * g7 + g2 * g8;
    C1(7, 15) = g4 * g7 + g5 * g8;
    C1(7, 17) = -g4 * g7 - g5 * g8;

    //  9
    C1(8, 15) = g1 * g7 + g2 * g8;
    C1(8, 17) = g1 * g7 + g2 * g8;
    C1(8, 19) = g4 * g7 + g5 * g8;
    C1(8, 21) = -g4 * g7 - g5 * g8;

    //  10
    C1(9, 1) = h1 * h7 + h2 * h8;
    C1(9, 3) = h1 * h7 + h2 * h8;
    C1(9, 7) = h4 * h7 + h5 * h8;
    C1(9, 9) = -h4 * h7 - h5 * h8;

    //  11
    C1(10, 8) = h1 * h7 + h2 * h8;
    C1(10, 10) = h1 * h7 + h2 * h8;
    C1(10, 13) = h4 * h7 + h5 * h8;
    C1(10, 15) = -h4 * h7 - h5 * h8;

    //
    C1(11, 2) = h1 * h7 + h2 * h8;
    C1(11, 4) = h1 * h7 + h2 * h8;
    C1(11, 8) = h4 * h7 + h5 * h8;
    C1(11, 10) = -h4 * h7 - h5 * h8;

    //  13
    C1(12, 14) = h1 * h7 + h2 * h8;
    C1(12, 16) = h1 * h7 + h2 * h8;
    C1(12, 18) = h4 * h7 + h5 * h8;
    C1(12, 20) = -h4 * h7 - h5 * h8;

    //
    C1(13, 3) = h1 * h7 + h2 * h8;
    C1(13, 5) = h1 * h7 + h2 * h8;
    C1(13, 9) = h4 * h7 + h5 * h8;
    C1(13, 11) = -h4 * h7 - h5 * h8;

    //
    C1(14, 4) = h1 * h7 + h2 * h8;
    C1(14, 6) = h1 * h7 + h2 * h8;
    C1(14, 10) = h4 * h7 + h5 * h8;
    C1(14, 12) = -h4 * h7 - h5 * h8;

    //
    C1(15, 9) = h1 * h7 + h2 * h8;
    C1(15, 11) = h1 * h7 + h2 * h8;
    C1(15, 14) = h4 * h7 + h5 * h8;
    C1(15, 16) = -h4 * h7 - h5 * h8;

    //
    C1(16, 10) = h1 * h7 + h2 * h8;
    C1(16, 12) = h1 * h7 + h2 * h8;
    C1(16, 15) = h4 * h7 + h5 * h8;
    C1(16, 17) = -h4 * h7 - h5 * h8;

    //  18
    C1(17, 15) = h1 * h7 + h2 * h8;
    C1(17, 17) = h1 * h7 + h2 * h8;
    C1(17, 19) = h4 * h7 + h5 * h8;
    C1(17, 21) = -h4 * h7 - h5 * h8;

    //  19
    C1(18, 1) = -2 * h4 * h7 - 2 * h5 * h8;
    C1(18, 3) = -2 * h4 * h7 - 2 * h5 * h8;
    C1(18, 7) = 2 * h1 * h7 + 2 * h2 * h8;
    C1(18, 9) = 2 * h1 * h7 + 2 * h2 * h8;
    C1(18, 14) = -4 * h4 * h7 - 4 * h5 * h8;

    //  20
    C1(19, 8) = -2 * h4 * h7 - 2 * h5 * h8;
    C1(19, 10) = -2 * h4 * h7 - 2 * h5 * h8;
    C1(19, 13) = 2 * h1 * h7 + 2 * h2 * h8;
    C1(19, 15) = 2 * h1 * h7 + 2 * h2 * h8;
    C1(19, 19) = -4 * h4 * h7 - 4 * h5 * h8;

    //  21
    C1(20, 2) = -2 * h4 * h7 - 2 * h5 * h8;
    C1(20, 4) = -2 * h4 * h7 - 2 * h5 * h8;
    C1(20, 8) = 2 * h1 * h7 + 2 * h2 * h8;
    C1(20, 10) = 2 * h1 * h7 + 2 * h2 * h8;
    C1(20, 15) = -4 * h4 * h7 - 4 * h5 * h8;

    //  22
    C1(21, 3) = -2 * h4 * h7 - 2 * h5 * h8;
    C1(21, 5) = -2 * h4 * h7 - 2 * h5 * h8;
    C1(21, 9) = 2 * h1 * h7 + 2 * h2 * h8;
    C1(21, 11) = 2 * h1 * h7 + 2 * h2 * h8;
    C1(21, 16) = -4 * h4 * h7 - 4 * h5 * h8;

    // Fill C2
    C2(0, 0) = -g1 * g4 - g2 * g5;
    C2(0, 2) = -g1 * g4 - g2 * g5;
    C2(0, 8) = g4 * g4 + g5 * g5 - 1;

    //  2
    C2(1, 7) = -g1 * g4 - g2 * g5;
    C2(1, 9) = -g1 * g4 - g2 * g5;
    C2(1, 14) = g4 * g4 + g5 * g5 - 1;

    //  3
    C2(2, 1) = -g1 * g4 - g2 * g5;
    C2(2, 3) = -g1 * g4 - g2 * g5;
    C2(2, 9) = g4 * g4 + g5 * g5 - 1;

    //  4
    C2(3, 13) = -g1 * g4 - g2 * g5;
    C2(3, 15) = -g1 * g4 - g2 * g5;
    C2(3, 19) = g4 * g4 + g5 * g5 - 1;

    //  5
    C2(4, 2) = -g1 * g4 - g2 * g5;
    C2(4, 4) = -g1 * g4 - g2 * g5;
    C2(4, 10) = g4 * g4 + g5 * g5 - 1;

    //  6
    C2(5, 3) = -g1 * g4 - g2 * g5;
    C2(5, 5) = -g1 * g4 - g2 * g5;
    C2(5, 11) = g4 * g4 + g5 * g5 - 1;

    //  7
    C2(6, 8) = -g1 * g4 - g2 * g5;
    C2(6, 10) = -g1 * g4 - g2 * g5;
    C2(6, 15) = g4 * g4 + g5 * g5 - 1;

    //  8
    C2(7, 9) = -g1 * g4 - g2 * g5;
    C2(7, 11) = -g1 * g4 - g2 * g5;
    C2(7, 16) = g4 * g4 + g5 * g5 - 1;

    //  9
    C2(8, 14) = -g1 * g4 - g2 * g5;
    C2(8, 16) = -g1 * g4 - g2 * g5;
    C2(8, 20) = g4 * g4 + g5 * g5 - 1;

    //  10
    C2(9, 0) = -h1 * h4 - h2 * h5;
    C2(9, 2) = -h1 * h4 - h2 * h5;
    C2(9, 8) = h4 * h4 + h5 * h5 - 1;

    //  11
    C2(10, 7) = -h1 * h4 - h2 * h5;
    C2(10, 9) = -h1 * h4 - h2 * h5;
    C2(10, 14) = h4 * h4 + h5 * h5 - 1;

    //
    C2(11, 1) = -h1 * h4 - h2 * h5;
    C2(11, 3) = -h1 * h4 - h2 * h5;
    C2(11, 9) = h4 * h4 + h5 * h5 - 1;

    //  13
    C2(12, 13) = -h1 * h4 - h2 * h5;
    C2(12, 15) = -h1 * h4 - h2 * h5;
    C2(12, 19) = h4 * h4 + h5 * h5 - 1;

    //
    C2(13, 2) = -h1 * h4 - h2 * h5;
    C2(13, 4) = -h1 * h4 - h2 * h5;
    C2(13, 10) = h4 * h4 + h5 * h5 - 1;

    //
    C2(14, 3) = -h1 * h4 - h2 * h5;
    C2(14, 5) = -h1 * h4 - h2 * h5;
    C2(14, 11) = h4 * h4 + h5 * h5 - 1;

    //
    C2(15, 8) = -h1 * h4 - h2 * h5;
    C2(15, 10) = -h1 * h4 - h2 * h5;
    C2(15, 15) = h4 * h4 + h5 * h5 - 1;

    //
    C2(16, 9) = -h1 * h4 - h2 * h5;
    C2(16, 11) = -h1 * h4 - h2 * h5;
    C2(16, 16) = h4 * h4 + h5 * h5 - 1;

    //  18
    C2(17, 14) = -h1 * h4 - h2 * h5;
    C2(17, 16) = -h1 * h4 - h2 * h5;
    C2(17, 20) = h4 * h4 + h5 * h5 - 1;

    //  19
    C2(18, 0) = -h1 * h1 - h2 * h2 + h4 * h4 + h5 * h5;
    C2(18, 2) = -2 * h1 * h1 - 2 * h2 * h2 + h4 * h4 + h5 * h5 + 1;
    C2(18, 4) = -h1 * h1 - h2 * h2 + 1;
    C2(18, 8) = 2 * h1 * h4 + 2 * h2 * h5;
    C2(18, 10) = 2 * h1 * h4 + 2 * h2 * h5;
    C2(18, 13) = h4 * h4 + h5 * h5 - 1;
    C2(18, 15) = -h4 * h4 - h5 * h5 + 1;

    //  20
    C2(19, 7) = -h1 * h1 - h2 * h2 + h4 * h4 + h5 * h5;
    C2(19, 9) = -2 * h1 * h1 - 2 * h2 * h2 + h4 * h4 + h5 * h5 + 1;
    C2(19, 11) = -h1 * h1 - h2 * h2 + 1;
    C2(19, 14) = 2 * h1 * h4 + 2 * h2 * h5;
    C2(19, 16) = 2 * h1 * h4 + 2 * h2 * h5;
    C2(19, 18) = h4 * h4 + h5 * h5 - 1;
    C2(19, 20) = -h4 * h4 - h5 * h5 + 1;

    //  21
    C2(20, 1) = -h1 * h1 - h2 * h2 + h4 * h4 + h5 * h5;
    C2(20, 3) = -2 * h1 * h1 - 2 * h2 * h2 + h4 * h4 + h5 * h5 + 1;
    C2(20, 5) = -h1 * h1 - h2 * h2 + 1;
    C2(20, 9) = 2 * h1 * h4 + 2 * h2 * h5;
    C2(20, 11) = 2 * h1 * h4 + 2 * h2 * h5;
    C2(20, 14) = h4 * h4 + h5 * h5 - 1;
    C2(20, 16) = -h4 * h4 - h5 * h5 + 1;

    //  22
    C2(21, 2) = -h1 * h1 - h2 * h2 + h4 * h4 + h5 * h5;
    C2(21, 4) = -2 * h1 * h1 - 2 * h2 * h2 + h4 * h4 + h5 * h5 + 1;
    C2(21, 6) = -h1 * h1 - h2 * h2 + 1;
    C2(21, 10) = 2 * h1 * h4 + 2 * h2 * h5;
    C2(21, 12) = 2 * h1 * h4 + 2 * h2 * h5;
    C2(21, 15) = h4 * h4 + h5 * h5 - 1;
    C2(21, 17) = -h4 * h4 - h5 * h5 + 1;

    // Fill C3
    C3(0, 1) = g3;
    C3(0, 3) = g3;
    C3(0, 7) = g6;
    C3(0, 9) = -g6;

    //  2
    C3(1, 8) = g3;
    C3(1, 10) = g3;
    C3(1, 13) = g6;
    C3(1, 15) = -g6;

    //  3
    C3(2, 2) = g3;
    C3(2, 4) = g3;
    C3(2, 8) = g6;
    C3(2, 10) = -g6;

    //  4
    C3(3, 14) = g3;
    C3(3, 16) = g3;
    C3(3, 18) = g6;
    C3(3, 20) = -g6;

    //  5
    C3(4, 3) = g3;
    C3(4, 5) = g3;
    C3(4, 9) = g6;
    C3(4, 11) = -g6;

    //  6
    C3(5, 4) = g3;
    C3(5, 6) = g3;
    C3(5, 10) = g6;
    C3(5, 12) = -g6;

    //  7
    C3(6, 9) = g3;
    C3(6, 11) = g3;
    C3(6, 14) = g6;
    C3(6, 16) = -g6;

    //  8
    C3(7, 10) = g3;
    C3(7, 12) = g3;
    C3(7, 15) = g6;
    C3(7, 17) = -g6;

    //  9
    C3(8, 15) = g3;
    C3(8, 17) = g3;
    C3(8, 19) = g6;
    C3(8, 21) = -g6;

    //  10
    C3(9, 1) = h3;
    C3(9, 3) = h3;
    C3(9, 7) = h6;
    C3(9, 9) = -h6;

    //  11
    C3(10, 8) = h3;
    C3(10, 10) = h3;
    C3(10, 13) = h6;
    C3(10, 15) = -h6;

    //
    C3(11, 2) = h3;
    C3(11, 4) = h3;
    C3(11, 8) = h6;
    C3(11, 10) = -h6;

    //  13
    C3(12, 14) = h3;
    C3(12, 16) = h3;
    C3(12, 18) = h6;
    C3(12, 20) = -h6;

    //
    C3(13, 3) = h3;
    C3(13, 5) = h3;
    C3(13, 9) = h6;
    C3(13, 11) = -h6;

    //
    C3(14, 4) = h3;
    C3(14, 6) = h3;
    C3(14, 10) = h6;
    C3(14, 12) = -h6;

    //
    C3(15, 9) = h3;
    C3(15, 11) = h3;
    C3(15, 14) = h6;
    C3(15, 16) = -h6;

    //
    C3(16, 10) = h3;
    C3(16, 12) = h3;
    C3(16, 15) = h6;
    C3(16, 17) = -h6;

    //  18
    C3(17, 15) = h3;
    C3(17, 17) = h3;
    C3(17, 19) = h6;
    C3(17, 21) = -h6;

    //  19
    C3(18, 1) = -2 * h6;
    C3(18, 3) = -2 * h6;
    C3(18, 7) = 2 * h3;
    C3(18, 9) = 2 * h3;
    C3(18, 14) = -4 * h6;

    //  20
    C3(19, 8) = -2 * h6;
    C3(19, 10) = -2 * h6;
    C3(19, 13) = 2 * h3;
    C3(19, 15) = 2 * h3;
    C3(19, 19) = -4 * h6;

    //  21
    C3(20, 2) = -2 * h6;
    C3(20, 4) = -2 * h6;
    C3(20, 8) = 2 * h3;
    C3(20, 10) = 2 * h3;
    C3(20, 15) = -4 * h6;

    //  22
    C3(21, 3) = -2 * h6;
    C3(21, 5) = -2 * h6;
    C3(21, 9) = 2 * h3;
    C3(21, 11) = 2 * h3;
    C3(21, 16) = -4 * h6;

    // Fill C4
    C4(0, 0) = -g3 * g6;
    C4(0, 2) = -g3 * g6;
    C4(0, 8) = g6 * g6;

    //  2
    C4(1, 7) = -g3 * g6;
    C4(1, 9) = -g3 * g6;
    C4(1, 14) = g6 * g6;

    //  3
    C4(2, 1) = -g3 * g6;
    C4(2, 3) = -g3 * g6;
    C4(2, 9) = g6 * g6;

    //  4
    C4(3, 13) = -g3 * g6;
    C4(3, 15) = -g3 * g6;
    C4(3, 19) = g6 * g6;

    //  5
    C4(4, 2) = -g3 * g6;
    C4(4, 4) = -g3 * g6;
    C4(4, 10) = g6 * g6;

    //  6
    C4(5, 3) = -g3 * g6;
    C4(5, 5) = -g3 * g6;
    C4(5, 11) = g6 * g6;

    //  7
    C4(6, 8) = -g3 * g6;
    C4(6, 10) = -g3 * g6;
    C4(6, 15) = g6 * g6;

    //  8
    C4(7, 9) = -g3 * g6;
    C4(7, 11) = -g3 * g6;
    C4(7, 16) = g6 * g6;

    //  9
    C4(8, 14) = -g3 * g6;
    C4(8, 16) = -g3 * g6;
    C4(8, 20) = g6 * g6;

    //  10
    C4(9, 0) = -h3 * h6;
    C4(9, 2) = -h3 * h6;
    C4(9, 8) = h6 * h6;

    //  11
    C4(10, 7) = -h3 * h6;
    C4(10, 9) = -h3 * h6;
    C4(10, 14) = h6 * h6;

    //
    C4(11, 1) = -h3 * h6;
    C4(11, 3) = -h3 * h6;
    C4(11, 9) = h6 * h6;

    //  13
    C4(12, 13) = -h3 * h6;
    C4(12, 15) = -h3 * h6;
    C4(12, 19) = h6 * h6;

    //
    C4(13, 2) = -h3 * h6;
    C4(13, 4) = -h3 * h6;
    C4(13, 10) = h6 * h6;

    //
    C4(14, 3) = -h3 * h6;
    C4(14, 5) = -h3 * h6;
    C4(14, 11) = h6 * h6;

    //
    C4(15, 8) = -h3 * h6;
    C4(15, 10) = -h3 * h6;
    C4(15, 15) = h6 * h6;

    //
    C4(16, 9) = -h3 * h6;
    C4(16, 11) = -h3 * h6;
    C4(16, 16) = h6 * h6;

    //
    C4(17, 14) = -h3 * h6;
    C4(17, 16) = -h3 * h6;
    C4(17, 20) = h6 * h6;

    //
    C4(18, 0) = -h3 * h3 + h6 * h6;
    C4(18, 2) = -2 * h3 * h3 + h6 * h6;
    C4(18, 4) = -h3 * h3;
    C4(18, 8) = 2 * h3 * h6;
    C4(18, 10) = 2 * h3 * h6;
    C4(18, 13) = h6 * h6;
    C4(18, 15) = -h6 * h6;

    //
    C4(19, 7) = -h3 * h3 + h6 * h6;
    C4(19, 9) = -2 * h3 * h3 + h6 * h6;
    C4(19, 11) = -h3 * h3;
    C4(19, 14) = 2 * h3 * h6;
    C4(19, 16) = 2 * h3 * h6;
    C4(19, 18) = h6 * h6;
    C4(19, 20) = -h6 * h6;

    //
    C4(20, 1) = -h3 * h3 + h6 * h6;
    C4(20, 3) = -2 * h3 * h3 + h6 * h6;
    C4(20, 5) = -h3 * h3;
    C4(20, 9) = 2 * h3 * h6;
    C4(20, 11) = 2 * h3 * h6;
    C4(20, 14) = h6 * h6;
    C4(20, 16) = -h6 * h6;

    //
    C4(21, 2) = -h3 * h3 + h6 * h6;
    C4(21, 4) = -2 * h3 * h3 + h6 * h6;
    C4(21, 6) = -h3 * h3;
    C4(21, 10) = 2 * h3 * h6;
    C4(21, 12) = 2 * h3 * h6;
    C4(21, 15) = h6 * h6;
    C4(21, 17) = -h6 * h6;

    // Construct A and B matrices
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(22, 22);
    Eigen::MatrixXd g = Eigen::MatrixXd::Identity(22, 22);

    Eigen::MatrixXd A(88, 88);
    A << z, g, z, z,
        z, z, g, z,
        z, z, z, g,
        -C0, -C1, -C2, -C3;

    Eigen::MatrixXd B(88, 88);
    B << g, z, z, z,
        z, g, z, z,
        z, z, g, z,
        z, z, z, C4;

    // Remove specific rows and columns
    std::vector<int> ii = {0, 1, 7, 12, 21, 22};
    Eigen::MatrixXd C = A;
    Eigen::MatrixXd D = B;
    for (int i = ii.size() - 1; i >= 0; --i) {
        removeRowAndColumn(C, ii[i]);
        removeRowAndColumn(D, ii[i]);
    }

    // Compute eigenvalues
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> es(C, D);
    Eigen::VectorXcd fs = es.eigenvalues();

    std::vector<double> focals;
    focals.reserve(fs.size());

    for (int i = 0; i < fs.size(); i++){
        if (std::abs(fs[i].imag()) < 1e-8 and fs[i].real() > 0.0){
            focals.emplace_back(fs[i].real());
        }
    }

    return focals;
}
} // namespace poselib