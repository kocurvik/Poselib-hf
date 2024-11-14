#include <Eigen/Dense>
#include "PoseLib/solvers/var/coeff.h"
using namespace Eigen;

MatrixXd svar_eq7(double a0, double a1, double a2, double a3, double a4,
                  double a5, double b0, double b1, double b2, double b3,
                  double b4, double b5, double c0, double c1, double c2,
                  double c3, double c4, double c5, double d0, double d1,
                  double d2, double d3, double d4, double d5) {

  MatrixXd sols(1, 16);
  //   Matrix<long double, Dynamic, Dynamic> sols(1, 28);
  sols << 4 * pow(b5, 3) * pow(d1, 3) - 4 * pow(b1, 3) * pow(d5, 3) +
              b1 * pow(b3, 2) * pow(d0, 3) - 4 * pow(b0, 2) * b3 * pow(d1, 3) -
              4 * b1 * pow(b4, 2) * pow(d0, 3) -
              8 * b0 * pow(b5, 2) * pow(d1, 3) + b1 * pow(b5, 2) * pow(d0, 3) -
              pow(b0, 2) * b1 * pow(d5, 3) + 8 * pow(b0, 2) * b4 * pow(d2, 3) +
              4 * pow(b0, 2) * b5 * pow(d1, 3) -
              16 * pow(b2, 2) * b3 * pow(d1, 3) +
              32 * pow(b1, 2) * b4 * pow(d2, 3) +
              16 * pow(b2, 2) * b5 * pow(d1, 3) - b1 * pow(b3, 2) * pow(d5, 3) -
              4 * b3 * pow(b5, 2) * pow(d1, 3) +
              8 * pow(b3, 2) * b4 * pow(d2, 3) - pow(b0, 3) * d1 * pow(d3, 2) +
              4 * pow(b1, 3) * pow(d0, 2) * d3 +
              4 * pow(b0, 3) * d1 * pow(d4, 2) - pow(b0, 3) * d1 * pow(d5, 2) +
              8 * pow(b1, 3) * d0 * pow(d5, 2) -
              4 * pow(b1, 3) * pow(d0, 2) * d5 +
              16 * pow(b1, 3) * pow(d2, 2) * d3 -
              8 * pow(b2, 3) * pow(d0, 2) * d4 + pow(b5, 3) * pow(d0, 2) * d1 -
              32 * pow(b2, 3) * pow(d1, 2) * d4 -
              16 * pow(b1, 3) * pow(d2, 2) * d5 +
              4 * pow(b1, 3) * d3 * pow(d5, 2) -
              8 * pow(b2, 3) * pow(d3, 2) * d4 + pow(b5, 3) * d1 * pow(d3, 2) +
              2 * b0 * b1 * b3 * pow(d5, 3) - 16 * b0 * b3 * b4 * pow(d2, 3) +
              8 * b0 * b3 * b5 * pow(d1, 3) - 2 * b1 * b3 * b5 * pow(d0, 3) +
              4 * b2 * b3 * b4 * pow(d0, 3) - 4 * b2 * b4 * b5 * pow(d0, 3) +
              2 * pow(b0, 3) * d1 * d3 * d5 - 4 * pow(b0, 3) * d2 * d3 * d4 -
              8 * pow(b1, 3) * d0 * d3 * d5 + 16 * pow(b2, 3) * d0 * d3 * d4 -
              2 * pow(b5, 3) * d0 * d1 * d3 + 4 * pow(b0, 3) * d2 * d4 * d5 -
              b0 * pow(b3, 2) * pow(d0, 2) * d1 +
              pow(b0, 2) * b1 * d0 * pow(d3, 2) +
              4 * b0 * pow(b4, 2) * pow(d0, 2) * d1 -
              4 * pow(b0, 2) * b1 * d0 * pow(d4, 2) +
              4 * pow(b0, 2) * b1 * pow(d1, 2) * d3 -
              4 * pow(b1, 2) * b3 * pow(d0, 2) * d1 -
              4 * b0 * pow(b2, 2) * d1 * pow(d3, 2) +
              4 * b0 * pow(b3, 2) * d1 * pow(d2, 2) -
              b0 * pow(b5, 2) * pow(d0, 2) * d1 -
              4 * b1 * pow(b2, 2) * d0 * pow(d3, 2) +
              4 * b1 * pow(b2, 2) * pow(d0, 2) * d3 +
              4 * b1 * pow(b3, 2) * d0 * pow(d2, 2) +
              pow(b0, 2) * b1 * d0 * pow(d5, 2) +
              4 * pow(b0, 2) * b1 * pow(d2, 2) * d3 -
              4 * pow(b0, 2) * b3 * d1 * pow(d2, 2) -
              4 * pow(b2, 2) * b3 * pow(d0, 2) * d1 -
              8 * b0 * pow(b1, 2) * d1 * pow(d5, 2) +
              16 * b0 * pow(b2, 2) * d1 * pow(d4, 2) +
              16 * b0 * pow(b4, 2) * d1 * pow(d2, 2) -
              16 * b1 * pow(b2, 2) * d0 * pow(d4, 2) +
              16 * b1 * pow(b2, 2) * pow(d1, 2) * d3 -
              16 * b1 * pow(b4, 2) * d0 * pow(d2, 2) +
              8 * b1 * pow(b5, 2) * d0 * pow(d1, 2) -
              4 * pow(b0, 2) * b1 * pow(d1, 2) * d5 -
              8 * pow(b0, 2) * b2 * pow(d1, 2) * d4 +
              8 * pow(b0, 2) * b4 * pow(d1, 2) * d2 -
              8 * pow(b1, 2) * b2 * pow(d0, 2) * d4 -
              16 * pow(b1, 2) * b3 * d1 * pow(d2, 2) +
              8 * pow(b1, 2) * b4 * pow(d0, 2) * d2 +
              4 * pow(b1, 2) * b5 * pow(d0, 2) * d1 -
              4 * b0 * pow(b2, 2) * d1 * pow(d5, 2) -
              4 * b0 * pow(b5, 2) * d1 * pow(d2, 2) +
              4 * b1 * pow(b2, 2) * d0 * pow(d5, 2) -
              4 * b1 * pow(b2, 2) * pow(d0, 2) * d5 +
              4 * b1 * pow(b4, 2) * pow(d0, 2) * d3 +
              4 * b1 * pow(b5, 2) * d0 * pow(d2, 2) -
              4 * b3 * pow(b4, 2) * pow(d0, 2) * d1 +
              4 * pow(b0, 2) * b1 * d3 * pow(d4, 2) -
              4 * pow(b0, 2) * b1 * pow(d2, 2) * d5 -
              8 * pow(b0, 2) * b2 * pow(d2, 2) * d4 -
              4 * pow(b0, 2) * b3 * d1 * pow(d4, 2) +
              4 * pow(b0, 2) * b5 * d1 * pow(d2, 2) +
              8 * pow(b2, 2) * b4 * pow(d0, 2) * d2 +
              4 * pow(b2, 2) * b5 * pow(d0, 2) * d1 -
              b0 * pow(b3, 2) * d1 * pow(d5, 2) -
              3 * b0 * pow(b5, 2) * d1 * pow(d3, 2) -
              16 * b1 * pow(b2, 2) * pow(d1, 2) * d5 +
              3 * b1 * pow(b3, 2) * d0 * pow(d5, 2) -
              3 * b1 * pow(b3, 2) * pow(d0, 2) * d5 +
              b1 * pow(b5, 2) * d0 * pow(d3, 2) -
              2 * b1 * pow(b5, 2) * pow(d0, 2) * d3 +
              2 * b2 * pow(b3, 2) * pow(d0, 2) * d4 -
              2 * b3 * pow(b5, 2) * pow(d0, 2) * d1 +
              2 * pow(b0, 2) * b1 * d3 * pow(d5, 2) -
              pow(b0, 2) * b1 * pow(d3, 2) * d5 -
              2 * pow(b0, 2) * b2 * pow(d3, 2) * d4 +
              2 * pow(b0, 2) * b3 * d1 * pow(d5, 2) -
              2 * pow(b0, 2) * b4 * d2 * pow(d3, 2) +
              3 * pow(b0, 2) * b5 * d1 * pow(d3, 2) -
              32 * pow(b1, 2) * b2 * pow(d2, 2) * d4 +
              16 * pow(b1, 2) * b5 * d1 * pow(d2, 2) +
              32 * pow(b2, 2) * b4 * pow(d1, 2) * d2 +
              2 * pow(b3, 2) * b4 * pow(d0, 2) * d2 +
              pow(b3, 2) * b5 * pow(d0, 2) * d1 +
              4 * b0 * pow(b4, 2) * d1 * pow(d5, 2) +
              4 * b0 * pow(b5, 2) * d1 * pow(d4, 2) +
              16 * b1 * pow(b2, 2) * d3 * pow(d4, 2) -
              4 * b1 * pow(b4, 2) * d0 * pow(d5, 2) +
              8 * b1 * pow(b4, 2) * pow(d0, 2) * d5 +
              16 * b1 * pow(b4, 2) * pow(d2, 2) * d3 -
              4 * b1 * pow(b5, 2) * d0 * pow(d4, 2) +
              4 * b1 * pow(b5, 2) * pow(d1, 2) * d3 -
              16 * b3 * pow(b4, 2) * d1 * pow(d2, 2) -
              8 * pow(b0, 2) * b5 * d1 * pow(d4, 2) -
              4 * pow(b1, 2) * b3 * d1 * pow(d5, 2) -
              16 * pow(b2, 2) * b3 * d1 * pow(d4, 2) -
              4 * b1 * pow(b2, 2) * d3 * pow(d5, 2) +
              4 * b1 * pow(b2, 2) * pow(d3, 2) * d5 -
              4 * b1 * pow(b3, 2) * pow(d2, 2) * d5 -
              b1 * pow(b5, 2) * pow(d0, 2) * d5 -
              4 * b1 * pow(b5, 2) * pow(d2, 2) * d3 -
              8 * b2 * pow(b3, 2) * pow(d2, 2) * d4 -
              2 * b2 * pow(b5, 2) * pow(d0, 2) * d4 +
              4 * b3 * pow(b5, 2) * d1 * pow(d2, 2) -
              2 * b4 * pow(b5, 2) * pow(d0, 2) * d2 +
              2 * pow(b0, 2) * b2 * d4 * pow(d5, 2) +
              2 * pow(b0, 2) * b4 * d2 * pow(d5, 2) +
              pow(b0, 2) * b5 * d1 * pow(d5, 2) +
              4 * pow(b2, 2) * b3 * d1 * pow(d5, 2) +
              8 * pow(b2, 2) * b4 * d2 * pow(d3, 2) +
              4 * pow(b2, 2) * b5 * d1 * pow(d3, 2) -
              4 * pow(b3, 2) * b5 * d1 * pow(d2, 2) -
              12 * b1 * pow(b5, 2) * pow(d1, 2) * d5 -
              8 * b2 * pow(b5, 2) * pow(d1, 2) * d4 -
              8 * b4 * pow(b5, 2) * pow(d1, 2) * d2 +
              8 * pow(b1, 2) * b2 * d4 * pow(d5, 2) +
              8 * pow(b1, 2) * b4 * d2 * pow(d5, 2) +
              12 * pow(b1, 2) * b5 * d1 * pow(d5, 2) +
              4 * b1 * pow(b4, 2) * d3 * pow(d5, 2) +
              4 * b1 * pow(b5, 2) * d3 * pow(d4, 2) -
              4 * b3 * pow(b4, 2) * d1 * pow(d5, 2) -
              4 * b3 * pow(b5, 2) * d1 * pow(d4, 2) -
              b1 * pow(b5, 2) * pow(d3, 2) * d5 +
              2 * b2 * pow(b3, 2) * d4 * pow(d5, 2) -
              2 * b2 * pow(b5, 2) * pow(d3, 2) * d4 -
              2 * b4 * pow(b5, 2) * d2 * pow(d3, 2) +
              2 * pow(b3, 2) * b4 * d2 * pow(d5, 2) +
              pow(b3, 2) * b5 * d1 * pow(d5, 2) +
              8 * b0 * b1 * b3 * d0 * pow(d1, 2) -
              2 * b0 * b1 * b3 * pow(d0, 2) * d3 -
              8 * b0 * b1 * b5 * d0 * pow(d1, 2) +
              8 * b0 * b1 * b2 * d2 * pow(d3, 2) -
              16 * b0 * b2 * b4 * d0 * pow(d2, 2) -
              4 * b0 * b1 * b3 * d0 * pow(d5, 2) +
              2 * b0 * b1 * b3 * pow(d0, 2) * d5 -
              8 * b0 * b1 * b3 * pow(d2, 2) * d3 +
              8 * b0 * b1 * b4 * pow(d0, 2) * d4 -
              2 * b0 * b1 * b5 * d0 * pow(d3, 2) +
              2 * b0 * b1 * b5 * pow(d0, 2) * d3 -
              4 * b0 * b2 * b3 * pow(d0, 2) * d4 +
              4 * b0 * b2 * b4 * d0 * pow(d3, 2) -
              4 * b0 * b2 * b4 * pow(d0, 2) * d3 -
              4 * b0 * b3 * b4 * pow(d0, 2) * d2 +
              2 * b0 * b3 * b5 * pow(d0, 2) * d1 +
              32 * b1 * b2 * b3 * pow(d1, 2) * d2 -
              8 * b0 * b1 * b3 * pow(d1, 2) * d5 +
              8 * b0 * b1 * b5 * d0 * pow(d4, 2) -
              8 * b0 * b1 * b5 * pow(d1, 2) * d3 -
              16 * b0 * b3 * b4 * pow(d1, 2) * d2 -
              64 * b1 * b2 * b4 * d1 * pow(d2, 2) -
              8 * b1 * b3 * b5 * d0 * pow(d1, 2) +
              16 * b2 * b3 * b4 * d0 * pow(d1, 2) +
              8 * b0 * b1 * b3 * pow(d2, 2) * d5 +
              2 * b0 * b1 * b5 * d0 * pow(d5, 2) -
              2 * b0 * b1 * b5 * pow(d0, 2) * d5 +
              16 * b0 * b2 * b3 * pow(d2, 2) * d4 -
              4 * b0 * b2 * b4 * d0 * pow(d5, 2) +
              4 * b0 * b2 * b4 * pow(d0, 2) * d5 +
              16 * b0 * b2 * b4 * pow(d2, 2) * d3 +
              4 * b0 * b2 * b5 * pow(d0, 2) * d4 +
              4 * b0 * b4 * b5 * pow(d0, 2) * d2 -
              32 * b1 * b2 * b5 * pow(d1, 2) * d2 -
              8 * b1 * b3 * b5 * d0 * pow(d2, 2) +
              16 * b2 * b3 * b4 * d0 * pow(d2, 2) -
              2 * b0 * b1 * b3 * d3 * pow(d5, 2) +
              16 * b0 * b1 * b5 * pow(d1, 2) * d5 +
              16 * b0 * b2 * b5 * pow(d1, 2) * d4 +
              2 * b1 * b3 * b5 * pow(d0, 2) * d3 -
              4 * b2 * b3 * b4 * pow(d0, 2) * d3 -
              16 * b2 * b4 * b5 * d0 * pow(d1, 2) -
              8 * b0 * b1 * b5 * d3 * pow(d4, 2) +
              8 * b0 * b3 * b5 * d1 * pow(d4, 2) -
              16 * b1 * b2 * b4 * d1 * pow(d5, 2) -
              8 * b1 * b2 * b5 * d2 * pow(d3, 2) -
              2 * b0 * b1 * b5 * d3 * pow(d5, 2) +
              2 * b0 * b1 * b5 * pow(d3, 2) * d5 -
              4 * b0 * b2 * b3 * d4 * pow(d5, 2) +
              4 * b0 * b2 * b4 * d3 * pow(d5, 2) -
              4 * b0 * b2 * b4 * pow(d3, 2) * d5 +
              4 * b0 * b2 * b5 * pow(d3, 2) * d4 -
              4 * b0 * b3 * b4 * d2 * pow(d5, 2) -
              2 * b0 * b3 * b5 * d1 * pow(d5, 2) +
              4 * b0 * b4 * b5 * d2 * pow(d3, 2) -
              2 * b1 * b3 * b5 * d0 * pow(d5, 2) +
              4 * b1 * b3 * b5 * pow(d0, 2) * d5 +
              8 * b1 * b3 * b5 * pow(d2, 2) * d3 -
              8 * b1 * b4 * b5 * pow(d0, 2) * d4 +
              4 * b2 * b3 * b4 * d0 * pow(d5, 2) -
              8 * b2 * b3 * b4 * pow(d0, 2) * d5 -
              16 * b2 * b3 * b4 * pow(d2, 2) * d3 -
              4 * b2 * b4 * b5 * d0 * pow(d3, 2) +
              8 * b2 * b4 * b5 * pow(d0, 2) * d3 +
              8 * b1 * b3 * b5 * pow(d1, 2) * d5 -
              16 * b2 * b3 * b4 * pow(d1, 2) * d5 +
              16 * b3 * b4 * b5 * pow(d1, 2) * d2 +
              4 * b2 * b4 * b5 * pow(d0, 2) * d5 +
              2 * b1 * b3 * b5 * d3 * pow(d5, 2) -
              4 * b2 * b3 * b4 * d3 * pow(d5, 2) +
              16 * b2 * b4 * b5 * pow(d1, 2) * d5 +
              4 * b2 * b4 * b5 * pow(d3, 2) * d5 -
              8 * b0 * pow(b1, 2) * d0 * d1 * d3 +
              8 * b0 * pow(b1, 2) * d0 * d1 * d5 +
              2 * pow(b0, 2) * b3 * d0 * d1 * d3 +
              16 * b0 * pow(b2, 2) * d0 * d2 * d4 -
              8 * b2 * pow(b3, 2) * d0 * d1 * d2 +
              2 * b0 * pow(b3, 2) * d0 * d1 * d5 -
              4 * b0 * pow(b3, 2) * d0 * d2 * d4 +
              4 * b0 * pow(b5, 2) * d0 * d1 * d3 -
              2 * pow(b0, 2) * b1 * d0 * d3 * d5 +
              4 * pow(b0, 2) * b2 * d0 * d3 * d4 -
              2 * pow(b0, 2) * b3 * d0 * d1 * d5 +
              4 * pow(b0, 2) * b3 * d0 * d2 * d4 -
              8 * pow(b0, 2) * b4 * d0 * d1 * d4 +
              4 * pow(b0, 2) * b4 * d0 * d2 * d3 -
              2 * pow(b0, 2) * b5 * d0 * d1 * d3 -
              32 * pow(b1, 2) * b2 * d1 * d2 * d3 +
              8 * pow(b2, 2) * b3 * d0 * d1 * d3 +
              8 * b0 * pow(b1, 2) * d1 * d3 * d5 -
              16 * b0 * pow(b1, 2) * d2 * d3 * d4 -
              8 * b0 * pow(b4, 2) * d0 * d1 * d5 +
              64 * b1 * pow(b2, 2) * d1 * d2 * d4 +
              16 * pow(b1, 2) * b2 * d0 * d3 * d4 +
              8 * pow(b1, 2) * b3 * d0 * d1 * d5 +
              8 * pow(b1, 2) * b5 * d0 * d1 * d3 +
              8 * b0 * pow(b2, 2) * d1 * d3 * d5 -
              16 * b0 * pow(b2, 2) * d2 * d3 * d4 -
              2 * b0 * pow(b5, 2) * d0 * d1 * d5 +
              4 * b0 * pow(b5, 2) * d0 * d2 * d4 -
              4 * pow(b0, 2) * b2 * d0 * d4 * d5 -
              4 * pow(b0, 2) * b4 * d0 * d2 * d5 +
              2 * pow(b0, 2) * b5 * d0 * d1 * d5 -
              4 * pow(b0, 2) * b5 * d0 * d2 * d4 +
              32 * pow(b1, 2) * b2 * d1 * d2 * d5 -
              16 * pow(b2, 2) * b3 * d0 * d2 * d4 -
              16 * pow(b2, 2) * b4 * d0 * d2 * d3 -
              8 * pow(b2, 2) * b5 * d0 * d1 * d3 +
              16 * b0 * pow(b1, 2) * d2 * d4 * d5 +
              2 * b3 * pow(b5, 2) * d0 * d1 * d3 -
              2 * pow(b0, 2) * b3 * d1 * d3 * d5 +
              4 * pow(b0, 2) * b3 * d2 * d3 * d4 -
              16 * pow(b1, 2) * b4 * d0 * d2 * d5 -
              16 * pow(b1, 2) * b5 * d0 * d1 * d5 -
              8 * b1 * pow(b4, 2) * d0 * d3 * d5 +
              16 * b1 * pow(b5, 2) * d1 * d2 * d4 +
              8 * b2 * pow(b3, 2) * d1 * d2 * d5 +
              8 * b3 * pow(b4, 2) * d0 * d1 * d5 +
              4 * b0 * pow(b3, 2) * d2 * d4 * d5 +
              2 * b0 * pow(b5, 2) * d1 * d3 * d5 -
              4 * b0 * pow(b5, 2) * d2 * d3 * d4 +
              2 * b1 * pow(b5, 2) * d0 * d3 * d5 -
              4 * b2 * pow(b3, 2) * d0 * d4 * d5 +
              4 * b2 * pow(b5, 2) * d0 * d3 * d4 +
              2 * b3 * pow(b5, 2) * d0 * d1 * d5 -
              4 * b3 * pow(b5, 2) * d0 * d2 * d4 +
              4 * b4 * pow(b5, 2) * d0 * d2 * d3 -
              8 * pow(b0, 2) * b3 * d2 * d4 * d5 +
              8 * pow(b0, 2) * b4 * d1 * d4 * d5 -
              4 * pow(b0, 2) * b5 * d1 * d3 * d5 +
              8 * pow(b0, 2) * b5 * d2 * d3 * d4 -
              8 * pow(b2, 2) * b3 * d1 * d3 * d5 +
              16 * pow(b2, 2) * b3 * d2 * d3 * d4 -
              4 * pow(b3, 2) * b4 * d0 * d2 * d5 -
              2 * pow(b3, 2) * b5 * d0 * d1 * d5 +
              4 * pow(b3, 2) * b5 * d0 * d2 * d4 -
              16 * pow(b1, 2) * b2 * d3 * d4 * d5 -
              8 * pow(b1, 2) * b5 * d1 * d3 * d5 +
              16 * pow(b1, 2) * b5 * d2 * d3 * d4 -
              4 * pow(b0, 2) * b5 * d2 * d4 * d5 -
              2 * b3 * pow(b5, 2) * d1 * d3 * d5 +
              4 * b3 * pow(b5, 2) * d2 * d3 * d4 -
              16 * pow(b1, 2) * b5 * d2 * d4 * d5 -
              4 * pow(b3, 2) * b5 * d2 * d4 * d5 +
              16 * b0 * b1 * b2 * d0 * d1 * d4 -
              8 * b0 * b1 * b2 * d0 * d2 * d3 -
              16 * b0 * b1 * b4 * d0 * d1 * d2 +
              8 * b0 * b2 * b3 * d0 * d1 * d2 +
              8 * b0 * b1 * b2 * d0 * d2 * d5 -
              8 * b0 * b2 * b5 * d0 * d1 * d2 +
              16 * b0 * b1 * b3 * d1 * d2 * d4 +
              16 * b0 * b1 * b4 * d1 * d2 * d3 -
              16 * b1 * b2 * b3 * d0 * d1 * d4 -
              16 * b1 * b2 * b4 * d0 * d1 * d3 +
              4 * b0 * b1 * b3 * d0 * d3 * d5 -
              8 * b0 * b1 * b4 * d0 * d3 * d4 +
              8 * b0 * b3 * b4 * d0 * d1 * d4 -
              4 * b0 * b3 * b5 * d0 * d1 * d3 -
              16 * b0 * b1 * b2 * d1 * d4 * d5 -
              8 * b0 * b1 * b2 * d2 * d3 * d5 -
              16 * b0 * b1 * b5 * d1 * d2 * d4 -
              8 * b0 * b2 * b3 * d1 * d2 * d5 -
              32 * b0 * b2 * b4 * d1 * d2 * d4 +
              16 * b1 * b2 * b4 * d0 * d1 * d5 +
              32 * b1 * b2 * b4 * d0 * d2 * d4 +
              8 * b1 * b2 * b5 * d0 * d2 * d3 +
              16 * b1 * b4 * b5 * d0 * d1 * d2 +
              8 * b2 * b3 * b5 * d0 * d1 * d2 -
              8 * b0 * b1 * b4 * d0 * d4 * d5 +
              8 * b0 * b2 * b3 * d0 * d4 * d5 -
              8 * b0 * b2 * b5 * d0 * d3 * d4 +
              8 * b0 * b3 * b4 * d0 * d2 * d5 +
              8 * b0 * b4 * b5 * d0 * d1 * d4 -
              8 * b0 * b4 * b5 * d0 * d2 * d3 +
              8 * b0 * b2 * b5 * d1 * d2 * d5 -
              8 * b1 * b2 * b5 * d0 * d2 * d5 +
              16 * b1 * b2 * b3 * d1 * d4 * d5 +
              16 * b1 * b2 * b4 * d1 * d3 * d5 -
              32 * b1 * b2 * b4 * d2 * d3 * d4 -
              16 * b1 * b3 * b5 * d1 * d2 * d4 -
              16 * b1 * b4 * b5 * d1 * d2 * d3 +
              32 * b2 * b3 * b4 * d1 * d2 * d4 +
              8 * b0 * b1 * b4 * d3 * d4 * d5 -
              8 * b0 * b3 * b4 * d1 * d4 * d5 +
              4 * b0 * b3 * b5 * d1 * d3 * d5 -
              8 * b0 * b3 * b5 * d2 * d3 * d4 -
              4 * b1 * b3 * b5 * d0 * d3 * d5 +
              8 * b1 * b4 * b5 * d0 * d3 * d4 +
              8 * b2 * b3 * b4 * d0 * d3 * d5 -
              8 * b3 * b4 * b5 * d0 * d1 * d4 +
              8 * b1 * b2 * b5 * d2 * d3 * d5 -
              8 * b2 * b3 * b5 * d1 * d2 * d5 +
              8 * b0 * b3 * b5 * d2 * d4 * d5 -
              8 * b0 * b4 * b5 * d1 * d4 * d5 +
              8 * b1 * b4 * b5 * d0 * d4 * d5 -
              8 * b2 * b4 * b5 * d0 * d3 * d5 -
              8 * b1 * b4 * b5 * d3 * d4 * d5 + 8 * b3 * b4 * b5 * d1 * d4 * d5,
      4 * pow(b1, 3) * c3 * pow(d0, 2) - pow(b0, 3) * c1 * pow(d3, 2) +
          4 * pow(b0, 3) * c1 * pow(d4, 2) - pow(b0, 3) * c1 * pow(d5, 2) +
          8 * pow(b1, 3) * c0 * pow(d5, 2) + 16 * pow(b1, 3) * c3 * pow(d2, 2) -
          4 * pow(b1, 3) * c5 * pow(d0, 2) - 8 * pow(b2, 3) * c4 * pow(d0, 2) +
          pow(b5, 3) * c1 * pow(d0, 2) - 32 * pow(b2, 3) * c4 * pow(d1, 2) +
          12 * pow(b5, 3) * c1 * pow(d1, 2) -
          16 * pow(b1, 3) * c5 * pow(d2, 2) + 4 * pow(b1, 3) * c3 * pow(d5, 2) -
          8 * pow(b2, 3) * c4 * pow(d3, 2) + pow(b5, 3) * c1 * pow(d3, 2) -
          12 * pow(b1, 3) * c5 * pow(d5, 2) + 8 * pow(b1, 3) * c0 * d0 * d3 -
          8 * pow(b1, 3) * c0 * d0 * d5 - 16 * pow(b2, 3) * c0 * d0 * d4 +
          2 * pow(b5, 3) * c0 * d0 * d1 - 2 * pow(b0, 3) * c3 * d1 * d3 +
          32 * pow(b1, 3) * c2 * d2 * d3 - 64 * pow(b2, 3) * c1 * d1 * d4 +
          2 * pow(b0, 3) * c1 * d3 * d5 - 4 * pow(b0, 3) * c2 * d3 * d4 +
          2 * pow(b0, 3) * c3 * d1 * d5 - 4 * pow(b0, 3) * c3 * d2 * d4 +
          8 * pow(b0, 3) * c4 * d1 * d4 - 4 * pow(b0, 3) * c4 * d2 * d3 +
          2 * pow(b0, 3) * c5 * d1 * d3 - 8 * pow(b1, 3) * c0 * d3 * d5 -
          8 * pow(b1, 3) * c3 * d0 * d5 - 8 * pow(b1, 3) * c5 * d0 * d3 +
          16 * pow(b2, 3) * c0 * d3 * d4 + 16 * pow(b2, 3) * c3 * d0 * d4 +
          16 * pow(b2, 3) * c4 * d0 * d3 - 2 * pow(b5, 3) * c0 * d1 * d3 -
          2 * pow(b5, 3) * c1 * d0 * d3 - 2 * pow(b5, 3) * c3 * d0 * d1 -
          32 * pow(b1, 3) * c2 * d2 * d5 + 4 * pow(b0, 3) * c2 * d4 * d5 +
          4 * pow(b0, 3) * c4 * d2 * d5 - 2 * pow(b0, 3) * c5 * d1 * d5 +
          4 * pow(b0, 3) * c5 * d2 * d4 + 16 * pow(b1, 3) * c5 * d0 * d5 -
          16 * pow(b2, 3) * c3 * d3 * d4 + 2 * pow(b5, 3) * c3 * d1 * d3 +
          8 * pow(b1, 3) * c5 * d3 * d5 - b0 * pow(b3, 2) * c1 * pow(d0, 2) +
          3 * b1 * pow(b3, 2) * c0 * pow(d0, 2) +
          pow(b0, 2) * b1 * c0 * pow(d3, 2) +
          4 * b0 * pow(b4, 2) * c1 * pow(d0, 2) -
          12 * b1 * pow(b4, 2) * c0 * pow(d0, 2) -
          4 * pow(b0, 2) * b1 * c0 * pow(d4, 2) +
          4 * pow(b0, 2) * b1 * c3 * pow(d1, 2) -
          12 * pow(b0, 2) * b3 * c1 * pow(d1, 2) -
          4 * pow(b1, 2) * b3 * c1 * pow(d0, 2) -
          4 * b0 * pow(b2, 2) * c1 * pow(d3, 2) +
          4 * b0 * pow(b3, 2) * c1 * pow(d2, 2) -
          b0 * pow(b5, 2) * c1 * pow(d0, 2) -
          4 * b1 * pow(b2, 2) * c0 * pow(d3, 2) +
          4 * b1 * pow(b2, 2) * c3 * pow(d0, 2) +
          4 * b1 * pow(b3, 2) * c0 * pow(d2, 2) +
          3 * b1 * pow(b5, 2) * c0 * pow(d0, 2) +
          pow(b0, 2) * b1 * c0 * pow(d5, 2) +
          4 * pow(b0, 2) * b1 * c3 * pow(d2, 2) -
          4 * pow(b0, 2) * b3 * c1 * pow(d2, 2) -
          4 * pow(b2, 2) * b3 * c1 * pow(d0, 2) -
          8 * b0 * pow(b1, 2) * c1 * pow(d5, 2) +
          16 * b0 * pow(b2, 2) * c1 * pow(d4, 2) +
          16 * b0 * pow(b4, 2) * c1 * pow(d2, 2) -
          24 * b0 * pow(b5, 2) * c1 * pow(d1, 2) -
          16 * b1 * pow(b2, 2) * c0 * pow(d4, 2) +
          16 * b1 * pow(b2, 2) * c3 * pow(d1, 2) -
          16 * b1 * pow(b4, 2) * c0 * pow(d2, 2) +
          8 * b1 * pow(b5, 2) * c0 * pow(d1, 2) -
          4 * pow(b0, 2) * b1 * c5 * pow(d1, 2) -
          8 * pow(b0, 2) * b2 * c4 * pow(d1, 2) +
          8 * pow(b0, 2) * b4 * c2 * pow(d1, 2) +
          12 * pow(b0, 2) * b5 * c1 * pow(d1, 2) -
          8 * pow(b1, 2) * b2 * c4 * pow(d0, 2) -
          16 * pow(b1, 2) * b3 * c1 * pow(d2, 2) +
          8 * pow(b1, 2) * b4 * c2 * pow(d0, 2) +
          4 * pow(b1, 2) * b5 * c1 * pow(d0, 2) -
          48 * pow(b2, 2) * b3 * c1 * pow(d1, 2) -
          4 * b0 * pow(b2, 2) * c1 * pow(d5, 2) -
          4 * b0 * pow(b5, 2) * c1 * pow(d2, 2) +
          4 * b1 * pow(b2, 2) * c0 * pow(d5, 2) -
          4 * b1 * pow(b2, 2) * c5 * pow(d0, 2) +
          4 * b1 * pow(b4, 2) * c3 * pow(d0, 2) +
          4 * b1 * pow(b5, 2) * c0 * pow(d2, 2) -
          4 * b3 * pow(b4, 2) * c1 * pow(d0, 2) +
          4 * pow(b0, 2) * b1 * c3 * pow(d4, 2) -
          4 * pow(b0, 2) * b1 * c5 * pow(d2, 2) -
          8 * pow(b0, 2) * b2 * c4 * pow(d2, 2) -
          4 * pow(b0, 2) * b3 * c1 * pow(d4, 2) +
          24 * pow(b0, 2) * b4 * c2 * pow(d2, 2) +
          4 * pow(b0, 2) * b5 * c1 * pow(d2, 2) +
          8 * pow(b2, 2) * b4 * c2 * pow(d0, 2) +
          4 * pow(b2, 2) * b5 * c1 * pow(d0, 2) -
          b0 * pow(b3, 2) * c1 * pow(d5, 2) -
          3 * b0 * pow(b5, 2) * c1 * pow(d3, 2) -
          16 * b1 * pow(b2, 2) * c5 * pow(d1, 2) +
          3 * b1 * pow(b3, 2) * c0 * pow(d5, 2) -
          3 * b1 * pow(b3, 2) * c5 * pow(d0, 2) +
          b1 * pow(b5, 2) * c0 * pow(d3, 2) -
          2 * b1 * pow(b5, 2) * c3 * pow(d0, 2) +
          2 * b2 * pow(b3, 2) * c4 * pow(d0, 2) -
          2 * b3 * pow(b5, 2) * c1 * pow(d0, 2) +
          2 * pow(b0, 2) * b1 * c3 * pow(d5, 2) -
          pow(b0, 2) * b1 * c5 * pow(d3, 2) -
          2 * pow(b0, 2) * b2 * c4 * pow(d3, 2) +
          2 * pow(b0, 2) * b3 * c1 * pow(d5, 2) -
          2 * pow(b0, 2) * b4 * c2 * pow(d3, 2) +
          3 * pow(b0, 2) * b5 * c1 * pow(d3, 2) -
          32 * pow(b1, 2) * b2 * c4 * pow(d2, 2) +
          96 * pow(b1, 2) * b4 * c2 * pow(d2, 2) +
          16 * pow(b1, 2) * b5 * c1 * pow(d2, 2) +
          32 * pow(b2, 2) * b4 * c2 * pow(d1, 2) +
          48 * pow(b2, 2) * b5 * c1 * pow(d1, 2) +
          2 * pow(b3, 2) * b4 * c2 * pow(d0, 2) +
          pow(b3, 2) * b5 * c1 * pow(d0, 2) +
          4 * b0 * pow(b4, 2) * c1 * pow(d5, 2) +
          4 * b0 * pow(b5, 2) * c1 * pow(d4, 2) +
          16 * b1 * pow(b2, 2) * c3 * pow(d4, 2) -
          4 * b1 * pow(b4, 2) * c0 * pow(d5, 2) +
          16 * b1 * pow(b4, 2) * c3 * pow(d2, 2) +
          8 * b1 * pow(b4, 2) * c5 * pow(d0, 2) -
          4 * b1 * pow(b5, 2) * c0 * pow(d4, 2) +
          4 * b1 * pow(b5, 2) * c3 * pow(d1, 2) -
          16 * b3 * pow(b4, 2) * c1 * pow(d2, 2) -
          12 * b3 * pow(b5, 2) * c1 * pow(d1, 2) -
          8 * pow(b0, 2) * b5 * c1 * pow(d4, 2) -
          4 * pow(b1, 2) * b3 * c1 * pow(d5, 2) -
          16 * pow(b2, 2) * b3 * c1 * pow(d4, 2) -
          4 * b1 * pow(b2, 2) * c3 * pow(d5, 2) +
          4 * b1 * pow(b2, 2) * c5 * pow(d3, 2) -
          4 * b1 * pow(b3, 2) * c5 * pow(d2, 2) -
          4 * b1 * pow(b5, 2) * c3 * pow(d2, 2) -
          b1 * pow(b5, 2) * c5 * pow(d0, 2) -
          8 * b2 * pow(b3, 2) * c4 * pow(d2, 2) -
          2 * b2 * pow(b5, 2) * c4 * pow(d0, 2) +
          4 * b3 * pow(b5, 2) * c1 * pow(d2, 2) -
          2 * b4 * pow(b5, 2) * c2 * pow(d0, 2) -
          3 * pow(b0, 2) * b1 * c5 * pow(d5, 2) +
          2 * pow(b0, 2) * b2 * c4 * pow(d5, 2) +
          2 * pow(b0, 2) * b4 * c2 * pow(d5, 2) +
          pow(b0, 2) * b5 * c1 * pow(d5, 2) +
          4 * pow(b2, 2) * b3 * c1 * pow(d5, 2) +
          8 * pow(b2, 2) * b4 * c2 * pow(d3, 2) +
          4 * pow(b2, 2) * b5 * c1 * pow(d3, 2) +
          24 * pow(b3, 2) * b4 * c2 * pow(d2, 2) -
          4 * pow(b3, 2) * b5 * c1 * pow(d2, 2) -
          12 * b1 * pow(b5, 2) * c5 * pow(d1, 2) -
          8 * b2 * pow(b5, 2) * c4 * pow(d1, 2) -
          8 * b4 * pow(b5, 2) * c2 * pow(d1, 2) +
          8 * pow(b1, 2) * b2 * c4 * pow(d5, 2) +
          8 * pow(b1, 2) * b4 * c2 * pow(d5, 2) +
          12 * pow(b1, 2) * b5 * c1 * pow(d5, 2) +
          4 * b1 * pow(b4, 2) * c3 * pow(d5, 2) +
          4 * b1 * pow(b5, 2) * c3 * pow(d4, 2) -
          4 * b3 * pow(b4, 2) * c1 * pow(d5, 2) -
          4 * b3 * pow(b5, 2) * c1 * pow(d4, 2) -
          3 * b1 * pow(b3, 2) * c5 * pow(d5, 2) -
          b1 * pow(b5, 2) * c5 * pow(d3, 2) +
          2 * b2 * pow(b3, 2) * c4 * pow(d5, 2) -
          2 * b2 * pow(b5, 2) * c4 * pow(d3, 2) -
          2 * b4 * pow(b5, 2) * c2 * pow(d3, 2) +
          2 * pow(b3, 2) * b4 * c2 * pow(d5, 2) +
          pow(b3, 2) * b5 * c1 * pow(d5, 2) +
          8 * b0 * b1 * b3 * c0 * pow(d1, 2) -
          2 * b0 * b1 * b3 * c3 * pow(d0, 2) -
          8 * b0 * b1 * b5 * c0 * pow(d1, 2) +
          8 * b0 * b1 * b2 * c2 * pow(d3, 2) -
          16 * b0 * b2 * b4 * c0 * pow(d2, 2) -
          4 * b0 * b1 * b3 * c0 * pow(d5, 2) -
          8 * b0 * b1 * b3 * c3 * pow(d2, 2) +
          2 * b0 * b1 * b3 * c5 * pow(d0, 2) +
          8 * b0 * b1 * b4 * c4 * pow(d0, 2) -
          2 * b0 * b1 * b5 * c0 * pow(d3, 2) +
          2 * b0 * b1 * b5 * c3 * pow(d0, 2) -
          4 * b0 * b2 * b3 * c4 * pow(d0, 2) +
          4 * b0 * b2 * b4 * c0 * pow(d3, 2) -
          4 * b0 * b2 * b4 * c3 * pow(d0, 2) -
          4 * b0 * b3 * b4 * c2 * pow(d0, 2) +
          2 * b0 * b3 * b5 * c1 * pow(d0, 2) +
          32 * b1 * b2 * b3 * c2 * pow(d1, 2) -
          6 * b1 * b3 * b5 * c0 * pow(d0, 2) +
          12 * b2 * b3 * b4 * c0 * pow(d0, 2) -
          8 * b0 * b1 * b3 * c5 * pow(d1, 2) +
          8 * b0 * b1 * b5 * c0 * pow(d4, 2) -
          8 * b0 * b1 * b5 * c3 * pow(d1, 2) -
          16 * b0 * b3 * b4 * c2 * pow(d1, 2) +
          24 * b0 * b3 * b5 * c1 * pow(d1, 2) -
          64 * b1 * b2 * b4 * c1 * pow(d2, 2) -
          8 * b1 * b3 * b5 * c0 * pow(d1, 2) +
          16 * b2 * b3 * b4 * c0 * pow(d1, 2) +
          8 * b0 * b1 * b3 * c5 * pow(d2, 2) +
          2 * b0 * b1 * b5 * c0 * pow(d5, 2) -
          2 * b0 * b1 * b5 * c5 * pow(d0, 2) +
          16 * b0 * b2 * b3 * c4 * pow(d2, 2) -
          4 * b0 * b2 * b4 * c0 * pow(d5, 2) +
          16 * b0 * b2 * b4 * c3 * pow(d2, 2) +
          4 * b0 * b2 * b4 * c5 * pow(d0, 2) +
          4 * b0 * b2 * b5 * c4 * pow(d0, 2) -
          48 * b0 * b3 * b4 * c2 * pow(d2, 2) +
          4 * b0 * b4 * b5 * c2 * pow(d0, 2) -
          32 * b1 * b2 * b5 * c2 * pow(d1, 2) -
          8 * b1 * b3 * b5 * c0 * pow(d2, 2) +
          16 * b2 * b3 * b4 * c0 * pow(d2, 2) -
          12 * b2 * b4 * b5 * c0 * pow(d0, 2) -
          2 * b0 * b1 * b3 * c3 * pow(d5, 2) +
          16 * b0 * b1 * b5 * c5 * pow(d1, 2) +
          16 * b0 * b2 * b5 * c4 * pow(d1, 2) +
          2 * b1 * b3 * b5 * c3 * pow(d0, 2) -
          4 * b2 * b3 * b4 * c3 * pow(d0, 2) -
          16 * b2 * b4 * b5 * c0 * pow(d1, 2) -
          8 * b0 * b1 * b5 * c3 * pow(d4, 2) +
          8 * b0 * b3 * b5 * c1 * pow(d4, 2) -
          16 * b1 * b2 * b4 * c1 * pow(d5, 2) -
          8 * b1 * b2 * b5 * c2 * pow(d3, 2) +
          6 * b0 * b1 * b3 * c5 * pow(d5, 2) -
          2 * b0 * b1 * b5 * c3 * pow(d5, 2) +
          2 * b0 * b1 * b5 * c5 * pow(d3, 2) -
          4 * b0 * b2 * b3 * c4 * pow(d5, 2) +
          4 * b0 * b2 * b4 * c3 * pow(d5, 2) -
          4 * b0 * b2 * b4 * c5 * pow(d3, 2) +
          4 * b0 * b2 * b5 * c4 * pow(d3, 2) -
          4 * b0 * b3 * b4 * c2 * pow(d5, 2) -
          2 * b0 * b3 * b5 * c1 * pow(d5, 2) +
          4 * b0 * b4 * b5 * c2 * pow(d3, 2) -
          2 * b1 * b3 * b5 * c0 * pow(d5, 2) +
          8 * b1 * b3 * b5 * c3 * pow(d2, 2) +
          4 * b1 * b3 * b5 * c5 * pow(d0, 2) -
          8 * b1 * b4 * b5 * c4 * pow(d0, 2) +
          4 * b2 * b3 * b4 * c0 * pow(d5, 2) -
          16 * b2 * b3 * b4 * c3 * pow(d2, 2) -
          8 * b2 * b3 * b4 * c5 * pow(d0, 2) -
          4 * b2 * b4 * b5 * c0 * pow(d3, 2) +
          8 * b2 * b4 * b5 * c3 * pow(d0, 2) +
          8 * b1 * b3 * b5 * c5 * pow(d1, 2) -
          16 * b2 * b3 * b4 * c5 * pow(d1, 2) +
          16 * b3 * b4 * b5 * c2 * pow(d1, 2) +
          4 * b2 * b4 * b5 * c5 * pow(d0, 2) +
          2 * b1 * b3 * b5 * c3 * pow(d5, 2) -
          4 * b2 * b3 * b4 * c3 * pow(d5, 2) +
          16 * b2 * b4 * b5 * c5 * pow(d1, 2) +
          4 * b2 * b4 * b5 * c5 * pow(d3, 2) -
          2 * b0 * pow(b3, 2) * c0 * d0 * d1 -
          8 * b0 * pow(b1, 2) * c0 * d1 * d3 -
          8 * b0 * pow(b1, 2) * c1 * d0 * d3 -
          8 * b0 * pow(b1, 2) * c3 * d0 * d1 +
          8 * b0 * pow(b4, 2) * c0 * d0 * d1 -
          8 * pow(b1, 2) * b3 * c0 * d0 * d1 -
          2 * b0 * pow(b5, 2) * c0 * d0 * d1 +
          8 * b1 * pow(b2, 2) * c0 * d0 * d3 +
          8 * pow(b0, 2) * b1 * c1 * d1 * d3 -
          8 * pow(b2, 2) * b3 * c0 * d0 * d1 +
          8 * b0 * pow(b1, 2) * c0 * d1 * d5 +
          8 * b0 * pow(b1, 2) * c1 * d0 * d5 +
          8 * b0 * pow(b1, 2) * c5 * d0 * d1 +
          2 * pow(b0, 2) * b1 * c3 * d0 * d3 +
          2 * pow(b0, 2) * b3 * c0 * d1 * d3 +
          2 * pow(b0, 2) * b3 * c1 * d0 * d3 +
          2 * pow(b0, 2) * b3 * c3 * d0 * d1 -
          16 * pow(b1, 2) * b2 * c0 * d0 * d4 +
          16 * pow(b1, 2) * b4 * c0 * d0 * d2 +
          8 * pow(b1, 2) * b5 * c0 * d0 * d1 +
          16 * b0 * pow(b2, 2) * c0 * d2 * d4 +
          16 * b0 * pow(b2, 2) * c2 * d0 * d4 +
          16 * b0 * pow(b2, 2) * c4 * d0 * d2 +
          8 * b0 * pow(b3, 2) * c2 * d1 * d2 -
          8 * b1 * pow(b2, 2) * c0 * d0 * d5 +
          32 * b1 * pow(b2, 2) * c1 * d1 * d3 +
          8 * b1 * pow(b3, 2) * c2 * d0 * d2 +
          8 * b1 * pow(b4, 2) * c0 * d0 * d3 +
          16 * b1 * pow(b5, 2) * c1 * d0 * d1 -
          8 * b2 * pow(b3, 2) * c0 * d1 * d2 -
          8 * b2 * pow(b3, 2) * c1 * d0 * d2 -
          8 * b2 * pow(b3, 2) * c2 * d0 * d1 -
          8 * b3 * pow(b4, 2) * c0 * d0 * d1 -
          8 * pow(b0, 2) * b1 * c1 * d1 * d5 +
          8 * pow(b0, 2) * b1 * c2 * d2 * d3 -
          16 * pow(b0, 2) * b2 * c1 * d1 * d4 -
          8 * pow(b0, 2) * b3 * c2 * d1 * d2 +
          16 * pow(b0, 2) * b4 * c1 * d1 * d2 +
          16 * pow(b2, 2) * b4 * c0 * d0 * d2 +
          8 * pow(b2, 2) * b5 * c0 * d0 * d1 -
          8 * b0 * pow(b2, 2) * c3 * d1 * d3 +
          2 * b0 * pow(b3, 2) * c0 * d1 * d5 -
          4 * b0 * pow(b3, 2) * c0 * d2 * d4 +
          2 * b0 * pow(b3, 2) * c1 * d0 * d5 -
          4 * b0 * pow(b3, 2) * c2 * d0 * d4 -
          4 * b0 * pow(b3, 2) * c4 * d0 * d2 +
          2 * b0 * pow(b3, 2) * c5 * d0 * d1 +
          32 * b0 * pow(b4, 2) * c2 * d1 * d2 +
          4 * b0 * pow(b5, 2) * c0 * d1 * d3 +
          4 * b0 * pow(b5, 2) * c1 * d0 * d3 +
          4 * b0 * pow(b5, 2) * c3 * d0 * d1 -
          8 * b1 * pow(b2, 2) * c3 * d0 * d3 -
          6 * b1 * pow(b3, 2) * c0 * d0 * d5 -
          32 * b1 * pow(b4, 2) * c2 * d0 * d2 -
          4 * b1 * pow(b5, 2) * c0 * d0 * d3 +
          4 * b2 * pow(b3, 2) * c0 * d0 * d4 -
          4 * b3 * pow(b5, 2) * c0 * d0 * d1 -
          2 * pow(b0, 2) * b1 * c0 * d3 * d5 -
          2 * pow(b0, 2) * b1 * c3 * d0 * d5 -
          8 * pow(b0, 2) * b1 * c4 * d0 * d4 -
          2 * pow(b0, 2) * b1 * c5 * d0 * d3 +
          4 * pow(b0, 2) * b2 * c0 * d3 * d4 +
          4 * pow(b0, 2) * b2 * c3 * d0 * d4 +
          4 * pow(b0, 2) * b2 * c4 * d0 * d3 -
          2 * pow(b0, 2) * b3 * c0 * d1 * d5 +
          4 * pow(b0, 2) * b3 * c0 * d2 * d4 -
          2 * pow(b0, 2) * b3 * c1 * d0 * d5 +
          4 * pow(b0, 2) * b3 * c2 * d0 * d4 +
          4 * pow(b0, 2) * b3 * c4 * d0 * d2 -
          2 * pow(b0, 2) * b3 * c5 * d0 * d1 -
          8 * pow(b0, 2) * b4 * c0 * d1 * d4 +
          4 * pow(b0, 2) * b4 * c0 * d2 * d3 -
          8 * pow(b0, 2) * b4 * c1 * d0 * d4 +
          4 * pow(b0, 2) * b4 * c2 * d0 * d3 +
          4 * pow(b0, 2) * b4 * c3 * d0 * d2 -
          8 * pow(b0, 2) * b4 * c4 * d0 * d1 -
          2 * pow(b0, 2) * b5 * c0 * d1 * d3 -
          2 * pow(b0, 2) * b5 * c1 * d0 * d3 -
          2 * pow(b0, 2) * b5 * c3 * d0 * d1 -
          32 * pow(b1, 2) * b2 * c1 * d2 * d3 -
          32 * pow(b1, 2) * b2 * c2 * d1 * d3 -
          32 * pow(b1, 2) * b2 * c3 * d1 * d2 -
          32 * pow(b1, 2) * b3 * c2 * d1 * d2 +
          8 * pow(b2, 2) * b3 * c0 * d1 * d3 +
          8 * pow(b2, 2) * b3 * c1 * d0 * d3 +
          8 * pow(b2, 2) * b3 * c3 * d0 * d1 +
          4 * pow(b3, 2) * b4 * c0 * d0 * d2 +
          2 * pow(b3, 2) * b5 * c0 * d0 * d1 +
          8 * b0 * pow(b1, 2) * c1 * d3 * d5 -
          16 * b0 * pow(b1, 2) * c2 * d3 * d4 +
          8 * b0 * pow(b1, 2) * c3 * d1 * d5 -
          16 * b0 * pow(b1, 2) * c3 * d2 * d4 -
          16 * b0 * pow(b1, 2) * c4 * d2 * d3 +
          8 * b0 * pow(b1, 2) * c5 * d1 * d3 -
          8 * b0 * pow(b4, 2) * c0 * d1 * d5 -
          8 * b0 * pow(b4, 2) * c1 * d0 * d5 -
          8 * b0 * pow(b4, 2) * c5 * d0 * d1 -
          8 * b0 * pow(b5, 2) * c2 * d1 * d2 -
          32 * b1 * pow(b2, 2) * c1 * d1 * d5 +
          64 * b1 * pow(b2, 2) * c1 * d2 * d4 +
          64 * b1 * pow(b2, 2) * c2 * d1 * d4 +
          64 * b1 * pow(b2, 2) * c4 * d1 * d2 +
          16 * b1 * pow(b4, 2) * c0 * d0 * d5 +
          8 * b1 * pow(b5, 2) * c2 * d0 * d2 -
          8 * pow(b0, 2) * b1 * c2 * d2 * d5 -
          16 * pow(b0, 2) * b2 * c2 * d2 * d4 +
          8 * pow(b0, 2) * b5 * c2 * d1 * d2 +
          16 * pow(b1, 2) * b2 * c0 * d3 * d4 +
          16 * pow(b1, 2) * b2 * c3 * d0 * d4 +
          16 * pow(b1, 2) * b2 * c4 * d0 * d3 +
          8 * pow(b1, 2) * b3 * c0 * d1 * d5 +
          8 * pow(b1, 2) * b3 * c1 * d0 * d5 +
          8 * pow(b1, 2) * b3 * c5 * d0 * d1 +
          8 * pow(b1, 2) * b5 * c0 * d1 * d3 +
          8 * pow(b1, 2) * b5 * c1 * d0 * d3 +
          8 * pow(b1, 2) * b5 * c3 * d0 * d1 +
          64 * pow(b2, 2) * b4 * c1 * d1 * d2 +
          8 * b0 * pow(b2, 2) * c1 * d3 * d5 -
          16 * b0 * pow(b2, 2) * c2 * d3 * d4 +
          8 * b0 * pow(b2, 2) * c3 * d1 * d5 -
          16 * b0 * pow(b2, 2) * c3 * d2 * d4 +
          32 * b0 * pow(b2, 2) * c4 * d1 * d4 -
          16 * b0 * pow(b2, 2) * c4 * d2 * d3 +
          8 * b0 * pow(b2, 2) * c5 * d1 * d3 -
          2 * b0 * pow(b5, 2) * c0 * d1 * d5 +
          4 * b0 * pow(b5, 2) * c0 * d2 * d4 -
          2 * b0 * pow(b5, 2) * c1 * d0 * d5 +
          4 * b0 * pow(b5, 2) * c2 * d0 * d4 +
          4 * b0 * pow(b5, 2) * c4 * d0 * d2 -
          2 * b0 * pow(b5, 2) * c5 * d0 * d1 -
          32 * b1 * pow(b2, 2) * c4 * d0 * d4 -
          2 * b1 * pow(b5, 2) * c0 * d0 * d5 +
          8 * b1 * pow(b5, 2) * c1 * d1 * d3 -
          4 * b2 * pow(b5, 2) * c0 * d0 * d4 -
          4 * b4 * pow(b5, 2) * c0 * d0 * d2 +
          2 * pow(b0, 2) * b1 * c5 * d0 * d5 -
          4 * pow(b0, 2) * b2 * c0 * d4 * d5 -
          4 * pow(b0, 2) * b2 * c4 * d0 * d5 -
          4 * pow(b0, 2) * b2 * c5 * d0 * d4 -
          4 * pow(b0, 2) * b4 * c0 * d2 * d5 -
          4 * pow(b0, 2) * b4 * c2 * d0 * d5 -
          4 * pow(b0, 2) * b4 * c5 * d0 * d2 +
          2 * pow(b0, 2) * b5 * c0 * d1 * d5 -
          4 * pow(b0, 2) * b5 * c0 * d2 * d4 +
          2 * pow(b0, 2) * b5 * c1 * d0 * d5 -
          4 * pow(b0, 2) * b5 * c2 * d0 * d4 -
          4 * pow(b0, 2) * b5 * c4 * d0 * d2 +
          2 * pow(b0, 2) * b5 * c5 * d0 * d1 +
          32 * pow(b1, 2) * b2 * c1 * d2 * d5 +
          32 * pow(b1, 2) * b2 * c2 * d1 * d5 -
          64 * pow(b1, 2) * b2 * c2 * d2 * d4 +
          32 * pow(b1, 2) * b2 * c5 * d1 * d2 +
          32 * pow(b1, 2) * b5 * c2 * d1 * d2 -
          16 * pow(b2, 2) * b3 * c0 * d2 * d4 -
          16 * pow(b2, 2) * b3 * c2 * d0 * d4 -
          16 * pow(b2, 2) * b3 * c4 * d0 * d2 -
          16 * pow(b2, 2) * b4 * c0 * d2 * d3 -
          16 * pow(b2, 2) * b4 * c2 * d0 * d3 -
          16 * pow(b2, 2) * b4 * c3 * d0 * d2 -
          8 * pow(b2, 2) * b5 * c0 * d1 * d3 -
          8 * pow(b2, 2) * b5 * c1 * d0 * d3 -
          8 * pow(b2, 2) * b5 * c3 * d0 * d1 +
          16 * b0 * pow(b1, 2) * c2 * d4 * d5 +
          16 * b0 * pow(b1, 2) * c4 * d2 * d5 -
          16 * b0 * pow(b1, 2) * c5 * d1 * d5 +
          16 * b0 * pow(b1, 2) * c5 * d2 * d4 -
          6 * b0 * pow(b5, 2) * c3 * d1 * d3 +
          32 * b1 * pow(b4, 2) * c2 * d2 * d3 +
          2 * b1 * pow(b5, 2) * c3 * d0 * d3 -
          32 * b3 * pow(b4, 2) * c2 * d1 * d2 +
          2 * b3 * pow(b5, 2) * c0 * d1 * d3 +
          2 * b3 * pow(b5, 2) * c1 * d0 * d3 +
          2 * b3 * pow(b5, 2) * c3 * d0 * d1 -
          2 * pow(b0, 2) * b1 * c3 * d3 * d5 +
          8 * pow(b0, 2) * b1 * c4 * d3 * d4 -
          4 * pow(b0, 2) * b2 * c3 * d3 * d4 -
          2 * pow(b0, 2) * b3 * c1 * d3 * d5 +
          4 * pow(b0, 2) * b3 * c2 * d3 * d4 -
          2 * pow(b0, 2) * b3 * c3 * d1 * d5 +
          4 * pow(b0, 2) * b3 * c3 * d2 * d4 -
          8 * pow(b0, 2) * b3 * c4 * d1 * d4 +
          4 * pow(b0, 2) * b3 * c4 * d2 * d3 -
          2 * pow(b0, 2) * b3 * c5 * d1 * d3 -
          4 * pow(b0, 2) * b4 * c3 * d2 * d3 +
          6 * pow(b0, 2) * b5 * c3 * d1 * d3 -
          16 * pow(b1, 2) * b4 * c0 * d2 * d5 -
          16 * pow(b1, 2) * b4 * c2 * d0 * d5 -
          16 * pow(b1, 2) * b4 * c5 * d0 * d2 -
          16 * pow(b1, 2) * b5 * c0 * d1 * d5 -
          16 * pow(b1, 2) * b5 * c1 * d0 * d5 -
          16 * pow(b1, 2) * b5 * c5 * d0 * d1 -
          8 * b0 * pow(b2, 2) * c5 * d1 * d5 +
          8 * b1 * pow(b2, 2) * c5 * d0 * d5 -
          8 * b1 * pow(b3, 2) * c2 * d2 * d5 -
          8 * b1 * pow(b4, 2) * c0 * d3 * d5 -
          8 * b1 * pow(b4, 2) * c3 * d0 * d5 -
          8 * b1 * pow(b4, 2) * c5 * d0 * d3 -
          24 * b1 * pow(b5, 2) * c1 * d1 * d5 +
          16 * b1 * pow(b5, 2) * c1 * d2 * d4 +
          16 * b1 * pow(b5, 2) * c2 * d1 * d4 -
          8 * b1 * pow(b5, 2) * c2 * d2 * d3 +
          16 * b1 * pow(b5, 2) * c4 * d1 * d2 +
          8 * b2 * pow(b3, 2) * c1 * d2 * d5 +
          8 * b2 * pow(b3, 2) * c2 * d1 * d5 -
          16 * b2 * pow(b3, 2) * c2 * d2 * d4 +
          8 * b2 * pow(b3, 2) * c5 * d1 * d2 -
          16 * b2 * pow(b5, 2) * c1 * d1 * d4 +
          8 * b3 * pow(b4, 2) * c0 * d1 * d5 +
          8 * b3 * pow(b4, 2) * c1 * d0 * d5 +
          8 * b3 * pow(b4, 2) * c5 * d0 * d1 +
          8 * b3 * pow(b5, 2) * c2 * d1 * d2 -
          16 * b4 * pow(b5, 2) * c1 * d1 * d2 -
          8 * pow(b3, 2) * b5 * c2 * d1 * d2 +
          4 * b0 * pow(b3, 2) * c2 * d4 * d5 +
          4 * b0 * pow(b3, 2) * c4 * d2 * d5 -
          2 * b0 * pow(b3, 2) * c5 * d1 * d5 +
          4 * b0 * pow(b3, 2) * c5 * d2 * d4 +
          2 * b0 * pow(b5, 2) * c1 * d3 * d5 -
          4 * b0 * pow(b5, 2) * c2 * d3 * d4 +
          2 * b0 * pow(b5, 2) * c3 * d1 * d5 -
          4 * b0 * pow(b5, 2) * c3 * d2 * d4 +
          8 * b0 * pow(b5, 2) * c4 * d1 * d4 -
          4 * b0 * pow(b5, 2) * c4 * d2 * d3 +
          2 * b0 * pow(b5, 2) * c5 * d1 * d3 +
          8 * b1 * pow(b2, 2) * c3 * d3 * d5 +
          32 * b1 * pow(b2, 2) * c4 * d3 * d4 +
          6 * b1 * pow(b3, 2) * c5 * d0 * d5 +
          2 * b1 * pow(b5, 2) * c0 * d3 * d5 +
          2 * b1 * pow(b5, 2) * c3 * d0 * d5 -
          8 * b1 * pow(b5, 2) * c4 * d0 * d4 +
          2 * b1 * pow(b5, 2) * c5 * d0 * d3 -
          4 * b2 * pow(b3, 2) * c0 * d4 * d5 -
          4 * b2 * pow(b3, 2) * c4 * d0 * d5 -
          4 * b2 * pow(b3, 2) * c5 * d0 * d4 +
          4 * b2 * pow(b5, 2) * c0 * d3 * d4 +
          4 * b2 * pow(b5, 2) * c3 * d0 * d4 +
          4 * b2 * pow(b5, 2) * c4 * d0 * d3 +
          2 * b3 * pow(b5, 2) * c0 * d1 * d5 -
          4 * b3 * pow(b5, 2) * c0 * d2 * d4 +
          2 * b3 * pow(b5, 2) * c1 * d0 * d5 -
          4 * b3 * pow(b5, 2) * c2 * d0 * d4 -
          4 * b3 * pow(b5, 2) * c4 * d0 * d2 +
          2 * b3 * pow(b5, 2) * c5 * d0 * d1 +
          4 * b4 * pow(b5, 2) * c0 * d2 * d3 +
          4 * b4 * pow(b5, 2) * c2 * d0 * d3 +
          4 * b4 * pow(b5, 2) * c3 * d0 * d2 +
          4 * pow(b0, 2) * b1 * c5 * d3 * d5 -
          8 * pow(b0, 2) * b3 * c2 * d4 * d5 -
          8 * pow(b0, 2) * b3 * c4 * d2 * d5 +
          4 * pow(b0, 2) * b3 * c5 * d1 * d5 -
          8 * pow(b0, 2) * b3 * c5 * d2 * d4 +
          8 * pow(b0, 2) * b4 * c1 * d4 * d5 +
          8 * pow(b0, 2) * b4 * c4 * d1 * d5 +
          8 * pow(b0, 2) * b4 * c5 * d1 * d4 -
          4 * pow(b0, 2) * b5 * c1 * d3 * d5 +
          8 * pow(b0, 2) * b5 * c2 * d3 * d4 -
          4 * pow(b0, 2) * b5 * c3 * d1 * d5 +
          8 * pow(b0, 2) * b5 * c3 * d2 * d4 -
          16 * pow(b0, 2) * b5 * c4 * d1 * d4 +
          8 * pow(b0, 2) * b5 * c4 * d2 * d3 -
          4 * pow(b0, 2) * b5 * c5 * d1 * d3 -
          8 * pow(b2, 2) * b3 * c1 * d3 * d5 +
          16 * pow(b2, 2) * b3 * c2 * d3 * d4 -
          8 * pow(b2, 2) * b3 * c3 * d1 * d5 +
          16 * pow(b2, 2) * b3 * c3 * d2 * d4 -
          32 * pow(b2, 2) * b3 * c4 * d1 * d4 +
          16 * pow(b2, 2) * b3 * c4 * d2 * d3 -
          8 * pow(b2, 2) * b3 * c5 * d1 * d3 +
          16 * pow(b2, 2) * b4 * c3 * d2 * d3 +
          8 * pow(b2, 2) * b5 * c3 * d1 * d3 -
          4 * pow(b3, 2) * b4 * c0 * d2 * d5 -
          4 * pow(b3, 2) * b4 * c2 * d0 * d5 -
          4 * pow(b3, 2) * b4 * c5 * d0 * d2 -
          2 * pow(b3, 2) * b5 * c0 * d1 * d5 +
          4 * pow(b3, 2) * b5 * c0 * d2 * d4 -
          2 * pow(b3, 2) * b5 * c1 * d0 * d5 +
          4 * pow(b3, 2) * b5 * c2 * d0 * d4 +
          4 * pow(b3, 2) * b5 * c4 * d0 * d2 -
          2 * pow(b3, 2) * b5 * c5 * d0 * d1 +
          8 * b0 * pow(b4, 2) * c5 * d1 * d5 -
          8 * b1 * pow(b4, 2) * c5 * d0 * d5 -
          16 * pow(b1, 2) * b2 * c3 * d4 * d5 -
          16 * pow(b1, 2) * b2 * c4 * d3 * d5 -
          16 * pow(b1, 2) * b2 * c5 * d3 * d4 -
          8 * pow(b1, 2) * b3 * c5 * d1 * d5 -
          8 * pow(b1, 2) * b5 * c1 * d3 * d5 +
          16 * pow(b1, 2) * b5 * c2 * d3 * d4 -
          8 * pow(b1, 2) * b5 * c3 * d1 * d5 +
          16 * pow(b1, 2) * b5 * c3 * d2 * d4 +
          16 * pow(b1, 2) * b5 * c4 * d2 * d3 -
          8 * pow(b1, 2) * b5 * c5 * d1 * d3 -
          8 * b1 * pow(b2, 2) * c5 * d3 * d5 +
          4 * pow(b0, 2) * b2 * c5 * d4 * d5 +
          4 * pow(b0, 2) * b4 * c5 * d2 * d5 -
          4 * pow(b0, 2) * b5 * c2 * d4 * d5 -
          4 * pow(b0, 2) * b5 * c4 * d2 * d5 +
          2 * pow(b0, 2) * b5 * c5 * d1 * d5 -
          4 * pow(b0, 2) * b5 * c5 * d2 * d4 +
          8 * pow(b2, 2) * b3 * c5 * d1 * d5 -
          2 * b1 * pow(b5, 2) * c3 * d3 * d5 +
          8 * b1 * pow(b5, 2) * c4 * d3 * d4 -
          4 * b2 * pow(b5, 2) * c3 * d3 * d4 -
          2 * b3 * pow(b5, 2) * c1 * d3 * d5 +
          4 * b3 * pow(b5, 2) * c2 * d3 * d4 -
          2 * b3 * pow(b5, 2) * c3 * d1 * d5 +
          4 * b3 * pow(b5, 2) * c3 * d2 * d4 -
          8 * b3 * pow(b5, 2) * c4 * d1 * d4 +
          4 * b3 * pow(b5, 2) * c4 * d2 * d3 -
          2 * b3 * pow(b5, 2) * c5 * d1 * d3 -
          4 * b4 * pow(b5, 2) * c3 * d2 * d3 +
          16 * pow(b1, 2) * b2 * c5 * d4 * d5 +
          16 * pow(b1, 2) * b4 * c5 * d2 * d5 -
          16 * pow(b1, 2) * b5 * c2 * d4 * d5 -
          16 * pow(b1, 2) * b5 * c4 * d2 * d5 +
          24 * pow(b1, 2) * b5 * c5 * d1 * d5 -
          16 * pow(b1, 2) * b5 * c5 * d2 * d4 +
          8 * b1 * pow(b4, 2) * c5 * d3 * d5 -
          8 * b3 * pow(b4, 2) * c5 * d1 * d5 +
          4 * b2 * pow(b3, 2) * c5 * d4 * d5 +
          4 * pow(b3, 2) * b4 * c5 * d2 * d5 -
          4 * pow(b3, 2) * b5 * c2 * d4 * d5 -
          4 * pow(b3, 2) * b5 * c4 * d2 * d5 +
          2 * pow(b3, 2) * b5 * c5 * d1 * d5 -
          4 * pow(b3, 2) * b5 * c5 * d2 * d4 +
          16 * b0 * b1 * b3 * c1 * d0 * d1 - 4 * b0 * b1 * b3 * c0 * d0 * d3 +
          16 * b0 * b1 * b2 * c0 * d1 * d4 - 8 * b0 * b1 * b2 * c0 * d2 * d3 +
          16 * b0 * b1 * b2 * c1 * d0 * d4 - 8 * b0 * b1 * b2 * c2 * d0 * d3 -
          8 * b0 * b1 * b2 * c3 * d0 * d2 + 16 * b0 * b1 * b2 * c4 * d0 * d1 -
          16 * b0 * b1 * b4 * c0 * d1 * d2 - 16 * b0 * b1 * b4 * c1 * d0 * d2 -
          16 * b0 * b1 * b4 * c2 * d0 * d1 - 16 * b0 * b1 * b5 * c1 * d0 * d1 +
          8 * b0 * b2 * b3 * c0 * d1 * d2 + 8 * b0 * b2 * b3 * c1 * d0 * d2 +
          8 * b0 * b2 * b3 * c2 * d0 * d1 + 4 * b0 * b1 * b3 * c0 * d0 * d5 +
          16 * b0 * b1 * b4 * c0 * d0 * d4 + 4 * b0 * b1 * b5 * c0 * d0 * d3 -
          8 * b0 * b2 * b3 * c0 * d0 * d4 - 8 * b0 * b2 * b4 * c0 * d0 * d3 -
          8 * b0 * b3 * b4 * c0 * d0 * d2 + 4 * b0 * b3 * b5 * c0 * d0 * d1 +
          8 * b0 * b1 * b2 * c0 * d2 * d5 + 8 * b0 * b1 * b2 * c2 * d0 * d5 +
          8 * b0 * b1 * b2 * c5 * d0 * d2 - 32 * b0 * b2 * b4 * c2 * d0 * d2 -
          8 * b0 * b2 * b5 * c0 * d1 * d2 - 8 * b0 * b2 * b5 * c1 * d0 * d2 -
          8 * b0 * b2 * b5 * c2 * d0 * d1 + 64 * b1 * b2 * b3 * c1 * d1 * d2 +
          16 * b0 * b1 * b2 * c3 * d2 * d3 - 16 * b0 * b1 * b3 * c1 * d1 * d5 +
          16 * b0 * b1 * b3 * c1 * d2 * d4 + 16 * b0 * b1 * b3 * c2 * d1 * d4 -
          16 * b0 * b1 * b3 * c2 * d2 * d3 + 16 * b0 * b1 * b3 * c4 * d1 * d2 +
          16 * b0 * b1 * b4 * c1 * d2 * d3 + 16 * b0 * b1 * b4 * c2 * d1 * d3 +
          16 * b0 * b1 * b4 * c3 * d1 * d2 - 4 * b0 * b1 * b5 * c0 * d0 * d5 -
          16 * b0 * b1 * b5 * c1 * d1 * d3 + 8 * b0 * b2 * b4 * c0 * d0 * d5 +
          8 * b0 * b2 * b5 * c0 * d0 * d4 - 32 * b0 * b3 * b4 * c1 * d1 * d2 +
          8 * b0 * b4 * b5 * c0 * d0 * d2 - 16 * b1 * b2 * b3 * c0 * d1 * d4 -
          16 * b1 * b2 * b3 * c1 * d0 * d4 - 16 * b1 * b2 * b3 * c4 * d0 * d1 -
          16 * b1 * b2 * b4 * c0 * d1 * d3 - 16 * b1 * b2 * b4 * c1 * d0 * d3 -
          16 * b1 * b2 * b4 * c3 * d0 * d1 - 16 * b1 * b3 * b5 * c1 * d0 * d1 +
          32 * b2 * b3 * b4 * c1 * d0 * d1 + 4 * b0 * b1 * b3 * c0 * d3 * d5 +
          4 * b0 * b1 * b3 * c3 * d0 * d5 + 4 * b0 * b1 * b3 * c5 * d0 * d3 -
          8 * b0 * b1 * b4 * c0 * d3 * d4 - 8 * b0 * b1 * b4 * c3 * d0 * d4 -
          8 * b0 * b1 * b4 * c4 * d0 * d3 - 4 * b0 * b1 * b5 * c3 * d0 * d3 +
          8 * b0 * b2 * b4 * c3 * d0 * d3 + 8 * b0 * b3 * b4 * c0 * d1 * d4 +
          8 * b0 * b3 * b4 * c1 * d0 * d4 + 8 * b0 * b3 * b4 * c4 * d0 * d1 -
          4 * b0 * b3 * b5 * c0 * d1 * d3 - 4 * b0 * b3 * b5 * c1 * d0 * d3 -
          4 * b0 * b3 * b5 * c3 * d0 * d1 - 128 * b1 * b2 * b4 * c2 * d1 * d2 -
          64 * b1 * b2 * b5 * c1 * d1 * d2 + 4 * b1 * b3 * b5 * c0 * d0 * d3 -
          8 * b2 * b3 * b4 * c0 * d0 * d3 - 16 * b0 * b1 * b2 * c1 * d4 * d5 -
          8 * b0 * b1 * b2 * c2 * d3 * d5 - 8 * b0 * b1 * b2 * c3 * d2 * d5 -
          16 * b0 * b1 * b2 * c4 * d1 * d5 - 16 * b0 * b1 * b2 * c5 * d1 * d4 -
          8 * b0 * b1 * b2 * c5 * d2 * d3 + 16 * b0 * b1 * b3 * c2 * d2 * d5 +
          32 * b0 * b1 * b5 * c1 * d1 * d5 - 16 * b0 * b1 * b5 * c1 * d2 * d4 -
          16 * b0 * b1 * b5 * c2 * d1 * d4 - 16 * b0 * b1 * b5 * c4 * d1 * d2 -
          8 * b0 * b2 * b3 * c1 * d2 * d5 - 8 * b0 * b2 * b3 * c2 * d1 * d5 +
          32 * b0 * b2 * b3 * c2 * d2 * d4 - 8 * b0 * b2 * b3 * c5 * d1 * d2 -
          32 * b0 * b2 * b4 * c1 * d2 * d4 - 32 * b0 * b2 * b4 * c2 * d1 * d4 +
          32 * b0 * b2 * b4 * c2 * d2 * d3 - 32 * b0 * b2 * b4 * c4 * d1 * d2 +
          32 * b0 * b2 * b5 * c1 * d1 * d4 + 16 * b1 * b2 * b4 * c0 * d1 * d5 +
          32 * b1 * b2 * b4 * c0 * d2 * d4 + 16 * b1 * b2 * b4 * c1 * d0 * d5 +
          32 * b1 * b2 * b4 * c2 * d0 * d4 + 32 * b1 * b2 * b4 * c4 * d0 * d2 +
          16 * b1 * b2 * b4 * c5 * d0 * d1 + 8 * b1 * b2 * b5 * c0 * d2 * d3 +
          8 * b1 * b2 * b5 * c2 * d0 * d3 + 8 * b1 * b2 * b5 * c3 * d0 * d2 -
          16 * b1 * b3 * b5 * c2 * d0 * d2 + 16 * b1 * b4 * b5 * c0 * d1 * d2 +
          16 * b1 * b4 * b5 * c1 * d0 * d2 + 16 * b1 * b4 * b5 * c2 * d0 * d1 +
          32 * b2 * b3 * b4 * c2 * d0 * d2 + 8 * b2 * b3 * b5 * c0 * d1 * d2 +
          8 * b2 * b3 * b5 * c1 * d0 * d2 + 8 * b2 * b3 * b5 * c2 * d0 * d1 -
          32 * b2 * b4 * b5 * c1 * d0 * d1 - 8 * b0 * b1 * b3 * c5 * d0 * d5 -
          8 * b0 * b1 * b4 * c0 * d4 * d5 - 8 * b0 * b1 * b4 * c4 * d0 * d5 -
          8 * b0 * b1 * b4 * c5 * d0 * d4 + 16 * b0 * b1 * b5 * c4 * d0 * d4 +
          8 * b0 * b2 * b3 * c0 * d4 * d5 + 8 * b0 * b2 * b3 * c4 * d0 * d5 +
          8 * b0 * b2 * b3 * c5 * d0 * d4 - 8 * b0 * b2 * b5 * c0 * d3 * d4 -
          8 * b0 * b2 * b5 * c3 * d0 * d4 - 8 * b0 * b2 * b5 * c4 * d0 * d3 +
          8 * b0 * b3 * b4 * c0 * d2 * d5 + 8 * b0 * b3 * b4 * c2 * d0 * d5 +
          8 * b0 * b3 * b4 * c5 * d0 * d2 + 8 * b0 * b4 * b5 * c0 * d1 * d4 -
          8 * b0 * b4 * b5 * c0 * d2 * d3 + 8 * b0 * b4 * b5 * c1 * d0 * d4 -
          8 * b0 * b4 * b5 * c2 * d0 * d3 - 8 * b0 * b4 * b5 * c3 * d0 * d2 +
          8 * b0 * b4 * b5 * c4 * d0 * d1 + 8 * b1 * b3 * b5 * c0 * d0 * d5 -
          16 * b1 * b4 * b5 * c0 * d0 * d4 - 16 * b2 * b3 * b4 * c0 * d0 * d5 +
          16 * b2 * b4 * b5 * c0 * d0 * d3 + 8 * b0 * b2 * b5 * c1 * d2 * d5 +
          8 * b0 * b2 * b5 * c2 * d1 * d5 + 8 * b0 * b2 * b5 * c5 * d1 * d2 -
          8 * b1 * b2 * b5 * c0 * d2 * d5 - 8 * b1 * b2 * b5 * c2 * d0 * d5 -
          8 * b1 * b2 * b5 * c5 * d0 * d2 + 4 * b0 * b1 * b5 * c5 * d0 * d5 -
          8 * b0 * b2 * b4 * c5 * d0 * d5 + 16 * b1 * b2 * b3 * c1 * d4 * d5 +
          16 * b1 * b2 * b3 * c4 * d1 * d5 + 16 * b1 * b2 * b3 * c5 * d1 * d4 +
          16 * b1 * b2 * b4 * c1 * d3 * d5 - 32 * b1 * b2 * b4 * c2 * d3 * d4 +
          16 * b1 * b2 * b4 * c3 * d1 * d5 - 32 * b1 * b2 * b4 * c3 * d2 * d4 -
          32 * b1 * b2 * b4 * c4 * d2 * d3 + 16 * b1 * b2 * b4 * c5 * d1 * d3 -
          16 * b1 * b2 * b5 * c3 * d2 * d3 + 16 * b1 * b3 * b5 * c1 * d1 * d5 -
          16 * b1 * b3 * b5 * c1 * d2 * d4 - 16 * b1 * b3 * b5 * c2 * d1 * d4 +
          16 * b1 * b3 * b5 * c2 * d2 * d3 - 16 * b1 * b3 * b5 * c4 * d1 * d2 -
          16 * b1 * b4 * b5 * c1 * d2 * d3 - 16 * b1 * b4 * b5 * c2 * d1 * d3 -
          16 * b1 * b4 * b5 * c3 * d1 * d2 - 32 * b2 * b3 * b4 * c1 * d1 * d5 +
          32 * b2 * b3 * b4 * c1 * d2 * d4 + 32 * b2 * b3 * b4 * c2 * d1 * d4 -
          32 * b2 * b3 * b4 * c2 * d2 * d3 + 32 * b2 * b3 * b4 * c4 * d1 * d2 +
          8 * b2 * b4 * b5 * c0 * d0 * d5 + 32 * b3 * b4 * b5 * c1 * d1 * d2 -
          4 * b0 * b1 * b3 * c5 * d3 * d5 + 8 * b0 * b1 * b4 * c3 * d4 * d5 +
          8 * b0 * b1 * b4 * c4 * d3 * d5 + 8 * b0 * b1 * b4 * c5 * d3 * d4 +
          4 * b0 * b1 * b5 * c3 * d3 * d5 - 16 * b0 * b1 * b5 * c4 * d3 * d4 -
          8 * b0 * b2 * b4 * c3 * d3 * d5 + 8 * b0 * b2 * b5 * c3 * d3 * d4 -
          8 * b0 * b3 * b4 * c1 * d4 * d5 - 8 * b0 * b3 * b4 * c4 * d1 * d5 -
          8 * b0 * b3 * b4 * c5 * d1 * d4 + 4 * b0 * b3 * b5 * c1 * d3 * d5 -
          8 * b0 * b3 * b5 * c2 * d3 * d4 + 4 * b0 * b3 * b5 * c3 * d1 * d5 -
          8 * b0 * b3 * b5 * c3 * d2 * d4 + 16 * b0 * b3 * b5 * c4 * d1 * d4 -
          8 * b0 * b3 * b5 * c4 * d2 * d3 + 4 * b0 * b3 * b5 * c5 * d1 * d3 +
          8 * b0 * b4 * b5 * c3 * d2 * d3 - 4 * b1 * b3 * b5 * c0 * d3 * d5 -
          4 * b1 * b3 * b5 * c3 * d0 * d5 - 4 * b1 * b3 * b5 * c5 * d0 * d3 +
          8 * b1 * b4 * b5 * c0 * d3 * d4 + 8 * b1 * b4 * b5 * c3 * d0 * d4 +
          8 * b1 * b4 * b5 * c4 * d0 * d3 + 8 * b2 * b3 * b4 * c0 * d3 * d5 +
          8 * b2 * b3 * b4 * c3 * d0 * d5 + 8 * b2 * b3 * b4 * c5 * d0 * d3 -
          8 * b2 * b4 * b5 * c3 * d0 * d3 - 8 * b3 * b4 * b5 * c0 * d1 * d4 -
          8 * b3 * b4 * b5 * c1 * d0 * d4 - 8 * b3 * b4 * b5 * c4 * d0 * d1 -
          32 * b1 * b2 * b4 * c5 * d1 * d5 + 8 * b1 * b2 * b5 * c2 * d3 * d5 +
          8 * b1 * b2 * b5 * c3 * d2 * d5 + 8 * b1 * b2 * b5 * c5 * d2 * d3 -
          8 * b2 * b3 * b5 * c1 * d2 * d5 - 8 * b2 * b3 * b5 * c2 * d1 * d5 -
          8 * b2 * b3 * b5 * c5 * d1 * d2 + 32 * b2 * b4 * b5 * c1 * d1 * d5 -
          4 * b0 * b1 * b5 * c5 * d3 * d5 - 8 * b0 * b2 * b3 * c5 * d4 * d5 +
          8 * b0 * b2 * b4 * c5 * d3 * d5 - 8 * b0 * b3 * b4 * c5 * d2 * d5 +
          8 * b0 * b3 * b5 * c2 * d4 * d5 + 8 * b0 * b3 * b5 * c4 * d2 * d5 -
          4 * b0 * b3 * b5 * c5 * d1 * d5 + 8 * b0 * b3 * b5 * c5 * d2 * d4 -
          8 * b0 * b4 * b5 * c1 * d4 * d5 - 8 * b0 * b4 * b5 * c4 * d1 * d5 -
          8 * b0 * b4 * b5 * c5 * d1 * d4 - 4 * b1 * b3 * b5 * c5 * d0 * d5 +
          8 * b1 * b4 * b5 * c0 * d4 * d5 + 8 * b1 * b4 * b5 * c4 * d0 * d5 +
          8 * b1 * b4 * b5 * c5 * d0 * d4 + 8 * b2 * b3 * b4 * c5 * d0 * d5 -
          8 * b2 * b4 * b5 * c0 * d3 * d5 - 8 * b2 * b4 * b5 * c3 * d0 * d5 -
          8 * b2 * b4 * b5 * c5 * d0 * d3 + 4 * b1 * b3 * b5 * c5 * d3 * d5 -
          8 * b1 * b4 * b5 * c3 * d4 * d5 - 8 * b1 * b4 * b5 * c4 * d3 * d5 -
          8 * b1 * b4 * b5 * c5 * d3 * d4 - 8 * b2 * b3 * b4 * c5 * d3 * d5 +
          8 * b2 * b4 * b5 * c3 * d3 * d5 + 8 * b3 * b4 * b5 * c1 * d4 * d5 +
          8 * b3 * b4 * b5 * c4 * d1 * d5 + 8 * b3 * b4 * b5 * c5 * d1 * d4,
      4 * pow(b1, 3) * pow(c0, 2) * d3 - pow(b0, 3) * pow(c3, 2) * d1 +
          4 * pow(b0, 3) * pow(c4, 2) * d1 - pow(b0, 3) * pow(c5, 2) * d1 -
          4 * pow(b1, 3) * pow(c0, 2) * d5 + 16 * pow(b1, 3) * pow(c2, 2) * d3 +
          8 * pow(b1, 3) * pow(c5, 2) * d0 - 8 * pow(b2, 3) * pow(c0, 2) * d4 +
          pow(b5, 3) * pow(c0, 2) * d1 - 32 * pow(b2, 3) * pow(c1, 2) * d4 +
          12 * pow(b5, 3) * pow(c1, 2) * d1 -
          16 * pow(b1, 3) * pow(c2, 2) * d5 + 4 * pow(b1, 3) * pow(c5, 2) * d3 -
          8 * pow(b2, 3) * pow(c3, 2) * d4 + pow(b5, 3) * pow(c3, 2) * d1 -
          12 * pow(b1, 3) * pow(c5, 2) * d5 + 8 * pow(b1, 3) * c0 * c3 * d0 -
          8 * pow(b1, 3) * c0 * c5 * d0 - 16 * pow(b2, 3) * c0 * c4 * d0 +
          2 * pow(b5, 3) * c0 * c1 * d0 - 2 * pow(b0, 3) * c1 * c3 * d3 +
          32 * pow(b1, 3) * c2 * c3 * d2 - 64 * pow(b2, 3) * c1 * c4 * d1 +
          2 * pow(b0, 3) * c1 * c3 * d5 + 8 * pow(b0, 3) * c1 * c4 * d4 +
          2 * pow(b0, 3) * c1 * c5 * d3 - 4 * pow(b0, 3) * c2 * c3 * d4 -
          4 * pow(b0, 3) * c2 * c4 * d3 - 4 * pow(b0, 3) * c3 * c4 * d2 +
          2 * pow(b0, 3) * c3 * c5 * d1 - 8 * pow(b1, 3) * c0 * c3 * d5 -
          8 * pow(b1, 3) * c0 * c5 * d3 - 8 * pow(b1, 3) * c3 * c5 * d0 +
          16 * pow(b2, 3) * c0 * c3 * d4 + 16 * pow(b2, 3) * c0 * c4 * d3 +
          16 * pow(b2, 3) * c3 * c4 * d0 - 2 * pow(b5, 3) * c0 * c1 * d3 -
          2 * pow(b5, 3) * c0 * c3 * d1 - 2 * pow(b5, 3) * c1 * c3 * d0 -
          32 * pow(b1, 3) * c2 * c5 * d2 - 2 * pow(b0, 3) * c1 * c5 * d5 +
          4 * pow(b0, 3) * c2 * c4 * d5 + 4 * pow(b0, 3) * c2 * c5 * d4 +
          4 * pow(b0, 3) * c4 * c5 * d2 + 16 * pow(b1, 3) * c0 * c5 * d5 -
          16 * pow(b2, 3) * c3 * c4 * d3 + 2 * pow(b5, 3) * c1 * c3 * d3 +
          8 * pow(b1, 3) * c3 * c5 * d5 - b0 * pow(b3, 2) * pow(c0, 2) * d1 +
          3 * b1 * pow(b3, 2) * pow(c0, 2) * d0 +
          pow(b0, 2) * b1 * pow(c3, 2) * d0 +
          4 * b0 * pow(b4, 2) * pow(c0, 2) * d1 -
          12 * b1 * pow(b4, 2) * pow(c0, 2) * d0 +
          4 * pow(b0, 2) * b1 * pow(c1, 2) * d3 -
          4 * pow(b0, 2) * b1 * pow(c4, 2) * d0 -
          12 * pow(b0, 2) * b3 * pow(c1, 2) * d1 -
          4 * pow(b1, 2) * b3 * pow(c0, 2) * d1 -
          4 * b0 * pow(b2, 2) * pow(c3, 2) * d1 +
          4 * b0 * pow(b3, 2) * pow(c2, 2) * d1 -
          b0 * pow(b5, 2) * pow(c0, 2) * d1 +
          4 * b1 * pow(b2, 2) * pow(c0, 2) * d3 -
          4 * b1 * pow(b2, 2) * pow(c3, 2) * d0 +
          4 * b1 * pow(b3, 2) * pow(c2, 2) * d0 +
          3 * b1 * pow(b5, 2) * pow(c0, 2) * d0 +
          4 * pow(b0, 2) * b1 * pow(c2, 2) * d3 +
          pow(b0, 2) * b1 * pow(c5, 2) * d0 -
          4 * pow(b0, 2) * b3 * pow(c2, 2) * d1 -
          4 * pow(b2, 2) * b3 * pow(c0, 2) * d1 -
          8 * b0 * pow(b1, 2) * pow(c5, 2) * d1 +
          16 * b0 * pow(b2, 2) * pow(c4, 2) * d1 +
          16 * b0 * pow(b4, 2) * pow(c2, 2) * d1 -
          24 * b0 * pow(b5, 2) * pow(c1, 2) * d1 +
          16 * b1 * pow(b2, 2) * pow(c1, 2) * d3 -
          16 * b1 * pow(b2, 2) * pow(c4, 2) * d0 -
          16 * b1 * pow(b4, 2) * pow(c2, 2) * d0 +
          8 * b1 * pow(b5, 2) * pow(c1, 2) * d0 -
          4 * pow(b0, 2) * b1 * pow(c1, 2) * d5 -
          8 * pow(b0, 2) * b2 * pow(c1, 2) * d4 +
          8 * pow(b0, 2) * b4 * pow(c1, 2) * d2 +
          12 * pow(b0, 2) * b5 * pow(c1, 2) * d1 -
          8 * pow(b1, 2) * b2 * pow(c0, 2) * d4 -
          16 * pow(b1, 2) * b3 * pow(c2, 2) * d1 +
          8 * pow(b1, 2) * b4 * pow(c0, 2) * d2 +
          4 * pow(b1, 2) * b5 * pow(c0, 2) * d1 -
          48 * pow(b2, 2) * b3 * pow(c1, 2) * d1 -
          4 * b0 * pow(b2, 2) * pow(c5, 2) * d1 -
          4 * b0 * pow(b5, 2) * pow(c2, 2) * d1 -
          4 * b1 * pow(b2, 2) * pow(c0, 2) * d5 +
          4 * b1 * pow(b2, 2) * pow(c5, 2) * d0 +
          4 * b1 * pow(b4, 2) * pow(c0, 2) * d3 +
          4 * b1 * pow(b5, 2) * pow(c2, 2) * d0 -
          4 * b3 * pow(b4, 2) * pow(c0, 2) * d1 -
          4 * pow(b0, 2) * b1 * pow(c2, 2) * d5 +
          4 * pow(b0, 2) * b1 * pow(c4, 2) * d3 -
          8 * pow(b0, 2) * b2 * pow(c2, 2) * d4 -
          4 * pow(b0, 2) * b3 * pow(c4, 2) * d1 +
          24 * pow(b0, 2) * b4 * pow(c2, 2) * d2 +
          4 * pow(b0, 2) * b5 * pow(c2, 2) * d1 +
          8 * pow(b2, 2) * b4 * pow(c0, 2) * d2 +
          4 * pow(b2, 2) * b5 * pow(c0, 2) * d1 -
          b0 * pow(b3, 2) * pow(c5, 2) * d1 -
          3 * b0 * pow(b5, 2) * pow(c3, 2) * d1 -
          16 * b1 * pow(b2, 2) * pow(c1, 2) * d5 -
          3 * b1 * pow(b3, 2) * pow(c0, 2) * d5 +
          3 * b1 * pow(b3, 2) * pow(c5, 2) * d0 -
          2 * b1 * pow(b5, 2) * pow(c0, 2) * d3 +
          b1 * pow(b5, 2) * pow(c3, 2) * d0 +
          2 * b2 * pow(b3, 2) * pow(c0, 2) * d4 -
          2 * b3 * pow(b5, 2) * pow(c0, 2) * d1 -
          pow(b0, 2) * b1 * pow(c3, 2) * d5 +
          2 * pow(b0, 2) * b1 * pow(c5, 2) * d3 -
          2 * pow(b0, 2) * b2 * pow(c3, 2) * d4 +
          2 * pow(b0, 2) * b3 * pow(c5, 2) * d1 -
          2 * pow(b0, 2) * b4 * pow(c3, 2) * d2 +
          3 * pow(b0, 2) * b5 * pow(c3, 2) * d1 -
          32 * pow(b1, 2) * b2 * pow(c2, 2) * d4 +
          96 * pow(b1, 2) * b4 * pow(c2, 2) * d2 +
          16 * pow(b1, 2) * b5 * pow(c2, 2) * d1 +
          32 * pow(b2, 2) * b4 * pow(c1, 2) * d2 +
          48 * pow(b2, 2) * b5 * pow(c1, 2) * d1 +
          2 * pow(b3, 2) * b4 * pow(c0, 2) * d2 +
          pow(b3, 2) * b5 * pow(c0, 2) * d1 +
          4 * b0 * pow(b4, 2) * pow(c5, 2) * d1 +
          4 * b0 * pow(b5, 2) * pow(c4, 2) * d1 +
          16 * b1 * pow(b2, 2) * pow(c4, 2) * d3 +
          8 * b1 * pow(b4, 2) * pow(c0, 2) * d5 +
          16 * b1 * pow(b4, 2) * pow(c2, 2) * d3 -
          4 * b1 * pow(b4, 2) * pow(c5, 2) * d0 +
          4 * b1 * pow(b5, 2) * pow(c1, 2) * d3 -
          4 * b1 * pow(b5, 2) * pow(c4, 2) * d0 -
          16 * b3 * pow(b4, 2) * pow(c2, 2) * d1 -
          12 * b3 * pow(b5, 2) * pow(c1, 2) * d1 -
          8 * pow(b0, 2) * b5 * pow(c4, 2) * d1 -
          4 * pow(b1, 2) * b3 * pow(c5, 2) * d1 -
          16 * pow(b2, 2) * b3 * pow(c4, 2) * d1 +
          4 * b1 * pow(b2, 2) * pow(c3, 2) * d5 -
          4 * b1 * pow(b2, 2) * pow(c5, 2) * d3 -
          4 * b1 * pow(b3, 2) * pow(c2, 2) * d5 -
          b1 * pow(b5, 2) * pow(c0, 2) * d5 -
          4 * b1 * pow(b5, 2) * pow(c2, 2) * d3 -
          8 * b2 * pow(b3, 2) * pow(c2, 2) * d4 -
          2 * b2 * pow(b5, 2) * pow(c0, 2) * d4 +
          4 * b3 * pow(b5, 2) * pow(c2, 2) * d1 -
          2 * b4 * pow(b5, 2) * pow(c0, 2) * d2 -
          3 * pow(b0, 2) * b1 * pow(c5, 2) * d5 +
          2 * pow(b0, 2) * b2 * pow(c5, 2) * d4 +
          2 * pow(b0, 2) * b4 * pow(c5, 2) * d2 +
          pow(b0, 2) * b5 * pow(c5, 2) * d1 +
          4 * pow(b2, 2) * b3 * pow(c5, 2) * d1 +
          8 * pow(b2, 2) * b4 * pow(c3, 2) * d2 +
          4 * pow(b2, 2) * b5 * pow(c3, 2) * d1 +
          24 * pow(b3, 2) * b4 * pow(c2, 2) * d2 -
          4 * pow(b3, 2) * b5 * pow(c2, 2) * d1 -
          12 * b1 * pow(b5, 2) * pow(c1, 2) * d5 -
          8 * b2 * pow(b5, 2) * pow(c1, 2) * d4 -
          8 * b4 * pow(b5, 2) * pow(c1, 2) * d2 +
          8 * pow(b1, 2) * b2 * pow(c5, 2) * d4 +
          8 * pow(b1, 2) * b4 * pow(c5, 2) * d2 +
          12 * pow(b1, 2) * b5 * pow(c5, 2) * d1 +
          4 * b1 * pow(b4, 2) * pow(c5, 2) * d3 +
          4 * b1 * pow(b5, 2) * pow(c4, 2) * d3 -
          4 * b3 * pow(b4, 2) * pow(c5, 2) * d1 -
          4 * b3 * pow(b5, 2) * pow(c4, 2) * d1 -
          3 * b1 * pow(b3, 2) * pow(c5, 2) * d5 -
          b1 * pow(b5, 2) * pow(c3, 2) * d5 +
          2 * b2 * pow(b3, 2) * pow(c5, 2) * d4 -
          2 * b2 * pow(b5, 2) * pow(c3, 2) * d4 -
          2 * b4 * pow(b5, 2) * pow(c3, 2) * d2 +
          2 * pow(b3, 2) * b4 * pow(c5, 2) * d2 +
          pow(b3, 2) * b5 * pow(c5, 2) * d1 +
          8 * b0 * b1 * b3 * pow(c1, 2) * d0 -
          2 * b0 * b1 * b3 * pow(c0, 2) * d3 -
          8 * b0 * b1 * b5 * pow(c1, 2) * d0 +
          8 * b0 * b1 * b2 * pow(c3, 2) * d2 -
          16 * b0 * b2 * b4 * pow(c2, 2) * d0 +
          2 * b0 * b1 * b3 * pow(c0, 2) * d5 -
          8 * b0 * b1 * b3 * pow(c2, 2) * d3 -
          4 * b0 * b1 * b3 * pow(c5, 2) * d0 +
          8 * b0 * b1 * b4 * pow(c0, 2) * d4 +
          2 * b0 * b1 * b5 * pow(c0, 2) * d3 -
          2 * b0 * b1 * b5 * pow(c3, 2) * d0 -
          4 * b0 * b2 * b3 * pow(c0, 2) * d4 -
          4 * b0 * b2 * b4 * pow(c0, 2) * d3 +
          4 * b0 * b2 * b4 * pow(c3, 2) * d0 -
          4 * b0 * b3 * b4 * pow(c0, 2) * d2 +
          2 * b0 * b3 * b5 * pow(c0, 2) * d1 +
          32 * b1 * b2 * b3 * pow(c1, 2) * d2 -
          6 * b1 * b3 * b5 * pow(c0, 2) * d0 +
          12 * b2 * b3 * b4 * pow(c0, 2) * d0 -
          8 * b0 * b1 * b3 * pow(c1, 2) * d5 -
          8 * b0 * b1 * b5 * pow(c1, 2) * d3 +
          8 * b0 * b1 * b5 * pow(c4, 2) * d0 -
          16 * b0 * b3 * b4 * pow(c1, 2) * d2 +
          24 * b0 * b3 * b5 * pow(c1, 2) * d1 -
          64 * b1 * b2 * b4 * pow(c2, 2) * d1 -
          8 * b1 * b3 * b5 * pow(c1, 2) * d0 +
          16 * b2 * b3 * b4 * pow(c1, 2) * d0 +
          8 * b0 * b1 * b3 * pow(c2, 2) * d5 -
          2 * b0 * b1 * b5 * pow(c0, 2) * d5 +
          2 * b0 * b1 * b5 * pow(c5, 2) * d0 +
          16 * b0 * b2 * b3 * pow(c2, 2) * d4 +
          4 * b0 * b2 * b4 * pow(c0, 2) * d5 +
          16 * b0 * b2 * b4 * pow(c2, 2) * d3 -
          4 * b0 * b2 * b4 * pow(c5, 2) * d0 +
          4 * b0 * b2 * b5 * pow(c0, 2) * d4 -
          48 * b0 * b3 * b4 * pow(c2, 2) * d2 +
          4 * b0 * b4 * b5 * pow(c0, 2) * d2 -
          32 * b1 * b2 * b5 * pow(c1, 2) * d2 -
          8 * b1 * b3 * b5 * pow(c2, 2) * d0 +
          16 * b2 * b3 * b4 * pow(c2, 2) * d0 -
          12 * b2 * b4 * b5 * pow(c0, 2) * d0 -
          2 * b0 * b1 * b3 * pow(c5, 2) * d3 +
          16 * b0 * b1 * b5 * pow(c1, 2) * d5 +
          16 * b0 * b2 * b5 * pow(c1, 2) * d4 +
          2 * b1 * b3 * b5 * pow(c0, 2) * d3 -
          4 * b2 * b3 * b4 * pow(c0, 2) * d3 -
          16 * b2 * b4 * b5 * pow(c1, 2) * d0 -
          8 * b0 * b1 * b5 * pow(c4, 2) * d3 +
          8 * b0 * b3 * b5 * pow(c4, 2) * d1 -
          16 * b1 * b2 * b4 * pow(c5, 2) * d1 -
          8 * b1 * b2 * b5 * pow(c3, 2) * d2 +
          6 * b0 * b1 * b3 * pow(c5, 2) * d5 +
          2 * b0 * b1 * b5 * pow(c3, 2) * d5 -
          2 * b0 * b1 * b5 * pow(c5, 2) * d3 -
          4 * b0 * b2 * b3 * pow(c5, 2) * d4 -
          4 * b0 * b2 * b4 * pow(c3, 2) * d5 +
          4 * b0 * b2 * b4 * pow(c5, 2) * d3 +
          4 * b0 * b2 * b5 * pow(c3, 2) * d4 -
          4 * b0 * b3 * b4 * pow(c5, 2) * d2 -
          2 * b0 * b3 * b5 * pow(c5, 2) * d1 +
          4 * b0 * b4 * b5 * pow(c3, 2) * d2 +
          4 * b1 * b3 * b5 * pow(c0, 2) * d5 +
          8 * b1 * b3 * b5 * pow(c2, 2) * d3 -
          2 * b1 * b3 * b5 * pow(c5, 2) * d0 -
          8 * b1 * b4 * b5 * pow(c0, 2) * d4 -
          8 * b2 * b3 * b4 * pow(c0, 2) * d5 -
          16 * b2 * b3 * b4 * pow(c2, 2) * d3 +
          4 * b2 * b3 * b4 * pow(c5, 2) * d0 +
          8 * b2 * b4 * b5 * pow(c0, 2) * d3 -
          4 * b2 * b4 * b5 * pow(c3, 2) * d0 +
          8 * b1 * b3 * b5 * pow(c1, 2) * d5 -
          16 * b2 * b3 * b4 * pow(c1, 2) * d5 +
          16 * b3 * b4 * b5 * pow(c1, 2) * d2 +
          4 * b2 * b4 * b5 * pow(c0, 2) * d5 +
          2 * b1 * b3 * b5 * pow(c5, 2) * d3 -
          4 * b2 * b3 * b4 * pow(c5, 2) * d3 +
          16 * b2 * b4 * b5 * pow(c1, 2) * d5 +
          4 * b2 * b4 * b5 * pow(c3, 2) * d5 -
          2 * b0 * pow(b3, 2) * c0 * c1 * d0 -
          8 * b0 * pow(b1, 2) * c0 * c1 * d3 -
          8 * b0 * pow(b1, 2) * c0 * c3 * d1 -
          8 * b0 * pow(b1, 2) * c1 * c3 * d0 +
          8 * b0 * pow(b4, 2) * c0 * c1 * d0 -
          8 * pow(b1, 2) * b3 * c0 * c1 * d0 -
          2 * b0 * pow(b5, 2) * c0 * c1 * d0 +
          8 * b1 * pow(b2, 2) * c0 * c3 * d0 +
          8 * pow(b0, 2) * b1 * c1 * c3 * d1 -
          8 * pow(b2, 2) * b3 * c0 * c1 * d0 +
          8 * b0 * pow(b1, 2) * c0 * c1 * d5 +
          8 * b0 * pow(b1, 2) * c0 * c5 * d1 +
          8 * b0 * pow(b1, 2) * c1 * c5 * d0 +
          2 * pow(b0, 2) * b1 * c0 * c3 * d3 +
          2 * pow(b0, 2) * b3 * c0 * c1 * d3 +
          2 * pow(b0, 2) * b3 * c0 * c3 * d1 +
          2 * pow(b0, 2) * b3 * c1 * c3 * d0 -
          16 * pow(b1, 2) * b2 * c0 * c4 * d0 +
          16 * pow(b1, 2) * b4 * c0 * c2 * d0 +
          8 * pow(b1, 2) * b5 * c0 * c1 * d0 +
          16 * b0 * pow(b2, 2) * c0 * c2 * d4 +
          16 * b0 * pow(b2, 2) * c0 * c4 * d2 +
          16 * b0 * pow(b2, 2) * c2 * c4 * d0 +
          8 * b0 * pow(b3, 2) * c1 * c2 * d2 -
          8 * b1 * pow(b2, 2) * c0 * c5 * d0 +
          32 * b1 * pow(b2, 2) * c1 * c3 * d1 +
          8 * b1 * pow(b3, 2) * c0 * c2 * d2 +
          8 * b1 * pow(b4, 2) * c0 * c3 * d0 +
          16 * b1 * pow(b5, 2) * c0 * c1 * d1 -
          8 * b2 * pow(b3, 2) * c0 * c1 * d2 -
          8 * b2 * pow(b3, 2) * c0 * c2 * d1 -
          8 * b2 * pow(b3, 2) * c1 * c2 * d0 -
          8 * b3 * pow(b4, 2) * c0 * c1 * d0 -
          8 * pow(b0, 2) * b1 * c1 * c5 * d1 +
          8 * pow(b0, 2) * b1 * c2 * c3 * d2 -
          16 * pow(b0, 2) * b2 * c1 * c4 * d1 -
          8 * pow(b0, 2) * b3 * c1 * c2 * d2 +
          16 * pow(b0, 2) * b4 * c1 * c2 * d1 +
          16 * pow(b2, 2) * b4 * c0 * c2 * d0 +
          8 * pow(b2, 2) * b5 * c0 * c1 * d0 -
          8 * b0 * pow(b2, 2) * c1 * c3 * d3 +
          2 * b0 * pow(b3, 2) * c0 * c1 * d5 -
          4 * b0 * pow(b3, 2) * c0 * c2 * d4 -
          4 * b0 * pow(b3, 2) * c0 * c4 * d2 +
          2 * b0 * pow(b3, 2) * c0 * c5 * d1 +
          2 * b0 * pow(b3, 2) * c1 * c5 * d0 -
          4 * b0 * pow(b3, 2) * c2 * c4 * d0 +
          32 * b0 * pow(b4, 2) * c1 * c2 * d2 +
          4 * b0 * pow(b5, 2) * c0 * c1 * d3 +
          4 * b0 * pow(b5, 2) * c0 * c3 * d1 +
          4 * b0 * pow(b5, 2) * c1 * c3 * d0 -
          8 * b1 * pow(b2, 2) * c0 * c3 * d3 -
          6 * b1 * pow(b3, 2) * c0 * c5 * d0 -
          32 * b1 * pow(b4, 2) * c0 * c2 * d2 -
          4 * b1 * pow(b5, 2) * c0 * c3 * d0 +
          4 * b2 * pow(b3, 2) * c0 * c4 * d0 -
          4 * b3 * pow(b5, 2) * c0 * c1 * d0 -
          2 * pow(b0, 2) * b1 * c0 * c3 * d5 -
          8 * pow(b0, 2) * b1 * c0 * c4 * d4 -
          2 * pow(b0, 2) * b1 * c0 * c5 * d3 -
          2 * pow(b0, 2) * b1 * c3 * c5 * d0 +
          4 * pow(b0, 2) * b2 * c0 * c3 * d4 +
          4 * pow(b0, 2) * b2 * c0 * c4 * d3 +
          4 * pow(b0, 2) * b2 * c3 * c4 * d0 -
          2 * pow(b0, 2) * b3 * c0 * c1 * d5 +
          4 * pow(b0, 2) * b3 * c0 * c2 * d4 +
          4 * pow(b0, 2) * b3 * c0 * c4 * d2 -
          2 * pow(b0, 2) * b3 * c0 * c5 * d1 -
          2 * pow(b0, 2) * b3 * c1 * c5 * d0 +
          4 * pow(b0, 2) * b3 * c2 * c4 * d0 -
          8 * pow(b0, 2) * b4 * c0 * c1 * d4 +
          4 * pow(b0, 2) * b4 * c0 * c2 * d3 +
          4 * pow(b0, 2) * b4 * c0 * c3 * d2 -
          8 * pow(b0, 2) * b4 * c0 * c4 * d1 -
          8 * pow(b0, 2) * b4 * c1 * c4 * d0 +
          4 * pow(b0, 2) * b4 * c2 * c3 * d0 -
          2 * pow(b0, 2) * b5 * c0 * c1 * d3 -
          2 * pow(b0, 2) * b5 * c0 * c3 * d1 -
          2 * pow(b0, 2) * b5 * c1 * c3 * d0 -
          32 * pow(b1, 2) * b2 * c1 * c2 * d3 -
          32 * pow(b1, 2) * b2 * c1 * c3 * d2 -
          32 * pow(b1, 2) * b2 * c2 * c3 * d1 -
          32 * pow(b1, 2) * b3 * c1 * c2 * d2 +
          8 * pow(b2, 2) * b3 * c0 * c1 * d3 +
          8 * pow(b2, 2) * b3 * c0 * c3 * d1 +
          8 * pow(b2, 2) * b3 * c1 * c3 * d0 +
          4 * pow(b3, 2) * b4 * c0 * c2 * d0 +
          2 * pow(b3, 2) * b5 * c0 * c1 * d0 +
          8 * b0 * pow(b1, 2) * c1 * c3 * d5 +
          8 * b0 * pow(b1, 2) * c1 * c5 * d3 -
          16 * b0 * pow(b1, 2) * c2 * c3 * d4 -
          16 * b0 * pow(b1, 2) * c2 * c4 * d3 -
          16 * b0 * pow(b1, 2) * c3 * c4 * d2 +
          8 * b0 * pow(b1, 2) * c3 * c5 * d1 -
          8 * b0 * pow(b4, 2) * c0 * c1 * d5 -
          8 * b0 * pow(b4, 2) * c0 * c5 * d1 -
          8 * b0 * pow(b4, 2) * c1 * c5 * d0 -
          8 * b0 * pow(b5, 2) * c1 * c2 * d2 +
          64 * b1 * pow(b2, 2) * c1 * c2 * d4 +
          64 * b1 * pow(b2, 2) * c1 * c4 * d2 -
          32 * b1 * pow(b2, 2) * c1 * c5 * d1 +
          64 * b1 * pow(b2, 2) * c2 * c4 * d1 +
          16 * b1 * pow(b4, 2) * c0 * c5 * d0 +
          8 * b1 * pow(b5, 2) * c0 * c2 * d2 -
          8 * pow(b0, 2) * b1 * c2 * c5 * d2 -
          16 * pow(b0, 2) * b2 * c2 * c4 * d2 +
          8 * pow(b0, 2) * b5 * c1 * c2 * d2 +
          16 * pow(b1, 2) * b2 * c0 * c3 * d4 +
          16 * pow(b1, 2) * b2 * c0 * c4 * d3 +
          16 * pow(b1, 2) * b2 * c3 * c4 * d0 +
          8 * pow(b1, 2) * b3 * c0 * c1 * d5 +
          8 * pow(b1, 2) * b3 * c0 * c5 * d1 +
          8 * pow(b1, 2) * b3 * c1 * c5 * d0 +
          8 * pow(b1, 2) * b5 * c0 * c1 * d3 +
          8 * pow(b1, 2) * b5 * c0 * c3 * d1 +
          8 * pow(b1, 2) * b5 * c1 * c3 * d0 +
          64 * pow(b2, 2) * b4 * c1 * c2 * d1 +
          8 * b0 * pow(b2, 2) * c1 * c3 * d5 +
          32 * b0 * pow(b2, 2) * c1 * c4 * d4 +
          8 * b0 * pow(b2, 2) * c1 * c5 * d3 -
          16 * b0 * pow(b2, 2) * c2 * c3 * d4 -
          16 * b0 * pow(b2, 2) * c2 * c4 * d3 -
          16 * b0 * pow(b2, 2) * c3 * c4 * d2 +
          8 * b0 * pow(b2, 2) * c3 * c5 * d1 -
          2 * b0 * pow(b5, 2) * c0 * c1 * d5 +
          4 * b0 * pow(b5, 2) * c0 * c2 * d4 +
          4 * b0 * pow(b5, 2) * c0 * c4 * d2 -
          2 * b0 * pow(b5, 2) * c0 * c5 * d1 -
          2 * b0 * pow(b5, 2) * c1 * c5 * d0 +
          4 * b0 * pow(b5, 2) * c2 * c4 * d0 -
          32 * b1 * pow(b2, 2) * c0 * c4 * d4 -
          2 * b1 * pow(b5, 2) * c0 * c5 * d0 +
          8 * b1 * pow(b5, 2) * c1 * c3 * d1 -
          4 * b2 * pow(b5, 2) * c0 * c4 * d0 -
          4 * b4 * pow(b5, 2) * c0 * c2 * d0 +
          2 * pow(b0, 2) * b1 * c0 * c5 * d5 -
          4 * pow(b0, 2) * b2 * c0 * c4 * d5 -
          4 * pow(b0, 2) * b2 * c0 * c5 * d4 -
          4 * pow(b0, 2) * b2 * c4 * c5 * d0 -
          4 * pow(b0, 2) * b4 * c0 * c2 * d5 -
          4 * pow(b0, 2) * b4 * c0 * c5 * d2 -
          4 * pow(b0, 2) * b4 * c2 * c5 * d0 +
          2 * pow(b0, 2) * b5 * c0 * c1 * d5 -
          4 * pow(b0, 2) * b5 * c0 * c2 * d4 -
          4 * pow(b0, 2) * b5 * c0 * c4 * d2 +
          2 * pow(b0, 2) * b5 * c0 * c5 * d1 +
          2 * pow(b0, 2) * b5 * c1 * c5 * d0 -
          4 * pow(b0, 2) * b5 * c2 * c4 * d0 +
          32 * pow(b1, 2) * b2 * c1 * c2 * d5 +
          32 * pow(b1, 2) * b2 * c1 * c5 * d2 -
          64 * pow(b1, 2) * b2 * c2 * c4 * d2 +
          32 * pow(b1, 2) * b2 * c2 * c5 * d1 +
          32 * pow(b1, 2) * b5 * c1 * c2 * d2 -
          16 * pow(b2, 2) * b3 * c0 * c2 * d4 -
          16 * pow(b2, 2) * b3 * c0 * c4 * d2 -
          16 * pow(b2, 2) * b3 * c2 * c4 * d0 -
          16 * pow(b2, 2) * b4 * c0 * c2 * d3 -
          16 * pow(b2, 2) * b4 * c0 * c3 * d2 -
          16 * pow(b2, 2) * b4 * c2 * c3 * d0 -
          8 * pow(b2, 2) * b5 * c0 * c1 * d3 -
          8 * pow(b2, 2) * b5 * c0 * c3 * d1 -
          8 * pow(b2, 2) * b5 * c1 * c3 * d0 -
          16 * b0 * pow(b1, 2) * c1 * c5 * d5 +
          16 * b0 * pow(b1, 2) * c2 * c4 * d5 +
          16 * b0 * pow(b1, 2) * c2 * c5 * d4 +
          16 * b0 * pow(b1, 2) * c4 * c5 * d2 -
          6 * b0 * pow(b5, 2) * c1 * c3 * d3 +
          32 * b1 * pow(b4, 2) * c2 * c3 * d2 +
          2 * b1 * pow(b5, 2) * c0 * c3 * d3 -
          32 * b3 * pow(b4, 2) * c1 * c2 * d2 +
          2 * b3 * pow(b5, 2) * c0 * c1 * d3 +
          2 * b3 * pow(b5, 2) * c0 * c3 * d1 +
          2 * b3 * pow(b5, 2) * c1 * c3 * d0 +
          8 * pow(b0, 2) * b1 * c3 * c4 * d4 -
          2 * pow(b0, 2) * b1 * c3 * c5 * d3 -
          4 * pow(b0, 2) * b2 * c3 * c4 * d3 -
          2 * pow(b0, 2) * b3 * c1 * c3 * d5 -
          8 * pow(b0, 2) * b3 * c1 * c4 * d4 -
          2 * pow(b0, 2) * b3 * c1 * c5 * d3 +
          4 * pow(b0, 2) * b3 * c2 * c3 * d4 +
          4 * pow(b0, 2) * b3 * c2 * c4 * d3 +
          4 * pow(b0, 2) * b3 * c3 * c4 * d2 -
          2 * pow(b0, 2) * b3 * c3 * c5 * d1 -
          4 * pow(b0, 2) * b4 * c2 * c3 * d3 +
          6 * pow(b0, 2) * b5 * c1 * c3 * d3 -
          16 * pow(b1, 2) * b4 * c0 * c2 * d5 -
          16 * pow(b1, 2) * b4 * c0 * c5 * d2 -
          16 * pow(b1, 2) * b4 * c2 * c5 * d0 -
          16 * pow(b1, 2) * b5 * c0 * c1 * d5 -
          16 * pow(b1, 2) * b5 * c0 * c5 * d1 -
          16 * pow(b1, 2) * b5 * c1 * c5 * d0 -
          8 * b0 * pow(b2, 2) * c1 * c5 * d5 +
          8 * b1 * pow(b2, 2) * c0 * c5 * d5 -
          8 * b1 * pow(b3, 2) * c2 * c5 * d2 -
          8 * b1 * pow(b4, 2) * c0 * c3 * d5 -
          8 * b1 * pow(b4, 2) * c0 * c5 * d3 -
          8 * b1 * pow(b4, 2) * c3 * c5 * d0 +
          16 * b1 * pow(b5, 2) * c1 * c2 * d4 +
          16 * b1 * pow(b5, 2) * c1 * c4 * d2 -
          24 * b1 * pow(b5, 2) * c1 * c5 * d1 -
          8 * b1 * pow(b5, 2) * c2 * c3 * d2 +
          16 * b1 * pow(b5, 2) * c2 * c4 * d1 +
          8 * b2 * pow(b3, 2) * c1 * c2 * d5 +
          8 * b2 * pow(b3, 2) * c1 * c5 * d2 -
          16 * b2 * pow(b3, 2) * c2 * c4 * d2 +
          8 * b2 * pow(b3, 2) * c2 * c5 * d1 -
          16 * b2 * pow(b5, 2) * c1 * c4 * d1 +
          8 * b3 * pow(b4, 2) * c0 * c1 * d5 +
          8 * b3 * pow(b4, 2) * c0 * c5 * d1 +
          8 * b3 * pow(b4, 2) * c1 * c5 * d0 +
          8 * b3 * pow(b5, 2) * c1 * c2 * d2 -
          16 * b4 * pow(b5, 2) * c1 * c2 * d1 -
          8 * pow(b3, 2) * b5 * c1 * c2 * d2 -
          2 * b0 * pow(b3, 2) * c1 * c5 * d5 +
          4 * b0 * pow(b3, 2) * c2 * c4 * d5 +
          4 * b0 * pow(b3, 2) * c2 * c5 * d4 +
          4 * b0 * pow(b3, 2) * c4 * c5 * d2 +
          2 * b0 * pow(b5, 2) * c1 * c3 * d5 +
          8 * b0 * pow(b5, 2) * c1 * c4 * d4 +
          2 * b0 * pow(b5, 2) * c1 * c5 * d3 -
          4 * b0 * pow(b5, 2) * c2 * c3 * d4 -
          4 * b0 * pow(b5, 2) * c2 * c4 * d3 -
          4 * b0 * pow(b5, 2) * c3 * c4 * d2 +
          2 * b0 * pow(b5, 2) * c3 * c5 * d1 +
          32 * b1 * pow(b2, 2) * c3 * c4 * d4 +
          8 * b1 * pow(b2, 2) * c3 * c5 * d3 +
          6 * b1 * pow(b3, 2) * c0 * c5 * d5 +
          2 * b1 * pow(b5, 2) * c0 * c3 * d5 -
          8 * b1 * pow(b5, 2) * c0 * c4 * d4 +
          2 * b1 * pow(b5, 2) * c0 * c5 * d3 +
          2 * b1 * pow(b5, 2) * c3 * c5 * d0 -
          4 * b2 * pow(b3, 2) * c0 * c4 * d5 -
          4 * b2 * pow(b3, 2) * c0 * c5 * d4 -
          4 * b2 * pow(b3, 2) * c4 * c5 * d0 +
          4 * b2 * pow(b5, 2) * c0 * c3 * d4 +
          4 * b2 * pow(b5, 2) * c0 * c4 * d3 +
          4 * b2 * pow(b5, 2) * c3 * c4 * d0 +
          2 * b3 * pow(b5, 2) * c0 * c1 * d5 -
          4 * b3 * pow(b5, 2) * c0 * c2 * d4 -
          4 * b3 * pow(b5, 2) * c0 * c4 * d2 +
          2 * b3 * pow(b5, 2) * c0 * c5 * d1 +
          2 * b3 * pow(b5, 2) * c1 * c5 * d0 -
          4 * b3 * pow(b5, 2) * c2 * c4 * d0 +
          4 * b4 * pow(b5, 2) * c0 * c2 * d3 +
          4 * b4 * pow(b5, 2) * c0 * c3 * d2 +
          4 * b4 * pow(b5, 2) * c2 * c3 * d0 +
          4 * pow(b0, 2) * b1 * c3 * c5 * d5 +
          4 * pow(b0, 2) * b3 * c1 * c5 * d5 -
          8 * pow(b0, 2) * b3 * c2 * c4 * d5 -
          8 * pow(b0, 2) * b3 * c2 * c5 * d4 -
          8 * pow(b0, 2) * b3 * c4 * c5 * d2 +
          8 * pow(b0, 2) * b4 * c1 * c4 * d5 +
          8 * pow(b0, 2) * b4 * c1 * c5 * d4 +
          8 * pow(b0, 2) * b4 * c4 * c5 * d1 -
          4 * pow(b0, 2) * b5 * c1 * c3 * d5 -
          16 * pow(b0, 2) * b5 * c1 * c4 * d4 -
          4 * pow(b0, 2) * b5 * c1 * c5 * d3 +
          8 * pow(b0, 2) * b5 * c2 * c3 * d4 +
          8 * pow(b0, 2) * b5 * c2 * c4 * d3 +
          8 * pow(b0, 2) * b5 * c3 * c4 * d2 -
          4 * pow(b0, 2) * b5 * c3 * c5 * d1 -
          8 * pow(b2, 2) * b3 * c1 * c3 * d5 -
          32 * pow(b2, 2) * b3 * c1 * c4 * d4 -
          8 * pow(b2, 2) * b3 * c1 * c5 * d3 +
          16 * pow(b2, 2) * b3 * c2 * c3 * d4 +
          16 * pow(b2, 2) * b3 * c2 * c4 * d3 +
          16 * pow(b2, 2) * b3 * c3 * c4 * d2 -
          8 * pow(b2, 2) * b3 * c3 * c5 * d1 +
          16 * pow(b2, 2) * b4 * c2 * c3 * d3 +
          8 * pow(b2, 2) * b5 * c1 * c3 * d3 -
          4 * pow(b3, 2) * b4 * c0 * c2 * d5 -
          4 * pow(b3, 2) * b4 * c0 * c5 * d2 -
          4 * pow(b3, 2) * b4 * c2 * c5 * d0 -
          2 * pow(b3, 2) * b5 * c0 * c1 * d5 +
          4 * pow(b3, 2) * b5 * c0 * c2 * d4 +
          4 * pow(b3, 2) * b5 * c0 * c4 * d2 -
          2 * pow(b3, 2) * b5 * c0 * c5 * d1 -
          2 * pow(b3, 2) * b5 * c1 * c5 * d0 +
          4 * pow(b3, 2) * b5 * c2 * c4 * d0 +
          8 * b0 * pow(b4, 2) * c1 * c5 * d5 -
          8 * b1 * pow(b4, 2) * c0 * c5 * d5 -
          16 * pow(b1, 2) * b2 * c3 * c4 * d5 -
          16 * pow(b1, 2) * b2 * c3 * c5 * d4 -
          16 * pow(b1, 2) * b2 * c4 * c5 * d3 -
          8 * pow(b1, 2) * b3 * c1 * c5 * d5 -
          8 * pow(b1, 2) * b5 * c1 * c3 * d5 -
          8 * pow(b1, 2) * b5 * c1 * c5 * d3 +
          16 * pow(b1, 2) * b5 * c2 * c3 * d4 +
          16 * pow(b1, 2) * b5 * c2 * c4 * d3 +
          16 * pow(b1, 2) * b5 * c3 * c4 * d2 -
          8 * pow(b1, 2) * b5 * c3 * c5 * d1 -
          8 * b1 * pow(b2, 2) * c3 * c5 * d5 +
          4 * pow(b0, 2) * b2 * c4 * c5 * d5 +
          4 * pow(b0, 2) * b4 * c2 * c5 * d5 +
          2 * pow(b0, 2) * b5 * c1 * c5 * d5 -
          4 * pow(b0, 2) * b5 * c2 * c4 * d5 -
          4 * pow(b0, 2) * b5 * c2 * c5 * d4 -
          4 * pow(b0, 2) * b5 * c4 * c5 * d2 +
          8 * pow(b2, 2) * b3 * c1 * c5 * d5 +
          8 * b1 * pow(b5, 2) * c3 * c4 * d4 -
          2 * b1 * pow(b5, 2) * c3 * c5 * d3 -
          4 * b2 * pow(b5, 2) * c3 * c4 * d3 -
          2 * b3 * pow(b5, 2) * c1 * c3 * d5 -
          8 * b3 * pow(b5, 2) * c1 * c4 * d4 -
          2 * b3 * pow(b5, 2) * c1 * c5 * d3 +
          4 * b3 * pow(b5, 2) * c2 * c3 * d4 +
          4 * b3 * pow(b5, 2) * c2 * c4 * d3 +
          4 * b3 * pow(b5, 2) * c3 * c4 * d2 -
          2 * b3 * pow(b5, 2) * c3 * c5 * d1 -
          4 * b4 * pow(b5, 2) * c2 * c3 * d3 +
          16 * pow(b1, 2) * b2 * c4 * c5 * d5 +
          16 * pow(b1, 2) * b4 * c2 * c5 * d5 +
          24 * pow(b1, 2) * b5 * c1 * c5 * d5 -
          16 * pow(b1, 2) * b5 * c2 * c4 * d5 -
          16 * pow(b1, 2) * b5 * c2 * c5 * d4 -
          16 * pow(b1, 2) * b5 * c4 * c5 * d2 +
          8 * b1 * pow(b4, 2) * c3 * c5 * d5 -
          8 * b3 * pow(b4, 2) * c1 * c5 * d5 +
          4 * b2 * pow(b3, 2) * c4 * c5 * d5 +
          4 * pow(b3, 2) * b4 * c2 * c5 * d5 +
          2 * pow(b3, 2) * b5 * c1 * c5 * d5 -
          4 * pow(b3, 2) * b5 * c2 * c4 * d5 -
          4 * pow(b3, 2) * b5 * c2 * c5 * d4 -
          4 * pow(b3, 2) * b5 * c4 * c5 * d2 +
          16 * b0 * b1 * b3 * c0 * c1 * d1 - 4 * b0 * b1 * b3 * c0 * c3 * d0 +
          16 * b0 * b1 * b2 * c0 * c1 * d4 - 8 * b0 * b1 * b2 * c0 * c2 * d3 -
          8 * b0 * b1 * b2 * c0 * c3 * d2 + 16 * b0 * b1 * b2 * c0 * c4 * d1 +
          16 * b0 * b1 * b2 * c1 * c4 * d0 - 8 * b0 * b1 * b2 * c2 * c3 * d0 -
          16 * b0 * b1 * b4 * c0 * c1 * d2 - 16 * b0 * b1 * b4 * c0 * c2 * d1 -
          16 * b0 * b1 * b4 * c1 * c2 * d0 - 16 * b0 * b1 * b5 * c0 * c1 * d1 +
          8 * b0 * b2 * b3 * c0 * c1 * d2 + 8 * b0 * b2 * b3 * c0 * c2 * d1 +
          8 * b0 * b2 * b3 * c1 * c2 * d0 + 4 * b0 * b1 * b3 * c0 * c5 * d0 +
          16 * b0 * b1 * b4 * c0 * c4 * d0 + 4 * b0 * b1 * b5 * c0 * c3 * d0 -
          8 * b0 * b2 * b3 * c0 * c4 * d0 - 8 * b0 * b2 * b4 * c0 * c3 * d0 -
          8 * b0 * b3 * b4 * c0 * c2 * d0 + 4 * b0 * b3 * b5 * c0 * c1 * d0 +
          8 * b0 * b1 * b2 * c0 * c2 * d5 + 8 * b0 * b1 * b2 * c0 * c5 * d2 +
          8 * b0 * b1 * b2 * c2 * c5 * d0 - 32 * b0 * b2 * b4 * c0 * c2 * d2 -
          8 * b0 * b2 * b5 * c0 * c1 * d2 - 8 * b0 * b2 * b5 * c0 * c2 * d1 -
          8 * b0 * b2 * b5 * c1 * c2 * d0 + 64 * b1 * b2 * b3 * c1 * c2 * d1 +
          16 * b0 * b1 * b2 * c2 * c3 * d3 + 16 * b0 * b1 * b3 * c1 * c2 * d4 +
          16 * b0 * b1 * b3 * c1 * c4 * d2 - 16 * b0 * b1 * b3 * c1 * c5 * d1 -
          16 * b0 * b1 * b3 * c2 * c3 * d2 + 16 * b0 * b1 * b3 * c2 * c4 * d1 +
          16 * b0 * b1 * b4 * c1 * c2 * d3 + 16 * b0 * b1 * b4 * c1 * c3 * d2 +
          16 * b0 * b1 * b4 * c2 * c3 * d1 - 4 * b0 * b1 * b5 * c0 * c5 * d0 -
          16 * b0 * b1 * b5 * c1 * c3 * d1 + 8 * b0 * b2 * b4 * c0 * c5 * d0 +
          8 * b0 * b2 * b5 * c0 * c4 * d0 - 32 * b0 * b3 * b4 * c1 * c2 * d1 +
          8 * b0 * b4 * b5 * c0 * c2 * d0 - 16 * b1 * b2 * b3 * c0 * c1 * d4 -
          16 * b1 * b2 * b3 * c0 * c4 * d1 - 16 * b1 * b2 * b3 * c1 * c4 * d0 -
          16 * b1 * b2 * b4 * c0 * c1 * d3 - 16 * b1 * b2 * b4 * c0 * c3 * d1 -
          16 * b1 * b2 * b4 * c1 * c3 * d0 - 16 * b1 * b3 * b5 * c0 * c1 * d1 +
          32 * b2 * b3 * b4 * c0 * c1 * d1 + 4 * b0 * b1 * b3 * c0 * c3 * d5 +
          4 * b0 * b1 * b3 * c0 * c5 * d3 + 4 * b0 * b1 * b3 * c3 * c5 * d0 -
          8 * b0 * b1 * b4 * c0 * c3 * d4 - 8 * b0 * b1 * b4 * c0 * c4 * d3 -
          8 * b0 * b1 * b4 * c3 * c4 * d0 - 4 * b0 * b1 * b5 * c0 * c3 * d3 +
          8 * b0 * b2 * b4 * c0 * c3 * d3 + 8 * b0 * b3 * b4 * c0 * c1 * d4 +
          8 * b0 * b3 * b4 * c0 * c4 * d1 + 8 * b0 * b3 * b4 * c1 * c4 * d0 -
          4 * b0 * b3 * b5 * c0 * c1 * d3 - 4 * b0 * b3 * b5 * c0 * c3 * d1 -
          4 * b0 * b3 * b5 * c1 * c3 * d0 - 128 * b1 * b2 * b4 * c1 * c2 * d2 -
          64 * b1 * b2 * b5 * c1 * c2 * d1 + 4 * b1 * b3 * b5 * c0 * c3 * d0 -
          8 * b2 * b3 * b4 * c0 * c3 * d0 - 16 * b0 * b1 * b2 * c1 * c4 * d5 -
          16 * b0 * b1 * b2 * c1 * c5 * d4 - 8 * b0 * b1 * b2 * c2 * c3 * d5 -
          8 * b0 * b1 * b2 * c2 * c5 * d3 - 8 * b0 * b1 * b2 * c3 * c5 * d2 -
          16 * b0 * b1 * b2 * c4 * c5 * d1 + 16 * b0 * b1 * b3 * c2 * c5 * d2 -
          16 * b0 * b1 * b5 * c1 * c2 * d4 - 16 * b0 * b1 * b5 * c1 * c4 * d2 +
          32 * b0 * b1 * b5 * c1 * c5 * d1 - 16 * b0 * b1 * b5 * c2 * c4 * d1 -
          8 * b0 * b2 * b3 * c1 * c2 * d5 - 8 * b0 * b2 * b3 * c1 * c5 * d2 +
          32 * b0 * b2 * b3 * c2 * c4 * d2 - 8 * b0 * b2 * b3 * c2 * c5 * d1 -
          32 * b0 * b2 * b4 * c1 * c2 * d4 - 32 * b0 * b2 * b4 * c1 * c4 * d2 +
          32 * b0 * b2 * b4 * c2 * c3 * d2 - 32 * b0 * b2 * b4 * c2 * c4 * d1 +
          32 * b0 * b2 * b5 * c1 * c4 * d1 + 16 * b1 * b2 * b4 * c0 * c1 * d5 +
          32 * b1 * b2 * b4 * c0 * c2 * d4 + 32 * b1 * b2 * b4 * c0 * c4 * d2 +
          16 * b1 * b2 * b4 * c0 * c5 * d1 + 16 * b1 * b2 * b4 * c1 * c5 * d0 +
          32 * b1 * b2 * b4 * c2 * c4 * d0 + 8 * b1 * b2 * b5 * c0 * c2 * d3 +
          8 * b1 * b2 * b5 * c0 * c3 * d2 + 8 * b1 * b2 * b5 * c2 * c3 * d0 -
          16 * b1 * b3 * b5 * c0 * c2 * d2 + 16 * b1 * b4 * b5 * c0 * c1 * d2 +
          16 * b1 * b4 * b5 * c0 * c2 * d1 + 16 * b1 * b4 * b5 * c1 * c2 * d0 +
          32 * b2 * b3 * b4 * c0 * c2 * d2 + 8 * b2 * b3 * b5 * c0 * c1 * d2 +
          8 * b2 * b3 * b5 * c0 * c2 * d1 + 8 * b2 * b3 * b5 * c1 * c2 * d0 -
          32 * b2 * b4 * b5 * c0 * c1 * d1 - 8 * b0 * b1 * b3 * c0 * c5 * d5 -
          8 * b0 * b1 * b4 * c0 * c4 * d5 - 8 * b0 * b1 * b4 * c0 * c5 * d4 -
          8 * b0 * b1 * b4 * c4 * c5 * d0 + 16 * b0 * b1 * b5 * c0 * c4 * d4 +
          8 * b0 * b2 * b3 * c0 * c4 * d5 + 8 * b0 * b2 * b3 * c0 * c5 * d4 +
          8 * b0 * b2 * b3 * c4 * c5 * d0 - 8 * b0 * b2 * b5 * c0 * c3 * d4 -
          8 * b0 * b2 * b5 * c0 * c4 * d3 - 8 * b0 * b2 * b5 * c3 * c4 * d0 +
          8 * b0 * b3 * b4 * c0 * c2 * d5 + 8 * b0 * b3 * b4 * c0 * c5 * d2 +
          8 * b0 * b3 * b4 * c2 * c5 * d0 + 8 * b0 * b4 * b5 * c0 * c1 * d4 -
          8 * b0 * b4 * b5 * c0 * c2 * d3 - 8 * b0 * b4 * b5 * c0 * c3 * d2 +
          8 * b0 * b4 * b5 * c0 * c4 * d1 + 8 * b0 * b4 * b5 * c1 * c4 * d0 -
          8 * b0 * b4 * b5 * c2 * c3 * d0 + 8 * b1 * b3 * b5 * c0 * c5 * d0 -
          16 * b1 * b4 * b5 * c0 * c4 * d0 - 16 * b2 * b3 * b4 * c0 * c5 * d0 +
          16 * b2 * b4 * b5 * c0 * c3 * d0 + 8 * b0 * b2 * b5 * c1 * c2 * d5 +
          8 * b0 * b2 * b5 * c1 * c5 * d2 + 8 * b0 * b2 * b5 * c2 * c5 * d1 -
          8 * b1 * b2 * b5 * c0 * c2 * d5 - 8 * b1 * b2 * b5 * c0 * c5 * d2 -
          8 * b1 * b2 * b5 * c2 * c5 * d0 + 4 * b0 * b1 * b5 * c0 * c5 * d5 -
          8 * b0 * b2 * b4 * c0 * c5 * d5 + 16 * b1 * b2 * b3 * c1 * c4 * d5 +
          16 * b1 * b2 * b3 * c1 * c5 * d4 + 16 * b1 * b2 * b3 * c4 * c5 * d1 +
          16 * b1 * b2 * b4 * c1 * c3 * d5 + 16 * b1 * b2 * b4 * c1 * c5 * d3 -
          32 * b1 * b2 * b4 * c2 * c3 * d4 - 32 * b1 * b2 * b4 * c2 * c4 * d3 -
          32 * b1 * b2 * b4 * c3 * c4 * d2 + 16 * b1 * b2 * b4 * c3 * c5 * d1 -
          16 * b1 * b2 * b5 * c2 * c3 * d3 - 16 * b1 * b3 * b5 * c1 * c2 * d4 -
          16 * b1 * b3 * b5 * c1 * c4 * d2 + 16 * b1 * b3 * b5 * c1 * c5 * d1 +
          16 * b1 * b3 * b5 * c2 * c3 * d2 - 16 * b1 * b3 * b5 * c2 * c4 * d1 -
          16 * b1 * b4 * b5 * c1 * c2 * d3 - 16 * b1 * b4 * b5 * c1 * c3 * d2 -
          16 * b1 * b4 * b5 * c2 * c3 * d1 + 32 * b2 * b3 * b4 * c1 * c2 * d4 +
          32 * b2 * b3 * b4 * c1 * c4 * d2 - 32 * b2 * b3 * b4 * c1 * c5 * d1 -
          32 * b2 * b3 * b4 * c2 * c3 * d2 + 32 * b2 * b3 * b4 * c2 * c4 * d1 +
          8 * b2 * b4 * b5 * c0 * c5 * d0 + 32 * b3 * b4 * b5 * c1 * c2 * d1 -
          4 * b0 * b1 * b3 * c3 * c5 * d5 + 8 * b0 * b1 * b4 * c3 * c4 * d5 +
          8 * b0 * b1 * b4 * c3 * c5 * d4 + 8 * b0 * b1 * b4 * c4 * c5 * d3 -
          16 * b0 * b1 * b5 * c3 * c4 * d4 + 4 * b0 * b1 * b5 * c3 * c5 * d3 -
          8 * b0 * b2 * b4 * c3 * c5 * d3 + 8 * b0 * b2 * b5 * c3 * c4 * d3 -
          8 * b0 * b3 * b4 * c1 * c4 * d5 - 8 * b0 * b3 * b4 * c1 * c5 * d4 -
          8 * b0 * b3 * b4 * c4 * c5 * d1 + 4 * b0 * b3 * b5 * c1 * c3 * d5 +
          16 * b0 * b3 * b5 * c1 * c4 * d4 + 4 * b0 * b3 * b5 * c1 * c5 * d3 -
          8 * b0 * b3 * b5 * c2 * c3 * d4 - 8 * b0 * b3 * b5 * c2 * c4 * d3 -
          8 * b0 * b3 * b5 * c3 * c4 * d2 + 4 * b0 * b3 * b5 * c3 * c5 * d1 +
          8 * b0 * b4 * b5 * c2 * c3 * d3 - 4 * b1 * b3 * b5 * c0 * c3 * d5 -
          4 * b1 * b3 * b5 * c0 * c5 * d3 - 4 * b1 * b3 * b5 * c3 * c5 * d0 +
          8 * b1 * b4 * b5 * c0 * c3 * d4 + 8 * b1 * b4 * b5 * c0 * c4 * d3 +
          8 * b1 * b4 * b5 * c3 * c4 * d0 + 8 * b2 * b3 * b4 * c0 * c3 * d5 +
          8 * b2 * b3 * b4 * c0 * c5 * d3 + 8 * b2 * b3 * b4 * c3 * c5 * d0 -
          8 * b2 * b4 * b5 * c0 * c3 * d3 - 8 * b3 * b4 * b5 * c0 * c1 * d4 -
          8 * b3 * b4 * b5 * c0 * c4 * d1 - 8 * b3 * b4 * b5 * c1 * c4 * d0 -
          32 * b1 * b2 * b4 * c1 * c5 * d5 + 8 * b1 * b2 * b5 * c2 * c3 * d5 +
          8 * b1 * b2 * b5 * c2 * c5 * d3 + 8 * b1 * b2 * b5 * c3 * c5 * d2 -
          8 * b2 * b3 * b5 * c1 * c2 * d5 - 8 * b2 * b3 * b5 * c1 * c5 * d2 -
          8 * b2 * b3 * b5 * c2 * c5 * d1 + 32 * b2 * b4 * b5 * c1 * c5 * d1 -
          4 * b0 * b1 * b5 * c3 * c5 * d5 - 8 * b0 * b2 * b3 * c4 * c5 * d5 +
          8 * b0 * b2 * b4 * c3 * c5 * d5 - 8 * b0 * b3 * b4 * c2 * c5 * d5 -
          4 * b0 * b3 * b5 * c1 * c5 * d5 + 8 * b0 * b3 * b5 * c2 * c4 * d5 +
          8 * b0 * b3 * b5 * c2 * c5 * d4 + 8 * b0 * b3 * b5 * c4 * c5 * d2 -
          8 * b0 * b4 * b5 * c1 * c4 * d5 - 8 * b0 * b4 * b5 * c1 * c5 * d4 -
          8 * b0 * b4 * b5 * c4 * c5 * d1 - 4 * b1 * b3 * b5 * c0 * c5 * d5 +
          8 * b1 * b4 * b5 * c0 * c4 * d5 + 8 * b1 * b4 * b5 * c0 * c5 * d4 +
          8 * b1 * b4 * b5 * c4 * c5 * d0 + 8 * b2 * b3 * b4 * c0 * c5 * d5 -
          8 * b2 * b4 * b5 * c0 * c3 * d5 - 8 * b2 * b4 * b5 * c0 * c5 * d3 -
          8 * b2 * b4 * b5 * c3 * c5 * d0 + 4 * b1 * b3 * b5 * c3 * c5 * d5 -
          8 * b1 * b4 * b5 * c3 * c4 * d5 - 8 * b1 * b4 * b5 * c3 * c5 * d4 -
          8 * b1 * b4 * b5 * c4 * c5 * d3 - 8 * b2 * b3 * b4 * c3 * c5 * d5 +
          8 * b2 * b4 * b5 * c3 * c5 * d3 + 8 * b3 * b4 * b5 * c1 * c4 * d5 +
          8 * b3 * b4 * b5 * c1 * c5 * d4 + 8 * b3 * b4 * b5 * c4 * c5 * d1,
      4 * pow(b5, 3) * pow(c1, 3) - 4 * pow(b1, 3) * pow(c5, 3) +
          b1 * pow(b3, 2) * pow(c0, 3) - 4 * pow(b0, 2) * b3 * pow(c1, 3) -
          4 * b1 * pow(b4, 2) * pow(c0, 3) - 8 * b0 * pow(b5, 2) * pow(c1, 3) +
          b1 * pow(b5, 2) * pow(c0, 3) - pow(b0, 2) * b1 * pow(c5, 3) +
          8 * pow(b0, 2) * b4 * pow(c2, 3) + 4 * pow(b0, 2) * b5 * pow(c1, 3) -
          16 * pow(b2, 2) * b3 * pow(c1, 3) +
          32 * pow(b1, 2) * b4 * pow(c2, 3) +
          16 * pow(b2, 2) * b5 * pow(c1, 3) - b1 * pow(b3, 2) * pow(c5, 3) -
          4 * b3 * pow(b5, 2) * pow(c1, 3) + 8 * pow(b3, 2) * b4 * pow(c2, 3) -
          pow(b0, 3) * c1 * pow(c3, 2) + 4 * pow(b1, 3) * pow(c0, 2) * c3 +
          4 * pow(b0, 3) * c1 * pow(c4, 2) - pow(b0, 3) * c1 * pow(c5, 2) +
          8 * pow(b1, 3) * c0 * pow(c5, 2) - 4 * pow(b1, 3) * pow(c0, 2) * c5 +
          16 * pow(b1, 3) * pow(c2, 2) * c3 - 8 * pow(b2, 3) * pow(c0, 2) * c4 +
          pow(b5, 3) * pow(c0, 2) * c1 - 32 * pow(b2, 3) * pow(c1, 2) * c4 -
          16 * pow(b1, 3) * pow(c2, 2) * c5 + 4 * pow(b1, 3) * c3 * pow(c5, 2) -
          8 * pow(b2, 3) * pow(c3, 2) * c4 + pow(b5, 3) * c1 * pow(c3, 2) +
          2 * b0 * b1 * b3 * pow(c5, 3) - 16 * b0 * b3 * b4 * pow(c2, 3) +
          8 * b0 * b3 * b5 * pow(c1, 3) - 2 * b1 * b3 * b5 * pow(c0, 3) +
          4 * b2 * b3 * b4 * pow(c0, 3) - 4 * b2 * b4 * b5 * pow(c0, 3) +
          2 * pow(b0, 3) * c1 * c3 * c5 - 4 * pow(b0, 3) * c2 * c3 * c4 -
          8 * pow(b1, 3) * c0 * c3 * c5 + 16 * pow(b2, 3) * c0 * c3 * c4 -
          2 * pow(b5, 3) * c0 * c1 * c3 + 4 * pow(b0, 3) * c2 * c4 * c5 -
          b0 * pow(b3, 2) * pow(c0, 2) * c1 +
          pow(b0, 2) * b1 * c0 * pow(c3, 2) +
          4 * b0 * pow(b4, 2) * pow(c0, 2) * c1 -
          4 * pow(b0, 2) * b1 * c0 * pow(c4, 2) +
          4 * pow(b0, 2) * b1 * pow(c1, 2) * c3 -
          4 * pow(b1, 2) * b3 * pow(c0, 2) * c1 -
          4 * b0 * pow(b2, 2) * c1 * pow(c3, 2) +
          4 * b0 * pow(b3, 2) * c1 * pow(c2, 2) -
          b0 * pow(b5, 2) * pow(c0, 2) * c1 -
          4 * b1 * pow(b2, 2) * c0 * pow(c3, 2) +
          4 * b1 * pow(b2, 2) * pow(c0, 2) * c3 +
          4 * b1 * pow(b3, 2) * c0 * pow(c2, 2) +
          pow(b0, 2) * b1 * c0 * pow(c5, 2) +
          4 * pow(b0, 2) * b1 * pow(c2, 2) * c3 -
          4 * pow(b0, 2) * b3 * c1 * pow(c2, 2) -
          4 * pow(b2, 2) * b3 * pow(c0, 2) * c1 -
          8 * b0 * pow(b1, 2) * c1 * pow(c5, 2) +
          16 * b0 * pow(b2, 2) * c1 * pow(c4, 2) +
          16 * b0 * pow(b4, 2) * c1 * pow(c2, 2) -
          16 * b1 * pow(b2, 2) * c0 * pow(c4, 2) +
          16 * b1 * pow(b2, 2) * pow(c1, 2) * c3 -
          16 * b1 * pow(b4, 2) * c0 * pow(c2, 2) +
          8 * b1 * pow(b5, 2) * c0 * pow(c1, 2) -
          4 * pow(b0, 2) * b1 * pow(c1, 2) * c5 -
          8 * pow(b0, 2) * b2 * pow(c1, 2) * c4 +
          8 * pow(b0, 2) * b4 * pow(c1, 2) * c2 -
          8 * pow(b1, 2) * b2 * pow(c0, 2) * c4 -
          16 * pow(b1, 2) * b3 * c1 * pow(c2, 2) +
          8 * pow(b1, 2) * b4 * pow(c0, 2) * c2 +
          4 * pow(b1, 2) * b5 * pow(c0, 2) * c1 -
          4 * b0 * pow(b2, 2) * c1 * pow(c5, 2) -
          4 * b0 * pow(b5, 2) * c1 * pow(c2, 2) +
          4 * b1 * pow(b2, 2) * c0 * pow(c5, 2) -
          4 * b1 * pow(b2, 2) * pow(c0, 2) * c5 +
          4 * b1 * pow(b4, 2) * pow(c0, 2) * c3 +
          4 * b1 * pow(b5, 2) * c0 * pow(c2, 2) -
          4 * b3 * pow(b4, 2) * pow(c0, 2) * c1 +
          4 * pow(b0, 2) * b1 * c3 * pow(c4, 2) -
          4 * pow(b0, 2) * b1 * pow(c2, 2) * c5 -
          8 * pow(b0, 2) * b2 * pow(c2, 2) * c4 -
          4 * pow(b0, 2) * b3 * c1 * pow(c4, 2) +
          4 * pow(b0, 2) * b5 * c1 * pow(c2, 2) +
          8 * pow(b2, 2) * b4 * pow(c0, 2) * c2 +
          4 * pow(b2, 2) * b5 * pow(c0, 2) * c1 -
          b0 * pow(b3, 2) * c1 * pow(c5, 2) -
          3 * b0 * pow(b5, 2) * c1 * pow(c3, 2) -
          16 * b1 * pow(b2, 2) * pow(c1, 2) * c5 +
          3 * b1 * pow(b3, 2) * c0 * pow(c5, 2) -
          3 * b1 * pow(b3, 2) * pow(c0, 2) * c5 +
          b1 * pow(b5, 2) * c0 * pow(c3, 2) -
          2 * b1 * pow(b5, 2) * pow(c0, 2) * c3 +
          2 * b2 * pow(b3, 2) * pow(c0, 2) * c4 -
          2 * b3 * pow(b5, 2) * pow(c0, 2) * c1 +
          2 * pow(b0, 2) * b1 * c3 * pow(c5, 2) -
          pow(b0, 2) * b1 * pow(c3, 2) * c5 -
          2 * pow(b0, 2) * b2 * pow(c3, 2) * c4 +
          2 * pow(b0, 2) * b3 * c1 * pow(c5, 2) -
          2 * pow(b0, 2) * b4 * c2 * pow(c3, 2) +
          3 * pow(b0, 2) * b5 * c1 * pow(c3, 2) -
          32 * pow(b1, 2) * b2 * pow(c2, 2) * c4 +
          16 * pow(b1, 2) * b5 * c1 * pow(c2, 2) +
          32 * pow(b2, 2) * b4 * pow(c1, 2) * c2 +
          2 * pow(b3, 2) * b4 * pow(c0, 2) * c2 +
          pow(b3, 2) * b5 * pow(c0, 2) * c1 +
          4 * b0 * pow(b4, 2) * c1 * pow(c5, 2) +
          4 * b0 * pow(b5, 2) * c1 * pow(c4, 2) +
          16 * b1 * pow(b2, 2) * c3 * pow(c4, 2) -
          4 * b1 * pow(b4, 2) * c0 * pow(c5, 2) +
          8 * b1 * pow(b4, 2) * pow(c0, 2) * c5 +
          16 * b1 * pow(b4, 2) * pow(c2, 2) * c3 -
          4 * b1 * pow(b5, 2) * c0 * pow(c4, 2) +
          4 * b1 * pow(b5, 2) * pow(c1, 2) * c3 -
          16 * b3 * pow(b4, 2) * c1 * pow(c2, 2) -
          8 * pow(b0, 2) * b5 * c1 * pow(c4, 2) -
          4 * pow(b1, 2) * b3 * c1 * pow(c5, 2) -
          16 * pow(b2, 2) * b3 * c1 * pow(c4, 2) -
          4 * b1 * pow(b2, 2) * c3 * pow(c5, 2) +
          4 * b1 * pow(b2, 2) * pow(c3, 2) * c5 -
          4 * b1 * pow(b3, 2) * pow(c2, 2) * c5 -
          b1 * pow(b5, 2) * pow(c0, 2) * c5 -
          4 * b1 * pow(b5, 2) * pow(c2, 2) * c3 -
          8 * b2 * pow(b3, 2) * pow(c2, 2) * c4 -
          2 * b2 * pow(b5, 2) * pow(c0, 2) * c4 +
          4 * b3 * pow(b5, 2) * c1 * pow(c2, 2) -
          2 * b4 * pow(b5, 2) * pow(c0, 2) * c2 +
          2 * pow(b0, 2) * b2 * c4 * pow(c5, 2) +
          2 * pow(b0, 2) * b4 * c2 * pow(c5, 2) +
          pow(b0, 2) * b5 * c1 * pow(c5, 2) +
          4 * pow(b2, 2) * b3 * c1 * pow(c5, 2) +
          8 * pow(b2, 2) * b4 * c2 * pow(c3, 2) +
          4 * pow(b2, 2) * b5 * c1 * pow(c3, 2) -
          4 * pow(b3, 2) * b5 * c1 * pow(c2, 2) -
          12 * b1 * pow(b5, 2) * pow(c1, 2) * c5 -
          8 * b2 * pow(b5, 2) * pow(c1, 2) * c4 -
          8 * b4 * pow(b5, 2) * pow(c1, 2) * c2 +
          8 * pow(b1, 2) * b2 * c4 * pow(c5, 2) +
          8 * pow(b1, 2) * b4 * c2 * pow(c5, 2) +
          12 * pow(b1, 2) * b5 * c1 * pow(c5, 2) +
          4 * b1 * pow(b4, 2) * c3 * pow(c5, 2) +
          4 * b1 * pow(b5, 2) * c3 * pow(c4, 2) -
          4 * b3 * pow(b4, 2) * c1 * pow(c5, 2) -
          4 * b3 * pow(b5, 2) * c1 * pow(c4, 2) -
          b1 * pow(b5, 2) * pow(c3, 2) * c5 +
          2 * b2 * pow(b3, 2) * c4 * pow(c5, 2) -
          2 * b2 * pow(b5, 2) * pow(c3, 2) * c4 -
          2 * b4 * pow(b5, 2) * c2 * pow(c3, 2) +
          2 * pow(b3, 2) * b4 * c2 * pow(c5, 2) +
          pow(b3, 2) * b5 * c1 * pow(c5, 2) +
          8 * b0 * b1 * b3 * c0 * pow(c1, 2) -
          2 * b0 * b1 * b3 * pow(c0, 2) * c3 -
          8 * b0 * b1 * b5 * c0 * pow(c1, 2) +
          8 * b0 * b1 * b2 * c2 * pow(c3, 2) -
          16 * b0 * b2 * b4 * c0 * pow(c2, 2) -
          4 * b0 * b1 * b3 * c0 * pow(c5, 2) +
          2 * b0 * b1 * b3 * pow(c0, 2) * c5 -
          8 * b0 * b1 * b3 * pow(c2, 2) * c3 +
          8 * b0 * b1 * b4 * pow(c0, 2) * c4 -
          2 * b0 * b1 * b5 * c0 * pow(c3, 2) +
          2 * b0 * b1 * b5 * pow(c0, 2) * c3 -
          4 * b0 * b2 * b3 * pow(c0, 2) * c4 +
          4 * b0 * b2 * b4 * c0 * pow(c3, 2) -
          4 * b0 * b2 * b4 * pow(c0, 2) * c3 -
          4 * b0 * b3 * b4 * pow(c0, 2) * c2 +
          2 * b0 * b3 * b5 * pow(c0, 2) * c1 +
          32 * b1 * b2 * b3 * pow(c1, 2) * c2 -
          8 * b0 * b1 * b3 * pow(c1, 2) * c5 +
          8 * b0 * b1 * b5 * c0 * pow(c4, 2) -
          8 * b0 * b1 * b5 * pow(c1, 2) * c3 -
          16 * b0 * b3 * b4 * pow(c1, 2) * c2 -
          64 * b1 * b2 * b4 * c1 * pow(c2, 2) -
          8 * b1 * b3 * b5 * c0 * pow(c1, 2) +
          16 * b2 * b3 * b4 * c0 * pow(c1, 2) +
          8 * b0 * b1 * b3 * pow(c2, 2) * c5 +
          2 * b0 * b1 * b5 * c0 * pow(c5, 2) -
          2 * b0 * b1 * b5 * pow(c0, 2) * c5 +
          16 * b0 * b2 * b3 * pow(c2, 2) * c4 -
          4 * b0 * b2 * b4 * c0 * pow(c5, 2) +
          4 * b0 * b2 * b4 * pow(c0, 2) * c5 +
          16 * b0 * b2 * b4 * pow(c2, 2) * c3 +
          4 * b0 * b2 * b5 * pow(c0, 2) * c4 +
          4 * b0 * b4 * b5 * pow(c0, 2) * c2 -
          32 * b1 * b2 * b5 * pow(c1, 2) * c2 -
          8 * b1 * b3 * b5 * c0 * pow(c2, 2) +
          16 * b2 * b3 * b4 * c0 * pow(c2, 2) -
          2 * b0 * b1 * b3 * c3 * pow(c5, 2) +
          16 * b0 * b1 * b5 * pow(c1, 2) * c5 +
          16 * b0 * b2 * b5 * pow(c1, 2) * c4 +
          2 * b1 * b3 * b5 * pow(c0, 2) * c3 -
          4 * b2 * b3 * b4 * pow(c0, 2) * c3 -
          16 * b2 * b4 * b5 * c0 * pow(c1, 2) -
          8 * b0 * b1 * b5 * c3 * pow(c4, 2) +
          8 * b0 * b3 * b5 * c1 * pow(c4, 2) -
          16 * b1 * b2 * b4 * c1 * pow(c5, 2) -
          8 * b1 * b2 * b5 * c2 * pow(c3, 2) -
          2 * b0 * b1 * b5 * c3 * pow(c5, 2) +
          2 * b0 * b1 * b5 * pow(c3, 2) * c5 -
          4 * b0 * b2 * b3 * c4 * pow(c5, 2) +
          4 * b0 * b2 * b4 * c3 * pow(c5, 2) -
          4 * b0 * b2 * b4 * pow(c3, 2) * c5 +
          4 * b0 * b2 * b5 * pow(c3, 2) * c4 -
          4 * b0 * b3 * b4 * c2 * pow(c5, 2) -
          2 * b0 * b3 * b5 * c1 * pow(c5, 2) +
          4 * b0 * b4 * b5 * c2 * pow(c3, 2) -
          2 * b1 * b3 * b5 * c0 * pow(c5, 2) +
          4 * b1 * b3 * b5 * pow(c0, 2) * c5 +
          8 * b1 * b3 * b5 * pow(c2, 2) * c3 -
          8 * b1 * b4 * b5 * pow(c0, 2) * c4 +
          4 * b2 * b3 * b4 * c0 * pow(c5, 2) -
          8 * b2 * b3 * b4 * pow(c0, 2) * c5 -
          16 * b2 * b3 * b4 * pow(c2, 2) * c3 -
          4 * b2 * b4 * b5 * c0 * pow(c3, 2) +
          8 * b2 * b4 * b5 * pow(c0, 2) * c3 +
          8 * b1 * b3 * b5 * pow(c1, 2) * c5 -
          16 * b2 * b3 * b4 * pow(c1, 2) * c5 +
          16 * b3 * b4 * b5 * pow(c1, 2) * c2 +
          4 * b2 * b4 * b5 * pow(c0, 2) * c5 +
          2 * b1 * b3 * b5 * c3 * pow(c5, 2) -
          4 * b2 * b3 * b4 * c3 * pow(c5, 2) +
          16 * b2 * b4 * b5 * pow(c1, 2) * c5 +
          4 * b2 * b4 * b5 * pow(c3, 2) * c5 -
          8 * b0 * pow(b1, 2) * c0 * c1 * c3 +
          8 * b0 * pow(b1, 2) * c0 * c1 * c5 +
          2 * pow(b0, 2) * b3 * c0 * c1 * c3 +
          16 * b0 * pow(b2, 2) * c0 * c2 * c4 -
          8 * b2 * pow(b3, 2) * c0 * c1 * c2 +
          2 * b0 * pow(b3, 2) * c0 * c1 * c5 -
          4 * b0 * pow(b3, 2) * c0 * c2 * c4 +
          4 * b0 * pow(b5, 2) * c0 * c1 * c3 -
          2 * pow(b0, 2) * b1 * c0 * c3 * c5 +
          4 * pow(b0, 2) * b2 * c0 * c3 * c4 -
          2 * pow(b0, 2) * b3 * c0 * c1 * c5 +
          4 * pow(b0, 2) * b3 * c0 * c2 * c4 -
          8 * pow(b0, 2) * b4 * c0 * c1 * c4 +
          4 * pow(b0, 2) * b4 * c0 * c2 * c3 -
          2 * pow(b0, 2) * b5 * c0 * c1 * c3 -
          32 * pow(b1, 2) * b2 * c1 * c2 * c3 +
          8 * pow(b2, 2) * b3 * c0 * c1 * c3 +
          8 * b0 * pow(b1, 2) * c1 * c3 * c5 -
          16 * b0 * pow(b1, 2) * c2 * c3 * c4 -
          8 * b0 * pow(b4, 2) * c0 * c1 * c5 +
          64 * b1 * pow(b2, 2) * c1 * c2 * c4 +
          16 * pow(b1, 2) * b2 * c0 * c3 * c4 +
          8 * pow(b1, 2) * b3 * c0 * c1 * c5 +
          8 * pow(b1, 2) * b5 * c0 * c1 * c3 +
          8 * b0 * pow(b2, 2) * c1 * c3 * c5 -
          16 * b0 * pow(b2, 2) * c2 * c3 * c4 -
          2 * b0 * pow(b5, 2) * c0 * c1 * c5 +
          4 * b0 * pow(b5, 2) * c0 * c2 * c4 -
          4 * pow(b0, 2) * b2 * c0 * c4 * c5 -
          4 * pow(b0, 2) * b4 * c0 * c2 * c5 +
          2 * pow(b0, 2) * b5 * c0 * c1 * c5 -
          4 * pow(b0, 2) * b5 * c0 * c2 * c4 +
          32 * pow(b1, 2) * b2 * c1 * c2 * c5 -
          16 * pow(b2, 2) * b3 * c0 * c2 * c4 -
          16 * pow(b2, 2) * b4 * c0 * c2 * c3 -
          8 * pow(b2, 2) * b5 * c0 * c1 * c3 +
          16 * b0 * pow(b1, 2) * c2 * c4 * c5 +
          2 * b3 * pow(b5, 2) * c0 * c1 * c3 -
          2 * pow(b0, 2) * b3 * c1 * c3 * c5 +
          4 * pow(b0, 2) * b3 * c2 * c3 * c4 -
          16 * pow(b1, 2) * b4 * c0 * c2 * c5 -
          16 * pow(b1, 2) * b5 * c0 * c1 * c5 -
          8 * b1 * pow(b4, 2) * c0 * c3 * c5 +
          16 * b1 * pow(b5, 2) * c1 * c2 * c4 +
          8 * b2 * pow(b3, 2) * c1 * c2 * c5 +
          8 * b3 * pow(b4, 2) * c0 * c1 * c5 +
          4 * b0 * pow(b3, 2) * c2 * c4 * c5 +
          2 * b0 * pow(b5, 2) * c1 * c3 * c5 -
          4 * b0 * pow(b5, 2) * c2 * c3 * c4 +
          2 * b1 * pow(b5, 2) * c0 * c3 * c5 -
          4 * b2 * pow(b3, 2) * c0 * c4 * c5 +
          4 * b2 * pow(b5, 2) * c0 * c3 * c4 +
          2 * b3 * pow(b5, 2) * c0 * c1 * c5 -
          4 * b3 * pow(b5, 2) * c0 * c2 * c4 +
          4 * b4 * pow(b5, 2) * c0 * c2 * c3 -
          8 * pow(b0, 2) * b3 * c2 * c4 * c5 +
          8 * pow(b0, 2) * b4 * c1 * c4 * c5 -
          4 * pow(b0, 2) * b5 * c1 * c3 * c5 +
          8 * pow(b0, 2) * b5 * c2 * c3 * c4 -
          8 * pow(b2, 2) * b3 * c1 * c3 * c5 +
          16 * pow(b2, 2) * b3 * c2 * c3 * c4 -
          4 * pow(b3, 2) * b4 * c0 * c2 * c5 -
          2 * pow(b3, 2) * b5 * c0 * c1 * c5 +
          4 * pow(b3, 2) * b5 * c0 * c2 * c4 -
          16 * pow(b1, 2) * b2 * c3 * c4 * c5 -
          8 * pow(b1, 2) * b5 * c1 * c3 * c5 +
          16 * pow(b1, 2) * b5 * c2 * c3 * c4 -
          4 * pow(b0, 2) * b5 * c2 * c4 * c5 -
          2 * b3 * pow(b5, 2) * c1 * c3 * c5 +
          4 * b3 * pow(b5, 2) * c2 * c3 * c4 -
          16 * pow(b1, 2) * b5 * c2 * c4 * c5 -
          4 * pow(b3, 2) * b5 * c2 * c4 * c5 +
          16 * b0 * b1 * b2 * c0 * c1 * c4 - 8 * b0 * b1 * b2 * c0 * c2 * c3 -
          16 * b0 * b1 * b4 * c0 * c1 * c2 + 8 * b0 * b2 * b3 * c0 * c1 * c2 +
          8 * b0 * b1 * b2 * c0 * c2 * c5 - 8 * b0 * b2 * b5 * c0 * c1 * c2 +
          16 * b0 * b1 * b3 * c1 * c2 * c4 + 16 * b0 * b1 * b4 * c1 * c2 * c3 -
          16 * b1 * b2 * b3 * c0 * c1 * c4 - 16 * b1 * b2 * b4 * c0 * c1 * c3 +
          4 * b0 * b1 * b3 * c0 * c3 * c5 - 8 * b0 * b1 * b4 * c0 * c3 * c4 +
          8 * b0 * b3 * b4 * c0 * c1 * c4 - 4 * b0 * b3 * b5 * c0 * c1 * c3 -
          16 * b0 * b1 * b2 * c1 * c4 * c5 - 8 * b0 * b1 * b2 * c2 * c3 * c5 -
          16 * b0 * b1 * b5 * c1 * c2 * c4 - 8 * b0 * b2 * b3 * c1 * c2 * c5 -
          32 * b0 * b2 * b4 * c1 * c2 * c4 + 16 * b1 * b2 * b4 * c0 * c1 * c5 +
          32 * b1 * b2 * b4 * c0 * c2 * c4 + 8 * b1 * b2 * b5 * c0 * c2 * c3 +
          16 * b1 * b4 * b5 * c0 * c1 * c2 + 8 * b2 * b3 * b5 * c0 * c1 * c2 -
          8 * b0 * b1 * b4 * c0 * c4 * c5 + 8 * b0 * b2 * b3 * c0 * c4 * c5 -
          8 * b0 * b2 * b5 * c0 * c3 * c4 + 8 * b0 * b3 * b4 * c0 * c2 * c5 +
          8 * b0 * b4 * b5 * c0 * c1 * c4 - 8 * b0 * b4 * b5 * c0 * c2 * c3 +
          8 * b0 * b2 * b5 * c1 * c2 * c5 - 8 * b1 * b2 * b5 * c0 * c2 * c5 +
          16 * b1 * b2 * b3 * c1 * c4 * c5 + 16 * b1 * b2 * b4 * c1 * c3 * c5 -
          32 * b1 * b2 * b4 * c2 * c3 * c4 - 16 * b1 * b3 * b5 * c1 * c2 * c4 -
          16 * b1 * b4 * b5 * c1 * c2 * c3 + 32 * b2 * b3 * b4 * c1 * c2 * c4 +
          8 * b0 * b1 * b4 * c3 * c4 * c5 - 8 * b0 * b3 * b4 * c1 * c4 * c5 +
          4 * b0 * b3 * b5 * c1 * c3 * c5 - 8 * b0 * b3 * b5 * c2 * c3 * c4 -
          4 * b1 * b3 * b5 * c0 * c3 * c5 + 8 * b1 * b4 * b5 * c0 * c3 * c4 +
          8 * b2 * b3 * b4 * c0 * c3 * c5 - 8 * b3 * b4 * b5 * c0 * c1 * c4 +
          8 * b1 * b2 * b5 * c2 * c3 * c5 - 8 * b2 * b3 * b5 * c1 * c2 * c5 +
          8 * b0 * b3 * b5 * c2 * c4 * c5 - 8 * b0 * b4 * b5 * c1 * c4 * c5 +
          8 * b1 * b4 * b5 * c0 * c4 * c5 - 8 * b2 * b4 * b5 * c0 * c3 * c5 -
          8 * b1 * b4 * b5 * c3 * c4 * c5 + 8 * b3 * b4 * b5 * c1 * c4 * c5,
      a1 * pow(b3, 2) * pow(d0, 3) - 4 * a3 * pow(b0, 2) * pow(d1, 3) -
          4 * a1 * pow(b4, 2) * pow(d0, 3) - 8 * a0 * pow(b5, 2) * pow(d1, 3) -
          a1 * pow(b0, 2) * pow(d5, 3) + a1 * pow(b5, 2) * pow(d0, 3) -
          16 * a3 * pow(b2, 2) * pow(d1, 3) + 8 * a4 * pow(b0, 2) * pow(d2, 3) +
          4 * a5 * pow(b0, 2) * pow(d1, 3) - 12 * a1 * pow(b1, 2) * pow(d5, 3) +
          32 * a4 * pow(b1, 2) * pow(d2, 3) +
          16 * a5 * pow(b2, 2) * pow(d1, 3) - a1 * pow(b3, 2) * pow(d5, 3) -
          4 * a3 * pow(b5, 2) * pow(d1, 3) + 8 * a4 * pow(b3, 2) * pow(d2, 3) +
          12 * a5 * pow(b5, 2) * pow(d1, 3) - 8 * a0 * b0 * b3 * pow(d1, 3) -
          2 * a0 * b0 * b1 * pow(d5, 3) + 16 * a0 * b0 * b4 * pow(d2, 3) +
          8 * a0 * b0 * b5 * pow(d1, 3) + 2 * a3 * b1 * b3 * pow(d0, 3) +
          64 * a1 * b1 * b4 * pow(d2, 3) - 32 * a2 * b2 * b3 * pow(d1, 3) +
          2 * a0 * b1 * b3 * pow(d5, 3) - 16 * a0 * b3 * b4 * pow(d2, 3) +
          8 * a0 * b3 * b5 * pow(d1, 3) + 2 * a1 * b0 * b3 * pow(d5, 3) -
          2 * a1 * b3 * b5 * pow(d0, 3) + 4 * a2 * b3 * b4 * pow(d0, 3) +
          2 * a3 * b0 * b1 * pow(d5, 3) - 16 * a3 * b0 * b4 * pow(d2, 3) +
          8 * a3 * b0 * b5 * pow(d1, 3) - 2 * a3 * b1 * b5 * pow(d0, 3) +
          4 * a3 * b2 * b4 * pow(d0, 3) - 16 * a4 * b0 * b3 * pow(d2, 3) -
          8 * a4 * b1 * b4 * pow(d0, 3) + 4 * a4 * b2 * b3 * pow(d0, 3) +
          8 * a5 * b0 * b3 * pow(d1, 3) - 2 * a5 * b1 * b3 * pow(d0, 3) +
          32 * a2 * b2 * b5 * pow(d1, 3) - 4 * a2 * b4 * b5 * pow(d0, 3) -
          4 * a4 * b2 * b5 * pow(d0, 3) - 16 * a5 * b0 * b5 * pow(d1, 3) +
          2 * a5 * b1 * b5 * pow(d0, 3) - 4 * a5 * b2 * b4 * pow(d0, 3) -
          2 * a3 * b1 * b3 * pow(d5, 3) + 16 * a3 * b3 * b4 * pow(d2, 3) -
          8 * a5 * b3 * b5 * pow(d1, 3) -
          3 * a0 * pow(b0, 2) * d1 * pow(d3, 2) -
          a0 * pow(b3, 2) * pow(d0, 2) * d1 +
          a1 * pow(b0, 2) * d0 * pow(d3, 2) +
          12 * a0 * pow(b0, 2) * d1 * pow(d4, 2) +
          4 * a0 * pow(b4, 2) * pow(d0, 2) * d1 -
          4 * a1 * pow(b0, 2) * d0 * pow(d4, 2) +
          4 * a1 * pow(b0, 2) * pow(d1, 2) * d3 +
          12 * a1 * pow(b1, 2) * pow(d0, 2) * d3 -
          4 * a3 * pow(b1, 2) * pow(d0, 2) * d1 -
          3 * a0 * pow(b0, 2) * d1 * pow(d5, 2) -
          4 * a0 * pow(b2, 2) * d1 * pow(d3, 2) +
          4 * a0 * pow(b3, 2) * d1 * pow(d2, 2) -
          a0 * pow(b5, 2) * pow(d0, 2) * d1 +
          a1 * pow(b0, 2) * d0 * pow(d5, 2) +
          4 * a1 * pow(b0, 2) * pow(d2, 2) * d3 -
          4 * a1 * pow(b2, 2) * d0 * pow(d3, 2) +
          4 * a1 * pow(b2, 2) * pow(d0, 2) * d3 +
          4 * a1 * pow(b3, 2) * d0 * pow(d2, 2) -
          4 * a3 * pow(b0, 2) * d1 * pow(d2, 2) -
          4 * a3 * pow(b2, 2) * pow(d0, 2) * d1 -
          8 * a0 * pow(b1, 2) * d1 * pow(d5, 2) +
          16 * a0 * pow(b2, 2) * d1 * pow(d4, 2) +
          16 * a0 * pow(b4, 2) * d1 * pow(d2, 2) -
          4 * a1 * pow(b0, 2) * pow(d1, 2) * d5 +
          24 * a1 * pow(b1, 2) * d0 * pow(d5, 2) -
          12 * a1 * pow(b1, 2) * pow(d0, 2) * d5 +
          48 * a1 * pow(b1, 2) * pow(d2, 2) * d3 -
          16 * a1 * pow(b2, 2) * d0 * pow(d4, 2) +
          16 * a1 * pow(b2, 2) * pow(d1, 2) * d3 -
          16 * a1 * pow(b4, 2) * d0 * pow(d2, 2) +
          8 * a1 * pow(b5, 2) * d0 * pow(d1, 2) -
          8 * a2 * pow(b0, 2) * pow(d1, 2) * d4 -
          8 * a2 * pow(b1, 2) * pow(d0, 2) * d4 -
          16 * a3 * pow(b1, 2) * d1 * pow(d2, 2) +
          8 * a4 * pow(b0, 2) * pow(d1, 2) * d2 +
          8 * a4 * pow(b1, 2) * pow(d0, 2) * d2 +
          4 * a5 * pow(b1, 2) * pow(d0, 2) * d1 -
          4 * a0 * pow(b2, 2) * d1 * pow(d5, 2) -
          4 * a0 * pow(b5, 2) * d1 * pow(d2, 2) +
          4 * a1 * pow(b0, 2) * d3 * pow(d4, 2) -
          4 * a1 * pow(b0, 2) * pow(d2, 2) * d5 +
          4 * a1 * pow(b2, 2) * d0 * pow(d5, 2) -
          4 * a1 * pow(b2, 2) * pow(d0, 2) * d5 +
          4 * a1 * pow(b4, 2) * pow(d0, 2) * d3 +
          4 * a1 * pow(b5, 2) * d0 * pow(d2, 2) -
          8 * a2 * pow(b0, 2) * pow(d2, 2) * d4 -
          24 * a2 * pow(b2, 2) * pow(d0, 2) * d4 -
          4 * a3 * pow(b0, 2) * d1 * pow(d4, 2) -
          4 * a3 * pow(b4, 2) * pow(d0, 2) * d1 +
          8 * a4 * pow(b2, 2) * pow(d0, 2) * d2 +
          4 * a5 * pow(b0, 2) * d1 * pow(d2, 2) +
          4 * a5 * pow(b2, 2) * pow(d0, 2) * d1 -
          a0 * pow(b3, 2) * d1 * pow(d5, 2) -
          3 * a0 * pow(b5, 2) * d1 * pow(d3, 2) +
          2 * a1 * pow(b0, 2) * d3 * pow(d5, 2) -
          a1 * pow(b0, 2) * pow(d3, 2) * d5 -
          48 * a1 * pow(b1, 2) * pow(d2, 2) * d5 -
          16 * a1 * pow(b2, 2) * pow(d1, 2) * d5 +
          3 * a1 * pow(b3, 2) * d0 * pow(d5, 2) -
          3 * a1 * pow(b3, 2) * pow(d0, 2) * d5 +
          a1 * pow(b5, 2) * d0 * pow(d3, 2) -
          2 * a1 * pow(b5, 2) * pow(d0, 2) * d3 -
          2 * a2 * pow(b0, 2) * pow(d3, 2) * d4 -
          32 * a2 * pow(b1, 2) * pow(d2, 2) * d4 -
          96 * a2 * pow(b2, 2) * pow(d1, 2) * d4 +
          2 * a2 * pow(b3, 2) * pow(d0, 2) * d4 +
          2 * a3 * pow(b0, 2) * d1 * pow(d5, 2) -
          2 * a3 * pow(b5, 2) * pow(d0, 2) * d1 -
          2 * a4 * pow(b0, 2) * d2 * pow(d3, 2) +
          32 * a4 * pow(b2, 2) * pow(d1, 2) * d2 +
          2 * a4 * pow(b3, 2) * pow(d0, 2) * d2 +
          3 * a5 * pow(b0, 2) * d1 * pow(d3, 2) +
          16 * a5 * pow(b1, 2) * d1 * pow(d2, 2) +
          a5 * pow(b3, 2) * pow(d0, 2) * d1 +
          4 * a0 * pow(b4, 2) * d1 * pow(d5, 2) +
          4 * a0 * pow(b5, 2) * d1 * pow(d4, 2) +
          12 * a1 * pow(b1, 2) * d3 * pow(d5, 2) +
          16 * a1 * pow(b2, 2) * d3 * pow(d4, 2) -
          4 * a1 * pow(b4, 2) * d0 * pow(d5, 2) +
          8 * a1 * pow(b4, 2) * pow(d0, 2) * d5 +
          16 * a1 * pow(b4, 2) * pow(d2, 2) * d3 -
          4 * a1 * pow(b5, 2) * d0 * pow(d4, 2) +
          4 * a1 * pow(b5, 2) * pow(d1, 2) * d3 -
          4 * a3 * pow(b1, 2) * d1 * pow(d5, 2) -
          16 * a3 * pow(b2, 2) * d1 * pow(d4, 2) -
          16 * a3 * pow(b4, 2) * d1 * pow(d2, 2) -
          8 * a5 * pow(b0, 2) * d1 * pow(d4, 2) -
          4 * a1 * pow(b2, 2) * d3 * pow(d5, 2) +
          4 * a1 * pow(b2, 2) * pow(d3, 2) * d5 -
          4 * a1 * pow(b3, 2) * pow(d2, 2) * d5 -
          a1 * pow(b5, 2) * pow(d0, 2) * d5 -
          4 * a1 * pow(b5, 2) * pow(d2, 2) * d3 +
          2 * a2 * pow(b0, 2) * d4 * pow(d5, 2) -
          24 * a2 * pow(b2, 2) * pow(d3, 2) * d4 -
          8 * a2 * pow(b3, 2) * pow(d2, 2) * d4 -
          2 * a2 * pow(b5, 2) * pow(d0, 2) * d4 +
          4 * a3 * pow(b2, 2) * d1 * pow(d5, 2) +
          4 * a3 * pow(b5, 2) * d1 * pow(d2, 2) +
          2 * a4 * pow(b0, 2) * d2 * pow(d5, 2) +
          8 * a4 * pow(b2, 2) * d2 * pow(d3, 2) -
          2 * a4 * pow(b5, 2) * pow(d0, 2) * d2 +
          a5 * pow(b0, 2) * d1 * pow(d5, 2) +
          4 * a5 * pow(b2, 2) * d1 * pow(d3, 2) -
          4 * a5 * pow(b3, 2) * d1 * pow(d2, 2) +
          3 * a5 * pow(b5, 2) * pow(d0, 2) * d1 -
          12 * a1 * pow(b5, 2) * pow(d1, 2) * d5 +
          8 * a2 * pow(b1, 2) * d4 * pow(d5, 2) -
          8 * a2 * pow(b5, 2) * pow(d1, 2) * d4 +
          8 * a4 * pow(b1, 2) * d2 * pow(d5, 2) -
          8 * a4 * pow(b5, 2) * pow(d1, 2) * d2 +
          12 * a5 * pow(b1, 2) * d1 * pow(d5, 2) +
          4 * a1 * pow(b4, 2) * d3 * pow(d5, 2) +
          4 * a1 * pow(b5, 2) * d3 * pow(d4, 2) -
          4 * a3 * pow(b4, 2) * d1 * pow(d5, 2) -
          4 * a3 * pow(b5, 2) * d1 * pow(d4, 2) -
          a1 * pow(b5, 2) * pow(d3, 2) * d5 +
          2 * a2 * pow(b3, 2) * d4 * pow(d5, 2) -
          2 * a2 * pow(b5, 2) * pow(d3, 2) * d4 +
          2 * a4 * pow(b3, 2) * d2 * pow(d5, 2) -
          2 * a4 * pow(b5, 2) * d2 * pow(d3, 2) +
          a5 * pow(b3, 2) * d1 * pow(d5, 2) +
          3 * a5 * pow(b5, 2) * d1 * pow(d3, 2) +
          2 * a0 * b0 * b1 * d0 * pow(d3, 2) -
          8 * a0 * b0 * b1 * d0 * pow(d4, 2) +
          8 * a0 * b0 * b1 * pow(d1, 2) * d3 +
          8 * a0 * b1 * b3 * d0 * pow(d1, 2) +
          8 * a1 * b0 * b3 * d0 * pow(d1, 2) +
          8 * a3 * b0 * b1 * d0 * pow(d1, 2) +
          2 * a0 * b0 * b1 * d0 * pow(d5, 2) +
          8 * a0 * b0 * b1 * pow(d2, 2) * d3 -
          8 * a0 * b0 * b3 * d1 * pow(d2, 2) -
          8 * a1 * b1 * b3 * pow(d0, 2) * d1 -
          8 * a0 * b0 * b1 * pow(d1, 2) * d5 -
          16 * a0 * b0 * b2 * pow(d1, 2) * d4 +
          16 * a0 * b0 * b4 * pow(d1, 2) * d2 -
          2 * a0 * b1 * b3 * pow(d0, 2) * d3 -
          8 * a0 * b1 * b5 * d0 * pow(d1, 2) -
          2 * a1 * b0 * b3 * pow(d0, 2) * d3 -
          8 * a1 * b0 * b5 * d0 * pow(d1, 2) -
          2 * a3 * b0 * b1 * pow(d0, 2) * d3 -
          2 * a3 * b0 * b3 * pow(d0, 2) * d1 -
          8 * a5 * b0 * b1 * d0 * pow(d1, 2) +
          8 * a0 * b0 * b1 * d3 * pow(d4, 2) -
          8 * a0 * b0 * b1 * pow(d2, 2) * d5 -
          16 * a0 * b0 * b2 * pow(d2, 2) * d4 -
          8 * a0 * b0 * b3 * d1 * pow(d4, 2) +
          8 * a0 * b0 * b5 * d1 * pow(d2, 2) +
          8 * a0 * b1 * b2 * d2 * pow(d3, 2) -
          16 * a0 * b2 * b4 * d0 * pow(d2, 2) -
          16 * a1 * b0 * b1 * d1 * pow(d5, 2) +
          8 * a1 * b0 * b2 * d2 * pow(d3, 2) -
          16 * a1 * b1 * b2 * pow(d0, 2) * d4 -
          32 * a1 * b1 * b3 * d1 * pow(d2, 2) +
          16 * a1 * b1 * b4 * pow(d0, 2) * d2 +
          8 * a1 * b1 * b5 * pow(d0, 2) * d1 +
          8 * a2 * b0 * b1 * d2 * pow(d3, 2) -
          8 * a2 * b0 * b2 * d1 * pow(d3, 2) -
          16 * a2 * b0 * b4 * d0 * pow(d2, 2) -
          8 * a2 * b1 * b2 * d0 * pow(d3, 2) +
          8 * a2 * b1 * b2 * pow(d0, 2) * d3 -
          8 * a2 * b2 * b3 * pow(d0, 2) * d1 -
          16 * a4 * b0 * b2 * d0 * pow(d2, 2) +
          4 * a0 * b0 * b1 * d3 * pow(d5, 2) -
          2 * a0 * b0 * b1 * pow(d3, 2) * d5 -
          4 * a0 * b0 * b2 * pow(d3, 2) * d4 +
          4 * a0 * b0 * b3 * d1 * pow(d5, 2) -
          4 * a0 * b0 * b4 * d2 * pow(d3, 2) +
          6 * a0 * b0 * b5 * d1 * pow(d3, 2) -
          4 * a0 * b1 * b3 * d0 * pow(d5, 2) +
          2 * a0 * b1 * b3 * pow(d0, 2) * d5 -
          8 * a0 * b1 * b3 * pow(d2, 2) * d3 +
          8 * a0 * b1 * b4 * pow(d0, 2) * d4 -
          2 * a0 * b1 * b5 * d0 * pow(d3, 2) +
          2 * a0 * b1 * b5 * pow(d0, 2) * d3 -
          4 * a0 * b2 * b3 * pow(d0, 2) * d4 +
          4 * a0 * b2 * b4 * d0 * pow(d3, 2) -
          4 * a0 * b2 * b4 * pow(d0, 2) * d3 -
          4 * a0 * b3 * b4 * pow(d0, 2) * d2 +
          2 * a0 * b3 * b5 * pow(d0, 2) * d1 -
          4 * a1 * b0 * b3 * d0 * pow(d5, 2) +
          2 * a1 * b0 * b3 * pow(d0, 2) * d5 -
          8 * a1 * b0 * b3 * pow(d2, 2) * d3 +
          8 * a1 * b0 * b4 * pow(d0, 2) * d4 -
          2 * a1 * b0 * b5 * d0 * pow(d3, 2) +
          2 * a1 * b0 * b5 * pow(d0, 2) * d3 +
          32 * a1 * b2 * b3 * pow(d1, 2) * d2 +
          32 * a2 * b0 * b2 * d1 * pow(d4, 2) -
          4 * a2 * b0 * b3 * pow(d0, 2) * d4 +
          4 * a2 * b0 * b4 * d0 * pow(d3, 2) -
          4 * a2 * b0 * b4 * pow(d0, 2) * d3 -
          32 * a2 * b1 * b2 * d0 * pow(d4, 2) +
          32 * a2 * b1 * b2 * pow(d1, 2) * d3 +
          32 * a2 * b1 * b3 * pow(d1, 2) * d2 -
          4 * a3 * b0 * b1 * d0 * pow(d5, 2) +
          2 * a3 * b0 * b1 * pow(d0, 2) * d5 -
          8 * a3 * b0 * b1 * pow(d2, 2) * d3 -
          4 * a3 * b0 * b2 * pow(d0, 2) * d4 +
          8 * a3 * b0 * b3 * d1 * pow(d2, 2) -
          4 * a3 * b0 * b4 * pow(d0, 2) * d2 +
          2 * a3 * b0 * b5 * pow(d0, 2) * d1 +
          32 * a3 * b1 * b2 * pow(d1, 2) * d2 +
          8 * a3 * b1 * b3 * d0 * pow(d2, 2) +
          8 * a4 * b0 * b1 * pow(d0, 2) * d4 +
          4 * a4 * b0 * b2 * d0 * pow(d3, 2) -
          4 * a4 * b0 * b2 * pow(d0, 2) * d3 -
          4 * a4 * b0 * b3 * pow(d0, 2) * d2 +
          8 * a4 * b0 * b4 * pow(d0, 2) * d1 -
          2 * a5 * b0 * b1 * d0 * pow(d3, 2) +
          2 * a5 * b0 * b1 * pow(d0, 2) * d3 +
          2 * a5 * b0 * b3 * pow(d0, 2) * d1 -
          16 * a0 * b0 * b5 * d1 * pow(d4, 2) -
          8 * a0 * b1 * b3 * pow(d1, 2) * d5 +
          8 * a0 * b1 * b5 * d0 * pow(d4, 2) -
          8 * a0 * b1 * b5 * pow(d1, 2) * d3 -
          16 * a0 * b3 * b4 * pow(d1, 2) * d2 -
          8 * a1 * b0 * b3 * pow(d1, 2) * d5 +
          8 * a1 * b0 * b5 * d0 * pow(d4, 2) -
          8 * a1 * b0 * b5 * pow(d1, 2) * d3 -
          64 * a1 * b1 * b2 * pow(d2, 2) * d4 +
          32 * a1 * b1 * b5 * d1 * pow(d2, 2) -
          64 * a1 * b2 * b4 * d1 * pow(d2, 2) -
          8 * a1 * b3 * b5 * d0 * pow(d1, 2) -
          8 * a2 * b0 * b2 * d1 * pow(d5, 2) +
          8 * a2 * b1 * b2 * d0 * pow(d5, 2) -
          8 * a2 * b1 * b2 * pow(d0, 2) * d5 -
          64 * a2 * b1 * b4 * d1 * pow(d2, 2) +
          16 * a2 * b2 * b4 * pow(d0, 2) * d2 +
          8 * a2 * b2 * b5 * pow(d0, 2) * d1 +
          16 * a2 * b3 * b4 * d0 * pow(d1, 2) -
          8 * a3 * b0 * b1 * pow(d1, 2) * d5 -
          16 * a3 * b0 * b4 * pow(d1, 2) * d2 -
          8 * a3 * b1 * b5 * d0 * pow(d1, 2) +
          16 * a3 * b2 * b4 * d0 * pow(d1, 2) -
          16 * a4 * b0 * b3 * pow(d1, 2) * d2 -
          64 * a4 * b1 * b2 * d1 * pow(d2, 2) +
          16 * a4 * b2 * b3 * d0 * pow(d1, 2) +
          8 * a5 * b0 * b1 * d0 * pow(d4, 2) -
          8 * a5 * b0 * b1 * pow(d1, 2) * d3 -
          8 * a5 * b1 * b3 * d0 * pow(d1, 2) +
          4 * a0 * b0 * b2 * d4 * pow(d5, 2) +
          4 * a0 * b0 * b4 * d2 * pow(d5, 2) +
          2 * a0 * b0 * b5 * d1 * pow(d5, 2) +
          8 * a0 * b1 * b3 * pow(d2, 2) * d5 +
          2 * a0 * b1 * b5 * d0 * pow(d5, 2) -
          2 * a0 * b1 * b5 * pow(d0, 2) * d5 +
          16 * a0 * b2 * b3 * pow(d2, 2) * d4 -
          4 * a0 * b2 * b4 * d0 * pow(d5, 2) +
          4 * a0 * b2 * b4 * pow(d0, 2) * d5 +
          16 * a0 * b2 * b4 * pow(d2, 2) * d3 +
          4 * a0 * b2 * b5 * pow(d0, 2) * d4 +
          4 * a0 * b4 * b5 * pow(d0, 2) * d2 +
          8 * a1 * b0 * b3 * pow(d2, 2) * d5 +
          2 * a1 * b0 * b5 * d0 * pow(d5, 2) -
          2 * a1 * b0 * b5 * pow(d0, 2) * d5 -
          8 * a1 * b1 * b3 * d1 * pow(d5, 2) -
          32 * a1 * b2 * b5 * pow(d1, 2) * d2 -
          8 * a1 * b3 * b5 * d0 * pow(d2, 2) +
          16 * a2 * b0 * b3 * pow(d2, 2) * d4 -
          4 * a2 * b0 * b4 * d0 * pow(d5, 2) +
          4 * a2 * b0 * b4 * pow(d0, 2) * d5 +
          16 * a2 * b0 * b4 * pow(d2, 2) * d3 +
          4 * a2 * b0 * b5 * pow(d0, 2) * d4 -
          32 * a2 * b1 * b2 * pow(d1, 2) * d5 -
          32 * a2 * b1 * b5 * pow(d1, 2) * d2 +
          64 * a2 * b2 * b4 * pow(d1, 2) * d2 +
          16 * a2 * b3 * b4 * d0 * pow(d2, 2) +
          8 * a3 * b0 * b1 * pow(d2, 2) * d5 +
          16 * a3 * b0 * b2 * pow(d2, 2) * d4 -
          8 * a3 * b1 * b5 * d0 * pow(d2, 2) +
          16 * a3 * b2 * b4 * d0 * pow(d2, 2) -
          4 * a4 * b0 * b2 * d0 * pow(d5, 2) +
          4 * a4 * b0 * b2 * pow(d0, 2) * d5 +
          16 * a4 * b0 * b2 * pow(d2, 2) * d3 +
          32 * a4 * b0 * b4 * d1 * pow(d2, 2) +
          4 * a4 * b0 * b5 * pow(d0, 2) * d2 -
          32 * a4 * b1 * b4 * d0 * pow(d2, 2) +
          16 * a4 * b2 * b3 * d0 * pow(d2, 2) +
          2 * a5 * b0 * b1 * d0 * pow(d5, 2) -
          2 * a5 * b0 * b1 * pow(d0, 2) * d5 +
          4 * a5 * b0 * b2 * pow(d0, 2) * d4 +
          4 * a5 * b0 * b4 * pow(d0, 2) * d2 -
          2 * a5 * b0 * b5 * pow(d0, 2) * d1 -
          32 * a5 * b1 * b2 * pow(d1, 2) * d2 -
          8 * a5 * b1 * b3 * d0 * pow(d2, 2) -
          2 * a0 * b1 * b3 * d3 * pow(d5, 2) +
          16 * a0 * b1 * b5 * pow(d1, 2) * d5 +
          16 * a0 * b2 * b5 * pow(d1, 2) * d4 -
          2 * a1 * b0 * b3 * d3 * pow(d5, 2) +
          16 * a1 * b0 * b5 * pow(d1, 2) * d5 +
          2 * a1 * b3 * b5 * pow(d0, 2) * d3 +
          16 * a2 * b0 * b5 * pow(d1, 2) * d4 +
          32 * a2 * b1 * b2 * d3 * pow(d4, 2) -
          32 * a2 * b2 * b3 * d1 * pow(d4, 2) -
          4 * a2 * b3 * b4 * pow(d0, 2) * d3 -
          16 * a2 * b4 * b5 * d0 * pow(d1, 2) -
          2 * a3 * b0 * b1 * d3 * pow(d5, 2) -
          2 * a3 * b0 * b3 * d1 * pow(d5, 2) +
          6 * a3 * b1 * b3 * d0 * pow(d5, 2) -
          6 * a3 * b1 * b3 * pow(d0, 2) * d5 +
          2 * a3 * b1 * b5 * pow(d0, 2) * d3 +
          4 * a3 * b2 * b3 * pow(d0, 2) * d4 -
          4 * a3 * b2 * b4 * pow(d0, 2) * d3 +
          4 * a3 * b3 * b4 * pow(d0, 2) * d2 +
          2 * a3 * b3 * b5 * pow(d0, 2) * d1 +
          8 * a4 * b1 * b4 * pow(d0, 2) * d3 -
          4 * a4 * b2 * b3 * pow(d0, 2) * d3 -
          16 * a4 * b2 * b5 * d0 * pow(d1, 2) -
          8 * a4 * b3 * b4 * pow(d0, 2) * d1 +
          16 * a5 * b0 * b1 * pow(d1, 2) * d5 +
          16 * a5 * b0 * b2 * pow(d1, 2) * d4 +
          2 * a5 * b1 * b3 * pow(d0, 2) * d3 +
          16 * a5 * b1 * b5 * d0 * pow(d1, 2) -
          16 * a5 * b2 * b4 * d0 * pow(d1, 2) -
          8 * a0 * b1 * b5 * d3 * pow(d4, 2) +
          8 * a0 * b3 * b5 * d1 * pow(d4, 2) -
          8 * a1 * b0 * b5 * d3 * pow(d4, 2) +
          16 * a1 * b1 * b2 * d4 * pow(d5, 2) +
          16 * a1 * b1 * b4 * d2 * pow(d5, 2) +
          24 * a1 * b1 * b5 * d1 * pow(d5, 2) -
          16 * a1 * b2 * b4 * d1 * pow(d5, 2) -
          8 * a1 * b2 * b5 * d2 * pow(d3, 2) -
          8 * a2 * b1 * b2 * d3 * pow(d5, 2) +
          8 * a2 * b1 * b2 * pow(d3, 2) * d5 -
          16 * a2 * b1 * b4 * d1 * pow(d5, 2) -
          8 * a2 * b1 * b5 * d2 * pow(d3, 2) +
          8 * a2 * b2 * b3 * d1 * pow(d5, 2) +
          16 * a2 * b2 * b4 * d2 * pow(d3, 2) +
          8 * a2 * b2 * b5 * d1 * pow(d3, 2) +
          8 * a3 * b0 * b5 * d1 * pow(d4, 2) -
          16 * a4 * b1 * b2 * d1 * pow(d5, 2) -
          8 * a5 * b0 * b1 * d3 * pow(d4, 2) +
          8 * a5 * b0 * b3 * d1 * pow(d4, 2) -
          8 * a5 * b0 * b5 * d1 * pow(d2, 2) -
          8 * a5 * b1 * b2 * d2 * pow(d3, 2) +
          8 * a5 * b1 * b5 * d0 * pow(d2, 2) -
          2 * a0 * b1 * b5 * d3 * pow(d5, 2) +
          2 * a0 * b1 * b5 * pow(d3, 2) * d5 -
          4 * a0 * b2 * b3 * d4 * pow(d5, 2) +
          4 * a0 * b2 * b4 * d3 * pow(d5, 2) -
          4 * a0 * b2 * b4 * pow(d3, 2) * d5 +
          4 * a0 * b2 * b5 * pow(d3, 2) * d4 -
          4 * a0 * b3 * b4 * d2 * pow(d5, 2) -
          2 * a0 * b3 * b5 * d1 * pow(d5, 2) +
          4 * a0 * b4 * b5 * d2 * pow(d3, 2) -
          2 * a1 * b0 * b5 * d3 * pow(d5, 2) +
          2 * a1 * b0 * b5 * pow(d3, 2) * d5 -
          2 * a1 * b3 * b5 * d0 * pow(d5, 2) +
          4 * a1 * b3 * b5 * pow(d0, 2) * d5 +
          8 * a1 * b3 * b5 * pow(d2, 2) * d3 -
          8 * a1 * b4 * b5 * pow(d0, 2) * d4 -
          4 * a2 * b0 * b3 * d4 * pow(d5, 2) +
          4 * a2 * b0 * b4 * d3 * pow(d5, 2) -
          4 * a2 * b0 * b4 * pow(d3, 2) * d5 +
          4 * a2 * b0 * b5 * pow(d3, 2) * d4 +
          4 * a2 * b3 * b4 * d0 * pow(d5, 2) -
          8 * a2 * b3 * b4 * pow(d0, 2) * d5 -
          16 * a2 * b3 * b4 * pow(d2, 2) * d3 -
          4 * a2 * b4 * b5 * d0 * pow(d3, 2) +
          8 * a2 * b4 * b5 * pow(d0, 2) * d3 -
          4 * a3 * b0 * b2 * d4 * pow(d5, 2) -
          4 * a3 * b0 * b4 * d2 * pow(d5, 2) -
          2 * a3 * b0 * b5 * d1 * pow(d5, 2) -
          8 * a3 * b1 * b3 * pow(d2, 2) * d5 -
          2 * a3 * b1 * b5 * d0 * pow(d5, 2) +
          4 * a3 * b1 * b5 * pow(d0, 2) * d5 +
          8 * a3 * b1 * b5 * pow(d2, 2) * d3 -
          16 * a3 * b2 * b3 * pow(d2, 2) * d4 +
          4 * a3 * b2 * b4 * d0 * pow(d5, 2) -
          8 * a3 * b2 * b4 * pow(d0, 2) * d5 -
          16 * a3 * b2 * b4 * pow(d2, 2) * d3 -
          8 * a3 * b3 * b5 * d1 * pow(d2, 2) +
          4 * a4 * b0 * b2 * d3 * pow(d5, 2) -
          4 * a4 * b0 * b2 * pow(d3, 2) * d5 -
          4 * a4 * b0 * b3 * d2 * pow(d5, 2) +
          8 * a4 * b0 * b4 * d1 * pow(d5, 2) +
          4 * a4 * b0 * b5 * d2 * pow(d3, 2) -
          8 * a4 * b1 * b4 * d0 * pow(d5, 2) +
          16 * a4 * b1 * b4 * pow(d0, 2) * d5 +
          32 * a4 * b1 * b4 * pow(d2, 2) * d3 -
          8 * a4 * b1 * b5 * pow(d0, 2) * d4 +
          4 * a4 * b2 * b3 * d0 * pow(d5, 2) -
          8 * a4 * b2 * b3 * pow(d0, 2) * d5 -
          16 * a4 * b2 * b3 * pow(d2, 2) * d3 -
          4 * a4 * b2 * b5 * d0 * pow(d3, 2) +
          8 * a4 * b2 * b5 * pow(d0, 2) * d3 -
          32 * a4 * b3 * b4 * d1 * pow(d2, 2) -
          2 * a5 * b0 * b1 * d3 * pow(d5, 2) +
          2 * a5 * b0 * b1 * pow(d3, 2) * d5 +
          4 * a5 * b0 * b2 * pow(d3, 2) * d4 -
          2 * a5 * b0 * b3 * d1 * pow(d5, 2) +
          4 * a5 * b0 * b4 * d2 * pow(d3, 2) -
          6 * a5 * b0 * b5 * d1 * pow(d3, 2) -
          2 * a5 * b1 * b3 * d0 * pow(d5, 2) +
          4 * a5 * b1 * b3 * pow(d0, 2) * d5 +
          8 * a5 * b1 * b3 * pow(d2, 2) * d3 -
          8 * a5 * b1 * b4 * pow(d0, 2) * d4 +
          2 * a5 * b1 * b5 * d0 * pow(d3, 2) -
          4 * a5 * b1 * b5 * pow(d0, 2) * d3 -
          4 * a5 * b2 * b4 * d0 * pow(d3, 2) +
          8 * a5 * b2 * b4 * pow(d0, 2) * d3 -
          4 * a5 * b3 * b5 * pow(d0, 2) * d1 +
          8 * a1 * b3 * b5 * pow(d1, 2) * d5 -
          16 * a2 * b3 * b4 * pow(d1, 2) * d5 +
          8 * a3 * b1 * b5 * pow(d1, 2) * d5 -
          16 * a3 * b2 * b4 * pow(d1, 2) * d5 +
          16 * a3 * b4 * b5 * pow(d1, 2) * d2 -
          16 * a4 * b2 * b3 * pow(d1, 2) * d5 +
          16 * a4 * b3 * b5 * pow(d1, 2) * d2 +
          8 * a5 * b0 * b5 * d1 * pow(d4, 2) +
          8 * a5 * b1 * b3 * pow(d1, 2) * d5 -
          8 * a5 * b1 * b5 * d0 * pow(d4, 2) +
          8 * a5 * b1 * b5 * pow(d1, 2) * d3 +
          16 * a5 * b3 * b4 * pow(d1, 2) * d2 +
          4 * a2 * b4 * b5 * pow(d0, 2) * d5 +
          4 * a4 * b2 * b5 * pow(d0, 2) * d5 -
          2 * a5 * b1 * b5 * pow(d0, 2) * d5 -
          8 * a5 * b1 * b5 * pow(d2, 2) * d3 +
          4 * a5 * b2 * b4 * pow(d0, 2) * d5 -
          4 * a5 * b2 * b5 * pow(d0, 2) * d4 +
          8 * a5 * b3 * b5 * d1 * pow(d2, 2) -
          4 * a5 * b4 * b5 * pow(d0, 2) * d2 +
          2 * a1 * b3 * b5 * d3 * pow(d5, 2) -
          4 * a2 * b3 * b4 * d3 * pow(d5, 2) +
          16 * a2 * b4 * b5 * pow(d1, 2) * d5 +
          2 * a3 * b1 * b5 * d3 * pow(d5, 2) +
          4 * a3 * b2 * b3 * d4 * pow(d5, 2) -
          4 * a3 * b2 * b4 * d3 * pow(d5, 2) +
          4 * a3 * b3 * b4 * d2 * pow(d5, 2) +
          2 * a3 * b3 * b5 * d1 * pow(d5, 2) +
          8 * a4 * b1 * b4 * d3 * pow(d5, 2) -
          4 * a4 * b2 * b3 * d3 * pow(d5, 2) +
          16 * a4 * b2 * b5 * pow(d1, 2) * d5 -
          8 * a4 * b3 * b4 * d1 * pow(d5, 2) +
          2 * a5 * b1 * b3 * d3 * pow(d5, 2) -
          24 * a5 * b1 * b5 * pow(d1, 2) * d5 +
          16 * a5 * b2 * b4 * pow(d1, 2) * d5 -
          16 * a5 * b2 * b5 * pow(d1, 2) * d4 -
          16 * a5 * b4 * b5 * pow(d1, 2) * d2 +
          8 * a5 * b1 * b5 * d3 * pow(d4, 2) -
          8 * a5 * b3 * b5 * d1 * pow(d4, 2) +
          4 * a2 * b4 * b5 * pow(d3, 2) * d5 +
          4 * a4 * b2 * b5 * pow(d3, 2) * d5 -
          2 * a5 * b1 * b5 * pow(d3, 2) * d5 +
          4 * a5 * b2 * b4 * pow(d3, 2) * d5 -
          4 * a5 * b2 * b5 * pow(d3, 2) * d4 -
          4 * a5 * b4 * b5 * d2 * pow(d3, 2) -
          8 * a0 * pow(b1, 2) * d0 * d1 * d3 +
          8 * a0 * pow(b1, 2) * d0 * d1 * d5 +
          2 * a3 * pow(b0, 2) * d0 * d1 * d3 +
          16 * a0 * pow(b2, 2) * d0 * d2 * d4 -
          8 * a2 * pow(b3, 2) * d0 * d1 * d2 +
          6 * a0 * pow(b0, 2) * d1 * d3 * d5 -
          12 * a0 * pow(b0, 2) * d2 * d3 * d4 +
          2 * a0 * pow(b3, 2) * d0 * d1 * d5 -
          4 * a0 * pow(b3, 2) * d0 * d2 * d4 +
          4 * a0 * pow(b5, 2) * d0 * d1 * d3 -
          2 * a1 * pow(b0, 2) * d0 * d3 * d5 +
          4 * a2 * pow(b0, 2) * d0 * d3 * d4 -
          32 * a2 * pow(b1, 2) * d1 * d2 * d3 -
          2 * a3 * pow(b0, 2) * d0 * d1 * d5 +
          4 * a3 * pow(b0, 2) * d0 * d2 * d4 +
          8 * a3 * pow(b2, 2) * d0 * d1 * d3 -
          8 * a4 * pow(b0, 2) * d0 * d1 * d4 +
          4 * a4 * pow(b0, 2) * d0 * d2 * d3 -
          2 * a5 * pow(b0, 2) * d0 * d1 * d3 +
          8 * a0 * pow(b1, 2) * d1 * d3 * d5 -
          16 * a0 * pow(b1, 2) * d2 * d3 * d4 -
          8 * a0 * pow(b4, 2) * d0 * d1 * d5 -
          24 * a1 * pow(b1, 2) * d0 * d3 * d5 +
          64 * a1 * pow(b2, 2) * d1 * d2 * d4 +
          16 * a2 * pow(b1, 2) * d0 * d3 * d4 +
          8 * a3 * pow(b1, 2) * d0 * d1 * d5 +
          8 * a5 * pow(b1, 2) * d0 * d1 * d3 +
          12 * a0 * pow(b0, 2) * d2 * d4 * d5 +
          8 * a0 * pow(b2, 2) * d1 * d3 * d5 -
          16 * a0 * pow(b2, 2) * d2 * d3 * d4 -
          2 * a0 * pow(b5, 2) * d0 * d1 * d5 +
          4 * a0 * pow(b5, 2) * d0 * d2 * d4 -
          4 * a2 * pow(b0, 2) * d0 * d4 * d5 +
          32 * a2 * pow(b1, 2) * d1 * d2 * d5 +
          48 * a2 * pow(b2, 2) * d0 * d3 * d4 -
          16 * a3 * pow(b2, 2) * d0 * d2 * d4 -
          4 * a4 * pow(b0, 2) * d0 * d2 * d5 -
          16 * a4 * pow(b2, 2) * d0 * d2 * d3 +
          2 * a5 * pow(b0, 2) * d0 * d1 * d5 -
          4 * a5 * pow(b0, 2) * d0 * d2 * d4 -
          8 * a5 * pow(b2, 2) * d0 * d1 * d3 +
          16 * a0 * pow(b1, 2) * d2 * d4 * d5 -
          2 * a3 * pow(b0, 2) * d1 * d3 * d5 +
          4 * a3 * pow(b0, 2) * d2 * d3 * d4 +
          2 * a3 * pow(b5, 2) * d0 * d1 * d3 -
          16 * a4 * pow(b1, 2) * d0 * d2 * d5 -
          16 * a5 * pow(b1, 2) * d0 * d1 * d5 -
          8 * a1 * pow(b4, 2) * d0 * d3 * d5 +
          16 * a1 * pow(b5, 2) * d1 * d2 * d4 +
          8 * a2 * pow(b3, 2) * d1 * d2 * d5 +
          8 * a3 * pow(b4, 2) * d0 * d1 * d5 +
          4 * a0 * pow(b3, 2) * d2 * d4 * d5 +
          2 * a0 * pow(b5, 2) * d1 * d3 * d5 -
          4 * a0 * pow(b5, 2) * d2 * d3 * d4 +
          2 * a1 * pow(b5, 2) * d0 * d3 * d5 -
          4 * a2 * pow(b3, 2) * d0 * d4 * d5 +
          4 * a2 * pow(b5, 2) * d0 * d3 * d4 -
          8 * a3 * pow(b0, 2) * d2 * d4 * d5 -
          8 * a3 * pow(b2, 2) * d1 * d3 * d5 +
          16 * a3 * pow(b2, 2) * d2 * d3 * d4 +
          2 * a3 * pow(b5, 2) * d0 * d1 * d5 -
          4 * a3 * pow(b5, 2) * d0 * d2 * d4 +
          8 * a4 * pow(b0, 2) * d1 * d4 * d5 -
          4 * a4 * pow(b3, 2) * d0 * d2 * d5 +
          4 * a4 * pow(b5, 2) * d0 * d2 * d3 -
          4 * a5 * pow(b0, 2) * d1 * d3 * d5 +
          8 * a5 * pow(b0, 2) * d2 * d3 * d4 -
          2 * a5 * pow(b3, 2) * d0 * d1 * d5 +
          4 * a5 * pow(b3, 2) * d0 * d2 * d4 -
          6 * a5 * pow(b5, 2) * d0 * d1 * d3 -
          16 * a2 * pow(b1, 2) * d3 * d4 * d5 -
          8 * a5 * pow(b1, 2) * d1 * d3 * d5 +
          16 * a5 * pow(b1, 2) * d2 * d3 * d4 -
          4 * a5 * pow(b0, 2) * d2 * d4 * d5 -
          2 * a3 * pow(b5, 2) * d1 * d3 * d5 +
          4 * a3 * pow(b5, 2) * d2 * d3 * d4 -
          16 * a5 * pow(b1, 2) * d2 * d4 * d5 -
          4 * a5 * pow(b3, 2) * d2 * d4 * d5 -
          16 * a1 * b0 * b1 * d0 * d1 * d3 + 4 * a0 * b0 * b3 * d0 * d1 * d3 +
          16 * a0 * b1 * b2 * d0 * d1 * d4 - 8 * a0 * b1 * b2 * d0 * d2 * d3 -
          16 * a0 * b1 * b4 * d0 * d1 * d2 + 8 * a0 * b2 * b3 * d0 * d1 * d2 +
          16 * a1 * b0 * b1 * d0 * d1 * d5 + 16 * a1 * b0 * b2 * d0 * d1 * d4 -
          8 * a1 * b0 * b2 * d0 * d2 * d3 - 16 * a1 * b0 * b4 * d0 * d1 * d2 +
          16 * a2 * b0 * b1 * d0 * d1 * d4 - 8 * a2 * b0 * b1 * d0 * d2 * d3 +
          8 * a2 * b0 * b3 * d0 * d1 * d2 + 8 * a3 * b0 * b2 * d0 * d1 * d2 -
          16 * a4 * b0 * b1 * d0 * d1 * d2 - 4 * a0 * b0 * b1 * d0 * d3 * d5 +
          8 * a0 * b0 * b2 * d0 * d3 * d4 - 4 * a0 * b0 * b3 * d0 * d1 * d5 +
          8 * a0 * b0 * b3 * d0 * d2 * d4 - 16 * a0 * b0 * b4 * d0 * d1 * d4 +
          8 * a0 * b0 * b4 * d0 * d2 * d3 - 4 * a0 * b0 * b5 * d0 * d1 * d3 +
          8 * a0 * b1 * b2 * d0 * d2 * d5 - 8 * a0 * b2 * b5 * d0 * d1 * d2 +
          8 * a1 * b0 * b2 * d0 * d2 * d5 - 64 * a1 * b1 * b2 * d1 * d2 * d3 +
          8 * a2 * b0 * b1 * d0 * d2 * d5 + 32 * a2 * b0 * b2 * d0 * d2 * d4 -
          8 * a2 * b0 * b5 * d0 * d1 * d2 - 8 * a5 * b0 * b2 * d0 * d1 * d2 -
          8 * a0 * b0 * b2 * d0 * d4 * d5 - 8 * a0 * b0 * b4 * d0 * d2 * d5 +
          4 * a0 * b0 * b5 * d0 * d1 * d5 - 8 * a0 * b0 * b5 * d0 * d2 * d4 +
          16 * a0 * b1 * b3 * d1 * d2 * d4 + 16 * a0 * b1 * b4 * d1 * d2 * d3 +
          16 * a1 * b0 * b1 * d1 * d3 * d5 - 32 * a1 * b0 * b1 * d2 * d3 * d4 +
          16 * a1 * b0 * b3 * d1 * d2 * d4 + 16 * a1 * b0 * b4 * d1 * d2 * d3 +
          32 * a1 * b1 * b2 * d0 * d3 * d4 + 16 * a1 * b1 * b3 * d0 * d1 * d5 +
          16 * a1 * b1 * b5 * d0 * d1 * d3 - 16 * a1 * b2 * b3 * d0 * d1 * d4 -
          16 * a1 * b2 * b4 * d0 * d1 * d3 - 16 * a2 * b1 * b3 * d0 * d1 * d4 -
          16 * a2 * b1 * b4 * d0 * d1 * d3 + 16 * a2 * b2 * b3 * d0 * d1 * d3 +
          16 * a3 * b0 * b1 * d1 * d2 * d4 - 16 * a3 * b1 * b2 * d0 * d1 * d4 -
          16 * a3 * b2 * b3 * d0 * d1 * d2 + 16 * a4 * b0 * b1 * d1 * d2 * d3 -
          16 * a4 * b1 * b2 * d0 * d1 * d3 - 4 * a0 * b0 * b3 * d1 * d3 * d5 +
          8 * a0 * b0 * b3 * d2 * d3 * d4 + 4 * a0 * b1 * b3 * d0 * d3 * d5 -
          8 * a0 * b1 * b4 * d0 * d3 * d4 + 8 * a0 * b3 * b4 * d0 * d1 * d4 -
          4 * a0 * b3 * b5 * d0 * d1 * d3 + 4 * a1 * b0 * b3 * d0 * d3 * d5 -
          8 * a1 * b0 * b4 * d0 * d3 * d4 + 64 * a1 * b1 * b2 * d1 * d2 * d5 +
          128 * a2 * b1 * b2 * d1 * d2 * d4 + 4 * a3 * b0 * b1 * d0 * d3 * d5 +
          4 * a3 * b0 * b3 * d0 * d1 * d5 - 8 * a3 * b0 * b3 * d0 * d2 * d4 +
          8 * a3 * b0 * b4 * d0 * d1 * d4 - 4 * a3 * b0 * b5 * d0 * d1 * d3 -
          8 * a4 * b0 * b1 * d0 * d3 * d4 + 8 * a4 * b0 * b3 * d0 * d1 * d4 -
          4 * a5 * b0 * b3 * d0 * d1 * d3 - 16 * a0 * b1 * b2 * d1 * d4 * d5 -
          8 * a0 * b1 * b2 * d2 * d3 * d5 - 16 * a0 * b1 * b5 * d1 * d2 * d4 -
          8 * a0 * b2 * b3 * d1 * d2 * d5 - 32 * a0 * b2 * b4 * d1 * d2 * d4 +
          32 * a1 * b0 * b1 * d2 * d4 * d5 - 16 * a1 * b0 * b2 * d1 * d4 * d5 -
          8 * a1 * b0 * b2 * d2 * d3 * d5 - 16 * a1 * b0 * b5 * d1 * d2 * d4 -
          32 * a1 * b1 * b4 * d0 * d2 * d5 - 32 * a1 * b1 * b5 * d0 * d1 * d5 +
          16 * a1 * b2 * b4 * d0 * d1 * d5 + 32 * a1 * b2 * b4 * d0 * d2 * d4 +
          8 * a1 * b2 * b5 * d0 * d2 * d3 + 16 * a1 * b4 * b5 * d0 * d1 * d2 -
          16 * a2 * b0 * b1 * d1 * d4 * d5 - 8 * a2 * b0 * b1 * d2 * d3 * d5 +
          16 * a2 * b0 * b2 * d1 * d3 * d5 - 32 * a2 * b0 * b2 * d2 * d3 * d4 -
          8 * a2 * b0 * b3 * d1 * d2 * d5 - 32 * a2 * b0 * b4 * d1 * d2 * d4 +
          16 * a2 * b1 * b4 * d0 * d1 * d5 + 32 * a2 * b1 * b4 * d0 * d2 * d4 +
          8 * a2 * b1 * b5 * d0 * d2 * d3 - 32 * a2 * b2 * b3 * d0 * d2 * d4 -
          32 * a2 * b2 * b4 * d0 * d2 * d3 - 16 * a2 * b2 * b5 * d0 * d1 * d3 +
          8 * a2 * b3 * b5 * d0 * d1 * d2 - 8 * a3 * b0 * b2 * d1 * d2 * d5 +
          8 * a3 * b2 * b5 * d0 * d1 * d2 - 32 * a4 * b0 * b2 * d1 * d2 * d4 +
          16 * a4 * b1 * b2 * d0 * d1 * d5 + 32 * a4 * b1 * b2 * d0 * d2 * d4 +
          16 * a4 * b1 * b5 * d0 * d1 * d2 - 16 * a5 * b0 * b1 * d1 * d2 * d4 +
          8 * a5 * b1 * b2 * d0 * d2 * d3 + 16 * a5 * b1 * b4 * d0 * d1 * d2 +
          8 * a5 * b2 * b3 * d0 * d1 * d2 - 16 * a0 * b0 * b3 * d2 * d4 * d5 +
          16 * a0 * b0 * b4 * d1 * d4 * d5 - 8 * a0 * b0 * b5 * d1 * d3 * d5 +
          16 * a0 * b0 * b5 * d2 * d3 * d4 - 8 * a0 * b1 * b4 * d0 * d4 * d5 +
          8 * a0 * b2 * b3 * d0 * d4 * d5 - 8 * a0 * b2 * b5 * d0 * d3 * d4 +
          8 * a0 * b3 * b4 * d0 * d2 * d5 + 8 * a0 * b4 * b5 * d0 * d1 * d4 -
          8 * a0 * b4 * b5 * d0 * d2 * d3 - 8 * a1 * b0 * b4 * d0 * d4 * d5 +
          8 * a2 * b0 * b3 * d0 * d4 * d5 - 8 * a2 * b0 * b5 * d0 * d3 * d4 +
          8 * a3 * b0 * b2 * d0 * d4 * d5 + 8 * a3 * b0 * b4 * d0 * d2 * d5 -
          8 * a4 * b0 * b1 * d0 * d4 * d5 + 8 * a4 * b0 * b3 * d0 * d2 * d5 -
          16 * a4 * b0 * b4 * d0 * d1 * d5 + 8 * a4 * b0 * b5 * d0 * d1 * d4 -
          8 * a4 * b0 * b5 * d0 * d2 * d3 - 8 * a5 * b0 * b2 * d0 * d3 * d4 +
          8 * a5 * b0 * b4 * d0 * d1 * d4 - 8 * a5 * b0 * b4 * d0 * d2 * d3 +
          8 * a5 * b0 * b5 * d0 * d1 * d3 + 8 * a0 * b2 * b5 * d1 * d2 * d5 -
          8 * a1 * b2 * b5 * d0 * d2 * d5 + 8 * a2 * b0 * b5 * d1 * d2 * d5 -
          8 * a2 * b1 * b5 * d0 * d2 * d5 + 8 * a5 * b0 * b2 * d1 * d2 * d5 -
          8 * a5 * b1 * b2 * d0 * d2 * d5 - 8 * a0 * b0 * b5 * d2 * d4 * d5 -
          32 * a1 * b1 * b2 * d3 * d4 * d5 - 16 * a1 * b1 * b5 * d1 * d3 * d5 +
          32 * a1 * b1 * b5 * d2 * d3 * d4 + 16 * a1 * b2 * b3 * d1 * d4 * d5 +
          16 * a1 * b2 * b4 * d1 * d3 * d5 - 32 * a1 * b2 * b4 * d2 * d3 * d4 -
          16 * a1 * b3 * b5 * d1 * d2 * d4 - 16 * a1 * b4 * b5 * d1 * d2 * d3 +
          16 * a2 * b1 * b3 * d1 * d4 * d5 + 16 * a2 * b1 * b4 * d1 * d3 * d5 -
          32 * a2 * b1 * b4 * d2 * d3 * d4 - 16 * a2 * b2 * b3 * d1 * d3 * d5 +
          32 * a2 * b2 * b3 * d2 * d3 * d4 + 32 * a2 * b3 * b4 * d1 * d2 * d4 +
          16 * a3 * b1 * b2 * d1 * d4 * d5 - 16 * a3 * b1 * b5 * d1 * d2 * d4 +
          16 * a3 * b2 * b3 * d1 * d2 * d5 + 32 * a3 * b2 * b4 * d1 * d2 * d4 +
          16 * a4 * b1 * b2 * d1 * d3 * d5 - 32 * a4 * b1 * b2 * d2 * d3 * d4 -
          16 * a4 * b1 * b5 * d1 * d2 * d3 + 32 * a4 * b2 * b3 * d1 * d2 * d4 -
          4 * a5 * b0 * b5 * d0 * d1 * d5 + 8 * a5 * b0 * b5 * d0 * d2 * d4 -
          16 * a5 * b1 * b3 * d1 * d2 * d4 - 16 * a5 * b1 * b4 * d1 * d2 * d3 +
          8 * a0 * b1 * b4 * d3 * d4 * d5 - 8 * a0 * b3 * b4 * d1 * d4 * d5 +
          4 * a0 * b3 * b5 * d1 * d3 * d5 - 8 * a0 * b3 * b5 * d2 * d3 * d4 +
          8 * a1 * b0 * b4 * d3 * d4 * d5 - 4 * a1 * b3 * b5 * d0 * d3 * d5 +
          8 * a1 * b4 * b5 * d0 * d3 * d4 + 8 * a2 * b3 * b4 * d0 * d3 * d5 +
          8 * a3 * b0 * b3 * d2 * d4 * d5 - 8 * a3 * b0 * b4 * d1 * d4 * d5 +
          4 * a3 * b0 * b5 * d1 * d3 * d5 - 8 * a3 * b0 * b5 * d2 * d3 * d4 -
          4 * a3 * b1 * b5 * d0 * d3 * d5 - 8 * a3 * b2 * b3 * d0 * d4 * d5 +
          8 * a3 * b2 * b4 * d0 * d3 * d5 - 8 * a3 * b3 * b4 * d0 * d2 * d5 -
          4 * a3 * b3 * b5 * d0 * d1 * d5 + 8 * a3 * b3 * b5 * d0 * d2 * d4 -
          8 * a3 * b4 * b5 * d0 * d1 * d4 + 8 * a4 * b0 * b1 * d3 * d4 * d5 -
          8 * a4 * b0 * b3 * d1 * d4 * d5 - 16 * a4 * b1 * b4 * d0 * d3 * d5 +
          8 * a4 * b1 * b5 * d0 * d3 * d4 + 8 * a4 * b2 * b3 * d0 * d3 * d5 +
          16 * a4 * b3 * b4 * d0 * d1 * d5 - 8 * a4 * b3 * b5 * d0 * d1 * d4 +
          4 * a5 * b0 * b3 * d1 * d3 * d5 - 8 * a5 * b0 * b3 * d2 * d3 * d4 -
          4 * a5 * b1 * b3 * d0 * d3 * d5 + 8 * a5 * b1 * b4 * d0 * d3 * d4 -
          8 * a5 * b3 * b4 * d0 * d1 * d4 + 4 * a5 * b3 * b5 * d0 * d1 * d3 -
          32 * a1 * b1 * b5 * d2 * d4 * d5 + 8 * a1 * b2 * b5 * d2 * d3 * d5 +
          8 * a2 * b1 * b5 * d2 * d3 * d5 - 8 * a2 * b3 * b5 * d1 * d2 * d5 -
          8 * a3 * b2 * b5 * d1 * d2 * d5 + 8 * a5 * b1 * b2 * d2 * d3 * d5 +
          32 * a5 * b1 * b5 * d1 * d2 * d4 - 8 * a5 * b2 * b3 * d1 * d2 * d5 +
          8 * a0 * b3 * b5 * d2 * d4 * d5 - 8 * a0 * b4 * b5 * d1 * d4 * d5 +
          8 * a1 * b4 * b5 * d0 * d4 * d5 - 8 * a2 * b4 * b5 * d0 * d3 * d5 +
          8 * a3 * b0 * b5 * d2 * d4 * d5 - 8 * a4 * b0 * b5 * d1 * d4 * d5 +
          8 * a4 * b1 * b5 * d0 * d4 * d5 - 8 * a4 * b2 * b5 * d0 * d3 * d5 +
          8 * a5 * b0 * b3 * d2 * d4 * d5 - 8 * a5 * b0 * b4 * d1 * d4 * d5 +
          4 * a5 * b0 * b5 * d1 * d3 * d5 - 8 * a5 * b0 * b5 * d2 * d3 * d4 +
          8 * a5 * b1 * b4 * d0 * d4 * d5 + 4 * a5 * b1 * b5 * d0 * d3 * d5 -
          8 * a5 * b2 * b4 * d0 * d3 * d5 + 8 * a5 * b2 * b5 * d0 * d3 * d4 +
          4 * a5 * b3 * b5 * d0 * d1 * d5 - 8 * a5 * b3 * b5 * d0 * d2 * d4 +
          8 * a5 * b4 * b5 * d0 * d2 * d3 - 8 * a1 * b4 * b5 * d3 * d4 * d5 -
          8 * a3 * b3 * b5 * d2 * d4 * d5 + 8 * a3 * b4 * b5 * d1 * d4 * d5 -
          8 * a4 * b1 * b5 * d3 * d4 * d5 + 8 * a4 * b3 * b5 * d1 * d4 * d5 -
          8 * a5 * b1 * b4 * d3 * d4 * d5 + 8 * a5 * b3 * b4 * d1 * d4 * d5 -
          4 * a5 * b3 * b5 * d1 * d3 * d5 + 8 * a5 * b3 * b5 * d2 * d3 * d4,
      a1 * pow(b0, 2) * c0 * pow(d3, 2) - a0 * pow(b3, 2) * c1 * pow(d0, 2) -
          3 * a0 * pow(b0, 2) * c1 * pow(d3, 2) +
          3 * a1 * pow(b3, 2) * c0 * pow(d0, 2) +
          12 * a0 * pow(b0, 2) * c1 * pow(d4, 2) +
          4 * a0 * pow(b4, 2) * c1 * pow(d0, 2) -
          4 * a1 * pow(b0, 2) * c0 * pow(d4, 2) +
          4 * a1 * pow(b0, 2) * c3 * pow(d1, 2) +
          12 * a1 * pow(b1, 2) * c3 * pow(d0, 2) -
          12 * a1 * pow(b4, 2) * c0 * pow(d0, 2) -
          12 * a3 * pow(b0, 2) * c1 * pow(d1, 2) -
          4 * a3 * pow(b1, 2) * c1 * pow(d0, 2) -
          3 * a0 * pow(b0, 2) * c1 * pow(d5, 2) -
          4 * a0 * pow(b2, 2) * c1 * pow(d3, 2) +
          4 * a0 * pow(b3, 2) * c1 * pow(d2, 2) -
          a0 * pow(b5, 2) * c1 * pow(d0, 2) +
          a1 * pow(b0, 2) * c0 * pow(d5, 2) +
          4 * a1 * pow(b0, 2) * c3 * pow(d2, 2) -
          4 * a1 * pow(b2, 2) * c0 * pow(d3, 2) +
          4 * a1 * pow(b2, 2) * c3 * pow(d0, 2) +
          4 * a1 * pow(b3, 2) * c0 * pow(d2, 2) +
          3 * a1 * pow(b5, 2) * c0 * pow(d0, 2) -
          4 * a3 * pow(b0, 2) * c1 * pow(d2, 2) -
          4 * a3 * pow(b2, 2) * c1 * pow(d0, 2) -
          8 * a0 * pow(b1, 2) * c1 * pow(d5, 2) +
          16 * a0 * pow(b2, 2) * c1 * pow(d4, 2) +
          16 * a0 * pow(b4, 2) * c1 * pow(d2, 2) -
          24 * a0 * pow(b5, 2) * c1 * pow(d1, 2) -
          4 * a1 * pow(b0, 2) * c5 * pow(d1, 2) +
          24 * a1 * pow(b1, 2) * c0 * pow(d5, 2) +
          48 * a1 * pow(b1, 2) * c3 * pow(d2, 2) -
          12 * a1 * pow(b1, 2) * c5 * pow(d0, 2) -
          16 * a1 * pow(b2, 2) * c0 * pow(d4, 2) +
          16 * a1 * pow(b2, 2) * c3 * pow(d1, 2) -
          16 * a1 * pow(b4, 2) * c0 * pow(d2, 2) +
          8 * a1 * pow(b5, 2) * c0 * pow(d1, 2) -
          8 * a2 * pow(b0, 2) * c4 * pow(d1, 2) -
          8 * a2 * pow(b1, 2) * c4 * pow(d0, 2) -
          16 * a3 * pow(b1, 2) * c1 * pow(d2, 2) -
          48 * a3 * pow(b2, 2) * c1 * pow(d1, 2) +
          8 * a4 * pow(b0, 2) * c2 * pow(d1, 2) +
          8 * a4 * pow(b1, 2) * c2 * pow(d0, 2) +
          12 * a5 * pow(b0, 2) * c1 * pow(d1, 2) +
          4 * a5 * pow(b1, 2) * c1 * pow(d0, 2) -
          4 * a0 * pow(b2, 2) * c1 * pow(d5, 2) -
          4 * a0 * pow(b5, 2) * c1 * pow(d2, 2) +
          4 * a1 * pow(b0, 2) * c3 * pow(d4, 2) -
          4 * a1 * pow(b0, 2) * c5 * pow(d2, 2) +
          4 * a1 * pow(b2, 2) * c0 * pow(d5, 2) -
          4 * a1 * pow(b2, 2) * c5 * pow(d0, 2) +
          4 * a1 * pow(b4, 2) * c3 * pow(d0, 2) +
          4 * a1 * pow(b5, 2) * c0 * pow(d2, 2) -
          8 * a2 * pow(b0, 2) * c4 * pow(d2, 2) -
          24 * a2 * pow(b2, 2) * c4 * pow(d0, 2) -
          4 * a3 * pow(b0, 2) * c1 * pow(d4, 2) -
          4 * a3 * pow(b4, 2) * c1 * pow(d0, 2) +
          24 * a4 * pow(b0, 2) * c2 * pow(d2, 2) +
          8 * a4 * pow(b2, 2) * c2 * pow(d0, 2) +
          4 * a5 * pow(b0, 2) * c1 * pow(d2, 2) +
          4 * a5 * pow(b2, 2) * c1 * pow(d0, 2) -
          a0 * pow(b3, 2) * c1 * pow(d5, 2) -
          3 * a0 * pow(b5, 2) * c1 * pow(d3, 2) +
          2 * a1 * pow(b0, 2) * c3 * pow(d5, 2) -
          a1 * pow(b0, 2) * c5 * pow(d3, 2) -
          48 * a1 * pow(b1, 2) * c5 * pow(d2, 2) -
          16 * a1 * pow(b2, 2) * c5 * pow(d1, 2) +
          3 * a1 * pow(b3, 2) * c0 * pow(d5, 2) -
          3 * a1 * pow(b3, 2) * c5 * pow(d0, 2) +
          a1 * pow(b5, 2) * c0 * pow(d3, 2) -
          2 * a1 * pow(b5, 2) * c3 * pow(d0, 2) -
          2 * a2 * pow(b0, 2) * c4 * pow(d3, 2) -
          32 * a2 * pow(b1, 2) * c4 * pow(d2, 2) -
          96 * a2 * pow(b2, 2) * c4 * pow(d1, 2) +
          2 * a2 * pow(b3, 2) * c4 * pow(d0, 2) +
          2 * a3 * pow(b0, 2) * c1 * pow(d5, 2) -
          2 * a3 * pow(b5, 2) * c1 * pow(d0, 2) -
          2 * a4 * pow(b0, 2) * c2 * pow(d3, 2) +
          96 * a4 * pow(b1, 2) * c2 * pow(d2, 2) +
          32 * a4 * pow(b2, 2) * c2 * pow(d1, 2) +
          2 * a4 * pow(b3, 2) * c2 * pow(d0, 2) +
          3 * a5 * pow(b0, 2) * c1 * pow(d3, 2) +
          16 * a5 * pow(b1, 2) * c1 * pow(d2, 2) +
          48 * a5 * pow(b2, 2) * c1 * pow(d1, 2) +
          a5 * pow(b3, 2) * c1 * pow(d0, 2) +
          4 * a0 * pow(b4, 2) * c1 * pow(d5, 2) +
          4 * a0 * pow(b5, 2) * c1 * pow(d4, 2) +
          12 * a1 * pow(b1, 2) * c3 * pow(d5, 2) +
          16 * a1 * pow(b2, 2) * c3 * pow(d4, 2) -
          4 * a1 * pow(b4, 2) * c0 * pow(d5, 2) +
          16 * a1 * pow(b4, 2) * c3 * pow(d2, 2) +
          8 * a1 * pow(b4, 2) * c5 * pow(d0, 2) -
          4 * a1 * pow(b5, 2) * c0 * pow(d4, 2) +
          4 * a1 * pow(b5, 2) * c3 * pow(d1, 2) -
          4 * a3 * pow(b1, 2) * c1 * pow(d5, 2) -
          16 * a3 * pow(b2, 2) * c1 * pow(d4, 2) -
          16 * a3 * pow(b4, 2) * c1 * pow(d2, 2) -
          12 * a3 * pow(b5, 2) * c1 * pow(d1, 2) -
          8 * a5 * pow(b0, 2) * c1 * pow(d4, 2) -
          3 * a1 * pow(b0, 2) * c5 * pow(d5, 2) -
          4 * a1 * pow(b2, 2) * c3 * pow(d5, 2) +
          4 * a1 * pow(b2, 2) * c5 * pow(d3, 2) -
          4 * a1 * pow(b3, 2) * c5 * pow(d2, 2) -
          4 * a1 * pow(b5, 2) * c3 * pow(d2, 2) -
          a1 * pow(b5, 2) * c5 * pow(d0, 2) +
          2 * a2 * pow(b0, 2) * c4 * pow(d5, 2) -
          24 * a2 * pow(b2, 2) * c4 * pow(d3, 2) -
          8 * a2 * pow(b3, 2) * c4 * pow(d2, 2) -
          2 * a2 * pow(b5, 2) * c4 * pow(d0, 2) +
          4 * a3 * pow(b2, 2) * c1 * pow(d5, 2) +
          4 * a3 * pow(b5, 2) * c1 * pow(d2, 2) +
          2 * a4 * pow(b0, 2) * c2 * pow(d5, 2) +
          8 * a4 * pow(b2, 2) * c2 * pow(d3, 2) +
          24 * a4 * pow(b3, 2) * c2 * pow(d2, 2) -
          2 * a4 * pow(b5, 2) * c2 * pow(d0, 2) +
          a5 * pow(b0, 2) * c1 * pow(d5, 2) +
          4 * a5 * pow(b2, 2) * c1 * pow(d3, 2) -
          4 * a5 * pow(b3, 2) * c1 * pow(d2, 2) +
          3 * a5 * pow(b5, 2) * c1 * pow(d0, 2) -
          36 * a1 * pow(b1, 2) * c5 * pow(d5, 2) -
          12 * a1 * pow(b5, 2) * c5 * pow(d1, 2) +
          8 * a2 * pow(b1, 2) * c4 * pow(d5, 2) -
          8 * a2 * pow(b5, 2) * c4 * pow(d1, 2) +
          8 * a4 * pow(b1, 2) * c2 * pow(d5, 2) -
          8 * a4 * pow(b5, 2) * c2 * pow(d1, 2) +
          12 * a5 * pow(b1, 2) * c1 * pow(d5, 2) +
          36 * a5 * pow(b5, 2) * c1 * pow(d1, 2) +
          4 * a1 * pow(b4, 2) * c3 * pow(d5, 2) +
          4 * a1 * pow(b5, 2) * c3 * pow(d4, 2) -
          4 * a3 * pow(b4, 2) * c1 * pow(d5, 2) -
          4 * a3 * pow(b5, 2) * c1 * pow(d4, 2) -
          3 * a1 * pow(b3, 2) * c5 * pow(d5, 2) -
          a1 * pow(b5, 2) * c5 * pow(d3, 2) +
          2 * a2 * pow(b3, 2) * c4 * pow(d5, 2) -
          2 * a2 * pow(b5, 2) * c4 * pow(d3, 2) +
          2 * a4 * pow(b3, 2) * c2 * pow(d5, 2) -
          2 * a4 * pow(b5, 2) * c2 * pow(d3, 2) +
          a5 * pow(b3, 2) * c1 * pow(d5, 2) +
          3 * a5 * pow(b5, 2) * c1 * pow(d3, 2) +
          2 * a0 * b0 * b1 * c0 * pow(d3, 2) -
          8 * a0 * b0 * b1 * c0 * pow(d4, 2) +
          8 * a0 * b0 * b1 * c3 * pow(d1, 2) -
          24 * a0 * b0 * b3 * c1 * pow(d1, 2) +
          8 * a0 * b1 * b3 * c0 * pow(d1, 2) +
          8 * a1 * b0 * b3 * c0 * pow(d1, 2) +
          8 * a3 * b0 * b1 * c0 * pow(d1, 2) +
          2 * a0 * b0 * b1 * c0 * pow(d5, 2) +
          8 * a0 * b0 * b1 * c3 * pow(d2, 2) -
          8 * a0 * b0 * b3 * c1 * pow(d2, 2) -
          8 * a1 * b1 * b3 * c1 * pow(d0, 2) -
          8 * a0 * b0 * b1 * c5 * pow(d1, 2) -
          16 * a0 * b0 * b2 * c4 * pow(d1, 2) +
          16 * a0 * b0 * b4 * c2 * pow(d1, 2) +
          24 * a0 * b0 * b5 * c1 * pow(d1, 2) -
          2 * a0 * b1 * b3 * c3 * pow(d0, 2) -
          8 * a0 * b1 * b5 * c0 * pow(d1, 2) -
          2 * a1 * b0 * b3 * c3 * pow(d0, 2) -
          8 * a1 * b0 * b5 * c0 * pow(d1, 2) -
          2 * a3 * b0 * b1 * c3 * pow(d0, 2) -
          2 * a3 * b0 * b3 * c1 * pow(d0, 2) +
          6 * a3 * b1 * b3 * c0 * pow(d0, 2) -
          8 * a5 * b0 * b1 * c0 * pow(d1, 2) +
          8 * a0 * b0 * b1 * c3 * pow(d4, 2) -
          8 * a0 * b0 * b1 * c5 * pow(d2, 2) -
          16 * a0 * b0 * b2 * c4 * pow(d2, 2) -
          8 * a0 * b0 * b3 * c1 * pow(d4, 2) +
          48 * a0 * b0 * b4 * c2 * pow(d2, 2) +
          8 * a0 * b0 * b5 * c1 * pow(d2, 2) +
          8 * a0 * b1 * b2 * c2 * pow(d3, 2) -
          16 * a0 * b2 * b4 * c0 * pow(d2, 2) -
          16 * a1 * b0 * b1 * c1 * pow(d5, 2) +
          8 * a1 * b0 * b2 * c2 * pow(d3, 2) -
          16 * a1 * b1 * b2 * c4 * pow(d0, 2) -
          32 * a1 * b1 * b3 * c1 * pow(d2, 2) +
          16 * a1 * b1 * b4 * c2 * pow(d0, 2) +
          8 * a1 * b1 * b5 * c1 * pow(d0, 2) +
          8 * a2 * b0 * b1 * c2 * pow(d3, 2) -
          8 * a2 * b0 * b2 * c1 * pow(d3, 2) -
          16 * a2 * b0 * b4 * c0 * pow(d2, 2) -
          8 * a2 * b1 * b2 * c0 * pow(d3, 2) +
          8 * a2 * b1 * b2 * c3 * pow(d0, 2) -
          8 * a2 * b2 * b3 * c1 * pow(d0, 2) -
          16 * a4 * b0 * b2 * c0 * pow(d2, 2) +
          4 * a0 * b0 * b1 * c3 * pow(d5, 2) -
          2 * a0 * b0 * b1 * c5 * pow(d3, 2) -
          4 * a0 * b0 * b2 * c4 * pow(d3, 2) +
          4 * a0 * b0 * b3 * c1 * pow(d5, 2) -
          4 * a0 * b0 * b4 * c2 * pow(d3, 2) +
          6 * a0 * b0 * b5 * c1 * pow(d3, 2) -
          4 * a0 * b1 * b3 * c0 * pow(d5, 2) -
          8 * a0 * b1 * b3 * c3 * pow(d2, 2) +
          2 * a0 * b1 * b3 * c5 * pow(d0, 2) +
          8 * a0 * b1 * b4 * c4 * pow(d0, 2) -
          2 * a0 * b1 * b5 * c0 * pow(d3, 2) +
          2 * a0 * b1 * b5 * c3 * pow(d0, 2) -
          4 * a0 * b2 * b3 * c4 * pow(d0, 2) +
          4 * a0 * b2 * b4 * c0 * pow(d3, 2) -
          4 * a0 * b2 * b4 * c3 * pow(d0, 2) -
          4 * a0 * b3 * b4 * c2 * pow(d0, 2) +
          2 * a0 * b3 * b5 * c1 * pow(d0, 2) -
          4 * a1 * b0 * b3 * c0 * pow(d5, 2) -
          8 * a1 * b0 * b3 * c3 * pow(d2, 2) +
          2 * a1 * b0 * b3 * c5 * pow(d0, 2) +
          8 * a1 * b0 * b4 * c4 * pow(d0, 2) -
          2 * a1 * b0 * b5 * c0 * pow(d3, 2) +
          2 * a1 * b0 * b5 * c3 * pow(d0, 2) +
          32 * a1 * b2 * b3 * c2 * pow(d1, 2) -
          6 * a1 * b3 * b5 * c0 * pow(d0, 2) +
          32 * a2 * b0 * b2 * c1 * pow(d4, 2) -
          4 * a2 * b0 * b3 * c4 * pow(d0, 2) +
          4 * a2 * b0 * b4 * c0 * pow(d3, 2) -
          4 * a2 * b0 * b4 * c3 * pow(d0, 2) -
          32 * a2 * b1 * b2 * c0 * pow(d4, 2) +
          32 * a2 * b1 * b2 * c3 * pow(d1, 2) +
          32 * a2 * b1 * b3 * c2 * pow(d1, 2) -
          96 * a2 * b2 * b3 * c1 * pow(d1, 2) +
          12 * a2 * b3 * b4 * c0 * pow(d0, 2) -
          4 * a3 * b0 * b1 * c0 * pow(d5, 2) -
          8 * a3 * b0 * b1 * c3 * pow(d2, 2) +
          2 * a3 * b0 * b1 * c5 * pow(d0, 2) -
          4 * a3 * b0 * b2 * c4 * pow(d0, 2) +
          8 * a3 * b0 * b3 * c1 * pow(d2, 2) -
          4 * a3 * b0 * b4 * c2 * pow(d0, 2) +
          2 * a3 * b0 * b5 * c1 * pow(d0, 2) +
          32 * a3 * b1 * b2 * c2 * pow(d1, 2) +
          8 * a3 * b1 * b3 * c0 * pow(d2, 2) -
          6 * a3 * b1 * b5 * c0 * pow(d0, 2) +
          12 * a3 * b2 * b4 * c0 * pow(d0, 2) +
          8 * a4 * b0 * b1 * c4 * pow(d0, 2) +
          4 * a4 * b0 * b2 * c0 * pow(d3, 2) -
          4 * a4 * b0 * b2 * c3 * pow(d0, 2) -
          4 * a4 * b0 * b3 * c2 * pow(d0, 2) +
          8 * a4 * b0 * b4 * c1 * pow(d0, 2) -
          24 * a4 * b1 * b4 * c0 * pow(d0, 2) +
          12 * a4 * b2 * b3 * c0 * pow(d0, 2) -
          2 * a5 * b0 * b1 * c0 * pow(d3, 2) +
          2 * a5 * b0 * b1 * c3 * pow(d0, 2) +
          2 * a5 * b0 * b3 * c1 * pow(d0, 2) -
          6 * a5 * b1 * b3 * c0 * pow(d0, 2) -
          16 * a0 * b0 * b5 * c1 * pow(d4, 2) -
          8 * a0 * b1 * b3 * c5 * pow(d1, 2) +
          8 * a0 * b1 * b5 * c0 * pow(d4, 2) -
          8 * a0 * b1 * b5 * c3 * pow(d1, 2) -
          16 * a0 * b3 * b4 * c2 * pow(d1, 2) +
          24 * a0 * b3 * b5 * c1 * pow(d1, 2) -
          8 * a1 * b0 * b3 * c5 * pow(d1, 2) +
          8 * a1 * b0 * b5 * c0 * pow(d4, 2) -
          8 * a1 * b0 * b5 * c3 * pow(d1, 2) -
          64 * a1 * b1 * b2 * c4 * pow(d2, 2) +
          192 * a1 * b1 * b4 * c2 * pow(d2, 2) +
          32 * a1 * b1 * b5 * c1 * pow(d2, 2) -
          64 * a1 * b2 * b4 * c1 * pow(d2, 2) -
          8 * a1 * b3 * b5 * c0 * pow(d1, 2) -
          8 * a2 * b0 * b2 * c1 * pow(d5, 2) +
          8 * a2 * b1 * b2 * c0 * pow(d5, 2) -
          8 * a2 * b1 * b2 * c5 * pow(d0, 2) -
          64 * a2 * b1 * b4 * c1 * pow(d2, 2) +
          16 * a2 * b2 * b4 * c2 * pow(d0, 2) +
          8 * a2 * b2 * b5 * c1 * pow(d0, 2) +
          16 * a2 * b3 * b4 * c0 * pow(d1, 2) -
          8 * a3 * b0 * b1 * c5 * pow(d1, 2) -
          16 * a3 * b0 * b4 * c2 * pow(d1, 2) +
          24 * a3 * b0 * b5 * c1 * pow(d1, 2) -
          8 * a3 * b1 * b5 * c0 * pow(d1, 2) +
          16 * a3 * b2 * b4 * c0 * pow(d1, 2) -
          16 * a4 * b0 * b3 * c2 * pow(d1, 2) -
          64 * a4 * b1 * b2 * c1 * pow(d2, 2) +
          16 * a4 * b2 * b3 * c0 * pow(d1, 2) +
          8 * a5 * b0 * b1 * c0 * pow(d4, 2) -
          8 * a5 * b0 * b1 * c3 * pow(d1, 2) +
          24 * a5 * b0 * b3 * c1 * pow(d1, 2) -
          8 * a5 * b1 * b3 * c0 * pow(d1, 2) -
          6 * a0 * b0 * b1 * c5 * pow(d5, 2) +
          4 * a0 * b0 * b2 * c4 * pow(d5, 2) +
          4 * a0 * b0 * b4 * c2 * pow(d5, 2) +
          2 * a0 * b0 * b5 * c1 * pow(d5, 2) +
          8 * a0 * b1 * b3 * c5 * pow(d2, 2) +
          2 * a0 * b1 * b5 * c0 * pow(d5, 2) -
          2 * a0 * b1 * b5 * c5 * pow(d0, 2) +
          16 * a0 * b2 * b3 * c4 * pow(d2, 2) -
          4 * a0 * b2 * b4 * c0 * pow(d5, 2) +
          16 * a0 * b2 * b4 * c3 * pow(d2, 2) +
          4 * a0 * b2 * b4 * c5 * pow(d0, 2) +
          4 * a0 * b2 * b5 * c4 * pow(d0, 2) -
          48 * a0 * b3 * b4 * c2 * pow(d2, 2) +
          4 * a0 * b4 * b5 * c2 * pow(d0, 2) +
          8 * a1 * b0 * b3 * c5 * pow(d2, 2) +
          2 * a1 * b0 * b5 * c0 * pow(d5, 2) -
          2 * a1 * b0 * b5 * c5 * pow(d0, 2) -
          8 * a1 * b1 * b3 * c1 * pow(d5, 2) -
          32 * a1 * b2 * b5 * c2 * pow(d1, 2) -
          8 * a1 * b3 * b5 * c0 * pow(d2, 2) +
          16 * a2 * b0 * b3 * c4 * pow(d2, 2) -
          4 * a2 * b0 * b4 * c0 * pow(d5, 2) +
          16 * a2 * b0 * b4 * c3 * pow(d2, 2) +
          4 * a2 * b0 * b4 * c5 * pow(d0, 2) +
          4 * a2 * b0 * b5 * c4 * pow(d0, 2) -
          32 * a2 * b1 * b2 * c5 * pow(d1, 2) -
          32 * a2 * b1 * b5 * c2 * pow(d1, 2) +
          64 * a2 * b2 * b4 * c2 * pow(d1, 2) +
          96 * a2 * b2 * b5 * c1 * pow(d1, 2) +
          16 * a2 * b3 * b4 * c0 * pow(d2, 2) -
          12 * a2 * b4 * b5 * c0 * pow(d0, 2) +
          8 * a3 * b0 * b1 * c5 * pow(d2, 2) +
          16 * a3 * b0 * b2 * c4 * pow(d2, 2) -
          48 * a3 * b0 * b4 * c2 * pow(d2, 2) -
          8 * a3 * b1 * b5 * c0 * pow(d2, 2) +
          16 * a3 * b2 * b4 * c0 * pow(d2, 2) -
          4 * a4 * b0 * b2 * c0 * pow(d5, 2) +
          16 * a4 * b0 * b2 * c3 * pow(d2, 2) +
          4 * a4 * b0 * b2 * c5 * pow(d0, 2) -
          48 * a4 * b0 * b3 * c2 * pow(d2, 2) +
          32 * a4 * b0 * b4 * c1 * pow(d2, 2) +
          4 * a4 * b0 * b5 * c2 * pow(d0, 2) -
          32 * a4 * b1 * b4 * c0 * pow(d2, 2) +
          16 * a4 * b2 * b3 * c0 * pow(d2, 2) -
          12 * a4 * b2 * b5 * c0 * pow(d0, 2) +
          2 * a5 * b0 * b1 * c0 * pow(d5, 2) -
          2 * a5 * b0 * b1 * c5 * pow(d0, 2) +
          4 * a5 * b0 * b2 * c4 * pow(d0, 2) +
          4 * a5 * b0 * b4 * c2 * pow(d0, 2) -
          2 * a5 * b0 * b5 * c1 * pow(d0, 2) -
          32 * a5 * b1 * b2 * c2 * pow(d1, 2) -
          8 * a5 * b1 * b3 * c0 * pow(d2, 2) +
          6 * a5 * b1 * b5 * c0 * pow(d0, 2) -
          12 * a5 * b2 * b4 * c0 * pow(d0, 2) -
          2 * a0 * b1 * b3 * c3 * pow(d5, 2) +
          16 * a0 * b1 * b5 * c5 * pow(d1, 2) +
          16 * a0 * b2 * b5 * c4 * pow(d1, 2) -
          2 * a1 * b0 * b3 * c3 * pow(d5, 2) +
          16 * a1 * b0 * b5 * c5 * pow(d1, 2) +
          2 * a1 * b3 * b5 * c3 * pow(d0, 2) +
          16 * a2 * b0 * b5 * c4 * pow(d1, 2) +
          32 * a2 * b1 * b2 * c3 * pow(d4, 2) -
          32 * a2 * b2 * b3 * c1 * pow(d4, 2) -
          4 * a2 * b3 * b4 * c3 * pow(d0, 2) -
          16 * a2 * b4 * b5 * c0 * pow(d1, 2) -
          2 * a3 * b0 * b1 * c3 * pow(d5, 2) -
          2 * a3 * b0 * b3 * c1 * pow(d5, 2) +
          6 * a3 * b1 * b3 * c0 * pow(d5, 2) -
          6 * a3 * b1 * b3 * c5 * pow(d0, 2) +
          2 * a3 * b1 * b5 * c3 * pow(d0, 2) +
          4 * a3 * b2 * b3 * c4 * pow(d0, 2) -
          4 * a3 * b2 * b4 * c3 * pow(d0, 2) +
          4 * a3 * b3 * b4 * c2 * pow(d0, 2) +
          2 * a3 * b3 * b5 * c1 * pow(d0, 2) +
          8 * a4 * b1 * b4 * c3 * pow(d0, 2) -
          4 * a4 * b2 * b3 * c3 * pow(d0, 2) -
          16 * a4 * b2 * b5 * c0 * pow(d1, 2) -
          8 * a4 * b3 * b4 * c1 * pow(d0, 2) +
          16 * a5 * b0 * b1 * c5 * pow(d1, 2) +
          16 * a5 * b0 * b2 * c4 * pow(d1, 2) -
          48 * a5 * b0 * b5 * c1 * pow(d1, 2) +
          2 * a5 * b1 * b3 * c3 * pow(d0, 2) +
          16 * a5 * b1 * b5 * c0 * pow(d1, 2) -
          16 * a5 * b2 * b4 * c0 * pow(d1, 2) -
          8 * a0 * b1 * b5 * c3 * pow(d4, 2) +
          8 * a0 * b3 * b5 * c1 * pow(d4, 2) -
          8 * a1 * b0 * b5 * c3 * pow(d4, 2) +
          16 * a1 * b1 * b2 * c4 * pow(d5, 2) +
          16 * a1 * b1 * b4 * c2 * pow(d5, 2) +
          24 * a1 * b1 * b5 * c1 * pow(d5, 2) -
          16 * a1 * b2 * b4 * c1 * pow(d5, 2) -
          8 * a1 * b2 * b5 * c2 * pow(d3, 2) -
          8 * a2 * b1 * b2 * c3 * pow(d5, 2) +
          8 * a2 * b1 * b2 * c5 * pow(d3, 2) -
          16 * a2 * b1 * b4 * c1 * pow(d5, 2) -
          8 * a2 * b1 * b5 * c2 * pow(d3, 2) +
          8 * a2 * b2 * b3 * c1 * pow(d5, 2) +
          16 * a2 * b2 * b4 * c2 * pow(d3, 2) +
          8 * a2 * b2 * b5 * c1 * pow(d3, 2) +
          8 * a3 * b0 * b5 * c1 * pow(d4, 2) -
          16 * a4 * b1 * b2 * c1 * pow(d5, 2) -
          8 * a5 * b0 * b1 * c3 * pow(d4, 2) +
          8 * a5 * b0 * b3 * c1 * pow(d4, 2) -
          8 * a5 * b0 * b5 * c1 * pow(d2, 2) -
          8 * a5 * b1 * b2 * c2 * pow(d3, 2) +
          8 * a5 * b1 * b5 * c0 * pow(d2, 2) +
          6 * a0 * b1 * b3 * c5 * pow(d5, 2) -
          2 * a0 * b1 * b5 * c3 * pow(d5, 2) +
          2 * a0 * b1 * b5 * c5 * pow(d3, 2) -
          4 * a0 * b2 * b3 * c4 * pow(d5, 2) +
          4 * a0 * b2 * b4 * c3 * pow(d5, 2) -
          4 * a0 * b2 * b4 * c5 * pow(d3, 2) +
          4 * a0 * b2 * b5 * c4 * pow(d3, 2) -
          4 * a0 * b3 * b4 * c2 * pow(d5, 2) -
          2 * a0 * b3 * b5 * c1 * pow(d5, 2) +
          4 * a0 * b4 * b5 * c2 * pow(d3, 2) +
          6 * a1 * b0 * b3 * c5 * pow(d5, 2) -
          2 * a1 * b0 * b5 * c3 * pow(d5, 2) +
          2 * a1 * b0 * b5 * c5 * pow(d3, 2) -
          2 * a1 * b3 * b5 * c0 * pow(d5, 2) +
          8 * a1 * b3 * b5 * c3 * pow(d2, 2) +
          4 * a1 * b3 * b5 * c5 * pow(d0, 2) -
          8 * a1 * b4 * b5 * c4 * pow(d0, 2) -
          4 * a2 * b0 * b3 * c4 * pow(d5, 2) +
          4 * a2 * b0 * b4 * c3 * pow(d5, 2) -
          4 * a2 * b0 * b4 * c5 * pow(d3, 2) +
          4 * a2 * b0 * b5 * c4 * pow(d3, 2) +
          4 * a2 * b3 * b4 * c0 * pow(d5, 2) -
          16 * a2 * b3 * b4 * c3 * pow(d2, 2) -
          8 * a2 * b3 * b4 * c5 * pow(d0, 2) -
          4 * a2 * b4 * b5 * c0 * pow(d3, 2) +
          8 * a2 * b4 * b5 * c3 * pow(d0, 2) +
          6 * a3 * b0 * b1 * c5 * pow(d5, 2) -
          4 * a3 * b0 * b2 * c4 * pow(d5, 2) -
          4 * a3 * b0 * b4 * c2 * pow(d5, 2) -
          2 * a3 * b0 * b5 * c1 * pow(d5, 2) -
          8 * a3 * b1 * b3 * c5 * pow(d2, 2) -
          2 * a3 * b1 * b5 * c0 * pow(d5, 2) +
          8 * a3 * b1 * b5 * c3 * pow(d2, 2) +
          4 * a3 * b1 * b5 * c5 * pow(d0, 2) -
          16 * a3 * b2 * b3 * c4 * pow(d2, 2) +
          4 * a3 * b2 * b4 * c0 * pow(d5, 2) -
          16 * a3 * b2 * b4 * c3 * pow(d2, 2) -
          8 * a3 * b2 * b4 * c5 * pow(d0, 2) +
          48 * a3 * b3 * b4 * c2 * pow(d2, 2) -
          8 * a3 * b3 * b5 * c1 * pow(d2, 2) +
          4 * a4 * b0 * b2 * c3 * pow(d5, 2) -
          4 * a4 * b0 * b2 * c5 * pow(d3, 2) -
          4 * a4 * b0 * b3 * c2 * pow(d5, 2) +
          8 * a4 * b0 * b4 * c1 * pow(d5, 2) +
          4 * a4 * b0 * b5 * c2 * pow(d3, 2) -
          8 * a4 * b1 * b4 * c0 * pow(d5, 2) +
          32 * a4 * b1 * b4 * c3 * pow(d2, 2) +
          16 * a4 * b1 * b4 * c5 * pow(d0, 2) -
          8 * a4 * b1 * b5 * c4 * pow(d0, 2) +
          4 * a4 * b2 * b3 * c0 * pow(d5, 2) -
          16 * a4 * b2 * b3 * c3 * pow(d2, 2) -
          8 * a4 * b2 * b3 * c5 * pow(d0, 2) -
          4 * a4 * b2 * b5 * c0 * pow(d3, 2) +
          8 * a4 * b2 * b5 * c3 * pow(d0, 2) -
          32 * a4 * b3 * b4 * c1 * pow(d2, 2) -
          2 * a5 * b0 * b1 * c3 * pow(d5, 2) +
          2 * a5 * b0 * b1 * c5 * pow(d3, 2) +
          4 * a5 * b0 * b2 * c4 * pow(d3, 2) -
          2 * a5 * b0 * b3 * c1 * pow(d5, 2) +
          4 * a5 * b0 * b4 * c2 * pow(d3, 2) -
          6 * a5 * b0 * b5 * c1 * pow(d3, 2) -
          2 * a5 * b1 * b3 * c0 * pow(d5, 2) +
          8 * a5 * b1 * b3 * c3 * pow(d2, 2) +
          4 * a5 * b1 * b3 * c5 * pow(d0, 2) -
          8 * a5 * b1 * b4 * c4 * pow(d0, 2) +
          2 * a5 * b1 * b5 * c0 * pow(d3, 2) -
          4 * a5 * b1 * b5 * c3 * pow(d0, 2) -
          4 * a5 * b2 * b4 * c0 * pow(d3, 2) +
          8 * a5 * b2 * b4 * c3 * pow(d0, 2) -
          4 * a5 * b3 * b5 * c1 * pow(d0, 2) +
          8 * a1 * b3 * b5 * c5 * pow(d1, 2) -
          16 * a2 * b3 * b4 * c5 * pow(d1, 2) +
          8 * a3 * b1 * b5 * c5 * pow(d1, 2) -
          16 * a3 * b2 * b4 * c5 * pow(d1, 2) +
          16 * a3 * b4 * b5 * c2 * pow(d1, 2) -
          16 * a4 * b2 * b3 * c5 * pow(d1, 2) +
          16 * a4 * b3 * b5 * c2 * pow(d1, 2) +
          8 * a5 * b0 * b5 * c1 * pow(d4, 2) +
          8 * a5 * b1 * b3 * c5 * pow(d1, 2) -
          8 * a5 * b1 * b5 * c0 * pow(d4, 2) +
          8 * a5 * b1 * b5 * c3 * pow(d1, 2) +
          16 * a5 * b3 * b4 * c2 * pow(d1, 2) -
          24 * a5 * b3 * b5 * c1 * pow(d1, 2) +
          4 * a2 * b4 * b5 * c5 * pow(d0, 2) +
          4 * a4 * b2 * b5 * c5 * pow(d0, 2) -
          8 * a5 * b1 * b5 * c3 * pow(d2, 2) -
          2 * a5 * b1 * b5 * c5 * pow(d0, 2) +
          4 * a5 * b2 * b4 * c5 * pow(d0, 2) -
          4 * a5 * b2 * b5 * c4 * pow(d0, 2) +
          8 * a5 * b3 * b5 * c1 * pow(d2, 2) -
          4 * a5 * b4 * b5 * c2 * pow(d0, 2) +
          2 * a1 * b3 * b5 * c3 * pow(d5, 2) -
          4 * a2 * b3 * b4 * c3 * pow(d5, 2) +
          16 * a2 * b4 * b5 * c5 * pow(d1, 2) -
          6 * a3 * b1 * b3 * c5 * pow(d5, 2) +
          2 * a3 * b1 * b5 * c3 * pow(d5, 2) +
          4 * a3 * b2 * b3 * c4 * pow(d5, 2) -
          4 * a3 * b2 * b4 * c3 * pow(d5, 2) +
          4 * a3 * b3 * b4 * c2 * pow(d5, 2) +
          2 * a3 * b3 * b5 * c1 * pow(d5, 2) +
          8 * a4 * b1 * b4 * c3 * pow(d5, 2) -
          4 * a4 * b2 * b3 * c3 * pow(d5, 2) +
          16 * a4 * b2 * b5 * c5 * pow(d1, 2) -
          8 * a4 * b3 * b4 * c1 * pow(d5, 2) +
          2 * a5 * b1 * b3 * c3 * pow(d5, 2) -
          24 * a5 * b1 * b5 * c5 * pow(d1, 2) +
          16 * a5 * b2 * b4 * c5 * pow(d1, 2) -
          16 * a5 * b2 * b5 * c4 * pow(d1, 2) -
          16 * a5 * b4 * b5 * c2 * pow(d1, 2) +
          8 * a5 * b1 * b5 * c3 * pow(d4, 2) -
          8 * a5 * b3 * b5 * c1 * pow(d4, 2) +
          4 * a2 * b4 * b5 * c5 * pow(d3, 2) +
          4 * a4 * b2 * b5 * c5 * pow(d3, 2) -
          2 * a5 * b1 * b5 * c5 * pow(d3, 2) +
          4 * a5 * b2 * b4 * c5 * pow(d3, 2) -
          4 * a5 * b2 * b5 * c4 * pow(d3, 2) -
          4 * a5 * b4 * b5 * c2 * pow(d3, 2) -
          2 * a0 * pow(b3, 2) * c0 * d0 * d1 -
          8 * a0 * pow(b1, 2) * c0 * d1 * d3 -
          8 * a0 * pow(b1, 2) * c1 * d0 * d3 -
          8 * a0 * pow(b1, 2) * c3 * d0 * d1 +
          8 * a0 * pow(b4, 2) * c0 * d0 * d1 +
          24 * a1 * pow(b1, 2) * c0 * d0 * d3 -
          8 * a3 * pow(b1, 2) * c0 * d0 * d1 -
          2 * a0 * pow(b5, 2) * c0 * d0 * d1 +
          8 * a1 * pow(b0, 2) * c1 * d1 * d3 +
          8 * a1 * pow(b2, 2) * c0 * d0 * d3 -
          8 * a3 * pow(b2, 2) * c0 * d0 * d1 -
          6 * a0 * pow(b0, 2) * c3 * d1 * d3 +
          8 * a0 * pow(b1, 2) * c0 * d1 * d5 +
          8 * a0 * pow(b1, 2) * c1 * d0 * d5 +
          8 * a0 * pow(b1, 2) * c5 * d0 * d1 +
          2 * a1 * pow(b0, 2) * c3 * d0 * d3 -
          24 * a1 * pow(b1, 2) * c0 * d0 * d5 -
          16 * a2 * pow(b1, 2) * c0 * d0 * d4 +
          2 * a3 * pow(b0, 2) * c0 * d1 * d3 +
          2 * a3 * pow(b0, 2) * c1 * d0 * d3 +
          2 * a3 * pow(b0, 2) * c3 * d0 * d1 +
          16 * a4 * pow(b1, 2) * c0 * d0 * d2 +
          8 * a5 * pow(b1, 2) * c0 * d0 * d1 +
          16 * a0 * pow(b2, 2) * c0 * d2 * d4 +
          16 * a0 * pow(b2, 2) * c2 * d0 * d4 +
          16 * a0 * pow(b2, 2) * c4 * d0 * d2 +
          8 * a0 * pow(b3, 2) * c2 * d1 * d2 -
          8 * a1 * pow(b0, 2) * c1 * d1 * d5 +
          8 * a1 * pow(b0, 2) * c2 * d2 * d3 -
          8 * a1 * pow(b2, 2) * c0 * d0 * d5 +
          32 * a1 * pow(b2, 2) * c1 * d1 * d3 +
          8 * a1 * pow(b3, 2) * c2 * d0 * d2 +
          8 * a1 * pow(b4, 2) * c0 * d0 * d3 +
          16 * a1 * pow(b5, 2) * c1 * d0 * d1 -
          16 * a2 * pow(b0, 2) * c1 * d1 * d4 -
          48 * a2 * pow(b2, 2) * c0 * d0 * d4 -
          8 * a2 * pow(b3, 2) * c0 * d1 * d2 -
          8 * a2 * pow(b3, 2) * c1 * d0 * d2 -
          8 * a2 * pow(b3, 2) * c2 * d0 * d1 -
          8 * a3 * pow(b0, 2) * c2 * d1 * d2 -
          8 * a3 * pow(b4, 2) * c0 * d0 * d1 +
          16 * a4 * pow(b0, 2) * c1 * d1 * d2 +
          16 * a4 * pow(b2, 2) * c0 * d0 * d2 +
          8 * a5 * pow(b2, 2) * c0 * d0 * d1 +
          6 * a0 * pow(b0, 2) * c1 * d3 * d5 -
          12 * a0 * pow(b0, 2) * c2 * d3 * d4 +
          6 * a0 * pow(b0, 2) * c3 * d1 * d5 -
          12 * a0 * pow(b0, 2) * c3 * d2 * d4 +
          24 * a0 * pow(b0, 2) * c4 * d1 * d4 -
          12 * a0 * pow(b0, 2) * c4 * d2 * d3 +
          6 * a0 * pow(b0, 2) * c5 * d1 * d3 -
          8 * a0 * pow(b2, 2) * c3 * d1 * d3 +
          2 * a0 * pow(b3, 2) * c0 * d1 * d5 -
          4 * a0 * pow(b3, 2) * c0 * d2 * d4 +
          2 * a0 * pow(b3, 2) * c1 * d0 * d5 -
          4 * a0 * pow(b3, 2) * c2 * d0 * d4 -
          4 * a0 * pow(b3, 2) * c4 * d0 * d2 +
          2 * a0 * pow(b3, 2) * c5 * d0 * d1 +
          32 * a0 * pow(b4, 2) * c2 * d1 * d2 +
          4 * a0 * pow(b5, 2) * c0 * d1 * d3 +
          4 * a0 * pow(b5, 2) * c1 * d0 * d3 +
          4 * a0 * pow(b5, 2) * c3 * d0 * d1 -
          2 * a1 * pow(b0, 2) * c0 * d3 * d5 -
          2 * a1 * pow(b0, 2) * c3 * d0 * d5 -
          8 * a1 * pow(b0, 2) * c4 * d0 * d4 -
          2 * a1 * pow(b0, 2) * c5 * d0 * d3 +
          96 * a1 * pow(b1, 2) * c2 * d2 * d3 -
          8 * a1 * pow(b2, 2) * c3 * d0 * d3 -
          6 * a1 * pow(b3, 2) * c0 * d0 * d5 -
          32 * a1 * pow(b4, 2) * c2 * d0 * d2 -
          4 * a1 * pow(b5, 2) * c0 * d0 * d3 +
          4 * a2 * pow(b0, 2) * c0 * d3 * d4 +
          4 * a2 * pow(b0, 2) * c3 * d0 * d4 +
          4 * a2 * pow(b0, 2) * c4 * d0 * d3 -
          32 * a2 * pow(b1, 2) * c1 * d2 * d3 -
          32 * a2 * pow(b1, 2) * c2 * d1 * d3 -
          32 * a2 * pow(b1, 2) * c3 * d1 * d2 +
          4 * a2 * pow(b3, 2) * c0 * d0 * d4 -
          2 * a3 * pow(b0, 2) * c0 * d1 * d5 +
          4 * a3 * pow(b0, 2) * c0 * d2 * d4 -
          2 * a3 * pow(b0, 2) * c1 * d0 * d5 +
          4 * a3 * pow(b0, 2) * c2 * d0 * d4 +
          4 * a3 * pow(b0, 2) * c4 * d0 * d2 -
          2 * a3 * pow(b0, 2) * c5 * d0 * d1 -
          32 * a3 * pow(b1, 2) * c2 * d1 * d2 +
          8 * a3 * pow(b2, 2) * c0 * d1 * d3 +
          8 * a3 * pow(b2, 2) * c1 * d0 * d3 +
          8 * a3 * pow(b2, 2) * c3 * d0 * d1 -
          4 * a3 * pow(b5, 2) * c0 * d0 * d1 -
          8 * a4 * pow(b0, 2) * c0 * d1 * d4 +
          4 * a4 * pow(b0, 2) * c0 * d2 * d3 -
          8 * a4 * pow(b0, 2) * c1 * d0 * d4 +
          4 * a4 * pow(b0, 2) * c2 * d0 * d3 +
          4 * a4 * pow(b0, 2) * c3 * d0 * d2 -
          8 * a4 * pow(b0, 2) * c4 * d0 * d1 +
          4 * a4 * pow(b3, 2) * c0 * d0 * d2 -
          2 * a5 * pow(b0, 2) * c0 * d1 * d3 -
          2 * a5 * pow(b0, 2) * c1 * d0 * d3 -
          2 * a5 * pow(b0, 2) * c3 * d0 * d1 +
          2 * a5 * pow(b3, 2) * c0 * d0 * d1 +
          8 * a0 * pow(b1, 2) * c1 * d3 * d5 -
          16 * a0 * pow(b1, 2) * c2 * d3 * d4 +
          8 * a0 * pow(b1, 2) * c3 * d1 * d5 -
          16 * a0 * pow(b1, 2) * c3 * d2 * d4 -
          16 * a0 * pow(b1, 2) * c4 * d2 * d3 +
          8 * a0 * pow(b1, 2) * c5 * d1 * d3 -
          8 * a0 * pow(b4, 2) * c0 * d1 * d5 -
          8 * a0 * pow(b4, 2) * c1 * d0 * d5 -
          8 * a0 * pow(b4, 2) * c5 * d0 * d1 -
          8 * a0 * pow(b5, 2) * c2 * d1 * d2 -
          8 * a1 * pow(b0, 2) * c2 * d2 * d5 -
          24 * a1 * pow(b1, 2) * c0 * d3 * d5 -
          24 * a1 * pow(b1, 2) * c3 * d0 * d5 -
          24 * a1 * pow(b1, 2) * c5 * d0 * d3 -
          32 * a1 * pow(b2, 2) * c1 * d1 * d5 +
          64 * a1 * pow(b2, 2) * c1 * d2 * d4 +
          64 * a1 * pow(b2, 2) * c2 * d1 * d4 +
          64 * a1 * pow(b2, 2) * c4 * d1 * d2 +
          16 * a1 * pow(b4, 2) * c0 * d0 * d5 +
          8 * a1 * pow(b5, 2) * c2 * d0 * d2 -
          16 * a2 * pow(b0, 2) * c2 * d2 * d4 +
          16 * a2 * pow(b1, 2) * c0 * d3 * d4 +
          16 * a2 * pow(b1, 2) * c3 * d0 * d4 +
          16 * a2 * pow(b1, 2) * c4 * d0 * d3 -
          192 * a2 * pow(b2, 2) * c1 * d1 * d4 +
          8 * a3 * pow(b1, 2) * c0 * d1 * d5 +
          8 * a3 * pow(b1, 2) * c1 * d0 * d5 +
          8 * a3 * pow(b1, 2) * c5 * d0 * d1 +
          64 * a4 * pow(b2, 2) * c1 * d1 * d2 +
          8 * a5 * pow(b0, 2) * c2 * d1 * d2 +
          8 * a5 * pow(b1, 2) * c0 * d1 * d3 +
          8 * a5 * pow(b1, 2) * c1 * d0 * d3 +
          8 * a5 * pow(b1, 2) * c3 * d0 * d1 +
          12 * a0 * pow(b0, 2) * c2 * d4 * d5 +
          12 * a0 * pow(b0, 2) * c4 * d2 * d5 -
          6 * a0 * pow(b0, 2) * c5 * d1 * d5 +
          12 * a0 * pow(b0, 2) * c5 * d2 * d4 +
          8 * a0 * pow(b2, 2) * c1 * d3 * d5 -
          16 * a0 * pow(b2, 2) * c2 * d3 * d4 +
          8 * a0 * pow(b2, 2) * c3 * d1 * d5 -
          16 * a0 * pow(b2, 2) * c3 * d2 * d4 +
          32 * a0 * pow(b2, 2) * c4 * d1 * d4 -
          16 * a0 * pow(b2, 2) * c4 * d2 * d3 +
          8 * a0 * pow(b2, 2) * c5 * d1 * d3 -
          2 * a0 * pow(b5, 2) * c0 * d1 * d5 +
          4 * a0 * pow(b5, 2) * c0 * d2 * d4 -
          2 * a0 * pow(b5, 2) * c1 * d0 * d5 +
          4 * a0 * pow(b5, 2) * c2 * d0 * d4 +
          4 * a0 * pow(b5, 2) * c4 * d0 * d2 -
          2 * a0 * pow(b5, 2) * c5 * d0 * d1 +
          2 * a1 * pow(b0, 2) * c5 * d0 * d5 -
          96 * a1 * pow(b1, 2) * c2 * d2 * d5 -
          32 * a1 * pow(b2, 2) * c4 * d0 * d4 -
          2 * a1 * pow(b5, 2) * c0 * d0 * d5 +
          8 * a1 * pow(b5, 2) * c1 * d1 * d3 -
          4 * a2 * pow(b0, 2) * c0 * d4 * d5 -
          4 * a2 * pow(b0, 2) * c4 * d0 * d5 -
          4 * a2 * pow(b0, 2) * c5 * d0 * d4 +
          32 * a2 * pow(b1, 2) * c1 * d2 * d5 +
          32 * a2 * pow(b1, 2) * c2 * d1 * d5 -
          64 * a2 * pow(b1, 2) * c2 * d2 * d4 +
          32 * a2 * pow(b1, 2) * c5 * d1 * d2 +
          48 * a2 * pow(b2, 2) * c0 * d3 * d4 +
          48 * a2 * pow(b2, 2) * c3 * d0 * d4 +
          48 * a2 * pow(b2, 2) * c4 * d0 * d3 -
          4 * a2 * pow(b5, 2) * c0 * d0 * d4 -
          16 * a3 * pow(b2, 2) * c0 * d2 * d4 -
          16 * a3 * pow(b2, 2) * c2 * d0 * d4 -
          16 * a3 * pow(b2, 2) * c4 * d0 * d2 -
          4 * a4 * pow(b0, 2) * c0 * d2 * d5 -
          4 * a4 * pow(b0, 2) * c2 * d0 * d5 -
          4 * a4 * pow(b0, 2) * c5 * d0 * d2 -
          16 * a4 * pow(b2, 2) * c0 * d2 * d3 -
          16 * a4 * pow(b2, 2) * c2 * d0 * d3 -
          16 * a4 * pow(b2, 2) * c3 * d0 * d2 -
          4 * a4 * pow(b5, 2) * c0 * d0 * d2 +
          2 * a5 * pow(b0, 2) * c0 * d1 * d5 -
          4 * a5 * pow(b0, 2) * c0 * d2 * d4 +
          2 * a5 * pow(b0, 2) * c1 * d0 * d5 -
          4 * a5 * pow(b0, 2) * c2 * d0 * d4 -
          4 * a5 * pow(b0, 2) * c4 * d0 * d2 +
          2 * a5 * pow(b0, 2) * c5 * d0 * d1 +
          32 * a5 * pow(b1, 2) * c2 * d1 * d2 -
          8 * a5 * pow(b2, 2) * c0 * d1 * d3 -
          8 * a5 * pow(b2, 2) * c1 * d0 * d3 -
          8 * a5 * pow(b2, 2) * c3 * d0 * d1 +
          6 * a5 * pow(b5, 2) * c0 * d0 * d1 +
          16 * a0 * pow(b1, 2) * c2 * d4 * d5 +
          16 * a0 * pow(b1, 2) * c4 * d2 * d5 -
          16 * a0 * pow(b1, 2) * c5 * d1 * d5 +
          16 * a0 * pow(b1, 2) * c5 * d2 * d4 -
          6 * a0 * pow(b5, 2) * c3 * d1 * d3 -
          2 * a1 * pow(b0, 2) * c3 * d3 * d5 +
          8 * a1 * pow(b0, 2) * c4 * d3 * d4 +
          48 * a1 * pow(b1, 2) * c5 * d0 * d5 +
          32 * a1 * pow(b4, 2) * c2 * d2 * d3 +
          2 * a1 * pow(b5, 2) * c3 * d0 * d3 -
          4 * a2 * pow(b0, 2) * c3 * d3 * d4 -
          2 * a3 * pow(b0, 2) * c1 * d3 * d5 +
          4 * a3 * pow(b0, 2) * c2 * d3 * d4 -
          2 * a3 * pow(b0, 2) * c3 * d1 * d5 +
          4 * a3 * pow(b0, 2) * c3 * d2 * d4 -
          8 * a3 * pow(b0, 2) * c4 * d1 * d4 +
          4 * a3 * pow(b0, 2) * c4 * d2 * d3 -
          2 * a3 * pow(b0, 2) * c5 * d1 * d3 -
          32 * a3 * pow(b4, 2) * c2 * d1 * d2 +
          2 * a3 * pow(b5, 2) * c0 * d1 * d3 +
          2 * a3 * pow(b5, 2) * c1 * d0 * d3 +
          2 * a3 * pow(b5, 2) * c3 * d0 * d1 -
          4 * a4 * pow(b0, 2) * c3 * d2 * d3 -
          16 * a4 * pow(b1, 2) * c0 * d2 * d5 -
          16 * a4 * pow(b1, 2) * c2 * d0 * d5 -
          16 * a4 * pow(b1, 2) * c5 * d0 * d2 +
          6 * a5 * pow(b0, 2) * c3 * d1 * d3 -
          16 * a5 * pow(b1, 2) * c0 * d1 * d5 -
          16 * a5 * pow(b1, 2) * c1 * d0 * d5 -
          16 * a5 * pow(b1, 2) * c5 * d0 * d1 -
          8 * a0 * pow(b2, 2) * c5 * d1 * d5 +
          8 * a1 * pow(b2, 2) * c5 * d0 * d5 -
          8 * a1 * pow(b3, 2) * c2 * d2 * d5 -
          8 * a1 * pow(b4, 2) * c0 * d3 * d5 -
          8 * a1 * pow(b4, 2) * c3 * d0 * d5 -
          8 * a1 * pow(b4, 2) * c5 * d0 * d3 -
          24 * a1 * pow(b5, 2) * c1 * d1 * d5 +
          16 * a1 * pow(b5, 2) * c1 * d2 * d4 +
          16 * a1 * pow(b5, 2) * c2 * d1 * d4 -
          8 * a1 * pow(b5, 2) * c2 * d2 * d3 +
          16 * a1 * pow(b5, 2) * c4 * d1 * d2 +
          8 * a2 * pow(b3, 2) * c1 * d2 * d5 +
          8 * a2 * pow(b3, 2) * c2 * d1 * d5 -
          16 * a2 * pow(b3, 2) * c2 * d2 * d4 +
          8 * a2 * pow(b3, 2) * c5 * d1 * d2 -
          16 * a2 * pow(b5, 2) * c1 * d1 * d4 +
          8 * a3 * pow(b4, 2) * c0 * d1 * d5 +
          8 * a3 * pow(b4, 2) * c1 * d0 * d5 +
          8 * a3 * pow(b4, 2) * c5 * d0 * d1 +
          8 * a3 * pow(b5, 2) * c2 * d1 * d2 -
          16 * a4 * pow(b5, 2) * c1 * d1 * d2 -
          8 * a5 * pow(b3, 2) * c2 * d1 * d2 +
          4 * a0 * pow(b3, 2) * c2 * d4 * d5 +
          4 * a0 * pow(b3, 2) * c4 * d2 * d5 -
          2 * a0 * pow(b3, 2) * c5 * d1 * d5 +
          4 * a0 * pow(b3, 2) * c5 * d2 * d4 +
          2 * a0 * pow(b5, 2) * c1 * d3 * d5 -
          4 * a0 * pow(b5, 2) * c2 * d3 * d4 +
          2 * a0 * pow(b5, 2) * c3 * d1 * d5 -
          4 * a0 * pow(b5, 2) * c3 * d2 * d4 +
          8 * a0 * pow(b5, 2) * c4 * d1 * d4 -
          4 * a0 * pow(b5, 2) * c4 * d2 * d3 +
          2 * a0 * pow(b5, 2) * c5 * d1 * d3 +
          4 * a1 * pow(b0, 2) * c5 * d3 * d5 +
          8 * a1 * pow(b2, 2) * c3 * d3 * d5 +
          32 * a1 * pow(b2, 2) * c4 * d3 * d4 +
          6 * a1 * pow(b3, 2) * c5 * d0 * d5 +
          2 * a1 * pow(b5, 2) * c0 * d3 * d5 +
          2 * a1 * pow(b5, 2) * c3 * d0 * d5 -
          8 * a1 * pow(b5, 2) * c4 * d0 * d4 +
          2 * a1 * pow(b5, 2) * c5 * d0 * d3 -
          48 * a2 * pow(b2, 2) * c3 * d3 * d4 -
          4 * a2 * pow(b3, 2) * c0 * d4 * d5 -
          4 * a2 * pow(b3, 2) * c4 * d0 * d5 -
          4 * a2 * pow(b3, 2) * c5 * d0 * d4 +
          4 * a2 * pow(b5, 2) * c0 * d3 * d4 +
          4 * a2 * pow(b5, 2) * c3 * d0 * d4 +
          4 * a2 * pow(b5, 2) * c4 * d0 * d3 -
          8 * a3 * pow(b0, 2) * c2 * d4 * d5 -
          8 * a3 * pow(b0, 2) * c4 * d2 * d5 +
          4 * a3 * pow(b0, 2) * c5 * d1 * d5 -
          8 * a3 * pow(b0, 2) * c5 * d2 * d4 -
          8 * a3 * pow(b2, 2) * c1 * d3 * d5 +
          16 * a3 * pow(b2, 2) * c2 * d3 * d4 -
          8 * a3 * pow(b2, 2) * c3 * d1 * d5 +
          16 * a3 * pow(b2, 2) * c3 * d2 * d4 -
          32 * a3 * pow(b2, 2) * c4 * d1 * d4 +
          16 * a3 * pow(b2, 2) * c4 * d2 * d3 -
          8 * a3 * pow(b2, 2) * c5 * d1 * d3 +
          2 * a3 * pow(b5, 2) * c0 * d1 * d5 -
          4 * a3 * pow(b5, 2) * c0 * d2 * d4 +
          2 * a3 * pow(b5, 2) * c1 * d0 * d5 -
          4 * a3 * pow(b5, 2) * c2 * d0 * d4 -
          4 * a3 * pow(b5, 2) * c4 * d0 * d2 +
          2 * a3 * pow(b5, 2) * c5 * d0 * d1 +
          8 * a4 * pow(b0, 2) * c1 * d4 * d5 +
          8 * a4 * pow(b0, 2) * c4 * d1 * d5 +
          8 * a4 * pow(b0, 2) * c5 * d1 * d4 +
          16 * a4 * pow(b2, 2) * c3 * d2 * d3 -
          4 * a4 * pow(b3, 2) * c0 * d2 * d5 -
          4 * a4 * pow(b3, 2) * c2 * d0 * d5 -
          4 * a4 * pow(b3, 2) * c5 * d0 * d2 +
          4 * a4 * pow(b5, 2) * c0 * d2 * d3 +
          4 * a4 * pow(b5, 2) * c2 * d0 * d3 +
          4 * a4 * pow(b5, 2) * c3 * d0 * d2 -
          4 * a5 * pow(b0, 2) * c1 * d3 * d5 +
          8 * a5 * pow(b0, 2) * c2 * d3 * d4 -
          4 * a5 * pow(b0, 2) * c3 * d1 * d5 +
          8 * a5 * pow(b0, 2) * c3 * d2 * d4 -
          16 * a5 * pow(b0, 2) * c4 * d1 * d4 +
          8 * a5 * pow(b0, 2) * c4 * d2 * d3 -
          4 * a5 * pow(b0, 2) * c5 * d1 * d3 +
          8 * a5 * pow(b2, 2) * c3 * d1 * d3 -
          2 * a5 * pow(b3, 2) * c0 * d1 * d5 +
          4 * a5 * pow(b3, 2) * c0 * d2 * d4 -
          2 * a5 * pow(b3, 2) * c1 * d0 * d5 +
          4 * a5 * pow(b3, 2) * c2 * d0 * d4 +
          4 * a5 * pow(b3, 2) * c4 * d0 * d2 -
          2 * a5 * pow(b3, 2) * c5 * d0 * d1 -
          6 * a5 * pow(b5, 2) * c0 * d1 * d3 -
          6 * a5 * pow(b5, 2) * c1 * d0 * d3 -
          6 * a5 * pow(b5, 2) * c3 * d0 * d1 +
          8 * a0 * pow(b4, 2) * c5 * d1 * d5 +
          24 * a1 * pow(b1, 2) * c5 * d3 * d5 -
          8 * a1 * pow(b4, 2) * c5 * d0 * d5 -
          16 * a2 * pow(b1, 2) * c3 * d4 * d5 -
          16 * a2 * pow(b1, 2) * c4 * d3 * d5 -
          16 * a2 * pow(b1, 2) * c5 * d3 * d4 -
          8 * a3 * pow(b1, 2) * c5 * d1 * d5 -
          8 * a5 * pow(b1, 2) * c1 * d3 * d5 +
          16 * a5 * pow(b1, 2) * c2 * d3 * d4 -
          8 * a5 * pow(b1, 2) * c3 * d1 * d5 +
          16 * a5 * pow(b1, 2) * c3 * d2 * d4 +
          16 * a5 * pow(b1, 2) * c4 * d2 * d3 -
          8 * a5 * pow(b1, 2) * c5 * d1 * d3 -
          8 * a1 * pow(b2, 2) * c5 * d3 * d5 +
          4 * a2 * pow(b0, 2) * c5 * d4 * d5 +
          8 * a3 * pow(b2, 2) * c5 * d1 * d5 +
          4 * a4 * pow(b0, 2) * c5 * d2 * d5 -
          4 * a5 * pow(b0, 2) * c2 * d4 * d5 -
          4 * a5 * pow(b0, 2) * c4 * d2 * d5 +
          2 * a5 * pow(b0, 2) * c5 * d1 * d5 -
          4 * a5 * pow(b0, 2) * c5 * d2 * d4 -
          2 * a1 * pow(b5, 2) * c3 * d3 * d5 +
          8 * a1 * pow(b5, 2) * c4 * d3 * d4 +
          16 * a2 * pow(b1, 2) * c5 * d4 * d5 -
          4 * a2 * pow(b5, 2) * c3 * d3 * d4 -
          2 * a3 * pow(b5, 2) * c1 * d3 * d5 +
          4 * a3 * pow(b5, 2) * c2 * d3 * d4 -
          2 * a3 * pow(b5, 2) * c3 * d1 * d5 +
          4 * a3 * pow(b5, 2) * c3 * d2 * d4 -
          8 * a3 * pow(b5, 2) * c4 * d1 * d4 +
          4 * a3 * pow(b5, 2) * c4 * d2 * d3 -
          2 * a3 * pow(b5, 2) * c5 * d1 * d3 +
          16 * a4 * pow(b1, 2) * c5 * d2 * d5 -
          4 * a4 * pow(b5, 2) * c3 * d2 * d3 -
          16 * a5 * pow(b1, 2) * c2 * d4 * d5 -
          16 * a5 * pow(b1, 2) * c4 * d2 * d5 +
          24 * a5 * pow(b1, 2) * c5 * d1 * d5 -
          16 * a5 * pow(b1, 2) * c5 * d2 * d4 +
          6 * a5 * pow(b5, 2) * c3 * d1 * d3 +
          8 * a1 * pow(b4, 2) * c5 * d3 * d5 -
          8 * a3 * pow(b4, 2) * c5 * d1 * d5 +
          4 * a2 * pow(b3, 2) * c5 * d4 * d5 +
          4 * a4 * pow(b3, 2) * c5 * d2 * d5 -
          4 * a5 * pow(b3, 2) * c2 * d4 * d5 -
          4 * a5 * pow(b3, 2) * c4 * d2 * d5 +
          2 * a5 * pow(b3, 2) * c5 * d1 * d5 -
          4 * a5 * pow(b3, 2) * c5 * d2 * d4 +
          16 * a0 * b0 * b1 * c1 * d1 * d3 + 16 * a0 * b1 * b3 * c1 * d0 * d1 -
          16 * a1 * b0 * b1 * c0 * d1 * d3 - 16 * a1 * b0 * b1 * c1 * d0 * d3 -
          16 * a1 * b0 * b1 * c3 * d0 * d1 + 16 * a1 * b0 * b3 * c1 * d0 * d1 -
          16 * a1 * b1 * b3 * c0 * d0 * d1 + 16 * a3 * b0 * b1 * c1 * d0 * d1 +
          4 * a0 * b0 * b1 * c3 * d0 * d3 + 4 * a0 * b0 * b3 * c0 * d1 * d3 +
          4 * a0 * b0 * b3 * c1 * d0 * d3 + 4 * a0 * b0 * b3 * c3 * d0 * d1 -
          4 * a0 * b1 * b3 * c0 * d0 * d3 - 4 * a1 * b0 * b3 * c0 * d0 * d3 -
          4 * a3 * b0 * b1 * c0 * d0 * d3 - 4 * a3 * b0 * b3 * c0 * d0 * d1 -
          16 * a0 * b0 * b1 * c1 * d1 * d5 + 16 * a0 * b0 * b1 * c2 * d2 * d3 -
          32 * a0 * b0 * b2 * c1 * d1 * d4 - 16 * a0 * b0 * b3 * c2 * d1 * d2 +
          32 * a0 * b0 * b4 * c1 * d1 * d2 + 16 * a0 * b1 * b2 * c0 * d1 * d4 -
          8 * a0 * b1 * b2 * c0 * d2 * d3 + 16 * a0 * b1 * b2 * c1 * d0 * d4 -
          8 * a0 * b1 * b2 * c2 * d0 * d3 - 8 * a0 * b1 * b2 * c3 * d0 * d2 +
          16 * a0 * b1 * b2 * c4 * d0 * d1 - 16 * a0 * b1 * b4 * c0 * d1 * d2 -
          16 * a0 * b1 * b4 * c1 * d0 * d2 - 16 * a0 * b1 * b4 * c2 * d0 * d1 -
          16 * a0 * b1 * b5 * c1 * d0 * d1 + 8 * a0 * b2 * b3 * c0 * d1 * d2 +
          8 * a0 * b2 * b3 * c1 * d0 * d2 + 8 * a0 * b2 * b3 * c2 * d0 * d1 +
          16 * a1 * b0 * b1 * c0 * d1 * d5 + 16 * a1 * b0 * b1 * c1 * d0 * d5 +
          16 * a1 * b0 * b1 * c5 * d0 * d1 + 16 * a1 * b0 * b2 * c0 * d1 * d4 -
          8 * a1 * b0 * b2 * c0 * d2 * d3 + 16 * a1 * b0 * b2 * c1 * d0 * d4 -
          8 * a1 * b0 * b2 * c2 * d0 * d3 - 8 * a1 * b0 * b2 * c3 * d0 * d2 +
          16 * a1 * b0 * b2 * c4 * d0 * d1 - 16 * a1 * b0 * b4 * c0 * d1 * d2 -
          16 * a1 * b0 * b4 * c1 * d0 * d2 - 16 * a1 * b0 * b4 * c2 * d0 * d1 -
          16 * a1 * b0 * b5 * c1 * d0 * d1 - 32 * a1 * b1 * b2 * c0 * d0 * d4 +
          32 * a1 * b1 * b4 * c0 * d0 * d2 + 16 * a1 * b1 * b5 * c0 * d0 * d1 +
          16 * a2 * b0 * b1 * c0 * d1 * d4 - 8 * a2 * b0 * b1 * c0 * d2 * d3 +
          16 * a2 * b0 * b1 * c1 * d0 * d4 - 8 * a2 * b0 * b1 * c2 * d0 * d3 -
          8 * a2 * b0 * b1 * c3 * d0 * d2 + 16 * a2 * b0 * b1 * c4 * d0 * d1 +
          8 * a2 * b0 * b3 * c0 * d1 * d2 + 8 * a2 * b0 * b3 * c1 * d0 * d2 +
          8 * a2 * b0 * b3 * c2 * d0 * d1 + 16 * a2 * b1 * b2 * c0 * d0 * d3 -
          16 * a2 * b2 * b3 * c0 * d0 * d1 + 8 * a3 * b0 * b2 * c0 * d1 * d2 +
          8 * a3 * b0 * b2 * c1 * d0 * d2 + 8 * a3 * b0 * b2 * c2 * d0 * d1 -
          16 * a4 * b0 * b1 * c0 * d1 * d2 - 16 * a4 * b0 * b1 * c1 * d0 * d2 -
          16 * a4 * b0 * b1 * c2 * d0 * d1 - 16 * a5 * b0 * b1 * c1 * d0 * d1 -
          4 * a0 * b0 * b1 * c0 * d3 * d5 - 4 * a0 * b0 * b1 * c3 * d0 * d5 -
          16 * a0 * b0 * b1 * c4 * d0 * d4 - 4 * a0 * b0 * b1 * c5 * d0 * d3 +
          8 * a0 * b0 * b2 * c0 * d3 * d4 + 8 * a0 * b0 * b2 * c3 * d0 * d4 +
          8 * a0 * b0 * b2 * c4 * d0 * d3 - 4 * a0 * b0 * b3 * c0 * d1 * d5 +
          8 * a0 * b0 * b3 * c0 * d2 * d4 - 4 * a0 * b0 * b3 * c1 * d0 * d5 +
          8 * a0 * b0 * b3 * c2 * d0 * d4 + 8 * a0 * b0 * b3 * c4 * d0 * d2 -
          4 * a0 * b0 * b3 * c5 * d0 * d1 - 16 * a0 * b0 * b4 * c0 * d1 * d4 +
          8 * a0 * b0 * b4 * c0 * d2 * d3 - 16 * a0 * b0 * b4 * c1 * d0 * d4 +
          8 * a0 * b0 * b4 * c2 * d0 * d3 + 8 * a0 * b0 * b4 * c3 * d0 * d2 -
          16 * a0 * b0 * b4 * c4 * d0 * d1 - 4 * a0 * b0 * b5 * c0 * d1 * d3 -
          4 * a0 * b0 * b5 * c1 * d0 * d3 - 4 * a0 * b0 * b5 * c3 * d0 * d1 +
          4 * a0 * b1 * b3 * c0 * d0 * d5 + 16 * a0 * b1 * b4 * c0 * d0 * d4 +
          4 * a0 * b1 * b5 * c0 * d0 * d3 - 8 * a0 * b2 * b3 * c0 * d0 * d4 -
          8 * a0 * b2 * b4 * c0 * d0 * d3 - 8 * a0 * b3 * b4 * c0 * d0 * d2 +
          4 * a0 * b3 * b5 * c0 * d0 * d1 + 4 * a1 * b0 * b3 * c0 * d0 * d5 +
          16 * a1 * b0 * b4 * c0 * d0 * d4 + 4 * a1 * b0 * b5 * c0 * d0 * d3 -
          8 * a2 * b0 * b3 * c0 * d0 * d4 - 8 * a2 * b0 * b4 * c0 * d0 * d3 +
          4 * a3 * b0 * b1 * c0 * d0 * d5 - 8 * a3 * b0 * b2 * c0 * d0 * d4 -
          8 * a3 * b0 * b4 * c0 * d0 * d2 + 4 * a3 * b0 * b5 * c0 * d0 * d1 +
          16 * a4 * b0 * b1 * c0 * d0 * d4 - 8 * a4 * b0 * b2 * c0 * d0 * d3 -
          8 * a4 * b0 * b3 * c0 * d0 * d2 + 16 * a4 * b0 * b4 * c0 * d0 * d1 +
          4 * a5 * b0 * b1 * c0 * d0 * d3 + 4 * a5 * b0 * b3 * c0 * d0 * d1 -
          16 * a0 * b0 * b1 * c2 * d2 * d5 - 32 * a0 * b0 * b2 * c2 * d2 * d4 +
          16 * a0 * b0 * b5 * c2 * d1 * d2 + 8 * a0 * b1 * b2 * c0 * d2 * d5 +
          8 * a0 * b1 * b2 * c2 * d0 * d5 + 8 * a0 * b1 * b2 * c5 * d0 * d2 -
          32 * a0 * b2 * b4 * c2 * d0 * d2 - 8 * a0 * b2 * b5 * c0 * d1 * d2 -
          8 * a0 * b2 * b5 * c1 * d0 * d2 - 8 * a0 * b2 * b5 * c2 * d0 * d1 +
          8 * a1 * b0 * b2 * c0 * d2 * d5 + 8 * a1 * b0 * b2 * c2 * d0 * d5 +
          8 * a1 * b0 * b2 * c5 * d0 * d2 - 64 * a1 * b1 * b2 * c1 * d2 * d3 -
          64 * a1 * b1 * b2 * c2 * d1 * d3 - 64 * a1 * b1 * b2 * c3 * d1 * d2 -
          64 * a1 * b1 * b3 * c2 * d1 * d2 + 64 * a1 * b2 * b3 * c1 * d1 * d2 +
          8 * a2 * b0 * b1 * c0 * d2 * d5 + 8 * a2 * b0 * b1 * c2 * d0 * d5 +
          8 * a2 * b0 * b1 * c5 * d0 * d2 + 32 * a2 * b0 * b2 * c0 * d2 * d4 +
          32 * a2 * b0 * b2 * c2 * d0 * d4 + 32 * a2 * b0 * b2 * c4 * d0 * d2 -
          32 * a2 * b0 * b4 * c2 * d0 * d2 - 8 * a2 * b0 * b5 * c0 * d1 * d2 -
          8 * a2 * b0 * b5 * c1 * d0 * d2 - 8 * a2 * b0 * b5 * c2 * d0 * d1 -
          16 * a2 * b1 * b2 * c0 * d0 * d5 + 64 * a2 * b1 * b2 * c1 * d1 * d3 +
          64 * a2 * b1 * b3 * c1 * d1 * d2 + 32 * a2 * b2 * b4 * c0 * d0 * d2 +
          16 * a2 * b2 * b5 * c0 * d0 * d1 + 64 * a3 * b1 * b2 * c1 * d1 * d2 -
          32 * a4 * b0 * b2 * c2 * d0 * d2 - 8 * a5 * b0 * b2 * c0 * d1 * d2 -
          8 * a5 * b0 * b2 * c1 * d0 * d2 - 8 * a5 * b0 * b2 * c2 * d0 * d1 +
          4 * a0 * b0 * b1 * c5 * d0 * d5 - 8 * a0 * b0 * b2 * c0 * d4 * d5 -
          8 * a0 * b0 * b2 * c4 * d0 * d5 - 8 * a0 * b0 * b2 * c5 * d0 * d4 -
          8 * a0 * b0 * b4 * c0 * d2 * d5 - 8 * a0 * b0 * b4 * c2 * d0 * d5 -
          8 * a0 * b0 * b4 * c5 * d0 * d2 + 4 * a0 * b0 * b5 * c0 * d1 * d5 -
          8 * a0 * b0 * b5 * c0 * d2 * d4 + 4 * a0 * b0 * b5 * c1 * d0 * d5 -
          8 * a0 * b0 * b5 * c2 * d0 * d4 - 8 * a0 * b0 * b5 * c4 * d0 * d2 +
          4 * a0 * b0 * b5 * c5 * d0 * d1 + 16 * a0 * b1 * b2 * c3 * d2 * d3 -
          16 * a0 * b1 * b3 * c1 * d1 * d5 + 16 * a0 * b1 * b3 * c1 * d2 * d4 +
          16 * a0 * b1 * b3 * c2 * d1 * d4 - 16 * a0 * b1 * b3 * c2 * d2 * d3 +
          16 * a0 * b1 * b3 * c4 * d1 * d2 + 16 * a0 * b1 * b4 * c1 * d2 * d3 +
          16 * a0 * b1 * b4 * c2 * d1 * d3 + 16 * a0 * b1 * b4 * c3 * d1 * d2 -
          4 * a0 * b1 * b5 * c0 * d0 * d5 - 16 * a0 * b1 * b5 * c1 * d1 * d3 +
          8 * a0 * b2 * b4 * c0 * d0 * d5 + 8 * a0 * b2 * b5 * c0 * d0 * d4 -
          32 * a0 * b3 * b4 * c1 * d1 * d2 + 8 * a0 * b4 * b5 * c0 * d0 * d2 +
          16 * a1 * b0 * b1 * c1 * d3 * d5 - 32 * a1 * b0 * b1 * c2 * d3 * d4 +
          16 * a1 * b0 * b1 * c3 * d1 * d5 - 32 * a1 * b0 * b1 * c3 * d2 * d4 -
          32 * a1 * b0 * b1 * c4 * d2 * d3 + 16 * a1 * b0 * b1 * c5 * d1 * d3 +
          16 * a1 * b0 * b2 * c3 * d2 * d3 - 16 * a1 * b0 * b3 * c1 * d1 * d5 +
          16 * a1 * b0 * b3 * c1 * d2 * d4 + 16 * a1 * b0 * b3 * c2 * d1 * d4 -
          16 * a1 * b0 * b3 * c2 * d2 * d3 + 16 * a1 * b0 * b3 * c4 * d1 * d2 +
          16 * a1 * b0 * b4 * c1 * d2 * d3 + 16 * a1 * b0 * b4 * c2 * d1 * d3 +
          16 * a1 * b0 * b4 * c3 * d1 * d2 - 4 * a1 * b0 * b5 * c0 * d0 * d5 -
          16 * a1 * b0 * b5 * c1 * d1 * d3 + 32 * a1 * b1 * b2 * c0 * d3 * d4 +
          32 * a1 * b1 * b2 * c3 * d0 * d4 + 32 * a1 * b1 * b2 * c4 * d0 * d3 +
          16 * a1 * b1 * b3 * c0 * d1 * d5 + 16 * a1 * b1 * b3 * c1 * d0 * d5 +
          16 * a1 * b1 * b3 * c5 * d0 * d1 + 16 * a1 * b1 * b5 * c0 * d1 * d3 +
          16 * a1 * b1 * b5 * c1 * d0 * d3 + 16 * a1 * b1 * b5 * c3 * d0 * d1 -
          16 * a1 * b2 * b3 * c0 * d1 * d4 - 16 * a1 * b2 * b3 * c1 * d0 * d4 -
          16 * a1 * b2 * b3 * c4 * d0 * d1 - 16 * a1 * b2 * b4 * c0 * d1 * d3 -
          16 * a1 * b2 * b4 * c1 * d0 * d3 - 16 * a1 * b2 * b4 * c3 * d0 * d1 -
          16 * a1 * b3 * b5 * c1 * d0 * d1 + 16 * a2 * b0 * b1 * c3 * d2 * d3 -
          16 * a2 * b0 * b2 * c3 * d1 * d3 + 8 * a2 * b0 * b4 * c0 * d0 * d5 +
          8 * a2 * b0 * b5 * c0 * d0 * d4 - 16 * a2 * b1 * b2 * c3 * d0 * d3 -
          16 * a2 * b1 * b3 * c0 * d1 * d4 - 16 * a2 * b1 * b3 * c1 * d0 * d4 -
          16 * a2 * b1 * b3 * c4 * d0 * d1 - 16 * a2 * b1 * b4 * c0 * d1 * d3 -
          16 * a2 * b1 * b4 * c1 * d0 * d3 - 16 * a2 * b1 * b4 * c3 * d0 * d1 +
          16 * a2 * b2 * b3 * c0 * d1 * d3 + 16 * a2 * b2 * b3 * c1 * d0 * d3 +
          16 * a2 * b2 * b3 * c3 * d0 * d1 + 32 * a2 * b3 * b4 * c1 * d0 * d1 -
          16 * a3 * b0 * b1 * c1 * d1 * d5 + 16 * a3 * b0 * b1 * c1 * d2 * d4 +
          16 * a3 * b0 * b1 * c2 * d1 * d4 - 16 * a3 * b0 * b1 * c2 * d2 * d3 +
          16 * a3 * b0 * b1 * c4 * d1 * d2 + 16 * a3 * b0 * b3 * c2 * d1 * d2 -
          32 * a3 * b0 * b4 * c1 * d1 * d2 - 16 * a3 * b1 * b2 * c0 * d1 * d4 -
          16 * a3 * b1 * b2 * c1 * d0 * d4 - 16 * a3 * b1 * b2 * c4 * d0 * d1 +
          16 * a3 * b1 * b3 * c2 * d0 * d2 - 16 * a3 * b1 * b5 * c1 * d0 * d1 -
          16 * a3 * b2 * b3 * c0 * d1 * d2 - 16 * a3 * b2 * b3 * c1 * d0 * d2 -
          16 * a3 * b2 * b3 * c2 * d0 * d1 + 32 * a3 * b2 * b4 * c1 * d0 * d1 +
          16 * a4 * b0 * b1 * c1 * d2 * d3 + 16 * a4 * b0 * b1 * c2 * d1 * d3 +
          16 * a4 * b0 * b1 * c3 * d1 * d2 + 8 * a4 * b0 * b2 * c0 * d0 * d5 -
          32 * a4 * b0 * b3 * c1 * d1 * d2 + 8 * a4 * b0 * b5 * c0 * d0 * d2 -
          16 * a4 * b1 * b2 * c0 * d1 * d3 - 16 * a4 * b1 * b2 * c1 * d0 * d3 -
          16 * a4 * b1 * b2 * c3 * d0 * d1 + 32 * a4 * b2 * b3 * c1 * d0 * d1 -
          4 * a5 * b0 * b1 * c0 * d0 * d5 - 16 * a5 * b0 * b1 * c1 * d1 * d3 +
          8 * a5 * b0 * b2 * c0 * d0 * d4 + 8 * a5 * b0 * b4 * c0 * d0 * d2 -
          4 * a5 * b0 * b5 * c0 * d0 * d1 - 16 * a5 * b1 * b3 * c1 * d0 * d1 -
          4 * a0 * b0 * b1 * c3 * d3 * d5 + 16 * a0 * b0 * b1 * c4 * d3 * d4 -
          8 * a0 * b0 * b2 * c3 * d3 * d4 - 4 * a0 * b0 * b3 * c1 * d3 * d5 +
          8 * a0 * b0 * b3 * c2 * d3 * d4 - 4 * a0 * b0 * b3 * c3 * d1 * d5 +
          8 * a0 * b0 * b3 * c3 * d2 * d4 - 16 * a0 * b0 * b3 * c4 * d1 * d4 +
          8 * a0 * b0 * b3 * c4 * d2 * d3 - 4 * a0 * b0 * b3 * c5 * d1 * d3 -
          8 * a0 * b0 * b4 * c3 * d2 * d3 + 12 * a0 * b0 * b5 * c3 * d1 * d3 +
          4 * a0 * b1 * b3 * c0 * d3 * d5 + 4 * a0 * b1 * b3 * c3 * d0 * d5 +
          4 * a0 * b1 * b3 * c5 * d0 * d3 - 8 * a0 * b1 * b4 * c0 * d3 * d4 -
          8 * a0 * b1 * b4 * c3 * d0 * d4 - 8 * a0 * b1 * b4 * c4 * d0 * d3 -
          4 * a0 * b1 * b5 * c3 * d0 * d3 + 8 * a0 * b2 * b4 * c3 * d0 * d3 +
          8 * a0 * b3 * b4 * c0 * d1 * d4 + 8 * a0 * b3 * b4 * c1 * d0 * d4 +
          8 * a0 * b3 * b4 * c4 * d0 * d1 - 4 * a0 * b3 * b5 * c0 * d1 * d3 -
          4 * a0 * b3 * b5 * c1 * d0 * d3 - 4 * a0 * b3 * b5 * c3 * d0 * d1 +
          4 * a1 * b0 * b3 * c0 * d3 * d5 + 4 * a1 * b0 * b3 * c3 * d0 * d5 +
          4 * a1 * b0 * b3 * c5 * d0 * d3 - 8 * a1 * b0 * b4 * c0 * d3 * d4 -
          8 * a1 * b0 * b4 * c3 * d0 * d4 - 8 * a1 * b0 * b4 * c4 * d0 * d3 -
          4 * a1 * b0 * b5 * c3 * d0 * d3 + 64 * a1 * b1 * b2 * c1 * d2 * d5 +
          64 * a1 * b1 * b2 * c2 * d1 * d5 - 128 * a1 * b1 * b2 * c2 * d2 * d4 +
          64 * a1 * b1 * b2 * c5 * d1 * d2 + 64 * a1 * b1 * b5 * c2 * d1 * d2 -
          128 * a1 * b2 * b4 * c2 * d1 * d2 - 64 * a1 * b2 * b5 * c1 * d1 * d2 +
          4 * a1 * b3 * b5 * c0 * d0 * d3 + 8 * a2 * b0 * b4 * c3 * d0 * d3 -
          64 * a2 * b1 * b2 * c1 * d1 * d5 + 128 * a2 * b1 * b2 * c1 * d2 * d4 +
          128 * a2 * b1 * b2 * c2 * d1 * d4 +
          128 * a2 * b1 * b2 * c4 * d1 * d2 -
          128 * a2 * b1 * b4 * c2 * d1 * d2 - 64 * a2 * b1 * b5 * c1 * d1 * d2 +
          128 * a2 * b2 * b4 * c1 * d1 * d2 - 8 * a2 * b3 * b4 * c0 * d0 * d3 +
          4 * a3 * b0 * b1 * c0 * d3 * d5 + 4 * a3 * b0 * b1 * c3 * d0 * d5 +
          4 * a3 * b0 * b1 * c5 * d0 * d3 + 4 * a3 * b0 * b3 * c0 * d1 * d5 -
          8 * a3 * b0 * b3 * c0 * d2 * d4 + 4 * a3 * b0 * b3 * c1 * d0 * d5 -
          8 * a3 * b0 * b3 * c2 * d0 * d4 - 8 * a3 * b0 * b3 * c4 * d0 * d2 +
          4 * a3 * b0 * b3 * c5 * d0 * d1 + 8 * a3 * b0 * b4 * c0 * d1 * d4 +
          8 * a3 * b0 * b4 * c1 * d0 * d4 + 8 * a3 * b0 * b4 * c4 * d0 * d1 -
          4 * a3 * b0 * b5 * c0 * d1 * d3 - 4 * a3 * b0 * b5 * c1 * d0 * d3 -
          4 * a3 * b0 * b5 * c3 * d0 * d1 - 12 * a3 * b1 * b3 * c0 * d0 * d5 +
          4 * a3 * b1 * b5 * c0 * d0 * d3 + 8 * a3 * b2 * b3 * c0 * d0 * d4 -
          8 * a3 * b2 * b4 * c0 * d0 * d3 + 8 * a3 * b3 * b4 * c0 * d0 * d2 +
          4 * a3 * b3 * b5 * c0 * d0 * d1 - 8 * a4 * b0 * b1 * c0 * d3 * d4 -
          8 * a4 * b0 * b1 * c3 * d0 * d4 - 8 * a4 * b0 * b1 * c4 * d0 * d3 +
          8 * a4 * b0 * b2 * c3 * d0 * d3 + 8 * a4 * b0 * b3 * c0 * d1 * d4 +
          8 * a4 * b0 * b3 * c1 * d0 * d4 + 8 * a4 * b0 * b3 * c4 * d0 * d1 -
          128 * a4 * b1 * b2 * c2 * d1 * d2 + 16 * a4 * b1 * b4 * c0 * d0 * d3 -
          8 * a4 * b2 * b3 * c0 * d0 * d3 - 16 * a4 * b3 * b4 * c0 * d0 * d1 -
          4 * a5 * b0 * b1 * c3 * d0 * d3 - 4 * a5 * b0 * b3 * c0 * d1 * d3 -
          4 * a5 * b0 * b3 * c1 * d0 * d3 - 4 * a5 * b0 * b3 * c3 * d0 * d1 -
          64 * a5 * b1 * b2 * c1 * d1 * d2 + 4 * a5 * b1 * b3 * c0 * d0 * d3 -
          16 * a0 * b1 * b2 * c1 * d4 * d5 - 8 * a0 * b1 * b2 * c2 * d3 * d5 -
          8 * a0 * b1 * b2 * c3 * d2 * d5 - 16 * a0 * b1 * b2 * c4 * d1 * d5 -
          16 * a0 * b1 * b2 * c5 * d1 * d4 - 8 * a0 * b1 * b2 * c5 * d2 * d3 +
          16 * a0 * b1 * b3 * c2 * d2 * d5 + 32 * a0 * b1 * b5 * c1 * d1 * d5 -
          16 * a0 * b1 * b5 * c1 * d2 * d4 - 16 * a0 * b1 * b5 * c2 * d1 * d4 -
          16 * a0 * b1 * b5 * c4 * d1 * d2 - 8 * a0 * b2 * b3 * c1 * d2 * d5 -
          8 * a0 * b2 * b3 * c2 * d1 * d5 + 32 * a0 * b2 * b3 * c2 * d2 * d4 -
          8 * a0 * b2 * b3 * c5 * d1 * d2 - 32 * a0 * b2 * b4 * c1 * d2 * d4 -
          32 * a0 * b2 * b4 * c2 * d1 * d4 + 32 * a0 * b2 * b4 * c2 * d2 * d3 -
          32 * a0 * b2 * b4 * c4 * d1 * d2 + 32 * a0 * b2 * b5 * c1 * d1 * d4 +
          32 * a1 * b0 * b1 * c2 * d4 * d5 + 32 * a1 * b0 * b1 * c4 * d2 * d5 -
          32 * a1 * b0 * b1 * c5 * d1 * d5 + 32 * a1 * b0 * b1 * c5 * d2 * d4 -
          16 * a1 * b0 * b2 * c1 * d4 * d5 - 8 * a1 * b0 * b2 * c2 * d3 * d5 -
          8 * a1 * b0 * b2 * c3 * d2 * d5 - 16 * a1 * b0 * b2 * c4 * d1 * d5 -
          16 * a1 * b0 * b2 * c5 * d1 * d4 - 8 * a1 * b0 * b2 * c5 * d2 * d3 +
          16 * a1 * b0 * b3 * c2 * d2 * d5 + 32 * a1 * b0 * b5 * c1 * d1 * d5 -
          16 * a1 * b0 * b5 * c1 * d2 * d4 - 16 * a1 * b0 * b5 * c2 * d1 * d4 -
          16 * a1 * b0 * b5 * c4 * d1 * d2 - 32 * a1 * b1 * b4 * c0 * d2 * d5 -
          32 * a1 * b1 * b4 * c2 * d0 * d5 - 32 * a1 * b1 * b4 * c5 * d0 * d2 -
          32 * a1 * b1 * b5 * c0 * d1 * d5 - 32 * a1 * b1 * b5 * c1 * d0 * d5 -
          32 * a1 * b1 * b5 * c5 * d0 * d1 + 16 * a1 * b2 * b4 * c0 * d1 * d5 +
          32 * a1 * b2 * b4 * c0 * d2 * d4 + 16 * a1 * b2 * b4 * c1 * d0 * d5 +
          32 * a1 * b2 * b4 * c2 * d0 * d4 + 32 * a1 * b2 * b4 * c4 * d0 * d2 +
          16 * a1 * b2 * b4 * c5 * d0 * d1 + 8 * a1 * b2 * b5 * c0 * d2 * d3 +
          8 * a1 * b2 * b5 * c2 * d0 * d3 + 8 * a1 * b2 * b5 * c3 * d0 * d2 -
          16 * a1 * b3 * b5 * c2 * d0 * d2 + 16 * a1 * b4 * b5 * c0 * d1 * d2 +
          16 * a1 * b4 * b5 * c1 * d0 * d2 + 16 * a1 * b4 * b5 * c2 * d0 * d1 -
          16 * a2 * b0 * b1 * c1 * d4 * d5 - 8 * a2 * b0 * b1 * c2 * d3 * d5 -
          8 * a2 * b0 * b1 * c3 * d2 * d5 - 16 * a2 * b0 * b1 * c4 * d1 * d5 -
          16 * a2 * b0 * b1 * c5 * d1 * d4 - 8 * a2 * b0 * b1 * c5 * d2 * d3 +
          16 * a2 * b0 * b2 * c1 * d3 * d5 - 32 * a2 * b0 * b2 * c2 * d3 * d4 +
          16 * a2 * b0 * b2 * c3 * d1 * d5 - 32 * a2 * b0 * b2 * c3 * d2 * d4 +
          64 * a2 * b0 * b2 * c4 * d1 * d4 - 32 * a2 * b0 * b2 * c4 * d2 * d3 +
          16 * a2 * b0 * b2 * c5 * d1 * d3 - 8 * a2 * b0 * b3 * c1 * d2 * d5 -
          8 * a2 * b0 * b3 * c2 * d1 * d5 + 32 * a2 * b0 * b3 * c2 * d2 * d4 -
          8 * a2 * b0 * b3 * c5 * d1 * d2 - 32 * a2 * b0 * b4 * c1 * d2 * d4 -
          32 * a2 * b0 * b4 * c2 * d1 * d4 + 32 * a2 * b0 * b4 * c2 * d2 * d3 -
          32 * a2 * b0 * b4 * c4 * d1 * d2 + 32 * a2 * b0 * b5 * c1 * d1 * d4 -
          64 * a2 * b1 * b2 * c4 * d0 * d4 + 16 * a2 * b1 * b4 * c0 * d1 * d5 +
          32 * a2 * b1 * b4 * c0 * d2 * d4 + 16 * a2 * b1 * b4 * c1 * d0 * d5 +
          32 * a2 * b1 * b4 * c2 * d0 * d4 + 32 * a2 * b1 * b4 * c4 * d0 * d2 +
          16 * a2 * b1 * b4 * c5 * d0 * d1 + 8 * a2 * b1 * b5 * c0 * d2 * d3 +
          8 * a2 * b1 * b5 * c2 * d0 * d3 + 8 * a2 * b1 * b5 * c3 * d0 * d2 -
          32 * a2 * b2 * b3 * c0 * d2 * d4 - 32 * a2 * b2 * b3 * c2 * d0 * d4 -
          32 * a2 * b2 * b3 * c4 * d0 * d2 - 32 * a2 * b2 * b4 * c0 * d2 * d3 -
          32 * a2 * b2 * b4 * c2 * d0 * d3 - 32 * a2 * b2 * b4 * c3 * d0 * d2 -
          16 * a2 * b2 * b5 * c0 * d1 * d3 - 16 * a2 * b2 * b5 * c1 * d0 * d3 -
          16 * a2 * b2 * b5 * c3 * d0 * d1 + 32 * a2 * b3 * b4 * c2 * d0 * d2 +
          8 * a2 * b3 * b5 * c0 * d1 * d2 + 8 * a2 * b3 * b5 * c1 * d0 * d2 +
          8 * a2 * b3 * b5 * c2 * d0 * d1 - 32 * a2 * b4 * b5 * c1 * d0 * d1 +
          16 * a3 * b0 * b1 * c2 * d2 * d5 - 8 * a3 * b0 * b2 * c1 * d2 * d5 -
          8 * a3 * b0 * b2 * c2 * d1 * d5 + 32 * a3 * b0 * b2 * c2 * d2 * d4 -
          8 * a3 * b0 * b2 * c5 * d1 * d2 - 16 * a3 * b1 * b5 * c2 * d0 * d2 +
          32 * a3 * b2 * b4 * c2 * d0 * d2 + 8 * a3 * b2 * b5 * c0 * d1 * d2 +
          8 * a3 * b2 * b5 * c1 * d0 * d2 + 8 * a3 * b2 * b5 * c2 * d0 * d1 -
          32 * a4 * b0 * b2 * c1 * d2 * d4 - 32 * a4 * b0 * b2 * c2 * d1 * d4 +
          32 * a4 * b0 * b2 * c2 * d2 * d3 - 32 * a4 * b0 * b2 * c4 * d1 * d2 +
          64 * a4 * b0 * b4 * c2 * d1 * d2 + 16 * a4 * b1 * b2 * c0 * d1 * d5 +
          32 * a4 * b1 * b2 * c0 * d2 * d4 + 16 * a4 * b1 * b2 * c1 * d0 * d5 +
          32 * a4 * b1 * b2 * c2 * d0 * d4 + 32 * a4 * b1 * b2 * c4 * d0 * d2 +
          16 * a4 * b1 * b2 * c5 * d0 * d1 - 64 * a4 * b1 * b4 * c2 * d0 * d2 +
          16 * a4 * b1 * b5 * c0 * d1 * d2 + 16 * a4 * b1 * b5 * c1 * d0 * d2 +
          16 * a4 * b1 * b5 * c2 * d0 * d1 + 32 * a4 * b2 * b3 * c2 * d0 * d2 -
          32 * a4 * b2 * b5 * c1 * d0 * d1 + 32 * a5 * b0 * b1 * c1 * d1 * d5 -
          16 * a5 * b0 * b1 * c1 * d2 * d4 - 16 * a5 * b0 * b1 * c2 * d1 * d4 -
          16 * a5 * b0 * b1 * c4 * d1 * d2 + 32 * a5 * b0 * b2 * c1 * d1 * d4 +
          8 * a5 * b1 * b2 * c0 * d2 * d3 + 8 * a5 * b1 * b2 * c2 * d0 * d3 +
          8 * a5 * b1 * b2 * c3 * d0 * d2 - 16 * a5 * b1 * b3 * c2 * d0 * d2 +
          16 * a5 * b1 * b4 * c0 * d1 * d2 + 16 * a5 * b1 * b4 * c1 * d0 * d2 +
          16 * a5 * b1 * b4 * c2 * d0 * d1 + 32 * a5 * b1 * b5 * c1 * d0 * d1 +
          8 * a5 * b2 * b3 * c0 * d1 * d2 + 8 * a5 * b2 * b3 * c1 * d0 * d2 +
          8 * a5 * b2 * b3 * c2 * d0 * d1 - 32 * a5 * b2 * b4 * c1 * d0 * d1 +
          8 * a0 * b0 * b1 * c5 * d3 * d5 - 16 * a0 * b0 * b3 * c2 * d4 * d5 -
          16 * a0 * b0 * b3 * c4 * d2 * d5 + 8 * a0 * b0 * b3 * c5 * d1 * d5 -
          16 * a0 * b0 * b3 * c5 * d2 * d4 + 16 * a0 * b0 * b4 * c1 * d4 * d5 +
          16 * a0 * b0 * b4 * c4 * d1 * d5 + 16 * a0 * b0 * b4 * c5 * d1 * d4 -
          8 * a0 * b0 * b5 * c1 * d3 * d5 + 16 * a0 * b0 * b5 * c2 * d3 * d4 -
          8 * a0 * b0 * b5 * c3 * d1 * d5 + 16 * a0 * b0 * b5 * c3 * d2 * d4 -
          32 * a0 * b0 * b5 * c4 * d1 * d4 + 16 * a0 * b0 * b5 * c4 * d2 * d3 -
          8 * a0 * b0 * b5 * c5 * d1 * d3 - 8 * a0 * b1 * b3 * c5 * d0 * d5 -
          8 * a0 * b1 * b4 * c0 * d4 * d5 - 8 * a0 * b1 * b4 * c4 * d0 * d5 -
          8 * a0 * b1 * b4 * c5 * d0 * d4 + 16 * a0 * b1 * b5 * c4 * d0 * d4 +
          8 * a0 * b2 * b3 * c0 * d4 * d5 + 8 * a0 * b2 * b3 * c4 * d0 * d5 +
          8 * a0 * b2 * b3 * c5 * d0 * d4 - 8 * a0 * b2 * b5 * c0 * d3 * d4 -
          8 * a0 * b2 * b5 * c3 * d0 * d4 - 8 * a0 * b2 * b5 * c4 * d0 * d3 +
          8 * a0 * b3 * b4 * c0 * d2 * d5 + 8 * a0 * b3 * b4 * c2 * d0 * d5 +
          8 * a0 * b3 * b4 * c5 * d0 * d2 + 8 * a0 * b4 * b5 * c0 * d1 * d4 -
          8 * a0 * b4 * b5 * c0 * d2 * d3 + 8 * a0 * b4 * b5 * c1 * d0 * d4 -
          8 * a0 * b4 * b5 * c2 * d0 * d3 - 8 * a0 * b4 * b5 * c3 * d0 * d2 +
          8 * a0 * b4 * b5 * c4 * d0 * d1 - 8 * a1 * b0 * b3 * c5 * d0 * d5 -
          8 * a1 * b0 * b4 * c0 * d4 * d5 - 8 * a1 * b0 * b4 * c4 * d0 * d5 -
          8 * a1 * b0 * b4 * c5 * d0 * d4 + 16 * a1 * b0 * b5 * c4 * d0 * d4 +
          8 * a1 * b3 * b5 * c0 * d0 * d5 - 16 * a1 * b4 * b5 * c0 * d0 * d4 +
          8 * a2 * b0 * b3 * c0 * d4 * d5 + 8 * a2 * b0 * b3 * c4 * d0 * d5 +
          8 * a2 * b0 * b3 * c5 * d0 * d4 - 8 * a2 * b0 * b5 * c0 * d3 * d4 -
          8 * a2 * b0 * b5 * c3 * d0 * d4 - 8 * a2 * b0 * b5 * c4 * d0 * d3 -
          16 * a2 * b3 * b4 * c0 * d0 * d5 + 16 * a2 * b4 * b5 * c0 * d0 * d3 -
          8 * a3 * b0 * b1 * c5 * d0 * d5 + 8 * a3 * b0 * b2 * c0 * d4 * d5 +
          8 * a3 * b0 * b2 * c4 * d0 * d5 + 8 * a3 * b0 * b2 * c5 * d0 * d4 +
          8 * a3 * b0 * b4 * c0 * d2 * d5 + 8 * a3 * b0 * b4 * c2 * d0 * d5 +
          8 * a3 * b0 * b4 * c5 * d0 * d2 + 8 * a3 * b1 * b5 * c0 * d0 * d5 -
          16 * a3 * b2 * b4 * c0 * d0 * d5 - 8 * a4 * b0 * b1 * c0 * d4 * d5 -
          8 * a4 * b0 * b1 * c4 * d0 * d5 - 8 * a4 * b0 * b1 * c5 * d0 * d4 +
          8 * a4 * b0 * b3 * c0 * d2 * d5 + 8 * a4 * b0 * b3 * c2 * d0 * d5 +
          8 * a4 * b0 * b3 * c5 * d0 * d2 - 16 * a4 * b0 * b4 * c0 * d1 * d5 -
          16 * a4 * b0 * b4 * c1 * d0 * d5 - 16 * a4 * b0 * b4 * c5 * d0 * d1 +
          8 * a4 * b0 * b5 * c0 * d1 * d4 - 8 * a4 * b0 * b5 * c0 * d2 * d3 +
          8 * a4 * b0 * b5 * c1 * d0 * d4 - 8 * a4 * b0 * b5 * c2 * d0 * d3 -
          8 * a4 * b0 * b5 * c3 * d0 * d2 + 8 * a4 * b0 * b5 * c4 * d0 * d1 +
          32 * a4 * b1 * b4 * c0 * d0 * d5 - 16 * a4 * b1 * b5 * c0 * d0 * d4 -
          16 * a4 * b2 * b3 * c0 * d0 * d5 + 16 * a4 * b2 * b5 * c0 * d0 * d3 +
          16 * a5 * b0 * b1 * c4 * d0 * d4 - 8 * a5 * b0 * b2 * c0 * d3 * d4 -
          8 * a5 * b0 * b2 * c3 * d0 * d4 - 8 * a5 * b0 * b2 * c4 * d0 * d3 +
          8 * a5 * b0 * b4 * c0 * d1 * d4 - 8 * a5 * b0 * b4 * c0 * d2 * d3 +
          8 * a5 * b0 * b4 * c1 * d0 * d4 - 8 * a5 * b0 * b4 * c2 * d0 * d3 -
          8 * a5 * b0 * b4 * c3 * d0 * d2 + 8 * a5 * b0 * b4 * c4 * d0 * d1 +
          8 * a5 * b0 * b5 * c0 * d1 * d3 + 8 * a5 * b0 * b5 * c1 * d0 * d3 +
          8 * a5 * b0 * b5 * c3 * d0 * d1 + 8 * a5 * b1 * b3 * c0 * d0 * d5 -
          16 * a5 * b1 * b4 * c0 * d0 * d4 - 8 * a5 * b1 * b5 * c0 * d0 * d3 +
          16 * a5 * b2 * b4 * c0 * d0 * d3 - 8 * a5 * b3 * b5 * c0 * d0 * d1 +
          8 * a0 * b2 * b5 * c1 * d2 * d5 + 8 * a0 * b2 * b5 * c2 * d1 * d5 +
          8 * a0 * b2 * b5 * c5 * d1 * d2 - 8 * a1 * b2 * b5 * c0 * d2 * d5 -
          8 * a1 * b2 * b5 * c2 * d0 * d5 - 8 * a1 * b2 * b5 * c5 * d0 * d2 -
          16 * a2 * b0 * b2 * c5 * d1 * d5 + 8 * a2 * b0 * b5 * c1 * d2 * d5 +
          8 * a2 * b0 * b5 * c2 * d1 * d5 + 8 * a2 * b0 * b5 * c5 * d1 * d2 +
          16 * a2 * b1 * b2 * c5 * d0 * d5 - 8 * a2 * b1 * b5 * c0 * d2 * d5 -
          8 * a2 * b1 * b5 * c2 * d0 * d5 - 8 * a2 * b1 * b5 * c5 * d0 * d2 +
          8 * a5 * b0 * b2 * c1 * d2 * d5 + 8 * a5 * b0 * b2 * c2 * d1 * d5 +
          8 * a5 * b0 * b2 * c5 * d1 * d2 - 16 * a5 * b0 * b5 * c2 * d1 * d2 -
          8 * a5 * b1 * b2 * c0 * d2 * d5 - 8 * a5 * b1 * b2 * c2 * d0 * d5 -
          8 * a5 * b1 * b2 * c5 * d0 * d2 + 16 * a5 * b1 * b5 * c2 * d0 * d2 +
          8 * a0 * b0 * b2 * c5 * d4 * d5 + 8 * a0 * b0 * b4 * c5 * d2 * d5 -
          8 * a0 * b0 * b5 * c2 * d4 * d5 - 8 * a0 * b0 * b5 * c4 * d2 * d5 +
          4 * a0 * b0 * b5 * c5 * d1 * d5 - 8 * a0 * b0 * b5 * c5 * d2 * d4 +
          4 * a0 * b1 * b5 * c5 * d0 * d5 - 8 * a0 * b2 * b4 * c5 * d0 * d5 +
          4 * a1 * b0 * b5 * c5 * d0 * d5 - 32 * a1 * b1 * b2 * c3 * d4 * d5 -
          32 * a1 * b1 * b2 * c4 * d3 * d5 - 32 * a1 * b1 * b2 * c5 * d3 * d4 -
          16 * a1 * b1 * b3 * c5 * d1 * d5 - 16 * a1 * b1 * b5 * c1 * d3 * d5 +
          32 * a1 * b1 * b5 * c2 * d3 * d4 - 16 * a1 * b1 * b5 * c3 * d1 * d5 +
          32 * a1 * b1 * b5 * c3 * d2 * d4 + 32 * a1 * b1 * b5 * c4 * d2 * d3 -
          16 * a1 * b1 * b5 * c5 * d1 * d3 + 16 * a1 * b2 * b3 * c1 * d4 * d5 +
          16 * a1 * b2 * b3 * c4 * d1 * d5 + 16 * a1 * b2 * b3 * c5 * d1 * d4 +
          16 * a1 * b2 * b4 * c1 * d3 * d5 - 32 * a1 * b2 * b4 * c2 * d3 * d4 +
          16 * a1 * b2 * b4 * c3 * d1 * d5 - 32 * a1 * b2 * b4 * c3 * d2 * d4 -
          32 * a1 * b2 * b4 * c4 * d2 * d3 + 16 * a1 * b2 * b4 * c5 * d1 * d3 -
          16 * a1 * b2 * b5 * c3 * d2 * d3 + 16 * a1 * b3 * b5 * c1 * d1 * d5 -
          16 * a1 * b3 * b5 * c1 * d2 * d4 - 16 * a1 * b3 * b5 * c2 * d1 * d4 +
          16 * a1 * b3 * b5 * c2 * d2 * d3 - 16 * a1 * b3 * b5 * c4 * d1 * d2 -
          16 * a1 * b4 * b5 * c1 * d2 * d3 - 16 * a1 * b4 * b5 * c2 * d1 * d3 -
          16 * a1 * b4 * b5 * c3 * d1 * d2 - 8 * a2 * b0 * b4 * c5 * d0 * d5 +
          16 * a2 * b1 * b2 * c3 * d3 * d5 + 64 * a2 * b1 * b2 * c4 * d3 * d4 +
          16 * a2 * b1 * b3 * c1 * d4 * d5 + 16 * a2 * b1 * b3 * c4 * d1 * d5 +
          16 * a2 * b1 * b3 * c5 * d1 * d4 + 16 * a2 * b1 * b4 * c1 * d3 * d5 -
          32 * a2 * b1 * b4 * c2 * d3 * d4 + 16 * a2 * b1 * b4 * c3 * d1 * d5 -
          32 * a2 * b1 * b4 * c3 * d2 * d4 - 32 * a2 * b1 * b4 * c4 * d2 * d3 +
          16 * a2 * b1 * b4 * c5 * d1 * d3 - 16 * a2 * b1 * b5 * c3 * d2 * d3 -
          16 * a2 * b2 * b3 * c1 * d3 * d5 + 32 * a2 * b2 * b3 * c2 * d3 * d4 -
          16 * a2 * b2 * b3 * c3 * d1 * d5 + 32 * a2 * b2 * b3 * c3 * d2 * d4 -
          64 * a2 * b2 * b3 * c4 * d1 * d4 + 32 * a2 * b2 * b3 * c4 * d2 * d3 -
          16 * a2 * b2 * b3 * c5 * d1 * d3 + 32 * a2 * b2 * b4 * c3 * d2 * d3 +
          16 * a2 * b2 * b5 * c3 * d1 * d3 - 32 * a2 * b3 * b4 * c1 * d1 * d5 +
          32 * a2 * b3 * b4 * c1 * d2 * d4 + 32 * a2 * b3 * b4 * c2 * d1 * d4 -
          32 * a2 * b3 * b4 * c2 * d2 * d3 + 32 * a2 * b3 * b4 * c4 * d1 * d2 +
          8 * a2 * b4 * b5 * c0 * d0 * d5 + 16 * a3 * b1 * b2 * c1 * d4 * d5 +
          16 * a3 * b1 * b2 * c4 * d1 * d5 + 16 * a3 * b1 * b2 * c5 * d1 * d4 -
          16 * a3 * b1 * b3 * c2 * d2 * d5 + 16 * a3 * b1 * b5 * c1 * d1 * d5 -
          16 * a3 * b1 * b5 * c1 * d2 * d4 - 16 * a3 * b1 * b5 * c2 * d1 * d4 +
          16 * a3 * b1 * b5 * c2 * d2 * d3 - 16 * a3 * b1 * b5 * c4 * d1 * d2 +
          16 * a3 * b2 * b3 * c1 * d2 * d5 + 16 * a3 * b2 * b3 * c2 * d1 * d5 -
          32 * a3 * b2 * b3 * c2 * d2 * d4 + 16 * a3 * b2 * b3 * c5 * d1 * d2 -
          32 * a3 * b2 * b4 * c1 * d1 * d5 + 32 * a3 * b2 * b4 * c1 * d2 * d4 +
          32 * a3 * b2 * b4 * c2 * d1 * d4 - 32 * a3 * b2 * b4 * c2 * d2 * d3 +
          32 * a3 * b2 * b4 * c4 * d1 * d2 - 16 * a3 * b3 * b5 * c2 * d1 * d2 +
          32 * a3 * b4 * b5 * c1 * d1 * d2 - 8 * a4 * b0 * b2 * c5 * d0 * d5 +
          16 * a4 * b1 * b2 * c1 * d3 * d5 - 32 * a4 * b1 * b2 * c2 * d3 * d4 +
          16 * a4 * b1 * b2 * c3 * d1 * d5 - 32 * a4 * b1 * b2 * c3 * d2 * d4 -
          32 * a4 * b1 * b2 * c4 * d2 * d3 + 16 * a4 * b1 * b2 * c5 * d1 * d3 +
          64 * a4 * b1 * b4 * c2 * d2 * d3 - 16 * a4 * b1 * b5 * c1 * d2 * d3 -
          16 * a4 * b1 * b5 * c2 * d1 * d3 - 16 * a4 * b1 * b5 * c3 * d1 * d2 -
          32 * a4 * b2 * b3 * c1 * d1 * d5 + 32 * a4 * b2 * b3 * c1 * d2 * d4 +
          32 * a4 * b2 * b3 * c2 * d1 * d4 - 32 * a4 * b2 * b3 * c2 * d2 * d3 +
          32 * a4 * b2 * b3 * c4 * d1 * d2 + 8 * a4 * b2 * b5 * c0 * d0 * d5 -
          64 * a4 * b3 * b4 * c2 * d1 * d2 + 32 * a4 * b3 * b5 * c1 * d1 * d2 +
          4 * a5 * b0 * b1 * c5 * d0 * d5 - 4 * a5 * b0 * b5 * c0 * d1 * d5 +
          8 * a5 * b0 * b5 * c0 * d2 * d4 - 4 * a5 * b0 * b5 * c1 * d0 * d5 +
          8 * a5 * b0 * b5 * c2 * d0 * d4 + 8 * a5 * b0 * b5 * c4 * d0 * d2 -
          4 * a5 * b0 * b5 * c5 * d0 * d1 - 16 * a5 * b1 * b2 * c3 * d2 * d3 +
          16 * a5 * b1 * b3 * c1 * d1 * d5 - 16 * a5 * b1 * b3 * c1 * d2 * d4 -
          16 * a5 * b1 * b3 * c2 * d1 * d4 + 16 * a5 * b1 * b3 * c2 * d2 * d3 -
          16 * a5 * b1 * b3 * c4 * d1 * d2 - 16 * a5 * b1 * b4 * c1 * d2 * d3 -
          16 * a5 * b1 * b4 * c2 * d1 * d3 - 16 * a5 * b1 * b4 * c3 * d1 * d2 -
          4 * a5 * b1 * b5 * c0 * d0 * d5 + 16 * a5 * b1 * b5 * c1 * d1 * d3 +
          8 * a5 * b2 * b4 * c0 * d0 * d5 - 8 * a5 * b2 * b5 * c0 * d0 * d4 +
          32 * a5 * b3 * b4 * c1 * d1 * d2 - 8 * a5 * b4 * b5 * c0 * d0 * d2 -
          4 * a0 * b1 * b3 * c5 * d3 * d5 + 8 * a0 * b1 * b4 * c3 * d4 * d5 +
          8 * a0 * b1 * b4 * c4 * d3 * d5 + 8 * a0 * b1 * b4 * c5 * d3 * d4 +
          4 * a0 * b1 * b5 * c3 * d3 * d5 - 16 * a0 * b1 * b5 * c4 * d3 * d4 -
          8 * a0 * b2 * b4 * c3 * d3 * d5 + 8 * a0 * b2 * b5 * c3 * d3 * d4 -
          8 * a0 * b3 * b4 * c1 * d4 * d5 - 8 * a0 * b3 * b4 * c4 * d1 * d5 -
          8 * a0 * b3 * b4 * c5 * d1 * d4 + 4 * a0 * b3 * b5 * c1 * d3 * d5 -
          8 * a0 * b3 * b5 * c2 * d3 * d4 + 4 * a0 * b3 * b5 * c3 * d1 * d5 -
          8 * a0 * b3 * b5 * c3 * d2 * d4 + 16 * a0 * b3 * b5 * c4 * d1 * d4 -
          8 * a0 * b3 * b5 * c4 * d2 * d3 + 4 * a0 * b3 * b5 * c5 * d1 * d3 +
          8 * a0 * b4 * b5 * c3 * d2 * d3 - 4 * a1 * b0 * b3 * c5 * d3 * d5 +
          8 * a1 * b0 * b4 * c3 * d4 * d5 + 8 * a1 * b0 * b4 * c4 * d3 * d5 +
          8 * a1 * b0 * b4 * c5 * d3 * d4 + 4 * a1 * b0 * b5 * c3 * d3 * d5 -
          16 * a1 * b0 * b5 * c4 * d3 * d4 - 4 * a1 * b3 * b5 * c0 * d3 * d5 -
          4 * a1 * b3 * b5 * c3 * d0 * d5 - 4 * a1 * b3 * b5 * c5 * d0 * d3 +
          8 * a1 * b4 * b5 * c0 * d3 * d4 + 8 * a1 * b4 * b5 * c3 * d0 * d4 +
          8 * a1 * b4 * b5 * c4 * d0 * d3 - 8 * a2 * b0 * b4 * c3 * d3 * d5 +
          8 * a2 * b0 * b5 * c3 * d3 * d4 + 8 * a2 * b3 * b4 * c0 * d3 * d5 +
          8 * a2 * b3 * b4 * c3 * d0 * d5 + 8 * a2 * b3 * b4 * c5 * d0 * d3 -
          8 * a2 * b4 * b5 * c3 * d0 * d3 - 4 * a3 * b0 * b1 * c5 * d3 * d5 +
          8 * a3 * b0 * b3 * c2 * d4 * d5 + 8 * a3 * b0 * b3 * c4 * d2 * d5 -
          4 * a3 * b0 * b3 * c5 * d1 * d5 + 8 * a3 * b0 * b3 * c5 * d2 * d4 -
          8 * a3 * b0 * b4 * c1 * d4 * d5 - 8 * a3 * b0 * b4 * c4 * d1 * d5 -
          8 * a3 * b0 * b4 * c5 * d1 * d4 + 4 * a3 * b0 * b5 * c1 * d3 * d5 -
          8 * a3 * b0 * b5 * c2 * d3 * d4 + 4 * a3 * b0 * b5 * c3 * d1 * d5 -
          8 * a3 * b0 * b5 * c3 * d2 * d4 + 16 * a3 * b0 * b5 * c4 * d1 * d4 -
          8 * a3 * b0 * b5 * c4 * d2 * d3 + 4 * a3 * b0 * b5 * c5 * d1 * d3 +
          12 * a3 * b1 * b3 * c5 * d0 * d5 - 4 * a3 * b1 * b5 * c0 * d3 * d5 -
          4 * a3 * b1 * b5 * c3 * d0 * d5 - 4 * a3 * b1 * b5 * c5 * d0 * d3 -
          8 * a3 * b2 * b3 * c0 * d4 * d5 - 8 * a3 * b2 * b3 * c4 * d0 * d5 -
          8 * a3 * b2 * b3 * c5 * d0 * d4 + 8 * a3 * b2 * b4 * c0 * d3 * d5 +
          8 * a3 * b2 * b4 * c3 * d0 * d5 + 8 * a3 * b2 * b4 * c5 * d0 * d3 -
          8 * a3 * b3 * b4 * c0 * d2 * d5 - 8 * a3 * b3 * b4 * c2 * d0 * d5 -
          8 * a3 * b3 * b4 * c5 * d0 * d2 - 4 * a3 * b3 * b5 * c0 * d1 * d5 +
          8 * a3 * b3 * b5 * c0 * d2 * d4 - 4 * a3 * b3 * b5 * c1 * d0 * d5 +
          8 * a3 * b3 * b5 * c2 * d0 * d4 + 8 * a3 * b3 * b5 * c4 * d0 * d2 -
          4 * a3 * b3 * b5 * c5 * d0 * d1 - 8 * a3 * b4 * b5 * c0 * d1 * d4 -
          8 * a3 * b4 * b5 * c1 * d0 * d4 - 8 * a3 * b4 * b5 * c4 * d0 * d1 +
          8 * a4 * b0 * b1 * c3 * d4 * d5 + 8 * a4 * b0 * b1 * c4 * d3 * d5 +
          8 * a4 * b0 * b1 * c5 * d3 * d4 - 8 * a4 * b0 * b2 * c3 * d3 * d5 -
          8 * a4 * b0 * b3 * c1 * d4 * d5 - 8 * a4 * b0 * b3 * c4 * d1 * d5 -
          8 * a4 * b0 * b3 * c5 * d1 * d4 + 8 * a4 * b0 * b5 * c3 * d2 * d3 -
          16 * a4 * b1 * b4 * c0 * d3 * d5 - 16 * a4 * b1 * b4 * c3 * d0 * d5 -
          16 * a4 * b1 * b4 * c5 * d0 * d3 + 8 * a4 * b1 * b5 * c0 * d3 * d4 +
          8 * a4 * b1 * b5 * c3 * d0 * d4 + 8 * a4 * b1 * b5 * c4 * d0 * d3 +
          8 * a4 * b2 * b3 * c0 * d3 * d5 + 8 * a4 * b2 * b3 * c3 * d0 * d5 +
          8 * a4 * b2 * b3 * c5 * d0 * d3 - 8 * a4 * b2 * b5 * c3 * d0 * d3 +
          16 * a4 * b3 * b4 * c0 * d1 * d5 + 16 * a4 * b3 * b4 * c1 * d0 * d5 +
          16 * a4 * b3 * b4 * c5 * d0 * d1 - 8 * a4 * b3 * b5 * c0 * d1 * d4 -
          8 * a4 * b3 * b5 * c1 * d0 * d4 - 8 * a4 * b3 * b5 * c4 * d0 * d1 +
          4 * a5 * b0 * b1 * c3 * d3 * d5 - 16 * a5 * b0 * b1 * c4 * d3 * d4 +
          8 * a5 * b0 * b2 * c3 * d3 * d4 + 4 * a5 * b0 * b3 * c1 * d3 * d5 -
          8 * a5 * b0 * b3 * c2 * d3 * d4 + 4 * a5 * b0 * b3 * c3 * d1 * d5 -
          8 * a5 * b0 * b3 * c3 * d2 * d4 + 16 * a5 * b0 * b3 * c4 * d1 * d4 -
          8 * a5 * b0 * b3 * c4 * d2 * d3 + 4 * a5 * b0 * b3 * c5 * d1 * d3 +
          8 * a5 * b0 * b4 * c3 * d2 * d3 - 12 * a5 * b0 * b5 * c3 * d1 * d3 -
          4 * a5 * b1 * b3 * c0 * d3 * d5 - 4 * a5 * b1 * b3 * c3 * d0 * d5 -
          4 * a5 * b1 * b3 * c5 * d0 * d3 + 8 * a5 * b1 * b4 * c0 * d3 * d4 +
          8 * a5 * b1 * b4 * c3 * d0 * d4 + 8 * a5 * b1 * b4 * c4 * d0 * d3 +
          4 * a5 * b1 * b5 * c3 * d0 * d3 - 8 * a5 * b2 * b4 * c3 * d0 * d3 -
          8 * a5 * b3 * b4 * c0 * d1 * d4 - 8 * a5 * b3 * b4 * c1 * d0 * d4 -
          8 * a5 * b3 * b4 * c4 * d0 * d1 + 4 * a5 * b3 * b5 * c0 * d1 * d3 +
          4 * a5 * b3 * b5 * c1 * d0 * d3 + 4 * a5 * b3 * b5 * c3 * d0 * d1 +
          32 * a1 * b1 * b2 * c5 * d4 * d5 + 32 * a1 * b1 * b4 * c5 * d2 * d5 -
          32 * a1 * b1 * b5 * c2 * d4 * d5 - 32 * a1 * b1 * b5 * c4 * d2 * d5 +
          48 * a1 * b1 * b5 * c5 * d1 * d5 - 32 * a1 * b1 * b5 * c5 * d2 * d4 -
          32 * a1 * b2 * b4 * c5 * d1 * d5 + 8 * a1 * b2 * b5 * c2 * d3 * d5 +
          8 * a1 * b2 * b5 * c3 * d2 * d5 + 8 * a1 * b2 * b5 * c5 * d2 * d3 -
          16 * a2 * b1 * b2 * c5 * d3 * d5 - 32 * a2 * b1 * b4 * c5 * d1 * d5 +
          8 * a2 * b1 * b5 * c2 * d3 * d5 + 8 * a2 * b1 * b5 * c3 * d2 * d5 +
          8 * a2 * b1 * b5 * c5 * d2 * d3 + 16 * a2 * b2 * b3 * c5 * d1 * d5 -
          8 * a2 * b3 * b5 * c1 * d2 * d5 - 8 * a2 * b3 * b5 * c2 * d1 * d5 -
          8 * a2 * b3 * b5 * c5 * d1 * d2 + 32 * a2 * b4 * b5 * c1 * d1 * d5 -
          8 * a3 * b2 * b5 * c1 * d2 * d5 - 8 * a3 * b2 * b5 * c2 * d1 * d5 -
          8 * a3 * b2 * b5 * c5 * d1 * d2 - 32 * a4 * b1 * b2 * c5 * d1 * d5 +
          32 * a4 * b2 * b5 * c1 * d1 * d5 + 8 * a5 * b1 * b2 * c2 * d3 * d5 +
          8 * a5 * b1 * b2 * c3 * d2 * d5 + 8 * a5 * b1 * b2 * c5 * d2 * d3 -
          48 * a5 * b1 * b5 * c1 * d1 * d5 + 32 * a5 * b1 * b5 * c1 * d2 * d4 +
          32 * a5 * b1 * b5 * c2 * d1 * d4 - 16 * a5 * b1 * b5 * c2 * d2 * d3 +
          32 * a5 * b1 * b5 * c4 * d1 * d2 - 8 * a5 * b2 * b3 * c1 * d2 * d5 -
          8 * a5 * b2 * b3 * c2 * d1 * d5 - 8 * a5 * b2 * b3 * c5 * d1 * d2 +
          32 * a5 * b2 * b4 * c1 * d1 * d5 - 32 * a5 * b2 * b5 * c1 * d1 * d4 +
          16 * a5 * b3 * b5 * c2 * d1 * d2 - 32 * a5 * b4 * b5 * c1 * d1 * d2 -
          4 * a0 * b1 * b5 * c5 * d3 * d5 - 8 * a0 * b2 * b3 * c5 * d4 * d5 +
          8 * a0 * b2 * b4 * c5 * d3 * d5 - 8 * a0 * b3 * b4 * c5 * d2 * d5 +
          8 * a0 * b3 * b5 * c2 * d4 * d5 + 8 * a0 * b3 * b5 * c4 * d2 * d5 -
          4 * a0 * b3 * b5 * c5 * d1 * d5 + 8 * a0 * b3 * b5 * c5 * d2 * d4 -
          8 * a0 * b4 * b5 * c1 * d4 * d5 - 8 * a0 * b4 * b5 * c4 * d1 * d5 -
          8 * a0 * b4 * b5 * c5 * d1 * d4 - 4 * a1 * b0 * b5 * c5 * d3 * d5 -
          4 * a1 * b3 * b5 * c5 * d0 * d5 + 8 * a1 * b4 * b5 * c0 * d4 * d5 +
          8 * a1 * b4 * b5 * c4 * d0 * d5 + 8 * a1 * b4 * b5 * c5 * d0 * d4 -
          8 * a2 * b0 * b3 * c5 * d4 * d5 + 8 * a2 * b0 * b4 * c5 * d3 * d5 +
          8 * a2 * b3 * b4 * c5 * d0 * d5 - 8 * a2 * b4 * b5 * c0 * d3 * d5 -
          8 * a2 * b4 * b5 * c3 * d0 * d5 - 8 * a2 * b4 * b5 * c5 * d0 * d3 -
          8 * a3 * b0 * b2 * c5 * d4 * d5 - 8 * a3 * b0 * b4 * c5 * d2 * d5 +
          8 * a3 * b0 * b5 * c2 * d4 * d5 + 8 * a3 * b0 * b5 * c4 * d2 * d5 -
          4 * a3 * b0 * b5 * c5 * d1 * d5 + 8 * a3 * b0 * b5 * c5 * d2 * d4 -
          4 * a3 * b1 * b5 * c5 * d0 * d5 + 8 * a3 * b2 * b4 * c5 * d0 * d5 +
          8 * a4 * b0 * b2 * c5 * d3 * d5 - 8 * a4 * b0 * b3 * c5 * d2 * d5 +
          16 * a4 * b0 * b4 * c5 * d1 * d5 - 8 * a4 * b0 * b5 * c1 * d4 * d5 -
          8 * a4 * b0 * b5 * c4 * d1 * d5 - 8 * a4 * b0 * b5 * c5 * d1 * d4 -
          16 * a4 * b1 * b4 * c5 * d0 * d5 + 8 * a4 * b1 * b5 * c0 * d4 * d5 +
          8 * a4 * b1 * b5 * c4 * d0 * d5 + 8 * a4 * b1 * b5 * c5 * d0 * d4 +
          8 * a4 * b2 * b3 * c5 * d0 * d5 - 8 * a4 * b2 * b5 * c0 * d3 * d5 -
          8 * a4 * b2 * b5 * c3 * d0 * d5 - 8 * a4 * b2 * b5 * c5 * d0 * d3 -
          4 * a5 * b0 * b1 * c5 * d3 * d5 + 8 * a5 * b0 * b3 * c2 * d4 * d5 +
          8 * a5 * b0 * b3 * c4 * d2 * d5 - 4 * a5 * b0 * b3 * c5 * d1 * d5 +
          8 * a5 * b0 * b3 * c5 * d2 * d4 - 8 * a5 * b0 * b4 * c1 * d4 * d5 -
          8 * a5 * b0 * b4 * c4 * d1 * d5 - 8 * a5 * b0 * b4 * c5 * d1 * d4 +
          4 * a5 * b0 * b5 * c1 * d3 * d5 - 8 * a5 * b0 * b5 * c2 * d3 * d4 +
          4 * a5 * b0 * b5 * c3 * d1 * d5 - 8 * a5 * b0 * b5 * c3 * d2 * d4 +
          16 * a5 * b0 * b5 * c4 * d1 * d4 - 8 * a5 * b0 * b5 * c4 * d2 * d3 +
          4 * a5 * b0 * b5 * c5 * d1 * d3 - 4 * a5 * b1 * b3 * c5 * d0 * d5 +
          8 * a5 * b1 * b4 * c0 * d4 * d5 + 8 * a5 * b1 * b4 * c4 * d0 * d5 +
          8 * a5 * b1 * b4 * c5 * d0 * d4 + 4 * a5 * b1 * b5 * c0 * d3 * d5 +
          4 * a5 * b1 * b5 * c3 * d0 * d5 - 16 * a5 * b1 * b5 * c4 * d0 * d4 +
          4 * a5 * b1 * b5 * c5 * d0 * d3 - 8 * a5 * b2 * b4 * c0 * d3 * d5 -
          8 * a5 * b2 * b4 * c3 * d0 * d5 - 8 * a5 * b2 * b4 * c5 * d0 * d3 +
          8 * a5 * b2 * b5 * c0 * d3 * d4 + 8 * a5 * b2 * b5 * c3 * d0 * d4 +
          8 * a5 * b2 * b5 * c4 * d0 * d3 + 4 * a5 * b3 * b5 * c0 * d1 * d5 -
          8 * a5 * b3 * b5 * c0 * d2 * d4 + 4 * a5 * b3 * b5 * c1 * d0 * d5 -
          8 * a5 * b3 * b5 * c2 * d0 * d4 - 8 * a5 * b3 * b5 * c4 * d0 * d2 +
          4 * a5 * b3 * b5 * c5 * d0 * d1 + 8 * a5 * b4 * b5 * c0 * d2 * d3 +
          8 * a5 * b4 * b5 * c2 * d0 * d3 + 8 * a5 * b4 * b5 * c3 * d0 * d2 +
          4 * a1 * b3 * b5 * c5 * d3 * d5 - 8 * a1 * b4 * b5 * c3 * d4 * d5 -
          8 * a1 * b4 * b5 * c4 * d3 * d5 - 8 * a1 * b4 * b5 * c5 * d3 * d4 -
          8 * a2 * b3 * b4 * c5 * d3 * d5 + 8 * a2 * b4 * b5 * c3 * d3 * d5 +
          4 * a3 * b1 * b5 * c5 * d3 * d5 + 8 * a3 * b2 * b3 * c5 * d4 * d5 -
          8 * a3 * b2 * b4 * c5 * d3 * d5 + 8 * a3 * b3 * b4 * c5 * d2 * d5 -
          8 * a3 * b3 * b5 * c2 * d4 * d5 - 8 * a3 * b3 * b5 * c4 * d2 * d5 +
          4 * a3 * b3 * b5 * c5 * d1 * d5 - 8 * a3 * b3 * b5 * c5 * d2 * d4 +
          8 * a3 * b4 * b5 * c1 * d4 * d5 + 8 * a3 * b4 * b5 * c4 * d1 * d5 +
          8 * a3 * b4 * b5 * c5 * d1 * d4 + 16 * a4 * b1 * b4 * c5 * d3 * d5 -
          8 * a4 * b1 * b5 * c3 * d4 * d5 - 8 * a4 * b1 * b5 * c4 * d3 * d5 -
          8 * a4 * b1 * b5 * c5 * d3 * d4 - 8 * a4 * b2 * b3 * c5 * d3 * d5 +
          8 * a4 * b2 * b5 * c3 * d3 * d5 - 16 * a4 * b3 * b4 * c5 * d1 * d5 +
          8 * a4 * b3 * b5 * c1 * d4 * d5 + 8 * a4 * b3 * b5 * c4 * d1 * d5 +
          8 * a4 * b3 * b5 * c5 * d1 * d4 + 4 * a5 * b1 * b3 * c5 * d3 * d5 -
          8 * a5 * b1 * b4 * c3 * d4 * d5 - 8 * a5 * b1 * b4 * c4 * d3 * d5 -
          8 * a5 * b1 * b4 * c5 * d3 * d4 - 4 * a5 * b1 * b5 * c3 * d3 * d5 +
          16 * a5 * b1 * b5 * c4 * d3 * d4 + 8 * a5 * b2 * b4 * c3 * d3 * d5 -
          8 * a5 * b2 * b5 * c3 * d3 * d4 + 8 * a5 * b3 * b4 * c1 * d4 * d5 +
          8 * a5 * b3 * b4 * c4 * d1 * d5 + 8 * a5 * b3 * b4 * c5 * d1 * d4 -
          4 * a5 * b3 * b5 * c1 * d3 * d5 + 8 * a5 * b3 * b5 * c2 * d3 * d4 -
          4 * a5 * b3 * b5 * c3 * d1 * d5 + 8 * a5 * b3 * b5 * c3 * d2 * d4 -
          16 * a5 * b3 * b5 * c4 * d1 * d4 + 8 * a5 * b3 * b5 * c4 * d2 * d3 -
          4 * a5 * b3 * b5 * c5 * d1 * d3 - 8 * a5 * b4 * b5 * c3 * d2 * d3,
      a1 * pow(b0, 2) * pow(c3, 2) * d0 - a0 * pow(b3, 2) * pow(c0, 2) * d1 -
          3 * a0 * pow(b0, 2) * pow(c3, 2) * d1 +
          3 * a1 * pow(b3, 2) * pow(c0, 2) * d0 +
          12 * a0 * pow(b0, 2) * pow(c4, 2) * d1 +
          4 * a0 * pow(b4, 2) * pow(c0, 2) * d1 +
          4 * a1 * pow(b0, 2) * pow(c1, 2) * d3 -
          4 * a1 * pow(b0, 2) * pow(c4, 2) * d0 +
          12 * a1 * pow(b1, 2) * pow(c0, 2) * d3 -
          12 * a1 * pow(b4, 2) * pow(c0, 2) * d0 -
          12 * a3 * pow(b0, 2) * pow(c1, 2) * d1 -
          4 * a3 * pow(b1, 2) * pow(c0, 2) * d1 -
          3 * a0 * pow(b0, 2) * pow(c5, 2) * d1 -
          4 * a0 * pow(b2, 2) * pow(c3, 2) * d1 +
          4 * a0 * pow(b3, 2) * pow(c2, 2) * d1 -
          a0 * pow(b5, 2) * pow(c0, 2) * d1 +
          4 * a1 * pow(b0, 2) * pow(c2, 2) * d3 +
          a1 * pow(b0, 2) * pow(c5, 2) * d0 +
          4 * a1 * pow(b2, 2) * pow(c0, 2) * d3 -
          4 * a1 * pow(b2, 2) * pow(c3, 2) * d0 +
          4 * a1 * pow(b3, 2) * pow(c2, 2) * d0 +
          3 * a1 * pow(b5, 2) * pow(c0, 2) * d0 -
          4 * a3 * pow(b0, 2) * pow(c2, 2) * d1 -
          4 * a3 * pow(b2, 2) * pow(c0, 2) * d1 -
          8 * a0 * pow(b1, 2) * pow(c5, 2) * d1 +
          16 * a0 * pow(b2, 2) * pow(c4, 2) * d1 +
          16 * a0 * pow(b4, 2) * pow(c2, 2) * d1 -
          24 * a0 * pow(b5, 2) * pow(c1, 2) * d1 -
          4 * a1 * pow(b0, 2) * pow(c1, 2) * d5 -
          12 * a1 * pow(b1, 2) * pow(c0, 2) * d5 +
          48 * a1 * pow(b1, 2) * pow(c2, 2) * d3 +
          24 * a1 * pow(b1, 2) * pow(c5, 2) * d0 +
          16 * a1 * pow(b2, 2) * pow(c1, 2) * d3 -
          16 * a1 * pow(b2, 2) * pow(c4, 2) * d0 -
          16 * a1 * pow(b4, 2) * pow(c2, 2) * d0 +
          8 * a1 * pow(b5, 2) * pow(c1, 2) * d0 -
          8 * a2 * pow(b0, 2) * pow(c1, 2) * d4 -
          8 * a2 * pow(b1, 2) * pow(c0, 2) * d4 -
          16 * a3 * pow(b1, 2) * pow(c2, 2) * d1 -
          48 * a3 * pow(b2, 2) * pow(c1, 2) * d1 +
          8 * a4 * pow(b0, 2) * pow(c1, 2) * d2 +
          8 * a4 * pow(b1, 2) * pow(c0, 2) * d2 +
          12 * a5 * pow(b0, 2) * pow(c1, 2) * d1 +
          4 * a5 * pow(b1, 2) * pow(c0, 2) * d1 -
          4 * a0 * pow(b2, 2) * pow(c5, 2) * d1 -
          4 * a0 * pow(b5, 2) * pow(c2, 2) * d1 -
          4 * a1 * pow(b0, 2) * pow(c2, 2) * d5 +
          4 * a1 * pow(b0, 2) * pow(c4, 2) * d3 -
          4 * a1 * pow(b2, 2) * pow(c0, 2) * d5 +
          4 * a1 * pow(b2, 2) * pow(c5, 2) * d0 +
          4 * a1 * pow(b4, 2) * pow(c0, 2) * d3 +
          4 * a1 * pow(b5, 2) * pow(c2, 2) * d0 -
          8 * a2 * pow(b0, 2) * pow(c2, 2) * d4 -
          24 * a2 * pow(b2, 2) * pow(c0, 2) * d4 -
          4 * a3 * pow(b0, 2) * pow(c4, 2) * d1 -
          4 * a3 * pow(b4, 2) * pow(c0, 2) * d1 +
          24 * a4 * pow(b0, 2) * pow(c2, 2) * d2 +
          8 * a4 * pow(b2, 2) * pow(c0, 2) * d2 +
          4 * a5 * pow(b0, 2) * pow(c2, 2) * d1 +
          4 * a5 * pow(b2, 2) * pow(c0, 2) * d1 -
          a0 * pow(b3, 2) * pow(c5, 2) * d1 -
          3 * a0 * pow(b5, 2) * pow(c3, 2) * d1 -
          a1 * pow(b0, 2) * pow(c3, 2) * d5 +
          2 * a1 * pow(b0, 2) * pow(c5, 2) * d3 -
          48 * a1 * pow(b1, 2) * pow(c2, 2) * d5 -
          16 * a1 * pow(b2, 2) * pow(c1, 2) * d5 -
          3 * a1 * pow(b3, 2) * pow(c0, 2) * d5 +
          3 * a1 * pow(b3, 2) * pow(c5, 2) * d0 -
          2 * a1 * pow(b5, 2) * pow(c0, 2) * d3 +
          a1 * pow(b5, 2) * pow(c3, 2) * d0 -
          2 * a2 * pow(b0, 2) * pow(c3, 2) * d4 -
          32 * a2 * pow(b1, 2) * pow(c2, 2) * d4 -
          96 * a2 * pow(b2, 2) * pow(c1, 2) * d4 +
          2 * a2 * pow(b3, 2) * pow(c0, 2) * d4 +
          2 * a3 * pow(b0, 2) * pow(c5, 2) * d1 -
          2 * a3 * pow(b5, 2) * pow(c0, 2) * d1 -
          2 * a4 * pow(b0, 2) * pow(c3, 2) * d2 +
          96 * a4 * pow(b1, 2) * pow(c2, 2) * d2 +
          32 * a4 * pow(b2, 2) * pow(c1, 2) * d2 +
          2 * a4 * pow(b3, 2) * pow(c0, 2) * d2 +
          3 * a5 * pow(b0, 2) * pow(c3, 2) * d1 +
          16 * a5 * pow(b1, 2) * pow(c2, 2) * d1 +
          48 * a5 * pow(b2, 2) * pow(c1, 2) * d1 +
          a5 * pow(b3, 2) * pow(c0, 2) * d1 +
          4 * a0 * pow(b4, 2) * pow(c5, 2) * d1 +
          4 * a0 * pow(b5, 2) * pow(c4, 2) * d1 +
          12 * a1 * pow(b1, 2) * pow(c5, 2) * d3 +
          16 * a1 * pow(b2, 2) * pow(c4, 2) * d3 +
          8 * a1 * pow(b4, 2) * pow(c0, 2) * d5 +
          16 * a1 * pow(b4, 2) * pow(c2, 2) * d3 -
          4 * a1 * pow(b4, 2) * pow(c5, 2) * d0 +
          4 * a1 * pow(b5, 2) * pow(c1, 2) * d3 -
          4 * a1 * pow(b5, 2) * pow(c4, 2) * d0 -
          4 * a3 * pow(b1, 2) * pow(c5, 2) * d1 -
          16 * a3 * pow(b2, 2) * pow(c4, 2) * d1 -
          16 * a3 * pow(b4, 2) * pow(c2, 2) * d1 -
          12 * a3 * pow(b5, 2) * pow(c1, 2) * d1 -
          8 * a5 * pow(b0, 2) * pow(c4, 2) * d1 -
          3 * a1 * pow(b0, 2) * pow(c5, 2) * d5 +
          4 * a1 * pow(b2, 2) * pow(c3, 2) * d5 -
          4 * a1 * pow(b2, 2) * pow(c5, 2) * d3 -
          4 * a1 * pow(b3, 2) * pow(c2, 2) * d5 -
          a1 * pow(b5, 2) * pow(c0, 2) * d5 -
          4 * a1 * pow(b5, 2) * pow(c2, 2) * d3 +
          2 * a2 * pow(b0, 2) * pow(c5, 2) * d4 -
          24 * a2 * pow(b2, 2) * pow(c3, 2) * d4 -
          8 * a2 * pow(b3, 2) * pow(c2, 2) * d4 -
          2 * a2 * pow(b5, 2) * pow(c0, 2) * d4 +
          4 * a3 * pow(b2, 2) * pow(c5, 2) * d1 +
          4 * a3 * pow(b5, 2) * pow(c2, 2) * d1 +
          2 * a4 * pow(b0, 2) * pow(c5, 2) * d2 +
          8 * a4 * pow(b2, 2) * pow(c3, 2) * d2 +
          24 * a4 * pow(b3, 2) * pow(c2, 2) * d2 -
          2 * a4 * pow(b5, 2) * pow(c0, 2) * d2 +
          a5 * pow(b0, 2) * pow(c5, 2) * d1 +
          4 * a5 * pow(b2, 2) * pow(c3, 2) * d1 -
          4 * a5 * pow(b3, 2) * pow(c2, 2) * d1 +
          3 * a5 * pow(b5, 2) * pow(c0, 2) * d1 -
          36 * a1 * pow(b1, 2) * pow(c5, 2) * d5 -
          12 * a1 * pow(b5, 2) * pow(c1, 2) * d5 +
          8 * a2 * pow(b1, 2) * pow(c5, 2) * d4 -
          8 * a2 * pow(b5, 2) * pow(c1, 2) * d4 +
          8 * a4 * pow(b1, 2) * pow(c5, 2) * d2 -
          8 * a4 * pow(b5, 2) * pow(c1, 2) * d2 +
          12 * a5 * pow(b1, 2) * pow(c5, 2) * d1 +
          36 * a5 * pow(b5, 2) * pow(c1, 2) * d1 +
          4 * a1 * pow(b4, 2) * pow(c5, 2) * d3 +
          4 * a1 * pow(b5, 2) * pow(c4, 2) * d3 -
          4 * a3 * pow(b4, 2) * pow(c5, 2) * d1 -
          4 * a3 * pow(b5, 2) * pow(c4, 2) * d1 -
          3 * a1 * pow(b3, 2) * pow(c5, 2) * d5 -
          a1 * pow(b5, 2) * pow(c3, 2) * d5 +
          2 * a2 * pow(b3, 2) * pow(c5, 2) * d4 -
          2 * a2 * pow(b5, 2) * pow(c3, 2) * d4 +
          2 * a4 * pow(b3, 2) * pow(c5, 2) * d2 -
          2 * a4 * pow(b5, 2) * pow(c3, 2) * d2 +
          a5 * pow(b3, 2) * pow(c5, 2) * d1 +
          3 * a5 * pow(b5, 2) * pow(c3, 2) * d1 +
          2 * a0 * b0 * b1 * pow(c3, 2) * d0 +
          8 * a0 * b0 * b1 * pow(c1, 2) * d3 -
          8 * a0 * b0 * b1 * pow(c4, 2) * d0 -
          24 * a0 * b0 * b3 * pow(c1, 2) * d1 +
          8 * a0 * b1 * b3 * pow(c1, 2) * d0 +
          8 * a1 * b0 * b3 * pow(c1, 2) * d0 +
          8 * a3 * b0 * b1 * pow(c1, 2) * d0 +
          8 * a0 * b0 * b1 * pow(c2, 2) * d3 +
          2 * a0 * b0 * b1 * pow(c5, 2) * d0 -
          8 * a0 * b0 * b3 * pow(c2, 2) * d1 -
          8 * a1 * b1 * b3 * pow(c0, 2) * d1 -
          8 * a0 * b0 * b1 * pow(c1, 2) * d5 -
          16 * a0 * b0 * b2 * pow(c1, 2) * d4 +
          16 * a0 * b0 * b4 * pow(c1, 2) * d2 +
          24 * a0 * b0 * b5 * pow(c1, 2) * d1 -
          2 * a0 * b1 * b3 * pow(c0, 2) * d3 -
          8 * a0 * b1 * b5 * pow(c1, 2) * d0 -
          2 * a1 * b0 * b3 * pow(c0, 2) * d3 -
          8 * a1 * b0 * b5 * pow(c1, 2) * d0 -
          2 * a3 * b0 * b1 * pow(c0, 2) * d3 -
          2 * a3 * b0 * b3 * pow(c0, 2) * d1 +
          6 * a3 * b1 * b3 * pow(c0, 2) * d0 -
          8 * a5 * b0 * b1 * pow(c1, 2) * d0 -
          8 * a0 * b0 * b1 * pow(c2, 2) * d5 +
          8 * a0 * b0 * b1 * pow(c4, 2) * d3 -
          16 * a0 * b0 * b2 * pow(c2, 2) * d4 -
          8 * a0 * b0 * b3 * pow(c4, 2) * d1 +
          48 * a0 * b0 * b4 * pow(c2, 2) * d2 +
          8 * a0 * b0 * b5 * pow(c2, 2) * d1 +
          8 * a0 * b1 * b2 * pow(c3, 2) * d2 -
          16 * a0 * b2 * b4 * pow(c2, 2) * d0 -
          16 * a1 * b0 * b1 * pow(c5, 2) * d1 +
          8 * a1 * b0 * b2 * pow(c3, 2) * d2 -
          16 * a1 * b1 * b2 * pow(c0, 2) * d4 -
          32 * a1 * b1 * b3 * pow(c2, 2) * d1 +
          16 * a1 * b1 * b4 * pow(c0, 2) * d2 +
          8 * a1 * b1 * b5 * pow(c0, 2) * d1 +
          8 * a2 * b0 * b1 * pow(c3, 2) * d2 -
          8 * a2 * b0 * b2 * pow(c3, 2) * d1 -
          16 * a2 * b0 * b4 * pow(c2, 2) * d0 +
          8 * a2 * b1 * b2 * pow(c0, 2) * d3 -
          8 * a2 * b1 * b2 * pow(c3, 2) * d0 -
          8 * a2 * b2 * b3 * pow(c0, 2) * d1 -
          16 * a4 * b0 * b2 * pow(c2, 2) * d0 -
          2 * a0 * b0 * b1 * pow(c3, 2) * d5 +
          4 * a0 * b0 * b1 * pow(c5, 2) * d3 -
          4 * a0 * b0 * b2 * pow(c3, 2) * d4 +
          4 * a0 * b0 * b3 * pow(c5, 2) * d1 -
          4 * a0 * b0 * b4 * pow(c3, 2) * d2 +
          6 * a0 * b0 * b5 * pow(c3, 2) * d1 +
          2 * a0 * b1 * b3 * pow(c0, 2) * d5 -
          8 * a0 * b1 * b3 * pow(c2, 2) * d3 -
          4 * a0 * b1 * b3 * pow(c5, 2) * d0 +
          8 * a0 * b1 * b4 * pow(c0, 2) * d4 +
          2 * a0 * b1 * b5 * pow(c0, 2) * d3 -
          2 * a0 * b1 * b5 * pow(c3, 2) * d0 -
          4 * a0 * b2 * b3 * pow(c0, 2) * d4 -
          4 * a0 * b2 * b4 * pow(c0, 2) * d3 +
          4 * a0 * b2 * b4 * pow(c3, 2) * d0 -
          4 * a0 * b3 * b4 * pow(c0, 2) * d2 +
          2 * a0 * b3 * b5 * pow(c0, 2) * d1 +
          2 * a1 * b0 * b3 * pow(c0, 2) * d5 -
          8 * a1 * b0 * b3 * pow(c2, 2) * d3 -
          4 * a1 * b0 * b3 * pow(c5, 2) * d0 +
          8 * a1 * b0 * b4 * pow(c0, 2) * d4 +
          2 * a1 * b0 * b5 * pow(c0, 2) * d3 -
          2 * a1 * b0 * b5 * pow(c3, 2) * d0 +
          32 * a1 * b2 * b3 * pow(c1, 2) * d2 -
          6 * a1 * b3 * b5 * pow(c0, 2) * d0 +
          32 * a2 * b0 * b2 * pow(c4, 2) * d1 -
          4 * a2 * b0 * b3 * pow(c0, 2) * d4 -
          4 * a2 * b0 * b4 * pow(c0, 2) * d3 +
          4 * a2 * b0 * b4 * pow(c3, 2) * d0 +
          32 * a2 * b1 * b2 * pow(c1, 2) * d3 -
          32 * a2 * b1 * b2 * pow(c4, 2) * d0 +
          32 * a2 * b1 * b3 * pow(c1, 2) * d2 -
          96 * a2 * b2 * b3 * pow(c1, 2) * d1 +
          12 * a2 * b3 * b4 * pow(c0, 2) * d0 +
          2 * a3 * b0 * b1 * pow(c0, 2) * d5 -
          8 * a3 * b0 * b1 * pow(c2, 2) * d3 -
          4 * a3 * b0 * b1 * pow(c5, 2) * d0 -
          4 * a3 * b0 * b2 * pow(c0, 2) * d4 +
          8 * a3 * b0 * b3 * pow(c2, 2) * d1 -
          4 * a3 * b0 * b4 * pow(c0, 2) * d2 +
          2 * a3 * b0 * b5 * pow(c0, 2) * d1 +
          32 * a3 * b1 * b2 * pow(c1, 2) * d2 +
          8 * a3 * b1 * b3 * pow(c2, 2) * d0 -
          6 * a3 * b1 * b5 * pow(c0, 2) * d0 +
          12 * a3 * b2 * b4 * pow(c0, 2) * d0 +
          8 * a4 * b0 * b1 * pow(c0, 2) * d4 -
          4 * a4 * b0 * b2 * pow(c0, 2) * d3 +
          4 * a4 * b0 * b2 * pow(c3, 2) * d0 -
          4 * a4 * b0 * b3 * pow(c0, 2) * d2 +
          8 * a4 * b0 * b4 * pow(c0, 2) * d1 -
          24 * a4 * b1 * b4 * pow(c0, 2) * d0 +
          12 * a4 * b2 * b3 * pow(c0, 2) * d0 +
          2 * a5 * b0 * b1 * pow(c0, 2) * d3 -
          2 * a5 * b0 * b1 * pow(c3, 2) * d0 +
          2 * a5 * b0 * b3 * pow(c0, 2) * d1 -
          6 * a5 * b1 * b3 * pow(c0, 2) * d0 -
          16 * a0 * b0 * b5 * pow(c4, 2) * d1 -
          8 * a0 * b1 * b3 * pow(c1, 2) * d5 -
          8 * a0 * b1 * b5 * pow(c1, 2) * d3 +
          8 * a0 * b1 * b5 * pow(c4, 2) * d0 -
          16 * a0 * b3 * b4 * pow(c1, 2) * d2 +
          24 * a0 * b3 * b5 * pow(c1, 2) * d1 -
          8 * a1 * b0 * b3 * pow(c1, 2) * d5 -
          8 * a1 * b0 * b5 * pow(c1, 2) * d3 +
          8 * a1 * b0 * b5 * pow(c4, 2) * d0 -
          64 * a1 * b1 * b2 * pow(c2, 2) * d4 +
          192 * a1 * b1 * b4 * pow(c2, 2) * d2 +
          32 * a1 * b1 * b5 * pow(c2, 2) * d1 -
          64 * a1 * b2 * b4 * pow(c2, 2) * d1 -
          8 * a1 * b3 * b5 * pow(c1, 2) * d0 -
          8 * a2 * b0 * b2 * pow(c5, 2) * d1 -
          8 * a2 * b1 * b2 * pow(c0, 2) * d5 +
          8 * a2 * b1 * b2 * pow(c5, 2) * d0 -
          64 * a2 * b1 * b4 * pow(c2, 2) * d1 +
          16 * a2 * b2 * b4 * pow(c0, 2) * d2 +
          8 * a2 * b2 * b5 * pow(c0, 2) * d1 +
          16 * a2 * b3 * b4 * pow(c1, 2) * d0 -
          8 * a3 * b0 * b1 * pow(c1, 2) * d5 -
          16 * a3 * b0 * b4 * pow(c1, 2) * d2 +
          24 * a3 * b0 * b5 * pow(c1, 2) * d1 -
          8 * a3 * b1 * b5 * pow(c1, 2) * d0 +
          16 * a3 * b2 * b4 * pow(c1, 2) * d0 -
          16 * a4 * b0 * b3 * pow(c1, 2) * d2 -
          64 * a4 * b1 * b2 * pow(c2, 2) * d1 +
          16 * a4 * b2 * b3 * pow(c1, 2) * d0 -
          8 * a5 * b0 * b1 * pow(c1, 2) * d3 +
          8 * a5 * b0 * b1 * pow(c4, 2) * d0 +
          24 * a5 * b0 * b3 * pow(c1, 2) * d1 -
          8 * a5 * b1 * b3 * pow(c1, 2) * d0 -
          6 * a0 * b0 * b1 * pow(c5, 2) * d5 +
          4 * a0 * b0 * b2 * pow(c5, 2) * d4 +
          4 * a0 * b0 * b4 * pow(c5, 2) * d2 +
          2 * a0 * b0 * b5 * pow(c5, 2) * d1 +
          8 * a0 * b1 * b3 * pow(c2, 2) * d5 -
          2 * a0 * b1 * b5 * pow(c0, 2) * d5 +
          2 * a0 * b1 * b5 * pow(c5, 2) * d0 +
          16 * a0 * b2 * b3 * pow(c2, 2) * d4 +
          4 * a0 * b2 * b4 * pow(c0, 2) * d5 +
          16 * a0 * b2 * b4 * pow(c2, 2) * d3 -
          4 * a0 * b2 * b4 * pow(c5, 2) * d0 +
          4 * a0 * b2 * b5 * pow(c0, 2) * d4 -
          48 * a0 * b3 * b4 * pow(c2, 2) * d2 +
          4 * a0 * b4 * b5 * pow(c0, 2) * d2 +
          8 * a1 * b0 * b3 * pow(c2, 2) * d5 -
          2 * a1 * b0 * b5 * pow(c0, 2) * d5 +
          2 * a1 * b0 * b5 * pow(c5, 2) * d0 -
          8 * a1 * b1 * b3 * pow(c5, 2) * d1 -
          32 * a1 * b2 * b5 * pow(c1, 2) * d2 -
          8 * a1 * b3 * b5 * pow(c2, 2) * d0 +
          16 * a2 * b0 * b3 * pow(c2, 2) * d4 +
          4 * a2 * b0 * b4 * pow(c0, 2) * d5 +
          16 * a2 * b0 * b4 * pow(c2, 2) * d3 -
          4 * a2 * b0 * b4 * pow(c5, 2) * d0 +
          4 * a2 * b0 * b5 * pow(c0, 2) * d4 -
          32 * a2 * b1 * b2 * pow(c1, 2) * d5 -
          32 * a2 * b1 * b5 * pow(c1, 2) * d2 +
          64 * a2 * b2 * b4 * pow(c1, 2) * d2 +
          96 * a2 * b2 * b5 * pow(c1, 2) * d1 +
          16 * a2 * b3 * b4 * pow(c2, 2) * d0 -
          12 * a2 * b4 * b5 * pow(c0, 2) * d0 +
          8 * a3 * b0 * b1 * pow(c2, 2) * d5 +
          16 * a3 * b0 * b2 * pow(c2, 2) * d4 -
          48 * a3 * b0 * b4 * pow(c2, 2) * d2 -
          8 * a3 * b1 * b5 * pow(c2, 2) * d0 +
          16 * a3 * b2 * b4 * pow(c2, 2) * d0 +
          4 * a4 * b0 * b2 * pow(c0, 2) * d5 +
          16 * a4 * b0 * b2 * pow(c2, 2) * d3 -
          4 * a4 * b0 * b2 * pow(c5, 2) * d0 -
          48 * a4 * b0 * b3 * pow(c2, 2) * d2 +
          32 * a4 * b0 * b4 * pow(c2, 2) * d1 +
          4 * a4 * b0 * b5 * pow(c0, 2) * d2 -
          32 * a4 * b1 * b4 * pow(c2, 2) * d0 +
          16 * a4 * b2 * b3 * pow(c2, 2) * d0 -
          12 * a4 * b2 * b5 * pow(c0, 2) * d0 -
          2 * a5 * b0 * b1 * pow(c0, 2) * d5 +
          2 * a5 * b0 * b1 * pow(c5, 2) * d0 +
          4 * a5 * b0 * b2 * pow(c0, 2) * d4 +
          4 * a5 * b0 * b4 * pow(c0, 2) * d2 -
          2 * a5 * b0 * b5 * pow(c0, 2) * d1 -
          32 * a5 * b1 * b2 * pow(c1, 2) * d2 -
          8 * a5 * b1 * b3 * pow(c2, 2) * d0 +
          6 * a5 * b1 * b5 * pow(c0, 2) * d0 -
          12 * a5 * b2 * b4 * pow(c0, 2) * d0 -
          2 * a0 * b1 * b3 * pow(c5, 2) * d3 +
          16 * a0 * b1 * b5 * pow(c1, 2) * d5 +
          16 * a0 * b2 * b5 * pow(c1, 2) * d4 -
          2 * a1 * b0 * b3 * pow(c5, 2) * d3 +
          16 * a1 * b0 * b5 * pow(c1, 2) * d5 +
          2 * a1 * b3 * b5 * pow(c0, 2) * d3 +
          16 * a2 * b0 * b5 * pow(c1, 2) * d4 +
          32 * a2 * b1 * b2 * pow(c4, 2) * d3 -
          32 * a2 * b2 * b3 * pow(c4, 2) * d1 -
          4 * a2 * b3 * b4 * pow(c0, 2) * d3 -
          16 * a2 * b4 * b5 * pow(c1, 2) * d0 -
          2 * a3 * b0 * b1 * pow(c5, 2) * d3 -
          2 * a3 * b0 * b3 * pow(c5, 2) * d1 -
          6 * a3 * b1 * b3 * pow(c0, 2) * d5 +
          6 * a3 * b1 * b3 * pow(c5, 2) * d0 +
          2 * a3 * b1 * b5 * pow(c0, 2) * d3 +
          4 * a3 * b2 * b3 * pow(c0, 2) * d4 -
          4 * a3 * b2 * b4 * pow(c0, 2) * d3 +
          4 * a3 * b3 * b4 * pow(c0, 2) * d2 +
          2 * a3 * b3 * b5 * pow(c0, 2) * d1 +
          8 * a4 * b1 * b4 * pow(c0, 2) * d3 -
          4 * a4 * b2 * b3 * pow(c0, 2) * d3 -
          16 * a4 * b2 * b5 * pow(c1, 2) * d0 -
          8 * a4 * b3 * b4 * pow(c0, 2) * d1 +
          16 * a5 * b0 * b1 * pow(c1, 2) * d5 +
          16 * a5 * b0 * b2 * pow(c1, 2) * d4 -
          48 * a5 * b0 * b5 * pow(c1, 2) * d1 +
          2 * a5 * b1 * b3 * pow(c0, 2) * d3 +
          16 * a5 * b1 * b5 * pow(c1, 2) * d0 -
          16 * a5 * b2 * b4 * pow(c1, 2) * d0 -
          8 * a0 * b1 * b5 * pow(c4, 2) * d3 +
          8 * a0 * b3 * b5 * pow(c4, 2) * d1 -
          8 * a1 * b0 * b5 * pow(c4, 2) * d3 +
          16 * a1 * b1 * b2 * pow(c5, 2) * d4 +
          16 * a1 * b1 * b4 * pow(c5, 2) * d2 +
          24 * a1 * b1 * b5 * pow(c5, 2) * d1 -
          16 * a1 * b2 * b4 * pow(c5, 2) * d1 -
          8 * a1 * b2 * b5 * pow(c3, 2) * d2 +
          8 * a2 * b1 * b2 * pow(c3, 2) * d5 -
          8 * a2 * b1 * b2 * pow(c5, 2) * d3 -
          16 * a2 * b1 * b4 * pow(c5, 2) * d1 -
          8 * a2 * b1 * b5 * pow(c3, 2) * d2 +
          8 * a2 * b2 * b3 * pow(c5, 2) * d1 +
          16 * a2 * b2 * b4 * pow(c3, 2) * d2 +
          8 * a2 * b2 * b5 * pow(c3, 2) * d1 +
          8 * a3 * b0 * b5 * pow(c4, 2) * d1 -
          16 * a4 * b1 * b2 * pow(c5, 2) * d1 -
          8 * a5 * b0 * b1 * pow(c4, 2) * d3 +
          8 * a5 * b0 * b3 * pow(c4, 2) * d1 -
          8 * a5 * b0 * b5 * pow(c2, 2) * d1 -
          8 * a5 * b1 * b2 * pow(c3, 2) * d2 +
          8 * a5 * b1 * b5 * pow(c2, 2) * d0 +
          6 * a0 * b1 * b3 * pow(c5, 2) * d5 +
          2 * a0 * b1 * b5 * pow(c3, 2) * d5 -
          2 * a0 * b1 * b5 * pow(c5, 2) * d3 -
          4 * a0 * b2 * b3 * pow(c5, 2) * d4 -
          4 * a0 * b2 * b4 * pow(c3, 2) * d5 +
          4 * a0 * b2 * b4 * pow(c5, 2) * d3 +
          4 * a0 * b2 * b5 * pow(c3, 2) * d4 -
          4 * a0 * b3 * b4 * pow(c5, 2) * d2 -
          2 * a0 * b3 * b5 * pow(c5, 2) * d1 +
          4 * a0 * b4 * b5 * pow(c3, 2) * d2 +
          6 * a1 * b0 * b3 * pow(c5, 2) * d5 +
          2 * a1 * b0 * b5 * pow(c3, 2) * d5 -
          2 * a1 * b0 * b5 * pow(c5, 2) * d3 +
          4 * a1 * b3 * b5 * pow(c0, 2) * d5 +
          8 * a1 * b3 * b5 * pow(c2, 2) * d3 -
          2 * a1 * b3 * b5 * pow(c5, 2) * d0 -
          8 * a1 * b4 * b5 * pow(c0, 2) * d4 -
          4 * a2 * b0 * b3 * pow(c5, 2) * d4 -
          4 * a2 * b0 * b4 * pow(c3, 2) * d5 +
          4 * a2 * b0 * b4 * pow(c5, 2) * d3 +
          4 * a2 * b0 * b5 * pow(c3, 2) * d4 -
          8 * a2 * b3 * b4 * pow(c0, 2) * d5 -
          16 * a2 * b3 * b4 * pow(c2, 2) * d3 +
          4 * a2 * b3 * b4 * pow(c5, 2) * d0 +
          8 * a2 * b4 * b5 * pow(c0, 2) * d3 -
          4 * a2 * b4 * b5 * pow(c3, 2) * d0 +
          6 * a3 * b0 * b1 * pow(c5, 2) * d5 -
          4 * a3 * b0 * b2 * pow(c5, 2) * d4 -
          4 * a3 * b0 * b4 * pow(c5, 2) * d2 -
          2 * a3 * b0 * b5 * pow(c5, 2) * d1 -
          8 * a3 * b1 * b3 * pow(c2, 2) * d5 +
          4 * a3 * b1 * b5 * pow(c0, 2) * d5 +
          8 * a3 * b1 * b5 * pow(c2, 2) * d3 -
          2 * a3 * b1 * b5 * pow(c5, 2) * d0 -
          16 * a3 * b2 * b3 * pow(c2, 2) * d4 -
          8 * a3 * b2 * b4 * pow(c0, 2) * d5 -
          16 * a3 * b2 * b4 * pow(c2, 2) * d3 +
          4 * a3 * b2 * b4 * pow(c5, 2) * d0 +
          48 * a3 * b3 * b4 * pow(c2, 2) * d2 -
          8 * a3 * b3 * b5 * pow(c2, 2) * d1 -
          4 * a4 * b0 * b2 * pow(c3, 2) * d5 +
          4 * a4 * b0 * b2 * pow(c5, 2) * d3 -
          4 * a4 * b0 * b3 * pow(c5, 2) * d2 +
          8 * a4 * b0 * b4 * pow(c5, 2) * d1 +
          4 * a4 * b0 * b5 * pow(c3, 2) * d2 +
          16 * a4 * b1 * b4 * pow(c0, 2) * d5 +
          32 * a4 * b1 * b4 * pow(c2, 2) * d3 -
          8 * a4 * b1 * b4 * pow(c5, 2) * d0 -
          8 * a4 * b1 * b5 * pow(c0, 2) * d4 -
          8 * a4 * b2 * b3 * pow(c0, 2) * d5 -
          16 * a4 * b2 * b3 * pow(c2, 2) * d3 +
          4 * a4 * b2 * b3 * pow(c5, 2) * d0 +
          8 * a4 * b2 * b5 * pow(c0, 2) * d3 -
          4 * a4 * b2 * b5 * pow(c3, 2) * d0 -
          32 * a4 * b3 * b4 * pow(c2, 2) * d1 +
          2 * a5 * b0 * b1 * pow(c3, 2) * d5 -
          2 * a5 * b0 * b1 * pow(c5, 2) * d3 +
          4 * a5 * b0 * b2 * pow(c3, 2) * d4 -
          2 * a5 * b0 * b3 * pow(c5, 2) * d1 +
          4 * a5 * b0 * b4 * pow(c3, 2) * d2 -
          6 * a5 * b0 * b5 * pow(c3, 2) * d1 +
          4 * a5 * b1 * b3 * pow(c0, 2) * d5 +
          8 * a5 * b1 * b3 * pow(c2, 2) * d3 -
          2 * a5 * b1 * b3 * pow(c5, 2) * d0 -
          8 * a5 * b1 * b4 * pow(c0, 2) * d4 -
          4 * a5 * b1 * b5 * pow(c0, 2) * d3 +
          2 * a5 * b1 * b5 * pow(c3, 2) * d0 +
          8 * a5 * b2 * b4 * pow(c0, 2) * d3 -
          4 * a5 * b2 * b4 * pow(c3, 2) * d0 -
          4 * a5 * b3 * b5 * pow(c0, 2) * d1 +
          8 * a1 * b3 * b5 * pow(c1, 2) * d5 -
          16 * a2 * b3 * b4 * pow(c1, 2) * d5 +
          8 * a3 * b1 * b5 * pow(c1, 2) * d5 -
          16 * a3 * b2 * b4 * pow(c1, 2) * d5 +
          16 * a3 * b4 * b5 * pow(c1, 2) * d2 -
          16 * a4 * b2 * b3 * pow(c1, 2) * d5 +
          16 * a4 * b3 * b5 * pow(c1, 2) * d2 +
          8 * a5 * b0 * b5 * pow(c4, 2) * d1 +
          8 * a5 * b1 * b3 * pow(c1, 2) * d5 +
          8 * a5 * b1 * b5 * pow(c1, 2) * d3 -
          8 * a5 * b1 * b5 * pow(c4, 2) * d0 +
          16 * a5 * b3 * b4 * pow(c1, 2) * d2 -
          24 * a5 * b3 * b5 * pow(c1, 2) * d1 +
          4 * a2 * b4 * b5 * pow(c0, 2) * d5 +
          4 * a4 * b2 * b5 * pow(c0, 2) * d5 -
          2 * a5 * b1 * b5 * pow(c0, 2) * d5 -
          8 * a5 * b1 * b5 * pow(c2, 2) * d3 +
          4 * a5 * b2 * b4 * pow(c0, 2) * d5 -
          4 * a5 * b2 * b5 * pow(c0, 2) * d4 +
          8 * a5 * b3 * b5 * pow(c2, 2) * d1 -
          4 * a5 * b4 * b5 * pow(c0, 2) * d2 +
          2 * a1 * b3 * b5 * pow(c5, 2) * d3 -
          4 * a2 * b3 * b4 * pow(c5, 2) * d3 +
          16 * a2 * b4 * b5 * pow(c1, 2) * d5 -
          6 * a3 * b1 * b3 * pow(c5, 2) * d5 +
          2 * a3 * b1 * b5 * pow(c5, 2) * d3 +
          4 * a3 * b2 * b3 * pow(c5, 2) * d4 -
          4 * a3 * b2 * b4 * pow(c5, 2) * d3 +
          4 * a3 * b3 * b4 * pow(c5, 2) * d2 +
          2 * a3 * b3 * b5 * pow(c5, 2) * d1 +
          8 * a4 * b1 * b4 * pow(c5, 2) * d3 -
          4 * a4 * b2 * b3 * pow(c5, 2) * d3 +
          16 * a4 * b2 * b5 * pow(c1, 2) * d5 -
          8 * a4 * b3 * b4 * pow(c5, 2) * d1 +
          2 * a5 * b1 * b3 * pow(c5, 2) * d3 -
          24 * a5 * b1 * b5 * pow(c1, 2) * d5 +
          16 * a5 * b2 * b4 * pow(c1, 2) * d5 -
          16 * a5 * b2 * b5 * pow(c1, 2) * d4 -
          16 * a5 * b4 * b5 * pow(c1, 2) * d2 +
          8 * a5 * b1 * b5 * pow(c4, 2) * d3 -
          8 * a5 * b3 * b5 * pow(c4, 2) * d1 +
          4 * a2 * b4 * b5 * pow(c3, 2) * d5 +
          4 * a4 * b2 * b5 * pow(c3, 2) * d5 -
          2 * a5 * b1 * b5 * pow(c3, 2) * d5 +
          4 * a5 * b2 * b4 * pow(c3, 2) * d5 -
          4 * a5 * b2 * b5 * pow(c3, 2) * d4 -
          4 * a5 * b4 * b5 * pow(c3, 2) * d2 -
          2 * a0 * pow(b3, 2) * c0 * c1 * d0 -
          8 * a0 * pow(b1, 2) * c0 * c1 * d3 -
          8 * a0 * pow(b1, 2) * c0 * c3 * d1 -
          8 * a0 * pow(b1, 2) * c1 * c3 * d0 +
          8 * a0 * pow(b4, 2) * c0 * c1 * d0 +
          24 * a1 * pow(b1, 2) * c0 * c3 * d0 -
          8 * a3 * pow(b1, 2) * c0 * c1 * d0 -
          2 * a0 * pow(b5, 2) * c0 * c1 * d0 +
          8 * a1 * pow(b0, 2) * c1 * c3 * d1 +
          8 * a1 * pow(b2, 2) * c0 * c3 * d0 -
          8 * a3 * pow(b2, 2) * c0 * c1 * d0 -
          6 * a0 * pow(b0, 2) * c1 * c3 * d3 +
          8 * a0 * pow(b1, 2) * c0 * c1 * d5 +
          8 * a0 * pow(b1, 2) * c0 * c5 * d1 +
          8 * a0 * pow(b1, 2) * c1 * c5 * d0 +
          2 * a1 * pow(b0, 2) * c0 * c3 * d3 -
          24 * a1 * pow(b1, 2) * c0 * c5 * d0 -
          16 * a2 * pow(b1, 2) * c0 * c4 * d0 +
          2 * a3 * pow(b0, 2) * c0 * c1 * d3 +
          2 * a3 * pow(b0, 2) * c0 * c3 * d1 +
          2 * a3 * pow(b0, 2) * c1 * c3 * d0 +
          16 * a4 * pow(b1, 2) * c0 * c2 * d0 +
          8 * a5 * pow(b1, 2) * c0 * c1 * d0 +
          16 * a0 * pow(b2, 2) * c0 * c2 * d4 +
          16 * a0 * pow(b2, 2) * c0 * c4 * d2 +
          16 * a0 * pow(b2, 2) * c2 * c4 * d0 +
          8 * a0 * pow(b3, 2) * c1 * c2 * d2 -
          8 * a1 * pow(b0, 2) * c1 * c5 * d1 +
          8 * a1 * pow(b0, 2) * c2 * c3 * d2 -
          8 * a1 * pow(b2, 2) * c0 * c5 * d0 +
          32 * a1 * pow(b2, 2) * c1 * c3 * d1 +
          8 * a1 * pow(b3, 2) * c0 * c2 * d2 +
          8 * a1 * pow(b4, 2) * c0 * c3 * d0 +
          16 * a1 * pow(b5, 2) * c0 * c1 * d1 -
          16 * a2 * pow(b0, 2) * c1 * c4 * d1 -
          48 * a2 * pow(b2, 2) * c0 * c4 * d0 -
          8 * a2 * pow(b3, 2) * c0 * c1 * d2 -
          8 * a2 * pow(b3, 2) * c0 * c2 * d1 -
          8 * a2 * pow(b3, 2) * c1 * c2 * d0 -
          8 * a3 * pow(b0, 2) * c1 * c2 * d2 -
          8 * a3 * pow(b4, 2) * c0 * c1 * d0 +
          16 * a4 * pow(b0, 2) * c1 * c2 * d1 +
          16 * a4 * pow(b2, 2) * c0 * c2 * d0 +
          8 * a5 * pow(b2, 2) * c0 * c1 * d0 +
          6 * a0 * pow(b0, 2) * c1 * c3 * d5 +
          24 * a0 * pow(b0, 2) * c1 * c4 * d4 +
          6 * a0 * pow(b0, 2) * c1 * c5 * d3 -
          12 * a0 * pow(b0, 2) * c2 * c3 * d4 -
          12 * a0 * pow(b0, 2) * c2 * c4 * d3 -
          12 * a0 * pow(b0, 2) * c3 * c4 * d2 +
          6 * a0 * pow(b0, 2) * c3 * c5 * d1 -
          8 * a0 * pow(b2, 2) * c1 * c3 * d3 +
          2 * a0 * pow(b3, 2) * c0 * c1 * d5 -
          4 * a0 * pow(b3, 2) * c0 * c2 * d4 -
          4 * a0 * pow(b3, 2) * c0 * c4 * d2 +
          2 * a0 * pow(b3, 2) * c0 * c5 * d1 +
          2 * a0 * pow(b3, 2) * c1 * c5 * d0 -
          4 * a0 * pow(b3, 2) * c2 * c4 * d0 +
          32 * a0 * pow(b4, 2) * c1 * c2 * d2 +
          4 * a0 * pow(b5, 2) * c0 * c1 * d3 +
          4 * a0 * pow(b5, 2) * c0 * c3 * d1 +
          4 * a0 * pow(b5, 2) * c1 * c3 * d0 -
          2 * a1 * pow(b0, 2) * c0 * c3 * d5 -
          8 * a1 * pow(b0, 2) * c0 * c4 * d4 -
          2 * a1 * pow(b0, 2) * c0 * c5 * d3 -
          2 * a1 * pow(b0, 2) * c3 * c5 * d0 +
          96 * a1 * pow(b1, 2) * c2 * c3 * d2 -
          8 * a1 * pow(b2, 2) * c0 * c3 * d3 -
          6 * a1 * pow(b3, 2) * c0 * c5 * d0 -
          32 * a1 * pow(b4, 2) * c0 * c2 * d2 -
          4 * a1 * pow(b5, 2) * c0 * c3 * d0 +
          4 * a2 * pow(b0, 2) * c0 * c3 * d4 +
          4 * a2 * pow(b0, 2) * c0 * c4 * d3 +
          4 * a2 * pow(b0, 2) * c3 * c4 * d0 -
          32 * a2 * pow(b1, 2) * c1 * c2 * d3 -
          32 * a2 * pow(b1, 2) * c1 * c3 * d2 -
          32 * a2 * pow(b1, 2) * c2 * c3 * d1 +
          4 * a2 * pow(b3, 2) * c0 * c4 * d0 -
          2 * a3 * pow(b0, 2) * c0 * c1 * d5 +
          4 * a3 * pow(b0, 2) * c0 * c2 * d4 +
          4 * a3 * pow(b0, 2) * c0 * c4 * d2 -
          2 * a3 * pow(b0, 2) * c0 * c5 * d1 -
          2 * a3 * pow(b0, 2) * c1 * c5 * d0 +
          4 * a3 * pow(b0, 2) * c2 * c4 * d0 -
          32 * a3 * pow(b1, 2) * c1 * c2 * d2 +
          8 * a3 * pow(b2, 2) * c0 * c1 * d3 +
          8 * a3 * pow(b2, 2) * c0 * c3 * d1 +
          8 * a3 * pow(b2, 2) * c1 * c3 * d0 -
          4 * a3 * pow(b5, 2) * c0 * c1 * d0 -
          8 * a4 * pow(b0, 2) * c0 * c1 * d4 +
          4 * a4 * pow(b0, 2) * c0 * c2 * d3 +
          4 * a4 * pow(b0, 2) * c0 * c3 * d2 -
          8 * a4 * pow(b0, 2) * c0 * c4 * d1 -
          8 * a4 * pow(b0, 2) * c1 * c4 * d0 +
          4 * a4 * pow(b0, 2) * c2 * c3 * d0 +
          4 * a4 * pow(b3, 2) * c0 * c2 * d0 -
          2 * a5 * pow(b0, 2) * c0 * c1 * d3 -
          2 * a5 * pow(b0, 2) * c0 * c3 * d1 -
          2 * a5 * pow(b0, 2) * c1 * c3 * d0 +
          2 * a5 * pow(b3, 2) * c0 * c1 * d0 +
          8 * a0 * pow(b1, 2) * c1 * c3 * d5 +
          8 * a0 * pow(b1, 2) * c1 * c5 * d3 -
          16 * a0 * pow(b1, 2) * c2 * c3 * d4 -
          16 * a0 * pow(b1, 2) * c2 * c4 * d3 -
          16 * a0 * pow(b1, 2) * c3 * c4 * d2 +
          8 * a0 * pow(b1, 2) * c3 * c5 * d1 -
          8 * a0 * pow(b4, 2) * c0 * c1 * d5 -
          8 * a0 * pow(b4, 2) * c0 * c5 * d1 -
          8 * a0 * pow(b4, 2) * c1 * c5 * d0 -
          8 * a0 * pow(b5, 2) * c1 * c2 * d2 -
          8 * a1 * pow(b0, 2) * c2 * c5 * d2 -
          24 * a1 * pow(b1, 2) * c0 * c3 * d5 -
          24 * a1 * pow(b1, 2) * c0 * c5 * d3 -
          24 * a1 * pow(b1, 2) * c3 * c5 * d0 +
          64 * a1 * pow(b2, 2) * c1 * c2 * d4 +
          64 * a1 * pow(b2, 2) * c1 * c4 * d2 -
          32 * a1 * pow(b2, 2) * c1 * c5 * d1 +
          64 * a1 * pow(b2, 2) * c2 * c4 * d1 +
          16 * a1 * pow(b4, 2) * c0 * c5 * d0 +
          8 * a1 * pow(b5, 2) * c0 * c2 * d2 -
          16 * a2 * pow(b0, 2) * c2 * c4 * d2 +
          16 * a2 * pow(b1, 2) * c0 * c3 * d4 +
          16 * a2 * pow(b1, 2) * c0 * c4 * d3 +
          16 * a2 * pow(b1, 2) * c3 * c4 * d0 -
          192 * a2 * pow(b2, 2) * c1 * c4 * d1 +
          8 * a3 * pow(b1, 2) * c0 * c1 * d5 +
          8 * a3 * pow(b1, 2) * c0 * c5 * d1 +
          8 * a3 * pow(b1, 2) * c1 * c5 * d0 +
          64 * a4 * pow(b2, 2) * c1 * c2 * d1 +
          8 * a5 * pow(b0, 2) * c1 * c2 * d2 +
          8 * a5 * pow(b1, 2) * c0 * c1 * d3 +
          8 * a5 * pow(b1, 2) * c0 * c3 * d1 +
          8 * a5 * pow(b1, 2) * c1 * c3 * d0 -
          6 * a0 * pow(b0, 2) * c1 * c5 * d5 +
          12 * a0 * pow(b0, 2) * c2 * c4 * d5 +
          12 * a0 * pow(b0, 2) * c2 * c5 * d4 +
          12 * a0 * pow(b0, 2) * c4 * c5 * d2 +
          8 * a0 * pow(b2, 2) * c1 * c3 * d5 +
          32 * a0 * pow(b2, 2) * c1 * c4 * d4 +
          8 * a0 * pow(b2, 2) * c1 * c5 * d3 -
          16 * a0 * pow(b2, 2) * c2 * c3 * d4 -
          16 * a0 * pow(b2, 2) * c2 * c4 * d3 -
          16 * a0 * pow(b2, 2) * c3 * c4 * d2 +
          8 * a0 * pow(b2, 2) * c3 * c5 * d1 -
          2 * a0 * pow(b5, 2) * c0 * c1 * d5 +
          4 * a0 * pow(b5, 2) * c0 * c2 * d4 +
          4 * a0 * pow(b5, 2) * c0 * c4 * d2 -
          2 * a0 * pow(b5, 2) * c0 * c5 * d1 -
          2 * a0 * pow(b5, 2) * c1 * c5 * d0 +
          4 * a0 * pow(b5, 2) * c2 * c4 * d0 +
          2 * a1 * pow(b0, 2) * c0 * c5 * d5 -
          96 * a1 * pow(b1, 2) * c2 * c5 * d2 -
          32 * a1 * pow(b2, 2) * c0 * c4 * d4 -
          2 * a1 * pow(b5, 2) * c0 * c5 * d0 +
          8 * a1 * pow(b5, 2) * c1 * c3 * d1 -
          4 * a2 * pow(b0, 2) * c0 * c4 * d5 -
          4 * a2 * pow(b0, 2) * c0 * c5 * d4 -
          4 * a2 * pow(b0, 2) * c4 * c5 * d0 +
          32 * a2 * pow(b1, 2) * c1 * c2 * d5 +
          32 * a2 * pow(b1, 2) * c1 * c5 * d2 -
          64 * a2 * pow(b1, 2) * c2 * c4 * d2 +
          32 * a2 * pow(b1, 2) * c2 * c5 * d1 +
          48 * a2 * pow(b2, 2) * c0 * c3 * d4 +
          48 * a2 * pow(b2, 2) * c0 * c4 * d3 +
          48 * a2 * pow(b2, 2) * c3 * c4 * d0 -
          4 * a2 * pow(b5, 2) * c0 * c4 * d0 -
          16 * a3 * pow(b2, 2) * c0 * c2 * d4 -
          16 * a3 * pow(b2, 2) * c0 * c4 * d2 -
          16 * a3 * pow(b2, 2) * c2 * c4 * d0 -
          4 * a4 * pow(b0, 2) * c0 * c2 * d5 -
          4 * a4 * pow(b0, 2) * c0 * c5 * d2 -
          4 * a4 * pow(b0, 2) * c2 * c5 * d0 -
          16 * a4 * pow(b2, 2) * c0 * c2 * d3 -
          16 * a4 * pow(b2, 2) * c0 * c3 * d2 -
          16 * a4 * pow(b2, 2) * c2 * c3 * d0 -
          4 * a4 * pow(b5, 2) * c0 * c2 * d0 +
          2 * a5 * pow(b0, 2) * c0 * c1 * d5 -
          4 * a5 * pow(b0, 2) * c0 * c2 * d4 -
          4 * a5 * pow(b0, 2) * c0 * c4 * d2 +
          2 * a5 * pow(b0, 2) * c0 * c5 * d1 +
          2 * a5 * pow(b0, 2) * c1 * c5 * d0 -
          4 * a5 * pow(b0, 2) * c2 * c4 * d0 +
          32 * a5 * pow(b1, 2) * c1 * c2 * d2 -
          8 * a5 * pow(b2, 2) * c0 * c1 * d3 -
          8 * a5 * pow(b2, 2) * c0 * c3 * d1 -
          8 * a5 * pow(b2, 2) * c1 * c3 * d0 +
          6 * a5 * pow(b5, 2) * c0 * c1 * d0 -
          16 * a0 * pow(b1, 2) * c1 * c5 * d5 +
          16 * a0 * pow(b1, 2) * c2 * c4 * d5 +
          16 * a0 * pow(b1, 2) * c2 * c5 * d4 +
          16 * a0 * pow(b1, 2) * c4 * c5 * d2 -
          6 * a0 * pow(b5, 2) * c1 * c3 * d3 +
          8 * a1 * pow(b0, 2) * c3 * c4 * d4 -
          2 * a1 * pow(b0, 2) * c3 * c5 * d3 +
          48 * a1 * pow(b1, 2) * c0 * c5 * d5 +
          32 * a1 * pow(b4, 2) * c2 * c3 * d2 +
          2 * a1 * pow(b5, 2) * c0 * c3 * d3 -
          4 * a2 * pow(b0, 2) * c3 * c4 * d3 -
          2 * a3 * pow(b0, 2) * c1 * c3 * d5 -
          8 * a3 * pow(b0, 2) * c1 * c4 * d4 -
          2 * a3 * pow(b0, 2) * c1 * c5 * d3 +
          4 * a3 * pow(b0, 2) * c2 * c3 * d4 +
          4 * a3 * pow(b0, 2) * c2 * c4 * d3 +
          4 * a3 * pow(b0, 2) * c3 * c4 * d2 -
          2 * a3 * pow(b0, 2) * c3 * c5 * d1 -
          32 * a3 * pow(b4, 2) * c1 * c2 * d2 +
          2 * a3 * pow(b5, 2) * c0 * c1 * d3 +
          2 * a3 * pow(b5, 2) * c0 * c3 * d1 +
          2 * a3 * pow(b5, 2) * c1 * c3 * d0 -
          4 * a4 * pow(b0, 2) * c2 * c3 * d3 -
          16 * a4 * pow(b1, 2) * c0 * c2 * d5 -
          16 * a4 * pow(b1, 2) * c0 * c5 * d2 -
          16 * a4 * pow(b1, 2) * c2 * c5 * d0 +
          6 * a5 * pow(b0, 2) * c1 * c3 * d3 -
          16 * a5 * pow(b1, 2) * c0 * c1 * d5 -
          16 * a5 * pow(b1, 2) * c0 * c5 * d1 -
          16 * a5 * pow(b1, 2) * c1 * c5 * d0 -
          8 * a0 * pow(b2, 2) * c1 * c5 * d5 +
          8 * a1 * pow(b2, 2) * c0 * c5 * d5 -
          8 * a1 * pow(b3, 2) * c2 * c5 * d2 -
          8 * a1 * pow(b4, 2) * c0 * c3 * d5 -
          8 * a1 * pow(b4, 2) * c0 * c5 * d3 -
          8 * a1 * pow(b4, 2) * c3 * c5 * d0 +
          16 * a1 * pow(b5, 2) * c1 * c2 * d4 +
          16 * a1 * pow(b5, 2) * c1 * c4 * d2 -
          24 * a1 * pow(b5, 2) * c1 * c5 * d1 -
          8 * a1 * pow(b5, 2) * c2 * c3 * d2 +
          16 * a1 * pow(b5, 2) * c2 * c4 * d1 +
          8 * a2 * pow(b3, 2) * c1 * c2 * d5 +
          8 * a2 * pow(b3, 2) * c1 * c5 * d2 -
          16 * a2 * pow(b3, 2) * c2 * c4 * d2 +
          8 * a2 * pow(b3, 2) * c2 * c5 * d1 -
          16 * a2 * pow(b5, 2) * c1 * c4 * d1 +
          8 * a3 * pow(b4, 2) * c0 * c1 * d5 +
          8 * a3 * pow(b4, 2) * c0 * c5 * d1 +
          8 * a3 * pow(b4, 2) * c1 * c5 * d0 +
          8 * a3 * pow(b5, 2) * c1 * c2 * d2 -
          16 * a4 * pow(b5, 2) * c1 * c2 * d1 -
          8 * a5 * pow(b3, 2) * c1 * c2 * d2 -
          2 * a0 * pow(b3, 2) * c1 * c5 * d5 +
          4 * a0 * pow(b3, 2) * c2 * c4 * d5 +
          4 * a0 * pow(b3, 2) * c2 * c5 * d4 +
          4 * a0 * pow(b3, 2) * c4 * c5 * d2 +
          2 * a0 * pow(b5, 2) * c1 * c3 * d5 +
          8 * a0 * pow(b5, 2) * c1 * c4 * d4 +
          2 * a0 * pow(b5, 2) * c1 * c5 * d3 -
          4 * a0 * pow(b5, 2) * c2 * c3 * d4 -
          4 * a0 * pow(b5, 2) * c2 * c4 * d3 -
          4 * a0 * pow(b5, 2) * c3 * c4 * d2 +
          2 * a0 * pow(b5, 2) * c3 * c5 * d1 +
          4 * a1 * pow(b0, 2) * c3 * c5 * d5 +
          32 * a1 * pow(b2, 2) * c3 * c4 * d4 +
          8 * a1 * pow(b2, 2) * c3 * c5 * d3 +
          6 * a1 * pow(b3, 2) * c0 * c5 * d5 +
          2 * a1 * pow(b5, 2) * c0 * c3 * d5 -
          8 * a1 * pow(b5, 2) * c0 * c4 * d4 +
          2 * a1 * pow(b5, 2) * c0 * c5 * d3 +
          2 * a1 * pow(b5, 2) * c3 * c5 * d0 -
          48 * a2 * pow(b2, 2) * c3 * c4 * d3 -
          4 * a2 * pow(b3, 2) * c0 * c4 * d5 -
          4 * a2 * pow(b3, 2) * c0 * c5 * d4 -
          4 * a2 * pow(b3, 2) * c4 * c5 * d0 +
          4 * a2 * pow(b5, 2) * c0 * c3 * d4 +
          4 * a2 * pow(b5, 2) * c0 * c4 * d3 +
          4 * a2 * pow(b5, 2) * c3 * c4 * d0 +
          4 * a3 * pow(b0, 2) * c1 * c5 * d5 -
          8 * a3 * pow(b0, 2) * c2 * c4 * d5 -
          8 * a3 * pow(b0, 2) * c2 * c5 * d4 -
          8 * a3 * pow(b0, 2) * c4 * c5 * d2 -
          8 * a3 * pow(b2, 2) * c1 * c3 * d5 -
          32 * a3 * pow(b2, 2) * c1 * c4 * d4 -
          8 * a3 * pow(b2, 2) * c1 * c5 * d3 +
          16 * a3 * pow(b2, 2) * c2 * c3 * d4 +
          16 * a3 * pow(b2, 2) * c2 * c4 * d3 +
          16 * a3 * pow(b2, 2) * c3 * c4 * d2 -
          8 * a3 * pow(b2, 2) * c3 * c5 * d1 +
          2 * a3 * pow(b5, 2) * c0 * c1 * d5 -
          4 * a3 * pow(b5, 2) * c0 * c2 * d4 -
          4 * a3 * pow(b5, 2) * c0 * c4 * d2 +
          2 * a3 * pow(b5, 2) * c0 * c5 * d1 +
          2 * a3 * pow(b5, 2) * c1 * c5 * d0 -
          4 * a3 * pow(b5, 2) * c2 * c4 * d0 +
          8 * a4 * pow(b0, 2) * c1 * c4 * d5 +
          8 * a4 * pow(b0, 2) * c1 * c5 * d4 +
          8 * a4 * pow(b0, 2) * c4 * c5 * d1 +
          16 * a4 * pow(b2, 2) * c2 * c3 * d3 -
          4 * a4 * pow(b3, 2) * c0 * c2 * d5 -
          4 * a4 * pow(b3, 2) * c0 * c5 * d2 -
          4 * a4 * pow(b3, 2) * c2 * c5 * d0 +
          4 * a4 * pow(b5, 2) * c0 * c2 * d3 +
          4 * a4 * pow(b5, 2) * c0 * c3 * d2 +
          4 * a4 * pow(b5, 2) * c2 * c3 * d0 -
          4 * a5 * pow(b0, 2) * c1 * c3 * d5 -
          16 * a5 * pow(b0, 2) * c1 * c4 * d4 -
          4 * a5 * pow(b0, 2) * c1 * c5 * d3 +
          8 * a5 * pow(b0, 2) * c2 * c3 * d4 +
          8 * a5 * pow(b0, 2) * c2 * c4 * d3 +
          8 * a5 * pow(b0, 2) * c3 * c4 * d2 -
          4 * a5 * pow(b0, 2) * c3 * c5 * d1 +
          8 * a5 * pow(b2, 2) * c1 * c3 * d3 -
          2 * a5 * pow(b3, 2) * c0 * c1 * d5 +
          4 * a5 * pow(b3, 2) * c0 * c2 * d4 +
          4 * a5 * pow(b3, 2) * c0 * c4 * d2 -
          2 * a5 * pow(b3, 2) * c0 * c5 * d1 -
          2 * a5 * pow(b3, 2) * c1 * c5 * d0 +
          4 * a5 * pow(b3, 2) * c2 * c4 * d0 -
          6 * a5 * pow(b5, 2) * c0 * c1 * d3 -
          6 * a5 * pow(b5, 2) * c0 * c3 * d1 -
          6 * a5 * pow(b5, 2) * c1 * c3 * d0 +
          8 * a0 * pow(b4, 2) * c1 * c5 * d5 +
          24 * a1 * pow(b1, 2) * c3 * c5 * d5 -
          8 * a1 * pow(b4, 2) * c0 * c5 * d5 -
          16 * a2 * pow(b1, 2) * c3 * c4 * d5 -
          16 * a2 * pow(b1, 2) * c3 * c5 * d4 -
          16 * a2 * pow(b1, 2) * c4 * c5 * d3 -
          8 * a3 * pow(b1, 2) * c1 * c5 * d5 -
          8 * a5 * pow(b1, 2) * c1 * c3 * d5 -
          8 * a5 * pow(b1, 2) * c1 * c5 * d3 +
          16 * a5 * pow(b1, 2) * c2 * c3 * d4 +
          16 * a5 * pow(b1, 2) * c2 * c4 * d3 +
          16 * a5 * pow(b1, 2) * c3 * c4 * d2 -
          8 * a5 * pow(b1, 2) * c3 * c5 * d1 -
          8 * a1 * pow(b2, 2) * c3 * c5 * d5 +
          4 * a2 * pow(b0, 2) * c4 * c5 * d5 +
          8 * a3 * pow(b2, 2) * c1 * c5 * d5 +
          4 * a4 * pow(b0, 2) * c2 * c5 * d5 +
          2 * a5 * pow(b0, 2) * c1 * c5 * d5 -
          4 * a5 * pow(b0, 2) * c2 * c4 * d5 -
          4 * a5 * pow(b0, 2) * c2 * c5 * d4 -
          4 * a5 * pow(b0, 2) * c4 * c5 * d2 +
          8 * a1 * pow(b5, 2) * c3 * c4 * d4 -
          2 * a1 * pow(b5, 2) * c3 * c5 * d3 +
          16 * a2 * pow(b1, 2) * c4 * c5 * d5 -
          4 * a2 * pow(b5, 2) * c3 * c4 * d3 -
          2 * a3 * pow(b5, 2) * c1 * c3 * d5 -
          8 * a3 * pow(b5, 2) * c1 * c4 * d4 -
          2 * a3 * pow(b5, 2) * c1 * c5 * d3 +
          4 * a3 * pow(b5, 2) * c2 * c3 * d4 +
          4 * a3 * pow(b5, 2) * c2 * c4 * d3 +
          4 * a3 * pow(b5, 2) * c3 * c4 * d2 -
          2 * a3 * pow(b5, 2) * c3 * c5 * d1 +
          16 * a4 * pow(b1, 2) * c2 * c5 * d5 -
          4 * a4 * pow(b5, 2) * c2 * c3 * d3 +
          24 * a5 * pow(b1, 2) * c1 * c5 * d5 -
          16 * a5 * pow(b1, 2) * c2 * c4 * d5 -
          16 * a5 * pow(b1, 2) * c2 * c5 * d4 -
          16 * a5 * pow(b1, 2) * c4 * c5 * d2 +
          6 * a5 * pow(b5, 2) * c1 * c3 * d3 +
          8 * a1 * pow(b4, 2) * c3 * c5 * d5 -
          8 * a3 * pow(b4, 2) * c1 * c5 * d5 +
          4 * a2 * pow(b3, 2) * c4 * c5 * d5 +
          4 * a4 * pow(b3, 2) * c2 * c5 * d5 +
          2 * a5 * pow(b3, 2) * c1 * c5 * d5 -
          4 * a5 * pow(b3, 2) * c2 * c4 * d5 -
          4 * a5 * pow(b3, 2) * c2 * c5 * d4 -
          4 * a5 * pow(b3, 2) * c4 * c5 * d2 +
          16 * a0 * b0 * b1 * c1 * c3 * d1 + 16 * a0 * b1 * b3 * c0 * c1 * d1 -
          16 * a1 * b0 * b1 * c0 * c1 * d3 - 16 * a1 * b0 * b1 * c0 * c3 * d1 -
          16 * a1 * b0 * b1 * c1 * c3 * d0 + 16 * a1 * b0 * b3 * c0 * c1 * d1 -
          16 * a1 * b1 * b3 * c0 * c1 * d0 + 16 * a3 * b0 * b1 * c0 * c1 * d1 +
          4 * a0 * b0 * b1 * c0 * c3 * d3 + 4 * a0 * b0 * b3 * c0 * c1 * d3 +
          4 * a0 * b0 * b3 * c0 * c3 * d1 + 4 * a0 * b0 * b3 * c1 * c3 * d0 -
          4 * a0 * b1 * b3 * c0 * c3 * d0 - 4 * a1 * b0 * b3 * c0 * c3 * d0 -
          4 * a3 * b0 * b1 * c0 * c3 * d0 - 4 * a3 * b0 * b3 * c0 * c1 * d0 -
          16 * a0 * b0 * b1 * c1 * c5 * d1 + 16 * a0 * b0 * b1 * c2 * c3 * d2 -
          32 * a0 * b0 * b2 * c1 * c4 * d1 - 16 * a0 * b0 * b3 * c1 * c2 * d2 +
          32 * a0 * b0 * b4 * c1 * c2 * d1 + 16 * a0 * b1 * b2 * c0 * c1 * d4 -
          8 * a0 * b1 * b2 * c0 * c2 * d3 - 8 * a0 * b1 * b2 * c0 * c3 * d2 +
          16 * a0 * b1 * b2 * c0 * c4 * d1 + 16 * a0 * b1 * b2 * c1 * c4 * d0 -
          8 * a0 * b1 * b2 * c2 * c3 * d0 - 16 * a0 * b1 * b4 * c0 * c1 * d2 -
          16 * a0 * b1 * b4 * c0 * c2 * d1 - 16 * a0 * b1 * b4 * c1 * c2 * d0 -
          16 * a0 * b1 * b5 * c0 * c1 * d1 + 8 * a0 * b2 * b3 * c0 * c1 * d2 +
          8 * a0 * b2 * b3 * c0 * c2 * d1 + 8 * a0 * b2 * b3 * c1 * c2 * d0 +
          16 * a1 * b0 * b1 * c0 * c1 * d5 + 16 * a1 * b0 * b1 * c0 * c5 * d1 +
          16 * a1 * b0 * b1 * c1 * c5 * d0 + 16 * a1 * b0 * b2 * c0 * c1 * d4 -
          8 * a1 * b0 * b2 * c0 * c2 * d3 - 8 * a1 * b0 * b2 * c0 * c3 * d2 +
          16 * a1 * b0 * b2 * c0 * c4 * d1 + 16 * a1 * b0 * b2 * c1 * c4 * d0 -
          8 * a1 * b0 * b2 * c2 * c3 * d0 - 16 * a1 * b0 * b4 * c0 * c1 * d2 -
          16 * a1 * b0 * b4 * c0 * c2 * d1 - 16 * a1 * b0 * b4 * c1 * c2 * d0 -
          16 * a1 * b0 * b5 * c0 * c1 * d1 - 32 * a1 * b1 * b2 * c0 * c4 * d0 +
          32 * a1 * b1 * b4 * c0 * c2 * d0 + 16 * a1 * b1 * b5 * c0 * c1 * d0 +
          16 * a2 * b0 * b1 * c0 * c1 * d4 - 8 * a2 * b0 * b1 * c0 * c2 * d3 -
          8 * a2 * b0 * b1 * c0 * c3 * d2 + 16 * a2 * b0 * b1 * c0 * c4 * d1 +
          16 * a2 * b0 * b1 * c1 * c4 * d0 - 8 * a2 * b0 * b1 * c2 * c3 * d0 +
          8 * a2 * b0 * b3 * c0 * c1 * d2 + 8 * a2 * b0 * b3 * c0 * c2 * d1 +
          8 * a2 * b0 * b3 * c1 * c2 * d0 + 16 * a2 * b1 * b2 * c0 * c3 * d0 -
          16 * a2 * b2 * b3 * c0 * c1 * d0 + 8 * a3 * b0 * b2 * c0 * c1 * d2 +
          8 * a3 * b0 * b2 * c0 * c2 * d1 + 8 * a3 * b0 * b2 * c1 * c2 * d0 -
          16 * a4 * b0 * b1 * c0 * c1 * d2 - 16 * a4 * b0 * b1 * c0 * c2 * d1 -
          16 * a4 * b0 * b1 * c1 * c2 * d0 - 16 * a5 * b0 * b1 * c0 * c1 * d1 -
          4 * a0 * b0 * b1 * c0 * c3 * d5 - 16 * a0 * b0 * b1 * c0 * c4 * d4 -
          4 * a0 * b0 * b1 * c0 * c5 * d3 - 4 * a0 * b0 * b1 * c3 * c5 * d0 +
          8 * a0 * b0 * b2 * c0 * c3 * d4 + 8 * a0 * b0 * b2 * c0 * c4 * d3 +
          8 * a0 * b0 * b2 * c3 * c4 * d0 - 4 * a0 * b0 * b3 * c0 * c1 * d5 +
          8 * a0 * b0 * b3 * c0 * c2 * d4 + 8 * a0 * b0 * b3 * c0 * c4 * d2 -
          4 * a0 * b0 * b3 * c0 * c5 * d1 - 4 * a0 * b0 * b3 * c1 * c5 * d0 +
          8 * a0 * b0 * b3 * c2 * c4 * d0 - 16 * a0 * b0 * b4 * c0 * c1 * d4 +
          8 * a0 * b0 * b4 * c0 * c2 * d3 + 8 * a0 * b0 * b4 * c0 * c3 * d2 -
          16 * a0 * b0 * b4 * c0 * c4 * d1 - 16 * a0 * b0 * b4 * c1 * c4 * d0 +
          8 * a0 * b0 * b4 * c2 * c3 * d0 - 4 * a0 * b0 * b5 * c0 * c1 * d3 -
          4 * a0 * b0 * b5 * c0 * c3 * d1 - 4 * a0 * b0 * b5 * c1 * c3 * d0 +
          4 * a0 * b1 * b3 * c0 * c5 * d0 + 16 * a0 * b1 * b4 * c0 * c4 * d0 +
          4 * a0 * b1 * b5 * c0 * c3 * d0 - 8 * a0 * b2 * b3 * c0 * c4 * d0 -
          8 * a0 * b2 * b4 * c0 * c3 * d0 - 8 * a0 * b3 * b4 * c0 * c2 * d0 +
          4 * a0 * b3 * b5 * c0 * c1 * d0 + 4 * a1 * b0 * b3 * c0 * c5 * d0 +
          16 * a1 * b0 * b4 * c0 * c4 * d0 + 4 * a1 * b0 * b5 * c0 * c3 * d0 -
          8 * a2 * b0 * b3 * c0 * c4 * d0 - 8 * a2 * b0 * b4 * c0 * c3 * d0 +
          4 * a3 * b0 * b1 * c0 * c5 * d0 - 8 * a3 * b0 * b2 * c0 * c4 * d0 -
          8 * a3 * b0 * b4 * c0 * c2 * d0 + 4 * a3 * b0 * b5 * c0 * c1 * d0 +
          16 * a4 * b0 * b1 * c0 * c4 * d0 - 8 * a4 * b0 * b2 * c0 * c3 * d0 -
          8 * a4 * b0 * b3 * c0 * c2 * d0 + 16 * a4 * b0 * b4 * c0 * c1 * d0 +
          4 * a5 * b0 * b1 * c0 * c3 * d0 + 4 * a5 * b0 * b3 * c0 * c1 * d0 -
          16 * a0 * b0 * b1 * c2 * c5 * d2 - 32 * a0 * b0 * b2 * c2 * c4 * d2 +
          16 * a0 * b0 * b5 * c1 * c2 * d2 + 8 * a0 * b1 * b2 * c0 * c2 * d5 +
          8 * a0 * b1 * b2 * c0 * c5 * d2 + 8 * a0 * b1 * b2 * c2 * c5 * d0 -
          32 * a0 * b2 * b4 * c0 * c2 * d2 - 8 * a0 * b2 * b5 * c0 * c1 * d2 -
          8 * a0 * b2 * b5 * c0 * c2 * d1 - 8 * a0 * b2 * b5 * c1 * c2 * d0 +
          8 * a1 * b0 * b2 * c0 * c2 * d5 + 8 * a1 * b0 * b2 * c0 * c5 * d2 +
          8 * a1 * b0 * b2 * c2 * c5 * d0 - 64 * a1 * b1 * b2 * c1 * c2 * d3 -
          64 * a1 * b1 * b2 * c1 * c3 * d2 - 64 * a1 * b1 * b2 * c2 * c3 * d1 -
          64 * a1 * b1 * b3 * c1 * c2 * d2 + 64 * a1 * b2 * b3 * c1 * c2 * d1 +
          8 * a2 * b0 * b1 * c0 * c2 * d5 + 8 * a2 * b0 * b1 * c0 * c5 * d2 +
          8 * a2 * b0 * b1 * c2 * c5 * d0 + 32 * a2 * b0 * b2 * c0 * c2 * d4 +
          32 * a2 * b0 * b2 * c0 * c4 * d2 + 32 * a2 * b0 * b2 * c2 * c4 * d0 -
          32 * a2 * b0 * b4 * c0 * c2 * d2 - 8 * a2 * b0 * b5 * c0 * c1 * d2 -
          8 * a2 * b0 * b5 * c0 * c2 * d1 - 8 * a2 * b0 * b5 * c1 * c2 * d0 -
          16 * a2 * b1 * b2 * c0 * c5 * d0 + 64 * a2 * b1 * b2 * c1 * c3 * d1 +
          64 * a2 * b1 * b3 * c1 * c2 * d1 + 32 * a2 * b2 * b4 * c0 * c2 * d0 +
          16 * a2 * b2 * b5 * c0 * c1 * d0 + 64 * a3 * b1 * b2 * c1 * c2 * d1 -
          32 * a4 * b0 * b2 * c0 * c2 * d2 - 8 * a5 * b0 * b2 * c0 * c1 * d2 -
          8 * a5 * b0 * b2 * c0 * c2 * d1 - 8 * a5 * b0 * b2 * c1 * c2 * d0 +
          4 * a0 * b0 * b1 * c0 * c5 * d5 - 8 * a0 * b0 * b2 * c0 * c4 * d5 -
          8 * a0 * b0 * b2 * c0 * c5 * d4 - 8 * a0 * b0 * b2 * c4 * c5 * d0 -
          8 * a0 * b0 * b4 * c0 * c2 * d5 - 8 * a0 * b0 * b4 * c0 * c5 * d2 -
          8 * a0 * b0 * b4 * c2 * c5 * d0 + 4 * a0 * b0 * b5 * c0 * c1 * d5 -
          8 * a0 * b0 * b5 * c0 * c2 * d4 - 8 * a0 * b0 * b5 * c0 * c4 * d2 +
          4 * a0 * b0 * b5 * c0 * c5 * d1 + 4 * a0 * b0 * b5 * c1 * c5 * d0 -
          8 * a0 * b0 * b5 * c2 * c4 * d0 + 16 * a0 * b1 * b2 * c2 * c3 * d3 +
          16 * a0 * b1 * b3 * c1 * c2 * d4 + 16 * a0 * b1 * b3 * c1 * c4 * d2 -
          16 * a0 * b1 * b3 * c1 * c5 * d1 - 16 * a0 * b1 * b3 * c2 * c3 * d2 +
          16 * a0 * b1 * b3 * c2 * c4 * d1 + 16 * a0 * b1 * b4 * c1 * c2 * d3 +
          16 * a0 * b1 * b4 * c1 * c3 * d2 + 16 * a0 * b1 * b4 * c2 * c3 * d1 -
          4 * a0 * b1 * b5 * c0 * c5 * d0 - 16 * a0 * b1 * b5 * c1 * c3 * d1 +
          8 * a0 * b2 * b4 * c0 * c5 * d0 + 8 * a0 * b2 * b5 * c0 * c4 * d0 -
          32 * a0 * b3 * b4 * c1 * c2 * d1 + 8 * a0 * b4 * b5 * c0 * c2 * d0 +
          16 * a1 * b0 * b1 * c1 * c3 * d5 + 16 * a1 * b0 * b1 * c1 * c5 * d3 -
          32 * a1 * b0 * b1 * c2 * c3 * d4 - 32 * a1 * b0 * b1 * c2 * c4 * d3 -
          32 * a1 * b0 * b1 * c3 * c4 * d2 + 16 * a1 * b0 * b1 * c3 * c5 * d1 +
          16 * a1 * b0 * b2 * c2 * c3 * d3 + 16 * a1 * b0 * b3 * c1 * c2 * d4 +
          16 * a1 * b0 * b3 * c1 * c4 * d2 - 16 * a1 * b0 * b3 * c1 * c5 * d1 -
          16 * a1 * b0 * b3 * c2 * c3 * d2 + 16 * a1 * b0 * b3 * c2 * c4 * d1 +
          16 * a1 * b0 * b4 * c1 * c2 * d3 + 16 * a1 * b0 * b4 * c1 * c3 * d2 +
          16 * a1 * b0 * b4 * c2 * c3 * d1 - 4 * a1 * b0 * b5 * c0 * c5 * d0 -
          16 * a1 * b0 * b5 * c1 * c3 * d1 + 32 * a1 * b1 * b2 * c0 * c3 * d4 +
          32 * a1 * b1 * b2 * c0 * c4 * d3 + 32 * a1 * b1 * b2 * c3 * c4 * d0 +
          16 * a1 * b1 * b3 * c0 * c1 * d5 + 16 * a1 * b1 * b3 * c0 * c5 * d1 +
          16 * a1 * b1 * b3 * c1 * c5 * d0 + 16 * a1 * b1 * b5 * c0 * c1 * d3 +
          16 * a1 * b1 * b5 * c0 * c3 * d1 + 16 * a1 * b1 * b5 * c1 * c3 * d0 -
          16 * a1 * b2 * b3 * c0 * c1 * d4 - 16 * a1 * b2 * b3 * c0 * c4 * d1 -
          16 * a1 * b2 * b3 * c1 * c4 * d0 - 16 * a1 * b2 * b4 * c0 * c1 * d3 -
          16 * a1 * b2 * b4 * c0 * c3 * d1 - 16 * a1 * b2 * b4 * c1 * c3 * d0 -
          16 * a1 * b3 * b5 * c0 * c1 * d1 + 16 * a2 * b0 * b1 * c2 * c3 * d3 -
          16 * a2 * b0 * b2 * c1 * c3 * d3 + 8 * a2 * b0 * b4 * c0 * c5 * d0 +
          8 * a2 * b0 * b5 * c0 * c4 * d0 - 16 * a2 * b1 * b2 * c0 * c3 * d3 -
          16 * a2 * b1 * b3 * c0 * c1 * d4 - 16 * a2 * b1 * b3 * c0 * c4 * d1 -
          16 * a2 * b1 * b3 * c1 * c4 * d0 - 16 * a2 * b1 * b4 * c0 * c1 * d3 -
          16 * a2 * b1 * b4 * c0 * c3 * d1 - 16 * a2 * b1 * b4 * c1 * c3 * d0 +
          16 * a2 * b2 * b3 * c0 * c1 * d3 + 16 * a2 * b2 * b3 * c0 * c3 * d1 +
          16 * a2 * b2 * b3 * c1 * c3 * d0 + 32 * a2 * b3 * b4 * c0 * c1 * d1 +
          16 * a3 * b0 * b1 * c1 * c2 * d4 + 16 * a3 * b0 * b1 * c1 * c4 * d2 -
          16 * a3 * b0 * b1 * c1 * c5 * d1 - 16 * a3 * b0 * b1 * c2 * c3 * d2 +
          16 * a3 * b0 * b1 * c2 * c4 * d1 + 16 * a3 * b0 * b3 * c1 * c2 * d2 -
          32 * a3 * b0 * b4 * c1 * c2 * d1 - 16 * a3 * b1 * b2 * c0 * c1 * d4 -
          16 * a3 * b1 * b2 * c0 * c4 * d1 - 16 * a3 * b1 * b2 * c1 * c4 * d0 +
          16 * a3 * b1 * b3 * c0 * c2 * d2 - 16 * a3 * b1 * b5 * c0 * c1 * d1 -
          16 * a3 * b2 * b3 * c0 * c1 * d2 - 16 * a3 * b2 * b3 * c0 * c2 * d1 -
          16 * a3 * b2 * b3 * c1 * c2 * d0 + 32 * a3 * b2 * b4 * c0 * c1 * d1 +
          16 * a4 * b0 * b1 * c1 * c2 * d3 + 16 * a4 * b0 * b1 * c1 * c3 * d2 +
          16 * a4 * b0 * b1 * c2 * c3 * d1 + 8 * a4 * b0 * b2 * c0 * c5 * d0 -
          32 * a4 * b0 * b3 * c1 * c2 * d1 + 8 * a4 * b0 * b5 * c0 * c2 * d0 -
          16 * a4 * b1 * b2 * c0 * c1 * d3 - 16 * a4 * b1 * b2 * c0 * c3 * d1 -
          16 * a4 * b1 * b2 * c1 * c3 * d0 + 32 * a4 * b2 * b3 * c0 * c1 * d1 -
          4 * a5 * b0 * b1 * c0 * c5 * d0 - 16 * a5 * b0 * b1 * c1 * c3 * d1 +
          8 * a5 * b0 * b2 * c0 * c4 * d0 + 8 * a5 * b0 * b4 * c0 * c2 * d0 -
          4 * a5 * b0 * b5 * c0 * c1 * d0 - 16 * a5 * b1 * b3 * c0 * c1 * d1 +
          16 * a0 * b0 * b1 * c3 * c4 * d4 - 4 * a0 * b0 * b1 * c3 * c5 * d3 -
          8 * a0 * b0 * b2 * c3 * c4 * d3 - 4 * a0 * b0 * b3 * c1 * c3 * d5 -
          16 * a0 * b0 * b3 * c1 * c4 * d4 - 4 * a0 * b0 * b3 * c1 * c5 * d3 +
          8 * a0 * b0 * b3 * c2 * c3 * d4 + 8 * a0 * b0 * b3 * c2 * c4 * d3 +
          8 * a0 * b0 * b3 * c3 * c4 * d2 - 4 * a0 * b0 * b3 * c3 * c5 * d1 -
          8 * a0 * b0 * b4 * c2 * c3 * d3 + 12 * a0 * b0 * b5 * c1 * c3 * d3 +
          4 * a0 * b1 * b3 * c0 * c3 * d5 + 4 * a0 * b1 * b3 * c0 * c5 * d3 +
          4 * a0 * b1 * b3 * c3 * c5 * d0 - 8 * a0 * b1 * b4 * c0 * c3 * d4 -
          8 * a0 * b1 * b4 * c0 * c4 * d3 - 8 * a0 * b1 * b4 * c3 * c4 * d0 -
          4 * a0 * b1 * b5 * c0 * c3 * d3 + 8 * a0 * b2 * b4 * c0 * c3 * d3 +
          8 * a0 * b3 * b4 * c0 * c1 * d4 + 8 * a0 * b3 * b4 * c0 * c4 * d1 +
          8 * a0 * b3 * b4 * c1 * c4 * d0 - 4 * a0 * b3 * b5 * c0 * c1 * d3 -
          4 * a0 * b3 * b5 * c0 * c3 * d1 - 4 * a0 * b3 * b5 * c1 * c3 * d0 +
          4 * a1 * b0 * b3 * c0 * c3 * d5 + 4 * a1 * b0 * b3 * c0 * c5 * d3 +
          4 * a1 * b0 * b3 * c3 * c5 * d0 - 8 * a1 * b0 * b4 * c0 * c3 * d4 -
          8 * a1 * b0 * b4 * c0 * c4 * d3 - 8 * a1 * b0 * b4 * c3 * c4 * d0 -
          4 * a1 * b0 * b5 * c0 * c3 * d3 + 64 * a1 * b1 * b2 * c1 * c2 * d5 +
          64 * a1 * b1 * b2 * c1 * c5 * d2 - 128 * a1 * b1 * b2 * c2 * c4 * d2 +
          64 * a1 * b1 * b2 * c2 * c5 * d1 + 64 * a1 * b1 * b5 * c1 * c2 * d2 -
          128 * a1 * b2 * b4 * c1 * c2 * d2 - 64 * a1 * b2 * b5 * c1 * c2 * d1 +
          4 * a1 * b3 * b5 * c0 * c3 * d0 + 8 * a2 * b0 * b4 * c0 * c3 * d3 +
          128 * a2 * b1 * b2 * c1 * c2 * d4 +
          128 * a2 * b1 * b2 * c1 * c4 * d2 - 64 * a2 * b1 * b2 * c1 * c5 * d1 +
          128 * a2 * b1 * b2 * c2 * c4 * d1 -
          128 * a2 * b1 * b4 * c1 * c2 * d2 - 64 * a2 * b1 * b5 * c1 * c2 * d1 +
          128 * a2 * b2 * b4 * c1 * c2 * d1 - 8 * a2 * b3 * b4 * c0 * c3 * d0 +
          4 * a3 * b0 * b1 * c0 * c3 * d5 + 4 * a3 * b0 * b1 * c0 * c5 * d3 +
          4 * a3 * b0 * b1 * c3 * c5 * d0 + 4 * a3 * b0 * b3 * c0 * c1 * d5 -
          8 * a3 * b0 * b3 * c0 * c2 * d4 - 8 * a3 * b0 * b3 * c0 * c4 * d2 +
          4 * a3 * b0 * b3 * c0 * c5 * d1 + 4 * a3 * b0 * b3 * c1 * c5 * d0 -
          8 * a3 * b0 * b3 * c2 * c4 * d0 + 8 * a3 * b0 * b4 * c0 * c1 * d4 +
          8 * a3 * b0 * b4 * c0 * c4 * d1 + 8 * a3 * b0 * b4 * c1 * c4 * d0 -
          4 * a3 * b0 * b5 * c0 * c1 * d3 - 4 * a3 * b0 * b5 * c0 * c3 * d1 -
          4 * a3 * b0 * b5 * c1 * c3 * d0 - 12 * a3 * b1 * b3 * c0 * c5 * d0 +
          4 * a3 * b1 * b5 * c0 * c3 * d0 + 8 * a3 * b2 * b3 * c0 * c4 * d0 -
          8 * a3 * b2 * b4 * c0 * c3 * d0 + 8 * a3 * b3 * b4 * c0 * c2 * d0 +
          4 * a3 * b3 * b5 * c0 * c1 * d0 - 8 * a4 * b0 * b1 * c0 * c3 * d4 -
          8 * a4 * b0 * b1 * c0 * c4 * d3 - 8 * a4 * b0 * b1 * c3 * c4 * d0 +
          8 * a4 * b0 * b2 * c0 * c3 * d3 + 8 * a4 * b0 * b3 * c0 * c1 * d4 +
          8 * a4 * b0 * b3 * c0 * c4 * d1 + 8 * a4 * b0 * b3 * c1 * c4 * d0 -
          128 * a4 * b1 * b2 * c1 * c2 * d2 + 16 * a4 * b1 * b4 * c0 * c3 * d0 -
          8 * a4 * b2 * b3 * c0 * c3 * d0 - 16 * a4 * b3 * b4 * c0 * c1 * d0 -
          4 * a5 * b0 * b1 * c0 * c3 * d3 - 4 * a5 * b0 * b3 * c0 * c1 * d3 -
          4 * a5 * b0 * b3 * c0 * c3 * d1 - 4 * a5 * b0 * b3 * c1 * c3 * d0 -
          64 * a5 * b1 * b2 * c1 * c2 * d1 + 4 * a5 * b1 * b3 * c0 * c3 * d0 -
          16 * a0 * b1 * b2 * c1 * c4 * d5 - 16 * a0 * b1 * b2 * c1 * c5 * d4 -
          8 * a0 * b1 * b2 * c2 * c3 * d5 - 8 * a0 * b1 * b2 * c2 * c5 * d3 -
          8 * a0 * b1 * b2 * c3 * c5 * d2 - 16 * a0 * b1 * b2 * c4 * c5 * d1 +
          16 * a0 * b1 * b3 * c2 * c5 * d2 - 16 * a0 * b1 * b5 * c1 * c2 * d4 -
          16 * a0 * b1 * b5 * c1 * c4 * d2 + 32 * a0 * b1 * b5 * c1 * c5 * d1 -
          16 * a0 * b1 * b5 * c2 * c4 * d1 - 8 * a0 * b2 * b3 * c1 * c2 * d5 -
          8 * a0 * b2 * b3 * c1 * c5 * d2 + 32 * a0 * b2 * b3 * c2 * c4 * d2 -
          8 * a0 * b2 * b3 * c2 * c5 * d1 - 32 * a0 * b2 * b4 * c1 * c2 * d4 -
          32 * a0 * b2 * b4 * c1 * c4 * d2 + 32 * a0 * b2 * b4 * c2 * c3 * d2 -
          32 * a0 * b2 * b4 * c2 * c4 * d1 + 32 * a0 * b2 * b5 * c1 * c4 * d1 -
          32 * a1 * b0 * b1 * c1 * c5 * d5 + 32 * a1 * b0 * b1 * c2 * c4 * d5 +
          32 * a1 * b0 * b1 * c2 * c5 * d4 + 32 * a1 * b0 * b1 * c4 * c5 * d2 -
          16 * a1 * b0 * b2 * c1 * c4 * d5 - 16 * a1 * b0 * b2 * c1 * c5 * d4 -
          8 * a1 * b0 * b2 * c2 * c3 * d5 - 8 * a1 * b0 * b2 * c2 * c5 * d3 -
          8 * a1 * b0 * b2 * c3 * c5 * d2 - 16 * a1 * b0 * b2 * c4 * c5 * d1 +
          16 * a1 * b0 * b3 * c2 * c5 * d2 - 16 * a1 * b0 * b5 * c1 * c2 * d4 -
          16 * a1 * b0 * b5 * c1 * c4 * d2 + 32 * a1 * b0 * b5 * c1 * c5 * d1 -
          16 * a1 * b0 * b5 * c2 * c4 * d1 - 32 * a1 * b1 * b4 * c0 * c2 * d5 -
          32 * a1 * b1 * b4 * c0 * c5 * d2 - 32 * a1 * b1 * b4 * c2 * c5 * d0 -
          32 * a1 * b1 * b5 * c0 * c1 * d5 - 32 * a1 * b1 * b5 * c0 * c5 * d1 -
          32 * a1 * b1 * b5 * c1 * c5 * d0 + 16 * a1 * b2 * b4 * c0 * c1 * d5 +
          32 * a1 * b2 * b4 * c0 * c2 * d4 + 32 * a1 * b2 * b4 * c0 * c4 * d2 +
          16 * a1 * b2 * b4 * c0 * c5 * d1 + 16 * a1 * b2 * b4 * c1 * c5 * d0 +
          32 * a1 * b2 * b4 * c2 * c4 * d0 + 8 * a1 * b2 * b5 * c0 * c2 * d3 +
          8 * a1 * b2 * b5 * c0 * c3 * d2 + 8 * a1 * b2 * b5 * c2 * c3 * d0 -
          16 * a1 * b3 * b5 * c0 * c2 * d2 + 16 * a1 * b4 * b5 * c0 * c1 * d2 +
          16 * a1 * b4 * b5 * c0 * c2 * d1 + 16 * a1 * b4 * b5 * c1 * c2 * d0 -
          16 * a2 * b0 * b1 * c1 * c4 * d5 - 16 * a2 * b0 * b1 * c1 * c5 * d4 -
          8 * a2 * b0 * b1 * c2 * c3 * d5 - 8 * a2 * b0 * b1 * c2 * c5 * d3 -
          8 * a2 * b0 * b1 * c3 * c5 * d2 - 16 * a2 * b0 * b1 * c4 * c5 * d1 +
          16 * a2 * b0 * b2 * c1 * c3 * d5 + 64 * a2 * b0 * b2 * c1 * c4 * d4 +
          16 * a2 * b0 * b2 * c1 * c5 * d3 - 32 * a2 * b0 * b2 * c2 * c3 * d4 -
          32 * a2 * b0 * b2 * c2 * c4 * d3 - 32 * a2 * b0 * b2 * c3 * c4 * d2 +
          16 * a2 * b0 * b2 * c3 * c5 * d1 - 8 * a2 * b0 * b3 * c1 * c2 * d5 -
          8 * a2 * b0 * b3 * c1 * c5 * d2 + 32 * a2 * b0 * b3 * c2 * c4 * d2 -
          8 * a2 * b0 * b3 * c2 * c5 * d1 - 32 * a2 * b0 * b4 * c1 * c2 * d4 -
          32 * a2 * b0 * b4 * c1 * c4 * d2 + 32 * a2 * b0 * b4 * c2 * c3 * d2 -
          32 * a2 * b0 * b4 * c2 * c4 * d1 + 32 * a2 * b0 * b5 * c1 * c4 * d1 -
          64 * a2 * b1 * b2 * c0 * c4 * d4 + 16 * a2 * b1 * b4 * c0 * c1 * d5 +
          32 * a2 * b1 * b4 * c0 * c2 * d4 + 32 * a2 * b1 * b4 * c0 * c4 * d2 +
          16 * a2 * b1 * b4 * c0 * c5 * d1 + 16 * a2 * b1 * b4 * c1 * c5 * d0 +
          32 * a2 * b1 * b4 * c2 * c4 * d0 + 8 * a2 * b1 * b5 * c0 * c2 * d3 +
          8 * a2 * b1 * b5 * c0 * c3 * d2 + 8 * a2 * b1 * b5 * c2 * c3 * d0 -
          32 * a2 * b2 * b3 * c0 * c2 * d4 - 32 * a2 * b2 * b3 * c0 * c4 * d2 -
          32 * a2 * b2 * b3 * c2 * c4 * d0 - 32 * a2 * b2 * b4 * c0 * c2 * d3 -
          32 * a2 * b2 * b4 * c0 * c3 * d2 - 32 * a2 * b2 * b4 * c2 * c3 * d0 -
          16 * a2 * b2 * b5 * c0 * c1 * d3 - 16 * a2 * b2 * b5 * c0 * c3 * d1 -
          16 * a2 * b2 * b5 * c1 * c3 * d0 + 32 * a2 * b3 * b4 * c0 * c2 * d2 +
          8 * a2 * b3 * b5 * c0 * c1 * d2 + 8 * a2 * b3 * b5 * c0 * c2 * d1 +
          8 * a2 * b3 * b5 * c1 * c2 * d0 - 32 * a2 * b4 * b5 * c0 * c1 * d1 +
          16 * a3 * b0 * b1 * c2 * c5 * d2 - 8 * a3 * b0 * b2 * c1 * c2 * d5 -
          8 * a3 * b0 * b2 * c1 * c5 * d2 + 32 * a3 * b0 * b2 * c2 * c4 * d2 -
          8 * a3 * b0 * b2 * c2 * c5 * d1 - 16 * a3 * b1 * b5 * c0 * c2 * d2 +
          32 * a3 * b2 * b4 * c0 * c2 * d2 + 8 * a3 * b2 * b5 * c0 * c1 * d2 +
          8 * a3 * b2 * b5 * c0 * c2 * d1 + 8 * a3 * b2 * b5 * c1 * c2 * d0 -
          32 * a4 * b0 * b2 * c1 * c2 * d4 - 32 * a4 * b0 * b2 * c1 * c4 * d2 +
          32 * a4 * b0 * b2 * c2 * c3 * d2 - 32 * a4 * b0 * b2 * c2 * c4 * d1 +
          64 * a4 * b0 * b4 * c1 * c2 * d2 + 16 * a4 * b1 * b2 * c0 * c1 * d5 +
          32 * a4 * b1 * b2 * c0 * c2 * d4 + 32 * a4 * b1 * b2 * c0 * c4 * d2 +
          16 * a4 * b1 * b2 * c0 * c5 * d1 + 16 * a4 * b1 * b2 * c1 * c5 * d0 +
          32 * a4 * b1 * b2 * c2 * c4 * d0 - 64 * a4 * b1 * b4 * c0 * c2 * d2 +
          16 * a4 * b1 * b5 * c0 * c1 * d2 + 16 * a4 * b1 * b5 * c0 * c2 * d1 +
          16 * a4 * b1 * b5 * c1 * c2 * d0 + 32 * a4 * b2 * b3 * c0 * c2 * d2 -
          32 * a4 * b2 * b5 * c0 * c1 * d1 - 16 * a5 * b0 * b1 * c1 * c2 * d4 -
          16 * a5 * b0 * b1 * c1 * c4 * d2 + 32 * a5 * b0 * b1 * c1 * c5 * d1 -
          16 * a5 * b0 * b1 * c2 * c4 * d1 + 32 * a5 * b0 * b2 * c1 * c4 * d1 +
          8 * a5 * b1 * b2 * c0 * c2 * d3 + 8 * a5 * b1 * b2 * c0 * c3 * d2 +
          8 * a5 * b1 * b2 * c2 * c3 * d0 - 16 * a5 * b1 * b3 * c0 * c2 * d2 +
          16 * a5 * b1 * b4 * c0 * c1 * d2 + 16 * a5 * b1 * b4 * c0 * c2 * d1 +
          16 * a5 * b1 * b4 * c1 * c2 * d0 + 32 * a5 * b1 * b5 * c0 * c1 * d1 +
          8 * a5 * b2 * b3 * c0 * c1 * d2 + 8 * a5 * b2 * b3 * c0 * c2 * d1 +
          8 * a5 * b2 * b3 * c1 * c2 * d0 - 32 * a5 * b2 * b4 * c0 * c1 * d1 +
          8 * a0 * b0 * b1 * c3 * c5 * d5 + 8 * a0 * b0 * b3 * c1 * c5 * d5 -
          16 * a0 * b0 * b3 * c2 * c4 * d5 - 16 * a0 * b0 * b3 * c2 * c5 * d4 -
          16 * a0 * b0 * b3 * c4 * c5 * d2 + 16 * a0 * b0 * b4 * c1 * c4 * d5 +
          16 * a0 * b0 * b4 * c1 * c5 * d4 + 16 * a0 * b0 * b4 * c4 * c5 * d1 -
          8 * a0 * b0 * b5 * c1 * c3 * d5 - 32 * a0 * b0 * b5 * c1 * c4 * d4 -
          8 * a0 * b0 * b5 * c1 * c5 * d3 + 16 * a0 * b0 * b5 * c2 * c3 * d4 +
          16 * a0 * b0 * b5 * c2 * c4 * d3 + 16 * a0 * b0 * b5 * c3 * c4 * d2 -
          8 * a0 * b0 * b5 * c3 * c5 * d1 - 8 * a0 * b1 * b3 * c0 * c5 * d5 -
          8 * a0 * b1 * b4 * c0 * c4 * d5 - 8 * a0 * b1 * b4 * c0 * c5 * d4 -
          8 * a0 * b1 * b4 * c4 * c5 * d0 + 16 * a0 * b1 * b5 * c0 * c4 * d4 +
          8 * a0 * b2 * b3 * c0 * c4 * d5 + 8 * a0 * b2 * b3 * c0 * c5 * d4 +
          8 * a0 * b2 * b3 * c4 * c5 * d0 - 8 * a0 * b2 * b5 * c0 * c3 * d4 -
          8 * a0 * b2 * b5 * c0 * c4 * d3 - 8 * a0 * b2 * b5 * c3 * c4 * d0 +
          8 * a0 * b3 * b4 * c0 * c2 * d5 + 8 * a0 * b3 * b4 * c0 * c5 * d2 +
          8 * a0 * b3 * b4 * c2 * c5 * d0 + 8 * a0 * b4 * b5 * c0 * c1 * d4 -
          8 * a0 * b4 * b5 * c0 * c2 * d3 - 8 * a0 * b4 * b5 * c0 * c3 * d2 +
          8 * a0 * b4 * b5 * c0 * c4 * d1 + 8 * a0 * b4 * b5 * c1 * c4 * d0 -
          8 * a0 * b4 * b5 * c2 * c3 * d0 - 8 * a1 * b0 * b3 * c0 * c5 * d5 -
          8 * a1 * b0 * b4 * c0 * c4 * d5 - 8 * a1 * b0 * b4 * c0 * c5 * d4 -
          8 * a1 * b0 * b4 * c4 * c5 * d0 + 16 * a1 * b0 * b5 * c0 * c4 * d4 +
          8 * a1 * b3 * b5 * c0 * c5 * d0 - 16 * a1 * b4 * b5 * c0 * c4 * d0 +
          8 * a2 * b0 * b3 * c0 * c4 * d5 + 8 * a2 * b0 * b3 * c0 * c5 * d4 +
          8 * a2 * b0 * b3 * c4 * c5 * d0 - 8 * a2 * b0 * b5 * c0 * c3 * d4 -
          8 * a2 * b0 * b5 * c0 * c4 * d3 - 8 * a2 * b0 * b5 * c3 * c4 * d0 -
          16 * a2 * b3 * b4 * c0 * c5 * d0 + 16 * a2 * b4 * b5 * c0 * c3 * d0 -
          8 * a3 * b0 * b1 * c0 * c5 * d5 + 8 * a3 * b0 * b2 * c0 * c4 * d5 +
          8 * a3 * b0 * b2 * c0 * c5 * d4 + 8 * a3 * b0 * b2 * c4 * c5 * d0 +
          8 * a3 * b0 * b4 * c0 * c2 * d5 + 8 * a3 * b0 * b4 * c0 * c5 * d2 +
          8 * a3 * b0 * b4 * c2 * c5 * d0 + 8 * a3 * b1 * b5 * c0 * c5 * d0 -
          16 * a3 * b2 * b4 * c0 * c5 * d0 - 8 * a4 * b0 * b1 * c0 * c4 * d5 -
          8 * a4 * b0 * b1 * c0 * c5 * d4 - 8 * a4 * b0 * b1 * c4 * c5 * d0 +
          8 * a4 * b0 * b3 * c0 * c2 * d5 + 8 * a4 * b0 * b3 * c0 * c5 * d2 +
          8 * a4 * b0 * b3 * c2 * c5 * d0 - 16 * a4 * b0 * b4 * c0 * c1 * d5 -
          16 * a4 * b0 * b4 * c0 * c5 * d1 - 16 * a4 * b0 * b4 * c1 * c5 * d0 +
          8 * a4 * b0 * b5 * c0 * c1 * d4 - 8 * a4 * b0 * b5 * c0 * c2 * d3 -
          8 * a4 * b0 * b5 * c0 * c3 * d2 + 8 * a4 * b0 * b5 * c0 * c4 * d1 +
          8 * a4 * b0 * b5 * c1 * c4 * d0 - 8 * a4 * b0 * b5 * c2 * c3 * d0 +
          32 * a4 * b1 * b4 * c0 * c5 * d0 - 16 * a4 * b1 * b5 * c0 * c4 * d0 -
          16 * a4 * b2 * b3 * c0 * c5 * d0 + 16 * a4 * b2 * b5 * c0 * c3 * d0 +
          16 * a5 * b0 * b1 * c0 * c4 * d4 - 8 * a5 * b0 * b2 * c0 * c3 * d4 -
          8 * a5 * b0 * b2 * c0 * c4 * d3 - 8 * a5 * b0 * b2 * c3 * c4 * d0 +
          8 * a5 * b0 * b4 * c0 * c1 * d4 - 8 * a5 * b0 * b4 * c0 * c2 * d3 -
          8 * a5 * b0 * b4 * c0 * c3 * d2 + 8 * a5 * b0 * b4 * c0 * c4 * d1 +
          8 * a5 * b0 * b4 * c1 * c4 * d0 - 8 * a5 * b0 * b4 * c2 * c3 * d0 +
          8 * a5 * b0 * b5 * c0 * c1 * d3 + 8 * a5 * b0 * b5 * c0 * c3 * d1 +
          8 * a5 * b0 * b5 * c1 * c3 * d0 + 8 * a5 * b1 * b3 * c0 * c5 * d0 -
          16 * a5 * b1 * b4 * c0 * c4 * d0 - 8 * a5 * b1 * b5 * c0 * c3 * d0 +
          16 * a5 * b2 * b4 * c0 * c3 * d0 - 8 * a5 * b3 * b5 * c0 * c1 * d0 +
          8 * a0 * b2 * b5 * c1 * c2 * d5 + 8 * a0 * b2 * b5 * c1 * c5 * d2 +
          8 * a0 * b2 * b5 * c2 * c5 * d1 - 8 * a1 * b2 * b5 * c0 * c2 * d5 -
          8 * a1 * b2 * b5 * c0 * c5 * d2 - 8 * a1 * b2 * b5 * c2 * c5 * d0 -
          16 * a2 * b0 * b2 * c1 * c5 * d5 + 8 * a2 * b0 * b5 * c1 * c2 * d5 +
          8 * a2 * b0 * b5 * c1 * c5 * d2 + 8 * a2 * b0 * b5 * c2 * c5 * d1 +
          16 * a2 * b1 * b2 * c0 * c5 * d5 - 8 * a2 * b1 * b5 * c0 * c2 * d5 -
          8 * a2 * b1 * b5 * c0 * c5 * d2 - 8 * a2 * b1 * b5 * c2 * c5 * d0 +
          8 * a5 * b0 * b2 * c1 * c2 * d5 + 8 * a5 * b0 * b2 * c1 * c5 * d2 +
          8 * a5 * b0 * b2 * c2 * c5 * d1 - 16 * a5 * b0 * b5 * c1 * c2 * d2 -
          8 * a5 * b1 * b2 * c0 * c2 * d5 - 8 * a5 * b1 * b2 * c0 * c5 * d2 -
          8 * a5 * b1 * b2 * c2 * c5 * d0 + 16 * a5 * b1 * b5 * c0 * c2 * d2 +
          8 * a0 * b0 * b2 * c4 * c5 * d5 + 8 * a0 * b0 * b4 * c2 * c5 * d5 +
          4 * a0 * b0 * b5 * c1 * c5 * d5 - 8 * a0 * b0 * b5 * c2 * c4 * d5 -
          8 * a0 * b0 * b5 * c2 * c5 * d4 - 8 * a0 * b0 * b5 * c4 * c5 * d2 +
          4 * a0 * b1 * b5 * c0 * c5 * d5 - 8 * a0 * b2 * b4 * c0 * c5 * d5 +
          4 * a1 * b0 * b5 * c0 * c5 * d5 - 32 * a1 * b1 * b2 * c3 * c4 * d5 -
          32 * a1 * b1 * b2 * c3 * c5 * d4 - 32 * a1 * b1 * b2 * c4 * c5 * d3 -
          16 * a1 * b1 * b3 * c1 * c5 * d5 - 16 * a1 * b1 * b5 * c1 * c3 * d5 -
          16 * a1 * b1 * b5 * c1 * c5 * d3 + 32 * a1 * b1 * b5 * c2 * c3 * d4 +
          32 * a1 * b1 * b5 * c2 * c4 * d3 + 32 * a1 * b1 * b5 * c3 * c4 * d2 -
          16 * a1 * b1 * b5 * c3 * c5 * d1 + 16 * a1 * b2 * b3 * c1 * c4 * d5 +
          16 * a1 * b2 * b3 * c1 * c5 * d4 + 16 * a1 * b2 * b3 * c4 * c5 * d1 +
          16 * a1 * b2 * b4 * c1 * c3 * d5 + 16 * a1 * b2 * b4 * c1 * c5 * d3 -
          32 * a1 * b2 * b4 * c2 * c3 * d4 - 32 * a1 * b2 * b4 * c2 * c4 * d3 -
          32 * a1 * b2 * b4 * c3 * c4 * d2 + 16 * a1 * b2 * b4 * c3 * c5 * d1 -
          16 * a1 * b2 * b5 * c2 * c3 * d3 - 16 * a1 * b3 * b5 * c1 * c2 * d4 -
          16 * a1 * b3 * b5 * c1 * c4 * d2 + 16 * a1 * b3 * b5 * c1 * c5 * d1 +
          16 * a1 * b3 * b5 * c2 * c3 * d2 - 16 * a1 * b3 * b5 * c2 * c4 * d1 -
          16 * a1 * b4 * b5 * c1 * c2 * d3 - 16 * a1 * b4 * b5 * c1 * c3 * d2 -
          16 * a1 * b4 * b5 * c2 * c3 * d1 - 8 * a2 * b0 * b4 * c0 * c5 * d5 +
          64 * a2 * b1 * b2 * c3 * c4 * d4 + 16 * a2 * b1 * b2 * c3 * c5 * d3 +
          16 * a2 * b1 * b3 * c1 * c4 * d5 + 16 * a2 * b1 * b3 * c1 * c5 * d4 +
          16 * a2 * b1 * b3 * c4 * c5 * d1 + 16 * a2 * b1 * b4 * c1 * c3 * d5 +
          16 * a2 * b1 * b4 * c1 * c5 * d3 - 32 * a2 * b1 * b4 * c2 * c3 * d4 -
          32 * a2 * b1 * b4 * c2 * c4 * d3 - 32 * a2 * b1 * b4 * c3 * c4 * d2 +
          16 * a2 * b1 * b4 * c3 * c5 * d1 - 16 * a2 * b1 * b5 * c2 * c3 * d3 -
          16 * a2 * b2 * b3 * c1 * c3 * d5 - 64 * a2 * b2 * b3 * c1 * c4 * d4 -
          16 * a2 * b2 * b3 * c1 * c5 * d3 + 32 * a2 * b2 * b3 * c2 * c3 * d4 +
          32 * a2 * b2 * b3 * c2 * c4 * d3 + 32 * a2 * b2 * b3 * c3 * c4 * d2 -
          16 * a2 * b2 * b3 * c3 * c5 * d1 + 32 * a2 * b2 * b4 * c2 * c3 * d3 +
          16 * a2 * b2 * b5 * c1 * c3 * d3 + 32 * a2 * b3 * b4 * c1 * c2 * d4 +
          32 * a2 * b3 * b4 * c1 * c4 * d2 - 32 * a2 * b3 * b4 * c1 * c5 * d1 -
          32 * a2 * b3 * b4 * c2 * c3 * d2 + 32 * a2 * b3 * b4 * c2 * c4 * d1 +
          8 * a2 * b4 * b5 * c0 * c5 * d0 + 16 * a3 * b1 * b2 * c1 * c4 * d5 +
          16 * a3 * b1 * b2 * c1 * c5 * d4 + 16 * a3 * b1 * b2 * c4 * c5 * d1 -
          16 * a3 * b1 * b3 * c2 * c5 * d2 - 16 * a3 * b1 * b5 * c1 * c2 * d4 -
          16 * a3 * b1 * b5 * c1 * c4 * d2 + 16 * a3 * b1 * b5 * c1 * c5 * d1 +
          16 * a3 * b1 * b5 * c2 * c3 * d2 - 16 * a3 * b1 * b5 * c2 * c4 * d1 +
          16 * a3 * b2 * b3 * c1 * c2 * d5 + 16 * a3 * b2 * b3 * c1 * c5 * d2 -
          32 * a3 * b2 * b3 * c2 * c4 * d2 + 16 * a3 * b2 * b3 * c2 * c5 * d1 +
          32 * a3 * b2 * b4 * c1 * c2 * d4 + 32 * a3 * b2 * b4 * c1 * c4 * d2 -
          32 * a3 * b2 * b4 * c1 * c5 * d1 - 32 * a3 * b2 * b4 * c2 * c3 * d2 +
          32 * a3 * b2 * b4 * c2 * c4 * d1 - 16 * a3 * b3 * b5 * c1 * c2 * d2 +
          32 * a3 * b4 * b5 * c1 * c2 * d1 - 8 * a4 * b0 * b2 * c0 * c5 * d5 +
          16 * a4 * b1 * b2 * c1 * c3 * d5 + 16 * a4 * b1 * b2 * c1 * c5 * d3 -
          32 * a4 * b1 * b2 * c2 * c3 * d4 - 32 * a4 * b1 * b2 * c2 * c4 * d3 -
          32 * a4 * b1 * b2 * c3 * c4 * d2 + 16 * a4 * b1 * b2 * c3 * c5 * d1 +
          64 * a4 * b1 * b4 * c2 * c3 * d2 - 16 * a4 * b1 * b5 * c1 * c2 * d3 -
          16 * a4 * b1 * b5 * c1 * c3 * d2 - 16 * a4 * b1 * b5 * c2 * c3 * d1 +
          32 * a4 * b2 * b3 * c1 * c2 * d4 + 32 * a4 * b2 * b3 * c1 * c4 * d2 -
          32 * a4 * b2 * b3 * c1 * c5 * d1 - 32 * a4 * b2 * b3 * c2 * c3 * d2 +
          32 * a4 * b2 * b3 * c2 * c4 * d1 + 8 * a4 * b2 * b5 * c0 * c5 * d0 -
          64 * a4 * b3 * b4 * c1 * c2 * d2 + 32 * a4 * b3 * b5 * c1 * c2 * d1 +
          4 * a5 * b0 * b1 * c0 * c5 * d5 - 4 * a5 * b0 * b5 * c0 * c1 * d5 +
          8 * a5 * b0 * b5 * c0 * c2 * d4 + 8 * a5 * b0 * b5 * c0 * c4 * d2 -
          4 * a5 * b0 * b5 * c0 * c5 * d1 - 4 * a5 * b0 * b5 * c1 * c5 * d0 +
          8 * a5 * b0 * b5 * c2 * c4 * d0 - 16 * a5 * b1 * b2 * c2 * c3 * d3 -
          16 * a5 * b1 * b3 * c1 * c2 * d4 - 16 * a5 * b1 * b3 * c1 * c4 * d2 +
          16 * a5 * b1 * b3 * c1 * c5 * d1 + 16 * a5 * b1 * b3 * c2 * c3 * d2 -
          16 * a5 * b1 * b3 * c2 * c4 * d1 - 16 * a5 * b1 * b4 * c1 * c2 * d3 -
          16 * a5 * b1 * b4 * c1 * c3 * d2 - 16 * a5 * b1 * b4 * c2 * c3 * d1 -
          4 * a5 * b1 * b5 * c0 * c5 * d0 + 16 * a5 * b1 * b5 * c1 * c3 * d1 +
          8 * a5 * b2 * b4 * c0 * c5 * d0 - 8 * a5 * b2 * b5 * c0 * c4 * d0 +
          32 * a5 * b3 * b4 * c1 * c2 * d1 - 8 * a5 * b4 * b5 * c0 * c2 * d0 -
          4 * a0 * b1 * b3 * c3 * c5 * d5 + 8 * a0 * b1 * b4 * c3 * c4 * d5 +
          8 * a0 * b1 * b4 * c3 * c5 * d4 + 8 * a0 * b1 * b4 * c4 * c5 * d3 -
          16 * a0 * b1 * b5 * c3 * c4 * d4 + 4 * a0 * b1 * b5 * c3 * c5 * d3 -
          8 * a0 * b2 * b4 * c3 * c5 * d3 + 8 * a0 * b2 * b5 * c3 * c4 * d3 -
          8 * a0 * b3 * b4 * c1 * c4 * d5 - 8 * a0 * b3 * b4 * c1 * c5 * d4 -
          8 * a0 * b3 * b4 * c4 * c5 * d1 + 4 * a0 * b3 * b5 * c1 * c3 * d5 +
          16 * a0 * b3 * b5 * c1 * c4 * d4 + 4 * a0 * b3 * b5 * c1 * c5 * d3 -
          8 * a0 * b3 * b5 * c2 * c3 * d4 - 8 * a0 * b3 * b5 * c2 * c4 * d3 -
          8 * a0 * b3 * b5 * c3 * c4 * d2 + 4 * a0 * b3 * b5 * c3 * c5 * d1 +
          8 * a0 * b4 * b5 * c2 * c3 * d3 - 4 * a1 * b0 * b3 * c3 * c5 * d5 +
          8 * a1 * b0 * b4 * c3 * c4 * d5 + 8 * a1 * b0 * b4 * c3 * c5 * d4 +
          8 * a1 * b0 * b4 * c4 * c5 * d3 - 16 * a1 * b0 * b5 * c3 * c4 * d4 +
          4 * a1 * b0 * b5 * c3 * c5 * d3 - 4 * a1 * b3 * b5 * c0 * c3 * d5 -
          4 * a1 * b3 * b5 * c0 * c5 * d3 - 4 * a1 * b3 * b5 * c3 * c5 * d0 +
          8 * a1 * b4 * b5 * c0 * c3 * d4 + 8 * a1 * b4 * b5 * c0 * c4 * d3 +
          8 * a1 * b4 * b5 * c3 * c4 * d0 - 8 * a2 * b0 * b4 * c3 * c5 * d3 +
          8 * a2 * b0 * b5 * c3 * c4 * d3 + 8 * a2 * b3 * b4 * c0 * c3 * d5 +
          8 * a2 * b3 * b4 * c0 * c5 * d3 + 8 * a2 * b3 * b4 * c3 * c5 * d0 -
          8 * a2 * b4 * b5 * c0 * c3 * d3 - 4 * a3 * b0 * b1 * c3 * c5 * d5 -
          4 * a3 * b0 * b3 * c1 * c5 * d5 + 8 * a3 * b0 * b3 * c2 * c4 * d5 +
          8 * a3 * b0 * b3 * c2 * c5 * d4 + 8 * a3 * b0 * b3 * c4 * c5 * d2 -
          8 * a3 * b0 * b4 * c1 * c4 * d5 - 8 * a3 * b0 * b4 * c1 * c5 * d4 -
          8 * a3 * b0 * b4 * c4 * c5 * d1 + 4 * a3 * b0 * b5 * c1 * c3 * d5 +
          16 * a3 * b0 * b5 * c1 * c4 * d4 + 4 * a3 * b0 * b5 * c1 * c5 * d3 -
          8 * a3 * b0 * b5 * c2 * c3 * d4 - 8 * a3 * b0 * b5 * c2 * c4 * d3 -
          8 * a3 * b0 * b5 * c3 * c4 * d2 + 4 * a3 * b0 * b5 * c3 * c5 * d1 +
          12 * a3 * b1 * b3 * c0 * c5 * d5 - 4 * a3 * b1 * b5 * c0 * c3 * d5 -
          4 * a3 * b1 * b5 * c0 * c5 * d3 - 4 * a3 * b1 * b5 * c3 * c5 * d0 -
          8 * a3 * b2 * b3 * c0 * c4 * d5 - 8 * a3 * b2 * b3 * c0 * c5 * d4 -
          8 * a3 * b2 * b3 * c4 * c5 * d0 + 8 * a3 * b2 * b4 * c0 * c3 * d5 +
          8 * a3 * b2 * b4 * c0 * c5 * d3 + 8 * a3 * b2 * b4 * c3 * c5 * d0 -
          8 * a3 * b3 * b4 * c0 * c2 * d5 - 8 * a3 * b3 * b4 * c0 * c5 * d2 -
          8 * a3 * b3 * b4 * c2 * c5 * d0 - 4 * a3 * b3 * b5 * c0 * c1 * d5 +
          8 * a3 * b3 * b5 * c0 * c2 * d4 + 8 * a3 * b3 * b5 * c0 * c4 * d2 -
          4 * a3 * b3 * b5 * c0 * c5 * d1 - 4 * a3 * b3 * b5 * c1 * c5 * d0 +
          8 * a3 * b3 * b5 * c2 * c4 * d0 - 8 * a3 * b4 * b5 * c0 * c1 * d4 -
          8 * a3 * b4 * b5 * c0 * c4 * d1 - 8 * a3 * b4 * b5 * c1 * c4 * d0 +
          8 * a4 * b0 * b1 * c3 * c4 * d5 + 8 * a4 * b0 * b1 * c3 * c5 * d4 +
          8 * a4 * b0 * b1 * c4 * c5 * d3 - 8 * a4 * b0 * b2 * c3 * c5 * d3 -
          8 * a4 * b0 * b3 * c1 * c4 * d5 - 8 * a4 * b0 * b3 * c1 * c5 * d4 -
          8 * a4 * b0 * b3 * c4 * c5 * d1 + 8 * a4 * b0 * b5 * c2 * c3 * d3 -
          16 * a4 * b1 * b4 * c0 * c3 * d5 - 16 * a4 * b1 * b4 * c0 * c5 * d3 -
          16 * a4 * b1 * b4 * c3 * c5 * d0 + 8 * a4 * b1 * b5 * c0 * c3 * d4 +
          8 * a4 * b1 * b5 * c0 * c4 * d3 + 8 * a4 * b1 * b5 * c3 * c4 * d0 +
          8 * a4 * b2 * b3 * c0 * c3 * d5 + 8 * a4 * b2 * b3 * c0 * c5 * d3 +
          8 * a4 * b2 * b3 * c3 * c5 * d0 - 8 * a4 * b2 * b5 * c0 * c3 * d3 +
          16 * a4 * b3 * b4 * c0 * c1 * d5 + 16 * a4 * b3 * b4 * c0 * c5 * d1 +
          16 * a4 * b3 * b4 * c1 * c5 * d0 - 8 * a4 * b3 * b5 * c0 * c1 * d4 -
          8 * a4 * b3 * b5 * c0 * c4 * d1 - 8 * a4 * b3 * b5 * c1 * c4 * d0 -
          16 * a5 * b0 * b1 * c3 * c4 * d4 + 4 * a5 * b0 * b1 * c3 * c5 * d3 +
          8 * a5 * b0 * b2 * c3 * c4 * d3 + 4 * a5 * b0 * b3 * c1 * c3 * d5 +
          16 * a5 * b0 * b3 * c1 * c4 * d4 + 4 * a5 * b0 * b3 * c1 * c5 * d3 -
          8 * a5 * b0 * b3 * c2 * c3 * d4 - 8 * a5 * b0 * b3 * c2 * c4 * d3 -
          8 * a5 * b0 * b3 * c3 * c4 * d2 + 4 * a5 * b0 * b3 * c3 * c5 * d1 +
          8 * a5 * b0 * b4 * c2 * c3 * d3 - 12 * a5 * b0 * b5 * c1 * c3 * d3 -
          4 * a5 * b1 * b3 * c0 * c3 * d5 - 4 * a5 * b1 * b3 * c0 * c5 * d3 -
          4 * a5 * b1 * b3 * c3 * c5 * d0 + 8 * a5 * b1 * b4 * c0 * c3 * d4 +
          8 * a5 * b1 * b4 * c0 * c4 * d3 + 8 * a5 * b1 * b4 * c3 * c4 * d0 +
          4 * a5 * b1 * b5 * c0 * c3 * d3 - 8 * a5 * b2 * b4 * c0 * c3 * d3 -
          8 * a5 * b3 * b4 * c0 * c1 * d4 - 8 * a5 * b3 * b4 * c0 * c4 * d1 -
          8 * a5 * b3 * b4 * c1 * c4 * d0 + 4 * a5 * b3 * b5 * c0 * c1 * d3 +
          4 * a5 * b3 * b5 * c0 * c3 * d1 + 4 * a5 * b3 * b5 * c1 * c3 * d0 +
          32 * a1 * b1 * b2 * c4 * c5 * d5 + 32 * a1 * b1 * b4 * c2 * c5 * d5 +
          48 * a1 * b1 * b5 * c1 * c5 * d5 - 32 * a1 * b1 * b5 * c2 * c4 * d5 -
          32 * a1 * b1 * b5 * c2 * c5 * d4 - 32 * a1 * b1 * b5 * c4 * c5 * d2 -
          32 * a1 * b2 * b4 * c1 * c5 * d5 + 8 * a1 * b2 * b5 * c2 * c3 * d5 +
          8 * a1 * b2 * b5 * c2 * c5 * d3 + 8 * a1 * b2 * b5 * c3 * c5 * d2 -
          16 * a2 * b1 * b2 * c3 * c5 * d5 - 32 * a2 * b1 * b4 * c1 * c5 * d5 +
          8 * a2 * b1 * b5 * c2 * c3 * d5 + 8 * a2 * b1 * b5 * c2 * c5 * d3 +
          8 * a2 * b1 * b5 * c3 * c5 * d2 + 16 * a2 * b2 * b3 * c1 * c5 * d5 -
          8 * a2 * b3 * b5 * c1 * c2 * d5 - 8 * a2 * b3 * b5 * c1 * c5 * d2 -
          8 * a2 * b3 * b5 * c2 * c5 * d1 + 32 * a2 * b4 * b5 * c1 * c5 * d1 -
          8 * a3 * b2 * b5 * c1 * c2 * d5 - 8 * a3 * b2 * b5 * c1 * c5 * d2 -
          8 * a3 * b2 * b5 * c2 * c5 * d1 - 32 * a4 * b1 * b2 * c1 * c5 * d5 +
          32 * a4 * b2 * b5 * c1 * c5 * d1 + 8 * a5 * b1 * b2 * c2 * c3 * d5 +
          8 * a5 * b1 * b2 * c2 * c5 * d3 + 8 * a5 * b1 * b2 * c3 * c5 * d2 +
          32 * a5 * b1 * b5 * c1 * c2 * d4 + 32 * a5 * b1 * b5 * c1 * c4 * d2 -
          48 * a5 * b1 * b5 * c1 * c5 * d1 - 16 * a5 * b1 * b5 * c2 * c3 * d2 +
          32 * a5 * b1 * b5 * c2 * c4 * d1 - 8 * a5 * b2 * b3 * c1 * c2 * d5 -
          8 * a5 * b2 * b3 * c1 * c5 * d2 - 8 * a5 * b2 * b3 * c2 * c5 * d1 +
          32 * a5 * b2 * b4 * c1 * c5 * d1 - 32 * a5 * b2 * b5 * c1 * c4 * d1 +
          16 * a5 * b3 * b5 * c1 * c2 * d2 - 32 * a5 * b4 * b5 * c1 * c2 * d1 -
          4 * a0 * b1 * b5 * c3 * c5 * d5 - 8 * a0 * b2 * b3 * c4 * c5 * d5 +
          8 * a0 * b2 * b4 * c3 * c5 * d5 - 8 * a0 * b3 * b4 * c2 * c5 * d5 -
          4 * a0 * b3 * b5 * c1 * c5 * d5 + 8 * a0 * b3 * b5 * c2 * c4 * d5 +
          8 * a0 * b3 * b5 * c2 * c5 * d4 + 8 * a0 * b3 * b5 * c4 * c5 * d2 -
          8 * a0 * b4 * b5 * c1 * c4 * d5 - 8 * a0 * b4 * b5 * c1 * c5 * d4 -
          8 * a0 * b4 * b5 * c4 * c5 * d1 - 4 * a1 * b0 * b5 * c3 * c5 * d5 -
          4 * a1 * b3 * b5 * c0 * c5 * d5 + 8 * a1 * b4 * b5 * c0 * c4 * d5 +
          8 * a1 * b4 * b5 * c0 * c5 * d4 + 8 * a1 * b4 * b5 * c4 * c5 * d0 -
          8 * a2 * b0 * b3 * c4 * c5 * d5 + 8 * a2 * b0 * b4 * c3 * c5 * d5 +
          8 * a2 * b3 * b4 * c0 * c5 * d5 - 8 * a2 * b4 * b5 * c0 * c3 * d5 -
          8 * a2 * b4 * b5 * c0 * c5 * d3 - 8 * a2 * b4 * b5 * c3 * c5 * d0 -
          8 * a3 * b0 * b2 * c4 * c5 * d5 - 8 * a3 * b0 * b4 * c2 * c5 * d5 -
          4 * a3 * b0 * b5 * c1 * c5 * d5 + 8 * a3 * b0 * b5 * c2 * c4 * d5 +
          8 * a3 * b0 * b5 * c2 * c5 * d4 + 8 * a3 * b0 * b5 * c4 * c5 * d2 -
          4 * a3 * b1 * b5 * c0 * c5 * d5 + 8 * a3 * b2 * b4 * c0 * c5 * d5 +
          8 * a4 * b0 * b2 * c3 * c5 * d5 - 8 * a4 * b0 * b3 * c2 * c5 * d5 +
          16 * a4 * b0 * b4 * c1 * c5 * d5 - 8 * a4 * b0 * b5 * c1 * c4 * d5 -
          8 * a4 * b0 * b5 * c1 * c5 * d4 - 8 * a4 * b0 * b5 * c4 * c5 * d1 -
          16 * a4 * b1 * b4 * c0 * c5 * d5 + 8 * a4 * b1 * b5 * c0 * c4 * d5 +
          8 * a4 * b1 * b5 * c0 * c5 * d4 + 8 * a4 * b1 * b5 * c4 * c5 * d0 +
          8 * a4 * b2 * b3 * c0 * c5 * d5 - 8 * a4 * b2 * b5 * c0 * c3 * d5 -
          8 * a4 * b2 * b5 * c0 * c5 * d3 - 8 * a4 * b2 * b5 * c3 * c5 * d0 -
          4 * a5 * b0 * b1 * c3 * c5 * d5 - 4 * a5 * b0 * b3 * c1 * c5 * d5 +
          8 * a5 * b0 * b3 * c2 * c4 * d5 + 8 * a5 * b0 * b3 * c2 * c5 * d4 +
          8 * a5 * b0 * b3 * c4 * c5 * d2 - 8 * a5 * b0 * b4 * c1 * c4 * d5 -
          8 * a5 * b0 * b4 * c1 * c5 * d4 - 8 * a5 * b0 * b4 * c4 * c5 * d1 +
          4 * a5 * b0 * b5 * c1 * c3 * d5 + 16 * a5 * b0 * b5 * c1 * c4 * d4 +
          4 * a5 * b0 * b5 * c1 * c5 * d3 - 8 * a5 * b0 * b5 * c2 * c3 * d4 -
          8 * a5 * b0 * b5 * c2 * c4 * d3 - 8 * a5 * b0 * b5 * c3 * c4 * d2 +
          4 * a5 * b0 * b5 * c3 * c5 * d1 - 4 * a5 * b1 * b3 * c0 * c5 * d5 +
          8 * a5 * b1 * b4 * c0 * c4 * d5 + 8 * a5 * b1 * b4 * c0 * c5 * d4 +
          8 * a5 * b1 * b4 * c4 * c5 * d0 + 4 * a5 * b1 * b5 * c0 * c3 * d5 -
          16 * a5 * b1 * b5 * c0 * c4 * d4 + 4 * a5 * b1 * b5 * c0 * c5 * d3 +
          4 * a5 * b1 * b5 * c3 * c5 * d0 - 8 * a5 * b2 * b4 * c0 * c3 * d5 -
          8 * a5 * b2 * b4 * c0 * c5 * d3 - 8 * a5 * b2 * b4 * c3 * c5 * d0 +
          8 * a5 * b2 * b5 * c0 * c3 * d4 + 8 * a5 * b2 * b5 * c0 * c4 * d3 +
          8 * a5 * b2 * b5 * c3 * c4 * d0 + 4 * a5 * b3 * b5 * c0 * c1 * d5 -
          8 * a5 * b3 * b5 * c0 * c2 * d4 - 8 * a5 * b3 * b5 * c0 * c4 * d2 +
          4 * a5 * b3 * b5 * c0 * c5 * d1 + 4 * a5 * b3 * b5 * c1 * c5 * d0 -
          8 * a5 * b3 * b5 * c2 * c4 * d0 + 8 * a5 * b4 * b5 * c0 * c2 * d3 +
          8 * a5 * b4 * b5 * c0 * c3 * d2 + 8 * a5 * b4 * b5 * c2 * c3 * d0 +
          4 * a1 * b3 * b5 * c3 * c5 * d5 - 8 * a1 * b4 * b5 * c3 * c4 * d5 -
          8 * a1 * b4 * b5 * c3 * c5 * d4 - 8 * a1 * b4 * b5 * c4 * c5 * d3 -
          8 * a2 * b3 * b4 * c3 * c5 * d5 + 8 * a2 * b4 * b5 * c3 * c5 * d3 +
          4 * a3 * b1 * b5 * c3 * c5 * d5 + 8 * a3 * b2 * b3 * c4 * c5 * d5 -
          8 * a3 * b2 * b4 * c3 * c5 * d5 + 8 * a3 * b3 * b4 * c2 * c5 * d5 +
          4 * a3 * b3 * b5 * c1 * c5 * d5 - 8 * a3 * b3 * b5 * c2 * c4 * d5 -
          8 * a3 * b3 * b5 * c2 * c5 * d4 - 8 * a3 * b3 * b5 * c4 * c5 * d2 +
          8 * a3 * b4 * b5 * c1 * c4 * d5 + 8 * a3 * b4 * b5 * c1 * c5 * d4 +
          8 * a3 * b4 * b5 * c4 * c5 * d1 + 16 * a4 * b1 * b4 * c3 * c5 * d5 -
          8 * a4 * b1 * b5 * c3 * c4 * d5 - 8 * a4 * b1 * b5 * c3 * c5 * d4 -
          8 * a4 * b1 * b5 * c4 * c5 * d3 - 8 * a4 * b2 * b3 * c3 * c5 * d5 +
          8 * a4 * b2 * b5 * c3 * c5 * d3 - 16 * a4 * b3 * b4 * c1 * c5 * d5 +
          8 * a4 * b3 * b5 * c1 * c4 * d5 + 8 * a4 * b3 * b5 * c1 * c5 * d4 +
          8 * a4 * b3 * b5 * c4 * c5 * d1 + 4 * a5 * b1 * b3 * c3 * c5 * d5 -
          8 * a5 * b1 * b4 * c3 * c4 * d5 - 8 * a5 * b1 * b4 * c3 * c5 * d4 -
          8 * a5 * b1 * b4 * c4 * c5 * d3 + 16 * a5 * b1 * b5 * c3 * c4 * d4 -
          4 * a5 * b1 * b5 * c3 * c5 * d3 + 8 * a5 * b2 * b4 * c3 * c5 * d3 -
          8 * a5 * b2 * b5 * c3 * c4 * d3 + 8 * a5 * b3 * b4 * c1 * c4 * d5 +
          8 * a5 * b3 * b4 * c1 * c5 * d4 + 8 * a5 * b3 * b4 * c4 * c5 * d1 -
          4 * a5 * b3 * b5 * c1 * c3 * d5 - 16 * a5 * b3 * b5 * c1 * c4 * d4 -
          4 * a5 * b3 * b5 * c1 * c5 * d3 + 8 * a5 * b3 * b5 * c2 * c3 * d4 +
          8 * a5 * b3 * b5 * c2 * c4 * d3 + 8 * a5 * b3 * b5 * c3 * c4 * d2 -
          4 * a5 * b3 * b5 * c3 * c5 * d1 - 8 * a5 * b4 * b5 * c2 * c3 * d3,
      a1 * pow(b3, 2) * pow(c0, 3) - 4 * a3 * pow(b0, 2) * pow(c1, 3) -
          4 * a1 * pow(b4, 2) * pow(c0, 3) - 8 * a0 * pow(b5, 2) * pow(c1, 3) -
          a1 * pow(b0, 2) * pow(c5, 3) + a1 * pow(b5, 2) * pow(c0, 3) -
          16 * a3 * pow(b2, 2) * pow(c1, 3) + 8 * a4 * pow(b0, 2) * pow(c2, 3) +
          4 * a5 * pow(b0, 2) * pow(c1, 3) - 12 * a1 * pow(b1, 2) * pow(c5, 3) +
          32 * a4 * pow(b1, 2) * pow(c2, 3) +
          16 * a5 * pow(b2, 2) * pow(c1, 3) - a1 * pow(b3, 2) * pow(c5, 3) -
          4 * a3 * pow(b5, 2) * pow(c1, 3) + 8 * a4 * pow(b3, 2) * pow(c2, 3) +
          12 * a5 * pow(b5, 2) * pow(c1, 3) - 8 * a0 * b0 * b3 * pow(c1, 3) -
          2 * a0 * b0 * b1 * pow(c5, 3) + 16 * a0 * b0 * b4 * pow(c2, 3) +
          8 * a0 * b0 * b5 * pow(c1, 3) + 2 * a3 * b1 * b3 * pow(c0, 3) +
          64 * a1 * b1 * b4 * pow(c2, 3) - 32 * a2 * b2 * b3 * pow(c1, 3) +
          2 * a0 * b1 * b3 * pow(c5, 3) - 16 * a0 * b3 * b4 * pow(c2, 3) +
          8 * a0 * b3 * b5 * pow(c1, 3) + 2 * a1 * b0 * b3 * pow(c5, 3) -
          2 * a1 * b3 * b5 * pow(c0, 3) + 4 * a2 * b3 * b4 * pow(c0, 3) +
          2 * a3 * b0 * b1 * pow(c5, 3) - 16 * a3 * b0 * b4 * pow(c2, 3) +
          8 * a3 * b0 * b5 * pow(c1, 3) - 2 * a3 * b1 * b5 * pow(c0, 3) +
          4 * a3 * b2 * b4 * pow(c0, 3) - 16 * a4 * b0 * b3 * pow(c2, 3) -
          8 * a4 * b1 * b4 * pow(c0, 3) + 4 * a4 * b2 * b3 * pow(c0, 3) +
          8 * a5 * b0 * b3 * pow(c1, 3) - 2 * a5 * b1 * b3 * pow(c0, 3) +
          32 * a2 * b2 * b5 * pow(c1, 3) - 4 * a2 * b4 * b5 * pow(c0, 3) -
          4 * a4 * b2 * b5 * pow(c0, 3) - 16 * a5 * b0 * b5 * pow(c1, 3) +
          2 * a5 * b1 * b5 * pow(c0, 3) - 4 * a5 * b2 * b4 * pow(c0, 3) -
          2 * a3 * b1 * b3 * pow(c5, 3) + 16 * a3 * b3 * b4 * pow(c2, 3) -
          8 * a5 * b3 * b5 * pow(c1, 3) -
          3 * a0 * pow(b0, 2) * c1 * pow(c3, 2) -
          a0 * pow(b3, 2) * pow(c0, 2) * c1 +
          a1 * pow(b0, 2) * c0 * pow(c3, 2) +
          12 * a0 * pow(b0, 2) * c1 * pow(c4, 2) +
          4 * a0 * pow(b4, 2) * pow(c0, 2) * c1 -
          4 * a1 * pow(b0, 2) * c0 * pow(c4, 2) +
          4 * a1 * pow(b0, 2) * pow(c1, 2) * c3 +
          12 * a1 * pow(b1, 2) * pow(c0, 2) * c3 -
          4 * a3 * pow(b1, 2) * pow(c0, 2) * c1 -
          3 * a0 * pow(b0, 2) * c1 * pow(c5, 2) -
          4 * a0 * pow(b2, 2) * c1 * pow(c3, 2) +
          4 * a0 * pow(b3, 2) * c1 * pow(c2, 2) -
          a0 * pow(b5, 2) * pow(c0, 2) * c1 +
          a1 * pow(b0, 2) * c0 * pow(c5, 2) +
          4 * a1 * pow(b0, 2) * pow(c2, 2) * c3 -
          4 * a1 * pow(b2, 2) * c0 * pow(c3, 2) +
          4 * a1 * pow(b2, 2) * pow(c0, 2) * c3 +
          4 * a1 * pow(b3, 2) * c0 * pow(c2, 2) -
          4 * a3 * pow(b0, 2) * c1 * pow(c2, 2) -
          4 * a3 * pow(b2, 2) * pow(c0, 2) * c1 -
          8 * a0 * pow(b1, 2) * c1 * pow(c5, 2) +
          16 * a0 * pow(b2, 2) * c1 * pow(c4, 2) +
          16 * a0 * pow(b4, 2) * c1 * pow(c2, 2) -
          4 * a1 * pow(b0, 2) * pow(c1, 2) * c5 +
          24 * a1 * pow(b1, 2) * c0 * pow(c5, 2) -
          12 * a1 * pow(b1, 2) * pow(c0, 2) * c5 +
          48 * a1 * pow(b1, 2) * pow(c2, 2) * c3 -
          16 * a1 * pow(b2, 2) * c0 * pow(c4, 2) +
          16 * a1 * pow(b2, 2) * pow(c1, 2) * c3 -
          16 * a1 * pow(b4, 2) * c0 * pow(c2, 2) +
          8 * a1 * pow(b5, 2) * c0 * pow(c1, 2) -
          8 * a2 * pow(b0, 2) * pow(c1, 2) * c4 -
          8 * a2 * pow(b1, 2) * pow(c0, 2) * c4 -
          16 * a3 * pow(b1, 2) * c1 * pow(c2, 2) +
          8 * a4 * pow(b0, 2) * pow(c1, 2) * c2 +
          8 * a4 * pow(b1, 2) * pow(c0, 2) * c2 +
          4 * a5 * pow(b1, 2) * pow(c0, 2) * c1 -
          4 * a0 * pow(b2, 2) * c1 * pow(c5, 2) -
          4 * a0 * pow(b5, 2) * c1 * pow(c2, 2) +
          4 * a1 * pow(b0, 2) * c3 * pow(c4, 2) -
          4 * a1 * pow(b0, 2) * pow(c2, 2) * c5 +
          4 * a1 * pow(b2, 2) * c0 * pow(c5, 2) -
          4 * a1 * pow(b2, 2) * pow(c0, 2) * c5 +
          4 * a1 * pow(b4, 2) * pow(c0, 2) * c3 +
          4 * a1 * pow(b5, 2) * c0 * pow(c2, 2) -
          8 * a2 * pow(b0, 2) * pow(c2, 2) * c4 -
          24 * a2 * pow(b2, 2) * pow(c0, 2) * c4 -
          4 * a3 * pow(b0, 2) * c1 * pow(c4, 2) -
          4 * a3 * pow(b4, 2) * pow(c0, 2) * c1 +
          8 * a4 * pow(b2, 2) * pow(c0, 2) * c2 +
          4 * a5 * pow(b0, 2) * c1 * pow(c2, 2) +
          4 * a5 * pow(b2, 2) * pow(c0, 2) * c1 -
          a0 * pow(b3, 2) * c1 * pow(c5, 2) -
          3 * a0 * pow(b5, 2) * c1 * pow(c3, 2) +
          2 * a1 * pow(b0, 2) * c3 * pow(c5, 2) -
          a1 * pow(b0, 2) * pow(c3, 2) * c5 -
          48 * a1 * pow(b1, 2) * pow(c2, 2) * c5 -
          16 * a1 * pow(b2, 2) * pow(c1, 2) * c5 +
          3 * a1 * pow(b3, 2) * c0 * pow(c5, 2) -
          3 * a1 * pow(b3, 2) * pow(c0, 2) * c5 +
          a1 * pow(b5, 2) * c0 * pow(c3, 2) -
          2 * a1 * pow(b5, 2) * pow(c0, 2) * c3 -
          2 * a2 * pow(b0, 2) * pow(c3, 2) * c4 -
          32 * a2 * pow(b1, 2) * pow(c2, 2) * c4 -
          96 * a2 * pow(b2, 2) * pow(c1, 2) * c4 +
          2 * a2 * pow(b3, 2) * pow(c0, 2) * c4 +
          2 * a3 * pow(b0, 2) * c1 * pow(c5, 2) -
          2 * a3 * pow(b5, 2) * pow(c0, 2) * c1 -
          2 * a4 * pow(b0, 2) * c2 * pow(c3, 2) +
          32 * a4 * pow(b2, 2) * pow(c1, 2) * c2 +
          2 * a4 * pow(b3, 2) * pow(c0, 2) * c2 +
          3 * a5 * pow(b0, 2) * c1 * pow(c3, 2) +
          16 * a5 * pow(b1, 2) * c1 * pow(c2, 2) +
          a5 * pow(b3, 2) * pow(c0, 2) * c1 +
          4 * a0 * pow(b4, 2) * c1 * pow(c5, 2) +
          4 * a0 * pow(b5, 2) * c1 * pow(c4, 2) +
          12 * a1 * pow(b1, 2) * c3 * pow(c5, 2) +
          16 * a1 * pow(b2, 2) * c3 * pow(c4, 2) -
          4 * a1 * pow(b4, 2) * c0 * pow(c5, 2) +
          8 * a1 * pow(b4, 2) * pow(c0, 2) * c5 +
          16 * a1 * pow(b4, 2) * pow(c2, 2) * c3 -
          4 * a1 * pow(b5, 2) * c0 * pow(c4, 2) +
          4 * a1 * pow(b5, 2) * pow(c1, 2) * c3 -
          4 * a3 * pow(b1, 2) * c1 * pow(c5, 2) -
          16 * a3 * pow(b2, 2) * c1 * pow(c4, 2) -
          16 * a3 * pow(b4, 2) * c1 * pow(c2, 2) -
          8 * a5 * pow(b0, 2) * c1 * pow(c4, 2) -
          4 * a1 * pow(b2, 2) * c3 * pow(c5, 2) +
          4 * a1 * pow(b2, 2) * pow(c3, 2) * c5 -
          4 * a1 * pow(b3, 2) * pow(c2, 2) * c5 -
          a1 * pow(b5, 2) * pow(c0, 2) * c5 -
          4 * a1 * pow(b5, 2) * pow(c2, 2) * c3 +
          2 * a2 * pow(b0, 2) * c4 * pow(c5, 2) -
          24 * a2 * pow(b2, 2) * pow(c3, 2) * c4 -
          8 * a2 * pow(b3, 2) * pow(c2, 2) * c4 -
          2 * a2 * pow(b5, 2) * pow(c0, 2) * c4 +
          4 * a3 * pow(b2, 2) * c1 * pow(c5, 2) +
          4 * a3 * pow(b5, 2) * c1 * pow(c2, 2) +
          2 * a4 * pow(b0, 2) * c2 * pow(c5, 2) +
          8 * a4 * pow(b2, 2) * c2 * pow(c3, 2) -
          2 * a4 * pow(b5, 2) * pow(c0, 2) * c2 +
          a5 * pow(b0, 2) * c1 * pow(c5, 2) +
          4 * a5 * pow(b2, 2) * c1 * pow(c3, 2) -
          4 * a5 * pow(b3, 2) * c1 * pow(c2, 2) +
          3 * a5 * pow(b5, 2) * pow(c0, 2) * c1 -
          12 * a1 * pow(b5, 2) * pow(c1, 2) * c5 +
          8 * a2 * pow(b1, 2) * c4 * pow(c5, 2) -
          8 * a2 * pow(b5, 2) * pow(c1, 2) * c4 +
          8 * a4 * pow(b1, 2) * c2 * pow(c5, 2) -
          8 * a4 * pow(b5, 2) * pow(c1, 2) * c2 +
          12 * a5 * pow(b1, 2) * c1 * pow(c5, 2) +
          4 * a1 * pow(b4, 2) * c3 * pow(c5, 2) +
          4 * a1 * pow(b5, 2) * c3 * pow(c4, 2) -
          4 * a3 * pow(b4, 2) * c1 * pow(c5, 2) -
          4 * a3 * pow(b5, 2) * c1 * pow(c4, 2) -
          a1 * pow(b5, 2) * pow(c3, 2) * c5 +
          2 * a2 * pow(b3, 2) * c4 * pow(c5, 2) -
          2 * a2 * pow(b5, 2) * pow(c3, 2) * c4 +
          2 * a4 * pow(b3, 2) * c2 * pow(c5, 2) -
          2 * a4 * pow(b5, 2) * c2 * pow(c3, 2) +
          a5 * pow(b3, 2) * c1 * pow(c5, 2) +
          3 * a5 * pow(b5, 2) * c1 * pow(c3, 2) +
          2 * a0 * b0 * b1 * c0 * pow(c3, 2) -
          8 * a0 * b0 * b1 * c0 * pow(c4, 2) +
          8 * a0 * b0 * b1 * pow(c1, 2) * c3 +
          8 * a0 * b1 * b3 * c0 * pow(c1, 2) +
          8 * a1 * b0 * b3 * c0 * pow(c1, 2) +
          8 * a3 * b0 * b1 * c0 * pow(c1, 2) +
          2 * a0 * b0 * b1 * c0 * pow(c5, 2) +
          8 * a0 * b0 * b1 * pow(c2, 2) * c3 -
          8 * a0 * b0 * b3 * c1 * pow(c2, 2) -
          8 * a1 * b1 * b3 * pow(c0, 2) * c1 -
          8 * a0 * b0 * b1 * pow(c1, 2) * c5 -
          16 * a0 * b0 * b2 * pow(c1, 2) * c4 +
          16 * a0 * b0 * b4 * pow(c1, 2) * c2 -
          2 * a0 * b1 * b3 * pow(c0, 2) * c3 -
          8 * a0 * b1 * b5 * c0 * pow(c1, 2) -
          2 * a1 * b0 * b3 * pow(c0, 2) * c3 -
          8 * a1 * b0 * b5 * c0 * pow(c1, 2) -
          2 * a3 * b0 * b1 * pow(c0, 2) * c3 -
          2 * a3 * b0 * b3 * pow(c0, 2) * c1 -
          8 * a5 * b0 * b1 * c0 * pow(c1, 2) +
          8 * a0 * b0 * b1 * c3 * pow(c4, 2) -
          8 * a0 * b0 * b1 * pow(c2, 2) * c5 -
          16 * a0 * b0 * b2 * pow(c2, 2) * c4 -
          8 * a0 * b0 * b3 * c1 * pow(c4, 2) +
          8 * a0 * b0 * b5 * c1 * pow(c2, 2) +
          8 * a0 * b1 * b2 * c2 * pow(c3, 2) -
          16 * a0 * b2 * b4 * c0 * pow(c2, 2) -
          16 * a1 * b0 * b1 * c1 * pow(c5, 2) +
          8 * a1 * b0 * b2 * c2 * pow(c3, 2) -
          16 * a1 * b1 * b2 * pow(c0, 2) * c4 -
          32 * a1 * b1 * b3 * c1 * pow(c2, 2) +
          16 * a1 * b1 * b4 * pow(c0, 2) * c2 +
          8 * a1 * b1 * b5 * pow(c0, 2) * c1 +
          8 * a2 * b0 * b1 * c2 * pow(c3, 2) -
          8 * a2 * b0 * b2 * c1 * pow(c3, 2) -
          16 * a2 * b0 * b4 * c0 * pow(c2, 2) -
          8 * a2 * b1 * b2 * c0 * pow(c3, 2) +
          8 * a2 * b1 * b2 * pow(c0, 2) * c3 -
          8 * a2 * b2 * b3 * pow(c0, 2) * c1 -
          16 * a4 * b0 * b2 * c0 * pow(c2, 2) +
          4 * a0 * b0 * b1 * c3 * pow(c5, 2) -
          2 * a0 * b0 * b1 * pow(c3, 2) * c5 -
          4 * a0 * b0 * b2 * pow(c3, 2) * c4 +
          4 * a0 * b0 * b3 * c1 * pow(c5, 2) -
          4 * a0 * b0 * b4 * c2 * pow(c3, 2) +
          6 * a0 * b0 * b5 * c1 * pow(c3, 2) -
          4 * a0 * b1 * b3 * c0 * pow(c5, 2) +
          2 * a0 * b1 * b3 * pow(c0, 2) * c5 -
          8 * a0 * b1 * b3 * pow(c2, 2) * c3 +
          8 * a0 * b1 * b4 * pow(c0, 2) * c4 -
          2 * a0 * b1 * b5 * c0 * pow(c3, 2) +
          2 * a0 * b1 * b5 * pow(c0, 2) * c3 -
          4 * a0 * b2 * b3 * pow(c0, 2) * c4 +
          4 * a0 * b2 * b4 * c0 * pow(c3, 2) -
          4 * a0 * b2 * b4 * pow(c0, 2) * c3 -
          4 * a0 * b3 * b4 * pow(c0, 2) * c2 +
          2 * a0 * b3 * b5 * pow(c0, 2) * c1 -
          4 * a1 * b0 * b3 * c0 * pow(c5, 2) +
          2 * a1 * b0 * b3 * pow(c0, 2) * c5 -
          8 * a1 * b0 * b3 * pow(c2, 2) * c3 +
          8 * a1 * b0 * b4 * pow(c0, 2) * c4 -
          2 * a1 * b0 * b5 * c0 * pow(c3, 2) +
          2 * a1 * b0 * b5 * pow(c0, 2) * c3 +
          32 * a1 * b2 * b3 * pow(c1, 2) * c2 +
          32 * a2 * b0 * b2 * c1 * pow(c4, 2) -
          4 * a2 * b0 * b3 * pow(c0, 2) * c4 +
          4 * a2 * b0 * b4 * c0 * pow(c3, 2) -
          4 * a2 * b0 * b4 * pow(c0, 2) * c3 -
          32 * a2 * b1 * b2 * c0 * pow(c4, 2) +
          32 * a2 * b1 * b2 * pow(c1, 2) * c3 +
          32 * a2 * b1 * b3 * pow(c1, 2) * c2 -
          4 * a3 * b0 * b1 * c0 * pow(c5, 2) +
          2 * a3 * b0 * b1 * pow(c0, 2) * c5 -
          8 * a3 * b0 * b1 * pow(c2, 2) * c3 -
          4 * a3 * b0 * b2 * pow(c0, 2) * c4 +
          8 * a3 * b0 * b3 * c1 * pow(c2, 2) -
          4 * a3 * b0 * b4 * pow(c0, 2) * c2 +
          2 * a3 * b0 * b5 * pow(c0, 2) * c1 +
          32 * a3 * b1 * b2 * pow(c1, 2) * c2 +
          8 * a3 * b1 * b3 * c0 * pow(c2, 2) +
          8 * a4 * b0 * b1 * pow(c0, 2) * c4 +
          4 * a4 * b0 * b2 * c0 * pow(c3, 2) -
          4 * a4 * b0 * b2 * pow(c0, 2) * c3 -
          4 * a4 * b0 * b3 * pow(c0, 2) * c2 +
          8 * a4 * b0 * b4 * pow(c0, 2) * c1 -
          2 * a5 * b0 * b1 * c0 * pow(c3, 2) +
          2 * a5 * b0 * b1 * pow(c0, 2) * c3 +
          2 * a5 * b0 * b3 * pow(c0, 2) * c1 -
          16 * a0 * b0 * b5 * c1 * pow(c4, 2) -
          8 * a0 * b1 * b3 * pow(c1, 2) * c5 +
          8 * a0 * b1 * b5 * c0 * pow(c4, 2) -
          8 * a0 * b1 * b5 * pow(c1, 2) * c3 -
          16 * a0 * b3 * b4 * pow(c1, 2) * c2 -
          8 * a1 * b0 * b3 * pow(c1, 2) * c5 +
          8 * a1 * b0 * b5 * c0 * pow(c4, 2) -
          8 * a1 * b0 * b5 * pow(c1, 2) * c3 -
          64 * a1 * b1 * b2 * pow(c2, 2) * c4 +
          32 * a1 * b1 * b5 * c1 * pow(c2, 2) -
          64 * a1 * b2 * b4 * c1 * pow(c2, 2) -
          8 * a1 * b3 * b5 * c0 * pow(c1, 2) -
          8 * a2 * b0 * b2 * c1 * pow(c5, 2) +
          8 * a2 * b1 * b2 * c0 * pow(c5, 2) -
          8 * a2 * b1 * b2 * pow(c0, 2) * c5 -
          64 * a2 * b1 * b4 * c1 * pow(c2, 2) +
          16 * a2 * b2 * b4 * pow(c0, 2) * c2 +
          8 * a2 * b2 * b5 * pow(c0, 2) * c1 +
          16 * a2 * b3 * b4 * c0 * pow(c1, 2) -
          8 * a3 * b0 * b1 * pow(c1, 2) * c5 -
          16 * a3 * b0 * b4 * pow(c1, 2) * c2 -
          8 * a3 * b1 * b5 * c0 * pow(c1, 2) +
          16 * a3 * b2 * b4 * c0 * pow(c1, 2) -
          16 * a4 * b0 * b3 * pow(c1, 2) * c2 -
          64 * a4 * b1 * b2 * c1 * pow(c2, 2) +
          16 * a4 * b2 * b3 * c0 * pow(c1, 2) +
          8 * a5 * b0 * b1 * c0 * pow(c4, 2) -
          8 * a5 * b0 * b1 * pow(c1, 2) * c3 -
          8 * a5 * b1 * b3 * c0 * pow(c1, 2) +
          4 * a0 * b0 * b2 * c4 * pow(c5, 2) +
          4 * a0 * b0 * b4 * c2 * pow(c5, 2) +
          2 * a0 * b0 * b5 * c1 * pow(c5, 2) +
          8 * a0 * b1 * b3 * pow(c2, 2) * c5 +
          2 * a0 * b1 * b5 * c0 * pow(c5, 2) -
          2 * a0 * b1 * b5 * pow(c0, 2) * c5 +
          16 * a0 * b2 * b3 * pow(c2, 2) * c4 -
          4 * a0 * b2 * b4 * c0 * pow(c5, 2) +
          4 * a0 * b2 * b4 * pow(c0, 2) * c5 +
          16 * a0 * b2 * b4 * pow(c2, 2) * c3 +
          4 * a0 * b2 * b5 * pow(c0, 2) * c4 +
          4 * a0 * b4 * b5 * pow(c0, 2) * c2 +
          8 * a1 * b0 * b3 * pow(c2, 2) * c5 +
          2 * a1 * b0 * b5 * c0 * pow(c5, 2) -
          2 * a1 * b0 * b5 * pow(c0, 2) * c5 -
          8 * a1 * b1 * b3 * c1 * pow(c5, 2) -
          32 * a1 * b2 * b5 * pow(c1, 2) * c2 -
          8 * a1 * b3 * b5 * c0 * pow(c2, 2) +
          16 * a2 * b0 * b3 * pow(c2, 2) * c4 -
          4 * a2 * b0 * b4 * c0 * pow(c5, 2) +
          4 * a2 * b0 * b4 * pow(c0, 2) * c5 +
          16 * a2 * b0 * b4 * pow(c2, 2) * c3 +
          4 * a2 * b0 * b5 * pow(c0, 2) * c4 -
          32 * a2 * b1 * b2 * pow(c1, 2) * c5 -
          32 * a2 * b1 * b5 * pow(c1, 2) * c2 +
          64 * a2 * b2 * b4 * pow(c1, 2) * c2 +
          16 * a2 * b3 * b4 * c0 * pow(c2, 2) +
          8 * a3 * b0 * b1 * pow(c2, 2) * c5 +
          16 * a3 * b0 * b2 * pow(c2, 2) * c4 -
          8 * a3 * b1 * b5 * c0 * pow(c2, 2) +
          16 * a3 * b2 * b4 * c0 * pow(c2, 2) -
          4 * a4 * b0 * b2 * c0 * pow(c5, 2) +
          4 * a4 * b0 * b2 * pow(c0, 2) * c5 +
          16 * a4 * b0 * b2 * pow(c2, 2) * c3 +
          32 * a4 * b0 * b4 * c1 * pow(c2, 2) +
          4 * a4 * b0 * b5 * pow(c0, 2) * c2 -
          32 * a4 * b1 * b4 * c0 * pow(c2, 2) +
          16 * a4 * b2 * b3 * c0 * pow(c2, 2) +
          2 * a5 * b0 * b1 * c0 * pow(c5, 2) -
          2 * a5 * b0 * b1 * pow(c0, 2) * c5 +
          4 * a5 * b0 * b2 * pow(c0, 2) * c4 +
          4 * a5 * b0 * b4 * pow(c0, 2) * c2 -
          2 * a5 * b0 * b5 * pow(c0, 2) * c1 -
          32 * a5 * b1 * b2 * pow(c1, 2) * c2 -
          8 * a5 * b1 * b3 * c0 * pow(c2, 2) -
          2 * a0 * b1 * b3 * c3 * pow(c5, 2) +
          16 * a0 * b1 * b5 * pow(c1, 2) * c5 +
          16 * a0 * b2 * b5 * pow(c1, 2) * c4 -
          2 * a1 * b0 * b3 * c3 * pow(c5, 2) +
          16 * a1 * b0 * b5 * pow(c1, 2) * c5 +
          2 * a1 * b3 * b5 * pow(c0, 2) * c3 +
          16 * a2 * b0 * b5 * pow(c1, 2) * c4 +
          32 * a2 * b1 * b2 * c3 * pow(c4, 2) -
          32 * a2 * b2 * b3 * c1 * pow(c4, 2) -
          4 * a2 * b3 * b4 * pow(c0, 2) * c3 -
          16 * a2 * b4 * b5 * c0 * pow(c1, 2) -
          2 * a3 * b0 * b1 * c3 * pow(c5, 2) -
          2 * a3 * b0 * b3 * c1 * pow(c5, 2) +
          6 * a3 * b1 * b3 * c0 * pow(c5, 2) -
          6 * a3 * b1 * b3 * pow(c0, 2) * c5 +
          2 * a3 * b1 * b5 * pow(c0, 2) * c3 +
          4 * a3 * b2 * b3 * pow(c0, 2) * c4 -
          4 * a3 * b2 * b4 * pow(c0, 2) * c3 +
          4 * a3 * b3 * b4 * pow(c0, 2) * c2 +
          2 * a3 * b3 * b5 * pow(c0, 2) * c1 +
          8 * a4 * b1 * b4 * pow(c0, 2) * c3 -
          4 * a4 * b2 * b3 * pow(c0, 2) * c3 -
          16 * a4 * b2 * b5 * c0 * pow(c1, 2) -
          8 * a4 * b3 * b4 * pow(c0, 2) * c1 +
          16 * a5 * b0 * b1 * pow(c1, 2) * c5 +
          16 * a5 * b0 * b2 * pow(c1, 2) * c4 +
          2 * a5 * b1 * b3 * pow(c0, 2) * c3 +
          16 * a5 * b1 * b5 * c0 * pow(c1, 2) -
          16 * a5 * b2 * b4 * c0 * pow(c1, 2) -
          8 * a0 * b1 * b5 * c3 * pow(c4, 2) +
          8 * a0 * b3 * b5 * c1 * pow(c4, 2) -
          8 * a1 * b0 * b5 * c3 * pow(c4, 2) +
          16 * a1 * b1 * b2 * c4 * pow(c5, 2) +
          16 * a1 * b1 * b4 * c2 * pow(c5, 2) +
          24 * a1 * b1 * b5 * c1 * pow(c5, 2) -
          16 * a1 * b2 * b4 * c1 * pow(c5, 2) -
          8 * a1 * b2 * b5 * c2 * pow(c3, 2) -
          8 * a2 * b1 * b2 * c3 * pow(c5, 2) +
          8 * a2 * b1 * b2 * pow(c3, 2) * c5 -
          16 * a2 * b1 * b4 * c1 * pow(c5, 2) -
          8 * a2 * b1 * b5 * c2 * pow(c3, 2) +
          8 * a2 * b2 * b3 * c1 * pow(c5, 2) +
          16 * a2 * b2 * b4 * c2 * pow(c3, 2) +
          8 * a2 * b2 * b5 * c1 * pow(c3, 2) +
          8 * a3 * b0 * b5 * c1 * pow(c4, 2) -
          16 * a4 * b1 * b2 * c1 * pow(c5, 2) -
          8 * a5 * b0 * b1 * c3 * pow(c4, 2) +
          8 * a5 * b0 * b3 * c1 * pow(c4, 2) -
          8 * a5 * b0 * b5 * c1 * pow(c2, 2) -
          8 * a5 * b1 * b2 * c2 * pow(c3, 2) +
          8 * a5 * b1 * b5 * c0 * pow(c2, 2) -
          2 * a0 * b1 * b5 * c3 * pow(c5, 2) +
          2 * a0 * b1 * b5 * pow(c3, 2) * c5 -
          4 * a0 * b2 * b3 * c4 * pow(c5, 2) +
          4 * a0 * b2 * b4 * c3 * pow(c5, 2) -
          4 * a0 * b2 * b4 * pow(c3, 2) * c5 +
          4 * a0 * b2 * b5 * pow(c3, 2) * c4 -
          4 * a0 * b3 * b4 * c2 * pow(c5, 2) -
          2 * a0 * b3 * b5 * c1 * pow(c5, 2) +
          4 * a0 * b4 * b5 * c2 * pow(c3, 2) -
          2 * a1 * b0 * b5 * c3 * pow(c5, 2) +
          2 * a1 * b0 * b5 * pow(c3, 2) * c5 -
          2 * a1 * b3 * b5 * c0 * pow(c5, 2) +
          4 * a1 * b3 * b5 * pow(c0, 2) * c5 +
          8 * a1 * b3 * b5 * pow(c2, 2) * c3 -
          8 * a1 * b4 * b5 * pow(c0, 2) * c4 -
          4 * a2 * b0 * b3 * c4 * pow(c5, 2) +
          4 * a2 * b0 * b4 * c3 * pow(c5, 2) -
          4 * a2 * b0 * b4 * pow(c3, 2) * c5 +
          4 * a2 * b0 * b5 * pow(c3, 2) * c4 +
          4 * a2 * b3 * b4 * c0 * pow(c5, 2) -
          8 * a2 * b3 * b4 * pow(c0, 2) * c5 -
          16 * a2 * b3 * b4 * pow(c2, 2) * c3 -
          4 * a2 * b4 * b5 * c0 * pow(c3, 2) +
          8 * a2 * b4 * b5 * pow(c0, 2) * c3 -
          4 * a3 * b0 * b2 * c4 * pow(c5, 2) -
          4 * a3 * b0 * b4 * c2 * pow(c5, 2) -
          2 * a3 * b0 * b5 * c1 * pow(c5, 2) -
          8 * a3 * b1 * b3 * pow(c2, 2) * c5 -
          2 * a3 * b1 * b5 * c0 * pow(c5, 2) +
          4 * a3 * b1 * b5 * pow(c0, 2) * c5 +
          8 * a3 * b1 * b5 * pow(c2, 2) * c3 -
          16 * a3 * b2 * b3 * pow(c2, 2) * c4 +
          4 * a3 * b2 * b4 * c0 * pow(c5, 2) -
          8 * a3 * b2 * b4 * pow(c0, 2) * c5 -
          16 * a3 * b2 * b4 * pow(c2, 2) * c3 -
          8 * a3 * b3 * b5 * c1 * pow(c2, 2) +
          4 * a4 * b0 * b2 * c3 * pow(c5, 2) -
          4 * a4 * b0 * b2 * pow(c3, 2) * c5 -
          4 * a4 * b0 * b3 * c2 * pow(c5, 2) +
          8 * a4 * b0 * b4 * c1 * pow(c5, 2) +
          4 * a4 * b0 * b5 * c2 * pow(c3, 2) -
          8 * a4 * b1 * b4 * c0 * pow(c5, 2) +
          16 * a4 * b1 * b4 * pow(c0, 2) * c5 +
          32 * a4 * b1 * b4 * pow(c2, 2) * c3 -
          8 * a4 * b1 * b5 * pow(c0, 2) * c4 +
          4 * a4 * b2 * b3 * c0 * pow(c5, 2) -
          8 * a4 * b2 * b3 * pow(c0, 2) * c5 -
          16 * a4 * b2 * b3 * pow(c2, 2) * c3 -
          4 * a4 * b2 * b5 * c0 * pow(c3, 2) +
          8 * a4 * b2 * b5 * pow(c0, 2) * c3 -
          32 * a4 * b3 * b4 * c1 * pow(c2, 2) -
          2 * a5 * b0 * b1 * c3 * pow(c5, 2) +
          2 * a5 * b0 * b1 * pow(c3, 2) * c5 +
          4 * a5 * b0 * b2 * pow(c3, 2) * c4 -
          2 * a5 * b0 * b3 * c1 * pow(c5, 2) +
          4 * a5 * b0 * b4 * c2 * pow(c3, 2) -
          6 * a5 * b0 * b5 * c1 * pow(c3, 2) -
          2 * a5 * b1 * b3 * c0 * pow(c5, 2) +
          4 * a5 * b1 * b3 * pow(c0, 2) * c5 +
          8 * a5 * b1 * b3 * pow(c2, 2) * c3 -
          8 * a5 * b1 * b4 * pow(c0, 2) * c4 +
          2 * a5 * b1 * b5 * c0 * pow(c3, 2) -
          4 * a5 * b1 * b5 * pow(c0, 2) * c3 -
          4 * a5 * b2 * b4 * c0 * pow(c3, 2) +
          8 * a5 * b2 * b4 * pow(c0, 2) * c3 -
          4 * a5 * b3 * b5 * pow(c0, 2) * c1 +
          8 * a1 * b3 * b5 * pow(c1, 2) * c5 -
          16 * a2 * b3 * b4 * pow(c1, 2) * c5 +
          8 * a3 * b1 * b5 * pow(c1, 2) * c5 -
          16 * a3 * b2 * b4 * pow(c1, 2) * c5 +
          16 * a3 * b4 * b5 * pow(c1, 2) * c2 -
          16 * a4 * b2 * b3 * pow(c1, 2) * c5 +
          16 * a4 * b3 * b5 * pow(c1, 2) * c2 +
          8 * a5 * b0 * b5 * c1 * pow(c4, 2) +
          8 * a5 * b1 * b3 * pow(c1, 2) * c5 -
          8 * a5 * b1 * b5 * c0 * pow(c4, 2) +
          8 * a5 * b1 * b5 * pow(c1, 2) * c3 +
          16 * a5 * b3 * b4 * pow(c1, 2) * c2 +
          4 * a2 * b4 * b5 * pow(c0, 2) * c5 +
          4 * a4 * b2 * b5 * pow(c0, 2) * c5 -
          2 * a5 * b1 * b5 * pow(c0, 2) * c5 -
          8 * a5 * b1 * b5 * pow(c2, 2) * c3 +
          4 * a5 * b2 * b4 * pow(c0, 2) * c5 -
          4 * a5 * b2 * b5 * pow(c0, 2) * c4 +
          8 * a5 * b3 * b5 * c1 * pow(c2, 2) -
          4 * a5 * b4 * b5 * pow(c0, 2) * c2 +
          2 * a1 * b3 * b5 * c3 * pow(c5, 2) -
          4 * a2 * b3 * b4 * c3 * pow(c5, 2) +
          16 * a2 * b4 * b5 * pow(c1, 2) * c5 +
          2 * a3 * b1 * b5 * c3 * pow(c5, 2) +
          4 * a3 * b2 * b3 * c4 * pow(c5, 2) -
          4 * a3 * b2 * b4 * c3 * pow(c5, 2) +
          4 * a3 * b3 * b4 * c2 * pow(c5, 2) +
          2 * a3 * b3 * b5 * c1 * pow(c5, 2) +
          8 * a4 * b1 * b4 * c3 * pow(c5, 2) -
          4 * a4 * b2 * b3 * c3 * pow(c5, 2) +
          16 * a4 * b2 * b5 * pow(c1, 2) * c5 -
          8 * a4 * b3 * b4 * c1 * pow(c5, 2) +
          2 * a5 * b1 * b3 * c3 * pow(c5, 2) -
          24 * a5 * b1 * b5 * pow(c1, 2) * c5 +
          16 * a5 * b2 * b4 * pow(c1, 2) * c5 -
          16 * a5 * b2 * b5 * pow(c1, 2) * c4 -
          16 * a5 * b4 * b5 * pow(c1, 2) * c2 +
          8 * a5 * b1 * b5 * c3 * pow(c4, 2) -
          8 * a5 * b3 * b5 * c1 * pow(c4, 2) +
          4 * a2 * b4 * b5 * pow(c3, 2) * c5 +
          4 * a4 * b2 * b5 * pow(c3, 2) * c5 -
          2 * a5 * b1 * b5 * pow(c3, 2) * c5 +
          4 * a5 * b2 * b4 * pow(c3, 2) * c5 -
          4 * a5 * b2 * b5 * pow(c3, 2) * c4 -
          4 * a5 * b4 * b5 * c2 * pow(c3, 2) -
          8 * a0 * pow(b1, 2) * c0 * c1 * c3 +
          8 * a0 * pow(b1, 2) * c0 * c1 * c5 +
          2 * a3 * pow(b0, 2) * c0 * c1 * c3 +
          16 * a0 * pow(b2, 2) * c0 * c2 * c4 -
          8 * a2 * pow(b3, 2) * c0 * c1 * c2 +
          6 * a0 * pow(b0, 2) * c1 * c3 * c5 -
          12 * a0 * pow(b0, 2) * c2 * c3 * c4 +
          2 * a0 * pow(b3, 2) * c0 * c1 * c5 -
          4 * a0 * pow(b3, 2) * c0 * c2 * c4 +
          4 * a0 * pow(b5, 2) * c0 * c1 * c3 -
          2 * a1 * pow(b0, 2) * c0 * c3 * c5 +
          4 * a2 * pow(b0, 2) * c0 * c3 * c4 -
          32 * a2 * pow(b1, 2) * c1 * c2 * c3 -
          2 * a3 * pow(b0, 2) * c0 * c1 * c5 +
          4 * a3 * pow(b0, 2) * c0 * c2 * c4 +
          8 * a3 * pow(b2, 2) * c0 * c1 * c3 -
          8 * a4 * pow(b0, 2) * c0 * c1 * c4 +
          4 * a4 * pow(b0, 2) * c0 * c2 * c3 -
          2 * a5 * pow(b0, 2) * c0 * c1 * c3 +
          8 * a0 * pow(b1, 2) * c1 * c3 * c5 -
          16 * a0 * pow(b1, 2) * c2 * c3 * c4 -
          8 * a0 * pow(b4, 2) * c0 * c1 * c5 -
          24 * a1 * pow(b1, 2) * c0 * c3 * c5 +
          64 * a1 * pow(b2, 2) * c1 * c2 * c4 +
          16 * a2 * pow(b1, 2) * c0 * c3 * c4 +
          8 * a3 * pow(b1, 2) * c0 * c1 * c5 +
          8 * a5 * pow(b1, 2) * c0 * c1 * c3 +
          12 * a0 * pow(b0, 2) * c2 * c4 * c5 +
          8 * a0 * pow(b2, 2) * c1 * c3 * c5 -
          16 * a0 * pow(b2, 2) * c2 * c3 * c4 -
          2 * a0 * pow(b5, 2) * c0 * c1 * c5 +
          4 * a0 * pow(b5, 2) * c0 * c2 * c4 -
          4 * a2 * pow(b0, 2) * c0 * c4 * c5 +
          32 * a2 * pow(b1, 2) * c1 * c2 * c5 +
          48 * a2 * pow(b2, 2) * c0 * c3 * c4 -
          16 * a3 * pow(b2, 2) * c0 * c2 * c4 -
          4 * a4 * pow(b0, 2) * c0 * c2 * c5 -
          16 * a4 * pow(b2, 2) * c0 * c2 * c3 +
          2 * a5 * pow(b0, 2) * c0 * c1 * c5 -
          4 * a5 * pow(b0, 2) * c0 * c2 * c4 -
          8 * a5 * pow(b2, 2) * c0 * c1 * c3 +
          16 * a0 * pow(b1, 2) * c2 * c4 * c5 -
          2 * a3 * pow(b0, 2) * c1 * c3 * c5 +
          4 * a3 * pow(b0, 2) * c2 * c3 * c4 +
          2 * a3 * pow(b5, 2) * c0 * c1 * c3 -
          16 * a4 * pow(b1, 2) * c0 * c2 * c5 -
          16 * a5 * pow(b1, 2) * c0 * c1 * c5 -
          8 * a1 * pow(b4, 2) * c0 * c3 * c5 +
          16 * a1 * pow(b5, 2) * c1 * c2 * c4 +
          8 * a2 * pow(b3, 2) * c1 * c2 * c5 +
          8 * a3 * pow(b4, 2) * c0 * c1 * c5 +
          4 * a0 * pow(b3, 2) * c2 * c4 * c5 +
          2 * a0 * pow(b5, 2) * c1 * c3 * c5 -
          4 * a0 * pow(b5, 2) * c2 * c3 * c4 +
          2 * a1 * pow(b5, 2) * c0 * c3 * c5 -
          4 * a2 * pow(b3, 2) * c0 * c4 * c5 +
          4 * a2 * pow(b5, 2) * c0 * c3 * c4 -
          8 * a3 * pow(b0, 2) * c2 * c4 * c5 -
          8 * a3 * pow(b2, 2) * c1 * c3 * c5 +
          16 * a3 * pow(b2, 2) * c2 * c3 * c4 +
          2 * a3 * pow(b5, 2) * c0 * c1 * c5 -
          4 * a3 * pow(b5, 2) * c0 * c2 * c4 +
          8 * a4 * pow(b0, 2) * c1 * c4 * c5 -
          4 * a4 * pow(b3, 2) * c0 * c2 * c5 +
          4 * a4 * pow(b5, 2) * c0 * c2 * c3 -
          4 * a5 * pow(b0, 2) * c1 * c3 * c5 +
          8 * a5 * pow(b0, 2) * c2 * c3 * c4 -
          2 * a5 * pow(b3, 2) * c0 * c1 * c5 +
          4 * a5 * pow(b3, 2) * c0 * c2 * c4 -
          6 * a5 * pow(b5, 2) * c0 * c1 * c3 -
          16 * a2 * pow(b1, 2) * c3 * c4 * c5 -
          8 * a5 * pow(b1, 2) * c1 * c3 * c5 +
          16 * a5 * pow(b1, 2) * c2 * c3 * c4 -
          4 * a5 * pow(b0, 2) * c2 * c4 * c5 -
          2 * a3 * pow(b5, 2) * c1 * c3 * c5 +
          4 * a3 * pow(b5, 2) * c2 * c3 * c4 -
          16 * a5 * pow(b1, 2) * c2 * c4 * c5 -
          4 * a5 * pow(b3, 2) * c2 * c4 * c5 -
          16 * a1 * b0 * b1 * c0 * c1 * c3 + 4 * a0 * b0 * b3 * c0 * c1 * c3 +
          16 * a0 * b1 * b2 * c0 * c1 * c4 - 8 * a0 * b1 * b2 * c0 * c2 * c3 -
          16 * a0 * b1 * b4 * c0 * c1 * c2 + 8 * a0 * b2 * b3 * c0 * c1 * c2 +
          16 * a1 * b0 * b1 * c0 * c1 * c5 + 16 * a1 * b0 * b2 * c0 * c1 * c4 -
          8 * a1 * b0 * b2 * c0 * c2 * c3 - 16 * a1 * b0 * b4 * c0 * c1 * c2 +
          16 * a2 * b0 * b1 * c0 * c1 * c4 - 8 * a2 * b0 * b1 * c0 * c2 * c3 +
          8 * a2 * b0 * b3 * c0 * c1 * c2 + 8 * a3 * b0 * b2 * c0 * c1 * c2 -
          16 * a4 * b0 * b1 * c0 * c1 * c2 - 4 * a0 * b0 * b1 * c0 * c3 * c5 +
          8 * a0 * b0 * b2 * c0 * c3 * c4 - 4 * a0 * b0 * b3 * c0 * c1 * c5 +
          8 * a0 * b0 * b3 * c0 * c2 * c4 - 16 * a0 * b0 * b4 * c0 * c1 * c4 +
          8 * a0 * b0 * b4 * c0 * c2 * c3 - 4 * a0 * b0 * b5 * c0 * c1 * c3 +
          8 * a0 * b1 * b2 * c0 * c2 * c5 - 8 * a0 * b2 * b5 * c0 * c1 * c2 +
          8 * a1 * b0 * b2 * c0 * c2 * c5 - 64 * a1 * b1 * b2 * c1 * c2 * c3 +
          8 * a2 * b0 * b1 * c0 * c2 * c5 + 32 * a2 * b0 * b2 * c0 * c2 * c4 -
          8 * a2 * b0 * b5 * c0 * c1 * c2 - 8 * a5 * b0 * b2 * c0 * c1 * c2 -
          8 * a0 * b0 * b2 * c0 * c4 * c5 - 8 * a0 * b0 * b4 * c0 * c2 * c5 +
          4 * a0 * b0 * b5 * c0 * c1 * c5 - 8 * a0 * b0 * b5 * c0 * c2 * c4 +
          16 * a0 * b1 * b3 * c1 * c2 * c4 + 16 * a0 * b1 * b4 * c1 * c2 * c3 +
          16 * a1 * b0 * b1 * c1 * c3 * c5 - 32 * a1 * b0 * b1 * c2 * c3 * c4 +
          16 * a1 * b0 * b3 * c1 * c2 * c4 + 16 * a1 * b0 * b4 * c1 * c2 * c3 +
          32 * a1 * b1 * b2 * c0 * c3 * c4 + 16 * a1 * b1 * b3 * c0 * c1 * c5 +
          16 * a1 * b1 * b5 * c0 * c1 * c3 - 16 * a1 * b2 * b3 * c0 * c1 * c4 -
          16 * a1 * b2 * b4 * c0 * c1 * c3 - 16 * a2 * b1 * b3 * c0 * c1 * c4 -
          16 * a2 * b1 * b4 * c0 * c1 * c3 + 16 * a2 * b2 * b3 * c0 * c1 * c3 +
          16 * a3 * b0 * b1 * c1 * c2 * c4 - 16 * a3 * b1 * b2 * c0 * c1 * c4 -
          16 * a3 * b2 * b3 * c0 * c1 * c2 + 16 * a4 * b0 * b1 * c1 * c2 * c3 -
          16 * a4 * b1 * b2 * c0 * c1 * c3 - 4 * a0 * b0 * b3 * c1 * c3 * c5 +
          8 * a0 * b0 * b3 * c2 * c3 * c4 + 4 * a0 * b1 * b3 * c0 * c3 * c5 -
          8 * a0 * b1 * b4 * c0 * c3 * c4 + 8 * a0 * b3 * b4 * c0 * c1 * c4 -
          4 * a0 * b3 * b5 * c0 * c1 * c3 + 4 * a1 * b0 * b3 * c0 * c3 * c5 -
          8 * a1 * b0 * b4 * c0 * c3 * c4 + 64 * a1 * b1 * b2 * c1 * c2 * c5 +
          128 * a2 * b1 * b2 * c1 * c2 * c4 + 4 * a3 * b0 * b1 * c0 * c3 * c5 +
          4 * a3 * b0 * b3 * c0 * c1 * c5 - 8 * a3 * b0 * b3 * c0 * c2 * c4 +
          8 * a3 * b0 * b4 * c0 * c1 * c4 - 4 * a3 * b0 * b5 * c0 * c1 * c3 -
          8 * a4 * b0 * b1 * c0 * c3 * c4 + 8 * a4 * b0 * b3 * c0 * c1 * c4 -
          4 * a5 * b0 * b3 * c0 * c1 * c3 - 16 * a0 * b1 * b2 * c1 * c4 * c5 -
          8 * a0 * b1 * b2 * c2 * c3 * c5 - 16 * a0 * b1 * b5 * c1 * c2 * c4 -
          8 * a0 * b2 * b3 * c1 * c2 * c5 - 32 * a0 * b2 * b4 * c1 * c2 * c4 +
          32 * a1 * b0 * b1 * c2 * c4 * c5 - 16 * a1 * b0 * b2 * c1 * c4 * c5 -
          8 * a1 * b0 * b2 * c2 * c3 * c5 - 16 * a1 * b0 * b5 * c1 * c2 * c4 -
          32 * a1 * b1 * b4 * c0 * c2 * c5 - 32 * a1 * b1 * b5 * c0 * c1 * c5 +
          16 * a1 * b2 * b4 * c0 * c1 * c5 + 32 * a1 * b2 * b4 * c0 * c2 * c4 +
          8 * a1 * b2 * b5 * c0 * c2 * c3 + 16 * a1 * b4 * b5 * c0 * c1 * c2 -
          16 * a2 * b0 * b1 * c1 * c4 * c5 - 8 * a2 * b0 * b1 * c2 * c3 * c5 +
          16 * a2 * b0 * b2 * c1 * c3 * c5 - 32 * a2 * b0 * b2 * c2 * c3 * c4 -
          8 * a2 * b0 * b3 * c1 * c2 * c5 - 32 * a2 * b0 * b4 * c1 * c2 * c4 +
          16 * a2 * b1 * b4 * c0 * c1 * c5 + 32 * a2 * b1 * b4 * c0 * c2 * c4 +
          8 * a2 * b1 * b5 * c0 * c2 * c3 - 32 * a2 * b2 * b3 * c0 * c2 * c4 -
          32 * a2 * b2 * b4 * c0 * c2 * c3 - 16 * a2 * b2 * b5 * c0 * c1 * c3 +
          8 * a2 * b3 * b5 * c0 * c1 * c2 - 8 * a3 * b0 * b2 * c1 * c2 * c5 +
          8 * a3 * b2 * b5 * c0 * c1 * c2 - 32 * a4 * b0 * b2 * c1 * c2 * c4 +
          16 * a4 * b1 * b2 * c0 * c1 * c5 + 32 * a4 * b1 * b2 * c0 * c2 * c4 +
          16 * a4 * b1 * b5 * c0 * c1 * c2 - 16 * a5 * b0 * b1 * c1 * c2 * c4 +
          8 * a5 * b1 * b2 * c0 * c2 * c3 + 16 * a5 * b1 * b4 * c0 * c1 * c2 +
          8 * a5 * b2 * b3 * c0 * c1 * c2 - 16 * a0 * b0 * b3 * c2 * c4 * c5 +
          16 * a0 * b0 * b4 * c1 * c4 * c5 - 8 * a0 * b0 * b5 * c1 * c3 * c5 +
          16 * a0 * b0 * b5 * c2 * c3 * c4 - 8 * a0 * b1 * b4 * c0 * c4 * c5 +
          8 * a0 * b2 * b3 * c0 * c4 * c5 - 8 * a0 * b2 * b5 * c0 * c3 * c4 +
          8 * a0 * b3 * b4 * c0 * c2 * c5 + 8 * a0 * b4 * b5 * c0 * c1 * c4 -
          8 * a0 * b4 * b5 * c0 * c2 * c3 - 8 * a1 * b0 * b4 * c0 * c4 * c5 +
          8 * a2 * b0 * b3 * c0 * c4 * c5 - 8 * a2 * b0 * b5 * c0 * c3 * c4 +
          8 * a3 * b0 * b2 * c0 * c4 * c5 + 8 * a3 * b0 * b4 * c0 * c2 * c5 -
          8 * a4 * b0 * b1 * c0 * c4 * c5 + 8 * a4 * b0 * b3 * c0 * c2 * c5 -
          16 * a4 * b0 * b4 * c0 * c1 * c5 + 8 * a4 * b0 * b5 * c0 * c1 * c4 -
          8 * a4 * b0 * b5 * c0 * c2 * c3 - 8 * a5 * b0 * b2 * c0 * c3 * c4 +
          8 * a5 * b0 * b4 * c0 * c1 * c4 - 8 * a5 * b0 * b4 * c0 * c2 * c3 +
          8 * a5 * b0 * b5 * c0 * c1 * c3 + 8 * a0 * b2 * b5 * c1 * c2 * c5 -
          8 * a1 * b2 * b5 * c0 * c2 * c5 + 8 * a2 * b0 * b5 * c1 * c2 * c5 -
          8 * a2 * b1 * b5 * c0 * c2 * c5 + 8 * a5 * b0 * b2 * c1 * c2 * c5 -
          8 * a5 * b1 * b2 * c0 * c2 * c5 - 8 * a0 * b0 * b5 * c2 * c4 * c5 -
          32 * a1 * b1 * b2 * c3 * c4 * c5 - 16 * a1 * b1 * b5 * c1 * c3 * c5 +
          32 * a1 * b1 * b5 * c2 * c3 * c4 + 16 * a1 * b2 * b3 * c1 * c4 * c5 +
          16 * a1 * b2 * b4 * c1 * c3 * c5 - 32 * a1 * b2 * b4 * c2 * c3 * c4 -
          16 * a1 * b3 * b5 * c1 * c2 * c4 - 16 * a1 * b4 * b5 * c1 * c2 * c3 +
          16 * a2 * b1 * b3 * c1 * c4 * c5 + 16 * a2 * b1 * b4 * c1 * c3 * c5 -
          32 * a2 * b1 * b4 * c2 * c3 * c4 - 16 * a2 * b2 * b3 * c1 * c3 * c5 +
          32 * a2 * b2 * b3 * c2 * c3 * c4 + 32 * a2 * b3 * b4 * c1 * c2 * c4 +
          16 * a3 * b1 * b2 * c1 * c4 * c5 - 16 * a3 * b1 * b5 * c1 * c2 * c4 +
          16 * a3 * b2 * b3 * c1 * c2 * c5 + 32 * a3 * b2 * b4 * c1 * c2 * c4 +
          16 * a4 * b1 * b2 * c1 * c3 * c5 - 32 * a4 * b1 * b2 * c2 * c3 * c4 -
          16 * a4 * b1 * b5 * c1 * c2 * c3 + 32 * a4 * b2 * b3 * c1 * c2 * c4 -
          4 * a5 * b0 * b5 * c0 * c1 * c5 + 8 * a5 * b0 * b5 * c0 * c2 * c4 -
          16 * a5 * b1 * b3 * c1 * c2 * c4 - 16 * a5 * b1 * b4 * c1 * c2 * c3 +
          8 * a0 * b1 * b4 * c3 * c4 * c5 - 8 * a0 * b3 * b4 * c1 * c4 * c5 +
          4 * a0 * b3 * b5 * c1 * c3 * c5 - 8 * a0 * b3 * b5 * c2 * c3 * c4 +
          8 * a1 * b0 * b4 * c3 * c4 * c5 - 4 * a1 * b3 * b5 * c0 * c3 * c5 +
          8 * a1 * b4 * b5 * c0 * c3 * c4 + 8 * a2 * b3 * b4 * c0 * c3 * c5 +
          8 * a3 * b0 * b3 * c2 * c4 * c5 - 8 * a3 * b0 * b4 * c1 * c4 * c5 +
          4 * a3 * b0 * b5 * c1 * c3 * c5 - 8 * a3 * b0 * b5 * c2 * c3 * c4 -
          4 * a3 * b1 * b5 * c0 * c3 * c5 - 8 * a3 * b2 * b3 * c0 * c4 * c5 +
          8 * a3 * b2 * b4 * c0 * c3 * c5 - 8 * a3 * b3 * b4 * c0 * c2 * c5 -
          4 * a3 * b3 * b5 * c0 * c1 * c5 + 8 * a3 * b3 * b5 * c0 * c2 * c4 -
          8 * a3 * b4 * b5 * c0 * c1 * c4 + 8 * a4 * b0 * b1 * c3 * c4 * c5 -
          8 * a4 * b0 * b3 * c1 * c4 * c5 - 16 * a4 * b1 * b4 * c0 * c3 * c5 +
          8 * a4 * b1 * b5 * c0 * c3 * c4 + 8 * a4 * b2 * b3 * c0 * c3 * c5 +
          16 * a4 * b3 * b4 * c0 * c1 * c5 - 8 * a4 * b3 * b5 * c0 * c1 * c4 +
          4 * a5 * b0 * b3 * c1 * c3 * c5 - 8 * a5 * b0 * b3 * c2 * c3 * c4 -
          4 * a5 * b1 * b3 * c0 * c3 * c5 + 8 * a5 * b1 * b4 * c0 * c3 * c4 -
          8 * a5 * b3 * b4 * c0 * c1 * c4 + 4 * a5 * b3 * b5 * c0 * c1 * c3 -
          32 * a1 * b1 * b5 * c2 * c4 * c5 + 8 * a1 * b2 * b5 * c2 * c3 * c5 +
          8 * a2 * b1 * b5 * c2 * c3 * c5 - 8 * a2 * b3 * b5 * c1 * c2 * c5 -
          8 * a3 * b2 * b5 * c1 * c2 * c5 + 8 * a5 * b1 * b2 * c2 * c3 * c5 +
          32 * a5 * b1 * b5 * c1 * c2 * c4 - 8 * a5 * b2 * b3 * c1 * c2 * c5 +
          8 * a0 * b3 * b5 * c2 * c4 * c5 - 8 * a0 * b4 * b5 * c1 * c4 * c5 +
          8 * a1 * b4 * b5 * c0 * c4 * c5 - 8 * a2 * b4 * b5 * c0 * c3 * c5 +
          8 * a3 * b0 * b5 * c2 * c4 * c5 - 8 * a4 * b0 * b5 * c1 * c4 * c5 +
          8 * a4 * b1 * b5 * c0 * c4 * c5 - 8 * a4 * b2 * b5 * c0 * c3 * c5 +
          8 * a5 * b0 * b3 * c2 * c4 * c5 - 8 * a5 * b0 * b4 * c1 * c4 * c5 +
          4 * a5 * b0 * b5 * c1 * c3 * c5 - 8 * a5 * b0 * b5 * c2 * c3 * c4 +
          8 * a5 * b1 * b4 * c0 * c4 * c5 + 4 * a5 * b1 * b5 * c0 * c3 * c5 -
          8 * a5 * b2 * b4 * c0 * c3 * c5 + 8 * a5 * b2 * b5 * c0 * c3 * c4 +
          4 * a5 * b3 * b5 * c0 * c1 * c5 - 8 * a5 * b3 * b5 * c0 * c2 * c4 +
          8 * a5 * b4 * b5 * c0 * c2 * c3 - 8 * a1 * b4 * b5 * c3 * c4 * c5 -
          8 * a3 * b3 * b5 * c2 * c4 * c5 + 8 * a3 * b4 * b5 * c1 * c4 * c5 -
          8 * a4 * b1 * b5 * c3 * c4 * c5 + 8 * a4 * b3 * b5 * c1 * c4 * c5 -
          8 * a5 * b1 * b4 * c3 * c4 * c5 + 8 * a5 * b3 * b4 * c1 * c4 * c5 -
          4 * a5 * b3 * b5 * c1 * c3 * c5 + 8 * a5 * b3 * b5 * c2 * c3 * c4,
      pow(a3, 2) * b1 * pow(d0, 3) - 4 * pow(a0, 2) * b3 * pow(d1, 3) -
          4 * pow(a4, 2) * b1 * pow(d0, 3) - pow(a0, 2) * b1 * pow(d5, 3) +
          8 * pow(a0, 2) * b4 * pow(d2, 3) + 4 * pow(a0, 2) * b5 * pow(d1, 3) -
          16 * pow(a2, 2) * b3 * pow(d1, 3) - 8 * pow(a5, 2) * b0 * pow(d1, 3) +
          pow(a5, 2) * b1 * pow(d0, 3) - 12 * pow(a1, 2) * b1 * pow(d5, 3) +
          32 * pow(a1, 2) * b4 * pow(d2, 3) +
          16 * pow(a2, 2) * b5 * pow(d1, 3) - pow(a3, 2) * b1 * pow(d5, 3) +
          8 * pow(a3, 2) * b4 * pow(d2, 3) - 4 * pow(a5, 2) * b3 * pow(d1, 3) +
          12 * pow(a5, 2) * b5 * pow(d1, 3) - 8 * a0 * a3 * b0 * pow(d1, 3) -
          2 * a0 * a1 * b0 * pow(d5, 3) + 16 * a0 * a4 * b0 * pow(d2, 3) +
          8 * a0 * a5 * b0 * pow(d1, 3) + 2 * a1 * a3 * b3 * pow(d0, 3) +
          64 * a1 * a4 * b1 * pow(d2, 3) - 32 * a2 * a3 * b2 * pow(d1, 3) +
          2 * a0 * a1 * b3 * pow(d5, 3) + 2 * a0 * a3 * b1 * pow(d5, 3) -
          16 * a0 * a3 * b4 * pow(d2, 3) + 8 * a0 * a3 * b5 * pow(d1, 3) -
          16 * a0 * a4 * b3 * pow(d2, 3) + 8 * a0 * a5 * b3 * pow(d1, 3) +
          2 * a1 * a3 * b0 * pow(d5, 3) - 2 * a1 * a3 * b5 * pow(d0, 3) -
          8 * a1 * a4 * b4 * pow(d0, 3) - 2 * a1 * a5 * b3 * pow(d0, 3) +
          4 * a2 * a3 * b4 * pow(d0, 3) + 4 * a2 * a4 * b3 * pow(d0, 3) -
          16 * a3 * a4 * b0 * pow(d2, 3) + 4 * a3 * a4 * b2 * pow(d0, 3) +
          8 * a3 * a5 * b0 * pow(d1, 3) - 2 * a3 * a5 * b1 * pow(d0, 3) +
          32 * a2 * a5 * b2 * pow(d1, 3) - 16 * a0 * a5 * b5 * pow(d1, 3) +
          2 * a1 * a5 * b5 * pow(d0, 3) - 4 * a2 * a4 * b5 * pow(d0, 3) -
          4 * a2 * a5 * b4 * pow(d0, 3) - 4 * a4 * a5 * b2 * pow(d0, 3) -
          2 * a1 * a3 * b3 * pow(d5, 3) + 16 * a3 * a4 * b3 * pow(d2, 3) -
          8 * a3 * a5 * b5 * pow(d1, 3) -
          3 * pow(a0, 2) * b0 * d1 * pow(d3, 2) +
          pow(a0, 2) * b1 * d0 * pow(d3, 2) -
          pow(a3, 2) * b0 * pow(d0, 2) * d1 +
          12 * pow(a0, 2) * b0 * d1 * pow(d4, 2) -
          4 * pow(a0, 2) * b1 * d0 * pow(d4, 2) +
          4 * pow(a0, 2) * b1 * pow(d1, 2) * d3 +
          12 * pow(a1, 2) * b1 * pow(d0, 2) * d3 -
          4 * pow(a1, 2) * b3 * pow(d0, 2) * d1 +
          4 * pow(a4, 2) * b0 * pow(d0, 2) * d1 -
          3 * pow(a0, 2) * b0 * d1 * pow(d5, 2) +
          pow(a0, 2) * b1 * d0 * pow(d5, 2) +
          4 * pow(a0, 2) * b1 * pow(d2, 2) * d3 -
          4 * pow(a0, 2) * b3 * d1 * pow(d2, 2) -
          4 * pow(a2, 2) * b0 * d1 * pow(d3, 2) -
          4 * pow(a2, 2) * b1 * d0 * pow(d3, 2) +
          4 * pow(a2, 2) * b1 * pow(d0, 2) * d3 -
          4 * pow(a2, 2) * b3 * pow(d0, 2) * d1 +
          4 * pow(a3, 2) * b0 * d1 * pow(d2, 2) +
          4 * pow(a3, 2) * b1 * d0 * pow(d2, 2) -
          pow(a5, 2) * b0 * pow(d0, 2) * d1 -
          4 * pow(a0, 2) * b1 * pow(d1, 2) * d5 -
          8 * pow(a0, 2) * b2 * pow(d1, 2) * d4 +
          8 * pow(a0, 2) * b4 * pow(d1, 2) * d2 -
          8 * pow(a1, 2) * b0 * d1 * pow(d5, 2) +
          24 * pow(a1, 2) * b1 * d0 * pow(d5, 2) -
          12 * pow(a1, 2) * b1 * pow(d0, 2) * d5 +
          48 * pow(a1, 2) * b1 * pow(d2, 2) * d3 -
          8 * pow(a1, 2) * b2 * pow(d0, 2) * d4 -
          16 * pow(a1, 2) * b3 * d1 * pow(d2, 2) +
          8 * pow(a1, 2) * b4 * pow(d0, 2) * d2 +
          4 * pow(a1, 2) * b5 * pow(d0, 2) * d1 +
          16 * pow(a2, 2) * b0 * d1 * pow(d4, 2) -
          16 * pow(a2, 2) * b1 * d0 * pow(d4, 2) +
          16 * pow(a2, 2) * b1 * pow(d1, 2) * d3 +
          16 * pow(a4, 2) * b0 * d1 * pow(d2, 2) -
          16 * pow(a4, 2) * b1 * d0 * pow(d2, 2) +
          8 * pow(a5, 2) * b1 * d0 * pow(d1, 2) +
          4 * pow(a0, 2) * b1 * d3 * pow(d4, 2) -
          4 * pow(a0, 2) * b1 * pow(d2, 2) * d5 -
          8 * pow(a0, 2) * b2 * pow(d2, 2) * d4 -
          4 * pow(a0, 2) * b3 * d1 * pow(d4, 2) +
          4 * pow(a0, 2) * b5 * d1 * pow(d2, 2) -
          4 * pow(a2, 2) * b0 * d1 * pow(d5, 2) +
          4 * pow(a2, 2) * b1 * d0 * pow(d5, 2) -
          4 * pow(a2, 2) * b1 * pow(d0, 2) * d5 -
          24 * pow(a2, 2) * b2 * pow(d0, 2) * d4 +
          8 * pow(a2, 2) * b4 * pow(d0, 2) * d2 +
          4 * pow(a2, 2) * b5 * pow(d0, 2) * d1 +
          4 * pow(a4, 2) * b1 * pow(d0, 2) * d3 -
          4 * pow(a4, 2) * b3 * pow(d0, 2) * d1 -
          4 * pow(a5, 2) * b0 * d1 * pow(d2, 2) +
          4 * pow(a5, 2) * b1 * d0 * pow(d2, 2) +
          2 * pow(a0, 2) * b1 * d3 * pow(d5, 2) -
          pow(a0, 2) * b1 * pow(d3, 2) * d5 -
          2 * pow(a0, 2) * b2 * pow(d3, 2) * d4 +
          2 * pow(a0, 2) * b3 * d1 * pow(d5, 2) -
          2 * pow(a0, 2) * b4 * d2 * pow(d3, 2) +
          3 * pow(a0, 2) * b5 * d1 * pow(d3, 2) -
          48 * pow(a1, 2) * b1 * pow(d2, 2) * d5 -
          32 * pow(a1, 2) * b2 * pow(d2, 2) * d4 +
          16 * pow(a1, 2) * b5 * d1 * pow(d2, 2) -
          16 * pow(a2, 2) * b1 * pow(d1, 2) * d5 -
          96 * pow(a2, 2) * b2 * pow(d1, 2) * d4 +
          32 * pow(a2, 2) * b4 * pow(d1, 2) * d2 -
          pow(a3, 2) * b0 * d1 * pow(d5, 2) +
          3 * pow(a3, 2) * b1 * d0 * pow(d5, 2) -
          3 * pow(a3, 2) * b1 * pow(d0, 2) * d5 +
          2 * pow(a3, 2) * b2 * pow(d0, 2) * d4 +
          2 * pow(a3, 2) * b4 * pow(d0, 2) * d2 +
          pow(a3, 2) * b5 * pow(d0, 2) * d1 -
          3 * pow(a5, 2) * b0 * d1 * pow(d3, 2) +
          pow(a5, 2) * b1 * d0 * pow(d3, 2) -
          2 * pow(a5, 2) * b1 * pow(d0, 2) * d3 -
          2 * pow(a5, 2) * b3 * pow(d0, 2) * d1 -
          8 * pow(a0, 2) * b5 * d1 * pow(d4, 2) +
          12 * pow(a1, 2) * b1 * d3 * pow(d5, 2) -
          4 * pow(a1, 2) * b3 * d1 * pow(d5, 2) +
          16 * pow(a2, 2) * b1 * d3 * pow(d4, 2) -
          16 * pow(a2, 2) * b3 * d1 * pow(d4, 2) +
          4 * pow(a4, 2) * b0 * d1 * pow(d5, 2) -
          4 * pow(a4, 2) * b1 * d0 * pow(d5, 2) +
          8 * pow(a4, 2) * b1 * pow(d0, 2) * d5 +
          16 * pow(a4, 2) * b1 * pow(d2, 2) * d3 -
          16 * pow(a4, 2) * b3 * d1 * pow(d2, 2) +
          4 * pow(a5, 2) * b0 * d1 * pow(d4, 2) -
          4 * pow(a5, 2) * b1 * d0 * pow(d4, 2) +
          4 * pow(a5, 2) * b1 * pow(d1, 2) * d3 +
          2 * pow(a0, 2) * b2 * d4 * pow(d5, 2) +
          2 * pow(a0, 2) * b4 * d2 * pow(d5, 2) +
          pow(a0, 2) * b5 * d1 * pow(d5, 2) -
          4 * pow(a2, 2) * b1 * d3 * pow(d5, 2) +
          4 * pow(a2, 2) * b1 * pow(d3, 2) * d5 -
          24 * pow(a2, 2) * b2 * pow(d3, 2) * d4 +
          4 * pow(a2, 2) * b3 * d1 * pow(d5, 2) +
          8 * pow(a2, 2) * b4 * d2 * pow(d3, 2) +
          4 * pow(a2, 2) * b5 * d1 * pow(d3, 2) -
          4 * pow(a3, 2) * b1 * pow(d2, 2) * d5 -
          8 * pow(a3, 2) * b2 * pow(d2, 2) * d4 -
          4 * pow(a3, 2) * b5 * d1 * pow(d2, 2) -
          pow(a5, 2) * b1 * pow(d0, 2) * d5 -
          4 * pow(a5, 2) * b1 * pow(d2, 2) * d3 -
          2 * pow(a5, 2) * b2 * pow(d0, 2) * d4 +
          4 * pow(a5, 2) * b3 * d1 * pow(d2, 2) -
          2 * pow(a5, 2) * b4 * pow(d0, 2) * d2 +
          3 * pow(a5, 2) * b5 * pow(d0, 2) * d1 +
          8 * pow(a1, 2) * b2 * d4 * pow(d5, 2) +
          8 * pow(a1, 2) * b4 * d2 * pow(d5, 2) +
          12 * pow(a1, 2) * b5 * d1 * pow(d5, 2) -
          12 * pow(a5, 2) * b1 * pow(d1, 2) * d5 -
          8 * pow(a5, 2) * b2 * pow(d1, 2) * d4 -
          8 * pow(a5, 2) * b4 * pow(d1, 2) * d2 +
          4 * pow(a4, 2) * b1 * d3 * pow(d5, 2) -
          4 * pow(a4, 2) * b3 * d1 * pow(d5, 2) +
          4 * pow(a5, 2) * b1 * d3 * pow(d4, 2) -
          4 * pow(a5, 2) * b3 * d1 * pow(d4, 2) +
          2 * pow(a3, 2) * b2 * d4 * pow(d5, 2) +
          2 * pow(a3, 2) * b4 * d2 * pow(d5, 2) +
          pow(a3, 2) * b5 * d1 * pow(d5, 2) -
          pow(a5, 2) * b1 * pow(d3, 2) * d5 -
          2 * pow(a5, 2) * b2 * pow(d3, 2) * d4 -
          2 * pow(a5, 2) * b4 * d2 * pow(d3, 2) +
          3 * pow(a5, 2) * b5 * d1 * pow(d3, 2) +
          2 * a0 * a1 * b0 * d0 * pow(d3, 2) -
          8 * a0 * a1 * b0 * d0 * pow(d4, 2) +
          8 * a0 * a1 * b0 * pow(d1, 2) * d3 +
          8 * a0 * a1 * b3 * d0 * pow(d1, 2) +
          8 * a0 * a3 * b1 * d0 * pow(d1, 2) +
          8 * a1 * a3 * b0 * d0 * pow(d1, 2) +
          2 * a0 * a1 * b0 * d0 * pow(d5, 2) +
          8 * a0 * a1 * b0 * pow(d2, 2) * d3 -
          8 * a0 * a3 * b0 * d1 * pow(d2, 2) -
          8 * a1 * a3 * b1 * pow(d0, 2) * d1 -
          8 * a0 * a1 * b0 * pow(d1, 2) * d5 -
          2 * a0 * a1 * b3 * pow(d0, 2) * d3 -
          8 * a0 * a1 * b5 * d0 * pow(d1, 2) -
          16 * a0 * a2 * b0 * pow(d1, 2) * d4 -
          2 * a0 * a3 * b1 * pow(d0, 2) * d3 -
          2 * a0 * a3 * b3 * pow(d0, 2) * d1 +
          16 * a0 * a4 * b0 * pow(d1, 2) * d2 -
          8 * a0 * a5 * b1 * d0 * pow(d1, 2) -
          2 * a1 * a3 * b0 * pow(d0, 2) * d3 -
          8 * a1 * a5 * b0 * d0 * pow(d1, 2) +
          8 * a0 * a1 * b0 * d3 * pow(d4, 2) -
          8 * a0 * a1 * b0 * pow(d2, 2) * d5 -
          16 * a0 * a1 * b1 * d1 * pow(d5, 2) +
          8 * a0 * a1 * b2 * d2 * pow(d3, 2) -
          16 * a0 * a2 * b0 * pow(d2, 2) * d4 +
          8 * a0 * a2 * b1 * d2 * pow(d3, 2) -
          8 * a0 * a2 * b2 * d1 * pow(d3, 2) -
          16 * a0 * a2 * b4 * d0 * pow(d2, 2) -
          8 * a0 * a3 * b0 * d1 * pow(d4, 2) -
          16 * a0 * a4 * b2 * d0 * pow(d2, 2) +
          8 * a0 * a5 * b0 * d1 * pow(d2, 2) +
          8 * a1 * a2 * b0 * d2 * pow(d3, 2) -
          16 * a1 * a2 * b1 * pow(d0, 2) * d4 -
          8 * a1 * a2 * b2 * d0 * pow(d3, 2) +
          8 * a1 * a2 * b2 * pow(d0, 2) * d3 -
          32 * a1 * a3 * b1 * d1 * pow(d2, 2) +
          16 * a1 * a4 * b1 * pow(d0, 2) * d2 +
          8 * a1 * a5 * b1 * pow(d0, 2) * d1 -
          8 * a2 * a3 * b2 * pow(d0, 2) * d1 -
          16 * a2 * a4 * b0 * d0 * pow(d2, 2) +
          4 * a0 * a1 * b0 * d3 * pow(d5, 2) -
          2 * a0 * a1 * b0 * pow(d3, 2) * d5 -
          4 * a0 * a1 * b3 * d0 * pow(d5, 2) +
          2 * a0 * a1 * b3 * pow(d0, 2) * d5 -
          8 * a0 * a1 * b3 * pow(d2, 2) * d3 +
          8 * a0 * a1 * b4 * pow(d0, 2) * d4 -
          2 * a0 * a1 * b5 * d0 * pow(d3, 2) +
          2 * a0 * a1 * b5 * pow(d0, 2) * d3 -
          4 * a0 * a2 * b0 * pow(d3, 2) * d4 +
          32 * a0 * a2 * b2 * d1 * pow(d4, 2) -
          4 * a0 * a2 * b3 * pow(d0, 2) * d4 +
          4 * a0 * a2 * b4 * d0 * pow(d3, 2) -
          4 * a0 * a2 * b4 * pow(d0, 2) * d3 +
          4 * a0 * a3 * b0 * d1 * pow(d5, 2) -
          4 * a0 * a3 * b1 * d0 * pow(d5, 2) +
          2 * a0 * a3 * b1 * pow(d0, 2) * d5 -
          8 * a0 * a3 * b1 * pow(d2, 2) * d3 -
          4 * a0 * a3 * b2 * pow(d0, 2) * d4 +
          8 * a0 * a3 * b3 * d1 * pow(d2, 2) -
          4 * a0 * a3 * b4 * pow(d0, 2) * d2 +
          2 * a0 * a3 * b5 * pow(d0, 2) * d1 -
          4 * a0 * a4 * b0 * d2 * pow(d3, 2) +
          8 * a0 * a4 * b1 * pow(d0, 2) * d4 +
          4 * a0 * a4 * b2 * d0 * pow(d3, 2) -
          4 * a0 * a4 * b2 * pow(d0, 2) * d3 -
          4 * a0 * a4 * b3 * pow(d0, 2) * d2 +
          8 * a0 * a4 * b4 * pow(d0, 2) * d1 +
          6 * a0 * a5 * b0 * d1 * pow(d3, 2) -
          2 * a0 * a5 * b1 * d0 * pow(d3, 2) +
          2 * a0 * a5 * b1 * pow(d0, 2) * d3 +
          2 * a0 * a5 * b3 * pow(d0, 2) * d1 -
          32 * a1 * a2 * b2 * d0 * pow(d4, 2) +
          32 * a1 * a2 * b2 * pow(d1, 2) * d3 +
          32 * a1 * a2 * b3 * pow(d1, 2) * d2 -
          4 * a1 * a3 * b0 * d0 * pow(d5, 2) +
          2 * a1 * a3 * b0 * pow(d0, 2) * d5 -
          8 * a1 * a3 * b0 * pow(d2, 2) * d3 +
          32 * a1 * a3 * b2 * pow(d1, 2) * d2 +
          8 * a1 * a3 * b3 * d0 * pow(d2, 2) +
          8 * a1 * a4 * b0 * pow(d0, 2) * d4 -
          2 * a1 * a5 * b0 * d0 * pow(d3, 2) +
          2 * a1 * a5 * b0 * pow(d0, 2) * d3 -
          4 * a2 * a3 * b0 * pow(d0, 2) * d4 +
          32 * a2 * a3 * b1 * pow(d1, 2) * d2 +
          4 * a2 * a4 * b0 * d0 * pow(d3, 2) -
          4 * a2 * a4 * b0 * pow(d0, 2) * d3 -
          4 * a3 * a4 * b0 * pow(d0, 2) * d2 +
          2 * a3 * a5 * b0 * pow(d0, 2) * d1 -
          8 * a0 * a1 * b3 * pow(d1, 2) * d5 +
          8 * a0 * a1 * b5 * d0 * pow(d4, 2) -
          8 * a0 * a1 * b5 * pow(d1, 2) * d3 -
          8 * a0 * a2 * b2 * d1 * pow(d5, 2) -
          8 * a0 * a3 * b1 * pow(d1, 2) * d5 -
          16 * a0 * a3 * b4 * pow(d1, 2) * d2 -
          16 * a0 * a4 * b3 * pow(d1, 2) * d2 -
          16 * a0 * a5 * b0 * d1 * pow(d4, 2) +
          8 * a0 * a5 * b1 * d0 * pow(d4, 2) -
          8 * a0 * a5 * b1 * pow(d1, 2) * d3 -
          64 * a1 * a2 * b1 * pow(d2, 2) * d4 +
          8 * a1 * a2 * b2 * d0 * pow(d5, 2) -
          8 * a1 * a2 * b2 * pow(d0, 2) * d5 -
          64 * a1 * a2 * b4 * d1 * pow(d2, 2) -
          8 * a1 * a3 * b0 * pow(d1, 2) * d5 -
          8 * a1 * a3 * b5 * d0 * pow(d1, 2) -
          64 * a1 * a4 * b2 * d1 * pow(d2, 2) +
          8 * a1 * a5 * b0 * d0 * pow(d4, 2) -
          8 * a1 * a5 * b0 * pow(d1, 2) * d3 +
          32 * a1 * a5 * b1 * d1 * pow(d2, 2) -
          8 * a1 * a5 * b3 * d0 * pow(d1, 2) +
          16 * a2 * a3 * b4 * d0 * pow(d1, 2) -
          64 * a2 * a4 * b1 * d1 * pow(d2, 2) +
          16 * a2 * a4 * b2 * pow(d0, 2) * d2 +
          16 * a2 * a4 * b3 * d0 * pow(d1, 2) +
          8 * a2 * a5 * b2 * pow(d0, 2) * d1 -
          16 * a3 * a4 * b0 * pow(d1, 2) * d2 +
          16 * a3 * a4 * b2 * d0 * pow(d1, 2) -
          8 * a3 * a5 * b1 * d0 * pow(d1, 2) +
          8 * a0 * a1 * b3 * pow(d2, 2) * d5 +
          2 * a0 * a1 * b5 * d0 * pow(d5, 2) -
          2 * a0 * a1 * b5 * pow(d0, 2) * d5 +
          4 * a0 * a2 * b0 * d4 * pow(d5, 2) +
          16 * a0 * a2 * b3 * pow(d2, 2) * d4 -
          4 * a0 * a2 * b4 * d0 * pow(d5, 2) +
          4 * a0 * a2 * b4 * pow(d0, 2) * d5 +
          16 * a0 * a2 * b4 * pow(d2, 2) * d3 +
          4 * a0 * a2 * b5 * pow(d0, 2) * d4 +
          8 * a0 * a3 * b1 * pow(d2, 2) * d5 +
          16 * a0 * a3 * b2 * pow(d2, 2) * d4 +
          4 * a0 * a4 * b0 * d2 * pow(d5, 2) -
          4 * a0 * a4 * b2 * d0 * pow(d5, 2) +
          4 * a0 * a4 * b2 * pow(d0, 2) * d5 +
          16 * a0 * a4 * b2 * pow(d2, 2) * d3 +
          32 * a0 * a4 * b4 * d1 * pow(d2, 2) +
          4 * a0 * a4 * b5 * pow(d0, 2) * d2 +
          2 * a0 * a5 * b0 * d1 * pow(d5, 2) +
          2 * a0 * a5 * b1 * d0 * pow(d5, 2) -
          2 * a0 * a5 * b1 * pow(d0, 2) * d5 +
          4 * a0 * a5 * b2 * pow(d0, 2) * d4 +
          4 * a0 * a5 * b4 * pow(d0, 2) * d2 -
          2 * a0 * a5 * b5 * pow(d0, 2) * d1 -
          32 * a1 * a2 * b2 * pow(d1, 2) * d5 -
          32 * a1 * a2 * b5 * pow(d1, 2) * d2 +
          8 * a1 * a3 * b0 * pow(d2, 2) * d5 -
          8 * a1 * a3 * b1 * d1 * pow(d5, 2) -
          8 * a1 * a3 * b5 * d0 * pow(d2, 2) -
          32 * a1 * a4 * b4 * d0 * pow(d2, 2) +
          2 * a1 * a5 * b0 * d0 * pow(d5, 2) -
          2 * a1 * a5 * b0 * pow(d0, 2) * d5 -
          32 * a1 * a5 * b2 * pow(d1, 2) * d2 -
          8 * a1 * a5 * b3 * d0 * pow(d2, 2) +
          16 * a2 * a3 * b0 * pow(d2, 2) * d4 +
          16 * a2 * a3 * b4 * d0 * pow(d2, 2) -
          4 * a2 * a4 * b0 * d0 * pow(d5, 2) +
          4 * a2 * a4 * b0 * pow(d0, 2) * d5 +
          16 * a2 * a4 * b0 * pow(d2, 2) * d3 +
          64 * a2 * a4 * b2 * pow(d1, 2) * d2 +
          16 * a2 * a4 * b3 * d0 * pow(d2, 2) +
          4 * a2 * a5 * b0 * pow(d0, 2) * d4 -
          32 * a2 * a5 * b1 * pow(d1, 2) * d2 +
          16 * a3 * a4 * b2 * d0 * pow(d2, 2) -
          8 * a3 * a5 * b1 * d0 * pow(d2, 2) +
          4 * a4 * a5 * b0 * pow(d0, 2) * d2 -
          2 * a0 * a1 * b3 * d3 * pow(d5, 2) +
          16 * a0 * a1 * b5 * pow(d1, 2) * d5 +
          16 * a0 * a2 * b5 * pow(d1, 2) * d4 -
          2 * a0 * a3 * b1 * d3 * pow(d5, 2) -
          2 * a0 * a3 * b3 * d1 * pow(d5, 2) +
          16 * a0 * a5 * b1 * pow(d1, 2) * d5 +
          16 * a0 * a5 * b2 * pow(d1, 2) * d4 +
          32 * a1 * a2 * b2 * d3 * pow(d4, 2) -
          2 * a1 * a3 * b0 * d3 * pow(d5, 2) +
          6 * a1 * a3 * b3 * d0 * pow(d5, 2) -
          6 * a1 * a3 * b3 * pow(d0, 2) * d5 +
          2 * a1 * a3 * b5 * pow(d0, 2) * d3 +
          8 * a1 * a4 * b4 * pow(d0, 2) * d3 +
          16 * a1 * a5 * b0 * pow(d1, 2) * d5 +
          2 * a1 * a5 * b3 * pow(d0, 2) * d3 +
          16 * a1 * a5 * b5 * d0 * pow(d1, 2) -
          32 * a2 * a3 * b2 * d1 * pow(d4, 2) +
          4 * a2 * a3 * b3 * pow(d0, 2) * d4 -
          4 * a2 * a3 * b4 * pow(d0, 2) * d3 -
          4 * a2 * a4 * b3 * pow(d0, 2) * d3 -
          16 * a2 * a4 * b5 * d0 * pow(d1, 2) +
          16 * a2 * a5 * b0 * pow(d1, 2) * d4 -
          16 * a2 * a5 * b4 * d0 * pow(d1, 2) -
          4 * a3 * a4 * b2 * pow(d0, 2) * d3 +
          4 * a3 * a4 * b3 * pow(d0, 2) * d2 -
          8 * a3 * a4 * b4 * pow(d0, 2) * d1 +
          2 * a3 * a5 * b1 * pow(d0, 2) * d3 +
          2 * a3 * a5 * b3 * pow(d0, 2) * d1 -
          16 * a4 * a5 * b2 * d0 * pow(d1, 2) -
          8 * a0 * a1 * b5 * d3 * pow(d4, 2) +
          8 * a0 * a3 * b5 * d1 * pow(d4, 2) -
          8 * a0 * a5 * b1 * d3 * pow(d4, 2) +
          8 * a0 * a5 * b3 * d1 * pow(d4, 2) -
          8 * a0 * a5 * b5 * d1 * pow(d2, 2) +
          16 * a1 * a2 * b1 * d4 * pow(d5, 2) -
          8 * a1 * a2 * b2 * d3 * pow(d5, 2) +
          8 * a1 * a2 * b2 * pow(d3, 2) * d5 -
          16 * a1 * a2 * b4 * d1 * pow(d5, 2) -
          8 * a1 * a2 * b5 * d2 * pow(d3, 2) +
          16 * a1 * a4 * b1 * d2 * pow(d5, 2) -
          16 * a1 * a4 * b2 * d1 * pow(d5, 2) -
          8 * a1 * a5 * b0 * d3 * pow(d4, 2) +
          24 * a1 * a5 * b1 * d1 * pow(d5, 2) -
          8 * a1 * a5 * b2 * d2 * pow(d3, 2) +
          8 * a1 * a5 * b5 * d0 * pow(d2, 2) +
          8 * a2 * a3 * b2 * d1 * pow(d5, 2) -
          16 * a2 * a4 * b1 * d1 * pow(d5, 2) +
          16 * a2 * a4 * b2 * d2 * pow(d3, 2) -
          8 * a2 * a5 * b1 * d2 * pow(d3, 2) +
          8 * a2 * a5 * b2 * d1 * pow(d3, 2) +
          8 * a3 * a5 * b0 * d1 * pow(d4, 2) -
          2 * a0 * a1 * b5 * d3 * pow(d5, 2) +
          2 * a0 * a1 * b5 * pow(d3, 2) * d5 -
          4 * a0 * a2 * b3 * d4 * pow(d5, 2) +
          4 * a0 * a2 * b4 * d3 * pow(d5, 2) -
          4 * a0 * a2 * b4 * pow(d3, 2) * d5 +
          4 * a0 * a2 * b5 * pow(d3, 2) * d4 -
          4 * a0 * a3 * b2 * d4 * pow(d5, 2) -
          4 * a0 * a3 * b4 * d2 * pow(d5, 2) -
          2 * a0 * a3 * b5 * d1 * pow(d5, 2) +
          4 * a0 * a4 * b2 * d3 * pow(d5, 2) -
          4 * a0 * a4 * b2 * pow(d3, 2) * d5 -
          4 * a0 * a4 * b3 * d2 * pow(d5, 2) +
          8 * a0 * a4 * b4 * d1 * pow(d5, 2) +
          4 * a0 * a4 * b5 * d2 * pow(d3, 2) -
          2 * a0 * a5 * b1 * d3 * pow(d5, 2) +
          2 * a0 * a5 * b1 * pow(d3, 2) * d5 +
          4 * a0 * a5 * b2 * pow(d3, 2) * d4 -
          2 * a0 * a5 * b3 * d1 * pow(d5, 2) +
          4 * a0 * a5 * b4 * d2 * pow(d3, 2) -
          6 * a0 * a5 * b5 * d1 * pow(d3, 2) -
          8 * a1 * a3 * b3 * pow(d2, 2) * d5 -
          2 * a1 * a3 * b5 * d0 * pow(d5, 2) +
          4 * a1 * a3 * b5 * pow(d0, 2) * d5 +
          8 * a1 * a3 * b5 * pow(d2, 2) * d3 -
          8 * a1 * a4 * b4 * d0 * pow(d5, 2) +
          16 * a1 * a4 * b4 * pow(d0, 2) * d5 +
          32 * a1 * a4 * b4 * pow(d2, 2) * d3 -
          8 * a1 * a4 * b5 * pow(d0, 2) * d4 -
          2 * a1 * a5 * b0 * d3 * pow(d5, 2) +
          2 * a1 * a5 * b0 * pow(d3, 2) * d5 -
          2 * a1 * a5 * b3 * d0 * pow(d5, 2) +
          4 * a1 * a5 * b3 * pow(d0, 2) * d5 +
          8 * a1 * a5 * b3 * pow(d2, 2) * d3 -
          8 * a1 * a5 * b4 * pow(d0, 2) * d4 +
          2 * a1 * a5 * b5 * d0 * pow(d3, 2) -
          4 * a1 * a5 * b5 * pow(d0, 2) * d3 -
          4 * a2 * a3 * b0 * d4 * pow(d5, 2) -
          16 * a2 * a3 * b3 * pow(d2, 2) * d4 +
          4 * a2 * a3 * b4 * d0 * pow(d5, 2) -
          8 * a2 * a3 * b4 * pow(d0, 2) * d5 -
          16 * a2 * a3 * b4 * pow(d2, 2) * d3 +
          4 * a2 * a4 * b0 * d3 * pow(d5, 2) -
          4 * a2 * a4 * b0 * pow(d3, 2) * d5 +
          4 * a2 * a4 * b3 * d0 * pow(d5, 2) -
          8 * a2 * a4 * b3 * pow(d0, 2) * d5 -
          16 * a2 * a4 * b3 * pow(d2, 2) * d3 -
          4 * a2 * a4 * b5 * d0 * pow(d3, 2) +
          8 * a2 * a4 * b5 * pow(d0, 2) * d3 +
          4 * a2 * a5 * b0 * pow(d3, 2) * d4 -
          4 * a2 * a5 * b4 * d0 * pow(d3, 2) +
          8 * a2 * a5 * b4 * pow(d0, 2) * d3 -
          4 * a3 * a4 * b0 * d2 * pow(d5, 2) +
          4 * a3 * a4 * b2 * d0 * pow(d5, 2) -
          8 * a3 * a4 * b2 * pow(d0, 2) * d5 -
          16 * a3 * a4 * b2 * pow(d2, 2) * d3 -
          32 * a3 * a4 * b4 * d1 * pow(d2, 2) -
          2 * a3 * a5 * b0 * d1 * pow(d5, 2) -
          2 * a3 * a5 * b1 * d0 * pow(d5, 2) +
          4 * a3 * a5 * b1 * pow(d0, 2) * d5 +
          8 * a3 * a5 * b1 * pow(d2, 2) * d3 -
          8 * a3 * a5 * b3 * d1 * pow(d2, 2) -
          4 * a3 * a5 * b5 * pow(d0, 2) * d1 +
          4 * a4 * a5 * b0 * d2 * pow(d3, 2) -
          8 * a4 * a5 * b1 * pow(d0, 2) * d4 -
          4 * a4 * a5 * b2 * d0 * pow(d3, 2) +
          8 * a4 * a5 * b2 * pow(d0, 2) * d3 +
          8 * a0 * a5 * b5 * d1 * pow(d4, 2) +
          8 * a1 * a3 * b5 * pow(d1, 2) * d5 +
          8 * a1 * a5 * b3 * pow(d1, 2) * d5 -
          8 * a1 * a5 * b5 * d0 * pow(d4, 2) +
          8 * a1 * a5 * b5 * pow(d1, 2) * d3 -
          16 * a2 * a3 * b4 * pow(d1, 2) * d5 -
          16 * a2 * a4 * b3 * pow(d1, 2) * d5 -
          16 * a3 * a4 * b2 * pow(d1, 2) * d5 +
          16 * a3 * a4 * b5 * pow(d1, 2) * d2 +
          8 * a3 * a5 * b1 * pow(d1, 2) * d5 +
          16 * a3 * a5 * b4 * pow(d1, 2) * d2 +
          16 * a4 * a5 * b3 * pow(d1, 2) * d2 -
          2 * a1 * a5 * b5 * pow(d0, 2) * d5 -
          8 * a1 * a5 * b5 * pow(d2, 2) * d3 +
          4 * a2 * a4 * b5 * pow(d0, 2) * d5 +
          4 * a2 * a5 * b4 * pow(d0, 2) * d5 -
          4 * a2 * a5 * b5 * pow(d0, 2) * d4 +
          8 * a3 * a5 * b5 * d1 * pow(d2, 2) +
          4 * a4 * a5 * b2 * pow(d0, 2) * d5 -
          4 * a4 * a5 * b5 * pow(d0, 2) * d2 +
          2 * a1 * a3 * b5 * d3 * pow(d5, 2) +
          8 * a1 * a4 * b4 * d3 * pow(d5, 2) +
          2 * a1 * a5 * b3 * d3 * pow(d5, 2) -
          24 * a1 * a5 * b5 * pow(d1, 2) * d5 +
          4 * a2 * a3 * b3 * d4 * pow(d5, 2) -
          4 * a2 * a3 * b4 * d3 * pow(d5, 2) -
          4 * a2 * a4 * b3 * d3 * pow(d5, 2) +
          16 * a2 * a4 * b5 * pow(d1, 2) * d5 +
          16 * a2 * a5 * b4 * pow(d1, 2) * d5 -
          16 * a2 * a5 * b5 * pow(d1, 2) * d4 -
          4 * a3 * a4 * b2 * d3 * pow(d5, 2) +
          4 * a3 * a4 * b3 * d2 * pow(d5, 2) -
          8 * a3 * a4 * b4 * d1 * pow(d5, 2) +
          2 * a3 * a5 * b1 * d3 * pow(d5, 2) +
          2 * a3 * a5 * b3 * d1 * pow(d5, 2) +
          16 * a4 * a5 * b2 * pow(d1, 2) * d5 -
          16 * a4 * a5 * b5 * pow(d1, 2) * d2 +
          8 * a1 * a5 * b5 * d3 * pow(d4, 2) -
          8 * a3 * a5 * b5 * d1 * pow(d4, 2) -
          2 * a1 * a5 * b5 * pow(d3, 2) * d5 +
          4 * a2 * a4 * b5 * pow(d3, 2) * d5 +
          4 * a2 * a5 * b4 * pow(d3, 2) * d5 -
          4 * a2 * a5 * b5 * pow(d3, 2) * d4 +
          4 * a4 * a5 * b2 * pow(d3, 2) * d5 -
          4 * a4 * a5 * b5 * d2 * pow(d3, 2) -
          8 * pow(a1, 2) * b0 * d0 * d1 * d3 +
          2 * pow(a0, 2) * b3 * d0 * d1 * d3 +
          8 * pow(a1, 2) * b0 * d0 * d1 * d5 +
          16 * pow(a2, 2) * b0 * d0 * d2 * d4 -
          8 * pow(a3, 2) * b2 * d0 * d1 * d2 +
          6 * pow(a0, 2) * b0 * d1 * d3 * d5 -
          12 * pow(a0, 2) * b0 * d2 * d3 * d4 -
          2 * pow(a0, 2) * b1 * d0 * d3 * d5 +
          4 * pow(a0, 2) * b2 * d0 * d3 * d4 -
          2 * pow(a0, 2) * b3 * d0 * d1 * d5 +
          4 * pow(a0, 2) * b3 * d0 * d2 * d4 -
          8 * pow(a0, 2) * b4 * d0 * d1 * d4 +
          4 * pow(a0, 2) * b4 * d0 * d2 * d3 -
          2 * pow(a0, 2) * b5 * d0 * d1 * d3 -
          32 * pow(a1, 2) * b2 * d1 * d2 * d3 +
          8 * pow(a2, 2) * b3 * d0 * d1 * d3 +
          2 * pow(a3, 2) * b0 * d0 * d1 * d5 -
          4 * pow(a3, 2) * b0 * d0 * d2 * d4 +
          4 * pow(a5, 2) * b0 * d0 * d1 * d3 +
          8 * pow(a1, 2) * b0 * d1 * d3 * d5 -
          16 * pow(a1, 2) * b0 * d2 * d3 * d4 -
          24 * pow(a1, 2) * b1 * d0 * d3 * d5 +
          16 * pow(a1, 2) * b2 * d0 * d3 * d4 +
          8 * pow(a1, 2) * b3 * d0 * d1 * d5 +
          8 * pow(a1, 2) * b5 * d0 * d1 * d3 +
          64 * pow(a2, 2) * b1 * d1 * d2 * d4 -
          8 * pow(a4, 2) * b0 * d0 * d1 * d5 +
          12 * pow(a0, 2) * b0 * d2 * d4 * d5 -
          4 * pow(a0, 2) * b2 * d0 * d4 * d5 -
          4 * pow(a0, 2) * b4 * d0 * d2 * d5 +
          2 * pow(a0, 2) * b5 * d0 * d1 * d5 -
          4 * pow(a0, 2) * b5 * d0 * d2 * d4 +
          32 * pow(a1, 2) * b2 * d1 * d2 * d5 +
          8 * pow(a2, 2) * b0 * d1 * d3 * d5 -
          16 * pow(a2, 2) * b0 * d2 * d3 * d4 +
          48 * pow(a2, 2) * b2 * d0 * d3 * d4 -
          16 * pow(a2, 2) * b3 * d0 * d2 * d4 -
          16 * pow(a2, 2) * b4 * d0 * d2 * d3 -
          8 * pow(a2, 2) * b5 * d0 * d1 * d3 -
          2 * pow(a5, 2) * b0 * d0 * d1 * d5 +
          4 * pow(a5, 2) * b0 * d0 * d2 * d4 -
          2 * pow(a0, 2) * b3 * d1 * d3 * d5 +
          4 * pow(a0, 2) * b3 * d2 * d3 * d4 +
          16 * pow(a1, 2) * b0 * d2 * d4 * d5 -
          16 * pow(a1, 2) * b4 * d0 * d2 * d5 -
          16 * pow(a1, 2) * b5 * d0 * d1 * d5 +
          2 * pow(a5, 2) * b3 * d0 * d1 * d3 +
          8 * pow(a3, 2) * b2 * d1 * d2 * d5 -
          8 * pow(a4, 2) * b1 * d0 * d3 * d5 +
          8 * pow(a4, 2) * b3 * d0 * d1 * d5 +
          16 * pow(a5, 2) * b1 * d1 * d2 * d4 -
          8 * pow(a0, 2) * b3 * d2 * d4 * d5 +
          8 * pow(a0, 2) * b4 * d1 * d4 * d5 -
          4 * pow(a0, 2) * b5 * d1 * d3 * d5 +
          8 * pow(a0, 2) * b5 * d2 * d3 * d4 -
          8 * pow(a2, 2) * b3 * d1 * d3 * d5 +
          16 * pow(a2, 2) * b3 * d2 * d3 * d4 +
          4 * pow(a3, 2) * b0 * d2 * d4 * d5 -
          4 * pow(a3, 2) * b2 * d0 * d4 * d5 -
          4 * pow(a3, 2) * b4 * d0 * d2 * d5 -
          2 * pow(a3, 2) * b5 * d0 * d1 * d5 +
          4 * pow(a3, 2) * b5 * d0 * d2 * d4 +
          2 * pow(a5, 2) * b0 * d1 * d3 * d5 -
          4 * pow(a5, 2) * b0 * d2 * d3 * d4 +
          2 * pow(a5, 2) * b1 * d0 * d3 * d5 +
          4 * pow(a5, 2) * b2 * d0 * d3 * d4 +
          2 * pow(a5, 2) * b3 * d0 * d1 * d5 -
          4 * pow(a5, 2) * b3 * d0 * d2 * d4 +
          4 * pow(a5, 2) * b4 * d0 * d2 * d3 -
          6 * pow(a5, 2) * b5 * d0 * d1 * d3 -
          16 * pow(a1, 2) * b2 * d3 * d4 * d5 -
          8 * pow(a1, 2) * b5 * d1 * d3 * d5 +
          16 * pow(a1, 2) * b5 * d2 * d3 * d4 -
          4 * pow(a0, 2) * b5 * d2 * d4 * d5 -
          16 * pow(a1, 2) * b5 * d2 * d4 * d5 -
          2 * pow(a5, 2) * b3 * d1 * d3 * d5 +
          4 * pow(a5, 2) * b3 * d2 * d3 * d4 -
          4 * pow(a3, 2) * b5 * d2 * d4 * d5 -
          16 * a0 * a1 * b1 * d0 * d1 * d3 + 4 * a0 * a3 * b0 * d0 * d1 * d3 +
          16 * a0 * a1 * b1 * d0 * d1 * d5 + 16 * a0 * a1 * b2 * d0 * d1 * d4 -
          8 * a0 * a1 * b2 * d0 * d2 * d3 - 16 * a0 * a1 * b4 * d0 * d1 * d2 +
          16 * a0 * a2 * b1 * d0 * d1 * d4 - 8 * a0 * a2 * b1 * d0 * d2 * d3 +
          8 * a0 * a2 * b3 * d0 * d1 * d2 + 8 * a0 * a3 * b2 * d0 * d1 * d2 -
          16 * a0 * a4 * b1 * d0 * d1 * d2 + 16 * a1 * a2 * b0 * d0 * d1 * d4 -
          8 * a1 * a2 * b0 * d0 * d2 * d3 - 16 * a1 * a4 * b0 * d0 * d1 * d2 +
          8 * a2 * a3 * b0 * d0 * d1 * d2 - 4 * a0 * a1 * b0 * d0 * d3 * d5 +
          8 * a0 * a2 * b0 * d0 * d3 * d4 - 4 * a0 * a3 * b0 * d0 * d1 * d5 +
          8 * a0 * a3 * b0 * d0 * d2 * d4 - 16 * a0 * a4 * b0 * d0 * d1 * d4 +
          8 * a0 * a4 * b0 * d0 * d2 * d3 - 4 * a0 * a5 * b0 * d0 * d1 * d3 +
          8 * a0 * a1 * b2 * d0 * d2 * d5 + 8 * a0 * a2 * b1 * d0 * d2 * d5 +
          32 * a0 * a2 * b2 * d0 * d2 * d4 - 8 * a0 * a2 * b5 * d0 * d1 * d2 -
          8 * a0 * a5 * b2 * d0 * d1 * d2 + 8 * a1 * a2 * b0 * d0 * d2 * d5 -
          64 * a1 * a2 * b1 * d1 * d2 * d3 - 8 * a2 * a5 * b0 * d0 * d1 * d2 +
          16 * a0 * a1 * b1 * d1 * d3 * d5 - 32 * a0 * a1 * b1 * d2 * d3 * d4 +
          16 * a0 * a1 * b3 * d1 * d2 * d4 + 16 * a0 * a1 * b4 * d1 * d2 * d3 -
          8 * a0 * a2 * b0 * d0 * d4 * d5 + 16 * a0 * a3 * b1 * d1 * d2 * d4 -
          8 * a0 * a4 * b0 * d0 * d2 * d5 + 16 * a0 * a4 * b1 * d1 * d2 * d3 +
          4 * a0 * a5 * b0 * d0 * d1 * d5 - 8 * a0 * a5 * b0 * d0 * d2 * d4 +
          32 * a1 * a2 * b1 * d0 * d3 * d4 - 16 * a1 * a2 * b3 * d0 * d1 * d4 -
          16 * a1 * a2 * b4 * d0 * d1 * d3 + 16 * a1 * a3 * b0 * d1 * d2 * d4 +
          16 * a1 * a3 * b1 * d0 * d1 * d5 - 16 * a1 * a3 * b2 * d0 * d1 * d4 +
          16 * a1 * a4 * b0 * d1 * d2 * d3 - 16 * a1 * a4 * b2 * d0 * d1 * d3 +
          16 * a1 * a5 * b1 * d0 * d1 * d3 - 16 * a2 * a3 * b1 * d0 * d1 * d4 +
          16 * a2 * a3 * b2 * d0 * d1 * d3 - 16 * a2 * a3 * b3 * d0 * d1 * d2 -
          16 * a2 * a4 * b1 * d0 * d1 * d3 + 4 * a0 * a1 * b3 * d0 * d3 * d5 -
          8 * a0 * a1 * b4 * d0 * d3 * d4 - 4 * a0 * a3 * b0 * d1 * d3 * d5 +
          8 * a0 * a3 * b0 * d2 * d3 * d4 + 4 * a0 * a3 * b1 * d0 * d3 * d5 +
          4 * a0 * a3 * b3 * d0 * d1 * d5 - 8 * a0 * a3 * b3 * d0 * d2 * d4 +
          8 * a0 * a3 * b4 * d0 * d1 * d4 - 4 * a0 * a3 * b5 * d0 * d1 * d3 -
          8 * a0 * a4 * b1 * d0 * d3 * d4 + 8 * a0 * a4 * b3 * d0 * d1 * d4 -
          4 * a0 * a5 * b3 * d0 * d1 * d3 + 64 * a1 * a2 * b1 * d1 * d2 * d5 +
          128 * a1 * a2 * b2 * d1 * d2 * d4 + 4 * a1 * a3 * b0 * d0 * d3 * d5 -
          8 * a1 * a4 * b0 * d0 * d3 * d4 + 8 * a3 * a4 * b0 * d0 * d1 * d4 -
          4 * a3 * a5 * b0 * d0 * d1 * d3 + 32 * a0 * a1 * b1 * d2 * d4 * d5 -
          16 * a0 * a1 * b2 * d1 * d4 * d5 - 8 * a0 * a1 * b2 * d2 * d3 * d5 -
          16 * a0 * a1 * b5 * d1 * d2 * d4 - 16 * a0 * a2 * b1 * d1 * d4 * d5 -
          8 * a0 * a2 * b1 * d2 * d3 * d5 + 16 * a0 * a2 * b2 * d1 * d3 * d5 -
          32 * a0 * a2 * b2 * d2 * d3 * d4 - 8 * a0 * a2 * b3 * d1 * d2 * d5 -
          32 * a0 * a2 * b4 * d1 * d2 * d4 - 8 * a0 * a3 * b2 * d1 * d2 * d5 -
          32 * a0 * a4 * b2 * d1 * d2 * d4 - 16 * a0 * a5 * b1 * d1 * d2 * d4 -
          16 * a1 * a2 * b0 * d1 * d4 * d5 - 8 * a1 * a2 * b0 * d2 * d3 * d5 +
          16 * a1 * a2 * b4 * d0 * d1 * d5 + 32 * a1 * a2 * b4 * d0 * d2 * d4 +
          8 * a1 * a2 * b5 * d0 * d2 * d3 - 32 * a1 * a4 * b1 * d0 * d2 * d5 +
          16 * a1 * a4 * b2 * d0 * d1 * d5 + 32 * a1 * a4 * b2 * d0 * d2 * d4 +
          16 * a1 * a4 * b5 * d0 * d1 * d2 - 16 * a1 * a5 * b0 * d1 * d2 * d4 -
          32 * a1 * a5 * b1 * d0 * d1 * d5 + 8 * a1 * a5 * b2 * d0 * d2 * d3 +
          16 * a1 * a5 * b4 * d0 * d1 * d2 - 8 * a2 * a3 * b0 * d1 * d2 * d5 -
          32 * a2 * a3 * b2 * d0 * d2 * d4 + 8 * a2 * a3 * b5 * d0 * d1 * d2 -
          32 * a2 * a4 * b0 * d1 * d2 * d4 + 16 * a2 * a4 * b1 * d0 * d1 * d5 +
          32 * a2 * a4 * b1 * d0 * d2 * d4 - 32 * a2 * a4 * b2 * d0 * d2 * d3 +
          8 * a2 * a5 * b1 * d0 * d2 * d3 - 16 * a2 * a5 * b2 * d0 * d1 * d3 +
          8 * a2 * a5 * b3 * d0 * d1 * d2 + 8 * a3 * a5 * b2 * d0 * d1 * d2 +
          16 * a4 * a5 * b1 * d0 * d1 * d2 - 8 * a0 * a1 * b4 * d0 * d4 * d5 +
          8 * a0 * a2 * b3 * d0 * d4 * d5 - 8 * a0 * a2 * b5 * d0 * d3 * d4 -
          16 * a0 * a3 * b0 * d2 * d4 * d5 + 8 * a0 * a3 * b2 * d0 * d4 * d5 +
          8 * a0 * a3 * b4 * d0 * d2 * d5 + 16 * a0 * a4 * b0 * d1 * d4 * d5 -
          8 * a0 * a4 * b1 * d0 * d4 * d5 + 8 * a0 * a4 * b3 * d0 * d2 * d5 -
          16 * a0 * a4 * b4 * d0 * d1 * d5 + 8 * a0 * a4 * b5 * d0 * d1 * d4 -
          8 * a0 * a4 * b5 * d0 * d2 * d3 - 8 * a0 * a5 * b0 * d1 * d3 * d5 +
          16 * a0 * a5 * b0 * d2 * d3 * d4 - 8 * a0 * a5 * b2 * d0 * d3 * d4 +
          8 * a0 * a5 * b4 * d0 * d1 * d4 - 8 * a0 * a5 * b4 * d0 * d2 * d3 +
          8 * a0 * a5 * b5 * d0 * d1 * d3 - 8 * a1 * a4 * b0 * d0 * d4 * d5 +
          8 * a2 * a3 * b0 * d0 * d4 * d5 - 8 * a2 * a5 * b0 * d0 * d3 * d4 +
          8 * a3 * a4 * b0 * d0 * d2 * d5 + 8 * a4 * a5 * b0 * d0 * d1 * d4 -
          8 * a4 * a5 * b0 * d0 * d2 * d3 + 8 * a0 * a2 * b5 * d1 * d2 * d5 +
          8 * a0 * a5 * b2 * d1 * d2 * d5 - 8 * a1 * a2 * b5 * d0 * d2 * d5 -
          8 * a1 * a5 * b2 * d0 * d2 * d5 + 8 * a2 * a5 * b0 * d1 * d2 * d5 -
          8 * a2 * a5 * b1 * d0 * d2 * d5 - 8 * a0 * a5 * b0 * d2 * d4 * d5 -
          4 * a0 * a5 * b5 * d0 * d1 * d5 + 8 * a0 * a5 * b5 * d0 * d2 * d4 -
          32 * a1 * a2 * b1 * d3 * d4 * d5 + 16 * a1 * a2 * b3 * d1 * d4 * d5 +
          16 * a1 * a2 * b4 * d1 * d3 * d5 - 32 * a1 * a2 * b4 * d2 * d3 * d4 +
          16 * a1 * a3 * b2 * d1 * d4 * d5 - 16 * a1 * a3 * b5 * d1 * d2 * d4 +
          16 * a1 * a4 * b2 * d1 * d3 * d5 - 32 * a1 * a4 * b2 * d2 * d3 * d4 -
          16 * a1 * a4 * b5 * d1 * d2 * d3 - 16 * a1 * a5 * b1 * d1 * d3 * d5 +
          32 * a1 * a5 * b1 * d2 * d3 * d4 - 16 * a1 * a5 * b3 * d1 * d2 * d4 -
          16 * a1 * a5 * b4 * d1 * d2 * d3 + 16 * a2 * a3 * b1 * d1 * d4 * d5 -
          16 * a2 * a3 * b2 * d1 * d3 * d5 + 32 * a2 * a3 * b2 * d2 * d3 * d4 +
          16 * a2 * a3 * b3 * d1 * d2 * d5 + 32 * a2 * a3 * b4 * d1 * d2 * d4 +
          16 * a2 * a4 * b1 * d1 * d3 * d5 - 32 * a2 * a4 * b1 * d2 * d3 * d4 +
          32 * a2 * a4 * b3 * d1 * d2 * d4 + 32 * a3 * a4 * b2 * d1 * d2 * d4 -
          16 * a3 * a5 * b1 * d1 * d2 * d4 - 16 * a4 * a5 * b1 * d1 * d2 * d3 +
          8 * a0 * a1 * b4 * d3 * d4 * d5 + 8 * a0 * a3 * b3 * d2 * d4 * d5 -
          8 * a0 * a3 * b4 * d1 * d4 * d5 + 4 * a0 * a3 * b5 * d1 * d3 * d5 -
          8 * a0 * a3 * b5 * d2 * d3 * d4 + 8 * a0 * a4 * b1 * d3 * d4 * d5 -
          8 * a0 * a4 * b3 * d1 * d4 * d5 + 4 * a0 * a5 * b3 * d1 * d3 * d5 -
          8 * a0 * a5 * b3 * d2 * d3 * d4 - 4 * a1 * a3 * b5 * d0 * d3 * d5 +
          8 * a1 * a4 * b0 * d3 * d4 * d5 - 16 * a1 * a4 * b4 * d0 * d3 * d5 +
          8 * a1 * a4 * b5 * d0 * d3 * d4 - 4 * a1 * a5 * b3 * d0 * d3 * d5 +
          8 * a1 * a5 * b4 * d0 * d3 * d4 - 8 * a2 * a3 * b3 * d0 * d4 * d5 +
          8 * a2 * a3 * b4 * d0 * d3 * d5 + 8 * a2 * a4 * b3 * d0 * d3 * d5 -
          8 * a3 * a4 * b0 * d1 * d4 * d5 + 8 * a3 * a4 * b2 * d0 * d3 * d5 -
          8 * a3 * a4 * b3 * d0 * d2 * d5 + 16 * a3 * a4 * b4 * d0 * d1 * d5 -
          8 * a3 * a4 * b5 * d0 * d1 * d4 + 4 * a3 * a5 * b0 * d1 * d3 * d5 -
          8 * a3 * a5 * b0 * d2 * d3 * d4 - 4 * a3 * a5 * b1 * d0 * d3 * d5 -
          4 * a3 * a5 * b3 * d0 * d1 * d5 + 8 * a3 * a5 * b3 * d0 * d2 * d4 -
          8 * a3 * a5 * b4 * d0 * d1 * d4 + 4 * a3 * a5 * b5 * d0 * d1 * d3 +
          8 * a4 * a5 * b1 * d0 * d3 * d4 - 8 * a4 * a5 * b3 * d0 * d1 * d4 +
          8 * a1 * a2 * b5 * d2 * d3 * d5 - 32 * a1 * a5 * b1 * d2 * d4 * d5 +
          8 * a1 * a5 * b2 * d2 * d3 * d5 + 32 * a1 * a5 * b5 * d1 * d2 * d4 -
          8 * a2 * a3 * b5 * d1 * d2 * d5 + 8 * a2 * a5 * b1 * d2 * d3 * d5 -
          8 * a2 * a5 * b3 * d1 * d2 * d5 - 8 * a3 * a5 * b2 * d1 * d2 * d5 +
          8 * a0 * a3 * b5 * d2 * d4 * d5 - 8 * a0 * a4 * b5 * d1 * d4 * d5 +
          8 * a0 * a5 * b3 * d2 * d4 * d5 - 8 * a0 * a5 * b4 * d1 * d4 * d5 +
          4 * a0 * a5 * b5 * d1 * d3 * d5 - 8 * a0 * a5 * b5 * d2 * d3 * d4 +
          8 * a1 * a4 * b5 * d0 * d4 * d5 + 8 * a1 * a5 * b4 * d0 * d4 * d5 +
          4 * a1 * a5 * b5 * d0 * d3 * d5 - 8 * a2 * a4 * b5 * d0 * d3 * d5 -
          8 * a2 * a5 * b4 * d0 * d3 * d5 + 8 * a2 * a5 * b5 * d0 * d3 * d4 +
          8 * a3 * a5 * b0 * d2 * d4 * d5 + 4 * a3 * a5 * b5 * d0 * d1 * d5 -
          8 * a3 * a5 * b5 * d0 * d2 * d4 - 8 * a4 * a5 * b0 * d1 * d4 * d5 +
          8 * a4 * a5 * b1 * d0 * d4 * d5 - 8 * a4 * a5 * b2 * d0 * d3 * d5 +
          8 * a4 * a5 * b5 * d0 * d2 * d3 - 8 * a1 * a4 * b5 * d3 * d4 * d5 -
          8 * a1 * a5 * b4 * d3 * d4 * d5 + 8 * a3 * a4 * b5 * d1 * d4 * d5 -
          8 * a3 * a5 * b3 * d2 * d4 * d5 + 8 * a3 * a5 * b4 * d1 * d4 * d5 -
          4 * a3 * a5 * b5 * d1 * d3 * d5 + 8 * a3 * a5 * b5 * d2 * d3 * d4 -
          8 * a4 * a5 * b1 * d3 * d4 * d5 + 8 * a4 * a5 * b3 * d1 * d4 * d5,
      pow(a0, 2) * b1 * c0 * pow(d3, 2) -
          3 * pow(a0, 2) * b0 * c1 * pow(d3, 2) -
          pow(a3, 2) * b0 * c1 * pow(d0, 2) +
          3 * pow(a3, 2) * b1 * c0 * pow(d0, 2) +
          12 * pow(a0, 2) * b0 * c1 * pow(d4, 2) -
          4 * pow(a0, 2) * b1 * c0 * pow(d4, 2) +
          4 * pow(a0, 2) * b1 * c3 * pow(d1, 2) -
          12 * pow(a0, 2) * b3 * c1 * pow(d1, 2) +
          12 * pow(a1, 2) * b1 * c3 * pow(d0, 2) -
          4 * pow(a1, 2) * b3 * c1 * pow(d0, 2) +
          4 * pow(a4, 2) * b0 * c1 * pow(d0, 2) -
          12 * pow(a4, 2) * b1 * c0 * pow(d0, 2) -
          3 * pow(a0, 2) * b0 * c1 * pow(d5, 2) +
          pow(a0, 2) * b1 * c0 * pow(d5, 2) +
          4 * pow(a0, 2) * b1 * c3 * pow(d2, 2) -
          4 * pow(a0, 2) * b3 * c1 * pow(d2, 2) -
          4 * pow(a2, 2) * b0 * c1 * pow(d3, 2) -
          4 * pow(a2, 2) * b1 * c0 * pow(d3, 2) +
          4 * pow(a2, 2) * b1 * c3 * pow(d0, 2) -
          4 * pow(a2, 2) * b3 * c1 * pow(d0, 2) +
          4 * pow(a3, 2) * b0 * c1 * pow(d2, 2) +
          4 * pow(a3, 2) * b1 * c0 * pow(d2, 2) -
          pow(a5, 2) * b0 * c1 * pow(d0, 2) +
          3 * pow(a5, 2) * b1 * c0 * pow(d0, 2) -
          4 * pow(a0, 2) * b1 * c5 * pow(d1, 2) -
          8 * pow(a0, 2) * b2 * c4 * pow(d1, 2) +
          8 * pow(a0, 2) * b4 * c2 * pow(d1, 2) +
          12 * pow(a0, 2) * b5 * c1 * pow(d1, 2) -
          8 * pow(a1, 2) * b0 * c1 * pow(d5, 2) +
          24 * pow(a1, 2) * b1 * c0 * pow(d5, 2) +
          48 * pow(a1, 2) * b1 * c3 * pow(d2, 2) -
          12 * pow(a1, 2) * b1 * c5 * pow(d0, 2) -
          8 * pow(a1, 2) * b2 * c4 * pow(d0, 2) -
          16 * pow(a1, 2) * b3 * c1 * pow(d2, 2) +
          8 * pow(a1, 2) * b4 * c2 * pow(d0, 2) +
          4 * pow(a1, 2) * b5 * c1 * pow(d0, 2) +
          16 * pow(a2, 2) * b0 * c1 * pow(d4, 2) -
          16 * pow(a2, 2) * b1 * c0 * pow(d4, 2) +
          16 * pow(a2, 2) * b1 * c3 * pow(d1, 2) -
          48 * pow(a2, 2) * b3 * c1 * pow(d1, 2) +
          16 * pow(a4, 2) * b0 * c1 * pow(d2, 2) -
          16 * pow(a4, 2) * b1 * c0 * pow(d2, 2) -
          24 * pow(a5, 2) * b0 * c1 * pow(d1, 2) +
          8 * pow(a5, 2) * b1 * c0 * pow(d1, 2) +
          4 * pow(a0, 2) * b1 * c3 * pow(d4, 2) -
          4 * pow(a0, 2) * b1 * c5 * pow(d2, 2) -
          8 * pow(a0, 2) * b2 * c4 * pow(d2, 2) -
          4 * pow(a0, 2) * b3 * c1 * pow(d4, 2) +
          24 * pow(a0, 2) * b4 * c2 * pow(d2, 2) +
          4 * pow(a0, 2) * b5 * c1 * pow(d2, 2) -
          4 * pow(a2, 2) * b0 * c1 * pow(d5, 2) +
          4 * pow(a2, 2) * b1 * c0 * pow(d5, 2) -
          4 * pow(a2, 2) * b1 * c5 * pow(d0, 2) -
          24 * pow(a2, 2) * b2 * c4 * pow(d0, 2) +
          8 * pow(a2, 2) * b4 * c2 * pow(d0, 2) +
          4 * pow(a2, 2) * b5 * c1 * pow(d0, 2) +
          4 * pow(a4, 2) * b1 * c3 * pow(d0, 2) -
          4 * pow(a4, 2) * b3 * c1 * pow(d0, 2) -
          4 * pow(a5, 2) * b0 * c1 * pow(d2, 2) +
          4 * pow(a5, 2) * b1 * c0 * pow(d2, 2) +
          2 * pow(a0, 2) * b1 * c3 * pow(d5, 2) -
          pow(a0, 2) * b1 * c5 * pow(d3, 2) -
          2 * pow(a0, 2) * b2 * c4 * pow(d3, 2) +
          2 * pow(a0, 2) * b3 * c1 * pow(d5, 2) -
          2 * pow(a0, 2) * b4 * c2 * pow(d3, 2) +
          3 * pow(a0, 2) * b5 * c1 * pow(d3, 2) -
          48 * pow(a1, 2) * b1 * c5 * pow(d2, 2) -
          32 * pow(a1, 2) * b2 * c4 * pow(d2, 2) +
          96 * pow(a1, 2) * b4 * c2 * pow(d2, 2) +
          16 * pow(a1, 2) * b5 * c1 * pow(d2, 2) -
          16 * pow(a2, 2) * b1 * c5 * pow(d1, 2) -
          96 * pow(a2, 2) * b2 * c4 * pow(d1, 2) +
          32 * pow(a2, 2) * b4 * c2 * pow(d1, 2) +
          48 * pow(a2, 2) * b5 * c1 * pow(d1, 2) -
          pow(a3, 2) * b0 * c1 * pow(d5, 2) +
          3 * pow(a3, 2) * b1 * c0 * pow(d5, 2) -
          3 * pow(a3, 2) * b1 * c5 * pow(d0, 2) +
          2 * pow(a3, 2) * b2 * c4 * pow(d0, 2) +
          2 * pow(a3, 2) * b4 * c2 * pow(d0, 2) +
          pow(a3, 2) * b5 * c1 * pow(d0, 2) -
          3 * pow(a5, 2) * b0 * c1 * pow(d3, 2) +
          pow(a5, 2) * b1 * c0 * pow(d3, 2) -
          2 * pow(a5, 2) * b1 * c3 * pow(d0, 2) -
          2 * pow(a5, 2) * b3 * c1 * pow(d0, 2) -
          8 * pow(a0, 2) * b5 * c1 * pow(d4, 2) +
          12 * pow(a1, 2) * b1 * c3 * pow(d5, 2) -
          4 * pow(a1, 2) * b3 * c1 * pow(d5, 2) +
          16 * pow(a2, 2) * b1 * c3 * pow(d4, 2) -
          16 * pow(a2, 2) * b3 * c1 * pow(d4, 2) +
          4 * pow(a4, 2) * b0 * c1 * pow(d5, 2) -
          4 * pow(a4, 2) * b1 * c0 * pow(d5, 2) +
          16 * pow(a4, 2) * b1 * c3 * pow(d2, 2) +
          8 * pow(a4, 2) * b1 * c5 * pow(d0, 2) -
          16 * pow(a4, 2) * b3 * c1 * pow(d2, 2) +
          4 * pow(a5, 2) * b0 * c1 * pow(d4, 2) -
          4 * pow(a5, 2) * b1 * c0 * pow(d4, 2) +
          4 * pow(a5, 2) * b1 * c3 * pow(d1, 2) -
          12 * pow(a5, 2) * b3 * c1 * pow(d1, 2) -
          3 * pow(a0, 2) * b1 * c5 * pow(d5, 2) +
          2 * pow(a0, 2) * b2 * c4 * pow(d5, 2) +
          2 * pow(a0, 2) * b4 * c2 * pow(d5, 2) +
          pow(a0, 2) * b5 * c1 * pow(d5, 2) -
          4 * pow(a2, 2) * b1 * c3 * pow(d5, 2) +
          4 * pow(a2, 2) * b1 * c5 * pow(d3, 2) -
          24 * pow(a2, 2) * b2 * c4 * pow(d3, 2) +
          4 * pow(a2, 2) * b3 * c1 * pow(d5, 2) +
          8 * pow(a2, 2) * b4 * c2 * pow(d3, 2) +
          4 * pow(a2, 2) * b5 * c1 * pow(d3, 2) -
          4 * pow(a3, 2) * b1 * c5 * pow(d2, 2) -
          8 * pow(a3, 2) * b2 * c4 * pow(d2, 2) +
          24 * pow(a3, 2) * b4 * c2 * pow(d2, 2) -
          4 * pow(a3, 2) * b5 * c1 * pow(d2, 2) -
          4 * pow(a5, 2) * b1 * c3 * pow(d2, 2) -
          pow(a5, 2) * b1 * c5 * pow(d0, 2) -
          2 * pow(a5, 2) * b2 * c4 * pow(d0, 2) +
          4 * pow(a5, 2) * b3 * c1 * pow(d2, 2) -
          2 * pow(a5, 2) * b4 * c2 * pow(d0, 2) +
          3 * pow(a5, 2) * b5 * c1 * pow(d0, 2) -
          36 * pow(a1, 2) * b1 * c5 * pow(d5, 2) +
          8 * pow(a1, 2) * b2 * c4 * pow(d5, 2) +
          8 * pow(a1, 2) * b4 * c2 * pow(d5, 2) +
          12 * pow(a1, 2) * b5 * c1 * pow(d5, 2) -
          12 * pow(a5, 2) * b1 * c5 * pow(d1, 2) -
          8 * pow(a5, 2) * b2 * c4 * pow(d1, 2) -
          8 * pow(a5, 2) * b4 * c2 * pow(d1, 2) +
          36 * pow(a5, 2) * b5 * c1 * pow(d1, 2) +
          4 * pow(a4, 2) * b1 * c3 * pow(d5, 2) -
          4 * pow(a4, 2) * b3 * c1 * pow(d5, 2) +
          4 * pow(a5, 2) * b1 * c3 * pow(d4, 2) -
          4 * pow(a5, 2) * b3 * c1 * pow(d4, 2) -
          3 * pow(a3, 2) * b1 * c5 * pow(d5, 2) +
          2 * pow(a3, 2) * b2 * c4 * pow(d5, 2) +
          2 * pow(a3, 2) * b4 * c2 * pow(d5, 2) +
          pow(a3, 2) * b5 * c1 * pow(d5, 2) -
          pow(a5, 2) * b1 * c5 * pow(d3, 2) -
          2 * pow(a5, 2) * b2 * c4 * pow(d3, 2) -
          2 * pow(a5, 2) * b4 * c2 * pow(d3, 2) +
          3 * pow(a5, 2) * b5 * c1 * pow(d3, 2) +
          2 * a0 * a1 * b0 * c0 * pow(d3, 2) -
          8 * a0 * a1 * b0 * c0 * pow(d4, 2) +
          8 * a0 * a1 * b0 * c3 * pow(d1, 2) +
          8 * a0 * a1 * b3 * c0 * pow(d1, 2) -
          24 * a0 * a3 * b0 * c1 * pow(d1, 2) +
          8 * a0 * a3 * b1 * c0 * pow(d1, 2) +
          8 * a1 * a3 * b0 * c0 * pow(d1, 2) +
          2 * a0 * a1 * b0 * c0 * pow(d5, 2) +
          8 * a0 * a1 * b0 * c3 * pow(d2, 2) -
          8 * a0 * a3 * b0 * c1 * pow(d2, 2) -
          8 * a1 * a3 * b1 * c1 * pow(d0, 2) -
          8 * a0 * a1 * b0 * c5 * pow(d1, 2) -
          2 * a0 * a1 * b3 * c3 * pow(d0, 2) -
          8 * a0 * a1 * b5 * c0 * pow(d1, 2) -
          16 * a0 * a2 * b0 * c4 * pow(d1, 2) -
          2 * a0 * a3 * b1 * c3 * pow(d0, 2) -
          2 * a0 * a3 * b3 * c1 * pow(d0, 2) +
          16 * a0 * a4 * b0 * c2 * pow(d1, 2) +
          24 * a0 * a5 * b0 * c1 * pow(d1, 2) -
          8 * a0 * a5 * b1 * c0 * pow(d1, 2) -
          2 * a1 * a3 * b0 * c3 * pow(d0, 2) +
          6 * a1 * a3 * b3 * c0 * pow(d0, 2) -
          8 * a1 * a5 * b0 * c0 * pow(d1, 2) +
          8 * a0 * a1 * b0 * c3 * pow(d4, 2) -
          8 * a0 * a1 * b0 * c5 * pow(d2, 2) -
          16 * a0 * a1 * b1 * c1 * pow(d5, 2) +
          8 * a0 * a1 * b2 * c2 * pow(d3, 2) -
          16 * a0 * a2 * b0 * c4 * pow(d2, 2) +
          8 * a0 * a2 * b1 * c2 * pow(d3, 2) -
          8 * a0 * a2 * b2 * c1 * pow(d3, 2) -
          16 * a0 * a2 * b4 * c0 * pow(d2, 2) -
          8 * a0 * a3 * b0 * c1 * pow(d4, 2) +
          48 * a0 * a4 * b0 * c2 * pow(d2, 2) -
          16 * a0 * a4 * b2 * c0 * pow(d2, 2) +
          8 * a0 * a5 * b0 * c1 * pow(d2, 2) +
          8 * a1 * a2 * b0 * c2 * pow(d3, 2) -
          16 * a1 * a2 * b1 * c4 * pow(d0, 2) -
          8 * a1 * a2 * b2 * c0 * pow(d3, 2) +
          8 * a1 * a2 * b2 * c3 * pow(d0, 2) -
          32 * a1 * a3 * b1 * c1 * pow(d2, 2) +
          16 * a1 * a4 * b1 * c2 * pow(d0, 2) +
          8 * a1 * a5 * b1 * c1 * pow(d0, 2) -
          8 * a2 * a3 * b2 * c1 * pow(d0, 2) -
          16 * a2 * a4 * b0 * c0 * pow(d2, 2) +
          4 * a0 * a1 * b0 * c3 * pow(d5, 2) -
          2 * a0 * a1 * b0 * c5 * pow(d3, 2) -
          4 * a0 * a1 * b3 * c0 * pow(d5, 2) -
          8 * a0 * a1 * b3 * c3 * pow(d2, 2) +
          2 * a0 * a1 * b3 * c5 * pow(d0, 2) +
          8 * a0 * a1 * b4 * c4 * pow(d0, 2) -
          2 * a0 * a1 * b5 * c0 * pow(d3, 2) +
          2 * a0 * a1 * b5 * c3 * pow(d0, 2) -
          4 * a0 * a2 * b0 * c4 * pow(d3, 2) +
          32 * a0 * a2 * b2 * c1 * pow(d4, 2) -
          4 * a0 * a2 * b3 * c4 * pow(d0, 2) +
          4 * a0 * a2 * b4 * c0 * pow(d3, 2) -
          4 * a0 * a2 * b4 * c3 * pow(d0, 2) +
          4 * a0 * a3 * b0 * c1 * pow(d5, 2) -
          4 * a0 * a3 * b1 * c0 * pow(d5, 2) -
          8 * a0 * a3 * b1 * c3 * pow(d2, 2) +
          2 * a0 * a3 * b1 * c5 * pow(d0, 2) -
          4 * a0 * a3 * b2 * c4 * pow(d0, 2) +
          8 * a0 * a3 * b3 * c1 * pow(d2, 2) -
          4 * a0 * a3 * b4 * c2 * pow(d0, 2) +
          2 * a0 * a3 * b5 * c1 * pow(d0, 2) -
          4 * a0 * a4 * b0 * c2 * pow(d3, 2) +
          8 * a0 * a4 * b1 * c4 * pow(d0, 2) +
          4 * a0 * a4 * b2 * c0 * pow(d3, 2) -
          4 * a0 * a4 * b2 * c3 * pow(d0, 2) -
          4 * a0 * a4 * b3 * c2 * pow(d0, 2) +
          8 * a0 * a4 * b4 * c1 * pow(d0, 2) +
          6 * a0 * a5 * b0 * c1 * pow(d3, 2) -
          2 * a0 * a5 * b1 * c0 * pow(d3, 2) +
          2 * a0 * a5 * b1 * c3 * pow(d0, 2) +
          2 * a0 * a5 * b3 * c1 * pow(d0, 2) -
          32 * a1 * a2 * b2 * c0 * pow(d4, 2) +
          32 * a1 * a2 * b2 * c3 * pow(d1, 2) +
          32 * a1 * a2 * b3 * c2 * pow(d1, 2) -
          4 * a1 * a3 * b0 * c0 * pow(d5, 2) -
          8 * a1 * a3 * b0 * c3 * pow(d2, 2) +
          2 * a1 * a3 * b0 * c5 * pow(d0, 2) +
          32 * a1 * a3 * b2 * c2 * pow(d1, 2) +
          8 * a1 * a3 * b3 * c0 * pow(d2, 2) -
          6 * a1 * a3 * b5 * c0 * pow(d0, 2) +
          8 * a1 * a4 * b0 * c4 * pow(d0, 2) -
          24 * a1 * a4 * b4 * c0 * pow(d0, 2) -
          2 * a1 * a5 * b0 * c0 * pow(d3, 2) +
          2 * a1 * a5 * b0 * c3 * pow(d0, 2) -
          6 * a1 * a5 * b3 * c0 * pow(d0, 2) -
          4 * a2 * a3 * b0 * c4 * pow(d0, 2) +
          32 * a2 * a3 * b1 * c2 * pow(d1, 2) -
          96 * a2 * a3 * b2 * c1 * pow(d1, 2) +
          12 * a2 * a3 * b4 * c0 * pow(d0, 2) +
          4 * a2 * a4 * b0 * c0 * pow(d3, 2) -
          4 * a2 * a4 * b0 * c3 * pow(d0, 2) +
          12 * a2 * a4 * b3 * c0 * pow(d0, 2) -
          4 * a3 * a4 * b0 * c2 * pow(d0, 2) +
          12 * a3 * a4 * b2 * c0 * pow(d0, 2) +
          2 * a3 * a5 * b0 * c1 * pow(d0, 2) -
          6 * a3 * a5 * b1 * c0 * pow(d0, 2) -
          8 * a0 * a1 * b3 * c5 * pow(d1, 2) +
          8 * a0 * a1 * b5 * c0 * pow(d4, 2) -
          8 * a0 * a1 * b5 * c3 * pow(d1, 2) -
          8 * a0 * a2 * b2 * c1 * pow(d5, 2) -
          8 * a0 * a3 * b1 * c5 * pow(d1, 2) -
          16 * a0 * a3 * b4 * c2 * pow(d1, 2) +
          24 * a0 * a3 * b5 * c1 * pow(d1, 2) -
          16 * a0 * a4 * b3 * c2 * pow(d1, 2) -
          16 * a0 * a5 * b0 * c1 * pow(d4, 2) +
          8 * a0 * a5 * b1 * c0 * pow(d4, 2) -
          8 * a0 * a5 * b1 * c3 * pow(d1, 2) +
          24 * a0 * a5 * b3 * c1 * pow(d1, 2) -
          64 * a1 * a2 * b1 * c4 * pow(d2, 2) +
          8 * a1 * a2 * b2 * c0 * pow(d5, 2) -
          8 * a1 * a2 * b2 * c5 * pow(d0, 2) -
          64 * a1 * a2 * b4 * c1 * pow(d2, 2) -
          8 * a1 * a3 * b0 * c5 * pow(d1, 2) -
          8 * a1 * a3 * b5 * c0 * pow(d1, 2) +
          192 * a1 * a4 * b1 * c2 * pow(d2, 2) -
          64 * a1 * a4 * b2 * c1 * pow(d2, 2) +
          8 * a1 * a5 * b0 * c0 * pow(d4, 2) -
          8 * a1 * a5 * b0 * c3 * pow(d1, 2) +
          32 * a1 * a5 * b1 * c1 * pow(d2, 2) -
          8 * a1 * a5 * b3 * c0 * pow(d1, 2) +
          16 * a2 * a3 * b4 * c0 * pow(d1, 2) -
          64 * a2 * a4 * b1 * c1 * pow(d2, 2) +
          16 * a2 * a4 * b2 * c2 * pow(d0, 2) +
          16 * a2 * a4 * b3 * c0 * pow(d1, 2) +
          8 * a2 * a5 * b2 * c1 * pow(d0, 2) -
          16 * a3 * a4 * b0 * c2 * pow(d1, 2) +
          16 * a3 * a4 * b2 * c0 * pow(d1, 2) +
          24 * a3 * a5 * b0 * c1 * pow(d1, 2) -
          8 * a3 * a5 * b1 * c0 * pow(d1, 2) -
          6 * a0 * a1 * b0 * c5 * pow(d5, 2) +
          8 * a0 * a1 * b3 * c5 * pow(d2, 2) +
          2 * a0 * a1 * b5 * c0 * pow(d5, 2) -
          2 * a0 * a1 * b5 * c5 * pow(d0, 2) +
          4 * a0 * a2 * b0 * c4 * pow(d5, 2) +
          16 * a0 * a2 * b3 * c4 * pow(d2, 2) -
          4 * a0 * a2 * b4 * c0 * pow(d5, 2) +
          16 * a0 * a2 * b4 * c3 * pow(d2, 2) +
          4 * a0 * a2 * b4 * c5 * pow(d0, 2) +
          4 * a0 * a2 * b5 * c4 * pow(d0, 2) +
          8 * a0 * a3 * b1 * c5 * pow(d2, 2) +
          16 * a0 * a3 * b2 * c4 * pow(d2, 2) -
          48 * a0 * a3 * b4 * c2 * pow(d2, 2) +
          4 * a0 * a4 * b0 * c2 * pow(d5, 2) -
          4 * a0 * a4 * b2 * c0 * pow(d5, 2) +
          16 * a0 * a4 * b2 * c3 * pow(d2, 2) +
          4 * a0 * a4 * b2 * c5 * pow(d0, 2) -
          48 * a0 * a4 * b3 * c2 * pow(d2, 2) +
          32 * a0 * a4 * b4 * c1 * pow(d2, 2) +
          4 * a0 * a4 * b5 * c2 * pow(d0, 2) +
          2 * a0 * a5 * b0 * c1 * pow(d5, 2) +
          2 * a0 * a5 * b1 * c0 * pow(d5, 2) -
          2 * a0 * a5 * b1 * c5 * pow(d0, 2) +
          4 * a0 * a5 * b2 * c4 * pow(d0, 2) +
          4 * a0 * a5 * b4 * c2 * pow(d0, 2) -
          2 * a0 * a5 * b5 * c1 * pow(d0, 2) -
          32 * a1 * a2 * b2 * c5 * pow(d1, 2) -
          32 * a1 * a2 * b5 * c2 * pow(d1, 2) +
          8 * a1 * a3 * b0 * c5 * pow(d2, 2) -
          8 * a1 * a3 * b1 * c1 * pow(d5, 2) -
          8 * a1 * a3 * b5 * c0 * pow(d2, 2) -
          32 * a1 * a4 * b4 * c0 * pow(d2, 2) +
          2 * a1 * a5 * b0 * c0 * pow(d5, 2) -
          2 * a1 * a5 * b0 * c5 * pow(d0, 2) -
          32 * a1 * a5 * b2 * c2 * pow(d1, 2) -
          8 * a1 * a5 * b3 * c0 * pow(d2, 2) +
          6 * a1 * a5 * b5 * c0 * pow(d0, 2) +
          16 * a2 * a3 * b0 * c4 * pow(d2, 2) +
          16 * a2 * a3 * b4 * c0 * pow(d2, 2) -
          4 * a2 * a4 * b0 * c0 * pow(d5, 2) +
          16 * a2 * a4 * b0 * c3 * pow(d2, 2) +
          4 * a2 * a4 * b0 * c5 * pow(d0, 2) +
          64 * a2 * a4 * b2 * c2 * pow(d1, 2) +
          16 * a2 * a4 * b3 * c0 * pow(d2, 2) -
          12 * a2 * a4 * b5 * c0 * pow(d0, 2) +
          4 * a2 * a5 * b0 * c4 * pow(d0, 2) -
          32 * a2 * a5 * b1 * c2 * pow(d1, 2) +
          96 * a2 * a5 * b2 * c1 * pow(d1, 2) -
          12 * a2 * a5 * b4 * c0 * pow(d0, 2) -
          48 * a3 * a4 * b0 * c2 * pow(d2, 2) +
          16 * a3 * a4 * b2 * c0 * pow(d2, 2) -
          8 * a3 * a5 * b1 * c0 * pow(d2, 2) +
          4 * a4 * a5 * b0 * c2 * pow(d0, 2) -
          12 * a4 * a5 * b2 * c0 * pow(d0, 2) -
          2 * a0 * a1 * b3 * c3 * pow(d5, 2) +
          16 * a0 * a1 * b5 * c5 * pow(d1, 2) +
          16 * a0 * a2 * b5 * c4 * pow(d1, 2) -
          2 * a0 * a3 * b1 * c3 * pow(d5, 2) -
          2 * a0 * a3 * b3 * c1 * pow(d5, 2) +
          16 * a0 * a5 * b1 * c5 * pow(d1, 2) +
          16 * a0 * a5 * b2 * c4 * pow(d1, 2) -
          48 * a0 * a5 * b5 * c1 * pow(d1, 2) +
          32 * a1 * a2 * b2 * c3 * pow(d4, 2) -
          2 * a1 * a3 * b0 * c3 * pow(d5, 2) +
          6 * a1 * a3 * b3 * c0 * pow(d5, 2) -
          6 * a1 * a3 * b3 * c5 * pow(d0, 2) +
          2 * a1 * a3 * b5 * c3 * pow(d0, 2) +
          8 * a1 * a4 * b4 * c3 * pow(d0, 2) +
          16 * a1 * a5 * b0 * c5 * pow(d1, 2) +
          2 * a1 * a5 * b3 * c3 * pow(d0, 2) +
          16 * a1 * a5 * b5 * c0 * pow(d1, 2) -
          32 * a2 * a3 * b2 * c1 * pow(d4, 2) +
          4 * a2 * a3 * b3 * c4 * pow(d0, 2) -
          4 * a2 * a3 * b4 * c3 * pow(d0, 2) -
          4 * a2 * a4 * b3 * c3 * pow(d0, 2) -
          16 * a2 * a4 * b5 * c0 * pow(d1, 2) +
          16 * a2 * a5 * b0 * c4 * pow(d1, 2) -
          16 * a2 * a5 * b4 * c0 * pow(d1, 2) -
          4 * a3 * a4 * b2 * c3 * pow(d0, 2) +
          4 * a3 * a4 * b3 * c2 * pow(d0, 2) -
          8 * a3 * a4 * b4 * c1 * pow(d0, 2) +
          2 * a3 * a5 * b1 * c3 * pow(d0, 2) +
          2 * a3 * a5 * b3 * c1 * pow(d0, 2) -
          16 * a4 * a5 * b2 * c0 * pow(d1, 2) -
          8 * a0 * a1 * b5 * c3 * pow(d4, 2) +
          8 * a0 * a3 * b5 * c1 * pow(d4, 2) -
          8 * a0 * a5 * b1 * c3 * pow(d4, 2) +
          8 * a0 * a5 * b3 * c1 * pow(d4, 2) -
          8 * a0 * a5 * b5 * c1 * pow(d2, 2) +
          16 * a1 * a2 * b1 * c4 * pow(d5, 2) -
          8 * a1 * a2 * b2 * c3 * pow(d5, 2) +
          8 * a1 * a2 * b2 * c5 * pow(d3, 2) -
          16 * a1 * a2 * b4 * c1 * pow(d5, 2) -
          8 * a1 * a2 * b5 * c2 * pow(d3, 2) +
          16 * a1 * a4 * b1 * c2 * pow(d5, 2) -
          16 * a1 * a4 * b2 * c1 * pow(d5, 2) -
          8 * a1 * a5 * b0 * c3 * pow(d4, 2) +
          24 * a1 * a5 * b1 * c1 * pow(d5, 2) -
          8 * a1 * a5 * b2 * c2 * pow(d3, 2) +
          8 * a1 * a5 * b5 * c0 * pow(d2, 2) +
          8 * a2 * a3 * b2 * c1 * pow(d5, 2) -
          16 * a2 * a4 * b1 * c1 * pow(d5, 2) +
          16 * a2 * a4 * b2 * c2 * pow(d3, 2) -
          8 * a2 * a5 * b1 * c2 * pow(d3, 2) +
          8 * a2 * a5 * b2 * c1 * pow(d3, 2) +
          8 * a3 * a5 * b0 * c1 * pow(d4, 2) +
          6 * a0 * a1 * b3 * c5 * pow(d5, 2) -
          2 * a0 * a1 * b5 * c3 * pow(d5, 2) +
          2 * a0 * a1 * b5 * c5 * pow(d3, 2) -
          4 * a0 * a2 * b3 * c4 * pow(d5, 2) +
          4 * a0 * a2 * b4 * c3 * pow(d5, 2) -
          4 * a0 * a2 * b4 * c5 * pow(d3, 2) +
          4 * a0 * a2 * b5 * c4 * pow(d3, 2) +
          6 * a0 * a3 * b1 * c5 * pow(d5, 2) -
          4 * a0 * a3 * b2 * c4 * pow(d5, 2) -
          4 * a0 * a3 * b4 * c2 * pow(d5, 2) -
          2 * a0 * a3 * b5 * c1 * pow(d5, 2) +
          4 * a0 * a4 * b2 * c3 * pow(d5, 2) -
          4 * a0 * a4 * b2 * c5 * pow(d3, 2) -
          4 * a0 * a4 * b3 * c2 * pow(d5, 2) +
          8 * a0 * a4 * b4 * c1 * pow(d5, 2) +
          4 * a0 * a4 * b5 * c2 * pow(d3, 2) -
          2 * a0 * a5 * b1 * c3 * pow(d5, 2) +
          2 * a0 * a5 * b1 * c5 * pow(d3, 2) +
          4 * a0 * a5 * b2 * c4 * pow(d3, 2) -
          2 * a0 * a5 * b3 * c1 * pow(d5, 2) +
          4 * a0 * a5 * b4 * c2 * pow(d3, 2) -
          6 * a0 * a5 * b5 * c1 * pow(d3, 2) +
          6 * a1 * a3 * b0 * c5 * pow(d5, 2) -
          8 * a1 * a3 * b3 * c5 * pow(d2, 2) -
          2 * a1 * a3 * b5 * c0 * pow(d5, 2) +
          8 * a1 * a3 * b5 * c3 * pow(d2, 2) +
          4 * a1 * a3 * b5 * c5 * pow(d0, 2) -
          8 * a1 * a4 * b4 * c0 * pow(d5, 2) +
          32 * a1 * a4 * b4 * c3 * pow(d2, 2) +
          16 * a1 * a4 * b4 * c5 * pow(d0, 2) -
          8 * a1 * a4 * b5 * c4 * pow(d0, 2) -
          2 * a1 * a5 * b0 * c3 * pow(d5, 2) +
          2 * a1 * a5 * b0 * c5 * pow(d3, 2) -
          2 * a1 * a5 * b3 * c0 * pow(d5, 2) +
          8 * a1 * a5 * b3 * c3 * pow(d2, 2) +
          4 * a1 * a5 * b3 * c5 * pow(d0, 2) -
          8 * a1 * a5 * b4 * c4 * pow(d0, 2) +
          2 * a1 * a5 * b5 * c0 * pow(d3, 2) -
          4 * a1 * a5 * b5 * c3 * pow(d0, 2) -
          4 * a2 * a3 * b0 * c4 * pow(d5, 2) -
          16 * a2 * a3 * b3 * c4 * pow(d2, 2) +
          4 * a2 * a3 * b4 * c0 * pow(d5, 2) -
          16 * a2 * a3 * b4 * c3 * pow(d2, 2) -
          8 * a2 * a3 * b4 * c5 * pow(d0, 2) +
          4 * a2 * a4 * b0 * c3 * pow(d5, 2) -
          4 * a2 * a4 * b0 * c5 * pow(d3, 2) +
          4 * a2 * a4 * b3 * c0 * pow(d5, 2) -
          16 * a2 * a4 * b3 * c3 * pow(d2, 2) -
          8 * a2 * a4 * b3 * c5 * pow(d0, 2) -
          4 * a2 * a4 * b5 * c0 * pow(d3, 2) +
          8 * a2 * a4 * b5 * c3 * pow(d0, 2) +
          4 * a2 * a5 * b0 * c4 * pow(d3, 2) -
          4 * a2 * a5 * b4 * c0 * pow(d3, 2) +
          8 * a2 * a5 * b4 * c3 * pow(d0, 2) -
          4 * a3 * a4 * b0 * c2 * pow(d5, 2) +
          4 * a3 * a4 * b2 * c0 * pow(d5, 2) -
          16 * a3 * a4 * b2 * c3 * pow(d2, 2) -
          8 * a3 * a4 * b2 * c5 * pow(d0, 2) +
          48 * a3 * a4 * b3 * c2 * pow(d2, 2) -
          32 * a3 * a4 * b4 * c1 * pow(d2, 2) -
          2 * a3 * a5 * b0 * c1 * pow(d5, 2) -
          2 * a3 * a5 * b1 * c0 * pow(d5, 2) +
          8 * a3 * a5 * b1 * c3 * pow(d2, 2) +
          4 * a3 * a5 * b1 * c5 * pow(d0, 2) -
          8 * a3 * a5 * b3 * c1 * pow(d2, 2) -
          4 * a3 * a5 * b5 * c1 * pow(d0, 2) +
          4 * a4 * a5 * b0 * c2 * pow(d3, 2) -
          8 * a4 * a5 * b1 * c4 * pow(d0, 2) -
          4 * a4 * a5 * b2 * c0 * pow(d3, 2) +
          8 * a4 * a5 * b2 * c3 * pow(d0, 2) +
          8 * a0 * a5 * b5 * c1 * pow(d4, 2) +
          8 * a1 * a3 * b5 * c5 * pow(d1, 2) +
          8 * a1 * a5 * b3 * c5 * pow(d1, 2) -
          8 * a1 * a5 * b5 * c0 * pow(d4, 2) +
          8 * a1 * a5 * b5 * c3 * pow(d1, 2) -
          16 * a2 * a3 * b4 * c5 * pow(d1, 2) -
          16 * a2 * a4 * b3 * c5 * pow(d1, 2) -
          16 * a3 * a4 * b2 * c5 * pow(d1, 2) +
          16 * a3 * a4 * b5 * c2 * pow(d1, 2) +
          8 * a3 * a5 * b1 * c5 * pow(d1, 2) +
          16 * a3 * a5 * b4 * c2 * pow(d1, 2) -
          24 * a3 * a5 * b5 * c1 * pow(d1, 2) +
          16 * a4 * a5 * b3 * c2 * pow(d1, 2) -
          8 * a1 * a5 * b5 * c3 * pow(d2, 2) -
          2 * a1 * a5 * b5 * c5 * pow(d0, 2) +
          4 * a2 * a4 * b5 * c5 * pow(d0, 2) +
          4 * a2 * a5 * b4 * c5 * pow(d0, 2) -
          4 * a2 * a5 * b5 * c4 * pow(d0, 2) +
          8 * a3 * a5 * b5 * c1 * pow(d2, 2) +
          4 * a4 * a5 * b2 * c5 * pow(d0, 2) -
          4 * a4 * a5 * b5 * c2 * pow(d0, 2) -
          6 * a1 * a3 * b3 * c5 * pow(d5, 2) +
          2 * a1 * a3 * b5 * c3 * pow(d5, 2) +
          8 * a1 * a4 * b4 * c3 * pow(d5, 2) +
          2 * a1 * a5 * b3 * c3 * pow(d5, 2) -
          24 * a1 * a5 * b5 * c5 * pow(d1, 2) +
          4 * a2 * a3 * b3 * c4 * pow(d5, 2) -
          4 * a2 * a3 * b4 * c3 * pow(d5, 2) -
          4 * a2 * a4 * b3 * c3 * pow(d5, 2) +
          16 * a2 * a4 * b5 * c5 * pow(d1, 2) +
          16 * a2 * a5 * b4 * c5 * pow(d1, 2) -
          16 * a2 * a5 * b5 * c4 * pow(d1, 2) -
          4 * a3 * a4 * b2 * c3 * pow(d5, 2) +
          4 * a3 * a4 * b3 * c2 * pow(d5, 2) -
          8 * a3 * a4 * b4 * c1 * pow(d5, 2) +
          2 * a3 * a5 * b1 * c3 * pow(d5, 2) +
          2 * a3 * a5 * b3 * c1 * pow(d5, 2) +
          16 * a4 * a5 * b2 * c5 * pow(d1, 2) -
          16 * a4 * a5 * b5 * c2 * pow(d1, 2) +
          8 * a1 * a5 * b5 * c3 * pow(d4, 2) -
          8 * a3 * a5 * b5 * c1 * pow(d4, 2) -
          2 * a1 * a5 * b5 * c5 * pow(d3, 2) +
          4 * a2 * a4 * b5 * c5 * pow(d3, 2) +
          4 * a2 * a5 * b4 * c5 * pow(d3, 2) -
          4 * a2 * a5 * b5 * c4 * pow(d3, 2) +
          4 * a4 * a5 * b2 * c5 * pow(d3, 2) -
          4 * a4 * a5 * b5 * c2 * pow(d3, 2) -
          2 * pow(a3, 2) * b0 * c0 * d0 * d1 -
          8 * pow(a1, 2) * b0 * c0 * d1 * d3 -
          8 * pow(a1, 2) * b0 * c1 * d0 * d3 -
          8 * pow(a1, 2) * b0 * c3 * d0 * d1 +
          24 * pow(a1, 2) * b1 * c0 * d0 * d3 -
          8 * pow(a1, 2) * b3 * c0 * d0 * d1 +
          8 * pow(a4, 2) * b0 * c0 * d0 * d1 +
          8 * pow(a0, 2) * b1 * c1 * d1 * d3 +
          8 * pow(a2, 2) * b1 * c0 * d0 * d3 -
          8 * pow(a2, 2) * b3 * c0 * d0 * d1 -
          2 * pow(a5, 2) * b0 * c0 * d0 * d1 -
          6 * pow(a0, 2) * b0 * c3 * d1 * d3 +
          2 * pow(a0, 2) * b1 * c3 * d0 * d3 +
          2 * pow(a0, 2) * b3 * c0 * d1 * d3 +
          2 * pow(a0, 2) * b3 * c1 * d0 * d3 +
          2 * pow(a0, 2) * b3 * c3 * d0 * d1 +
          8 * pow(a1, 2) * b0 * c0 * d1 * d5 +
          8 * pow(a1, 2) * b0 * c1 * d0 * d5 +
          8 * pow(a1, 2) * b0 * c5 * d0 * d1 -
          24 * pow(a1, 2) * b1 * c0 * d0 * d5 -
          16 * pow(a1, 2) * b2 * c0 * d0 * d4 +
          16 * pow(a1, 2) * b4 * c0 * d0 * d2 +
          8 * pow(a1, 2) * b5 * c0 * d0 * d1 -
          8 * pow(a0, 2) * b1 * c1 * d1 * d5 +
          8 * pow(a0, 2) * b1 * c2 * d2 * d3 -
          16 * pow(a0, 2) * b2 * c1 * d1 * d4 -
          8 * pow(a0, 2) * b3 * c2 * d1 * d2 +
          16 * pow(a0, 2) * b4 * c1 * d1 * d2 +
          16 * pow(a2, 2) * b0 * c0 * d2 * d4 +
          16 * pow(a2, 2) * b0 * c2 * d0 * d4 +
          16 * pow(a2, 2) * b0 * c4 * d0 * d2 -
          8 * pow(a2, 2) * b1 * c0 * d0 * d5 +
          32 * pow(a2, 2) * b1 * c1 * d1 * d3 -
          48 * pow(a2, 2) * b2 * c0 * d0 * d4 +
          16 * pow(a2, 2) * b4 * c0 * d0 * d2 +
          8 * pow(a2, 2) * b5 * c0 * d0 * d1 +
          8 * pow(a3, 2) * b0 * c2 * d1 * d2 +
          8 * pow(a3, 2) * b1 * c2 * d0 * d2 -
          8 * pow(a3, 2) * b2 * c0 * d1 * d2 -
          8 * pow(a3, 2) * b2 * c1 * d0 * d2 -
          8 * pow(a3, 2) * b2 * c2 * d0 * d1 +
          8 * pow(a4, 2) * b1 * c0 * d0 * d3 -
          8 * pow(a4, 2) * b3 * c0 * d0 * d1 +
          16 * pow(a5, 2) * b1 * c1 * d0 * d1 +
          6 * pow(a0, 2) * b0 * c1 * d3 * d5 -
          12 * pow(a0, 2) * b0 * c2 * d3 * d4 +
          6 * pow(a0, 2) * b0 * c3 * d1 * d5 -
          12 * pow(a0, 2) * b0 * c3 * d2 * d4 +
          24 * pow(a0, 2) * b0 * c4 * d1 * d4 -
          12 * pow(a0, 2) * b0 * c4 * d2 * d3 +
          6 * pow(a0, 2) * b0 * c5 * d1 * d3 -
          2 * pow(a0, 2) * b1 * c0 * d3 * d5 -
          2 * pow(a0, 2) * b1 * c3 * d0 * d5 -
          8 * pow(a0, 2) * b1 * c4 * d0 * d4 -
          2 * pow(a0, 2) * b1 * c5 * d0 * d3 +
          4 * pow(a0, 2) * b2 * c0 * d3 * d4 +
          4 * pow(a0, 2) * b2 * c3 * d0 * d4 +
          4 * pow(a0, 2) * b2 * c4 * d0 * d3 -
          2 * pow(a0, 2) * b3 * c0 * d1 * d5 +
          4 * pow(a0, 2) * b3 * c0 * d2 * d4 -
          2 * pow(a0, 2) * b3 * c1 * d0 * d5 +
          4 * pow(a0, 2) * b3 * c2 * d0 * d4 +
          4 * pow(a0, 2) * b3 * c4 * d0 * d2 -
          2 * pow(a0, 2) * b3 * c5 * d0 * d1 -
          8 * pow(a0, 2) * b4 * c0 * d1 * d4 +
          4 * pow(a0, 2) * b4 * c0 * d2 * d3 -
          8 * pow(a0, 2) * b4 * c1 * d0 * d4 +
          4 * pow(a0, 2) * b4 * c2 * d0 * d3 +
          4 * pow(a0, 2) * b4 * c3 * d0 * d2 -
          8 * pow(a0, 2) * b4 * c4 * d0 * d1 -
          2 * pow(a0, 2) * b5 * c0 * d1 * d3 -
          2 * pow(a0, 2) * b5 * c1 * d0 * d3 -
          2 * pow(a0, 2) * b5 * c3 * d0 * d1 +
          96 * pow(a1, 2) * b1 * c2 * d2 * d3 -
          32 * pow(a1, 2) * b2 * c1 * d2 * d3 -
          32 * pow(a1, 2) * b2 * c2 * d1 * d3 -
          32 * pow(a1, 2) * b2 * c3 * d1 * d2 -
          32 * pow(a1, 2) * b3 * c2 * d1 * d2 -
          8 * pow(a2, 2) * b0 * c3 * d1 * d3 -
          8 * pow(a2, 2) * b1 * c3 * d0 * d3 +
          8 * pow(a2, 2) * b3 * c0 * d1 * d3 +
          8 * pow(a2, 2) * b3 * c1 * d0 * d3 +
          8 * pow(a2, 2) * b3 * c3 * d0 * d1 +
          2 * pow(a3, 2) * b0 * c0 * d1 * d5 -
          4 * pow(a3, 2) * b0 * c0 * d2 * d4 +
          2 * pow(a3, 2) * b0 * c1 * d0 * d5 -
          4 * pow(a3, 2) * b0 * c2 * d0 * d4 -
          4 * pow(a3, 2) * b0 * c4 * d0 * d2 +
          2 * pow(a3, 2) * b0 * c5 * d0 * d1 -
          6 * pow(a3, 2) * b1 * c0 * d0 * d5 +
          4 * pow(a3, 2) * b2 * c0 * d0 * d4 +
          4 * pow(a3, 2) * b4 * c0 * d0 * d2 +
          2 * pow(a3, 2) * b5 * c0 * d0 * d1 +
          32 * pow(a4, 2) * b0 * c2 * d1 * d2 -
          32 * pow(a4, 2) * b1 * c2 * d0 * d2 +
          4 * pow(a5, 2) * b0 * c0 * d1 * d3 +
          4 * pow(a5, 2) * b0 * c1 * d0 * d3 +
          4 * pow(a5, 2) * b0 * c3 * d0 * d1 -
          4 * pow(a5, 2) * b1 * c0 * d0 * d3 -
          4 * pow(a5, 2) * b3 * c0 * d0 * d1 -
          8 * pow(a0, 2) * b1 * c2 * d2 * d5 -
          16 * pow(a0, 2) * b2 * c2 * d2 * d4 +
          8 * pow(a0, 2) * b5 * c2 * d1 * d2 +
          8 * pow(a1, 2) * b0 * c1 * d3 * d5 -
          16 * pow(a1, 2) * b0 * c2 * d3 * d4 +
          8 * pow(a1, 2) * b0 * c3 * d1 * d5 -
          16 * pow(a1, 2) * b0 * c3 * d2 * d4 -
          16 * pow(a1, 2) * b0 * c4 * d2 * d3 +
          8 * pow(a1, 2) * b0 * c5 * d1 * d3 -
          24 * pow(a1, 2) * b1 * c0 * d3 * d5 -
          24 * pow(a1, 2) * b1 * c3 * d0 * d5 -
          24 * pow(a1, 2) * b1 * c5 * d0 * d3 +
          16 * pow(a1, 2) * b2 * c0 * d3 * d4 +
          16 * pow(a1, 2) * b2 * c3 * d0 * d4 +
          16 * pow(a1, 2) * b2 * c4 * d0 * d3 +
          8 * pow(a1, 2) * b3 * c0 * d1 * d5 +
          8 * pow(a1, 2) * b3 * c1 * d0 * d5 +
          8 * pow(a1, 2) * b3 * c5 * d0 * d1 +
          8 * pow(a1, 2) * b5 * c0 * d1 * d3 +
          8 * pow(a1, 2) * b5 * c1 * d0 * d3 +
          8 * pow(a1, 2) * b5 * c3 * d0 * d1 -
          32 * pow(a2, 2) * b1 * c1 * d1 * d5 +
          64 * pow(a2, 2) * b1 * c1 * d2 * d4 +
          64 * pow(a2, 2) * b1 * c2 * d1 * d4 +
          64 * pow(a2, 2) * b1 * c4 * d1 * d2 -
          192 * pow(a2, 2) * b2 * c1 * d1 * d4 +
          64 * pow(a2, 2) * b4 * c1 * d1 * d2 -
          8 * pow(a4, 2) * b0 * c0 * d1 * d5 -
          8 * pow(a4, 2) * b0 * c1 * d0 * d5 -
          8 * pow(a4, 2) * b0 * c5 * d0 * d1 +
          16 * pow(a4, 2) * b1 * c0 * d0 * d5 -
          8 * pow(a5, 2) * b0 * c2 * d1 * d2 +
          8 * pow(a5, 2) * b1 * c2 * d0 * d2 +
          12 * pow(a0, 2) * b0 * c2 * d4 * d5 +
          12 * pow(a0, 2) * b0 * c4 * d2 * d5 -
          6 * pow(a0, 2) * b0 * c5 * d1 * d5 +
          12 * pow(a0, 2) * b0 * c5 * d2 * d4 +
          2 * pow(a0, 2) * b1 * c5 * d0 * d5 -
          4 * pow(a0, 2) * b2 * c0 * d4 * d5 -
          4 * pow(a0, 2) * b2 * c4 * d0 * d5 -
          4 * pow(a0, 2) * b2 * c5 * d0 * d4 -
          4 * pow(a0, 2) * b4 * c0 * d2 * d5 -
          4 * pow(a0, 2) * b4 * c2 * d0 * d5 -
          4 * pow(a0, 2) * b4 * c5 * d0 * d2 +
          2 * pow(a0, 2) * b5 * c0 * d1 * d5 -
          4 * pow(a0, 2) * b5 * c0 * d2 * d4 +
          2 * pow(a0, 2) * b5 * c1 * d0 * d5 -
          4 * pow(a0, 2) * b5 * c2 * d0 * d4 -
          4 * pow(a0, 2) * b5 * c4 * d0 * d2 +
          2 * pow(a0, 2) * b5 * c5 * d0 * d1 -
          96 * pow(a1, 2) * b1 * c2 * d2 * d5 +
          32 * pow(a1, 2) * b2 * c1 * d2 * d5 +
          32 * pow(a1, 2) * b2 * c2 * d1 * d5 -
          64 * pow(a1, 2) * b2 * c2 * d2 * d4 +
          32 * pow(a1, 2) * b2 * c5 * d1 * d2 +
          32 * pow(a1, 2) * b5 * c2 * d1 * d2 +
          8 * pow(a2, 2) * b0 * c1 * d3 * d5 -
          16 * pow(a2, 2) * b0 * c2 * d3 * d4 +
          8 * pow(a2, 2) * b0 * c3 * d1 * d5 -
          16 * pow(a2, 2) * b0 * c3 * d2 * d4 +
          32 * pow(a2, 2) * b0 * c4 * d1 * d4 -
          16 * pow(a2, 2) * b0 * c4 * d2 * d3 +
          8 * pow(a2, 2) * b0 * c5 * d1 * d3 -
          32 * pow(a2, 2) * b1 * c4 * d0 * d4 +
          48 * pow(a2, 2) * b2 * c0 * d3 * d4 +
          48 * pow(a2, 2) * b2 * c3 * d0 * d4 +
          48 * pow(a2, 2) * b2 * c4 * d0 * d3 -
          16 * pow(a2, 2) * b3 * c0 * d2 * d4 -
          16 * pow(a2, 2) * b3 * c2 * d0 * d4 -
          16 * pow(a2, 2) * b3 * c4 * d0 * d2 -
          16 * pow(a2, 2) * b4 * c0 * d2 * d3 -
          16 * pow(a2, 2) * b4 * c2 * d0 * d3 -
          16 * pow(a2, 2) * b4 * c3 * d0 * d2 -
          8 * pow(a2, 2) * b5 * c0 * d1 * d3 -
          8 * pow(a2, 2) * b5 * c1 * d0 * d3 -
          8 * pow(a2, 2) * b5 * c3 * d0 * d1 -
          2 * pow(a5, 2) * b0 * c0 * d1 * d5 +
          4 * pow(a5, 2) * b0 * c0 * d2 * d4 -
          2 * pow(a5, 2) * b0 * c1 * d0 * d5 +
          4 * pow(a5, 2) * b0 * c2 * d0 * d4 +
          4 * pow(a5, 2) * b0 * c4 * d0 * d2 -
          2 * pow(a5, 2) * b0 * c5 * d0 * d1 -
          2 * pow(a5, 2) * b1 * c0 * d0 * d5 +
          8 * pow(a5, 2) * b1 * c1 * d1 * d3 -
          4 * pow(a5, 2) * b2 * c0 * d0 * d4 -
          4 * pow(a5, 2) * b4 * c0 * d0 * d2 +
          6 * pow(a5, 2) * b5 * c0 * d0 * d1 -
          2 * pow(a0, 2) * b1 * c3 * d3 * d5 +
          8 * pow(a0, 2) * b1 * c4 * d3 * d4 -
          4 * pow(a0, 2) * b2 * c3 * d3 * d4 -
          2 * pow(a0, 2) * b3 * c1 * d3 * d5 +
          4 * pow(a0, 2) * b3 * c2 * d3 * d4 -
          2 * pow(a0, 2) * b3 * c3 * d1 * d5 +
          4 * pow(a0, 2) * b3 * c3 * d2 * d4 -
          8 * pow(a0, 2) * b3 * c4 * d1 * d4 +
          4 * pow(a0, 2) * b3 * c4 * d2 * d3 -
          2 * pow(a0, 2) * b3 * c5 * d1 * d3 -
          4 * pow(a0, 2) * b4 * c3 * d2 * d3 +
          6 * pow(a0, 2) * b5 * c3 * d1 * d3 +
          16 * pow(a1, 2) * b0 * c2 * d4 * d5 +
          16 * pow(a1, 2) * b0 * c4 * d2 * d5 -
          16 * pow(a1, 2) * b0 * c5 * d1 * d5 +
          16 * pow(a1, 2) * b0 * c5 * d2 * d4 +
          48 * pow(a1, 2) * b1 * c5 * d0 * d5 -
          16 * pow(a1, 2) * b4 * c0 * d2 * d5 -
          16 * pow(a1, 2) * b4 * c2 * d0 * d5 -
          16 * pow(a1, 2) * b4 * c5 * d0 * d2 -
          16 * pow(a1, 2) * b5 * c0 * d1 * d5 -
          16 * pow(a1, 2) * b5 * c1 * d0 * d5 -
          16 * pow(a1, 2) * b5 * c5 * d0 * d1 +
          32 * pow(a4, 2) * b1 * c2 * d2 * d3 -
          32 * pow(a4, 2) * b3 * c2 * d1 * d2 -
          6 * pow(a5, 2) * b0 * c3 * d1 * d3 +
          2 * pow(a5, 2) * b1 * c3 * d0 * d3 +
          2 * pow(a5, 2) * b3 * c0 * d1 * d3 +
          2 * pow(a5, 2) * b3 * c1 * d0 * d3 +
          2 * pow(a5, 2) * b3 * c3 * d0 * d1 -
          8 * pow(a2, 2) * b0 * c5 * d1 * d5 +
          8 * pow(a2, 2) * b1 * c5 * d0 * d5 -
          8 * pow(a3, 2) * b1 * c2 * d2 * d5 +
          8 * pow(a3, 2) * b2 * c1 * d2 * d5 +
          8 * pow(a3, 2) * b2 * c2 * d1 * d5 -
          16 * pow(a3, 2) * b2 * c2 * d2 * d4 +
          8 * pow(a3, 2) * b2 * c5 * d1 * d2 -
          8 * pow(a3, 2) * b5 * c2 * d1 * d2 -
          8 * pow(a4, 2) * b1 * c0 * d3 * d5 -
          8 * pow(a4, 2) * b1 * c3 * d0 * d5 -
          8 * pow(a4, 2) * b1 * c5 * d0 * d3 +
          8 * pow(a4, 2) * b3 * c0 * d1 * d5 +
          8 * pow(a4, 2) * b3 * c1 * d0 * d5 +
          8 * pow(a4, 2) * b3 * c5 * d0 * d1 -
          24 * pow(a5, 2) * b1 * c1 * d1 * d5 +
          16 * pow(a5, 2) * b1 * c1 * d2 * d4 +
          16 * pow(a5, 2) * b1 * c2 * d1 * d4 -
          8 * pow(a5, 2) * b1 * c2 * d2 * d3 +
          16 * pow(a5, 2) * b1 * c4 * d1 * d2 -
          16 * pow(a5, 2) * b2 * c1 * d1 * d4 +
          8 * pow(a5, 2) * b3 * c2 * d1 * d2 -
          16 * pow(a5, 2) * b4 * c1 * d1 * d2 +
          4 * pow(a0, 2) * b1 * c5 * d3 * d5 -
          8 * pow(a0, 2) * b3 * c2 * d4 * d5 -
          8 * pow(a0, 2) * b3 * c4 * d2 * d5 +
          4 * pow(a0, 2) * b3 * c5 * d1 * d5 -
          8 * pow(a0, 2) * b3 * c5 * d2 * d4 +
          8 * pow(a0, 2) * b4 * c1 * d4 * d5 +
          8 * pow(a0, 2) * b4 * c4 * d1 * d5 +
          8 * pow(a0, 2) * b4 * c5 * d1 * d4 -
          4 * pow(a0, 2) * b5 * c1 * d3 * d5 +
          8 * pow(a0, 2) * b5 * c2 * d3 * d4 -
          4 * pow(a0, 2) * b5 * c3 * d1 * d5 +
          8 * pow(a0, 2) * b5 * c3 * d2 * d4 -
          16 * pow(a0, 2) * b5 * c4 * d1 * d4 +
          8 * pow(a0, 2) * b5 * c4 * d2 * d3 -
          4 * pow(a0, 2) * b5 * c5 * d1 * d3 +
          8 * pow(a2, 2) * b1 * c3 * d3 * d5 +
          32 * pow(a2, 2) * b1 * c4 * d3 * d4 -
          48 * pow(a2, 2) * b2 * c3 * d3 * d4 -
          8 * pow(a2, 2) * b3 * c1 * d3 * d5 +
          16 * pow(a2, 2) * b3 * c2 * d3 * d4 -
          8 * pow(a2, 2) * b3 * c3 * d1 * d5 +
          16 * pow(a2, 2) * b3 * c3 * d2 * d4 -
          32 * pow(a2, 2) * b3 * c4 * d1 * d4 +
          16 * pow(a2, 2) * b3 * c4 * d2 * d3 -
          8 * pow(a2, 2) * b3 * c5 * d1 * d3 +
          16 * pow(a2, 2) * b4 * c3 * d2 * d3 +
          8 * pow(a2, 2) * b5 * c3 * d1 * d3 +
          4 * pow(a3, 2) * b0 * c2 * d4 * d5 +
          4 * pow(a3, 2) * b0 * c4 * d2 * d5 -
          2 * pow(a3, 2) * b0 * c5 * d1 * d5 +
          4 * pow(a3, 2) * b0 * c5 * d2 * d4 +
          6 * pow(a3, 2) * b1 * c5 * d0 * d5 -
          4 * pow(a3, 2) * b2 * c0 * d4 * d5 -
          4 * pow(a3, 2) * b2 * c4 * d0 * d5 -
          4 * pow(a3, 2) * b2 * c5 * d0 * d4 -
          4 * pow(a3, 2) * b4 * c0 * d2 * d5 -
          4 * pow(a3, 2) * b4 * c2 * d0 * d5 -
          4 * pow(a3, 2) * b4 * c5 * d0 * d2 -
          2 * pow(a3, 2) * b5 * c0 * d1 * d5 +
          4 * pow(a3, 2) * b5 * c0 * d2 * d4 -
          2 * pow(a3, 2) * b5 * c1 * d0 * d5 +
          4 * pow(a3, 2) * b5 * c2 * d0 * d4 +
          4 * pow(a3, 2) * b5 * c4 * d0 * d2 -
          2 * pow(a3, 2) * b5 * c5 * d0 * d1 +
          2 * pow(a5, 2) * b0 * c1 * d3 * d5 -
          4 * pow(a5, 2) * b0 * c2 * d3 * d4 +
          2 * pow(a5, 2) * b0 * c3 * d1 * d5 -
          4 * pow(a5, 2) * b0 * c3 * d2 * d4 +
          8 * pow(a5, 2) * b0 * c4 * d1 * d4 -
          4 * pow(a5, 2) * b0 * c4 * d2 * d3 +
          2 * pow(a5, 2) * b0 * c5 * d1 * d3 +
          2 * pow(a5, 2) * b1 * c0 * d3 * d5 +
          2 * pow(a5, 2) * b1 * c3 * d0 * d5 -
          8 * pow(a5, 2) * b1 * c4 * d0 * d4 +
          2 * pow(a5, 2) * b1 * c5 * d0 * d3 +
          4 * pow(a5, 2) * b2 * c0 * d3 * d4 +
          4 * pow(a5, 2) * b2 * c3 * d0 * d4 +
          4 * pow(a5, 2) * b2 * c4 * d0 * d3 +
          2 * pow(a5, 2) * b3 * c0 * d1 * d5 -
          4 * pow(a5, 2) * b3 * c0 * d2 * d4 +
          2 * pow(a5, 2) * b3 * c1 * d0 * d5 -
          4 * pow(a5, 2) * b3 * c2 * d0 * d4 -
          4 * pow(a5, 2) * b3 * c4 * d0 * d2 +
          2 * pow(a5, 2) * b3 * c5 * d0 * d1 +
          4 * pow(a5, 2) * b4 * c0 * d2 * d3 +
          4 * pow(a5, 2) * b4 * c2 * d0 * d3 +
          4 * pow(a5, 2) * b4 * c3 * d0 * d2 -
          6 * pow(a5, 2) * b5 * c0 * d1 * d3 -
          6 * pow(a5, 2) * b5 * c1 * d0 * d3 -
          6 * pow(a5, 2) * b5 * c3 * d0 * d1 +
          24 * pow(a1, 2) * b1 * c5 * d3 * d5 -
          16 * pow(a1, 2) * b2 * c3 * d4 * d5 -
          16 * pow(a1, 2) * b2 * c4 * d3 * d5 -
          16 * pow(a1, 2) * b2 * c5 * d3 * d4 -
          8 * pow(a1, 2) * b3 * c5 * d1 * d5 -
          8 * pow(a1, 2) * b5 * c1 * d3 * d5 +
          16 * pow(a1, 2) * b5 * c2 * d3 * d4 -
          8 * pow(a1, 2) * b5 * c3 * d1 * d5 +
          16 * pow(a1, 2) * b5 * c3 * d2 * d4 +
          16 * pow(a1, 2) * b5 * c4 * d2 * d3 -
          8 * pow(a1, 2) * b5 * c5 * d1 * d3 +
          8 * pow(a4, 2) * b0 * c5 * d1 * d5 -
          8 * pow(a4, 2) * b1 * c5 * d0 * d5 +
          4 * pow(a0, 2) * b2 * c5 * d4 * d5 +
          4 * pow(a0, 2) * b4 * c5 * d2 * d5 -
          4 * pow(a0, 2) * b5 * c2 * d4 * d5 -
          4 * pow(a0, 2) * b5 * c4 * d2 * d5 +
          2 * pow(a0, 2) * b5 * c5 * d1 * d5 -
          4 * pow(a0, 2) * b5 * c5 * d2 * d4 -
          8 * pow(a2, 2) * b1 * c5 * d3 * d5 +
          8 * pow(a2, 2) * b3 * c5 * d1 * d5 +
          16 * pow(a1, 2) * b2 * c5 * d4 * d5 +
          16 * pow(a1, 2) * b4 * c5 * d2 * d5 -
          16 * pow(a1, 2) * b5 * c2 * d4 * d5 -
          16 * pow(a1, 2) * b5 * c4 * d2 * d5 +
          24 * pow(a1, 2) * b5 * c5 * d1 * d5 -
          16 * pow(a1, 2) * b5 * c5 * d2 * d4 -
          2 * pow(a5, 2) * b1 * c3 * d3 * d5 +
          8 * pow(a5, 2) * b1 * c4 * d3 * d4 -
          4 * pow(a5, 2) * b2 * c3 * d3 * d4 -
          2 * pow(a5, 2) * b3 * c1 * d3 * d5 +
          4 * pow(a5, 2) * b3 * c2 * d3 * d4 -
          2 * pow(a5, 2) * b3 * c3 * d1 * d5 +
          4 * pow(a5, 2) * b3 * c3 * d2 * d4 -
          8 * pow(a5, 2) * b3 * c4 * d1 * d4 +
          4 * pow(a5, 2) * b3 * c4 * d2 * d3 -
          2 * pow(a5, 2) * b3 * c5 * d1 * d3 -
          4 * pow(a5, 2) * b4 * c3 * d2 * d3 +
          6 * pow(a5, 2) * b5 * c3 * d1 * d3 +
          8 * pow(a4, 2) * b1 * c5 * d3 * d5 -
          8 * pow(a4, 2) * b3 * c5 * d1 * d5 +
          4 * pow(a3, 2) * b2 * c5 * d4 * d5 +
          4 * pow(a3, 2) * b4 * c5 * d2 * d5 -
          4 * pow(a3, 2) * b5 * c2 * d4 * d5 -
          4 * pow(a3, 2) * b5 * c4 * d2 * d5 +
          2 * pow(a3, 2) * b5 * c5 * d1 * d5 -
          4 * pow(a3, 2) * b5 * c5 * d2 * d4 +
          16 * a0 * a1 * b0 * c1 * d1 * d3 - 16 * a0 * a1 * b1 * c0 * d1 * d3 -
          16 * a0 * a1 * b1 * c1 * d0 * d3 - 16 * a0 * a1 * b1 * c3 * d0 * d1 +
          16 * a0 * a1 * b3 * c1 * d0 * d1 + 16 * a0 * a3 * b1 * c1 * d0 * d1 +
          16 * a1 * a3 * b0 * c1 * d0 * d1 - 16 * a1 * a3 * b1 * c0 * d0 * d1 +
          4 * a0 * a1 * b0 * c3 * d0 * d3 - 4 * a0 * a1 * b3 * c0 * d0 * d3 +
          4 * a0 * a3 * b0 * c0 * d1 * d3 + 4 * a0 * a3 * b0 * c1 * d0 * d3 +
          4 * a0 * a3 * b0 * c3 * d0 * d1 - 4 * a0 * a3 * b1 * c0 * d0 * d3 -
          4 * a0 * a3 * b3 * c0 * d0 * d1 - 4 * a1 * a3 * b0 * c0 * d0 * d3 -
          16 * a0 * a1 * b0 * c1 * d1 * d5 + 16 * a0 * a1 * b0 * c2 * d2 * d3 +
          16 * a0 * a1 * b1 * c0 * d1 * d5 + 16 * a0 * a1 * b1 * c1 * d0 * d5 +
          16 * a0 * a1 * b1 * c5 * d0 * d1 + 16 * a0 * a1 * b2 * c0 * d1 * d4 -
          8 * a0 * a1 * b2 * c0 * d2 * d3 + 16 * a0 * a1 * b2 * c1 * d0 * d4 -
          8 * a0 * a1 * b2 * c2 * d0 * d3 - 8 * a0 * a1 * b2 * c3 * d0 * d2 +
          16 * a0 * a1 * b2 * c4 * d0 * d1 - 16 * a0 * a1 * b4 * c0 * d1 * d2 -
          16 * a0 * a1 * b4 * c1 * d0 * d2 - 16 * a0 * a1 * b4 * c2 * d0 * d1 -
          16 * a0 * a1 * b5 * c1 * d0 * d1 - 32 * a0 * a2 * b0 * c1 * d1 * d4 +
          16 * a0 * a2 * b1 * c0 * d1 * d4 - 8 * a0 * a2 * b1 * c0 * d2 * d3 +
          16 * a0 * a2 * b1 * c1 * d0 * d4 - 8 * a0 * a2 * b1 * c2 * d0 * d3 -
          8 * a0 * a2 * b1 * c3 * d0 * d2 + 16 * a0 * a2 * b1 * c4 * d0 * d1 +
          8 * a0 * a2 * b3 * c0 * d1 * d2 + 8 * a0 * a2 * b3 * c1 * d0 * d2 +
          8 * a0 * a2 * b3 * c2 * d0 * d1 - 16 * a0 * a3 * b0 * c2 * d1 * d2 +
          8 * a0 * a3 * b2 * c0 * d1 * d2 + 8 * a0 * a3 * b2 * c1 * d0 * d2 +
          8 * a0 * a3 * b2 * c2 * d0 * d1 + 32 * a0 * a4 * b0 * c1 * d1 * d2 -
          16 * a0 * a4 * b1 * c0 * d1 * d2 - 16 * a0 * a4 * b1 * c1 * d0 * d2 -
          16 * a0 * a4 * b1 * c2 * d0 * d1 - 16 * a0 * a5 * b1 * c1 * d0 * d1 +
          16 * a1 * a2 * b0 * c0 * d1 * d4 - 8 * a1 * a2 * b0 * c0 * d2 * d3 +
          16 * a1 * a2 * b0 * c1 * d0 * d4 - 8 * a1 * a2 * b0 * c2 * d0 * d3 -
          8 * a1 * a2 * b0 * c3 * d0 * d2 + 16 * a1 * a2 * b0 * c4 * d0 * d1 -
          32 * a1 * a2 * b1 * c0 * d0 * d4 + 16 * a1 * a2 * b2 * c0 * d0 * d3 -
          16 * a1 * a4 * b0 * c0 * d1 * d2 - 16 * a1 * a4 * b0 * c1 * d0 * d2 -
          16 * a1 * a4 * b0 * c2 * d0 * d1 + 32 * a1 * a4 * b1 * c0 * d0 * d2 -
          16 * a1 * a5 * b0 * c1 * d0 * d1 + 16 * a1 * a5 * b1 * c0 * d0 * d1 +
          8 * a2 * a3 * b0 * c0 * d1 * d2 + 8 * a2 * a3 * b0 * c1 * d0 * d2 +
          8 * a2 * a3 * b0 * c2 * d0 * d1 - 16 * a2 * a3 * b2 * c0 * d0 * d1 -
          4 * a0 * a1 * b0 * c0 * d3 * d5 - 4 * a0 * a1 * b0 * c3 * d0 * d5 -
          16 * a0 * a1 * b0 * c4 * d0 * d4 - 4 * a0 * a1 * b0 * c5 * d0 * d3 +
          4 * a0 * a1 * b3 * c0 * d0 * d5 + 16 * a0 * a1 * b4 * c0 * d0 * d4 +
          4 * a0 * a1 * b5 * c0 * d0 * d3 + 8 * a0 * a2 * b0 * c0 * d3 * d4 +
          8 * a0 * a2 * b0 * c3 * d0 * d4 + 8 * a0 * a2 * b0 * c4 * d0 * d3 -
          8 * a0 * a2 * b3 * c0 * d0 * d4 - 8 * a0 * a2 * b4 * c0 * d0 * d3 -
          4 * a0 * a3 * b0 * c0 * d1 * d5 + 8 * a0 * a3 * b0 * c0 * d2 * d4 -
          4 * a0 * a3 * b0 * c1 * d0 * d5 + 8 * a0 * a3 * b0 * c2 * d0 * d4 +
          8 * a0 * a3 * b0 * c4 * d0 * d2 - 4 * a0 * a3 * b0 * c5 * d0 * d1 +
          4 * a0 * a3 * b1 * c0 * d0 * d5 - 8 * a0 * a3 * b2 * c0 * d0 * d4 -
          8 * a0 * a3 * b4 * c0 * d0 * d2 + 4 * a0 * a3 * b5 * c0 * d0 * d1 -
          16 * a0 * a4 * b0 * c0 * d1 * d4 + 8 * a0 * a4 * b0 * c0 * d2 * d3 -
          16 * a0 * a4 * b0 * c1 * d0 * d4 + 8 * a0 * a4 * b0 * c2 * d0 * d3 +
          8 * a0 * a4 * b0 * c3 * d0 * d2 - 16 * a0 * a4 * b0 * c4 * d0 * d1 +
          16 * a0 * a4 * b1 * c0 * d0 * d4 - 8 * a0 * a4 * b2 * c0 * d0 * d3 -
          8 * a0 * a4 * b3 * c0 * d0 * d2 + 16 * a0 * a4 * b4 * c0 * d0 * d1 -
          4 * a0 * a5 * b0 * c0 * d1 * d3 - 4 * a0 * a5 * b0 * c1 * d0 * d3 -
          4 * a0 * a5 * b0 * c3 * d0 * d1 + 4 * a0 * a5 * b1 * c0 * d0 * d3 +
          4 * a0 * a5 * b3 * c0 * d0 * d1 + 4 * a1 * a3 * b0 * c0 * d0 * d5 +
          16 * a1 * a4 * b0 * c0 * d0 * d4 + 4 * a1 * a5 * b0 * c0 * d0 * d3 -
          8 * a2 * a3 * b0 * c0 * d0 * d4 - 8 * a2 * a4 * b0 * c0 * d0 * d3 -
          8 * a3 * a4 * b0 * c0 * d0 * d2 + 4 * a3 * a5 * b0 * c0 * d0 * d1 -
          16 * a0 * a1 * b0 * c2 * d2 * d5 + 8 * a0 * a1 * b2 * c0 * d2 * d5 +
          8 * a0 * a1 * b2 * c2 * d0 * d5 + 8 * a0 * a1 * b2 * c5 * d0 * d2 -
          32 * a0 * a2 * b0 * c2 * d2 * d4 + 8 * a0 * a2 * b1 * c0 * d2 * d5 +
          8 * a0 * a2 * b1 * c2 * d0 * d5 + 8 * a0 * a2 * b1 * c5 * d0 * d2 +
          32 * a0 * a2 * b2 * c0 * d2 * d4 + 32 * a0 * a2 * b2 * c2 * d0 * d4 +
          32 * a0 * a2 * b2 * c4 * d0 * d2 - 32 * a0 * a2 * b4 * c2 * d0 * d2 -
          8 * a0 * a2 * b5 * c0 * d1 * d2 - 8 * a0 * a2 * b5 * c1 * d0 * d2 -
          8 * a0 * a2 * b5 * c2 * d0 * d1 - 32 * a0 * a4 * b2 * c2 * d0 * d2 +
          16 * a0 * a5 * b0 * c2 * d1 * d2 - 8 * a0 * a5 * b2 * c0 * d1 * d2 -
          8 * a0 * a5 * b2 * c1 * d0 * d2 - 8 * a0 * a5 * b2 * c2 * d0 * d1 +
          8 * a1 * a2 * b0 * c0 * d2 * d5 + 8 * a1 * a2 * b0 * c2 * d0 * d5 +
          8 * a1 * a2 * b0 * c5 * d0 * d2 - 64 * a1 * a2 * b1 * c1 * d2 * d3 -
          64 * a1 * a2 * b1 * c2 * d1 * d3 - 64 * a1 * a2 * b1 * c3 * d1 * d2 -
          16 * a1 * a2 * b2 * c0 * d0 * d5 + 64 * a1 * a2 * b2 * c1 * d1 * d3 +
          64 * a1 * a2 * b3 * c1 * d1 * d2 - 64 * a1 * a3 * b1 * c2 * d1 * d2 +
          64 * a1 * a3 * b2 * c1 * d1 * d2 + 64 * a2 * a3 * b1 * c1 * d1 * d2 -
          32 * a2 * a4 * b0 * c2 * d0 * d2 + 32 * a2 * a4 * b2 * c0 * d0 * d2 -
          8 * a2 * a5 * b0 * c0 * d1 * d2 - 8 * a2 * a5 * b0 * c1 * d0 * d2 -
          8 * a2 * a5 * b0 * c2 * d0 * d1 + 16 * a2 * a5 * b2 * c0 * d0 * d1 +
          4 * a0 * a1 * b0 * c5 * d0 * d5 + 16 * a0 * a1 * b1 * c1 * d3 * d5 -
          32 * a0 * a1 * b1 * c2 * d3 * d4 + 16 * a0 * a1 * b1 * c3 * d1 * d5 -
          32 * a0 * a1 * b1 * c3 * d2 * d4 - 32 * a0 * a1 * b1 * c4 * d2 * d3 +
          16 * a0 * a1 * b1 * c5 * d1 * d3 + 16 * a0 * a1 * b2 * c3 * d2 * d3 -
          16 * a0 * a1 * b3 * c1 * d1 * d5 + 16 * a0 * a1 * b3 * c1 * d2 * d4 +
          16 * a0 * a1 * b3 * c2 * d1 * d4 - 16 * a0 * a1 * b3 * c2 * d2 * d3 +
          16 * a0 * a1 * b3 * c4 * d1 * d2 + 16 * a0 * a1 * b4 * c1 * d2 * d3 +
          16 * a0 * a1 * b4 * c2 * d1 * d3 + 16 * a0 * a1 * b4 * c3 * d1 * d2 -
          4 * a0 * a1 * b5 * c0 * d0 * d5 - 16 * a0 * a1 * b5 * c1 * d1 * d3 -
          8 * a0 * a2 * b0 * c0 * d4 * d5 - 8 * a0 * a2 * b0 * c4 * d0 * d5 -
          8 * a0 * a2 * b0 * c5 * d0 * d4 + 16 * a0 * a2 * b1 * c3 * d2 * d3 -
          16 * a0 * a2 * b2 * c3 * d1 * d3 + 8 * a0 * a2 * b4 * c0 * d0 * d5 +
          8 * a0 * a2 * b5 * c0 * d0 * d4 - 16 * a0 * a3 * b1 * c1 * d1 * d5 +
          16 * a0 * a3 * b1 * c1 * d2 * d4 + 16 * a0 * a3 * b1 * c2 * d1 * d4 -
          16 * a0 * a3 * b1 * c2 * d2 * d3 + 16 * a0 * a3 * b1 * c4 * d1 * d2 +
          16 * a0 * a3 * b3 * c2 * d1 * d2 - 32 * a0 * a3 * b4 * c1 * d1 * d2 -
          8 * a0 * a4 * b0 * c0 * d2 * d5 - 8 * a0 * a4 * b0 * c2 * d0 * d5 -
          8 * a0 * a4 * b0 * c5 * d0 * d2 + 16 * a0 * a4 * b1 * c1 * d2 * d3 +
          16 * a0 * a4 * b1 * c2 * d1 * d3 + 16 * a0 * a4 * b1 * c3 * d1 * d2 +
          8 * a0 * a4 * b2 * c0 * d0 * d5 - 32 * a0 * a4 * b3 * c1 * d1 * d2 +
          8 * a0 * a4 * b5 * c0 * d0 * d2 + 4 * a0 * a5 * b0 * c0 * d1 * d5 -
          8 * a0 * a5 * b0 * c0 * d2 * d4 + 4 * a0 * a5 * b0 * c1 * d0 * d5 -
          8 * a0 * a5 * b0 * c2 * d0 * d4 - 8 * a0 * a5 * b0 * c4 * d0 * d2 +
          4 * a0 * a5 * b0 * c5 * d0 * d1 - 4 * a0 * a5 * b1 * c0 * d0 * d5 -
          16 * a0 * a5 * b1 * c1 * d1 * d3 + 8 * a0 * a5 * b2 * c0 * d0 * d4 +
          8 * a0 * a5 * b4 * c0 * d0 * d2 - 4 * a0 * a5 * b5 * c0 * d0 * d1 +
          16 * a1 * a2 * b0 * c3 * d2 * d3 + 32 * a1 * a2 * b1 * c0 * d3 * d4 +
          32 * a1 * a2 * b1 * c3 * d0 * d4 + 32 * a1 * a2 * b1 * c4 * d0 * d3 -
          16 * a1 * a2 * b2 * c3 * d0 * d3 - 16 * a1 * a2 * b3 * c0 * d1 * d4 -
          16 * a1 * a2 * b3 * c1 * d0 * d4 - 16 * a1 * a2 * b3 * c4 * d0 * d1 -
          16 * a1 * a2 * b4 * c0 * d1 * d3 - 16 * a1 * a2 * b4 * c1 * d0 * d3 -
          16 * a1 * a2 * b4 * c3 * d0 * d1 - 16 * a1 * a3 * b0 * c1 * d1 * d5 +
          16 * a1 * a3 * b0 * c1 * d2 * d4 + 16 * a1 * a3 * b0 * c2 * d1 * d4 -
          16 * a1 * a3 * b0 * c2 * d2 * d3 + 16 * a1 * a3 * b0 * c4 * d1 * d2 +
          16 * a1 * a3 * b1 * c0 * d1 * d5 + 16 * a1 * a3 * b1 * c1 * d0 * d5 +
          16 * a1 * a3 * b1 * c5 * d0 * d1 - 16 * a1 * a3 * b2 * c0 * d1 * d4 -
          16 * a1 * a3 * b2 * c1 * d0 * d4 - 16 * a1 * a3 * b2 * c4 * d0 * d1 +
          16 * a1 * a3 * b3 * c2 * d0 * d2 - 16 * a1 * a3 * b5 * c1 * d0 * d1 +
          16 * a1 * a4 * b0 * c1 * d2 * d3 + 16 * a1 * a4 * b0 * c2 * d1 * d3 +
          16 * a1 * a4 * b0 * c3 * d1 * d2 - 16 * a1 * a4 * b2 * c0 * d1 * d3 -
          16 * a1 * a4 * b2 * c1 * d0 * d3 - 16 * a1 * a4 * b2 * c3 * d0 * d1 -
          4 * a1 * a5 * b0 * c0 * d0 * d5 - 16 * a1 * a5 * b0 * c1 * d1 * d3 +
          16 * a1 * a5 * b1 * c0 * d1 * d3 + 16 * a1 * a5 * b1 * c1 * d0 * d3 +
          16 * a1 * a5 * b1 * c3 * d0 * d1 - 16 * a1 * a5 * b3 * c1 * d0 * d1 -
          16 * a2 * a3 * b1 * c0 * d1 * d4 - 16 * a2 * a3 * b1 * c1 * d0 * d4 -
          16 * a2 * a3 * b1 * c4 * d0 * d1 + 16 * a2 * a3 * b2 * c0 * d1 * d3 +
          16 * a2 * a3 * b2 * c1 * d0 * d3 + 16 * a2 * a3 * b2 * c3 * d0 * d1 -
          16 * a2 * a3 * b3 * c0 * d1 * d2 - 16 * a2 * a3 * b3 * c1 * d0 * d2 -
          16 * a2 * a3 * b3 * c2 * d0 * d1 + 32 * a2 * a3 * b4 * c1 * d0 * d1 +
          8 * a2 * a4 * b0 * c0 * d0 * d5 - 16 * a2 * a4 * b1 * c0 * d1 * d3 -
          16 * a2 * a4 * b1 * c1 * d0 * d3 - 16 * a2 * a4 * b1 * c3 * d0 * d1 +
          32 * a2 * a4 * b3 * c1 * d0 * d1 + 8 * a2 * a5 * b0 * c0 * d0 * d4 -
          32 * a3 * a4 * b0 * c1 * d1 * d2 + 32 * a3 * a4 * b2 * c1 * d0 * d1 -
          16 * a3 * a5 * b1 * c1 * d0 * d1 + 8 * a4 * a5 * b0 * c0 * d0 * d2 -
          4 * a0 * a1 * b0 * c3 * d3 * d5 + 16 * a0 * a1 * b0 * c4 * d3 * d4 +
          4 * a0 * a1 * b3 * c0 * d3 * d5 + 4 * a0 * a1 * b3 * c3 * d0 * d5 +
          4 * a0 * a1 * b3 * c5 * d0 * d3 - 8 * a0 * a1 * b4 * c0 * d3 * d4 -
          8 * a0 * a1 * b4 * c3 * d0 * d4 - 8 * a0 * a1 * b4 * c4 * d0 * d3 -
          4 * a0 * a1 * b5 * c3 * d0 * d3 - 8 * a0 * a2 * b0 * c3 * d3 * d4 +
          8 * a0 * a2 * b4 * c3 * d0 * d3 - 4 * a0 * a3 * b0 * c1 * d3 * d5 +
          8 * a0 * a3 * b0 * c2 * d3 * d4 - 4 * a0 * a3 * b0 * c3 * d1 * d5 +
          8 * a0 * a3 * b0 * c3 * d2 * d4 - 16 * a0 * a3 * b0 * c4 * d1 * d4 +
          8 * a0 * a3 * b0 * c4 * d2 * d3 - 4 * a0 * a3 * b0 * c5 * d1 * d3 +
          4 * a0 * a3 * b1 * c0 * d3 * d5 + 4 * a0 * a3 * b1 * c3 * d0 * d5 +
          4 * a0 * a3 * b1 * c5 * d0 * d3 + 4 * a0 * a3 * b3 * c0 * d1 * d5 -
          8 * a0 * a3 * b3 * c0 * d2 * d4 + 4 * a0 * a3 * b3 * c1 * d0 * d5 -
          8 * a0 * a3 * b3 * c2 * d0 * d4 - 8 * a0 * a3 * b3 * c4 * d0 * d2 +
          4 * a0 * a3 * b3 * c5 * d0 * d1 + 8 * a0 * a3 * b4 * c0 * d1 * d4 +
          8 * a0 * a3 * b4 * c1 * d0 * d4 + 8 * a0 * a3 * b4 * c4 * d0 * d1 -
          4 * a0 * a3 * b5 * c0 * d1 * d3 - 4 * a0 * a3 * b5 * c1 * d0 * d3 -
          4 * a0 * a3 * b5 * c3 * d0 * d1 - 8 * a0 * a4 * b0 * c3 * d2 * d3 -
          8 * a0 * a4 * b1 * c0 * d3 * d4 - 8 * a0 * a4 * b1 * c3 * d0 * d4 -
          8 * a0 * a4 * b1 * c4 * d0 * d3 + 8 * a0 * a4 * b2 * c3 * d0 * d3 +
          8 * a0 * a4 * b3 * c0 * d1 * d4 + 8 * a0 * a4 * b3 * c1 * d0 * d4 +
          8 * a0 * a4 * b3 * c4 * d0 * d1 + 12 * a0 * a5 * b0 * c3 * d1 * d3 -
          4 * a0 * a5 * b1 * c3 * d0 * d3 - 4 * a0 * a5 * b3 * c0 * d1 * d3 -
          4 * a0 * a5 * b3 * c1 * d0 * d3 - 4 * a0 * a5 * b3 * c3 * d0 * d1 +
          64 * a1 * a2 * b1 * c1 * d2 * d5 + 64 * a1 * a2 * b1 * c2 * d1 * d5 -
          128 * a1 * a2 * b1 * c2 * d2 * d4 + 64 * a1 * a2 * b1 * c5 * d1 * d2 -
          64 * a1 * a2 * b2 * c1 * d1 * d5 + 128 * a1 * a2 * b2 * c1 * d2 * d4 +
          128 * a1 * a2 * b2 * c2 * d1 * d4 +
          128 * a1 * a2 * b2 * c4 * d1 * d2 -
          128 * a1 * a2 * b4 * c2 * d1 * d2 - 64 * a1 * a2 * b5 * c1 * d1 * d2 +
          4 * a1 * a3 * b0 * c0 * d3 * d5 + 4 * a1 * a3 * b0 * c3 * d0 * d5 +
          4 * a1 * a3 * b0 * c5 * d0 * d3 - 12 * a1 * a3 * b3 * c0 * d0 * d5 +
          4 * a1 * a3 * b5 * c0 * d0 * d3 - 8 * a1 * a4 * b0 * c0 * d3 * d4 -
          8 * a1 * a4 * b0 * c3 * d0 * d4 - 8 * a1 * a4 * b0 * c4 * d0 * d3 -
          128 * a1 * a4 * b2 * c2 * d1 * d2 + 16 * a1 * a4 * b4 * c0 * d0 * d3 -
          4 * a1 * a5 * b0 * c3 * d0 * d3 + 64 * a1 * a5 * b1 * c2 * d1 * d2 -
          64 * a1 * a5 * b2 * c1 * d1 * d2 + 4 * a1 * a5 * b3 * c0 * d0 * d3 +
          8 * a2 * a3 * b3 * c0 * d0 * d4 - 8 * a2 * a3 * b4 * c0 * d0 * d3 +
          8 * a2 * a4 * b0 * c3 * d0 * d3 - 128 * a2 * a4 * b1 * c2 * d1 * d2 +
          128 * a2 * a4 * b2 * c1 * d1 * d2 - 8 * a2 * a4 * b3 * c0 * d0 * d3 -
          64 * a2 * a5 * b1 * c1 * d1 * d2 + 8 * a3 * a4 * b0 * c0 * d1 * d4 +
          8 * a3 * a4 * b0 * c1 * d0 * d4 + 8 * a3 * a4 * b0 * c4 * d0 * d1 -
          8 * a3 * a4 * b2 * c0 * d0 * d3 + 8 * a3 * a4 * b3 * c0 * d0 * d2 -
          16 * a3 * a4 * b4 * c0 * d0 * d1 - 4 * a3 * a5 * b0 * c0 * d1 * d3 -
          4 * a3 * a5 * b0 * c1 * d0 * d3 - 4 * a3 * a5 * b0 * c3 * d0 * d1 +
          4 * a3 * a5 * b1 * c0 * d0 * d3 + 4 * a3 * a5 * b3 * c0 * d0 * d1 +
          32 * a0 * a1 * b1 * c2 * d4 * d5 + 32 * a0 * a1 * b1 * c4 * d2 * d5 -
          32 * a0 * a1 * b1 * c5 * d1 * d5 + 32 * a0 * a1 * b1 * c5 * d2 * d4 -
          16 * a0 * a1 * b2 * c1 * d4 * d5 - 8 * a0 * a1 * b2 * c2 * d3 * d5 -
          8 * a0 * a1 * b2 * c3 * d2 * d5 - 16 * a0 * a1 * b2 * c4 * d1 * d5 -
          16 * a0 * a1 * b2 * c5 * d1 * d4 - 8 * a0 * a1 * b2 * c5 * d2 * d3 +
          16 * a0 * a1 * b3 * c2 * d2 * d5 + 32 * a0 * a1 * b5 * c1 * d1 * d5 -
          16 * a0 * a1 * b5 * c1 * d2 * d4 - 16 * a0 * a1 * b5 * c2 * d1 * d4 -
          16 * a0 * a1 * b5 * c4 * d1 * d2 - 16 * a0 * a2 * b1 * c1 * d4 * d5 -
          8 * a0 * a2 * b1 * c2 * d3 * d5 - 8 * a0 * a2 * b1 * c3 * d2 * d5 -
          16 * a0 * a2 * b1 * c4 * d1 * d5 - 16 * a0 * a2 * b1 * c5 * d1 * d4 -
          8 * a0 * a2 * b1 * c5 * d2 * d3 + 16 * a0 * a2 * b2 * c1 * d3 * d5 -
          32 * a0 * a2 * b2 * c2 * d3 * d4 + 16 * a0 * a2 * b2 * c3 * d1 * d5 -
          32 * a0 * a2 * b2 * c3 * d2 * d4 + 64 * a0 * a2 * b2 * c4 * d1 * d4 -
          32 * a0 * a2 * b2 * c4 * d2 * d3 + 16 * a0 * a2 * b2 * c5 * d1 * d3 -
          8 * a0 * a2 * b3 * c1 * d2 * d5 - 8 * a0 * a2 * b3 * c2 * d1 * d5 +
          32 * a0 * a2 * b3 * c2 * d2 * d4 - 8 * a0 * a2 * b3 * c5 * d1 * d2 -
          32 * a0 * a2 * b4 * c1 * d2 * d4 - 32 * a0 * a2 * b4 * c2 * d1 * d4 +
          32 * a0 * a2 * b4 * c2 * d2 * d3 - 32 * a0 * a2 * b4 * c4 * d1 * d2 +
          32 * a0 * a2 * b5 * c1 * d1 * d4 + 16 * a0 * a3 * b1 * c2 * d2 * d5 -
          8 * a0 * a3 * b2 * c1 * d2 * d5 - 8 * a0 * a3 * b2 * c2 * d1 * d5 +
          32 * a0 * a3 * b2 * c2 * d2 * d4 - 8 * a0 * a3 * b2 * c5 * d1 * d2 -
          32 * a0 * a4 * b2 * c1 * d2 * d4 - 32 * a0 * a4 * b2 * c2 * d1 * d4 +
          32 * a0 * a4 * b2 * c2 * d2 * d3 - 32 * a0 * a4 * b2 * c4 * d1 * d2 +
          64 * a0 * a4 * b4 * c2 * d1 * d2 + 32 * a0 * a5 * b1 * c1 * d1 * d5 -
          16 * a0 * a5 * b1 * c1 * d2 * d4 - 16 * a0 * a5 * b1 * c2 * d1 * d4 -
          16 * a0 * a5 * b1 * c4 * d1 * d2 + 32 * a0 * a5 * b2 * c1 * d1 * d4 -
          16 * a1 * a2 * b0 * c1 * d4 * d5 - 8 * a1 * a2 * b0 * c2 * d3 * d5 -
          8 * a1 * a2 * b0 * c3 * d2 * d5 - 16 * a1 * a2 * b0 * c4 * d1 * d5 -
          16 * a1 * a2 * b0 * c5 * d1 * d4 - 8 * a1 * a2 * b0 * c5 * d2 * d3 -
          64 * a1 * a2 * b2 * c4 * d0 * d4 + 16 * a1 * a2 * b4 * c0 * d1 * d5 +
          32 * a1 * a2 * b4 * c0 * d2 * d4 + 16 * a1 * a2 * b4 * c1 * d0 * d5 +
          32 * a1 * a2 * b4 * c2 * d0 * d4 + 32 * a1 * a2 * b4 * c4 * d0 * d2 +
          16 * a1 * a2 * b4 * c5 * d0 * d1 + 8 * a1 * a2 * b5 * c0 * d2 * d3 +
          8 * a1 * a2 * b5 * c2 * d0 * d3 + 8 * a1 * a2 * b5 * c3 * d0 * d2 +
          16 * a1 * a3 * b0 * c2 * d2 * d5 - 16 * a1 * a3 * b5 * c2 * d0 * d2 -
          32 * a1 * a4 * b1 * c0 * d2 * d5 - 32 * a1 * a4 * b1 * c2 * d0 * d5 -
          32 * a1 * a4 * b1 * c5 * d0 * d2 + 16 * a1 * a4 * b2 * c0 * d1 * d5 +
          32 * a1 * a4 * b2 * c0 * d2 * d4 + 16 * a1 * a4 * b2 * c1 * d0 * d5 +
          32 * a1 * a4 * b2 * c2 * d0 * d4 + 32 * a1 * a4 * b2 * c4 * d0 * d2 +
          16 * a1 * a4 * b2 * c5 * d0 * d1 - 64 * a1 * a4 * b4 * c2 * d0 * d2 +
          16 * a1 * a4 * b5 * c0 * d1 * d2 + 16 * a1 * a4 * b5 * c1 * d0 * d2 +
          16 * a1 * a4 * b5 * c2 * d0 * d1 + 32 * a1 * a5 * b0 * c1 * d1 * d5 -
          16 * a1 * a5 * b0 * c1 * d2 * d4 - 16 * a1 * a5 * b0 * c2 * d1 * d4 -
          16 * a1 * a5 * b0 * c4 * d1 * d2 - 32 * a1 * a5 * b1 * c0 * d1 * d5 -
          32 * a1 * a5 * b1 * c1 * d0 * d5 - 32 * a1 * a5 * b1 * c5 * d0 * d1 +
          8 * a1 * a5 * b2 * c0 * d2 * d3 + 8 * a1 * a5 * b2 * c2 * d0 * d3 +
          8 * a1 * a5 * b2 * c3 * d0 * d2 - 16 * a1 * a5 * b3 * c2 * d0 * d2 +
          16 * a1 * a5 * b4 * c0 * d1 * d2 + 16 * a1 * a5 * b4 * c1 * d0 * d2 +
          16 * a1 * a5 * b4 * c2 * d0 * d1 + 32 * a1 * a5 * b5 * c1 * d0 * d1 -
          8 * a2 * a3 * b0 * c1 * d2 * d5 - 8 * a2 * a3 * b0 * c2 * d1 * d5 +
          32 * a2 * a3 * b0 * c2 * d2 * d4 - 8 * a2 * a3 * b0 * c5 * d1 * d2 -
          32 * a2 * a3 * b2 * c0 * d2 * d4 - 32 * a2 * a3 * b2 * c2 * d0 * d4 -
          32 * a2 * a3 * b2 * c4 * d0 * d2 + 32 * a2 * a3 * b4 * c2 * d0 * d2 +
          8 * a2 * a3 * b5 * c0 * d1 * d2 + 8 * a2 * a3 * b5 * c1 * d0 * d2 +
          8 * a2 * a3 * b5 * c2 * d0 * d1 - 32 * a2 * a4 * b0 * c1 * d2 * d4 -
          32 * a2 * a4 * b0 * c2 * d1 * d4 + 32 * a2 * a4 * b0 * c2 * d2 * d3 -
          32 * a2 * a4 * b0 * c4 * d1 * d2 + 16 * a2 * a4 * b1 * c0 * d1 * d5 +
          32 * a2 * a4 * b1 * c0 * d2 * d4 + 16 * a2 * a4 * b1 * c1 * d0 * d5 +
          32 * a2 * a4 * b1 * c2 * d0 * d4 + 32 * a2 * a4 * b1 * c4 * d0 * d2 +
          16 * a2 * a4 * b1 * c5 * d0 * d1 - 32 * a2 * a4 * b2 * c0 * d2 * d3 -
          32 * a2 * a4 * b2 * c2 * d0 * d3 - 32 * a2 * a4 * b2 * c3 * d0 * d2 +
          32 * a2 * a4 * b3 * c2 * d0 * d2 - 32 * a2 * a4 * b5 * c1 * d0 * d1 +
          32 * a2 * a5 * b0 * c1 * d1 * d4 + 8 * a2 * a5 * b1 * c0 * d2 * d3 +
          8 * a2 * a5 * b1 * c2 * d0 * d3 + 8 * a2 * a5 * b1 * c3 * d0 * d2 -
          16 * a2 * a5 * b2 * c0 * d1 * d3 - 16 * a2 * a5 * b2 * c1 * d0 * d3 -
          16 * a2 * a5 * b2 * c3 * d0 * d1 + 8 * a2 * a5 * b3 * c0 * d1 * d2 +
          8 * a2 * a5 * b3 * c1 * d0 * d2 + 8 * a2 * a5 * b3 * c2 * d0 * d1 -
          32 * a2 * a5 * b4 * c1 * d0 * d1 + 32 * a3 * a4 * b2 * c2 * d0 * d2 -
          16 * a3 * a5 * b1 * c2 * d0 * d2 + 8 * a3 * a5 * b2 * c0 * d1 * d2 +
          8 * a3 * a5 * b2 * c1 * d0 * d2 + 8 * a3 * a5 * b2 * c2 * d0 * d1 +
          16 * a4 * a5 * b1 * c0 * d1 * d2 + 16 * a4 * a5 * b1 * c1 * d0 * d2 +
          16 * a4 * a5 * b1 * c2 * d0 * d1 - 32 * a4 * a5 * b2 * c1 * d0 * d1 +
          8 * a0 * a1 * b0 * c5 * d3 * d5 - 8 * a0 * a1 * b3 * c5 * d0 * d5 -
          8 * a0 * a1 * b4 * c0 * d4 * d5 - 8 * a0 * a1 * b4 * c4 * d0 * d5 -
          8 * a0 * a1 * b4 * c5 * d0 * d4 + 16 * a0 * a1 * b5 * c4 * d0 * d4 +
          8 * a0 * a2 * b3 * c0 * d4 * d5 + 8 * a0 * a2 * b3 * c4 * d0 * d5 +
          8 * a0 * a2 * b3 * c5 * d0 * d4 - 8 * a0 * a2 * b5 * c0 * d3 * d4 -
          8 * a0 * a2 * b5 * c3 * d0 * d4 - 8 * a0 * a2 * b5 * c4 * d0 * d3 -
          16 * a0 * a3 * b0 * c2 * d4 * d5 - 16 * a0 * a3 * b0 * c4 * d2 * d5 +
          8 * a0 * a3 * b0 * c5 * d1 * d5 - 16 * a0 * a3 * b0 * c5 * d2 * d4 -
          8 * a0 * a3 * b1 * c5 * d0 * d5 + 8 * a0 * a3 * b2 * c0 * d4 * d5 +
          8 * a0 * a3 * b2 * c4 * d0 * d5 + 8 * a0 * a3 * b2 * c5 * d0 * d4 +
          8 * a0 * a3 * b4 * c0 * d2 * d5 + 8 * a0 * a3 * b4 * c2 * d0 * d5 +
          8 * a0 * a3 * b4 * c5 * d0 * d2 + 16 * a0 * a4 * b0 * c1 * d4 * d5 +
          16 * a0 * a4 * b0 * c4 * d1 * d5 + 16 * a0 * a4 * b0 * c5 * d1 * d4 -
          8 * a0 * a4 * b1 * c0 * d4 * d5 - 8 * a0 * a4 * b1 * c4 * d0 * d5 -
          8 * a0 * a4 * b1 * c5 * d0 * d4 + 8 * a0 * a4 * b3 * c0 * d2 * d5 +
          8 * a0 * a4 * b3 * c2 * d0 * d5 + 8 * a0 * a4 * b3 * c5 * d0 * d2 -
          16 * a0 * a4 * b4 * c0 * d1 * d5 - 16 * a0 * a4 * b4 * c1 * d0 * d5 -
          16 * a0 * a4 * b4 * c5 * d0 * d1 + 8 * a0 * a4 * b5 * c0 * d1 * d4 -
          8 * a0 * a4 * b5 * c0 * d2 * d3 + 8 * a0 * a4 * b5 * c1 * d0 * d4 -
          8 * a0 * a4 * b5 * c2 * d0 * d3 - 8 * a0 * a4 * b5 * c3 * d0 * d2 +
          8 * a0 * a4 * b5 * c4 * d0 * d1 - 8 * a0 * a5 * b0 * c1 * d3 * d5 +
          16 * a0 * a5 * b0 * c2 * d3 * d4 - 8 * a0 * a5 * b0 * c3 * d1 * d5 +
          16 * a0 * a5 * b0 * c3 * d2 * d4 - 32 * a0 * a5 * b0 * c4 * d1 * d4 +
          16 * a0 * a5 * b0 * c4 * d2 * d3 - 8 * a0 * a5 * b0 * c5 * d1 * d3 +
          16 * a0 * a5 * b1 * c4 * d0 * d4 - 8 * a0 * a5 * b2 * c0 * d3 * d4 -
          8 * a0 * a5 * b2 * c3 * d0 * d4 - 8 * a0 * a5 * b2 * c4 * d0 * d3 +
          8 * a0 * a5 * b4 * c0 * d1 * d4 - 8 * a0 * a5 * b4 * c0 * d2 * d3 +
          8 * a0 * a5 * b4 * c1 * d0 * d4 - 8 * a0 * a5 * b4 * c2 * d0 * d3 -
          8 * a0 * a5 * b4 * c3 * d0 * d2 + 8 * a0 * a5 * b4 * c4 * d0 * d1 +
          8 * a0 * a5 * b5 * c0 * d1 * d3 + 8 * a0 * a5 * b5 * c1 * d0 * d3 +
          8 * a0 * a5 * b5 * c3 * d0 * d1 - 8 * a1 * a3 * b0 * c5 * d0 * d5 +
          8 * a1 * a3 * b5 * c0 * d0 * d5 - 8 * a1 * a4 * b0 * c0 * d4 * d5 -
          8 * a1 * a4 * b0 * c4 * d0 * d5 - 8 * a1 * a4 * b0 * c5 * d0 * d4 +
          32 * a1 * a4 * b4 * c0 * d0 * d5 - 16 * a1 * a4 * b5 * c0 * d0 * d4 +
          16 * a1 * a5 * b0 * c4 * d0 * d4 + 8 * a1 * a5 * b3 * c0 * d0 * d5 -
          16 * a1 * a5 * b4 * c0 * d0 * d4 - 8 * a1 * a5 * b5 * c0 * d0 * d3 +
          8 * a2 * a3 * b0 * c0 * d4 * d5 + 8 * a2 * a3 * b0 * c4 * d0 * d5 +
          8 * a2 * a3 * b0 * c5 * d0 * d4 - 16 * a2 * a3 * b4 * c0 * d0 * d5 -
          16 * a2 * a4 * b3 * c0 * d0 * d5 + 16 * a2 * a4 * b5 * c0 * d0 * d3 -
          8 * a2 * a5 * b0 * c0 * d3 * d4 - 8 * a2 * a5 * b0 * c3 * d0 * d4 -
          8 * a2 * a5 * b0 * c4 * d0 * d3 + 16 * a2 * a5 * b4 * c0 * d0 * d3 +
          8 * a3 * a4 * b0 * c0 * d2 * d5 + 8 * a3 * a4 * b0 * c2 * d0 * d5 +
          8 * a3 * a4 * b0 * c5 * d0 * d2 - 16 * a3 * a4 * b2 * c0 * d0 * d5 +
          8 * a3 * a5 * b1 * c0 * d0 * d5 - 8 * a3 * a5 * b5 * c0 * d0 * d1 +
          8 * a4 * a5 * b0 * c0 * d1 * d4 - 8 * a4 * a5 * b0 * c0 * d2 * d3 +
          8 * a4 * a5 * b0 * c1 * d0 * d4 - 8 * a4 * a5 * b0 * c2 * d0 * d3 -
          8 * a4 * a5 * b0 * c3 * d0 * d2 + 8 * a4 * a5 * b0 * c4 * d0 * d1 -
          16 * a4 * a5 * b1 * c0 * d0 * d4 + 16 * a4 * a5 * b2 * c0 * d0 * d3 -
          16 * a0 * a2 * b2 * c5 * d1 * d5 + 8 * a0 * a2 * b5 * c1 * d2 * d5 +
          8 * a0 * a2 * b5 * c2 * d1 * d5 + 8 * a0 * a2 * b5 * c5 * d1 * d2 +
          8 * a0 * a5 * b2 * c1 * d2 * d5 + 8 * a0 * a5 * b2 * c2 * d1 * d5 +
          8 * a0 * a5 * b2 * c5 * d1 * d2 - 16 * a0 * a5 * b5 * c2 * d1 * d2 +
          16 * a1 * a2 * b2 * c5 * d0 * d5 - 8 * a1 * a2 * b5 * c0 * d2 * d5 -
          8 * a1 * a2 * b5 * c2 * d0 * d5 - 8 * a1 * a2 * b5 * c5 * d0 * d2 -
          8 * a1 * a5 * b2 * c0 * d2 * d5 - 8 * a1 * a5 * b2 * c2 * d0 * d5 -
          8 * a1 * a5 * b2 * c5 * d0 * d2 + 16 * a1 * a5 * b5 * c2 * d0 * d2 +
          8 * a2 * a5 * b0 * c1 * d2 * d5 + 8 * a2 * a5 * b0 * c2 * d1 * d5 +
          8 * a2 * a5 * b0 * c5 * d1 * d2 - 8 * a2 * a5 * b1 * c0 * d2 * d5 -
          8 * a2 * a5 * b1 * c2 * d0 * d5 - 8 * a2 * a5 * b1 * c5 * d0 * d2 +
          4 * a0 * a1 * b5 * c5 * d0 * d5 + 8 * a0 * a2 * b0 * c5 * d4 * d5 -
          8 * a0 * a2 * b4 * c5 * d0 * d5 + 8 * a0 * a4 * b0 * c5 * d2 * d5 -
          8 * a0 * a4 * b2 * c5 * d0 * d5 - 8 * a0 * a5 * b0 * c2 * d4 * d5 -
          8 * a0 * a5 * b0 * c4 * d2 * d5 + 4 * a0 * a5 * b0 * c5 * d1 * d5 -
          8 * a0 * a5 * b0 * c5 * d2 * d4 + 4 * a0 * a5 * b1 * c5 * d0 * d5 -
          4 * a0 * a5 * b5 * c0 * d1 * d5 + 8 * a0 * a5 * b5 * c0 * d2 * d4 -
          4 * a0 * a5 * b5 * c1 * d0 * d5 + 8 * a0 * a5 * b5 * c2 * d0 * d4 +
          8 * a0 * a5 * b5 * c4 * d0 * d2 - 4 * a0 * a5 * b5 * c5 * d0 * d1 -
          32 * a1 * a2 * b1 * c3 * d4 * d5 - 32 * a1 * a2 * b1 * c4 * d3 * d5 -
          32 * a1 * a2 * b1 * c5 * d3 * d4 + 16 * a1 * a2 * b2 * c3 * d3 * d5 +
          64 * a1 * a2 * b2 * c4 * d3 * d4 + 16 * a1 * a2 * b3 * c1 * d4 * d5 +
          16 * a1 * a2 * b3 * c4 * d1 * d5 + 16 * a1 * a2 * b3 * c5 * d1 * d4 +
          16 * a1 * a2 * b4 * c1 * d3 * d5 - 32 * a1 * a2 * b4 * c2 * d3 * d4 +
          16 * a1 * a2 * b4 * c3 * d1 * d5 - 32 * a1 * a2 * b4 * c3 * d2 * d4 -
          32 * a1 * a2 * b4 * c4 * d2 * d3 + 16 * a1 * a2 * b4 * c5 * d1 * d3 -
          16 * a1 * a2 * b5 * c3 * d2 * d3 - 16 * a1 * a3 * b1 * c5 * d1 * d5 +
          16 * a1 * a3 * b2 * c1 * d4 * d5 + 16 * a1 * a3 * b2 * c4 * d1 * d5 +
          16 * a1 * a3 * b2 * c5 * d1 * d4 - 16 * a1 * a3 * b3 * c2 * d2 * d5 +
          16 * a1 * a3 * b5 * c1 * d1 * d5 - 16 * a1 * a3 * b5 * c1 * d2 * d4 -
          16 * a1 * a3 * b5 * c2 * d1 * d4 + 16 * a1 * a3 * b5 * c2 * d2 * d3 -
          16 * a1 * a3 * b5 * c4 * d1 * d2 + 16 * a1 * a4 * b2 * c1 * d3 * d5 -
          32 * a1 * a4 * b2 * c2 * d3 * d4 + 16 * a1 * a4 * b2 * c3 * d1 * d5 -
          32 * a1 * a4 * b2 * c3 * d2 * d4 - 32 * a1 * a4 * b2 * c4 * d2 * d3 +
          16 * a1 * a4 * b2 * c5 * d1 * d3 + 64 * a1 * a4 * b4 * c2 * d2 * d3 -
          16 * a1 * a4 * b5 * c1 * d2 * d3 - 16 * a1 * a4 * b5 * c2 * d1 * d3 -
          16 * a1 * a4 * b5 * c3 * d1 * d2 + 4 * a1 * a5 * b0 * c5 * d0 * d5 -
          16 * a1 * a5 * b1 * c1 * d3 * d5 + 32 * a1 * a5 * b1 * c2 * d3 * d4 -
          16 * a1 * a5 * b1 * c3 * d1 * d5 + 32 * a1 * a5 * b1 * c3 * d2 * d4 +
          32 * a1 * a5 * b1 * c4 * d2 * d3 - 16 * a1 * a5 * b1 * c5 * d1 * d3 -
          16 * a1 * a5 * b2 * c3 * d2 * d3 + 16 * a1 * a5 * b3 * c1 * d1 * d5 -
          16 * a1 * a5 * b3 * c1 * d2 * d4 - 16 * a1 * a5 * b3 * c2 * d1 * d4 +
          16 * a1 * a5 * b3 * c2 * d2 * d3 - 16 * a1 * a5 * b3 * c4 * d1 * d2 -
          16 * a1 * a5 * b4 * c1 * d2 * d3 - 16 * a1 * a5 * b4 * c2 * d1 * d3 -
          16 * a1 * a5 * b4 * c3 * d1 * d2 - 4 * a1 * a5 * b5 * c0 * d0 * d5 +
          16 * a1 * a5 * b5 * c1 * d1 * d3 + 16 * a2 * a3 * b1 * c1 * d4 * d5 +
          16 * a2 * a3 * b1 * c4 * d1 * d5 + 16 * a2 * a3 * b1 * c5 * d1 * d4 -
          16 * a2 * a3 * b2 * c1 * d3 * d5 + 32 * a2 * a3 * b2 * c2 * d3 * d4 -
          16 * a2 * a3 * b2 * c3 * d1 * d5 + 32 * a2 * a3 * b2 * c3 * d2 * d4 -
          64 * a2 * a3 * b2 * c4 * d1 * d4 + 32 * a2 * a3 * b2 * c4 * d2 * d3 -
          16 * a2 * a3 * b2 * c5 * d1 * d3 + 16 * a2 * a3 * b3 * c1 * d2 * d5 +
          16 * a2 * a3 * b3 * c2 * d1 * d5 - 32 * a2 * a3 * b3 * c2 * d2 * d4 +
          16 * a2 * a3 * b3 * c5 * d1 * d2 - 32 * a2 * a3 * b4 * c1 * d1 * d5 +
          32 * a2 * a3 * b4 * c1 * d2 * d4 + 32 * a2 * a3 * b4 * c2 * d1 * d4 -
          32 * a2 * a3 * b4 * c2 * d2 * d3 + 32 * a2 * a3 * b4 * c4 * d1 * d2 -
          8 * a2 * a4 * b0 * c5 * d0 * d5 + 16 * a2 * a4 * b1 * c1 * d3 * d5 -
          32 * a2 * a4 * b1 * c2 * d3 * d4 + 16 * a2 * a4 * b1 * c3 * d1 * d5 -
          32 * a2 * a4 * b1 * c3 * d2 * d4 - 32 * a2 * a4 * b1 * c4 * d2 * d3 +
          16 * a2 * a4 * b1 * c5 * d1 * d3 + 32 * a2 * a4 * b2 * c3 * d2 * d3 -
          32 * a2 * a4 * b3 * c1 * d1 * d5 + 32 * a2 * a4 * b3 * c1 * d2 * d4 +
          32 * a2 * a4 * b3 * c2 * d1 * d4 - 32 * a2 * a4 * b3 * c2 * d2 * d3 +
          32 * a2 * a4 * b3 * c4 * d1 * d2 + 8 * a2 * a4 * b5 * c0 * d0 * d5 -
          16 * a2 * a5 * b1 * c3 * d2 * d3 + 16 * a2 * a5 * b2 * c3 * d1 * d3 +
          8 * a2 * a5 * b4 * c0 * d0 * d5 - 8 * a2 * a5 * b5 * c0 * d0 * d4 -
          32 * a3 * a4 * b2 * c1 * d1 * d5 + 32 * a3 * a4 * b2 * c1 * d2 * d4 +
          32 * a3 * a4 * b2 * c2 * d1 * d4 - 32 * a3 * a4 * b2 * c2 * d2 * d3 +
          32 * a3 * a4 * b2 * c4 * d1 * d2 - 64 * a3 * a4 * b4 * c2 * d1 * d2 +
          32 * a3 * a4 * b5 * c1 * d1 * d2 + 16 * a3 * a5 * b1 * c1 * d1 * d5 -
          16 * a3 * a5 * b1 * c1 * d2 * d4 - 16 * a3 * a5 * b1 * c2 * d1 * d4 +
          16 * a3 * a5 * b1 * c2 * d2 * d3 - 16 * a3 * a5 * b1 * c4 * d1 * d2 -
          16 * a3 * a5 * b3 * c2 * d1 * d2 + 32 * a3 * a5 * b4 * c1 * d1 * d2 -
          16 * a4 * a5 * b1 * c1 * d2 * d3 - 16 * a4 * a5 * b1 * c2 * d1 * d3 -
          16 * a4 * a5 * b1 * c3 * d1 * d2 + 8 * a4 * a5 * b2 * c0 * d0 * d5 +
          32 * a4 * a5 * b3 * c1 * d1 * d2 - 8 * a4 * a5 * b5 * c0 * d0 * d2 -
          4 * a0 * a1 * b3 * c5 * d3 * d5 + 8 * a0 * a1 * b4 * c3 * d4 * d5 +
          8 * a0 * a1 * b4 * c4 * d3 * d5 + 8 * a0 * a1 * b4 * c5 * d3 * d4 +
          4 * a0 * a1 * b5 * c3 * d3 * d5 - 16 * a0 * a1 * b5 * c4 * d3 * d4 -
          8 * a0 * a2 * b4 * c3 * d3 * d5 + 8 * a0 * a2 * b5 * c3 * d3 * d4 -
          4 * a0 * a3 * b1 * c5 * d3 * d5 + 8 * a0 * a3 * b3 * c2 * d4 * d5 +
          8 * a0 * a3 * b3 * c4 * d2 * d5 - 4 * a0 * a3 * b3 * c5 * d1 * d5 +
          8 * a0 * a3 * b3 * c5 * d2 * d4 - 8 * a0 * a3 * b4 * c1 * d4 * d5 -
          8 * a0 * a3 * b4 * c4 * d1 * d5 - 8 * a0 * a3 * b4 * c5 * d1 * d4 +
          4 * a0 * a3 * b5 * c1 * d3 * d5 - 8 * a0 * a3 * b5 * c2 * d3 * d4 +
          4 * a0 * a3 * b5 * c3 * d1 * d5 - 8 * a0 * a3 * b5 * c3 * d2 * d4 +
          16 * a0 * a3 * b5 * c4 * d1 * d4 - 8 * a0 * a3 * b5 * c4 * d2 * d3 +
          4 * a0 * a3 * b5 * c5 * d1 * d3 + 8 * a0 * a4 * b1 * c3 * d4 * d5 +
          8 * a0 * a4 * b1 * c4 * d3 * d5 + 8 * a0 * a4 * b1 * c5 * d3 * d4 -
          8 * a0 * a4 * b2 * c3 * d3 * d5 - 8 * a0 * a4 * b3 * c1 * d4 * d5 -
          8 * a0 * a4 * b3 * c4 * d1 * d5 - 8 * a0 * a4 * b3 * c5 * d1 * d4 +
          8 * a0 * a4 * b5 * c3 * d2 * d3 + 4 * a0 * a5 * b1 * c3 * d3 * d5 -
          16 * a0 * a5 * b1 * c4 * d3 * d4 + 8 * a0 * a5 * b2 * c3 * d3 * d4 +
          4 * a0 * a5 * b3 * c1 * d3 * d5 - 8 * a0 * a5 * b3 * c2 * d3 * d4 +
          4 * a0 * a5 * b3 * c3 * d1 * d5 - 8 * a0 * a5 * b3 * c3 * d2 * d4 +
          16 * a0 * a5 * b3 * c4 * d1 * d4 - 8 * a0 * a5 * b3 * c4 * d2 * d3 +
          4 * a0 * a5 * b3 * c5 * d1 * d3 + 8 * a0 * a5 * b4 * c3 * d2 * d3 -
          12 * a0 * a5 * b5 * c3 * d1 * d3 - 4 * a1 * a3 * b0 * c5 * d3 * d5 +
          12 * a1 * a3 * b3 * c5 * d0 * d5 - 4 * a1 * a3 * b5 * c0 * d3 * d5 -
          4 * a1 * a3 * b5 * c3 * d0 * d5 - 4 * a1 * a3 * b5 * c5 * d0 * d3 +
          8 * a1 * a4 * b0 * c3 * d4 * d5 + 8 * a1 * a4 * b0 * c4 * d3 * d5 +
          8 * a1 * a4 * b0 * c5 * d3 * d4 - 16 * a1 * a4 * b4 * c0 * d3 * d5 -
          16 * a1 * a4 * b4 * c3 * d0 * d5 - 16 * a1 * a4 * b4 * c5 * d0 * d3 +
          8 * a1 * a4 * b5 * c0 * d3 * d4 + 8 * a1 * a4 * b5 * c3 * d0 * d4 +
          8 * a1 * a4 * b5 * c4 * d0 * d3 + 4 * a1 * a5 * b0 * c3 * d3 * d5 -
          16 * a1 * a5 * b0 * c4 * d3 * d4 - 4 * a1 * a5 * b3 * c0 * d3 * d5 -
          4 * a1 * a5 * b3 * c3 * d0 * d5 - 4 * a1 * a5 * b3 * c5 * d0 * d3 +
          8 * a1 * a5 * b4 * c0 * d3 * d4 + 8 * a1 * a5 * b4 * c3 * d0 * d4 +
          8 * a1 * a5 * b4 * c4 * d0 * d3 + 4 * a1 * a5 * b5 * c3 * d0 * d3 -
          8 * a2 * a3 * b3 * c0 * d4 * d5 - 8 * a2 * a3 * b3 * c4 * d0 * d5 -
          8 * a2 * a3 * b3 * c5 * d0 * d4 + 8 * a2 * a3 * b4 * c0 * d3 * d5 +
          8 * a2 * a3 * b4 * c3 * d0 * d5 + 8 * a2 * a3 * b4 * c5 * d0 * d3 -
          8 * a2 * a4 * b0 * c3 * d3 * d5 + 8 * a2 * a4 * b3 * c0 * d3 * d5 +
          8 * a2 * a4 * b3 * c3 * d0 * d5 + 8 * a2 * a4 * b3 * c5 * d0 * d3 -
          8 * a2 * a4 * b5 * c3 * d0 * d3 + 8 * a2 * a5 * b0 * c3 * d3 * d4 -
          8 * a2 * a5 * b4 * c3 * d0 * d3 - 8 * a3 * a4 * b0 * c1 * d4 * d5 -
          8 * a3 * a4 * b0 * c4 * d1 * d5 - 8 * a3 * a4 * b0 * c5 * d1 * d4 +
          8 * a3 * a4 * b2 * c0 * d3 * d5 + 8 * a3 * a4 * b2 * c3 * d0 * d5 +
          8 * a3 * a4 * b2 * c5 * d0 * d3 - 8 * a3 * a4 * b3 * c0 * d2 * d5 -
          8 * a3 * a4 * b3 * c2 * d0 * d5 - 8 * a3 * a4 * b3 * c5 * d0 * d2 +
          16 * a3 * a4 * b4 * c0 * d1 * d5 + 16 * a3 * a4 * b4 * c1 * d0 * d5 +
          16 * a3 * a4 * b4 * c5 * d0 * d1 - 8 * a3 * a4 * b5 * c0 * d1 * d4 -
          8 * a3 * a4 * b5 * c1 * d0 * d4 - 8 * a3 * a4 * b5 * c4 * d0 * d1 +
          4 * a3 * a5 * b0 * c1 * d3 * d5 - 8 * a3 * a5 * b0 * c2 * d3 * d4 +
          4 * a3 * a5 * b0 * c3 * d1 * d5 - 8 * a3 * a5 * b0 * c3 * d2 * d4 +
          16 * a3 * a5 * b0 * c4 * d1 * d4 - 8 * a3 * a5 * b0 * c4 * d2 * d3 +
          4 * a3 * a5 * b0 * c5 * d1 * d3 - 4 * a3 * a5 * b1 * c0 * d3 * d5 -
          4 * a3 * a5 * b1 * c3 * d0 * d5 - 4 * a3 * a5 * b1 * c5 * d0 * d3 -
          4 * a3 * a5 * b3 * c0 * d1 * d5 + 8 * a3 * a5 * b3 * c0 * d2 * d4 -
          4 * a3 * a5 * b3 * c1 * d0 * d5 + 8 * a3 * a5 * b3 * c2 * d0 * d4 +
          8 * a3 * a5 * b3 * c4 * d0 * d2 - 4 * a3 * a5 * b3 * c5 * d0 * d1 -
          8 * a3 * a5 * b4 * c0 * d1 * d4 - 8 * a3 * a5 * b4 * c1 * d0 * d4 -
          8 * a3 * a5 * b4 * c4 * d0 * d1 + 4 * a3 * a5 * b5 * c0 * d1 * d3 +
          4 * a3 * a5 * b5 * c1 * d0 * d3 + 4 * a3 * a5 * b5 * c3 * d0 * d1 +
          8 * a4 * a5 * b0 * c3 * d2 * d3 + 8 * a4 * a5 * b1 * c0 * d3 * d4 +
          8 * a4 * a5 * b1 * c3 * d0 * d4 + 8 * a4 * a5 * b1 * c4 * d0 * d3 -
          8 * a4 * a5 * b2 * c3 * d0 * d3 - 8 * a4 * a5 * b3 * c0 * d1 * d4 -
          8 * a4 * a5 * b3 * c1 * d0 * d4 - 8 * a4 * a5 * b3 * c4 * d0 * d1 +
          32 * a1 * a2 * b1 * c5 * d4 * d5 - 16 * a1 * a2 * b2 * c5 * d3 * d5 -
          32 * a1 * a2 * b4 * c5 * d1 * d5 + 8 * a1 * a2 * b5 * c2 * d3 * d5 +
          8 * a1 * a2 * b5 * c3 * d2 * d5 + 8 * a1 * a2 * b5 * c5 * d2 * d3 +
          32 * a1 * a4 * b1 * c5 * d2 * d5 - 32 * a1 * a4 * b2 * c5 * d1 * d5 -
          32 * a1 * a5 * b1 * c2 * d4 * d5 - 32 * a1 * a5 * b1 * c4 * d2 * d5 +
          48 * a1 * a5 * b1 * c5 * d1 * d5 - 32 * a1 * a5 * b1 * c5 * d2 * d4 +
          8 * a1 * a5 * b2 * c2 * d3 * d5 + 8 * a1 * a5 * b2 * c3 * d2 * d5 +
          8 * a1 * a5 * b2 * c5 * d2 * d3 - 48 * a1 * a5 * b5 * c1 * d1 * d5 +
          32 * a1 * a5 * b5 * c1 * d2 * d4 + 32 * a1 * a5 * b5 * c2 * d1 * d4 -
          16 * a1 * a5 * b5 * c2 * d2 * d3 + 32 * a1 * a5 * b5 * c4 * d1 * d2 +
          16 * a2 * a3 * b2 * c5 * d1 * d5 - 8 * a2 * a3 * b5 * c1 * d2 * d5 -
          8 * a2 * a3 * b5 * c2 * d1 * d5 - 8 * a2 * a3 * b5 * c5 * d1 * d2 -
          32 * a2 * a4 * b1 * c5 * d1 * d5 + 32 * a2 * a4 * b5 * c1 * d1 * d5 +
          8 * a2 * a5 * b1 * c2 * d3 * d5 + 8 * a2 * a5 * b1 * c3 * d2 * d5 +
          8 * a2 * a5 * b1 * c5 * d2 * d3 - 8 * a2 * a5 * b3 * c1 * d2 * d5 -
          8 * a2 * a5 * b3 * c2 * d1 * d5 - 8 * a2 * a5 * b3 * c5 * d1 * d2 +
          32 * a2 * a5 * b4 * c1 * d1 * d5 - 32 * a2 * a5 * b5 * c1 * d1 * d4 -
          8 * a3 * a5 * b2 * c1 * d2 * d5 - 8 * a3 * a5 * b2 * c2 * d1 * d5 -
          8 * a3 * a5 * b2 * c5 * d1 * d2 + 16 * a3 * a5 * b5 * c2 * d1 * d2 +
          32 * a4 * a5 * b2 * c1 * d1 * d5 - 32 * a4 * a5 * b5 * c1 * d1 * d2 -
          4 * a0 * a1 * b5 * c5 * d3 * d5 - 8 * a0 * a2 * b3 * c5 * d4 * d5 +
          8 * a0 * a2 * b4 * c5 * d3 * d5 - 8 * a0 * a3 * b2 * c5 * d4 * d5 -
          8 * a0 * a3 * b4 * c5 * d2 * d5 + 8 * a0 * a3 * b5 * c2 * d4 * d5 +
          8 * a0 * a3 * b5 * c4 * d2 * d5 - 4 * a0 * a3 * b5 * c5 * d1 * d5 +
          8 * a0 * a3 * b5 * c5 * d2 * d4 + 8 * a0 * a4 * b2 * c5 * d3 * d5 -
          8 * a0 * a4 * b3 * c5 * d2 * d5 + 16 * a0 * a4 * b4 * c5 * d1 * d5 -
          8 * a0 * a4 * b5 * c1 * d4 * d5 - 8 * a0 * a4 * b5 * c4 * d1 * d5 -
          8 * a0 * a4 * b5 * c5 * d1 * d4 - 4 * a0 * a5 * b1 * c5 * d3 * d5 +
          8 * a0 * a5 * b3 * c2 * d4 * d5 + 8 * a0 * a5 * b3 * c4 * d2 * d5 -
          4 * a0 * a5 * b3 * c5 * d1 * d5 + 8 * a0 * a5 * b3 * c5 * d2 * d4 -
          8 * a0 * a5 * b4 * c1 * d4 * d5 - 8 * a0 * a5 * b4 * c4 * d1 * d5 -
          8 * a0 * a5 * b4 * c5 * d1 * d4 + 4 * a0 * a5 * b5 * c1 * d3 * d5 -
          8 * a0 * a5 * b5 * c2 * d3 * d4 + 4 * a0 * a5 * b5 * c3 * d1 * d5 -
          8 * a0 * a5 * b5 * c3 * d2 * d4 + 16 * a0 * a5 * b5 * c4 * d1 * d4 -
          8 * a0 * a5 * b5 * c4 * d2 * d3 + 4 * a0 * a5 * b5 * c5 * d1 * d3 -
          4 * a1 * a3 * b5 * c5 * d0 * d5 - 16 * a1 * a4 * b4 * c5 * d0 * d5 +
          8 * a1 * a4 * b5 * c0 * d4 * d5 + 8 * a1 * a4 * b5 * c4 * d0 * d5 +
          8 * a1 * a4 * b5 * c5 * d0 * d4 - 4 * a1 * a5 * b0 * c5 * d3 * d5 -
          4 * a1 * a5 * b3 * c5 * d0 * d5 + 8 * a1 * a5 * b4 * c0 * d4 * d5 +
          8 * a1 * a5 * b4 * c4 * d0 * d5 + 8 * a1 * a5 * b4 * c5 * d0 * d4 +
          4 * a1 * a5 * b5 * c0 * d3 * d5 + 4 * a1 * a5 * b5 * c3 * d0 * d5 -
          16 * a1 * a5 * b5 * c4 * d0 * d4 + 4 * a1 * a5 * b5 * c5 * d0 * d3 -
          8 * a2 * a3 * b0 * c5 * d4 * d5 + 8 * a2 * a3 * b4 * c5 * d0 * d5 +
          8 * a2 * a4 * b0 * c5 * d3 * d5 + 8 * a2 * a4 * b3 * c5 * d0 * d5 -
          8 * a2 * a4 * b5 * c0 * d3 * d5 - 8 * a2 * a4 * b5 * c3 * d0 * d5 -
          8 * a2 * a4 * b5 * c5 * d0 * d3 - 8 * a2 * a5 * b4 * c0 * d3 * d5 -
          8 * a2 * a5 * b4 * c3 * d0 * d5 - 8 * a2 * a5 * b4 * c5 * d0 * d3 +
          8 * a2 * a5 * b5 * c0 * d3 * d4 + 8 * a2 * a5 * b5 * c3 * d0 * d4 +
          8 * a2 * a5 * b5 * c4 * d0 * d3 - 8 * a3 * a4 * b0 * c5 * d2 * d5 +
          8 * a3 * a4 * b2 * c5 * d0 * d5 + 8 * a3 * a5 * b0 * c2 * d4 * d5 +
          8 * a3 * a5 * b0 * c4 * d2 * d5 - 4 * a3 * a5 * b0 * c5 * d1 * d5 +
          8 * a3 * a5 * b0 * c5 * d2 * d4 - 4 * a3 * a5 * b1 * c5 * d0 * d5 +
          4 * a3 * a5 * b5 * c0 * d1 * d5 - 8 * a3 * a5 * b5 * c0 * d2 * d4 +
          4 * a3 * a5 * b5 * c1 * d0 * d5 - 8 * a3 * a5 * b5 * c2 * d0 * d4 -
          8 * a3 * a5 * b5 * c4 * d0 * d2 + 4 * a3 * a5 * b5 * c5 * d0 * d1 -
          8 * a4 * a5 * b0 * c1 * d4 * d5 - 8 * a4 * a5 * b0 * c4 * d1 * d5 -
          8 * a4 * a5 * b0 * c5 * d1 * d4 + 8 * a4 * a5 * b1 * c0 * d4 * d5 +
          8 * a4 * a5 * b1 * c4 * d0 * d5 + 8 * a4 * a5 * b1 * c5 * d0 * d4 -
          8 * a4 * a5 * b2 * c0 * d3 * d5 - 8 * a4 * a5 * b2 * c3 * d0 * d5 -
          8 * a4 * a5 * b2 * c5 * d0 * d3 + 8 * a4 * a5 * b5 * c0 * d2 * d3 +
          8 * a4 * a5 * b5 * c2 * d0 * d3 + 8 * a4 * a5 * b5 * c3 * d0 * d2 +
          4 * a1 * a3 * b5 * c5 * d3 * d5 + 16 * a1 * a4 * b4 * c5 * d3 * d5 -
          8 * a1 * a4 * b5 * c3 * d4 * d5 - 8 * a1 * a4 * b5 * c4 * d3 * d5 -
          8 * a1 * a4 * b5 * c5 * d3 * d4 + 4 * a1 * a5 * b3 * c5 * d3 * d5 -
          8 * a1 * a5 * b4 * c3 * d4 * d5 - 8 * a1 * a5 * b4 * c4 * d3 * d5 -
          8 * a1 * a5 * b4 * c5 * d3 * d4 - 4 * a1 * a5 * b5 * c3 * d3 * d5 +
          16 * a1 * a5 * b5 * c4 * d3 * d4 + 8 * a2 * a3 * b3 * c5 * d4 * d5 -
          8 * a2 * a3 * b4 * c5 * d3 * d5 - 8 * a2 * a4 * b3 * c5 * d3 * d5 +
          8 * a2 * a4 * b5 * c3 * d3 * d5 + 8 * a2 * a5 * b4 * c3 * d3 * d5 -
          8 * a2 * a5 * b5 * c3 * d3 * d4 - 8 * a3 * a4 * b2 * c5 * d3 * d5 +
          8 * a3 * a4 * b3 * c5 * d2 * d5 - 16 * a3 * a4 * b4 * c5 * d1 * d5 +
          8 * a3 * a4 * b5 * c1 * d4 * d5 + 8 * a3 * a4 * b5 * c4 * d1 * d5 +
          8 * a3 * a4 * b5 * c5 * d1 * d4 + 4 * a3 * a5 * b1 * c5 * d3 * d5 -
          8 * a3 * a5 * b3 * c2 * d4 * d5 - 8 * a3 * a5 * b3 * c4 * d2 * d5 +
          4 * a3 * a5 * b3 * c5 * d1 * d5 - 8 * a3 * a5 * b3 * c5 * d2 * d4 +
          8 * a3 * a5 * b4 * c1 * d4 * d5 + 8 * a3 * a5 * b4 * c4 * d1 * d5 +
          8 * a3 * a5 * b4 * c5 * d1 * d4 - 4 * a3 * a5 * b5 * c1 * d3 * d5 +
          8 * a3 * a5 * b5 * c2 * d3 * d4 - 4 * a3 * a5 * b5 * c3 * d1 * d5 +
          8 * a3 * a5 * b5 * c3 * d2 * d4 - 16 * a3 * a5 * b5 * c4 * d1 * d4 +
          8 * a3 * a5 * b5 * c4 * d2 * d3 - 4 * a3 * a5 * b5 * c5 * d1 * d3 -
          8 * a4 * a5 * b1 * c3 * d4 * d5 - 8 * a4 * a5 * b1 * c4 * d3 * d5 -
          8 * a4 * a5 * b1 * c5 * d3 * d4 + 8 * a4 * a5 * b2 * c3 * d3 * d5 +
          8 * a4 * a5 * b3 * c1 * d4 * d5 + 8 * a4 * a5 * b3 * c4 * d1 * d5 +
          8 * a4 * a5 * b3 * c5 * d1 * d4 - 8 * a4 * a5 * b5 * c3 * d2 * d3,
      pow(a0, 2) * b1 * pow(c3, 2) * d0 -
          3 * pow(a0, 2) * b0 * pow(c3, 2) * d1 -
          pow(a3, 2) * b0 * pow(c0, 2) * d1 +
          3 * pow(a3, 2) * b1 * pow(c0, 2) * d0 +
          12 * pow(a0, 2) * b0 * pow(c4, 2) * d1 +
          4 * pow(a0, 2) * b1 * pow(c1, 2) * d3 -
          4 * pow(a0, 2) * b1 * pow(c4, 2) * d0 -
          12 * pow(a0, 2) * b3 * pow(c1, 2) * d1 +
          12 * pow(a1, 2) * b1 * pow(c0, 2) * d3 -
          4 * pow(a1, 2) * b3 * pow(c0, 2) * d1 +
          4 * pow(a4, 2) * b0 * pow(c0, 2) * d1 -
          12 * pow(a4, 2) * b1 * pow(c0, 2) * d0 -
          3 * pow(a0, 2) * b0 * pow(c5, 2) * d1 +
          4 * pow(a0, 2) * b1 * pow(c2, 2) * d3 +
          pow(a0, 2) * b1 * pow(c5, 2) * d0 -
          4 * pow(a0, 2) * b3 * pow(c2, 2) * d1 -
          4 * pow(a2, 2) * b0 * pow(c3, 2) * d1 +
          4 * pow(a2, 2) * b1 * pow(c0, 2) * d3 -
          4 * pow(a2, 2) * b1 * pow(c3, 2) * d0 -
          4 * pow(a2, 2) * b3 * pow(c0, 2) * d1 +
          4 * pow(a3, 2) * b0 * pow(c2, 2) * d1 +
          4 * pow(a3, 2) * b1 * pow(c2, 2) * d0 -
          pow(a5, 2) * b0 * pow(c0, 2) * d1 +
          3 * pow(a5, 2) * b1 * pow(c0, 2) * d0 -
          4 * pow(a0, 2) * b1 * pow(c1, 2) * d5 -
          8 * pow(a0, 2) * b2 * pow(c1, 2) * d4 +
          8 * pow(a0, 2) * b4 * pow(c1, 2) * d2 +
          12 * pow(a0, 2) * b5 * pow(c1, 2) * d1 -
          8 * pow(a1, 2) * b0 * pow(c5, 2) * d1 -
          12 * pow(a1, 2) * b1 * pow(c0, 2) * d5 +
          48 * pow(a1, 2) * b1 * pow(c2, 2) * d3 +
          24 * pow(a1, 2) * b1 * pow(c5, 2) * d0 -
          8 * pow(a1, 2) * b2 * pow(c0, 2) * d4 -
          16 * pow(a1, 2) * b3 * pow(c2, 2) * d1 +
          8 * pow(a1, 2) * b4 * pow(c0, 2) * d2 +
          4 * pow(a1, 2) * b5 * pow(c0, 2) * d1 +
          16 * pow(a2, 2) * b0 * pow(c4, 2) * d1 +
          16 * pow(a2, 2) * b1 * pow(c1, 2) * d3 -
          16 * pow(a2, 2) * b1 * pow(c4, 2) * d0 -
          48 * pow(a2, 2) * b3 * pow(c1, 2) * d1 +
          16 * pow(a4, 2) * b0 * pow(c2, 2) * d1 -
          16 * pow(a4, 2) * b1 * pow(c2, 2) * d0 -
          24 * pow(a5, 2) * b0 * pow(c1, 2) * d1 +
          8 * pow(a5, 2) * b1 * pow(c1, 2) * d0 -
          4 * pow(a0, 2) * b1 * pow(c2, 2) * d5 +
          4 * pow(a0, 2) * b1 * pow(c4, 2) * d3 -
          8 * pow(a0, 2) * b2 * pow(c2, 2) * d4 -
          4 * pow(a0, 2) * b3 * pow(c4, 2) * d1 +
          24 * pow(a0, 2) * b4 * pow(c2, 2) * d2 +
          4 * pow(a0, 2) * b5 * pow(c2, 2) * d1 -
          4 * pow(a2, 2) * b0 * pow(c5, 2) * d1 -
          4 * pow(a2, 2) * b1 * pow(c0, 2) * d5 +
          4 * pow(a2, 2) * b1 * pow(c5, 2) * d0 -
          24 * pow(a2, 2) * b2 * pow(c0, 2) * d4 +
          8 * pow(a2, 2) * b4 * pow(c0, 2) * d2 +
          4 * pow(a2, 2) * b5 * pow(c0, 2) * d1 +
          4 * pow(a4, 2) * b1 * pow(c0, 2) * d3 -
          4 * pow(a4, 2) * b3 * pow(c0, 2) * d1 -
          4 * pow(a5, 2) * b0 * pow(c2, 2) * d1 +
          4 * pow(a5, 2) * b1 * pow(c2, 2) * d0 -
          pow(a0, 2) * b1 * pow(c3, 2) * d5 +
          2 * pow(a0, 2) * b1 * pow(c5, 2) * d3 -
          2 * pow(a0, 2) * b2 * pow(c3, 2) * d4 +
          2 * pow(a0, 2) * b3 * pow(c5, 2) * d1 -
          2 * pow(a0, 2) * b4 * pow(c3, 2) * d2 +
          3 * pow(a0, 2) * b5 * pow(c3, 2) * d1 -
          48 * pow(a1, 2) * b1 * pow(c2, 2) * d5 -
          32 * pow(a1, 2) * b2 * pow(c2, 2) * d4 +
          96 * pow(a1, 2) * b4 * pow(c2, 2) * d2 +
          16 * pow(a1, 2) * b5 * pow(c2, 2) * d1 -
          16 * pow(a2, 2) * b1 * pow(c1, 2) * d5 -
          96 * pow(a2, 2) * b2 * pow(c1, 2) * d4 +
          32 * pow(a2, 2) * b4 * pow(c1, 2) * d2 +
          48 * pow(a2, 2) * b5 * pow(c1, 2) * d1 -
          pow(a3, 2) * b0 * pow(c5, 2) * d1 -
          3 * pow(a3, 2) * b1 * pow(c0, 2) * d5 +
          3 * pow(a3, 2) * b1 * pow(c5, 2) * d0 +
          2 * pow(a3, 2) * b2 * pow(c0, 2) * d4 +
          2 * pow(a3, 2) * b4 * pow(c0, 2) * d2 +
          pow(a3, 2) * b5 * pow(c0, 2) * d1 -
          3 * pow(a5, 2) * b0 * pow(c3, 2) * d1 -
          2 * pow(a5, 2) * b1 * pow(c0, 2) * d3 +
          pow(a5, 2) * b1 * pow(c3, 2) * d0 -
          2 * pow(a5, 2) * b3 * pow(c0, 2) * d1 -
          8 * pow(a0, 2) * b5 * pow(c4, 2) * d1 +
          12 * pow(a1, 2) * b1 * pow(c5, 2) * d3 -
          4 * pow(a1, 2) * b3 * pow(c5, 2) * d1 +
          16 * pow(a2, 2) * b1 * pow(c4, 2) * d3 -
          16 * pow(a2, 2) * b3 * pow(c4, 2) * d1 +
          4 * pow(a4, 2) * b0 * pow(c5, 2) * d1 +
          8 * pow(a4, 2) * b1 * pow(c0, 2) * d5 +
          16 * pow(a4, 2) * b1 * pow(c2, 2) * d3 -
          4 * pow(a4, 2) * b1 * pow(c5, 2) * d0 -
          16 * pow(a4, 2) * b3 * pow(c2, 2) * d1 +
          4 * pow(a5, 2) * b0 * pow(c4, 2) * d1 +
          4 * pow(a5, 2) * b1 * pow(c1, 2) * d3 -
          4 * pow(a5, 2) * b1 * pow(c4, 2) * d0 -
          12 * pow(a5, 2) * b3 * pow(c1, 2) * d1 -
          3 * pow(a0, 2) * b1 * pow(c5, 2) * d5 +
          2 * pow(a0, 2) * b2 * pow(c5, 2) * d4 +
          2 * pow(a0, 2) * b4 * pow(c5, 2) * d2 +
          pow(a0, 2) * b5 * pow(c5, 2) * d1 +
          4 * pow(a2, 2) * b1 * pow(c3, 2) * d5 -
          4 * pow(a2, 2) * b1 * pow(c5, 2) * d3 -
          24 * pow(a2, 2) * b2 * pow(c3, 2) * d4 +
          4 * pow(a2, 2) * b3 * pow(c5, 2) * d1 +
          8 * pow(a2, 2) * b4 * pow(c3, 2) * d2 +
          4 * pow(a2, 2) * b5 * pow(c3, 2) * d1 -
          4 * pow(a3, 2) * b1 * pow(c2, 2) * d5 -
          8 * pow(a3, 2) * b2 * pow(c2, 2) * d4 +
          24 * pow(a3, 2) * b4 * pow(c2, 2) * d2 -
          4 * pow(a3, 2) * b5 * pow(c2, 2) * d1 -
          pow(a5, 2) * b1 * pow(c0, 2) * d5 -
          4 * pow(a5, 2) * b1 * pow(c2, 2) * d3 -
          2 * pow(a5, 2) * b2 * pow(c0, 2) * d4 +
          4 * pow(a5, 2) * b3 * pow(c2, 2) * d1 -
          2 * pow(a5, 2) * b4 * pow(c0, 2) * d2 +
          3 * pow(a5, 2) * b5 * pow(c0, 2) * d1 -
          36 * pow(a1, 2) * b1 * pow(c5, 2) * d5 +
          8 * pow(a1, 2) * b2 * pow(c5, 2) * d4 +
          8 * pow(a1, 2) * b4 * pow(c5, 2) * d2 +
          12 * pow(a1, 2) * b5 * pow(c5, 2) * d1 -
          12 * pow(a5, 2) * b1 * pow(c1, 2) * d5 -
          8 * pow(a5, 2) * b2 * pow(c1, 2) * d4 -
          8 * pow(a5, 2) * b4 * pow(c1, 2) * d2 +
          36 * pow(a5, 2) * b5 * pow(c1, 2) * d1 +
          4 * pow(a4, 2) * b1 * pow(c5, 2) * d3 -
          4 * pow(a4, 2) * b3 * pow(c5, 2) * d1 +
          4 * pow(a5, 2) * b1 * pow(c4, 2) * d3 -
          4 * pow(a5, 2) * b3 * pow(c4, 2) * d1 -
          3 * pow(a3, 2) * b1 * pow(c5, 2) * d5 +
          2 * pow(a3, 2) * b2 * pow(c5, 2) * d4 +
          2 * pow(a3, 2) * b4 * pow(c5, 2) * d2 +
          pow(a3, 2) * b5 * pow(c5, 2) * d1 -
          pow(a5, 2) * b1 * pow(c3, 2) * d5 -
          2 * pow(a5, 2) * b2 * pow(c3, 2) * d4 -
          2 * pow(a5, 2) * b4 * pow(c3, 2) * d2 +
          3 * pow(a5, 2) * b5 * pow(c3, 2) * d1 +
          2 * a0 * a1 * b0 * pow(c3, 2) * d0 +
          8 * a0 * a1 * b0 * pow(c1, 2) * d3 -
          8 * a0 * a1 * b0 * pow(c4, 2) * d0 +
          8 * a0 * a1 * b3 * pow(c1, 2) * d0 -
          24 * a0 * a3 * b0 * pow(c1, 2) * d1 +
          8 * a0 * a3 * b1 * pow(c1, 2) * d0 +
          8 * a1 * a3 * b0 * pow(c1, 2) * d0 +
          8 * a0 * a1 * b0 * pow(c2, 2) * d3 +
          2 * a0 * a1 * b0 * pow(c5, 2) * d0 -
          8 * a0 * a3 * b0 * pow(c2, 2) * d1 -
          8 * a1 * a3 * b1 * pow(c0, 2) * d1 -
          8 * a0 * a1 * b0 * pow(c1, 2) * d5 -
          2 * a0 * a1 * b3 * pow(c0, 2) * d3 -
          8 * a0 * a1 * b5 * pow(c1, 2) * d0 -
          16 * a0 * a2 * b0 * pow(c1, 2) * d4 -
          2 * a0 * a3 * b1 * pow(c0, 2) * d3 -
          2 * a0 * a3 * b3 * pow(c0, 2) * d1 +
          16 * a0 * a4 * b0 * pow(c1, 2) * d2 +
          24 * a0 * a5 * b0 * pow(c1, 2) * d1 -
          8 * a0 * a5 * b1 * pow(c1, 2) * d0 -
          2 * a1 * a3 * b0 * pow(c0, 2) * d3 +
          6 * a1 * a3 * b3 * pow(c0, 2) * d0 -
          8 * a1 * a5 * b0 * pow(c1, 2) * d0 -
          8 * a0 * a1 * b0 * pow(c2, 2) * d5 +
          8 * a0 * a1 * b0 * pow(c4, 2) * d3 -
          16 * a0 * a1 * b1 * pow(c5, 2) * d1 +
          8 * a0 * a1 * b2 * pow(c3, 2) * d2 -
          16 * a0 * a2 * b0 * pow(c2, 2) * d4 +
          8 * a0 * a2 * b1 * pow(c3, 2) * d2 -
          8 * a0 * a2 * b2 * pow(c3, 2) * d1 -
          16 * a0 * a2 * b4 * pow(c2, 2) * d0 -
          8 * a0 * a3 * b0 * pow(c4, 2) * d1 +
          48 * a0 * a4 * b0 * pow(c2, 2) * d2 -
          16 * a0 * a4 * b2 * pow(c2, 2) * d0 +
          8 * a0 * a5 * b0 * pow(c2, 2) * d1 +
          8 * a1 * a2 * b0 * pow(c3, 2) * d2 -
          16 * a1 * a2 * b1 * pow(c0, 2) * d4 +
          8 * a1 * a2 * b2 * pow(c0, 2) * d3 -
          8 * a1 * a2 * b2 * pow(c3, 2) * d0 -
          32 * a1 * a3 * b1 * pow(c2, 2) * d1 +
          16 * a1 * a4 * b1 * pow(c0, 2) * d2 +
          8 * a1 * a5 * b1 * pow(c0, 2) * d1 -
          8 * a2 * a3 * b2 * pow(c0, 2) * d1 -
          16 * a2 * a4 * b0 * pow(c2, 2) * d0 -
          2 * a0 * a1 * b0 * pow(c3, 2) * d5 +
          4 * a0 * a1 * b0 * pow(c5, 2) * d3 +
          2 * a0 * a1 * b3 * pow(c0, 2) * d5 -
          8 * a0 * a1 * b3 * pow(c2, 2) * d3 -
          4 * a0 * a1 * b3 * pow(c5, 2) * d0 +
          8 * a0 * a1 * b4 * pow(c0, 2) * d4 +
          2 * a0 * a1 * b5 * pow(c0, 2) * d3 -
          2 * a0 * a1 * b5 * pow(c3, 2) * d0 -
          4 * a0 * a2 * b0 * pow(c3, 2) * d4 +
          32 * a0 * a2 * b2 * pow(c4, 2) * d1 -
          4 * a0 * a2 * b3 * pow(c0, 2) * d4 -
          4 * a0 * a2 * b4 * pow(c0, 2) * d3 +
          4 * a0 * a2 * b4 * pow(c3, 2) * d0 +
          4 * a0 * a3 * b0 * pow(c5, 2) * d1 +
          2 * a0 * a3 * b1 * pow(c0, 2) * d5 -
          8 * a0 * a3 * b1 * pow(c2, 2) * d3 -
          4 * a0 * a3 * b1 * pow(c5, 2) * d0 -
          4 * a0 * a3 * b2 * pow(c0, 2) * d4 +
          8 * a0 * a3 * b3 * pow(c2, 2) * d1 -
          4 * a0 * a3 * b4 * pow(c0, 2) * d2 +
          2 * a0 * a3 * b5 * pow(c0, 2) * d1 -
          4 * a0 * a4 * b0 * pow(c3, 2) * d2 +
          8 * a0 * a4 * b1 * pow(c0, 2) * d4 -
          4 * a0 * a4 * b2 * pow(c0, 2) * d3 +
          4 * a0 * a4 * b2 * pow(c3, 2) * d0 -
          4 * a0 * a4 * b3 * pow(c0, 2) * d2 +
          8 * a0 * a4 * b4 * pow(c0, 2) * d1 +
          6 * a0 * a5 * b0 * pow(c3, 2) * d1 +
          2 * a0 * a5 * b1 * pow(c0, 2) * d3 -
          2 * a0 * a5 * b1 * pow(c3, 2) * d0 +
          2 * a0 * a5 * b3 * pow(c0, 2) * d1 +
          32 * a1 * a2 * b2 * pow(c1, 2) * d3 -
          32 * a1 * a2 * b2 * pow(c4, 2) * d0 +
          32 * a1 * a2 * b3 * pow(c1, 2) * d2 +
          2 * a1 * a3 * b0 * pow(c0, 2) * d5 -
          8 * a1 * a3 * b0 * pow(c2, 2) * d3 -
          4 * a1 * a3 * b0 * pow(c5, 2) * d0 +
          32 * a1 * a3 * b2 * pow(c1, 2) * d2 +
          8 * a1 * a3 * b3 * pow(c2, 2) * d0 -
          6 * a1 * a3 * b5 * pow(c0, 2) * d0 +
          8 * a1 * a4 * b0 * pow(c0, 2) * d4 -
          24 * a1 * a4 * b4 * pow(c0, 2) * d0 +
          2 * a1 * a5 * b0 * pow(c0, 2) * d3 -
          2 * a1 * a5 * b0 * pow(c3, 2) * d0 -
          6 * a1 * a5 * b3 * pow(c0, 2) * d0 -
          4 * a2 * a3 * b0 * pow(c0, 2) * d4 +
          32 * a2 * a3 * b1 * pow(c1, 2) * d2 -
          96 * a2 * a3 * b2 * pow(c1, 2) * d1 +
          12 * a2 * a3 * b4 * pow(c0, 2) * d0 -
          4 * a2 * a4 * b0 * pow(c0, 2) * d3 +
          4 * a2 * a4 * b0 * pow(c3, 2) * d0 +
          12 * a2 * a4 * b3 * pow(c0, 2) * d0 -
          4 * a3 * a4 * b0 * pow(c0, 2) * d2 +
          12 * a3 * a4 * b2 * pow(c0, 2) * d0 +
          2 * a3 * a5 * b0 * pow(c0, 2) * d1 -
          6 * a3 * a5 * b1 * pow(c0, 2) * d0 -
          8 * a0 * a1 * b3 * pow(c1, 2) * d5 -
          8 * a0 * a1 * b5 * pow(c1, 2) * d3 +
          8 * a0 * a1 * b5 * pow(c4, 2) * d0 -
          8 * a0 * a2 * b2 * pow(c5, 2) * d1 -
          8 * a0 * a3 * b1 * pow(c1, 2) * d5 -
          16 * a0 * a3 * b4 * pow(c1, 2) * d2 +
          24 * a0 * a3 * b5 * pow(c1, 2) * d1 -
          16 * a0 * a4 * b3 * pow(c1, 2) * d2 -
          16 * a0 * a5 * b0 * pow(c4, 2) * d1 -
          8 * a0 * a5 * b1 * pow(c1, 2) * d3 +
          8 * a0 * a5 * b1 * pow(c4, 2) * d0 +
          24 * a0 * a5 * b3 * pow(c1, 2) * d1 -
          64 * a1 * a2 * b1 * pow(c2, 2) * d4 -
          8 * a1 * a2 * b2 * pow(c0, 2) * d5 +
          8 * a1 * a2 * b2 * pow(c5, 2) * d0 -
          64 * a1 * a2 * b4 * pow(c2, 2) * d1 -
          8 * a1 * a3 * b0 * pow(c1, 2) * d5 -
          8 * a1 * a3 * b5 * pow(c1, 2) * d0 +
          192 * a1 * a4 * b1 * pow(c2, 2) * d2 -
          64 * a1 * a4 * b2 * pow(c2, 2) * d1 -
          8 * a1 * a5 * b0 * pow(c1, 2) * d3 +
          8 * a1 * a5 * b0 * pow(c4, 2) * d0 +
          32 * a1 * a5 * b1 * pow(c2, 2) * d1 -
          8 * a1 * a5 * b3 * pow(c1, 2) * d0 +
          16 * a2 * a3 * b4 * pow(c1, 2) * d0 -
          64 * a2 * a4 * b1 * pow(c2, 2) * d1 +
          16 * a2 * a4 * b2 * pow(c0, 2) * d2 +
          16 * a2 * a4 * b3 * pow(c1, 2) * d0 +
          8 * a2 * a5 * b2 * pow(c0, 2) * d1 -
          16 * a3 * a4 * b0 * pow(c1, 2) * d2 +
          16 * a3 * a4 * b2 * pow(c1, 2) * d0 +
          24 * a3 * a5 * b0 * pow(c1, 2) * d1 -
          8 * a3 * a5 * b1 * pow(c1, 2) * d0 -
          6 * a0 * a1 * b0 * pow(c5, 2) * d5 +
          8 * a0 * a1 * b3 * pow(c2, 2) * d5 -
          2 * a0 * a1 * b5 * pow(c0, 2) * d5 +
          2 * a0 * a1 * b5 * pow(c5, 2) * d0 +
          4 * a0 * a2 * b0 * pow(c5, 2) * d4 +
          16 * a0 * a2 * b3 * pow(c2, 2) * d4 +
          4 * a0 * a2 * b4 * pow(c0, 2) * d5 +
          16 * a0 * a2 * b4 * pow(c2, 2) * d3 -
          4 * a0 * a2 * b4 * pow(c5, 2) * d0 +
          4 * a0 * a2 * b5 * pow(c0, 2) * d4 +
          8 * a0 * a3 * b1 * pow(c2, 2) * d5 +
          16 * a0 * a3 * b2 * pow(c2, 2) * d4 -
          48 * a0 * a3 * b4 * pow(c2, 2) * d2 +
          4 * a0 * a4 * b0 * pow(c5, 2) * d2 +
          4 * a0 * a4 * b2 * pow(c0, 2) * d5 +
          16 * a0 * a4 * b2 * pow(c2, 2) * d3 -
          4 * a0 * a4 * b2 * pow(c5, 2) * d0 -
          48 * a0 * a4 * b3 * pow(c2, 2) * d2 +
          32 * a0 * a4 * b4 * pow(c2, 2) * d1 +
          4 * a0 * a4 * b5 * pow(c0, 2) * d2 +
          2 * a0 * a5 * b0 * pow(c5, 2) * d1 -
          2 * a0 * a5 * b1 * pow(c0, 2) * d5 +
          2 * a0 * a5 * b1 * pow(c5, 2) * d0 +
          4 * a0 * a5 * b2 * pow(c0, 2) * d4 +
          4 * a0 * a5 * b4 * pow(c0, 2) * d2 -
          2 * a0 * a5 * b5 * pow(c0, 2) * d1 -
          32 * a1 * a2 * b2 * pow(c1, 2) * d5 -
          32 * a1 * a2 * b5 * pow(c1, 2) * d2 +
          8 * a1 * a3 * b0 * pow(c2, 2) * d5 -
          8 * a1 * a3 * b1 * pow(c5, 2) * d1 -
          8 * a1 * a3 * b5 * pow(c2, 2) * d0 -
          32 * a1 * a4 * b4 * pow(c2, 2) * d0 -
          2 * a1 * a5 * b0 * pow(c0, 2) * d5 +
          2 * a1 * a5 * b0 * pow(c5, 2) * d0 -
          32 * a1 * a5 * b2 * pow(c1, 2) * d2 -
          8 * a1 * a5 * b3 * pow(c2, 2) * d0 +
          6 * a1 * a5 * b5 * pow(c0, 2) * d0 +
          16 * a2 * a3 * b0 * pow(c2, 2) * d4 +
          16 * a2 * a3 * b4 * pow(c2, 2) * d0 +
          4 * a2 * a4 * b0 * pow(c0, 2) * d5 +
          16 * a2 * a4 * b0 * pow(c2, 2) * d3 -
          4 * a2 * a4 * b0 * pow(c5, 2) * d0 +
          64 * a2 * a4 * b2 * pow(c1, 2) * d2 +
          16 * a2 * a4 * b3 * pow(c2, 2) * d0 -
          12 * a2 * a4 * b5 * pow(c0, 2) * d0 +
          4 * a2 * a5 * b0 * pow(c0, 2) * d4 -
          32 * a2 * a5 * b1 * pow(c1, 2) * d2 +
          96 * a2 * a5 * b2 * pow(c1, 2) * d1 -
          12 * a2 * a5 * b4 * pow(c0, 2) * d0 -
          48 * a3 * a4 * b0 * pow(c2, 2) * d2 +
          16 * a3 * a4 * b2 * pow(c2, 2) * d0 -
          8 * a3 * a5 * b1 * pow(c2, 2) * d0 +
          4 * a4 * a5 * b0 * pow(c0, 2) * d2 -
          12 * a4 * a5 * b2 * pow(c0, 2) * d0 -
          2 * a0 * a1 * b3 * pow(c5, 2) * d3 +
          16 * a0 * a1 * b5 * pow(c1, 2) * d5 +
          16 * a0 * a2 * b5 * pow(c1, 2) * d4 -
          2 * a0 * a3 * b1 * pow(c5, 2) * d3 -
          2 * a0 * a3 * b3 * pow(c5, 2) * d1 +
          16 * a0 * a5 * b1 * pow(c1, 2) * d5 +
          16 * a0 * a5 * b2 * pow(c1, 2) * d4 -
          48 * a0 * a5 * b5 * pow(c1, 2) * d1 +
          32 * a1 * a2 * b2 * pow(c4, 2) * d3 -
          2 * a1 * a3 * b0 * pow(c5, 2) * d3 -
          6 * a1 * a3 * b3 * pow(c0, 2) * d5 +
          6 * a1 * a3 * b3 * pow(c5, 2) * d0 +
          2 * a1 * a3 * b5 * pow(c0, 2) * d3 +
          8 * a1 * a4 * b4 * pow(c0, 2) * d3 +
          16 * a1 * a5 * b0 * pow(c1, 2) * d5 +
          2 * a1 * a5 * b3 * pow(c0, 2) * d3 +
          16 * a1 * a5 * b5 * pow(c1, 2) * d0 -
          32 * a2 * a3 * b2 * pow(c4, 2) * d1 +
          4 * a2 * a3 * b3 * pow(c0, 2) * d4 -
          4 * a2 * a3 * b4 * pow(c0, 2) * d3 -
          4 * a2 * a4 * b3 * pow(c0, 2) * d3 -
          16 * a2 * a4 * b5 * pow(c1, 2) * d0 +
          16 * a2 * a5 * b0 * pow(c1, 2) * d4 -
          16 * a2 * a5 * b4 * pow(c1, 2) * d0 -
          4 * a3 * a4 * b2 * pow(c0, 2) * d3 +
          4 * a3 * a4 * b3 * pow(c0, 2) * d2 -
          8 * a3 * a4 * b4 * pow(c0, 2) * d1 +
          2 * a3 * a5 * b1 * pow(c0, 2) * d3 +
          2 * a3 * a5 * b3 * pow(c0, 2) * d1 -
          16 * a4 * a5 * b2 * pow(c1, 2) * d0 -
          8 * a0 * a1 * b5 * pow(c4, 2) * d3 +
          8 * a0 * a3 * b5 * pow(c4, 2) * d1 -
          8 * a0 * a5 * b1 * pow(c4, 2) * d3 +
          8 * a0 * a5 * b3 * pow(c4, 2) * d1 -
          8 * a0 * a5 * b5 * pow(c2, 2) * d1 +
          16 * a1 * a2 * b1 * pow(c5, 2) * d4 +
          8 * a1 * a2 * b2 * pow(c3, 2) * d5 -
          8 * a1 * a2 * b2 * pow(c5, 2) * d3 -
          16 * a1 * a2 * b4 * pow(c5, 2) * d1 -
          8 * a1 * a2 * b5 * pow(c3, 2) * d2 +
          16 * a1 * a4 * b1 * pow(c5, 2) * d2 -
          16 * a1 * a4 * b2 * pow(c5, 2) * d1 -
          8 * a1 * a5 * b0 * pow(c4, 2) * d3 +
          24 * a1 * a5 * b1 * pow(c5, 2) * d1 -
          8 * a1 * a5 * b2 * pow(c3, 2) * d2 +
          8 * a1 * a5 * b5 * pow(c2, 2) * d0 +
          8 * a2 * a3 * b2 * pow(c5, 2) * d1 -
          16 * a2 * a4 * b1 * pow(c5, 2) * d1 +
          16 * a2 * a4 * b2 * pow(c3, 2) * d2 -
          8 * a2 * a5 * b1 * pow(c3, 2) * d2 +
          8 * a2 * a5 * b2 * pow(c3, 2) * d1 +
          8 * a3 * a5 * b0 * pow(c4, 2) * d1 +
          6 * a0 * a1 * b3 * pow(c5, 2) * d5 +
          2 * a0 * a1 * b5 * pow(c3, 2) * d5 -
          2 * a0 * a1 * b5 * pow(c5, 2) * d3 -
          4 * a0 * a2 * b3 * pow(c5, 2) * d4 -
          4 * a0 * a2 * b4 * pow(c3, 2) * d5 +
          4 * a0 * a2 * b4 * pow(c5, 2) * d3 +
          4 * a0 * a2 * b5 * pow(c3, 2) * d4 +
          6 * a0 * a3 * b1 * pow(c5, 2) * d5 -
          4 * a0 * a3 * b2 * pow(c5, 2) * d4 -
          4 * a0 * a3 * b4 * pow(c5, 2) * d2 -
          2 * a0 * a3 * b5 * pow(c5, 2) * d1 -
          4 * a0 * a4 * b2 * pow(c3, 2) * d5 +
          4 * a0 * a4 * b2 * pow(c5, 2) * d3 -
          4 * a0 * a4 * b3 * pow(c5, 2) * d2 +
          8 * a0 * a4 * b4 * pow(c5, 2) * d1 +
          4 * a0 * a4 * b5 * pow(c3, 2) * d2 +
          2 * a0 * a5 * b1 * pow(c3, 2) * d5 -
          2 * a0 * a5 * b1 * pow(c5, 2) * d3 +
          4 * a0 * a5 * b2 * pow(c3, 2) * d4 -
          2 * a0 * a5 * b3 * pow(c5, 2) * d1 +
          4 * a0 * a5 * b4 * pow(c3, 2) * d2 -
          6 * a0 * a5 * b5 * pow(c3, 2) * d1 +
          6 * a1 * a3 * b0 * pow(c5, 2) * d5 -
          8 * a1 * a3 * b3 * pow(c2, 2) * d5 +
          4 * a1 * a3 * b5 * pow(c0, 2) * d5 +
          8 * a1 * a3 * b5 * pow(c2, 2) * d3 -
          2 * a1 * a3 * b5 * pow(c5, 2) * d0 +
          16 * a1 * a4 * b4 * pow(c0, 2) * d5 +
          32 * a1 * a4 * b4 * pow(c2, 2) * d3 -
          8 * a1 * a4 * b4 * pow(c5, 2) * d0 -
          8 * a1 * a4 * b5 * pow(c0, 2) * d4 +
          2 * a1 * a5 * b0 * pow(c3, 2) * d5 -
          2 * a1 * a5 * b0 * pow(c5, 2) * d3 +
          4 * a1 * a5 * b3 * pow(c0, 2) * d5 +
          8 * a1 * a5 * b3 * pow(c2, 2) * d3 -
          2 * a1 * a5 * b3 * pow(c5, 2) * d0 -
          8 * a1 * a5 * b4 * pow(c0, 2) * d4 -
          4 * a1 * a5 * b5 * pow(c0, 2) * d3 +
          2 * a1 * a5 * b5 * pow(c3, 2) * d0 -
          4 * a2 * a3 * b0 * pow(c5, 2) * d4 -
          16 * a2 * a3 * b3 * pow(c2, 2) * d4 -
          8 * a2 * a3 * b4 * pow(c0, 2) * d5 -
          16 * a2 * a3 * b4 * pow(c2, 2) * d3 +
          4 * a2 * a3 * b4 * pow(c5, 2) * d0 -
          4 * a2 * a4 * b0 * pow(c3, 2) * d5 +
          4 * a2 * a4 * b0 * pow(c5, 2) * d3 -
          8 * a2 * a4 * b3 * pow(c0, 2) * d5 -
          16 * a2 * a4 * b3 * pow(c2, 2) * d3 +
          4 * a2 * a4 * b3 * pow(c5, 2) * d0 +
          8 * a2 * a4 * b5 * pow(c0, 2) * d3 -
          4 * a2 * a4 * b5 * pow(c3, 2) * d0 +
          4 * a2 * a5 * b0 * pow(c3, 2) * d4 +
          8 * a2 * a5 * b4 * pow(c0, 2) * d3 -
          4 * a2 * a5 * b4 * pow(c3, 2) * d0 -
          4 * a3 * a4 * b0 * pow(c5, 2) * d2 -
          8 * a3 * a4 * b2 * pow(c0, 2) * d5 -
          16 * a3 * a4 * b2 * pow(c2, 2) * d3 +
          4 * a3 * a4 * b2 * pow(c5, 2) * d0 +
          48 * a3 * a4 * b3 * pow(c2, 2) * d2 -
          32 * a3 * a4 * b4 * pow(c2, 2) * d1 -
          2 * a3 * a5 * b0 * pow(c5, 2) * d1 +
          4 * a3 * a5 * b1 * pow(c0, 2) * d5 +
          8 * a3 * a5 * b1 * pow(c2, 2) * d3 -
          2 * a3 * a5 * b1 * pow(c5, 2) * d0 -
          8 * a3 * a5 * b3 * pow(c2, 2) * d1 -
          4 * a3 * a5 * b5 * pow(c0, 2) * d1 +
          4 * a4 * a5 * b0 * pow(c3, 2) * d2 -
          8 * a4 * a5 * b1 * pow(c0, 2) * d4 +
          8 * a4 * a5 * b2 * pow(c0, 2) * d3 -
          4 * a4 * a5 * b2 * pow(c3, 2) * d0 +
          8 * a0 * a5 * b5 * pow(c4, 2) * d1 +
          8 * a1 * a3 * b5 * pow(c1, 2) * d5 +
          8 * a1 * a5 * b3 * pow(c1, 2) * d5 +
          8 * a1 * a5 * b5 * pow(c1, 2) * d3 -
          8 * a1 * a5 * b5 * pow(c4, 2) * d0 -
          16 * a2 * a3 * b4 * pow(c1, 2) * d5 -
          16 * a2 * a4 * b3 * pow(c1, 2) * d5 -
          16 * a3 * a4 * b2 * pow(c1, 2) * d5 +
          16 * a3 * a4 * b5 * pow(c1, 2) * d2 +
          8 * a3 * a5 * b1 * pow(c1, 2) * d5 +
          16 * a3 * a5 * b4 * pow(c1, 2) * d2 -
          24 * a3 * a5 * b5 * pow(c1, 2) * d1 +
          16 * a4 * a5 * b3 * pow(c1, 2) * d2 -
          2 * a1 * a5 * b5 * pow(c0, 2) * d5 -
          8 * a1 * a5 * b5 * pow(c2, 2) * d3 +
          4 * a2 * a4 * b5 * pow(c0, 2) * d5 +
          4 * a2 * a5 * b4 * pow(c0, 2) * d5 -
          4 * a2 * a5 * b5 * pow(c0, 2) * d4 +
          8 * a3 * a5 * b5 * pow(c2, 2) * d1 +
          4 * a4 * a5 * b2 * pow(c0, 2) * d5 -
          4 * a4 * a5 * b5 * pow(c0, 2) * d2 -
          6 * a1 * a3 * b3 * pow(c5, 2) * d5 +
          2 * a1 * a3 * b5 * pow(c5, 2) * d3 +
          8 * a1 * a4 * b4 * pow(c5, 2) * d3 +
          2 * a1 * a5 * b3 * pow(c5, 2) * d3 -
          24 * a1 * a5 * b5 * pow(c1, 2) * d5 +
          4 * a2 * a3 * b3 * pow(c5, 2) * d4 -
          4 * a2 * a3 * b4 * pow(c5, 2) * d3 -
          4 * a2 * a4 * b3 * pow(c5, 2) * d3 +
          16 * a2 * a4 * b5 * pow(c1, 2) * d5 +
          16 * a2 * a5 * b4 * pow(c1, 2) * d5 -
          16 * a2 * a5 * b5 * pow(c1, 2) * d4 -
          4 * a3 * a4 * b2 * pow(c5, 2) * d3 +
          4 * a3 * a4 * b3 * pow(c5, 2) * d2 -
          8 * a3 * a4 * b4 * pow(c5, 2) * d1 +
          2 * a3 * a5 * b1 * pow(c5, 2) * d3 +
          2 * a3 * a5 * b3 * pow(c5, 2) * d1 +
          16 * a4 * a5 * b2 * pow(c1, 2) * d5 -
          16 * a4 * a5 * b5 * pow(c1, 2) * d2 +
          8 * a1 * a5 * b5 * pow(c4, 2) * d3 -
          8 * a3 * a5 * b5 * pow(c4, 2) * d1 -
          2 * a1 * a5 * b5 * pow(c3, 2) * d5 +
          4 * a2 * a4 * b5 * pow(c3, 2) * d5 +
          4 * a2 * a5 * b4 * pow(c3, 2) * d5 -
          4 * a2 * a5 * b5 * pow(c3, 2) * d4 +
          4 * a4 * a5 * b2 * pow(c3, 2) * d5 -
          4 * a4 * a5 * b5 * pow(c3, 2) * d2 -
          2 * pow(a3, 2) * b0 * c0 * c1 * d0 -
          8 * pow(a1, 2) * b0 * c0 * c1 * d3 -
          8 * pow(a1, 2) * b0 * c0 * c3 * d1 -
          8 * pow(a1, 2) * b0 * c1 * c3 * d0 +
          24 * pow(a1, 2) * b1 * c0 * c3 * d0 -
          8 * pow(a1, 2) * b3 * c0 * c1 * d0 +
          8 * pow(a4, 2) * b0 * c0 * c1 * d0 +
          8 * pow(a0, 2) * b1 * c1 * c3 * d1 +
          8 * pow(a2, 2) * b1 * c0 * c3 * d0 -
          8 * pow(a2, 2) * b3 * c0 * c1 * d0 -
          2 * pow(a5, 2) * b0 * c0 * c1 * d0 -
          6 * pow(a0, 2) * b0 * c1 * c3 * d3 +
          2 * pow(a0, 2) * b1 * c0 * c3 * d3 +
          2 * pow(a0, 2) * b3 * c0 * c1 * d3 +
          2 * pow(a0, 2) * b3 * c0 * c3 * d1 +
          2 * pow(a0, 2) * b3 * c1 * c3 * d0 +
          8 * pow(a1, 2) * b0 * c0 * c1 * d5 +
          8 * pow(a1, 2) * b0 * c0 * c5 * d1 +
          8 * pow(a1, 2) * b0 * c1 * c5 * d0 -
          24 * pow(a1, 2) * b1 * c0 * c5 * d0 -
          16 * pow(a1, 2) * b2 * c0 * c4 * d0 +
          16 * pow(a1, 2) * b4 * c0 * c2 * d0 +
          8 * pow(a1, 2) * b5 * c0 * c1 * d0 -
          8 * pow(a0, 2) * b1 * c1 * c5 * d1 +
          8 * pow(a0, 2) * b1 * c2 * c3 * d2 -
          16 * pow(a0, 2) * b2 * c1 * c4 * d1 -
          8 * pow(a0, 2) * b3 * c1 * c2 * d2 +
          16 * pow(a0, 2) * b4 * c1 * c2 * d1 +
          16 * pow(a2, 2) * b0 * c0 * c2 * d4 +
          16 * pow(a2, 2) * b0 * c0 * c4 * d2 +
          16 * pow(a2, 2) * b0 * c2 * c4 * d0 -
          8 * pow(a2, 2) * b1 * c0 * c5 * d0 +
          32 * pow(a2, 2) * b1 * c1 * c3 * d1 -
          48 * pow(a2, 2) * b2 * c0 * c4 * d0 +
          16 * pow(a2, 2) * b4 * c0 * c2 * d0 +
          8 * pow(a2, 2) * b5 * c0 * c1 * d0 +
          8 * pow(a3, 2) * b0 * c1 * c2 * d2 +
          8 * pow(a3, 2) * b1 * c0 * c2 * d2 -
          8 * pow(a3, 2) * b2 * c0 * c1 * d2 -
          8 * pow(a3, 2) * b2 * c0 * c2 * d1 -
          8 * pow(a3, 2) * b2 * c1 * c2 * d0 +
          8 * pow(a4, 2) * b1 * c0 * c3 * d0 -
          8 * pow(a4, 2) * b3 * c0 * c1 * d0 +
          16 * pow(a5, 2) * b1 * c0 * c1 * d1 +
          6 * pow(a0, 2) * b0 * c1 * c3 * d5 +
          24 * pow(a0, 2) * b0 * c1 * c4 * d4 +
          6 * pow(a0, 2) * b0 * c1 * c5 * d3 -
          12 * pow(a0, 2) * b0 * c2 * c3 * d4 -
          12 * pow(a0, 2) * b0 * c2 * c4 * d3 -
          12 * pow(a0, 2) * b0 * c3 * c4 * d2 +
          6 * pow(a0, 2) * b0 * c3 * c5 * d1 -
          2 * pow(a0, 2) * b1 * c0 * c3 * d5 -
          8 * pow(a0, 2) * b1 * c0 * c4 * d4 -
          2 * pow(a0, 2) * b1 * c0 * c5 * d3 -
          2 * pow(a0, 2) * b1 * c3 * c5 * d0 +
          4 * pow(a0, 2) * b2 * c0 * c3 * d4 +
          4 * pow(a0, 2) * b2 * c0 * c4 * d3 +
          4 * pow(a0, 2) * b2 * c3 * c4 * d0 -
          2 * pow(a0, 2) * b3 * c0 * c1 * d5 +
          4 * pow(a0, 2) * b3 * c0 * c2 * d4 +
          4 * pow(a0, 2) * b3 * c0 * c4 * d2 -
          2 * pow(a0, 2) * b3 * c0 * c5 * d1 -
          2 * pow(a0, 2) * b3 * c1 * c5 * d0 +
          4 * pow(a0, 2) * b3 * c2 * c4 * d0 -
          8 * pow(a0, 2) * b4 * c0 * c1 * d4 +
          4 * pow(a0, 2) * b4 * c0 * c2 * d3 +
          4 * pow(a0, 2) * b4 * c0 * c3 * d2 -
          8 * pow(a0, 2) * b4 * c0 * c4 * d1 -
          8 * pow(a0, 2) * b4 * c1 * c4 * d0 +
          4 * pow(a0, 2) * b4 * c2 * c3 * d0 -
          2 * pow(a0, 2) * b5 * c0 * c1 * d3 -
          2 * pow(a0, 2) * b5 * c0 * c3 * d1 -
          2 * pow(a0, 2) * b5 * c1 * c3 * d0 +
          96 * pow(a1, 2) * b1 * c2 * c3 * d2 -
          32 * pow(a1, 2) * b2 * c1 * c2 * d3 -
          32 * pow(a1, 2) * b2 * c1 * c3 * d2 -
          32 * pow(a1, 2) * b2 * c2 * c3 * d1 -
          32 * pow(a1, 2) * b3 * c1 * c2 * d2 -
          8 * pow(a2, 2) * b0 * c1 * c3 * d3 -
          8 * pow(a2, 2) * b1 * c0 * c3 * d3 +
          8 * pow(a2, 2) * b3 * c0 * c1 * d3 +
          8 * pow(a2, 2) * b3 * c0 * c3 * d1 +
          8 * pow(a2, 2) * b3 * c1 * c3 * d0 +
          2 * pow(a3, 2) * b0 * c0 * c1 * d5 -
          4 * pow(a3, 2) * b0 * c0 * c2 * d4 -
          4 * pow(a3, 2) * b0 * c0 * c4 * d2 +
          2 * pow(a3, 2) * b0 * c0 * c5 * d1 +
          2 * pow(a3, 2) * b0 * c1 * c5 * d0 -
          4 * pow(a3, 2) * b0 * c2 * c4 * d0 -
          6 * pow(a3, 2) * b1 * c0 * c5 * d0 +
          4 * pow(a3, 2) * b2 * c0 * c4 * d0 +
          4 * pow(a3, 2) * b4 * c0 * c2 * d0 +
          2 * pow(a3, 2) * b5 * c0 * c1 * d0 +
          32 * pow(a4, 2) * b0 * c1 * c2 * d2 -
          32 * pow(a4, 2) * b1 * c0 * c2 * d2 +
          4 * pow(a5, 2) * b0 * c0 * c1 * d3 +
          4 * pow(a5, 2) * b0 * c0 * c3 * d1 +
          4 * pow(a5, 2) * b0 * c1 * c3 * d0 -
          4 * pow(a5, 2) * b1 * c0 * c3 * d0 -
          4 * pow(a5, 2) * b3 * c0 * c1 * d0 -
          8 * pow(a0, 2) * b1 * c2 * c5 * d2 -
          16 * pow(a0, 2) * b2 * c2 * c4 * d2 +
          8 * pow(a0, 2) * b5 * c1 * c2 * d2 +
          8 * pow(a1, 2) * b0 * c1 * c3 * d5 +
          8 * pow(a1, 2) * b0 * c1 * c5 * d3 -
          16 * pow(a1, 2) * b0 * c2 * c3 * d4 -
          16 * pow(a1, 2) * b0 * c2 * c4 * d3 -
          16 * pow(a1, 2) * b0 * c3 * c4 * d2 +
          8 * pow(a1, 2) * b0 * c3 * c5 * d1 -
          24 * pow(a1, 2) * b1 * c0 * c3 * d5 -
          24 * pow(a1, 2) * b1 * c0 * c5 * d3 -
          24 * pow(a1, 2) * b1 * c3 * c5 * d0 +
          16 * pow(a1, 2) * b2 * c0 * c3 * d4 +
          16 * pow(a1, 2) * b2 * c0 * c4 * d3 +
          16 * pow(a1, 2) * b2 * c3 * c4 * d0 +
          8 * pow(a1, 2) * b3 * c0 * c1 * d5 +
          8 * pow(a1, 2) * b3 * c0 * c5 * d1 +
          8 * pow(a1, 2) * b3 * c1 * c5 * d0 +
          8 * pow(a1, 2) * b5 * c0 * c1 * d3 +
          8 * pow(a1, 2) * b5 * c0 * c3 * d1 +
          8 * pow(a1, 2) * b5 * c1 * c3 * d0 +
          64 * pow(a2, 2) * b1 * c1 * c2 * d4 +
          64 * pow(a2, 2) * b1 * c1 * c4 * d2 -
          32 * pow(a2, 2) * b1 * c1 * c5 * d1 +
          64 * pow(a2, 2) * b1 * c2 * c4 * d1 -
          192 * pow(a2, 2) * b2 * c1 * c4 * d1 +
          64 * pow(a2, 2) * b4 * c1 * c2 * d1 -
          8 * pow(a4, 2) * b0 * c0 * c1 * d5 -
          8 * pow(a4, 2) * b0 * c0 * c5 * d1 -
          8 * pow(a4, 2) * b0 * c1 * c5 * d0 +
          16 * pow(a4, 2) * b1 * c0 * c5 * d0 -
          8 * pow(a5, 2) * b0 * c1 * c2 * d2 +
          8 * pow(a5, 2) * b1 * c0 * c2 * d2 -
          6 * pow(a0, 2) * b0 * c1 * c5 * d5 +
          12 * pow(a0, 2) * b0 * c2 * c4 * d5 +
          12 * pow(a0, 2) * b0 * c2 * c5 * d4 +
          12 * pow(a0, 2) * b0 * c4 * c5 * d2 +
          2 * pow(a0, 2) * b1 * c0 * c5 * d5 -
          4 * pow(a0, 2) * b2 * c0 * c4 * d5 -
          4 * pow(a0, 2) * b2 * c0 * c5 * d4 -
          4 * pow(a0, 2) * b2 * c4 * c5 * d0 -
          4 * pow(a0, 2) * b4 * c0 * c2 * d5 -
          4 * pow(a0, 2) * b4 * c0 * c5 * d2 -
          4 * pow(a0, 2) * b4 * c2 * c5 * d0 +
          2 * pow(a0, 2) * b5 * c0 * c1 * d5 -
          4 * pow(a0, 2) * b5 * c0 * c2 * d4 -
          4 * pow(a0, 2) * b5 * c0 * c4 * d2 +
          2 * pow(a0, 2) * b5 * c0 * c5 * d1 +
          2 * pow(a0, 2) * b5 * c1 * c5 * d0 -
          4 * pow(a0, 2) * b5 * c2 * c4 * d0 -
          96 * pow(a1, 2) * b1 * c2 * c5 * d2 +
          32 * pow(a1, 2) * b2 * c1 * c2 * d5 +
          32 * pow(a1, 2) * b2 * c1 * c5 * d2 -
          64 * pow(a1, 2) * b2 * c2 * c4 * d2 +
          32 * pow(a1, 2) * b2 * c2 * c5 * d1 +
          32 * pow(a1, 2) * b5 * c1 * c2 * d2 +
          8 * pow(a2, 2) * b0 * c1 * c3 * d5 +
          32 * pow(a2, 2) * b0 * c1 * c4 * d4 +
          8 * pow(a2, 2) * b0 * c1 * c5 * d3 -
          16 * pow(a2, 2) * b0 * c2 * c3 * d4 -
          16 * pow(a2, 2) * b0 * c2 * c4 * d3 -
          16 * pow(a2, 2) * b0 * c3 * c4 * d2 +
          8 * pow(a2, 2) * b0 * c3 * c5 * d1 -
          32 * pow(a2, 2) * b1 * c0 * c4 * d4 +
          48 * pow(a2, 2) * b2 * c0 * c3 * d4 +
          48 * pow(a2, 2) * b2 * c0 * c4 * d3 +
          48 * pow(a2, 2) * b2 * c3 * c4 * d0 -
          16 * pow(a2, 2) * b3 * c0 * c2 * d4 -
          16 * pow(a2, 2) * b3 * c0 * c4 * d2 -
          16 * pow(a2, 2) * b3 * c2 * c4 * d0 -
          16 * pow(a2, 2) * b4 * c0 * c2 * d3 -
          16 * pow(a2, 2) * b4 * c0 * c3 * d2 -
          16 * pow(a2, 2) * b4 * c2 * c3 * d0 -
          8 * pow(a2, 2) * b5 * c0 * c1 * d3 -
          8 * pow(a2, 2) * b5 * c0 * c3 * d1 -
          8 * pow(a2, 2) * b5 * c1 * c3 * d0 -
          2 * pow(a5, 2) * b0 * c0 * c1 * d5 +
          4 * pow(a5, 2) * b0 * c0 * c2 * d4 +
          4 * pow(a5, 2) * b0 * c0 * c4 * d2 -
          2 * pow(a5, 2) * b0 * c0 * c5 * d1 -
          2 * pow(a5, 2) * b0 * c1 * c5 * d0 +
          4 * pow(a5, 2) * b0 * c2 * c4 * d0 -
          2 * pow(a5, 2) * b1 * c0 * c5 * d0 +
          8 * pow(a5, 2) * b1 * c1 * c3 * d1 -
          4 * pow(a5, 2) * b2 * c0 * c4 * d0 -
          4 * pow(a5, 2) * b4 * c0 * c2 * d0 +
          6 * pow(a5, 2) * b5 * c0 * c1 * d0 +
          8 * pow(a0, 2) * b1 * c3 * c4 * d4 -
          2 * pow(a0, 2) * b1 * c3 * c5 * d3 -
          4 * pow(a0, 2) * b2 * c3 * c4 * d3 -
          2 * pow(a0, 2) * b3 * c1 * c3 * d5 -
          8 * pow(a0, 2) * b3 * c1 * c4 * d4 -
          2 * pow(a0, 2) * b3 * c1 * c5 * d3 +
          4 * pow(a0, 2) * b3 * c2 * c3 * d4 +
          4 * pow(a0, 2) * b3 * c2 * c4 * d3 +
          4 * pow(a0, 2) * b3 * c3 * c4 * d2 -
          2 * pow(a0, 2) * b3 * c3 * c5 * d1 -
          4 * pow(a0, 2) * b4 * c2 * c3 * d3 +
          6 * pow(a0, 2) * b5 * c1 * c3 * d3 -
          16 * pow(a1, 2) * b0 * c1 * c5 * d5 +
          16 * pow(a1, 2) * b0 * c2 * c4 * d5 +
          16 * pow(a1, 2) * b0 * c2 * c5 * d4 +
          16 * pow(a1, 2) * b0 * c4 * c5 * d2 +
          48 * pow(a1, 2) * b1 * c0 * c5 * d5 -
          16 * pow(a1, 2) * b4 * c0 * c2 * d5 -
          16 * pow(a1, 2) * b4 * c0 * c5 * d2 -
          16 * pow(a1, 2) * b4 * c2 * c5 * d0 -
          16 * pow(a1, 2) * b5 * c0 * c1 * d5 -
          16 * pow(a1, 2) * b5 * c0 * c5 * d1 -
          16 * pow(a1, 2) * b5 * c1 * c5 * d0 +
          32 * pow(a4, 2) * b1 * c2 * c3 * d2 -
          32 * pow(a4, 2) * b3 * c1 * c2 * d2 -
          6 * pow(a5, 2) * b0 * c1 * c3 * d3 +
          2 * pow(a5, 2) * b1 * c0 * c3 * d3 +
          2 * pow(a5, 2) * b3 * c0 * c1 * d3 +
          2 * pow(a5, 2) * b3 * c0 * c3 * d1 +
          2 * pow(a5, 2) * b3 * c1 * c3 * d0 -
          8 * pow(a2, 2) * b0 * c1 * c5 * d5 +
          8 * pow(a2, 2) * b1 * c0 * c5 * d5 -
          8 * pow(a3, 2) * b1 * c2 * c5 * d2 +
          8 * pow(a3, 2) * b2 * c1 * c2 * d5 +
          8 * pow(a3, 2) * b2 * c1 * c5 * d2 -
          16 * pow(a3, 2) * b2 * c2 * c4 * d2 +
          8 * pow(a3, 2) * b2 * c2 * c5 * d1 -
          8 * pow(a3, 2) * b5 * c1 * c2 * d2 -
          8 * pow(a4, 2) * b1 * c0 * c3 * d5 -
          8 * pow(a4, 2) * b1 * c0 * c5 * d3 -
          8 * pow(a4, 2) * b1 * c3 * c5 * d0 +
          8 * pow(a4, 2) * b3 * c0 * c1 * d5 +
          8 * pow(a4, 2) * b3 * c0 * c5 * d1 +
          8 * pow(a4, 2) * b3 * c1 * c5 * d0 +
          16 * pow(a5, 2) * b1 * c1 * c2 * d4 +
          16 * pow(a5, 2) * b1 * c1 * c4 * d2 -
          24 * pow(a5, 2) * b1 * c1 * c5 * d1 -
          8 * pow(a5, 2) * b1 * c2 * c3 * d2 +
          16 * pow(a5, 2) * b1 * c2 * c4 * d1 -
          16 * pow(a5, 2) * b2 * c1 * c4 * d1 +
          8 * pow(a5, 2) * b3 * c1 * c2 * d2 -
          16 * pow(a5, 2) * b4 * c1 * c2 * d1 +
          4 * pow(a0, 2) * b1 * c3 * c5 * d5 +
          4 * pow(a0, 2) * b3 * c1 * c5 * d5 -
          8 * pow(a0, 2) * b3 * c2 * c4 * d5 -
          8 * pow(a0, 2) * b3 * c2 * c5 * d4 -
          8 * pow(a0, 2) * b3 * c4 * c5 * d2 +
          8 * pow(a0, 2) * b4 * c1 * c4 * d5 +
          8 * pow(a0, 2) * b4 * c1 * c5 * d4 +
          8 * pow(a0, 2) * b4 * c4 * c5 * d1 -
          4 * pow(a0, 2) * b5 * c1 * c3 * d5 -
          16 * pow(a0, 2) * b5 * c1 * c4 * d4 -
          4 * pow(a0, 2) * b5 * c1 * c5 * d3 +
          8 * pow(a0, 2) * b5 * c2 * c3 * d4 +
          8 * pow(a0, 2) * b5 * c2 * c4 * d3 +
          8 * pow(a0, 2) * b5 * c3 * c4 * d2 -
          4 * pow(a0, 2) * b5 * c3 * c5 * d1 +
          32 * pow(a2, 2) * b1 * c3 * c4 * d4 +
          8 * pow(a2, 2) * b1 * c3 * c5 * d3 -
          48 * pow(a2, 2) * b2 * c3 * c4 * d3 -
          8 * pow(a2, 2) * b3 * c1 * c3 * d5 -
          32 * pow(a2, 2) * b3 * c1 * c4 * d4 -
          8 * pow(a2, 2) * b3 * c1 * c5 * d3 +
          16 * pow(a2, 2) * b3 * c2 * c3 * d4 +
          16 * pow(a2, 2) * b3 * c2 * c4 * d3 +
          16 * pow(a2, 2) * b3 * c3 * c4 * d2 -
          8 * pow(a2, 2) * b3 * c3 * c5 * d1 +
          16 * pow(a2, 2) * b4 * c2 * c3 * d3 +
          8 * pow(a2, 2) * b5 * c1 * c3 * d3 -
          2 * pow(a3, 2) * b0 * c1 * c5 * d5 +
          4 * pow(a3, 2) * b0 * c2 * c4 * d5 +
          4 * pow(a3, 2) * b0 * c2 * c5 * d4 +
          4 * pow(a3, 2) * b0 * c4 * c5 * d2 +
          6 * pow(a3, 2) * b1 * c0 * c5 * d5 -
          4 * pow(a3, 2) * b2 * c0 * c4 * d5 -
          4 * pow(a3, 2) * b2 * c0 * c5 * d4 -
          4 * pow(a3, 2) * b2 * c4 * c5 * d0 -
          4 * pow(a3, 2) * b4 * c0 * c2 * d5 -
          4 * pow(a3, 2) * b4 * c0 * c5 * d2 -
          4 * pow(a3, 2) * b4 * c2 * c5 * d0 -
          2 * pow(a3, 2) * b5 * c0 * c1 * d5 +
          4 * pow(a3, 2) * b5 * c0 * c2 * d4 +
          4 * pow(a3, 2) * b5 * c0 * c4 * d2 -
          2 * pow(a3, 2) * b5 * c0 * c5 * d1 -
          2 * pow(a3, 2) * b5 * c1 * c5 * d0 +
          4 * pow(a3, 2) * b5 * c2 * c4 * d0 +
          2 * pow(a5, 2) * b0 * c1 * c3 * d5 +
          8 * pow(a5, 2) * b0 * c1 * c4 * d4 +
          2 * pow(a5, 2) * b0 * c1 * c5 * d3 -
          4 * pow(a5, 2) * b0 * c2 * c3 * d4 -
          4 * pow(a5, 2) * b0 * c2 * c4 * d3 -
          4 * pow(a5, 2) * b0 * c3 * c4 * d2 +
          2 * pow(a5, 2) * b0 * c3 * c5 * d1 +
          2 * pow(a5, 2) * b1 * c0 * c3 * d5 -
          8 * pow(a5, 2) * b1 * c0 * c4 * d4 +
          2 * pow(a5, 2) * b1 * c0 * c5 * d3 +
          2 * pow(a5, 2) * b1 * c3 * c5 * d0 +
          4 * pow(a5, 2) * b2 * c0 * c3 * d4 +
          4 * pow(a5, 2) * b2 * c0 * c4 * d3 +
          4 * pow(a5, 2) * b2 * c3 * c4 * d0 +
          2 * pow(a5, 2) * b3 * c0 * c1 * d5 -
          4 * pow(a5, 2) * b3 * c0 * c2 * d4 -
          4 * pow(a5, 2) * b3 * c0 * c4 * d2 +
          2 * pow(a5, 2) * b3 * c0 * c5 * d1 +
          2 * pow(a5, 2) * b3 * c1 * c5 * d0 -
          4 * pow(a5, 2) * b3 * c2 * c4 * d0 +
          4 * pow(a5, 2) * b4 * c0 * c2 * d3 +
          4 * pow(a5, 2) * b4 * c0 * c3 * d2 +
          4 * pow(a5, 2) * b4 * c2 * c3 * d0 -
          6 * pow(a5, 2) * b5 * c0 * c1 * d3 -
          6 * pow(a5, 2) * b5 * c0 * c3 * d1 -
          6 * pow(a5, 2) * b5 * c1 * c3 * d0 +
          24 * pow(a1, 2) * b1 * c3 * c5 * d5 -
          16 * pow(a1, 2) * b2 * c3 * c4 * d5 -
          16 * pow(a1, 2) * b2 * c3 * c5 * d4 -
          16 * pow(a1, 2) * b2 * c4 * c5 * d3 -
          8 * pow(a1, 2) * b3 * c1 * c5 * d5 -
          8 * pow(a1, 2) * b5 * c1 * c3 * d5 -
          8 * pow(a1, 2) * b5 * c1 * c5 * d3 +
          16 * pow(a1, 2) * b5 * c2 * c3 * d4 +
          16 * pow(a1, 2) * b5 * c2 * c4 * d3 +
          16 * pow(a1, 2) * b5 * c3 * c4 * d2 -
          8 * pow(a1, 2) * b5 * c3 * c5 * d1 +
          8 * pow(a4, 2) * b0 * c1 * c5 * d5 -
          8 * pow(a4, 2) * b1 * c0 * c5 * d5 +
          4 * pow(a0, 2) * b2 * c4 * c5 * d5 +
          4 * pow(a0, 2) * b4 * c2 * c5 * d5 +
          2 * pow(a0, 2) * b5 * c1 * c5 * d5 -
          4 * pow(a0, 2) * b5 * c2 * c4 * d5 -
          4 * pow(a0, 2) * b5 * c2 * c5 * d4 -
          4 * pow(a0, 2) * b5 * c4 * c5 * d2 -
          8 * pow(a2, 2) * b1 * c3 * c5 * d5 +
          8 * pow(a2, 2) * b3 * c1 * c5 * d5 +
          16 * pow(a1, 2) * b2 * c4 * c5 * d5 +
          16 * pow(a1, 2) * b4 * c2 * c5 * d5 +
          24 * pow(a1, 2) * b5 * c1 * c5 * d5 -
          16 * pow(a1, 2) * b5 * c2 * c4 * d5 -
          16 * pow(a1, 2) * b5 * c2 * c5 * d4 -
          16 * pow(a1, 2) * b5 * c4 * c5 * d2 +
          8 * pow(a5, 2) * b1 * c3 * c4 * d4 -
          2 * pow(a5, 2) * b1 * c3 * c5 * d3 -
          4 * pow(a5, 2) * b2 * c3 * c4 * d3 -
          2 * pow(a5, 2) * b3 * c1 * c3 * d5 -
          8 * pow(a5, 2) * b3 * c1 * c4 * d4 -
          2 * pow(a5, 2) * b3 * c1 * c5 * d3 +
          4 * pow(a5, 2) * b3 * c2 * c3 * d4 +
          4 * pow(a5, 2) * b3 * c2 * c4 * d3 +
          4 * pow(a5, 2) * b3 * c3 * c4 * d2 -
          2 * pow(a5, 2) * b3 * c3 * c5 * d1 -
          4 * pow(a5, 2) * b4 * c2 * c3 * d3 +
          6 * pow(a5, 2) * b5 * c1 * c3 * d3 +
          8 * pow(a4, 2) * b1 * c3 * c5 * d5 -
          8 * pow(a4, 2) * b3 * c1 * c5 * d5 +
          4 * pow(a3, 2) * b2 * c4 * c5 * d5 +
          4 * pow(a3, 2) * b4 * c2 * c5 * d5 +
          2 * pow(a3, 2) * b5 * c1 * c5 * d5 -
          4 * pow(a3, 2) * b5 * c2 * c4 * d5 -
          4 * pow(a3, 2) * b5 * c2 * c5 * d4 -
          4 * pow(a3, 2) * b5 * c4 * c5 * d2 +
          16 * a0 * a1 * b0 * c1 * c3 * d1 - 16 * a0 * a1 * b1 * c0 * c1 * d3 -
          16 * a0 * a1 * b1 * c0 * c3 * d1 - 16 * a0 * a1 * b1 * c1 * c3 * d0 +
          16 * a0 * a1 * b3 * c0 * c1 * d1 + 16 * a0 * a3 * b1 * c0 * c1 * d1 +
          16 * a1 * a3 * b0 * c0 * c1 * d1 - 16 * a1 * a3 * b1 * c0 * c1 * d0 +
          4 * a0 * a1 * b0 * c0 * c3 * d3 - 4 * a0 * a1 * b3 * c0 * c3 * d0 +
          4 * a0 * a3 * b0 * c0 * c1 * d3 + 4 * a0 * a3 * b0 * c0 * c3 * d1 +
          4 * a0 * a3 * b0 * c1 * c3 * d0 - 4 * a0 * a3 * b1 * c0 * c3 * d0 -
          4 * a0 * a3 * b3 * c0 * c1 * d0 - 4 * a1 * a3 * b0 * c0 * c3 * d0 -
          16 * a0 * a1 * b0 * c1 * c5 * d1 + 16 * a0 * a1 * b0 * c2 * c3 * d2 +
          16 * a0 * a1 * b1 * c0 * c1 * d5 + 16 * a0 * a1 * b1 * c0 * c5 * d1 +
          16 * a0 * a1 * b1 * c1 * c5 * d0 + 16 * a0 * a1 * b2 * c0 * c1 * d4 -
          8 * a0 * a1 * b2 * c0 * c2 * d3 - 8 * a0 * a1 * b2 * c0 * c3 * d2 +
          16 * a0 * a1 * b2 * c0 * c4 * d1 + 16 * a0 * a1 * b2 * c1 * c4 * d0 -
          8 * a0 * a1 * b2 * c2 * c3 * d0 - 16 * a0 * a1 * b4 * c0 * c1 * d2 -
          16 * a0 * a1 * b4 * c0 * c2 * d1 - 16 * a0 * a1 * b4 * c1 * c2 * d0 -
          16 * a0 * a1 * b5 * c0 * c1 * d1 - 32 * a0 * a2 * b0 * c1 * c4 * d1 +
          16 * a0 * a2 * b1 * c0 * c1 * d4 - 8 * a0 * a2 * b1 * c0 * c2 * d3 -
          8 * a0 * a2 * b1 * c0 * c3 * d2 + 16 * a0 * a2 * b1 * c0 * c4 * d1 +
          16 * a0 * a2 * b1 * c1 * c4 * d0 - 8 * a0 * a2 * b1 * c2 * c3 * d0 +
          8 * a0 * a2 * b3 * c0 * c1 * d2 + 8 * a0 * a2 * b3 * c0 * c2 * d1 +
          8 * a0 * a2 * b3 * c1 * c2 * d0 - 16 * a0 * a3 * b0 * c1 * c2 * d2 +
          8 * a0 * a3 * b2 * c0 * c1 * d2 + 8 * a0 * a3 * b2 * c0 * c2 * d1 +
          8 * a0 * a3 * b2 * c1 * c2 * d0 + 32 * a0 * a4 * b0 * c1 * c2 * d1 -
          16 * a0 * a4 * b1 * c0 * c1 * d2 - 16 * a0 * a4 * b1 * c0 * c2 * d1 -
          16 * a0 * a4 * b1 * c1 * c2 * d0 - 16 * a0 * a5 * b1 * c0 * c1 * d1 +
          16 * a1 * a2 * b0 * c0 * c1 * d4 - 8 * a1 * a2 * b0 * c0 * c2 * d3 -
          8 * a1 * a2 * b0 * c0 * c3 * d2 + 16 * a1 * a2 * b0 * c0 * c4 * d1 +
          16 * a1 * a2 * b0 * c1 * c4 * d0 - 8 * a1 * a2 * b0 * c2 * c3 * d0 -
          32 * a1 * a2 * b1 * c0 * c4 * d0 + 16 * a1 * a2 * b2 * c0 * c3 * d0 -
          16 * a1 * a4 * b0 * c0 * c1 * d2 - 16 * a1 * a4 * b0 * c0 * c2 * d1 -
          16 * a1 * a4 * b0 * c1 * c2 * d0 + 32 * a1 * a4 * b1 * c0 * c2 * d0 -
          16 * a1 * a5 * b0 * c0 * c1 * d1 + 16 * a1 * a5 * b1 * c0 * c1 * d0 +
          8 * a2 * a3 * b0 * c0 * c1 * d2 + 8 * a2 * a3 * b0 * c0 * c2 * d1 +
          8 * a2 * a3 * b0 * c1 * c2 * d0 - 16 * a2 * a3 * b2 * c0 * c1 * d0 -
          4 * a0 * a1 * b0 * c0 * c3 * d5 - 16 * a0 * a1 * b0 * c0 * c4 * d4 -
          4 * a0 * a1 * b0 * c0 * c5 * d3 - 4 * a0 * a1 * b0 * c3 * c5 * d0 +
          4 * a0 * a1 * b3 * c0 * c5 * d0 + 16 * a0 * a1 * b4 * c0 * c4 * d0 +
          4 * a0 * a1 * b5 * c0 * c3 * d0 + 8 * a0 * a2 * b0 * c0 * c3 * d4 +
          8 * a0 * a2 * b0 * c0 * c4 * d3 + 8 * a0 * a2 * b0 * c3 * c4 * d0 -
          8 * a0 * a2 * b3 * c0 * c4 * d0 - 8 * a0 * a2 * b4 * c0 * c3 * d0 -
          4 * a0 * a3 * b0 * c0 * c1 * d5 + 8 * a0 * a3 * b0 * c0 * c2 * d4 +
          8 * a0 * a3 * b0 * c0 * c4 * d2 - 4 * a0 * a3 * b0 * c0 * c5 * d1 -
          4 * a0 * a3 * b0 * c1 * c5 * d0 + 8 * a0 * a3 * b0 * c2 * c4 * d0 +
          4 * a0 * a3 * b1 * c0 * c5 * d0 - 8 * a0 * a3 * b2 * c0 * c4 * d0 -
          8 * a0 * a3 * b4 * c0 * c2 * d0 + 4 * a0 * a3 * b5 * c0 * c1 * d0 -
          16 * a0 * a4 * b0 * c0 * c1 * d4 + 8 * a0 * a4 * b0 * c0 * c2 * d3 +
          8 * a0 * a4 * b0 * c0 * c3 * d2 - 16 * a0 * a4 * b0 * c0 * c4 * d1 -
          16 * a0 * a4 * b0 * c1 * c4 * d0 + 8 * a0 * a4 * b0 * c2 * c3 * d0 +
          16 * a0 * a4 * b1 * c0 * c4 * d0 - 8 * a0 * a4 * b2 * c0 * c3 * d0 -
          8 * a0 * a4 * b3 * c0 * c2 * d0 + 16 * a0 * a4 * b4 * c0 * c1 * d0 -
          4 * a0 * a5 * b0 * c0 * c1 * d3 - 4 * a0 * a5 * b0 * c0 * c3 * d1 -
          4 * a0 * a5 * b0 * c1 * c3 * d0 + 4 * a0 * a5 * b1 * c0 * c3 * d0 +
          4 * a0 * a5 * b3 * c0 * c1 * d0 + 4 * a1 * a3 * b0 * c0 * c5 * d0 +
          16 * a1 * a4 * b0 * c0 * c4 * d0 + 4 * a1 * a5 * b0 * c0 * c3 * d0 -
          8 * a2 * a3 * b0 * c0 * c4 * d0 - 8 * a2 * a4 * b0 * c0 * c3 * d0 -
          8 * a3 * a4 * b0 * c0 * c2 * d0 + 4 * a3 * a5 * b0 * c0 * c1 * d0 -
          16 * a0 * a1 * b0 * c2 * c5 * d2 + 8 * a0 * a1 * b2 * c0 * c2 * d5 +
          8 * a0 * a1 * b2 * c0 * c5 * d2 + 8 * a0 * a1 * b2 * c2 * c5 * d0 -
          32 * a0 * a2 * b0 * c2 * c4 * d2 + 8 * a0 * a2 * b1 * c0 * c2 * d5 +
          8 * a0 * a2 * b1 * c0 * c5 * d2 + 8 * a0 * a2 * b1 * c2 * c5 * d0 +
          32 * a0 * a2 * b2 * c0 * c2 * d4 + 32 * a0 * a2 * b2 * c0 * c4 * d2 +
          32 * a0 * a2 * b2 * c2 * c4 * d0 - 32 * a0 * a2 * b4 * c0 * c2 * d2 -
          8 * a0 * a2 * b5 * c0 * c1 * d2 - 8 * a0 * a2 * b5 * c0 * c2 * d1 -
          8 * a0 * a2 * b5 * c1 * c2 * d0 - 32 * a0 * a4 * b2 * c0 * c2 * d2 +
          16 * a0 * a5 * b0 * c1 * c2 * d2 - 8 * a0 * a5 * b2 * c0 * c1 * d2 -
          8 * a0 * a5 * b2 * c0 * c2 * d1 - 8 * a0 * a5 * b2 * c1 * c2 * d0 +
          8 * a1 * a2 * b0 * c0 * c2 * d5 + 8 * a1 * a2 * b0 * c0 * c5 * d2 +
          8 * a1 * a2 * b0 * c2 * c5 * d0 - 64 * a1 * a2 * b1 * c1 * c2 * d3 -
          64 * a1 * a2 * b1 * c1 * c3 * d2 - 64 * a1 * a2 * b1 * c2 * c3 * d1 -
          16 * a1 * a2 * b2 * c0 * c5 * d0 + 64 * a1 * a2 * b2 * c1 * c3 * d1 +
          64 * a1 * a2 * b3 * c1 * c2 * d1 - 64 * a1 * a3 * b1 * c1 * c2 * d2 +
          64 * a1 * a3 * b2 * c1 * c2 * d1 + 64 * a2 * a3 * b1 * c1 * c2 * d1 -
          32 * a2 * a4 * b0 * c0 * c2 * d2 + 32 * a2 * a4 * b2 * c0 * c2 * d0 -
          8 * a2 * a5 * b0 * c0 * c1 * d2 - 8 * a2 * a5 * b0 * c0 * c2 * d1 -
          8 * a2 * a5 * b0 * c1 * c2 * d0 + 16 * a2 * a5 * b2 * c0 * c1 * d0 +
          4 * a0 * a1 * b0 * c0 * c5 * d5 + 16 * a0 * a1 * b1 * c1 * c3 * d5 +
          16 * a0 * a1 * b1 * c1 * c5 * d3 - 32 * a0 * a1 * b1 * c2 * c3 * d4 -
          32 * a0 * a1 * b1 * c2 * c4 * d3 - 32 * a0 * a1 * b1 * c3 * c4 * d2 +
          16 * a0 * a1 * b1 * c3 * c5 * d1 + 16 * a0 * a1 * b2 * c2 * c3 * d3 +
          16 * a0 * a1 * b3 * c1 * c2 * d4 + 16 * a0 * a1 * b3 * c1 * c4 * d2 -
          16 * a0 * a1 * b3 * c1 * c5 * d1 - 16 * a0 * a1 * b3 * c2 * c3 * d2 +
          16 * a0 * a1 * b3 * c2 * c4 * d1 + 16 * a0 * a1 * b4 * c1 * c2 * d3 +
          16 * a0 * a1 * b4 * c1 * c3 * d2 + 16 * a0 * a1 * b4 * c2 * c3 * d1 -
          4 * a0 * a1 * b5 * c0 * c5 * d0 - 16 * a0 * a1 * b5 * c1 * c3 * d1 -
          8 * a0 * a2 * b0 * c0 * c4 * d5 - 8 * a0 * a2 * b0 * c0 * c5 * d4 -
          8 * a0 * a2 * b0 * c4 * c5 * d0 + 16 * a0 * a2 * b1 * c2 * c3 * d3 -
          16 * a0 * a2 * b2 * c1 * c3 * d3 + 8 * a0 * a2 * b4 * c0 * c5 * d0 +
          8 * a0 * a2 * b5 * c0 * c4 * d0 + 16 * a0 * a3 * b1 * c1 * c2 * d4 +
          16 * a0 * a3 * b1 * c1 * c4 * d2 - 16 * a0 * a3 * b1 * c1 * c5 * d1 -
          16 * a0 * a3 * b1 * c2 * c3 * d2 + 16 * a0 * a3 * b1 * c2 * c4 * d1 +
          16 * a0 * a3 * b3 * c1 * c2 * d2 - 32 * a0 * a3 * b4 * c1 * c2 * d1 -
          8 * a0 * a4 * b0 * c0 * c2 * d5 - 8 * a0 * a4 * b0 * c0 * c5 * d2 -
          8 * a0 * a4 * b0 * c2 * c5 * d0 + 16 * a0 * a4 * b1 * c1 * c2 * d3 +
          16 * a0 * a4 * b1 * c1 * c3 * d2 + 16 * a0 * a4 * b1 * c2 * c3 * d1 +
          8 * a0 * a4 * b2 * c0 * c5 * d0 - 32 * a0 * a4 * b3 * c1 * c2 * d1 +
          8 * a0 * a4 * b5 * c0 * c2 * d0 + 4 * a0 * a5 * b0 * c0 * c1 * d5 -
          8 * a0 * a5 * b0 * c0 * c2 * d4 - 8 * a0 * a5 * b0 * c0 * c4 * d2 +
          4 * a0 * a5 * b0 * c0 * c5 * d1 + 4 * a0 * a5 * b0 * c1 * c5 * d0 -
          8 * a0 * a5 * b0 * c2 * c4 * d0 - 4 * a0 * a5 * b1 * c0 * c5 * d0 -
          16 * a0 * a5 * b1 * c1 * c3 * d1 + 8 * a0 * a5 * b2 * c0 * c4 * d0 +
          8 * a0 * a5 * b4 * c0 * c2 * d0 - 4 * a0 * a5 * b5 * c0 * c1 * d0 +
          16 * a1 * a2 * b0 * c2 * c3 * d3 + 32 * a1 * a2 * b1 * c0 * c3 * d4 +
          32 * a1 * a2 * b1 * c0 * c4 * d3 + 32 * a1 * a2 * b1 * c3 * c4 * d0 -
          16 * a1 * a2 * b2 * c0 * c3 * d3 - 16 * a1 * a2 * b3 * c0 * c1 * d4 -
          16 * a1 * a2 * b3 * c0 * c4 * d1 - 16 * a1 * a2 * b3 * c1 * c4 * d0 -
          16 * a1 * a2 * b4 * c0 * c1 * d3 - 16 * a1 * a2 * b4 * c0 * c3 * d1 -
          16 * a1 * a2 * b4 * c1 * c3 * d0 + 16 * a1 * a3 * b0 * c1 * c2 * d4 +
          16 * a1 * a3 * b0 * c1 * c4 * d2 - 16 * a1 * a3 * b0 * c1 * c5 * d1 -
          16 * a1 * a3 * b0 * c2 * c3 * d2 + 16 * a1 * a3 * b0 * c2 * c4 * d1 +
          16 * a1 * a3 * b1 * c0 * c1 * d5 + 16 * a1 * a3 * b1 * c0 * c5 * d1 +
          16 * a1 * a3 * b1 * c1 * c5 * d0 - 16 * a1 * a3 * b2 * c0 * c1 * d4 -
          16 * a1 * a3 * b2 * c0 * c4 * d1 - 16 * a1 * a3 * b2 * c1 * c4 * d0 +
          16 * a1 * a3 * b3 * c0 * c2 * d2 - 16 * a1 * a3 * b5 * c0 * c1 * d1 +
          16 * a1 * a4 * b0 * c1 * c2 * d3 + 16 * a1 * a4 * b0 * c1 * c3 * d2 +
          16 * a1 * a4 * b0 * c2 * c3 * d1 - 16 * a1 * a4 * b2 * c0 * c1 * d3 -
          16 * a1 * a4 * b2 * c0 * c3 * d1 - 16 * a1 * a4 * b2 * c1 * c3 * d0 -
          4 * a1 * a5 * b0 * c0 * c5 * d0 - 16 * a1 * a5 * b0 * c1 * c3 * d1 +
          16 * a1 * a5 * b1 * c0 * c1 * d3 + 16 * a1 * a5 * b1 * c0 * c3 * d1 +
          16 * a1 * a5 * b1 * c1 * c3 * d0 - 16 * a1 * a5 * b3 * c0 * c1 * d1 -
          16 * a2 * a3 * b1 * c0 * c1 * d4 - 16 * a2 * a3 * b1 * c0 * c4 * d1 -
          16 * a2 * a3 * b1 * c1 * c4 * d0 + 16 * a2 * a3 * b2 * c0 * c1 * d3 +
          16 * a2 * a3 * b2 * c0 * c3 * d1 + 16 * a2 * a3 * b2 * c1 * c3 * d0 -
          16 * a2 * a3 * b3 * c0 * c1 * d2 - 16 * a2 * a3 * b3 * c0 * c2 * d1 -
          16 * a2 * a3 * b3 * c1 * c2 * d0 + 32 * a2 * a3 * b4 * c0 * c1 * d1 +
          8 * a2 * a4 * b0 * c0 * c5 * d0 - 16 * a2 * a4 * b1 * c0 * c1 * d3 -
          16 * a2 * a4 * b1 * c0 * c3 * d1 - 16 * a2 * a4 * b1 * c1 * c3 * d0 +
          32 * a2 * a4 * b3 * c0 * c1 * d1 + 8 * a2 * a5 * b0 * c0 * c4 * d0 -
          32 * a3 * a4 * b0 * c1 * c2 * d1 + 32 * a3 * a4 * b2 * c0 * c1 * d1 -
          16 * a3 * a5 * b1 * c0 * c1 * d1 + 8 * a4 * a5 * b0 * c0 * c2 * d0 +
          16 * a0 * a1 * b0 * c3 * c4 * d4 - 4 * a0 * a1 * b0 * c3 * c5 * d3 +
          4 * a0 * a1 * b3 * c0 * c3 * d5 + 4 * a0 * a1 * b3 * c0 * c5 * d3 +
          4 * a0 * a1 * b3 * c3 * c5 * d0 - 8 * a0 * a1 * b4 * c0 * c3 * d4 -
          8 * a0 * a1 * b4 * c0 * c4 * d3 - 8 * a0 * a1 * b4 * c3 * c4 * d0 -
          4 * a0 * a1 * b5 * c0 * c3 * d3 - 8 * a0 * a2 * b0 * c3 * c4 * d3 +
          8 * a0 * a2 * b4 * c0 * c3 * d3 - 4 * a0 * a3 * b0 * c1 * c3 * d5 -
          16 * a0 * a3 * b0 * c1 * c4 * d4 - 4 * a0 * a3 * b0 * c1 * c5 * d3 +
          8 * a0 * a3 * b0 * c2 * c3 * d4 + 8 * a0 * a3 * b0 * c2 * c4 * d3 +
          8 * a0 * a3 * b0 * c3 * c4 * d2 - 4 * a0 * a3 * b0 * c3 * c5 * d1 +
          4 * a0 * a3 * b1 * c0 * c3 * d5 + 4 * a0 * a3 * b1 * c0 * c5 * d3 +
          4 * a0 * a3 * b1 * c3 * c5 * d0 + 4 * a0 * a3 * b3 * c0 * c1 * d5 -
          8 * a0 * a3 * b3 * c0 * c2 * d4 - 8 * a0 * a3 * b3 * c0 * c4 * d2 +
          4 * a0 * a3 * b3 * c0 * c5 * d1 + 4 * a0 * a3 * b3 * c1 * c5 * d0 -
          8 * a0 * a3 * b3 * c2 * c4 * d0 + 8 * a0 * a3 * b4 * c0 * c1 * d4 +
          8 * a0 * a3 * b4 * c0 * c4 * d1 + 8 * a0 * a3 * b4 * c1 * c4 * d0 -
          4 * a0 * a3 * b5 * c0 * c1 * d3 - 4 * a0 * a3 * b5 * c0 * c3 * d1 -
          4 * a0 * a3 * b5 * c1 * c3 * d0 - 8 * a0 * a4 * b0 * c2 * c3 * d3 -
          8 * a0 * a4 * b1 * c0 * c3 * d4 - 8 * a0 * a4 * b1 * c0 * c4 * d3 -
          8 * a0 * a4 * b1 * c3 * c4 * d0 + 8 * a0 * a4 * b2 * c0 * c3 * d3 +
          8 * a0 * a4 * b3 * c0 * c1 * d4 + 8 * a0 * a4 * b3 * c0 * c4 * d1 +
          8 * a0 * a4 * b3 * c1 * c4 * d0 + 12 * a0 * a5 * b0 * c1 * c3 * d3 -
          4 * a0 * a5 * b1 * c0 * c3 * d3 - 4 * a0 * a5 * b3 * c0 * c1 * d3 -
          4 * a0 * a5 * b3 * c0 * c3 * d1 - 4 * a0 * a5 * b3 * c1 * c3 * d0 +
          64 * a1 * a2 * b1 * c1 * c2 * d5 + 64 * a1 * a2 * b1 * c1 * c5 * d2 -
          128 * a1 * a2 * b1 * c2 * c4 * d2 + 64 * a1 * a2 * b1 * c2 * c5 * d1 +
          128 * a1 * a2 * b2 * c1 * c2 * d4 +
          128 * a1 * a2 * b2 * c1 * c4 * d2 - 64 * a1 * a2 * b2 * c1 * c5 * d1 +
          128 * a1 * a2 * b2 * c2 * c4 * d1 -
          128 * a1 * a2 * b4 * c1 * c2 * d2 - 64 * a1 * a2 * b5 * c1 * c2 * d1 +
          4 * a1 * a3 * b0 * c0 * c3 * d5 + 4 * a1 * a3 * b0 * c0 * c5 * d3 +
          4 * a1 * a3 * b0 * c3 * c5 * d0 - 12 * a1 * a3 * b3 * c0 * c5 * d0 +
          4 * a1 * a3 * b5 * c0 * c3 * d0 - 8 * a1 * a4 * b0 * c0 * c3 * d4 -
          8 * a1 * a4 * b0 * c0 * c4 * d3 - 8 * a1 * a4 * b0 * c3 * c4 * d0 -
          128 * a1 * a4 * b2 * c1 * c2 * d2 + 16 * a1 * a4 * b4 * c0 * c3 * d0 -
          4 * a1 * a5 * b0 * c0 * c3 * d3 + 64 * a1 * a5 * b1 * c1 * c2 * d2 -
          64 * a1 * a5 * b2 * c1 * c2 * d1 + 4 * a1 * a5 * b3 * c0 * c3 * d0 +
          8 * a2 * a3 * b3 * c0 * c4 * d0 - 8 * a2 * a3 * b4 * c0 * c3 * d0 +
          8 * a2 * a4 * b0 * c0 * c3 * d3 - 128 * a2 * a4 * b1 * c1 * c2 * d2 +
          128 * a2 * a4 * b2 * c1 * c2 * d1 - 8 * a2 * a4 * b3 * c0 * c3 * d0 -
          64 * a2 * a5 * b1 * c1 * c2 * d1 + 8 * a3 * a4 * b0 * c0 * c1 * d4 +
          8 * a3 * a4 * b0 * c0 * c4 * d1 + 8 * a3 * a4 * b0 * c1 * c4 * d0 -
          8 * a3 * a4 * b2 * c0 * c3 * d0 + 8 * a3 * a4 * b3 * c0 * c2 * d0 -
          16 * a3 * a4 * b4 * c0 * c1 * d0 - 4 * a3 * a5 * b0 * c0 * c1 * d3 -
          4 * a3 * a5 * b0 * c0 * c3 * d1 - 4 * a3 * a5 * b0 * c1 * c3 * d0 +
          4 * a3 * a5 * b1 * c0 * c3 * d0 + 4 * a3 * a5 * b3 * c0 * c1 * d0 -
          32 * a0 * a1 * b1 * c1 * c5 * d5 + 32 * a0 * a1 * b1 * c2 * c4 * d5 +
          32 * a0 * a1 * b1 * c2 * c5 * d4 + 32 * a0 * a1 * b1 * c4 * c5 * d2 -
          16 * a0 * a1 * b2 * c1 * c4 * d5 - 16 * a0 * a1 * b2 * c1 * c5 * d4 -
          8 * a0 * a1 * b2 * c2 * c3 * d5 - 8 * a0 * a1 * b2 * c2 * c5 * d3 -
          8 * a0 * a1 * b2 * c3 * c5 * d2 - 16 * a0 * a1 * b2 * c4 * c5 * d1 +
          16 * a0 * a1 * b3 * c2 * c5 * d2 - 16 * a0 * a1 * b5 * c1 * c2 * d4 -
          16 * a0 * a1 * b5 * c1 * c4 * d2 + 32 * a0 * a1 * b5 * c1 * c5 * d1 -
          16 * a0 * a1 * b5 * c2 * c4 * d1 - 16 * a0 * a2 * b1 * c1 * c4 * d5 -
          16 * a0 * a2 * b1 * c1 * c5 * d4 - 8 * a0 * a2 * b1 * c2 * c3 * d5 -
          8 * a0 * a2 * b1 * c2 * c5 * d3 - 8 * a0 * a2 * b1 * c3 * c5 * d2 -
          16 * a0 * a2 * b1 * c4 * c5 * d1 + 16 * a0 * a2 * b2 * c1 * c3 * d5 +
          64 * a0 * a2 * b2 * c1 * c4 * d4 + 16 * a0 * a2 * b2 * c1 * c5 * d3 -
          32 * a0 * a2 * b2 * c2 * c3 * d4 - 32 * a0 * a2 * b2 * c2 * c4 * d3 -
          32 * a0 * a2 * b2 * c3 * c4 * d2 + 16 * a0 * a2 * b2 * c3 * c5 * d1 -
          8 * a0 * a2 * b3 * c1 * c2 * d5 - 8 * a0 * a2 * b3 * c1 * c5 * d2 +
          32 * a0 * a2 * b3 * c2 * c4 * d2 - 8 * a0 * a2 * b3 * c2 * c5 * d1 -
          32 * a0 * a2 * b4 * c1 * c2 * d4 - 32 * a0 * a2 * b4 * c1 * c4 * d2 +
          32 * a0 * a2 * b4 * c2 * c3 * d2 - 32 * a0 * a2 * b4 * c2 * c4 * d1 +
          32 * a0 * a2 * b5 * c1 * c4 * d1 + 16 * a0 * a3 * b1 * c2 * c5 * d2 -
          8 * a0 * a3 * b2 * c1 * c2 * d5 - 8 * a0 * a3 * b2 * c1 * c5 * d2 +
          32 * a0 * a3 * b2 * c2 * c4 * d2 - 8 * a0 * a3 * b2 * c2 * c5 * d1 -
          32 * a0 * a4 * b2 * c1 * c2 * d4 - 32 * a0 * a4 * b2 * c1 * c4 * d2 +
          32 * a0 * a4 * b2 * c2 * c3 * d2 - 32 * a0 * a4 * b2 * c2 * c4 * d1 +
          64 * a0 * a4 * b4 * c1 * c2 * d2 - 16 * a0 * a5 * b1 * c1 * c2 * d4 -
          16 * a0 * a5 * b1 * c1 * c4 * d2 + 32 * a0 * a5 * b1 * c1 * c5 * d1 -
          16 * a0 * a5 * b1 * c2 * c4 * d1 + 32 * a0 * a5 * b2 * c1 * c4 * d1 -
          16 * a1 * a2 * b0 * c1 * c4 * d5 - 16 * a1 * a2 * b0 * c1 * c5 * d4 -
          8 * a1 * a2 * b0 * c2 * c3 * d5 - 8 * a1 * a2 * b0 * c2 * c5 * d3 -
          8 * a1 * a2 * b0 * c3 * c5 * d2 - 16 * a1 * a2 * b0 * c4 * c5 * d1 -
          64 * a1 * a2 * b2 * c0 * c4 * d4 + 16 * a1 * a2 * b4 * c0 * c1 * d5 +
          32 * a1 * a2 * b4 * c0 * c2 * d4 + 32 * a1 * a2 * b4 * c0 * c4 * d2 +
          16 * a1 * a2 * b4 * c0 * c5 * d1 + 16 * a1 * a2 * b4 * c1 * c5 * d0 +
          32 * a1 * a2 * b4 * c2 * c4 * d0 + 8 * a1 * a2 * b5 * c0 * c2 * d3 +
          8 * a1 * a2 * b5 * c0 * c3 * d2 + 8 * a1 * a2 * b5 * c2 * c3 * d0 +
          16 * a1 * a3 * b0 * c2 * c5 * d2 - 16 * a1 * a3 * b5 * c0 * c2 * d2 -
          32 * a1 * a4 * b1 * c0 * c2 * d5 - 32 * a1 * a4 * b1 * c0 * c5 * d2 -
          32 * a1 * a4 * b1 * c2 * c5 * d0 + 16 * a1 * a4 * b2 * c0 * c1 * d5 +
          32 * a1 * a4 * b2 * c0 * c2 * d4 + 32 * a1 * a4 * b2 * c0 * c4 * d2 +
          16 * a1 * a4 * b2 * c0 * c5 * d1 + 16 * a1 * a4 * b2 * c1 * c5 * d0 +
          32 * a1 * a4 * b2 * c2 * c4 * d0 - 64 * a1 * a4 * b4 * c0 * c2 * d2 +
          16 * a1 * a4 * b5 * c0 * c1 * d2 + 16 * a1 * a4 * b5 * c0 * c2 * d1 +
          16 * a1 * a4 * b5 * c1 * c2 * d0 - 16 * a1 * a5 * b0 * c1 * c2 * d4 -
          16 * a1 * a5 * b0 * c1 * c4 * d2 + 32 * a1 * a5 * b0 * c1 * c5 * d1 -
          16 * a1 * a5 * b0 * c2 * c4 * d1 - 32 * a1 * a5 * b1 * c0 * c1 * d5 -
          32 * a1 * a5 * b1 * c0 * c5 * d1 - 32 * a1 * a5 * b1 * c1 * c5 * d0 +
          8 * a1 * a5 * b2 * c0 * c2 * d3 + 8 * a1 * a5 * b2 * c0 * c3 * d2 +
          8 * a1 * a5 * b2 * c2 * c3 * d0 - 16 * a1 * a5 * b3 * c0 * c2 * d2 +
          16 * a1 * a5 * b4 * c0 * c1 * d2 + 16 * a1 * a5 * b4 * c0 * c2 * d1 +
          16 * a1 * a5 * b4 * c1 * c2 * d0 + 32 * a1 * a5 * b5 * c0 * c1 * d1 -
          8 * a2 * a3 * b0 * c1 * c2 * d5 - 8 * a2 * a3 * b0 * c1 * c5 * d2 +
          32 * a2 * a3 * b0 * c2 * c4 * d2 - 8 * a2 * a3 * b0 * c2 * c5 * d1 -
          32 * a2 * a3 * b2 * c0 * c2 * d4 - 32 * a2 * a3 * b2 * c0 * c4 * d2 -
          32 * a2 * a3 * b2 * c2 * c4 * d0 + 32 * a2 * a3 * b4 * c0 * c2 * d2 +
          8 * a2 * a3 * b5 * c0 * c1 * d2 + 8 * a2 * a3 * b5 * c0 * c2 * d1 +
          8 * a2 * a3 * b5 * c1 * c2 * d0 - 32 * a2 * a4 * b0 * c1 * c2 * d4 -
          32 * a2 * a4 * b0 * c1 * c4 * d2 + 32 * a2 * a4 * b0 * c2 * c3 * d2 -
          32 * a2 * a4 * b0 * c2 * c4 * d1 + 16 * a2 * a4 * b1 * c0 * c1 * d5 +
          32 * a2 * a4 * b1 * c0 * c2 * d4 + 32 * a2 * a4 * b1 * c0 * c4 * d2 +
          16 * a2 * a4 * b1 * c0 * c5 * d1 + 16 * a2 * a4 * b1 * c1 * c5 * d0 +
          32 * a2 * a4 * b1 * c2 * c4 * d0 - 32 * a2 * a4 * b2 * c0 * c2 * d3 -
          32 * a2 * a4 * b2 * c0 * c3 * d2 - 32 * a2 * a4 * b2 * c2 * c3 * d0 +
          32 * a2 * a4 * b3 * c0 * c2 * d2 - 32 * a2 * a4 * b5 * c0 * c1 * d1 +
          32 * a2 * a5 * b0 * c1 * c4 * d1 + 8 * a2 * a5 * b1 * c0 * c2 * d3 +
          8 * a2 * a5 * b1 * c0 * c3 * d2 + 8 * a2 * a5 * b1 * c2 * c3 * d0 -
          16 * a2 * a5 * b2 * c0 * c1 * d3 - 16 * a2 * a5 * b2 * c0 * c3 * d1 -
          16 * a2 * a5 * b2 * c1 * c3 * d0 + 8 * a2 * a5 * b3 * c0 * c1 * d2 +
          8 * a2 * a5 * b3 * c0 * c2 * d1 + 8 * a2 * a5 * b3 * c1 * c2 * d0 -
          32 * a2 * a5 * b4 * c0 * c1 * d1 + 32 * a3 * a4 * b2 * c0 * c2 * d2 -
          16 * a3 * a5 * b1 * c0 * c2 * d2 + 8 * a3 * a5 * b2 * c0 * c1 * d2 +
          8 * a3 * a5 * b2 * c0 * c2 * d1 + 8 * a3 * a5 * b2 * c1 * c2 * d0 +
          16 * a4 * a5 * b1 * c0 * c1 * d2 + 16 * a4 * a5 * b1 * c0 * c2 * d1 +
          16 * a4 * a5 * b1 * c1 * c2 * d0 - 32 * a4 * a5 * b2 * c0 * c1 * d1 +
          8 * a0 * a1 * b0 * c3 * c5 * d5 - 8 * a0 * a1 * b3 * c0 * c5 * d5 -
          8 * a0 * a1 * b4 * c0 * c4 * d5 - 8 * a0 * a1 * b4 * c0 * c5 * d4 -
          8 * a0 * a1 * b4 * c4 * c5 * d0 + 16 * a0 * a1 * b5 * c0 * c4 * d4 +
          8 * a0 * a2 * b3 * c0 * c4 * d5 + 8 * a0 * a2 * b3 * c0 * c5 * d4 +
          8 * a0 * a2 * b3 * c4 * c5 * d0 - 8 * a0 * a2 * b5 * c0 * c3 * d4 -
          8 * a0 * a2 * b5 * c0 * c4 * d3 - 8 * a0 * a2 * b5 * c3 * c4 * d0 +
          8 * a0 * a3 * b0 * c1 * c5 * d5 - 16 * a0 * a3 * b0 * c2 * c4 * d5 -
          16 * a0 * a3 * b0 * c2 * c5 * d4 - 16 * a0 * a3 * b0 * c4 * c5 * d2 -
          8 * a0 * a3 * b1 * c0 * c5 * d5 + 8 * a0 * a3 * b2 * c0 * c4 * d5 +
          8 * a0 * a3 * b2 * c0 * c5 * d4 + 8 * a0 * a3 * b2 * c4 * c5 * d0 +
          8 * a0 * a3 * b4 * c0 * c2 * d5 + 8 * a0 * a3 * b4 * c0 * c5 * d2 +
          8 * a0 * a3 * b4 * c2 * c5 * d0 + 16 * a0 * a4 * b0 * c1 * c4 * d5 +
          16 * a0 * a4 * b0 * c1 * c5 * d4 + 16 * a0 * a4 * b0 * c4 * c5 * d1 -
          8 * a0 * a4 * b1 * c0 * c4 * d5 - 8 * a0 * a4 * b1 * c0 * c5 * d4 -
          8 * a0 * a4 * b1 * c4 * c5 * d0 + 8 * a0 * a4 * b3 * c0 * c2 * d5 +
          8 * a0 * a4 * b3 * c0 * c5 * d2 + 8 * a0 * a4 * b3 * c2 * c5 * d0 -
          16 * a0 * a4 * b4 * c0 * c1 * d5 - 16 * a0 * a4 * b4 * c0 * c5 * d1 -
          16 * a0 * a4 * b4 * c1 * c5 * d0 + 8 * a0 * a4 * b5 * c0 * c1 * d4 -
          8 * a0 * a4 * b5 * c0 * c2 * d3 - 8 * a0 * a4 * b5 * c0 * c3 * d2 +
          8 * a0 * a4 * b5 * c0 * c4 * d1 + 8 * a0 * a4 * b5 * c1 * c4 * d0 -
          8 * a0 * a4 * b5 * c2 * c3 * d0 - 8 * a0 * a5 * b0 * c1 * c3 * d5 -
          32 * a0 * a5 * b0 * c1 * c4 * d4 - 8 * a0 * a5 * b0 * c1 * c5 * d3 +
          16 * a0 * a5 * b0 * c2 * c3 * d4 + 16 * a0 * a5 * b0 * c2 * c4 * d3 +
          16 * a0 * a5 * b0 * c3 * c4 * d2 - 8 * a0 * a5 * b0 * c3 * c5 * d1 +
          16 * a0 * a5 * b1 * c0 * c4 * d4 - 8 * a0 * a5 * b2 * c0 * c3 * d4 -
          8 * a0 * a5 * b2 * c0 * c4 * d3 - 8 * a0 * a5 * b2 * c3 * c4 * d0 +
          8 * a0 * a5 * b4 * c0 * c1 * d4 - 8 * a0 * a5 * b4 * c0 * c2 * d3 -
          8 * a0 * a5 * b4 * c0 * c3 * d2 + 8 * a0 * a5 * b4 * c0 * c4 * d1 +
          8 * a0 * a5 * b4 * c1 * c4 * d0 - 8 * a0 * a5 * b4 * c2 * c3 * d0 +
          8 * a0 * a5 * b5 * c0 * c1 * d3 + 8 * a0 * a5 * b5 * c0 * c3 * d1 +
          8 * a0 * a5 * b5 * c1 * c3 * d0 - 8 * a1 * a3 * b0 * c0 * c5 * d5 +
          8 * a1 * a3 * b5 * c0 * c5 * d0 - 8 * a1 * a4 * b0 * c0 * c4 * d5 -
          8 * a1 * a4 * b0 * c0 * c5 * d4 - 8 * a1 * a4 * b0 * c4 * c5 * d0 +
          32 * a1 * a4 * b4 * c0 * c5 * d0 - 16 * a1 * a4 * b5 * c0 * c4 * d0 +
          16 * a1 * a5 * b0 * c0 * c4 * d4 + 8 * a1 * a5 * b3 * c0 * c5 * d0 -
          16 * a1 * a5 * b4 * c0 * c4 * d0 - 8 * a1 * a5 * b5 * c0 * c3 * d0 +
          8 * a2 * a3 * b0 * c0 * c4 * d5 + 8 * a2 * a3 * b0 * c0 * c5 * d4 +
          8 * a2 * a3 * b0 * c4 * c5 * d0 - 16 * a2 * a3 * b4 * c0 * c5 * d0 -
          16 * a2 * a4 * b3 * c0 * c5 * d0 + 16 * a2 * a4 * b5 * c0 * c3 * d0 -
          8 * a2 * a5 * b0 * c0 * c3 * d4 - 8 * a2 * a5 * b0 * c0 * c4 * d3 -
          8 * a2 * a5 * b0 * c3 * c4 * d0 + 16 * a2 * a5 * b4 * c0 * c3 * d0 +
          8 * a3 * a4 * b0 * c0 * c2 * d5 + 8 * a3 * a4 * b0 * c0 * c5 * d2 +
          8 * a3 * a4 * b0 * c2 * c5 * d0 - 16 * a3 * a4 * b2 * c0 * c5 * d0 +
          8 * a3 * a5 * b1 * c0 * c5 * d0 - 8 * a3 * a5 * b5 * c0 * c1 * d0 +
          8 * a4 * a5 * b0 * c0 * c1 * d4 - 8 * a4 * a5 * b0 * c0 * c2 * d3 -
          8 * a4 * a5 * b0 * c0 * c3 * d2 + 8 * a4 * a5 * b0 * c0 * c4 * d1 +
          8 * a4 * a5 * b0 * c1 * c4 * d0 - 8 * a4 * a5 * b0 * c2 * c3 * d0 -
          16 * a4 * a5 * b1 * c0 * c4 * d0 + 16 * a4 * a5 * b2 * c0 * c3 * d0 -
          16 * a0 * a2 * b2 * c1 * c5 * d5 + 8 * a0 * a2 * b5 * c1 * c2 * d5 +
          8 * a0 * a2 * b5 * c1 * c5 * d2 + 8 * a0 * a2 * b5 * c2 * c5 * d1 +
          8 * a0 * a5 * b2 * c1 * c2 * d5 + 8 * a0 * a5 * b2 * c1 * c5 * d2 +
          8 * a0 * a5 * b2 * c2 * c5 * d1 - 16 * a0 * a5 * b5 * c1 * c2 * d2 +
          16 * a1 * a2 * b2 * c0 * c5 * d5 - 8 * a1 * a2 * b5 * c0 * c2 * d5 -
          8 * a1 * a2 * b5 * c0 * c5 * d2 - 8 * a1 * a2 * b5 * c2 * c5 * d0 -
          8 * a1 * a5 * b2 * c0 * c2 * d5 - 8 * a1 * a5 * b2 * c0 * c5 * d2 -
          8 * a1 * a5 * b2 * c2 * c5 * d0 + 16 * a1 * a5 * b5 * c0 * c2 * d2 +
          8 * a2 * a5 * b0 * c1 * c2 * d5 + 8 * a2 * a5 * b0 * c1 * c5 * d2 +
          8 * a2 * a5 * b0 * c2 * c5 * d1 - 8 * a2 * a5 * b1 * c0 * c2 * d5 -
          8 * a2 * a5 * b1 * c0 * c5 * d2 - 8 * a2 * a5 * b1 * c2 * c5 * d0 +
          4 * a0 * a1 * b5 * c0 * c5 * d5 + 8 * a0 * a2 * b0 * c4 * c5 * d5 -
          8 * a0 * a2 * b4 * c0 * c5 * d5 + 8 * a0 * a4 * b0 * c2 * c5 * d5 -
          8 * a0 * a4 * b2 * c0 * c5 * d5 + 4 * a0 * a5 * b0 * c1 * c5 * d5 -
          8 * a0 * a5 * b0 * c2 * c4 * d5 - 8 * a0 * a5 * b0 * c2 * c5 * d4 -
          8 * a0 * a5 * b0 * c4 * c5 * d2 + 4 * a0 * a5 * b1 * c0 * c5 * d5 -
          4 * a0 * a5 * b5 * c0 * c1 * d5 + 8 * a0 * a5 * b5 * c0 * c2 * d4 +
          8 * a0 * a5 * b5 * c0 * c4 * d2 - 4 * a0 * a5 * b5 * c0 * c5 * d1 -
          4 * a0 * a5 * b5 * c1 * c5 * d0 + 8 * a0 * a5 * b5 * c2 * c4 * d0 -
          32 * a1 * a2 * b1 * c3 * c4 * d5 - 32 * a1 * a2 * b1 * c3 * c5 * d4 -
          32 * a1 * a2 * b1 * c4 * c5 * d3 + 64 * a1 * a2 * b2 * c3 * c4 * d4 +
          16 * a1 * a2 * b2 * c3 * c5 * d3 + 16 * a1 * a2 * b3 * c1 * c4 * d5 +
          16 * a1 * a2 * b3 * c1 * c5 * d4 + 16 * a1 * a2 * b3 * c4 * c5 * d1 +
          16 * a1 * a2 * b4 * c1 * c3 * d5 + 16 * a1 * a2 * b4 * c1 * c5 * d3 -
          32 * a1 * a2 * b4 * c2 * c3 * d4 - 32 * a1 * a2 * b4 * c2 * c4 * d3 -
          32 * a1 * a2 * b4 * c3 * c4 * d2 + 16 * a1 * a2 * b4 * c3 * c5 * d1 -
          16 * a1 * a2 * b5 * c2 * c3 * d3 - 16 * a1 * a3 * b1 * c1 * c5 * d5 +
          16 * a1 * a3 * b2 * c1 * c4 * d5 + 16 * a1 * a3 * b2 * c1 * c5 * d4 +
          16 * a1 * a3 * b2 * c4 * c5 * d1 - 16 * a1 * a3 * b3 * c2 * c5 * d2 -
          16 * a1 * a3 * b5 * c1 * c2 * d4 - 16 * a1 * a3 * b5 * c1 * c4 * d2 +
          16 * a1 * a3 * b5 * c1 * c5 * d1 + 16 * a1 * a3 * b5 * c2 * c3 * d2 -
          16 * a1 * a3 * b5 * c2 * c4 * d1 + 16 * a1 * a4 * b2 * c1 * c3 * d5 +
          16 * a1 * a4 * b2 * c1 * c5 * d3 - 32 * a1 * a4 * b2 * c2 * c3 * d4 -
          32 * a1 * a4 * b2 * c2 * c4 * d3 - 32 * a1 * a4 * b2 * c3 * c4 * d2 +
          16 * a1 * a4 * b2 * c3 * c5 * d1 + 64 * a1 * a4 * b4 * c2 * c3 * d2 -
          16 * a1 * a4 * b5 * c1 * c2 * d3 - 16 * a1 * a4 * b5 * c1 * c3 * d2 -
          16 * a1 * a4 * b5 * c2 * c3 * d1 + 4 * a1 * a5 * b0 * c0 * c5 * d5 -
          16 * a1 * a5 * b1 * c1 * c3 * d5 - 16 * a1 * a5 * b1 * c1 * c5 * d3 +
          32 * a1 * a5 * b1 * c2 * c3 * d4 + 32 * a1 * a5 * b1 * c2 * c4 * d3 +
          32 * a1 * a5 * b1 * c3 * c4 * d2 - 16 * a1 * a5 * b1 * c3 * c5 * d1 -
          16 * a1 * a5 * b2 * c2 * c3 * d3 - 16 * a1 * a5 * b3 * c1 * c2 * d4 -
          16 * a1 * a5 * b3 * c1 * c4 * d2 + 16 * a1 * a5 * b3 * c1 * c5 * d1 +
          16 * a1 * a5 * b3 * c2 * c3 * d2 - 16 * a1 * a5 * b3 * c2 * c4 * d1 -
          16 * a1 * a5 * b4 * c1 * c2 * d3 - 16 * a1 * a5 * b4 * c1 * c3 * d2 -
          16 * a1 * a5 * b4 * c2 * c3 * d1 - 4 * a1 * a5 * b5 * c0 * c5 * d0 +
          16 * a1 * a5 * b5 * c1 * c3 * d1 + 16 * a2 * a3 * b1 * c1 * c4 * d5 +
          16 * a2 * a3 * b1 * c1 * c5 * d4 + 16 * a2 * a3 * b1 * c4 * c5 * d1 -
          16 * a2 * a3 * b2 * c1 * c3 * d5 - 64 * a2 * a3 * b2 * c1 * c4 * d4 -
          16 * a2 * a3 * b2 * c1 * c5 * d3 + 32 * a2 * a3 * b2 * c2 * c3 * d4 +
          32 * a2 * a3 * b2 * c2 * c4 * d3 + 32 * a2 * a3 * b2 * c3 * c4 * d2 -
          16 * a2 * a3 * b2 * c3 * c5 * d1 + 16 * a2 * a3 * b3 * c1 * c2 * d5 +
          16 * a2 * a3 * b3 * c1 * c5 * d2 - 32 * a2 * a3 * b3 * c2 * c4 * d2 +
          16 * a2 * a3 * b3 * c2 * c5 * d1 + 32 * a2 * a3 * b4 * c1 * c2 * d4 +
          32 * a2 * a3 * b4 * c1 * c4 * d2 - 32 * a2 * a3 * b4 * c1 * c5 * d1 -
          32 * a2 * a3 * b4 * c2 * c3 * d2 + 32 * a2 * a3 * b4 * c2 * c4 * d1 -
          8 * a2 * a4 * b0 * c0 * c5 * d5 + 16 * a2 * a4 * b1 * c1 * c3 * d5 +
          16 * a2 * a4 * b1 * c1 * c5 * d3 - 32 * a2 * a4 * b1 * c2 * c3 * d4 -
          32 * a2 * a4 * b1 * c2 * c4 * d3 - 32 * a2 * a4 * b1 * c3 * c4 * d2 +
          16 * a2 * a4 * b1 * c3 * c5 * d1 + 32 * a2 * a4 * b2 * c2 * c3 * d3 +
          32 * a2 * a4 * b3 * c1 * c2 * d4 + 32 * a2 * a4 * b3 * c1 * c4 * d2 -
          32 * a2 * a4 * b3 * c1 * c5 * d1 - 32 * a2 * a4 * b3 * c2 * c3 * d2 +
          32 * a2 * a4 * b3 * c2 * c4 * d1 + 8 * a2 * a4 * b5 * c0 * c5 * d0 -
          16 * a2 * a5 * b1 * c2 * c3 * d3 + 16 * a2 * a5 * b2 * c1 * c3 * d3 +
          8 * a2 * a5 * b4 * c0 * c5 * d0 - 8 * a2 * a5 * b5 * c0 * c4 * d0 +
          32 * a3 * a4 * b2 * c1 * c2 * d4 + 32 * a3 * a4 * b2 * c1 * c4 * d2 -
          32 * a3 * a4 * b2 * c1 * c5 * d1 - 32 * a3 * a4 * b2 * c2 * c3 * d2 +
          32 * a3 * a4 * b2 * c2 * c4 * d1 - 64 * a3 * a4 * b4 * c1 * c2 * d2 +
          32 * a3 * a4 * b5 * c1 * c2 * d1 - 16 * a3 * a5 * b1 * c1 * c2 * d4 -
          16 * a3 * a5 * b1 * c1 * c4 * d2 + 16 * a3 * a5 * b1 * c1 * c5 * d1 +
          16 * a3 * a5 * b1 * c2 * c3 * d2 - 16 * a3 * a5 * b1 * c2 * c4 * d1 -
          16 * a3 * a5 * b3 * c1 * c2 * d2 + 32 * a3 * a5 * b4 * c1 * c2 * d1 -
          16 * a4 * a5 * b1 * c1 * c2 * d3 - 16 * a4 * a5 * b1 * c1 * c3 * d2 -
          16 * a4 * a5 * b1 * c2 * c3 * d1 + 8 * a4 * a5 * b2 * c0 * c5 * d0 +
          32 * a4 * a5 * b3 * c1 * c2 * d1 - 8 * a4 * a5 * b5 * c0 * c2 * d0 -
          4 * a0 * a1 * b3 * c3 * c5 * d5 + 8 * a0 * a1 * b4 * c3 * c4 * d5 +
          8 * a0 * a1 * b4 * c3 * c5 * d4 + 8 * a0 * a1 * b4 * c4 * c5 * d3 -
          16 * a0 * a1 * b5 * c3 * c4 * d4 + 4 * a0 * a1 * b5 * c3 * c5 * d3 -
          8 * a0 * a2 * b4 * c3 * c5 * d3 + 8 * a0 * a2 * b5 * c3 * c4 * d3 -
          4 * a0 * a3 * b1 * c3 * c5 * d5 - 4 * a0 * a3 * b3 * c1 * c5 * d5 +
          8 * a0 * a3 * b3 * c2 * c4 * d5 + 8 * a0 * a3 * b3 * c2 * c5 * d4 +
          8 * a0 * a3 * b3 * c4 * c5 * d2 - 8 * a0 * a3 * b4 * c1 * c4 * d5 -
          8 * a0 * a3 * b4 * c1 * c5 * d4 - 8 * a0 * a3 * b4 * c4 * c5 * d1 +
          4 * a0 * a3 * b5 * c1 * c3 * d5 + 16 * a0 * a3 * b5 * c1 * c4 * d4 +
          4 * a0 * a3 * b5 * c1 * c5 * d3 - 8 * a0 * a3 * b5 * c2 * c3 * d4 -
          8 * a0 * a3 * b5 * c2 * c4 * d3 - 8 * a0 * a3 * b5 * c3 * c4 * d2 +
          4 * a0 * a3 * b5 * c3 * c5 * d1 + 8 * a0 * a4 * b1 * c3 * c4 * d5 +
          8 * a0 * a4 * b1 * c3 * c5 * d4 + 8 * a0 * a4 * b1 * c4 * c5 * d3 -
          8 * a0 * a4 * b2 * c3 * c5 * d3 - 8 * a0 * a4 * b3 * c1 * c4 * d5 -
          8 * a0 * a4 * b3 * c1 * c5 * d4 - 8 * a0 * a4 * b3 * c4 * c5 * d1 +
          8 * a0 * a4 * b5 * c2 * c3 * d3 - 16 * a0 * a5 * b1 * c3 * c4 * d4 +
          4 * a0 * a5 * b1 * c3 * c5 * d3 + 8 * a0 * a5 * b2 * c3 * c4 * d3 +
          4 * a0 * a5 * b3 * c1 * c3 * d5 + 16 * a0 * a5 * b3 * c1 * c4 * d4 +
          4 * a0 * a5 * b3 * c1 * c5 * d3 - 8 * a0 * a5 * b3 * c2 * c3 * d4 -
          8 * a0 * a5 * b3 * c2 * c4 * d3 - 8 * a0 * a5 * b3 * c3 * c4 * d2 +
          4 * a0 * a5 * b3 * c3 * c5 * d1 + 8 * a0 * a5 * b4 * c2 * c3 * d3 -
          12 * a0 * a5 * b5 * c1 * c3 * d3 - 4 * a1 * a3 * b0 * c3 * c5 * d5 +
          12 * a1 * a3 * b3 * c0 * c5 * d5 - 4 * a1 * a3 * b5 * c0 * c3 * d5 -
          4 * a1 * a3 * b5 * c0 * c5 * d3 - 4 * a1 * a3 * b5 * c3 * c5 * d0 +
          8 * a1 * a4 * b0 * c3 * c4 * d5 + 8 * a1 * a4 * b0 * c3 * c5 * d4 +
          8 * a1 * a4 * b0 * c4 * c5 * d3 - 16 * a1 * a4 * b4 * c0 * c3 * d5 -
          16 * a1 * a4 * b4 * c0 * c5 * d3 - 16 * a1 * a4 * b4 * c3 * c5 * d0 +
          8 * a1 * a4 * b5 * c0 * c3 * d4 + 8 * a1 * a4 * b5 * c0 * c4 * d3 +
          8 * a1 * a4 * b5 * c3 * c4 * d0 - 16 * a1 * a5 * b0 * c3 * c4 * d4 +
          4 * a1 * a5 * b0 * c3 * c5 * d3 - 4 * a1 * a5 * b3 * c0 * c3 * d5 -
          4 * a1 * a5 * b3 * c0 * c5 * d3 - 4 * a1 * a5 * b3 * c3 * c5 * d0 +
          8 * a1 * a5 * b4 * c0 * c3 * d4 + 8 * a1 * a5 * b4 * c0 * c4 * d3 +
          8 * a1 * a5 * b4 * c3 * c4 * d0 + 4 * a1 * a5 * b5 * c0 * c3 * d3 -
          8 * a2 * a3 * b3 * c0 * c4 * d5 - 8 * a2 * a3 * b3 * c0 * c5 * d4 -
          8 * a2 * a3 * b3 * c4 * c5 * d0 + 8 * a2 * a3 * b4 * c0 * c3 * d5 +
          8 * a2 * a3 * b4 * c0 * c5 * d3 + 8 * a2 * a3 * b4 * c3 * c5 * d0 -
          8 * a2 * a4 * b0 * c3 * c5 * d3 + 8 * a2 * a4 * b3 * c0 * c3 * d5 +
          8 * a2 * a4 * b3 * c0 * c5 * d3 + 8 * a2 * a4 * b3 * c3 * c5 * d0 -
          8 * a2 * a4 * b5 * c0 * c3 * d3 + 8 * a2 * a5 * b0 * c3 * c4 * d3 -
          8 * a2 * a5 * b4 * c0 * c3 * d3 - 8 * a3 * a4 * b0 * c1 * c4 * d5 -
          8 * a3 * a4 * b0 * c1 * c5 * d4 - 8 * a3 * a4 * b0 * c4 * c5 * d1 +
          8 * a3 * a4 * b2 * c0 * c3 * d5 + 8 * a3 * a4 * b2 * c0 * c5 * d3 +
          8 * a3 * a4 * b2 * c3 * c5 * d0 - 8 * a3 * a4 * b3 * c0 * c2 * d5 -
          8 * a3 * a4 * b3 * c0 * c5 * d2 - 8 * a3 * a4 * b3 * c2 * c5 * d0 +
          16 * a3 * a4 * b4 * c0 * c1 * d5 + 16 * a3 * a4 * b4 * c0 * c5 * d1 +
          16 * a3 * a4 * b4 * c1 * c5 * d0 - 8 * a3 * a4 * b5 * c0 * c1 * d4 -
          8 * a3 * a4 * b5 * c0 * c4 * d1 - 8 * a3 * a4 * b5 * c1 * c4 * d0 +
          4 * a3 * a5 * b0 * c1 * c3 * d5 + 16 * a3 * a5 * b0 * c1 * c4 * d4 +
          4 * a3 * a5 * b0 * c1 * c5 * d3 - 8 * a3 * a5 * b0 * c2 * c3 * d4 -
          8 * a3 * a5 * b0 * c2 * c4 * d3 - 8 * a3 * a5 * b0 * c3 * c4 * d2 +
          4 * a3 * a5 * b0 * c3 * c5 * d1 - 4 * a3 * a5 * b1 * c0 * c3 * d5 -
          4 * a3 * a5 * b1 * c0 * c5 * d3 - 4 * a3 * a5 * b1 * c3 * c5 * d0 -
          4 * a3 * a5 * b3 * c0 * c1 * d5 + 8 * a3 * a5 * b3 * c0 * c2 * d4 +
          8 * a3 * a5 * b3 * c0 * c4 * d2 - 4 * a3 * a5 * b3 * c0 * c5 * d1 -
          4 * a3 * a5 * b3 * c1 * c5 * d0 + 8 * a3 * a5 * b3 * c2 * c4 * d0 -
          8 * a3 * a5 * b4 * c0 * c1 * d4 - 8 * a3 * a5 * b4 * c0 * c4 * d1 -
          8 * a3 * a5 * b4 * c1 * c4 * d0 + 4 * a3 * a5 * b5 * c0 * c1 * d3 +
          4 * a3 * a5 * b5 * c0 * c3 * d1 + 4 * a3 * a5 * b5 * c1 * c3 * d0 +
          8 * a4 * a5 * b0 * c2 * c3 * d3 + 8 * a4 * a5 * b1 * c0 * c3 * d4 +
          8 * a4 * a5 * b1 * c0 * c4 * d3 + 8 * a4 * a5 * b1 * c3 * c4 * d0 -
          8 * a4 * a5 * b2 * c0 * c3 * d3 - 8 * a4 * a5 * b3 * c0 * c1 * d4 -
          8 * a4 * a5 * b3 * c0 * c4 * d1 - 8 * a4 * a5 * b3 * c1 * c4 * d0 +
          32 * a1 * a2 * b1 * c4 * c5 * d5 - 16 * a1 * a2 * b2 * c3 * c5 * d5 -
          32 * a1 * a2 * b4 * c1 * c5 * d5 + 8 * a1 * a2 * b5 * c2 * c3 * d5 +
          8 * a1 * a2 * b5 * c2 * c5 * d3 + 8 * a1 * a2 * b5 * c3 * c5 * d2 +
          32 * a1 * a4 * b1 * c2 * c5 * d5 - 32 * a1 * a4 * b2 * c1 * c5 * d5 +
          48 * a1 * a5 * b1 * c1 * c5 * d5 - 32 * a1 * a5 * b1 * c2 * c4 * d5 -
          32 * a1 * a5 * b1 * c2 * c5 * d4 - 32 * a1 * a5 * b1 * c4 * c5 * d2 +
          8 * a1 * a5 * b2 * c2 * c3 * d5 + 8 * a1 * a5 * b2 * c2 * c5 * d3 +
          8 * a1 * a5 * b2 * c3 * c5 * d2 + 32 * a1 * a5 * b5 * c1 * c2 * d4 +
          32 * a1 * a5 * b5 * c1 * c4 * d2 - 48 * a1 * a5 * b5 * c1 * c5 * d1 -
          16 * a1 * a5 * b5 * c2 * c3 * d2 + 32 * a1 * a5 * b5 * c2 * c4 * d1 +
          16 * a2 * a3 * b2 * c1 * c5 * d5 - 8 * a2 * a3 * b5 * c1 * c2 * d5 -
          8 * a2 * a3 * b5 * c1 * c5 * d2 - 8 * a2 * a3 * b5 * c2 * c5 * d1 -
          32 * a2 * a4 * b1 * c1 * c5 * d5 + 32 * a2 * a4 * b5 * c1 * c5 * d1 +
          8 * a2 * a5 * b1 * c2 * c3 * d5 + 8 * a2 * a5 * b1 * c2 * c5 * d3 +
          8 * a2 * a5 * b1 * c3 * c5 * d2 - 8 * a2 * a5 * b3 * c1 * c2 * d5 -
          8 * a2 * a5 * b3 * c1 * c5 * d2 - 8 * a2 * a5 * b3 * c2 * c5 * d1 +
          32 * a2 * a5 * b4 * c1 * c5 * d1 - 32 * a2 * a5 * b5 * c1 * c4 * d1 -
          8 * a3 * a5 * b2 * c1 * c2 * d5 - 8 * a3 * a5 * b2 * c1 * c5 * d2 -
          8 * a3 * a5 * b2 * c2 * c5 * d1 + 16 * a3 * a5 * b5 * c1 * c2 * d2 +
          32 * a4 * a5 * b2 * c1 * c5 * d1 - 32 * a4 * a5 * b5 * c1 * c2 * d1 -
          4 * a0 * a1 * b5 * c3 * c5 * d5 - 8 * a0 * a2 * b3 * c4 * c5 * d5 +
          8 * a0 * a2 * b4 * c3 * c5 * d5 - 8 * a0 * a3 * b2 * c4 * c5 * d5 -
          8 * a0 * a3 * b4 * c2 * c5 * d5 - 4 * a0 * a3 * b5 * c1 * c5 * d5 +
          8 * a0 * a3 * b5 * c2 * c4 * d5 + 8 * a0 * a3 * b5 * c2 * c5 * d4 +
          8 * a0 * a3 * b5 * c4 * c5 * d2 + 8 * a0 * a4 * b2 * c3 * c5 * d5 -
          8 * a0 * a4 * b3 * c2 * c5 * d5 + 16 * a0 * a4 * b4 * c1 * c5 * d5 -
          8 * a0 * a4 * b5 * c1 * c4 * d5 - 8 * a0 * a4 * b5 * c1 * c5 * d4 -
          8 * a0 * a4 * b5 * c4 * c5 * d1 - 4 * a0 * a5 * b1 * c3 * c5 * d5 -
          4 * a0 * a5 * b3 * c1 * c5 * d5 + 8 * a0 * a5 * b3 * c2 * c4 * d5 +
          8 * a0 * a5 * b3 * c2 * c5 * d4 + 8 * a0 * a5 * b3 * c4 * c5 * d2 -
          8 * a0 * a5 * b4 * c1 * c4 * d5 - 8 * a0 * a5 * b4 * c1 * c5 * d4 -
          8 * a0 * a5 * b4 * c4 * c5 * d1 + 4 * a0 * a5 * b5 * c1 * c3 * d5 +
          16 * a0 * a5 * b5 * c1 * c4 * d4 + 4 * a0 * a5 * b5 * c1 * c5 * d3 -
          8 * a0 * a5 * b5 * c2 * c3 * d4 - 8 * a0 * a5 * b5 * c2 * c4 * d3 -
          8 * a0 * a5 * b5 * c3 * c4 * d2 + 4 * a0 * a5 * b5 * c3 * c5 * d1 -
          4 * a1 * a3 * b5 * c0 * c5 * d5 - 16 * a1 * a4 * b4 * c0 * c5 * d5 +
          8 * a1 * a4 * b5 * c0 * c4 * d5 + 8 * a1 * a4 * b5 * c0 * c5 * d4 +
          8 * a1 * a4 * b5 * c4 * c5 * d0 - 4 * a1 * a5 * b0 * c3 * c5 * d5 -
          4 * a1 * a5 * b3 * c0 * c5 * d5 + 8 * a1 * a5 * b4 * c0 * c4 * d5 +
          8 * a1 * a5 * b4 * c0 * c5 * d4 + 8 * a1 * a5 * b4 * c4 * c5 * d0 +
          4 * a1 * a5 * b5 * c0 * c3 * d5 - 16 * a1 * a5 * b5 * c0 * c4 * d4 +
          4 * a1 * a5 * b5 * c0 * c5 * d3 + 4 * a1 * a5 * b5 * c3 * c5 * d0 -
          8 * a2 * a3 * b0 * c4 * c5 * d5 + 8 * a2 * a3 * b4 * c0 * c5 * d5 +
          8 * a2 * a4 * b0 * c3 * c5 * d5 + 8 * a2 * a4 * b3 * c0 * c5 * d5 -
          8 * a2 * a4 * b5 * c0 * c3 * d5 - 8 * a2 * a4 * b5 * c0 * c5 * d3 -
          8 * a2 * a4 * b5 * c3 * c5 * d0 - 8 * a2 * a5 * b4 * c0 * c3 * d5 -
          8 * a2 * a5 * b4 * c0 * c5 * d3 - 8 * a2 * a5 * b4 * c3 * c5 * d0 +
          8 * a2 * a5 * b5 * c0 * c3 * d4 + 8 * a2 * a5 * b5 * c0 * c4 * d3 +
          8 * a2 * a5 * b5 * c3 * c4 * d0 - 8 * a3 * a4 * b0 * c2 * c5 * d5 +
          8 * a3 * a4 * b2 * c0 * c5 * d5 - 4 * a3 * a5 * b0 * c1 * c5 * d5 +
          8 * a3 * a5 * b0 * c2 * c4 * d5 + 8 * a3 * a5 * b0 * c2 * c5 * d4 +
          8 * a3 * a5 * b0 * c4 * c5 * d2 - 4 * a3 * a5 * b1 * c0 * c5 * d5 +
          4 * a3 * a5 * b5 * c0 * c1 * d5 - 8 * a3 * a5 * b5 * c0 * c2 * d4 -
          8 * a3 * a5 * b5 * c0 * c4 * d2 + 4 * a3 * a5 * b5 * c0 * c5 * d1 +
          4 * a3 * a5 * b5 * c1 * c5 * d0 - 8 * a3 * a5 * b5 * c2 * c4 * d0 -
          8 * a4 * a5 * b0 * c1 * c4 * d5 - 8 * a4 * a5 * b0 * c1 * c5 * d4 -
          8 * a4 * a5 * b0 * c4 * c5 * d1 + 8 * a4 * a5 * b1 * c0 * c4 * d5 +
          8 * a4 * a5 * b1 * c0 * c5 * d4 + 8 * a4 * a5 * b1 * c4 * c5 * d0 -
          8 * a4 * a5 * b2 * c0 * c3 * d5 - 8 * a4 * a5 * b2 * c0 * c5 * d3 -
          8 * a4 * a5 * b2 * c3 * c5 * d0 + 8 * a4 * a5 * b5 * c0 * c2 * d3 +
          8 * a4 * a5 * b5 * c0 * c3 * d2 + 8 * a4 * a5 * b5 * c2 * c3 * d0 +
          4 * a1 * a3 * b5 * c3 * c5 * d5 + 16 * a1 * a4 * b4 * c3 * c5 * d5 -
          8 * a1 * a4 * b5 * c3 * c4 * d5 - 8 * a1 * a4 * b5 * c3 * c5 * d4 -
          8 * a1 * a4 * b5 * c4 * c5 * d3 + 4 * a1 * a5 * b3 * c3 * c5 * d5 -
          8 * a1 * a5 * b4 * c3 * c4 * d5 - 8 * a1 * a5 * b4 * c3 * c5 * d4 -
          8 * a1 * a5 * b4 * c4 * c5 * d3 + 16 * a1 * a5 * b5 * c3 * c4 * d4 -
          4 * a1 * a5 * b5 * c3 * c5 * d3 + 8 * a2 * a3 * b3 * c4 * c5 * d5 -
          8 * a2 * a3 * b4 * c3 * c5 * d5 - 8 * a2 * a4 * b3 * c3 * c5 * d5 +
          8 * a2 * a4 * b5 * c3 * c5 * d3 + 8 * a2 * a5 * b4 * c3 * c5 * d3 -
          8 * a2 * a5 * b5 * c3 * c4 * d3 - 8 * a3 * a4 * b2 * c3 * c5 * d5 +
          8 * a3 * a4 * b3 * c2 * c5 * d5 - 16 * a3 * a4 * b4 * c1 * c5 * d5 +
          8 * a3 * a4 * b5 * c1 * c4 * d5 + 8 * a3 * a4 * b5 * c1 * c5 * d4 +
          8 * a3 * a4 * b5 * c4 * c5 * d1 + 4 * a3 * a5 * b1 * c3 * c5 * d5 +
          4 * a3 * a5 * b3 * c1 * c5 * d5 - 8 * a3 * a5 * b3 * c2 * c4 * d5 -
          8 * a3 * a5 * b3 * c2 * c5 * d4 - 8 * a3 * a5 * b3 * c4 * c5 * d2 +
          8 * a3 * a5 * b4 * c1 * c4 * d5 + 8 * a3 * a5 * b4 * c1 * c5 * d4 +
          8 * a3 * a5 * b4 * c4 * c5 * d1 - 4 * a3 * a5 * b5 * c1 * c3 * d5 -
          16 * a3 * a5 * b5 * c1 * c4 * d4 - 4 * a3 * a5 * b5 * c1 * c5 * d3 +
          8 * a3 * a5 * b5 * c2 * c3 * d4 + 8 * a3 * a5 * b5 * c2 * c4 * d3 +
          8 * a3 * a5 * b5 * c3 * c4 * d2 - 4 * a3 * a5 * b5 * c3 * c5 * d1 -
          8 * a4 * a5 * b1 * c3 * c4 * d5 - 8 * a4 * a5 * b1 * c3 * c5 * d4 -
          8 * a4 * a5 * b1 * c4 * c5 * d3 + 8 * a4 * a5 * b2 * c3 * c5 * d3 +
          8 * a4 * a5 * b3 * c1 * c4 * d5 + 8 * a4 * a5 * b3 * c1 * c5 * d4 +
          8 * a4 * a5 * b3 * c4 * c5 * d1 - 8 * a4 * a5 * b5 * c2 * c3 * d3,
      pow(a3, 2) * b1 * pow(c0, 3) - 4 * pow(a0, 2) * b3 * pow(c1, 3) -
          4 * pow(a4, 2) * b1 * pow(c0, 3) - pow(a0, 2) * b1 * pow(c5, 3) +
          8 * pow(a0, 2) * b4 * pow(c2, 3) + 4 * pow(a0, 2) * b5 * pow(c1, 3) -
          16 * pow(a2, 2) * b3 * pow(c1, 3) - 8 * pow(a5, 2) * b0 * pow(c1, 3) +
          pow(a5, 2) * b1 * pow(c0, 3) - 12 * pow(a1, 2) * b1 * pow(c5, 3) +
          32 * pow(a1, 2) * b4 * pow(c2, 3) +
          16 * pow(a2, 2) * b5 * pow(c1, 3) - pow(a3, 2) * b1 * pow(c5, 3) +
          8 * pow(a3, 2) * b4 * pow(c2, 3) - 4 * pow(a5, 2) * b3 * pow(c1, 3) +
          12 * pow(a5, 2) * b5 * pow(c1, 3) - 8 * a0 * a3 * b0 * pow(c1, 3) -
          2 * a0 * a1 * b0 * pow(c5, 3) + 16 * a0 * a4 * b0 * pow(c2, 3) +
          8 * a0 * a5 * b0 * pow(c1, 3) + 2 * a1 * a3 * b3 * pow(c0, 3) +
          64 * a1 * a4 * b1 * pow(c2, 3) - 32 * a2 * a3 * b2 * pow(c1, 3) +
          2 * a0 * a1 * b3 * pow(c5, 3) + 2 * a0 * a3 * b1 * pow(c5, 3) -
          16 * a0 * a3 * b4 * pow(c2, 3) + 8 * a0 * a3 * b5 * pow(c1, 3) -
          16 * a0 * a4 * b3 * pow(c2, 3) + 8 * a0 * a5 * b3 * pow(c1, 3) +
          2 * a1 * a3 * b0 * pow(c5, 3) - 2 * a1 * a3 * b5 * pow(c0, 3) -
          8 * a1 * a4 * b4 * pow(c0, 3) - 2 * a1 * a5 * b3 * pow(c0, 3) +
          4 * a2 * a3 * b4 * pow(c0, 3) + 4 * a2 * a4 * b3 * pow(c0, 3) -
          16 * a3 * a4 * b0 * pow(c2, 3) + 4 * a3 * a4 * b2 * pow(c0, 3) +
          8 * a3 * a5 * b0 * pow(c1, 3) - 2 * a3 * a5 * b1 * pow(c0, 3) +
          32 * a2 * a5 * b2 * pow(c1, 3) - 16 * a0 * a5 * b5 * pow(c1, 3) +
          2 * a1 * a5 * b5 * pow(c0, 3) - 4 * a2 * a4 * b5 * pow(c0, 3) -
          4 * a2 * a5 * b4 * pow(c0, 3) - 4 * a4 * a5 * b2 * pow(c0, 3) -
          2 * a1 * a3 * b3 * pow(c5, 3) + 16 * a3 * a4 * b3 * pow(c2, 3) -
          8 * a3 * a5 * b5 * pow(c1, 3) -
          3 * pow(a0, 2) * b0 * c1 * pow(c3, 2) +
          pow(a0, 2) * b1 * c0 * pow(c3, 2) -
          pow(a3, 2) * b0 * pow(c0, 2) * c1 +
          12 * pow(a0, 2) * b0 * c1 * pow(c4, 2) -
          4 * pow(a0, 2) * b1 * c0 * pow(c4, 2) +
          4 * pow(a0, 2) * b1 * pow(c1, 2) * c3 +
          12 * pow(a1, 2) * b1 * pow(c0, 2) * c3 -
          4 * pow(a1, 2) * b3 * pow(c0, 2) * c1 +
          4 * pow(a4, 2) * b0 * pow(c0, 2) * c1 -
          3 * pow(a0, 2) * b0 * c1 * pow(c5, 2) +
          pow(a0, 2) * b1 * c0 * pow(c5, 2) +
          4 * pow(a0, 2) * b1 * pow(c2, 2) * c3 -
          4 * pow(a0, 2) * b3 * c1 * pow(c2, 2) -
          4 * pow(a2, 2) * b0 * c1 * pow(c3, 2) -
          4 * pow(a2, 2) * b1 * c0 * pow(c3, 2) +
          4 * pow(a2, 2) * b1 * pow(c0, 2) * c3 -
          4 * pow(a2, 2) * b3 * pow(c0, 2) * c1 +
          4 * pow(a3, 2) * b0 * c1 * pow(c2, 2) +
          4 * pow(a3, 2) * b1 * c0 * pow(c2, 2) -
          pow(a5, 2) * b0 * pow(c0, 2) * c1 -
          4 * pow(a0, 2) * b1 * pow(c1, 2) * c5 -
          8 * pow(a0, 2) * b2 * pow(c1, 2) * c4 +
          8 * pow(a0, 2) * b4 * pow(c1, 2) * c2 -
          8 * pow(a1, 2) * b0 * c1 * pow(c5, 2) +
          24 * pow(a1, 2) * b1 * c0 * pow(c5, 2) -
          12 * pow(a1, 2) * b1 * pow(c0, 2) * c5 +
          48 * pow(a1, 2) * b1 * pow(c2, 2) * c3 -
          8 * pow(a1, 2) * b2 * pow(c0, 2) * c4 -
          16 * pow(a1, 2) * b3 * c1 * pow(c2, 2) +
          8 * pow(a1, 2) * b4 * pow(c0, 2) * c2 +
          4 * pow(a1, 2) * b5 * pow(c0, 2) * c1 +
          16 * pow(a2, 2) * b0 * c1 * pow(c4, 2) -
          16 * pow(a2, 2) * b1 * c0 * pow(c4, 2) +
          16 * pow(a2, 2) * b1 * pow(c1, 2) * c3 +
          16 * pow(a4, 2) * b0 * c1 * pow(c2, 2) -
          16 * pow(a4, 2) * b1 * c0 * pow(c2, 2) +
          8 * pow(a5, 2) * b1 * c0 * pow(c1, 2) +
          4 * pow(a0, 2) * b1 * c3 * pow(c4, 2) -
          4 * pow(a0, 2) * b1 * pow(c2, 2) * c5 -
          8 * pow(a0, 2) * b2 * pow(c2, 2) * c4 -
          4 * pow(a0, 2) * b3 * c1 * pow(c4, 2) +
          4 * pow(a0, 2) * b5 * c1 * pow(c2, 2) -
          4 * pow(a2, 2) * b0 * c1 * pow(c5, 2) +
          4 * pow(a2, 2) * b1 * c0 * pow(c5, 2) -
          4 * pow(a2, 2) * b1 * pow(c0, 2) * c5 -
          24 * pow(a2, 2) * b2 * pow(c0, 2) * c4 +
          8 * pow(a2, 2) * b4 * pow(c0, 2) * c2 +
          4 * pow(a2, 2) * b5 * pow(c0, 2) * c1 +
          4 * pow(a4, 2) * b1 * pow(c0, 2) * c3 -
          4 * pow(a4, 2) * b3 * pow(c0, 2) * c1 -
          4 * pow(a5, 2) * b0 * c1 * pow(c2, 2) +
          4 * pow(a5, 2) * b1 * c0 * pow(c2, 2) +
          2 * pow(a0, 2) * b1 * c3 * pow(c5, 2) -
          pow(a0, 2) * b1 * pow(c3, 2) * c5 -
          2 * pow(a0, 2) * b2 * pow(c3, 2) * c4 +
          2 * pow(a0, 2) * b3 * c1 * pow(c5, 2) -
          2 * pow(a0, 2) * b4 * c2 * pow(c3, 2) +
          3 * pow(a0, 2) * b5 * c1 * pow(c3, 2) -
          48 * pow(a1, 2) * b1 * pow(c2, 2) * c5 -
          32 * pow(a1, 2) * b2 * pow(c2, 2) * c4 +
          16 * pow(a1, 2) * b5 * c1 * pow(c2, 2) -
          16 * pow(a2, 2) * b1 * pow(c1, 2) * c5 -
          96 * pow(a2, 2) * b2 * pow(c1, 2) * c4 +
          32 * pow(a2, 2) * b4 * pow(c1, 2) * c2 -
          pow(a3, 2) * b0 * c1 * pow(c5, 2) +
          3 * pow(a3, 2) * b1 * c0 * pow(c5, 2) -
          3 * pow(a3, 2) * b1 * pow(c0, 2) * c5 +
          2 * pow(a3, 2) * b2 * pow(c0, 2) * c4 +
          2 * pow(a3, 2) * b4 * pow(c0, 2) * c2 +
          pow(a3, 2) * b5 * pow(c0, 2) * c1 -
          3 * pow(a5, 2) * b0 * c1 * pow(c3, 2) +
          pow(a5, 2) * b1 * c0 * pow(c3, 2) -
          2 * pow(a5, 2) * b1 * pow(c0, 2) * c3 -
          2 * pow(a5, 2) * b3 * pow(c0, 2) * c1 -
          8 * pow(a0, 2) * b5 * c1 * pow(c4, 2) +
          12 * pow(a1, 2) * b1 * c3 * pow(c5, 2) -
          4 * pow(a1, 2) * b3 * c1 * pow(c5, 2) +
          16 * pow(a2, 2) * b1 * c3 * pow(c4, 2) -
          16 * pow(a2, 2) * b3 * c1 * pow(c4, 2) +
          4 * pow(a4, 2) * b0 * c1 * pow(c5, 2) -
          4 * pow(a4, 2) * b1 * c0 * pow(c5, 2) +
          8 * pow(a4, 2) * b1 * pow(c0, 2) * c5 +
          16 * pow(a4, 2) * b1 * pow(c2, 2) * c3 -
          16 * pow(a4, 2) * b3 * c1 * pow(c2, 2) +
          4 * pow(a5, 2) * b0 * c1 * pow(c4, 2) -
          4 * pow(a5, 2) * b1 * c0 * pow(c4, 2) +
          4 * pow(a5, 2) * b1 * pow(c1, 2) * c3 +
          2 * pow(a0, 2) * b2 * c4 * pow(c5, 2) +
          2 * pow(a0, 2) * b4 * c2 * pow(c5, 2) +
          pow(a0, 2) * b5 * c1 * pow(c5, 2) -
          4 * pow(a2, 2) * b1 * c3 * pow(c5, 2) +
          4 * pow(a2, 2) * b1 * pow(c3, 2) * c5 -
          24 * pow(a2, 2) * b2 * pow(c3, 2) * c4 +
          4 * pow(a2, 2) * b3 * c1 * pow(c5, 2) +
          8 * pow(a2, 2) * b4 * c2 * pow(c3, 2) +
          4 * pow(a2, 2) * b5 * c1 * pow(c3, 2) -
          4 * pow(a3, 2) * b1 * pow(c2, 2) * c5 -
          8 * pow(a3, 2) * b2 * pow(c2, 2) * c4 -
          4 * pow(a3, 2) * b5 * c1 * pow(c2, 2) -
          pow(a5, 2) * b1 * pow(c0, 2) * c5 -
          4 * pow(a5, 2) * b1 * pow(c2, 2) * c3 -
          2 * pow(a5, 2) * b2 * pow(c0, 2) * c4 +
          4 * pow(a5, 2) * b3 * c1 * pow(c2, 2) -
          2 * pow(a5, 2) * b4 * pow(c0, 2) * c2 +
          3 * pow(a5, 2) * b5 * pow(c0, 2) * c1 +
          8 * pow(a1, 2) * b2 * c4 * pow(c5, 2) +
          8 * pow(a1, 2) * b4 * c2 * pow(c5, 2) +
          12 * pow(a1, 2) * b5 * c1 * pow(c5, 2) -
          12 * pow(a5, 2) * b1 * pow(c1, 2) * c5 -
          8 * pow(a5, 2) * b2 * pow(c1, 2) * c4 -
          8 * pow(a5, 2) * b4 * pow(c1, 2) * c2 +
          4 * pow(a4, 2) * b1 * c3 * pow(c5, 2) -
          4 * pow(a4, 2) * b3 * c1 * pow(c5, 2) +
          4 * pow(a5, 2) * b1 * c3 * pow(c4, 2) -
          4 * pow(a5, 2) * b3 * c1 * pow(c4, 2) +
          2 * pow(a3, 2) * b2 * c4 * pow(c5, 2) +
          2 * pow(a3, 2) * b4 * c2 * pow(c5, 2) +
          pow(a3, 2) * b5 * c1 * pow(c5, 2) -
          pow(a5, 2) * b1 * pow(c3, 2) * c5 -
          2 * pow(a5, 2) * b2 * pow(c3, 2) * c4 -
          2 * pow(a5, 2) * b4 * c2 * pow(c3, 2) +
          3 * pow(a5, 2) * b5 * c1 * pow(c3, 2) +
          2 * a0 * a1 * b0 * c0 * pow(c3, 2) -
          8 * a0 * a1 * b0 * c0 * pow(c4, 2) +
          8 * a0 * a1 * b0 * pow(c1, 2) * c3 +
          8 * a0 * a1 * b3 * c0 * pow(c1, 2) +
          8 * a0 * a3 * b1 * c0 * pow(c1, 2) +
          8 * a1 * a3 * b0 * c0 * pow(c1, 2) +
          2 * a0 * a1 * b0 * c0 * pow(c5, 2) +
          8 * a0 * a1 * b0 * pow(c2, 2) * c3 -
          8 * a0 * a3 * b0 * c1 * pow(c2, 2) -
          8 * a1 * a3 * b1 * pow(c0, 2) * c1 -
          8 * a0 * a1 * b0 * pow(c1, 2) * c5 -
          2 * a0 * a1 * b3 * pow(c0, 2) * c3 -
          8 * a0 * a1 * b5 * c0 * pow(c1, 2) -
          16 * a0 * a2 * b0 * pow(c1, 2) * c4 -
          2 * a0 * a3 * b1 * pow(c0, 2) * c3 -
          2 * a0 * a3 * b3 * pow(c0, 2) * c1 +
          16 * a0 * a4 * b0 * pow(c1, 2) * c2 -
          8 * a0 * a5 * b1 * c0 * pow(c1, 2) -
          2 * a1 * a3 * b0 * pow(c0, 2) * c3 -
          8 * a1 * a5 * b0 * c0 * pow(c1, 2) +
          8 * a0 * a1 * b0 * c3 * pow(c4, 2) -
          8 * a0 * a1 * b0 * pow(c2, 2) * c5 -
          16 * a0 * a1 * b1 * c1 * pow(c5, 2) +
          8 * a0 * a1 * b2 * c2 * pow(c3, 2) -
          16 * a0 * a2 * b0 * pow(c2, 2) * c4 +
          8 * a0 * a2 * b1 * c2 * pow(c3, 2) -
          8 * a0 * a2 * b2 * c1 * pow(c3, 2) -
          16 * a0 * a2 * b4 * c0 * pow(c2, 2) -
          8 * a0 * a3 * b0 * c1 * pow(c4, 2) -
          16 * a0 * a4 * b2 * c0 * pow(c2, 2) +
          8 * a0 * a5 * b0 * c1 * pow(c2, 2) +
          8 * a1 * a2 * b0 * c2 * pow(c3, 2) -
          16 * a1 * a2 * b1 * pow(c0, 2) * c4 -
          8 * a1 * a2 * b2 * c0 * pow(c3, 2) +
          8 * a1 * a2 * b2 * pow(c0, 2) * c3 -
          32 * a1 * a3 * b1 * c1 * pow(c2, 2) +
          16 * a1 * a4 * b1 * pow(c0, 2) * c2 +
          8 * a1 * a5 * b1 * pow(c0, 2) * c1 -
          8 * a2 * a3 * b2 * pow(c0, 2) * c1 -
          16 * a2 * a4 * b0 * c0 * pow(c2, 2) +
          4 * a0 * a1 * b0 * c3 * pow(c5, 2) -
          2 * a0 * a1 * b0 * pow(c3, 2) * c5 -
          4 * a0 * a1 * b3 * c0 * pow(c5, 2) +
          2 * a0 * a1 * b3 * pow(c0, 2) * c5 -
          8 * a0 * a1 * b3 * pow(c2, 2) * c3 +
          8 * a0 * a1 * b4 * pow(c0, 2) * c4 -
          2 * a0 * a1 * b5 * c0 * pow(c3, 2) +
          2 * a0 * a1 * b5 * pow(c0, 2) * c3 -
          4 * a0 * a2 * b0 * pow(c3, 2) * c4 +
          32 * a0 * a2 * b2 * c1 * pow(c4, 2) -
          4 * a0 * a2 * b3 * pow(c0, 2) * c4 +
          4 * a0 * a2 * b4 * c0 * pow(c3, 2) -
          4 * a0 * a2 * b4 * pow(c0, 2) * c3 +
          4 * a0 * a3 * b0 * c1 * pow(c5, 2) -
          4 * a0 * a3 * b1 * c0 * pow(c5, 2) +
          2 * a0 * a3 * b1 * pow(c0, 2) * c5 -
          8 * a0 * a3 * b1 * pow(c2, 2) * c3 -
          4 * a0 * a3 * b2 * pow(c0, 2) * c4 +
          8 * a0 * a3 * b3 * c1 * pow(c2, 2) -
          4 * a0 * a3 * b4 * pow(c0, 2) * c2 +
          2 * a0 * a3 * b5 * pow(c0, 2) * c1 -
          4 * a0 * a4 * b0 * c2 * pow(c3, 2) +
          8 * a0 * a4 * b1 * pow(c0, 2) * c4 +
          4 * a0 * a4 * b2 * c0 * pow(c3, 2) -
          4 * a0 * a4 * b2 * pow(c0, 2) * c3 -
          4 * a0 * a4 * b3 * pow(c0, 2) * c2 +
          8 * a0 * a4 * b4 * pow(c0, 2) * c1 +
          6 * a0 * a5 * b0 * c1 * pow(c3, 2) -
          2 * a0 * a5 * b1 * c0 * pow(c3, 2) +
          2 * a0 * a5 * b1 * pow(c0, 2) * c3 +
          2 * a0 * a5 * b3 * pow(c0, 2) * c1 -
          32 * a1 * a2 * b2 * c0 * pow(c4, 2) +
          32 * a1 * a2 * b2 * pow(c1, 2) * c3 +
          32 * a1 * a2 * b3 * pow(c1, 2) * c2 -
          4 * a1 * a3 * b0 * c0 * pow(c5, 2) +
          2 * a1 * a3 * b0 * pow(c0, 2) * c5 -
          8 * a1 * a3 * b0 * pow(c2, 2) * c3 +
          32 * a1 * a3 * b2 * pow(c1, 2) * c2 +
          8 * a1 * a3 * b3 * c0 * pow(c2, 2) +
          8 * a1 * a4 * b0 * pow(c0, 2) * c4 -
          2 * a1 * a5 * b0 * c0 * pow(c3, 2) +
          2 * a1 * a5 * b0 * pow(c0, 2) * c3 -
          4 * a2 * a3 * b0 * pow(c0, 2) * c4 +
          32 * a2 * a3 * b1 * pow(c1, 2) * c2 +
          4 * a2 * a4 * b0 * c0 * pow(c3, 2) -
          4 * a2 * a4 * b0 * pow(c0, 2) * c3 -
          4 * a3 * a4 * b0 * pow(c0, 2) * c2 +
          2 * a3 * a5 * b0 * pow(c0, 2) * c1 -
          8 * a0 * a1 * b3 * pow(c1, 2) * c5 +
          8 * a0 * a1 * b5 * c0 * pow(c4, 2) -
          8 * a0 * a1 * b5 * pow(c1, 2) * c3 -
          8 * a0 * a2 * b2 * c1 * pow(c5, 2) -
          8 * a0 * a3 * b1 * pow(c1, 2) * c5 -
          16 * a0 * a3 * b4 * pow(c1, 2) * c2 -
          16 * a0 * a4 * b3 * pow(c1, 2) * c2 -
          16 * a0 * a5 * b0 * c1 * pow(c4, 2) +
          8 * a0 * a5 * b1 * c0 * pow(c4, 2) -
          8 * a0 * a5 * b1 * pow(c1, 2) * c3 -
          64 * a1 * a2 * b1 * pow(c2, 2) * c4 +
          8 * a1 * a2 * b2 * c0 * pow(c5, 2) -
          8 * a1 * a2 * b2 * pow(c0, 2) * c5 -
          64 * a1 * a2 * b4 * c1 * pow(c2, 2) -
          8 * a1 * a3 * b0 * pow(c1, 2) * c5 -
          8 * a1 * a3 * b5 * c0 * pow(c1, 2) -
          64 * a1 * a4 * b2 * c1 * pow(c2, 2) +
          8 * a1 * a5 * b0 * c0 * pow(c4, 2) -
          8 * a1 * a5 * b0 * pow(c1, 2) * c3 +
          32 * a1 * a5 * b1 * c1 * pow(c2, 2) -
          8 * a1 * a5 * b3 * c0 * pow(c1, 2) +
          16 * a2 * a3 * b4 * c0 * pow(c1, 2) -
          64 * a2 * a4 * b1 * c1 * pow(c2, 2) +
          16 * a2 * a4 * b2 * pow(c0, 2) * c2 +
          16 * a2 * a4 * b3 * c0 * pow(c1, 2) +
          8 * a2 * a5 * b2 * pow(c0, 2) * c1 -
          16 * a3 * a4 * b0 * pow(c1, 2) * c2 +
          16 * a3 * a4 * b2 * c0 * pow(c1, 2) -
          8 * a3 * a5 * b1 * c0 * pow(c1, 2) +
          8 * a0 * a1 * b3 * pow(c2, 2) * c5 +
          2 * a0 * a1 * b5 * c0 * pow(c5, 2) -
          2 * a0 * a1 * b5 * pow(c0, 2) * c5 +
          4 * a0 * a2 * b0 * c4 * pow(c5, 2) +
          16 * a0 * a2 * b3 * pow(c2, 2) * c4 -
          4 * a0 * a2 * b4 * c0 * pow(c5, 2) +
          4 * a0 * a2 * b4 * pow(c0, 2) * c5 +
          16 * a0 * a2 * b4 * pow(c2, 2) * c3 +
          4 * a0 * a2 * b5 * pow(c0, 2) * c4 +
          8 * a0 * a3 * b1 * pow(c2, 2) * c5 +
          16 * a0 * a3 * b2 * pow(c2, 2) * c4 +
          4 * a0 * a4 * b0 * c2 * pow(c5, 2) -
          4 * a0 * a4 * b2 * c0 * pow(c5, 2) +
          4 * a0 * a4 * b2 * pow(c0, 2) * c5 +
          16 * a0 * a4 * b2 * pow(c2, 2) * c3 +
          32 * a0 * a4 * b4 * c1 * pow(c2, 2) +
          4 * a0 * a4 * b5 * pow(c0, 2) * c2 +
          2 * a0 * a5 * b0 * c1 * pow(c5, 2) +
          2 * a0 * a5 * b1 * c0 * pow(c5, 2) -
          2 * a0 * a5 * b1 * pow(c0, 2) * c5 +
          4 * a0 * a5 * b2 * pow(c0, 2) * c4 +
          4 * a0 * a5 * b4 * pow(c0, 2) * c2 -
          2 * a0 * a5 * b5 * pow(c0, 2) * c1 -
          32 * a1 * a2 * b2 * pow(c1, 2) * c5 -
          32 * a1 * a2 * b5 * pow(c1, 2) * c2 +
          8 * a1 * a3 * b0 * pow(c2, 2) * c5 -
          8 * a1 * a3 * b1 * c1 * pow(c5, 2) -
          8 * a1 * a3 * b5 * c0 * pow(c2, 2) -
          32 * a1 * a4 * b4 * c0 * pow(c2, 2) +
          2 * a1 * a5 * b0 * c0 * pow(c5, 2) -
          2 * a1 * a5 * b0 * pow(c0, 2) * c5 -
          32 * a1 * a5 * b2 * pow(c1, 2) * c2 -
          8 * a1 * a5 * b3 * c0 * pow(c2, 2) +
          16 * a2 * a3 * b0 * pow(c2, 2) * c4 +
          16 * a2 * a3 * b4 * c0 * pow(c2, 2) -
          4 * a2 * a4 * b0 * c0 * pow(c5, 2) +
          4 * a2 * a4 * b0 * pow(c0, 2) * c5 +
          16 * a2 * a4 * b0 * pow(c2, 2) * c3 +
          64 * a2 * a4 * b2 * pow(c1, 2) * c2 +
          16 * a2 * a4 * b3 * c0 * pow(c2, 2) +
          4 * a2 * a5 * b0 * pow(c0, 2) * c4 -
          32 * a2 * a5 * b1 * pow(c1, 2) * c2 +
          16 * a3 * a4 * b2 * c0 * pow(c2, 2) -
          8 * a3 * a5 * b1 * c0 * pow(c2, 2) +
          4 * a4 * a5 * b0 * pow(c0, 2) * c2 -
          2 * a0 * a1 * b3 * c3 * pow(c5, 2) +
          16 * a0 * a1 * b5 * pow(c1, 2) * c5 +
          16 * a0 * a2 * b5 * pow(c1, 2) * c4 -
          2 * a0 * a3 * b1 * c3 * pow(c5, 2) -
          2 * a0 * a3 * b3 * c1 * pow(c5, 2) +
          16 * a0 * a5 * b1 * pow(c1, 2) * c5 +
          16 * a0 * a5 * b2 * pow(c1, 2) * c4 +
          32 * a1 * a2 * b2 * c3 * pow(c4, 2) -
          2 * a1 * a3 * b0 * c3 * pow(c5, 2) +
          6 * a1 * a3 * b3 * c0 * pow(c5, 2) -
          6 * a1 * a3 * b3 * pow(c0, 2) * c5 +
          2 * a1 * a3 * b5 * pow(c0, 2) * c3 +
          8 * a1 * a4 * b4 * pow(c0, 2) * c3 +
          16 * a1 * a5 * b0 * pow(c1, 2) * c5 +
          2 * a1 * a5 * b3 * pow(c0, 2) * c3 +
          16 * a1 * a5 * b5 * c0 * pow(c1, 2) -
          32 * a2 * a3 * b2 * c1 * pow(c4, 2) +
          4 * a2 * a3 * b3 * pow(c0, 2) * c4 -
          4 * a2 * a3 * b4 * pow(c0, 2) * c3 -
          4 * a2 * a4 * b3 * pow(c0, 2) * c3 -
          16 * a2 * a4 * b5 * c0 * pow(c1, 2) +
          16 * a2 * a5 * b0 * pow(c1, 2) * c4 -
          16 * a2 * a5 * b4 * c0 * pow(c1, 2) -
          4 * a3 * a4 * b2 * pow(c0, 2) * c3 +
          4 * a3 * a4 * b3 * pow(c0, 2) * c2 -
          8 * a3 * a4 * b4 * pow(c0, 2) * c1 +
          2 * a3 * a5 * b1 * pow(c0, 2) * c3 +
          2 * a3 * a5 * b3 * pow(c0, 2) * c1 -
          16 * a4 * a5 * b2 * c0 * pow(c1, 2) -
          8 * a0 * a1 * b5 * c3 * pow(c4, 2) +
          8 * a0 * a3 * b5 * c1 * pow(c4, 2) -
          8 * a0 * a5 * b1 * c3 * pow(c4, 2) +
          8 * a0 * a5 * b3 * c1 * pow(c4, 2) -
          8 * a0 * a5 * b5 * c1 * pow(c2, 2) +
          16 * a1 * a2 * b1 * c4 * pow(c5, 2) -
          8 * a1 * a2 * b2 * c3 * pow(c5, 2) +
          8 * a1 * a2 * b2 * pow(c3, 2) * c5 -
          16 * a1 * a2 * b4 * c1 * pow(c5, 2) -
          8 * a1 * a2 * b5 * c2 * pow(c3, 2) +
          16 * a1 * a4 * b1 * c2 * pow(c5, 2) -
          16 * a1 * a4 * b2 * c1 * pow(c5, 2) -
          8 * a1 * a5 * b0 * c3 * pow(c4, 2) +
          24 * a1 * a5 * b1 * c1 * pow(c5, 2) -
          8 * a1 * a5 * b2 * c2 * pow(c3, 2) +
          8 * a1 * a5 * b5 * c0 * pow(c2, 2) +
          8 * a2 * a3 * b2 * c1 * pow(c5, 2) -
          16 * a2 * a4 * b1 * c1 * pow(c5, 2) +
          16 * a2 * a4 * b2 * c2 * pow(c3, 2) -
          8 * a2 * a5 * b1 * c2 * pow(c3, 2) +
          8 * a2 * a5 * b2 * c1 * pow(c3, 2) +
          8 * a3 * a5 * b0 * c1 * pow(c4, 2) -
          2 * a0 * a1 * b5 * c3 * pow(c5, 2) +
          2 * a0 * a1 * b5 * pow(c3, 2) * c5 -
          4 * a0 * a2 * b3 * c4 * pow(c5, 2) +
          4 * a0 * a2 * b4 * c3 * pow(c5, 2) -
          4 * a0 * a2 * b4 * pow(c3, 2) * c5 +
          4 * a0 * a2 * b5 * pow(c3, 2) * c4 -
          4 * a0 * a3 * b2 * c4 * pow(c5, 2) -
          4 * a0 * a3 * b4 * c2 * pow(c5, 2) -
          2 * a0 * a3 * b5 * c1 * pow(c5, 2) +
          4 * a0 * a4 * b2 * c3 * pow(c5, 2) -
          4 * a0 * a4 * b2 * pow(c3, 2) * c5 -
          4 * a0 * a4 * b3 * c2 * pow(c5, 2) +
          8 * a0 * a4 * b4 * c1 * pow(c5, 2) +
          4 * a0 * a4 * b5 * c2 * pow(c3, 2) -
          2 * a0 * a5 * b1 * c3 * pow(c5, 2) +
          2 * a0 * a5 * b1 * pow(c3, 2) * c5 +
          4 * a0 * a5 * b2 * pow(c3, 2) * c4 -
          2 * a0 * a5 * b3 * c1 * pow(c5, 2) +
          4 * a0 * a5 * b4 * c2 * pow(c3, 2) -
          6 * a0 * a5 * b5 * c1 * pow(c3, 2) -
          8 * a1 * a3 * b3 * pow(c2, 2) * c5 -
          2 * a1 * a3 * b5 * c0 * pow(c5, 2) +
          4 * a1 * a3 * b5 * pow(c0, 2) * c5 +
          8 * a1 * a3 * b5 * pow(c2, 2) * c3 -
          8 * a1 * a4 * b4 * c0 * pow(c5, 2) +
          16 * a1 * a4 * b4 * pow(c0, 2) * c5 +
          32 * a1 * a4 * b4 * pow(c2, 2) * c3 -
          8 * a1 * a4 * b5 * pow(c0, 2) * c4 -
          2 * a1 * a5 * b0 * c3 * pow(c5, 2) +
          2 * a1 * a5 * b0 * pow(c3, 2) * c5 -
          2 * a1 * a5 * b3 * c0 * pow(c5, 2) +
          4 * a1 * a5 * b3 * pow(c0, 2) * c5 +
          8 * a1 * a5 * b3 * pow(c2, 2) * c3 -
          8 * a1 * a5 * b4 * pow(c0, 2) * c4 +
          2 * a1 * a5 * b5 * c0 * pow(c3, 2) -
          4 * a1 * a5 * b5 * pow(c0, 2) * c3 -
          4 * a2 * a3 * b0 * c4 * pow(c5, 2) -
          16 * a2 * a3 * b3 * pow(c2, 2) * c4 +
          4 * a2 * a3 * b4 * c0 * pow(c5, 2) -
          8 * a2 * a3 * b4 * pow(c0, 2) * c5 -
          16 * a2 * a3 * b4 * pow(c2, 2) * c3 +
          4 * a2 * a4 * b0 * c3 * pow(c5, 2) -
          4 * a2 * a4 * b0 * pow(c3, 2) * c5 +
          4 * a2 * a4 * b3 * c0 * pow(c5, 2) -
          8 * a2 * a4 * b3 * pow(c0, 2) * c5 -
          16 * a2 * a4 * b3 * pow(c2, 2) * c3 -
          4 * a2 * a4 * b5 * c0 * pow(c3, 2) +
          8 * a2 * a4 * b5 * pow(c0, 2) * c3 +
          4 * a2 * a5 * b0 * pow(c3, 2) * c4 -
          4 * a2 * a5 * b4 * c0 * pow(c3, 2) +
          8 * a2 * a5 * b4 * pow(c0, 2) * c3 -
          4 * a3 * a4 * b0 * c2 * pow(c5, 2) +
          4 * a3 * a4 * b2 * c0 * pow(c5, 2) -
          8 * a3 * a4 * b2 * pow(c0, 2) * c5 -
          16 * a3 * a4 * b2 * pow(c2, 2) * c3 -
          32 * a3 * a4 * b4 * c1 * pow(c2, 2) -
          2 * a3 * a5 * b0 * c1 * pow(c5, 2) -
          2 * a3 * a5 * b1 * c0 * pow(c5, 2) +
          4 * a3 * a5 * b1 * pow(c0, 2) * c5 +
          8 * a3 * a5 * b1 * pow(c2, 2) * c3 -
          8 * a3 * a5 * b3 * c1 * pow(c2, 2) -
          4 * a3 * a5 * b5 * pow(c0, 2) * c1 +
          4 * a4 * a5 * b0 * c2 * pow(c3, 2) -
          8 * a4 * a5 * b1 * pow(c0, 2) * c4 -
          4 * a4 * a5 * b2 * c0 * pow(c3, 2) +
          8 * a4 * a5 * b2 * pow(c0, 2) * c3 +
          8 * a0 * a5 * b5 * c1 * pow(c4, 2) +
          8 * a1 * a3 * b5 * pow(c1, 2) * c5 +
          8 * a1 * a5 * b3 * pow(c1, 2) * c5 -
          8 * a1 * a5 * b5 * c0 * pow(c4, 2) +
          8 * a1 * a5 * b5 * pow(c1, 2) * c3 -
          16 * a2 * a3 * b4 * pow(c1, 2) * c5 -
          16 * a2 * a4 * b3 * pow(c1, 2) * c5 -
          16 * a3 * a4 * b2 * pow(c1, 2) * c5 +
          16 * a3 * a4 * b5 * pow(c1, 2) * c2 +
          8 * a3 * a5 * b1 * pow(c1, 2) * c5 +
          16 * a3 * a5 * b4 * pow(c1, 2) * c2 +
          16 * a4 * a5 * b3 * pow(c1, 2) * c2 -
          2 * a1 * a5 * b5 * pow(c0, 2) * c5 -
          8 * a1 * a5 * b5 * pow(c2, 2) * c3 +
          4 * a2 * a4 * b5 * pow(c0, 2) * c5 +
          4 * a2 * a5 * b4 * pow(c0, 2) * c5 -
          4 * a2 * a5 * b5 * pow(c0, 2) * c4 +
          8 * a3 * a5 * b5 * c1 * pow(c2, 2) +
          4 * a4 * a5 * b2 * pow(c0, 2) * c5 -
          4 * a4 * a5 * b5 * pow(c0, 2) * c2 +
          2 * a1 * a3 * b5 * c3 * pow(c5, 2) +
          8 * a1 * a4 * b4 * c3 * pow(c5, 2) +
          2 * a1 * a5 * b3 * c3 * pow(c5, 2) -
          24 * a1 * a5 * b5 * pow(c1, 2) * c5 +
          4 * a2 * a3 * b3 * c4 * pow(c5, 2) -
          4 * a2 * a3 * b4 * c3 * pow(c5, 2) -
          4 * a2 * a4 * b3 * c3 * pow(c5, 2) +
          16 * a2 * a4 * b5 * pow(c1, 2) * c5 +
          16 * a2 * a5 * b4 * pow(c1, 2) * c5 -
          16 * a2 * a5 * b5 * pow(c1, 2) * c4 -
          4 * a3 * a4 * b2 * c3 * pow(c5, 2) +
          4 * a3 * a4 * b3 * c2 * pow(c5, 2) -
          8 * a3 * a4 * b4 * c1 * pow(c5, 2) +
          2 * a3 * a5 * b1 * c3 * pow(c5, 2) +
          2 * a3 * a5 * b3 * c1 * pow(c5, 2) +
          16 * a4 * a5 * b2 * pow(c1, 2) * c5 -
          16 * a4 * a5 * b5 * pow(c1, 2) * c2 +
          8 * a1 * a5 * b5 * c3 * pow(c4, 2) -
          8 * a3 * a5 * b5 * c1 * pow(c4, 2) -
          2 * a1 * a5 * b5 * pow(c3, 2) * c5 +
          4 * a2 * a4 * b5 * pow(c3, 2) * c5 +
          4 * a2 * a5 * b4 * pow(c3, 2) * c5 -
          4 * a2 * a5 * b5 * pow(c3, 2) * c4 +
          4 * a4 * a5 * b2 * pow(c3, 2) * c5 -
          4 * a4 * a5 * b5 * c2 * pow(c3, 2) -
          8 * pow(a1, 2) * b0 * c0 * c1 * c3 +
          2 * pow(a0, 2) * b3 * c0 * c1 * c3 +
          8 * pow(a1, 2) * b0 * c0 * c1 * c5 +
          16 * pow(a2, 2) * b0 * c0 * c2 * c4 -
          8 * pow(a3, 2) * b2 * c0 * c1 * c2 +
          6 * pow(a0, 2) * b0 * c1 * c3 * c5 -
          12 * pow(a0, 2) * b0 * c2 * c3 * c4 -
          2 * pow(a0, 2) * b1 * c0 * c3 * c5 +
          4 * pow(a0, 2) * b2 * c0 * c3 * c4 -
          2 * pow(a0, 2) * b3 * c0 * c1 * c5 +
          4 * pow(a0, 2) * b3 * c0 * c2 * c4 -
          8 * pow(a0, 2) * b4 * c0 * c1 * c4 +
          4 * pow(a0, 2) * b4 * c0 * c2 * c3 -
          2 * pow(a0, 2) * b5 * c0 * c1 * c3 -
          32 * pow(a1, 2) * b2 * c1 * c2 * c3 +
          8 * pow(a2, 2) * b3 * c0 * c1 * c3 +
          2 * pow(a3, 2) * b0 * c0 * c1 * c5 -
          4 * pow(a3, 2) * b0 * c0 * c2 * c4 +
          4 * pow(a5, 2) * b0 * c0 * c1 * c3 +
          8 * pow(a1, 2) * b0 * c1 * c3 * c5 -
          16 * pow(a1, 2) * b0 * c2 * c3 * c4 -
          24 * pow(a1, 2) * b1 * c0 * c3 * c5 +
          16 * pow(a1, 2) * b2 * c0 * c3 * c4 +
          8 * pow(a1, 2) * b3 * c0 * c1 * c5 +
          8 * pow(a1, 2) * b5 * c0 * c1 * c3 +
          64 * pow(a2, 2) * b1 * c1 * c2 * c4 -
          8 * pow(a4, 2) * b0 * c0 * c1 * c5 +
          12 * pow(a0, 2) * b0 * c2 * c4 * c5 -
          4 * pow(a0, 2) * b2 * c0 * c4 * c5 -
          4 * pow(a0, 2) * b4 * c0 * c2 * c5 +
          2 * pow(a0, 2) * b5 * c0 * c1 * c5 -
          4 * pow(a0, 2) * b5 * c0 * c2 * c4 +
          32 * pow(a1, 2) * b2 * c1 * c2 * c5 +
          8 * pow(a2, 2) * b0 * c1 * c3 * c5 -
          16 * pow(a2, 2) * b0 * c2 * c3 * c4 +
          48 * pow(a2, 2) * b2 * c0 * c3 * c4 -
          16 * pow(a2, 2) * b3 * c0 * c2 * c4 -
          16 * pow(a2, 2) * b4 * c0 * c2 * c3 -
          8 * pow(a2, 2) * b5 * c0 * c1 * c3 -
          2 * pow(a5, 2) * b0 * c0 * c1 * c5 +
          4 * pow(a5, 2) * b0 * c0 * c2 * c4 -
          2 * pow(a0, 2) * b3 * c1 * c3 * c5 +
          4 * pow(a0, 2) * b3 * c2 * c3 * c4 +
          16 * pow(a1, 2) * b0 * c2 * c4 * c5 -
          16 * pow(a1, 2) * b4 * c0 * c2 * c5 -
          16 * pow(a1, 2) * b5 * c0 * c1 * c5 +
          2 * pow(a5, 2) * b3 * c0 * c1 * c3 +
          8 * pow(a3, 2) * b2 * c1 * c2 * c5 -
          8 * pow(a4, 2) * b1 * c0 * c3 * c5 +
          8 * pow(a4, 2) * b3 * c0 * c1 * c5 +
          16 * pow(a5, 2) * b1 * c1 * c2 * c4 -
          8 * pow(a0, 2) * b3 * c2 * c4 * c5 +
          8 * pow(a0, 2) * b4 * c1 * c4 * c5 -
          4 * pow(a0, 2) * b5 * c1 * c3 * c5 +
          8 * pow(a0, 2) * b5 * c2 * c3 * c4 -
          8 * pow(a2, 2) * b3 * c1 * c3 * c5 +
          16 * pow(a2, 2) * b3 * c2 * c3 * c4 +
          4 * pow(a3, 2) * b0 * c2 * c4 * c5 -
          4 * pow(a3, 2) * b2 * c0 * c4 * c5 -
          4 * pow(a3, 2) * b4 * c0 * c2 * c5 -
          2 * pow(a3, 2) * b5 * c0 * c1 * c5 +
          4 * pow(a3, 2) * b5 * c0 * c2 * c4 +
          2 * pow(a5, 2) * b0 * c1 * c3 * c5 -
          4 * pow(a5, 2) * b0 * c2 * c3 * c4 +
          2 * pow(a5, 2) * b1 * c0 * c3 * c5 +
          4 * pow(a5, 2) * b2 * c0 * c3 * c4 +
          2 * pow(a5, 2) * b3 * c0 * c1 * c5 -
          4 * pow(a5, 2) * b3 * c0 * c2 * c4 +
          4 * pow(a5, 2) * b4 * c0 * c2 * c3 -
          6 * pow(a5, 2) * b5 * c0 * c1 * c3 -
          16 * pow(a1, 2) * b2 * c3 * c4 * c5 -
          8 * pow(a1, 2) * b5 * c1 * c3 * c5 +
          16 * pow(a1, 2) * b5 * c2 * c3 * c4 -
          4 * pow(a0, 2) * b5 * c2 * c4 * c5 -
          16 * pow(a1, 2) * b5 * c2 * c4 * c5 -
          2 * pow(a5, 2) * b3 * c1 * c3 * c5 +
          4 * pow(a5, 2) * b3 * c2 * c3 * c4 -
          4 * pow(a3, 2) * b5 * c2 * c4 * c5 -
          16 * a0 * a1 * b1 * c0 * c1 * c3 + 4 * a0 * a3 * b0 * c0 * c1 * c3 +
          16 * a0 * a1 * b1 * c0 * c1 * c5 + 16 * a0 * a1 * b2 * c0 * c1 * c4 -
          8 * a0 * a1 * b2 * c0 * c2 * c3 - 16 * a0 * a1 * b4 * c0 * c1 * c2 +
          16 * a0 * a2 * b1 * c0 * c1 * c4 - 8 * a0 * a2 * b1 * c0 * c2 * c3 +
          8 * a0 * a2 * b3 * c0 * c1 * c2 + 8 * a0 * a3 * b2 * c0 * c1 * c2 -
          16 * a0 * a4 * b1 * c0 * c1 * c2 + 16 * a1 * a2 * b0 * c0 * c1 * c4 -
          8 * a1 * a2 * b0 * c0 * c2 * c3 - 16 * a1 * a4 * b0 * c0 * c1 * c2 +
          8 * a2 * a3 * b0 * c0 * c1 * c2 - 4 * a0 * a1 * b0 * c0 * c3 * c5 +
          8 * a0 * a2 * b0 * c0 * c3 * c4 - 4 * a0 * a3 * b0 * c0 * c1 * c5 +
          8 * a0 * a3 * b0 * c0 * c2 * c4 - 16 * a0 * a4 * b0 * c0 * c1 * c4 +
          8 * a0 * a4 * b0 * c0 * c2 * c3 - 4 * a0 * a5 * b0 * c0 * c1 * c3 +
          8 * a0 * a1 * b2 * c0 * c2 * c5 + 8 * a0 * a2 * b1 * c0 * c2 * c5 +
          32 * a0 * a2 * b2 * c0 * c2 * c4 - 8 * a0 * a2 * b5 * c0 * c1 * c2 -
          8 * a0 * a5 * b2 * c0 * c1 * c2 + 8 * a1 * a2 * b0 * c0 * c2 * c5 -
          64 * a1 * a2 * b1 * c1 * c2 * c3 - 8 * a2 * a5 * b0 * c0 * c1 * c2 +
          16 * a0 * a1 * b1 * c1 * c3 * c5 - 32 * a0 * a1 * b1 * c2 * c3 * c4 +
          16 * a0 * a1 * b3 * c1 * c2 * c4 + 16 * a0 * a1 * b4 * c1 * c2 * c3 -
          8 * a0 * a2 * b0 * c0 * c4 * c5 + 16 * a0 * a3 * b1 * c1 * c2 * c4 -
          8 * a0 * a4 * b0 * c0 * c2 * c5 + 16 * a0 * a4 * b1 * c1 * c2 * c3 +
          4 * a0 * a5 * b0 * c0 * c1 * c5 - 8 * a0 * a5 * b0 * c0 * c2 * c4 +
          32 * a1 * a2 * b1 * c0 * c3 * c4 - 16 * a1 * a2 * b3 * c0 * c1 * c4 -
          16 * a1 * a2 * b4 * c0 * c1 * c3 + 16 * a1 * a3 * b0 * c1 * c2 * c4 +
          16 * a1 * a3 * b1 * c0 * c1 * c5 - 16 * a1 * a3 * b2 * c0 * c1 * c4 +
          16 * a1 * a4 * b0 * c1 * c2 * c3 - 16 * a1 * a4 * b2 * c0 * c1 * c3 +
          16 * a1 * a5 * b1 * c0 * c1 * c3 - 16 * a2 * a3 * b1 * c0 * c1 * c4 +
          16 * a2 * a3 * b2 * c0 * c1 * c3 - 16 * a2 * a3 * b3 * c0 * c1 * c2 -
          16 * a2 * a4 * b1 * c0 * c1 * c3 + 4 * a0 * a1 * b3 * c0 * c3 * c5 -
          8 * a0 * a1 * b4 * c0 * c3 * c4 - 4 * a0 * a3 * b0 * c1 * c3 * c5 +
          8 * a0 * a3 * b0 * c2 * c3 * c4 + 4 * a0 * a3 * b1 * c0 * c3 * c5 +
          4 * a0 * a3 * b3 * c0 * c1 * c5 - 8 * a0 * a3 * b3 * c0 * c2 * c4 +
          8 * a0 * a3 * b4 * c0 * c1 * c4 - 4 * a0 * a3 * b5 * c0 * c1 * c3 -
          8 * a0 * a4 * b1 * c0 * c3 * c4 + 8 * a0 * a4 * b3 * c0 * c1 * c4 -
          4 * a0 * a5 * b3 * c0 * c1 * c3 + 64 * a1 * a2 * b1 * c1 * c2 * c5 +
          128 * a1 * a2 * b2 * c1 * c2 * c4 + 4 * a1 * a3 * b0 * c0 * c3 * c5 -
          8 * a1 * a4 * b0 * c0 * c3 * c4 + 8 * a3 * a4 * b0 * c0 * c1 * c4 -
          4 * a3 * a5 * b0 * c0 * c1 * c3 + 32 * a0 * a1 * b1 * c2 * c4 * c5 -
          16 * a0 * a1 * b2 * c1 * c4 * c5 - 8 * a0 * a1 * b2 * c2 * c3 * c5 -
          16 * a0 * a1 * b5 * c1 * c2 * c4 - 16 * a0 * a2 * b1 * c1 * c4 * c5 -
          8 * a0 * a2 * b1 * c2 * c3 * c5 + 16 * a0 * a2 * b2 * c1 * c3 * c5 -
          32 * a0 * a2 * b2 * c2 * c3 * c4 - 8 * a0 * a2 * b3 * c1 * c2 * c5 -
          32 * a0 * a2 * b4 * c1 * c2 * c4 - 8 * a0 * a3 * b2 * c1 * c2 * c5 -
          32 * a0 * a4 * b2 * c1 * c2 * c4 - 16 * a0 * a5 * b1 * c1 * c2 * c4 -
          16 * a1 * a2 * b0 * c1 * c4 * c5 - 8 * a1 * a2 * b0 * c2 * c3 * c5 +
          16 * a1 * a2 * b4 * c0 * c1 * c5 + 32 * a1 * a2 * b4 * c0 * c2 * c4 +
          8 * a1 * a2 * b5 * c0 * c2 * c3 - 32 * a1 * a4 * b1 * c0 * c2 * c5 +
          16 * a1 * a4 * b2 * c0 * c1 * c5 + 32 * a1 * a4 * b2 * c0 * c2 * c4 +
          16 * a1 * a4 * b5 * c0 * c1 * c2 - 16 * a1 * a5 * b0 * c1 * c2 * c4 -
          32 * a1 * a5 * b1 * c0 * c1 * c5 + 8 * a1 * a5 * b2 * c0 * c2 * c3 +
          16 * a1 * a5 * b4 * c0 * c1 * c2 - 8 * a2 * a3 * b0 * c1 * c2 * c5 -
          32 * a2 * a3 * b2 * c0 * c2 * c4 + 8 * a2 * a3 * b5 * c0 * c1 * c2 -
          32 * a2 * a4 * b0 * c1 * c2 * c4 + 16 * a2 * a4 * b1 * c0 * c1 * c5 +
          32 * a2 * a4 * b1 * c0 * c2 * c4 - 32 * a2 * a4 * b2 * c0 * c2 * c3 +
          8 * a2 * a5 * b1 * c0 * c2 * c3 - 16 * a2 * a5 * b2 * c0 * c1 * c3 +
          8 * a2 * a5 * b3 * c0 * c1 * c2 + 8 * a3 * a5 * b2 * c0 * c1 * c2 +
          16 * a4 * a5 * b1 * c0 * c1 * c2 - 8 * a0 * a1 * b4 * c0 * c4 * c5 +
          8 * a0 * a2 * b3 * c0 * c4 * c5 - 8 * a0 * a2 * b5 * c0 * c3 * c4 -
          16 * a0 * a3 * b0 * c2 * c4 * c5 + 8 * a0 * a3 * b2 * c0 * c4 * c5 +
          8 * a0 * a3 * b4 * c0 * c2 * c5 + 16 * a0 * a4 * b0 * c1 * c4 * c5 -
          8 * a0 * a4 * b1 * c0 * c4 * c5 + 8 * a0 * a4 * b3 * c0 * c2 * c5 -
          16 * a0 * a4 * b4 * c0 * c1 * c5 + 8 * a0 * a4 * b5 * c0 * c1 * c4 -
          8 * a0 * a4 * b5 * c0 * c2 * c3 - 8 * a0 * a5 * b0 * c1 * c3 * c5 +
          16 * a0 * a5 * b0 * c2 * c3 * c4 - 8 * a0 * a5 * b2 * c0 * c3 * c4 +
          8 * a0 * a5 * b4 * c0 * c1 * c4 - 8 * a0 * a5 * b4 * c0 * c2 * c3 +
          8 * a0 * a5 * b5 * c0 * c1 * c3 - 8 * a1 * a4 * b0 * c0 * c4 * c5 +
          8 * a2 * a3 * b0 * c0 * c4 * c5 - 8 * a2 * a5 * b0 * c0 * c3 * c4 +
          8 * a3 * a4 * b0 * c0 * c2 * c5 + 8 * a4 * a5 * b0 * c0 * c1 * c4 -
          8 * a4 * a5 * b0 * c0 * c2 * c3 + 8 * a0 * a2 * b5 * c1 * c2 * c5 +
          8 * a0 * a5 * b2 * c1 * c2 * c5 - 8 * a1 * a2 * b5 * c0 * c2 * c5 -
          8 * a1 * a5 * b2 * c0 * c2 * c5 + 8 * a2 * a5 * b0 * c1 * c2 * c5 -
          8 * a2 * a5 * b1 * c0 * c2 * c5 - 8 * a0 * a5 * b0 * c2 * c4 * c5 -
          4 * a0 * a5 * b5 * c0 * c1 * c5 + 8 * a0 * a5 * b5 * c0 * c2 * c4 -
          32 * a1 * a2 * b1 * c3 * c4 * c5 + 16 * a1 * a2 * b3 * c1 * c4 * c5 +
          16 * a1 * a2 * b4 * c1 * c3 * c5 - 32 * a1 * a2 * b4 * c2 * c3 * c4 +
          16 * a1 * a3 * b2 * c1 * c4 * c5 - 16 * a1 * a3 * b5 * c1 * c2 * c4 +
          16 * a1 * a4 * b2 * c1 * c3 * c5 - 32 * a1 * a4 * b2 * c2 * c3 * c4 -
          16 * a1 * a4 * b5 * c1 * c2 * c3 - 16 * a1 * a5 * b1 * c1 * c3 * c5 +
          32 * a1 * a5 * b1 * c2 * c3 * c4 - 16 * a1 * a5 * b3 * c1 * c2 * c4 -
          16 * a1 * a5 * b4 * c1 * c2 * c3 + 16 * a2 * a3 * b1 * c1 * c4 * c5 -
          16 * a2 * a3 * b2 * c1 * c3 * c5 + 32 * a2 * a3 * b2 * c2 * c3 * c4 +
          16 * a2 * a3 * b3 * c1 * c2 * c5 + 32 * a2 * a3 * b4 * c1 * c2 * c4 +
          16 * a2 * a4 * b1 * c1 * c3 * c5 - 32 * a2 * a4 * b1 * c2 * c3 * c4 +
          32 * a2 * a4 * b3 * c1 * c2 * c4 + 32 * a3 * a4 * b2 * c1 * c2 * c4 -
          16 * a3 * a5 * b1 * c1 * c2 * c4 - 16 * a4 * a5 * b1 * c1 * c2 * c3 +
          8 * a0 * a1 * b4 * c3 * c4 * c5 + 8 * a0 * a3 * b3 * c2 * c4 * c5 -
          8 * a0 * a3 * b4 * c1 * c4 * c5 + 4 * a0 * a3 * b5 * c1 * c3 * c5 -
          8 * a0 * a3 * b5 * c2 * c3 * c4 + 8 * a0 * a4 * b1 * c3 * c4 * c5 -
          8 * a0 * a4 * b3 * c1 * c4 * c5 + 4 * a0 * a5 * b3 * c1 * c3 * c5 -
          8 * a0 * a5 * b3 * c2 * c3 * c4 - 4 * a1 * a3 * b5 * c0 * c3 * c5 +
          8 * a1 * a4 * b0 * c3 * c4 * c5 - 16 * a1 * a4 * b4 * c0 * c3 * c5 +
          8 * a1 * a4 * b5 * c0 * c3 * c4 - 4 * a1 * a5 * b3 * c0 * c3 * c5 +
          8 * a1 * a5 * b4 * c0 * c3 * c4 - 8 * a2 * a3 * b3 * c0 * c4 * c5 +
          8 * a2 * a3 * b4 * c0 * c3 * c5 + 8 * a2 * a4 * b3 * c0 * c3 * c5 -
          8 * a3 * a4 * b0 * c1 * c4 * c5 + 8 * a3 * a4 * b2 * c0 * c3 * c5 -
          8 * a3 * a4 * b3 * c0 * c2 * c5 + 16 * a3 * a4 * b4 * c0 * c1 * c5 -
          8 * a3 * a4 * b5 * c0 * c1 * c4 + 4 * a3 * a5 * b0 * c1 * c3 * c5 -
          8 * a3 * a5 * b0 * c2 * c3 * c4 - 4 * a3 * a5 * b1 * c0 * c3 * c5 -
          4 * a3 * a5 * b3 * c0 * c1 * c5 + 8 * a3 * a5 * b3 * c0 * c2 * c4 -
          8 * a3 * a5 * b4 * c0 * c1 * c4 + 4 * a3 * a5 * b5 * c0 * c1 * c3 +
          8 * a4 * a5 * b1 * c0 * c3 * c4 - 8 * a4 * a5 * b3 * c0 * c1 * c4 +
          8 * a1 * a2 * b5 * c2 * c3 * c5 - 32 * a1 * a5 * b1 * c2 * c4 * c5 +
          8 * a1 * a5 * b2 * c2 * c3 * c5 + 32 * a1 * a5 * b5 * c1 * c2 * c4 -
          8 * a2 * a3 * b5 * c1 * c2 * c5 + 8 * a2 * a5 * b1 * c2 * c3 * c5 -
          8 * a2 * a5 * b3 * c1 * c2 * c5 - 8 * a3 * a5 * b2 * c1 * c2 * c5 +
          8 * a0 * a3 * b5 * c2 * c4 * c5 - 8 * a0 * a4 * b5 * c1 * c4 * c5 +
          8 * a0 * a5 * b3 * c2 * c4 * c5 - 8 * a0 * a5 * b4 * c1 * c4 * c5 +
          4 * a0 * a5 * b5 * c1 * c3 * c5 - 8 * a0 * a5 * b5 * c2 * c3 * c4 +
          8 * a1 * a4 * b5 * c0 * c4 * c5 + 8 * a1 * a5 * b4 * c0 * c4 * c5 +
          4 * a1 * a5 * b5 * c0 * c3 * c5 - 8 * a2 * a4 * b5 * c0 * c3 * c5 -
          8 * a2 * a5 * b4 * c0 * c3 * c5 + 8 * a2 * a5 * b5 * c0 * c3 * c4 +
          8 * a3 * a5 * b0 * c2 * c4 * c5 + 4 * a3 * a5 * b5 * c0 * c1 * c5 -
          8 * a3 * a5 * b5 * c0 * c2 * c4 - 8 * a4 * a5 * b0 * c1 * c4 * c5 +
          8 * a4 * a5 * b1 * c0 * c4 * c5 - 8 * a4 * a5 * b2 * c0 * c3 * c5 +
          8 * a4 * a5 * b5 * c0 * c2 * c3 - 8 * a1 * a4 * b5 * c3 * c4 * c5 -
          8 * a1 * a5 * b4 * c3 * c4 * c5 + 8 * a3 * a4 * b5 * c1 * c4 * c5 -
          8 * a3 * a5 * b3 * c2 * c4 * c5 + 8 * a3 * a5 * b4 * c1 * c4 * c5 -
          4 * a3 * a5 * b5 * c1 * c3 * c5 + 8 * a3 * a5 * b5 * c2 * c3 * c4 -
          8 * a4 * a5 * b1 * c3 * c4 * c5 + 8 * a4 * a5 * b3 * c1 * c4 * c5,
      4 * pow(a5, 3) * pow(d1, 3) - 4 * pow(a1, 3) * pow(d5, 3) +
          a1 * pow(a3, 2) * pow(d0, 3) - 4 * pow(a0, 2) * a3 * pow(d1, 3) -
          4 * a1 * pow(a4, 2) * pow(d0, 3) - 8 * a0 * pow(a5, 2) * pow(d1, 3) +
          a1 * pow(a5, 2) * pow(d0, 3) - pow(a0, 2) * a1 * pow(d5, 3) +
          8 * pow(a0, 2) * a4 * pow(d2, 3) + 4 * pow(a0, 2) * a5 * pow(d1, 3) -
          16 * pow(a2, 2) * a3 * pow(d1, 3) +
          32 * pow(a1, 2) * a4 * pow(d2, 3) +
          16 * pow(a2, 2) * a5 * pow(d1, 3) - a1 * pow(a3, 2) * pow(d5, 3) -
          4 * a3 * pow(a5, 2) * pow(d1, 3) + 8 * pow(a3, 2) * a4 * pow(d2, 3) -
          pow(a0, 3) * d1 * pow(d3, 2) + 4 * pow(a1, 3) * pow(d0, 2) * d3 +
          4 * pow(a0, 3) * d1 * pow(d4, 2) - pow(a0, 3) * d1 * pow(d5, 2) +
          8 * pow(a1, 3) * d0 * pow(d5, 2) - 4 * pow(a1, 3) * pow(d0, 2) * d5 +
          16 * pow(a1, 3) * pow(d2, 2) * d3 - 8 * pow(a2, 3) * pow(d0, 2) * d4 +
          pow(a5, 3) * pow(d0, 2) * d1 - 32 * pow(a2, 3) * pow(d1, 2) * d4 -
          16 * pow(a1, 3) * pow(d2, 2) * d5 + 4 * pow(a1, 3) * d3 * pow(d5, 2) -
          8 * pow(a2, 3) * pow(d3, 2) * d4 + pow(a5, 3) * d1 * pow(d3, 2) +
          2 * a0 * a1 * a3 * pow(d5, 3) - 16 * a0 * a3 * a4 * pow(d2, 3) +
          8 * a0 * a3 * a5 * pow(d1, 3) - 2 * a1 * a3 * a5 * pow(d0, 3) +
          4 * a2 * a3 * a4 * pow(d0, 3) - 4 * a2 * a4 * a5 * pow(d0, 3) +
          2 * pow(a0, 3) * d1 * d3 * d5 - 4 * pow(a0, 3) * d2 * d3 * d4 -
          8 * pow(a1, 3) * d0 * d3 * d5 + 16 * pow(a2, 3) * d0 * d3 * d4 -
          2 * pow(a5, 3) * d0 * d1 * d3 + 4 * pow(a0, 3) * d2 * d4 * d5 -
          a0 * pow(a3, 2) * pow(d0, 2) * d1 +
          pow(a0, 2) * a1 * d0 * pow(d3, 2) +
          4 * a0 * pow(a4, 2) * pow(d0, 2) * d1 -
          4 * pow(a0, 2) * a1 * d0 * pow(d4, 2) +
          4 * pow(a0, 2) * a1 * pow(d1, 2) * d3 -
          4 * pow(a1, 2) * a3 * pow(d0, 2) * d1 -
          4 * a0 * pow(a2, 2) * d1 * pow(d3, 2) +
          4 * a0 * pow(a3, 2) * d1 * pow(d2, 2) -
          a0 * pow(a5, 2) * pow(d0, 2) * d1 -
          4 * a1 * pow(a2, 2) * d0 * pow(d3, 2) +
          4 * a1 * pow(a2, 2) * pow(d0, 2) * d3 +
          4 * a1 * pow(a3, 2) * d0 * pow(d2, 2) +
          pow(a0, 2) * a1 * d0 * pow(d5, 2) +
          4 * pow(a0, 2) * a1 * pow(d2, 2) * d3 -
          4 * pow(a0, 2) * a3 * d1 * pow(d2, 2) -
          4 * pow(a2, 2) * a3 * pow(d0, 2) * d1 -
          8 * a0 * pow(a1, 2) * d1 * pow(d5, 2) +
          16 * a0 * pow(a2, 2) * d1 * pow(d4, 2) +
          16 * a0 * pow(a4, 2) * d1 * pow(d2, 2) -
          16 * a1 * pow(a2, 2) * d0 * pow(d4, 2) +
          16 * a1 * pow(a2, 2) * pow(d1, 2) * d3 -
          16 * a1 * pow(a4, 2) * d0 * pow(d2, 2) +
          8 * a1 * pow(a5, 2) * d0 * pow(d1, 2) -
          4 * pow(a0, 2) * a1 * pow(d1, 2) * d5 -
          8 * pow(a0, 2) * a2 * pow(d1, 2) * d4 +
          8 * pow(a0, 2) * a4 * pow(d1, 2) * d2 -
          8 * pow(a1, 2) * a2 * pow(d0, 2) * d4 -
          16 * pow(a1, 2) * a3 * d1 * pow(d2, 2) +
          8 * pow(a1, 2) * a4 * pow(d0, 2) * d2 +
          4 * pow(a1, 2) * a5 * pow(d0, 2) * d1 -
          4 * a0 * pow(a2, 2) * d1 * pow(d5, 2) -
          4 * a0 * pow(a5, 2) * d1 * pow(d2, 2) +
          4 * a1 * pow(a2, 2) * d0 * pow(d5, 2) -
          4 * a1 * pow(a2, 2) * pow(d0, 2) * d5 +
          4 * a1 * pow(a4, 2) * pow(d0, 2) * d3 +
          4 * a1 * pow(a5, 2) * d0 * pow(d2, 2) -
          4 * a3 * pow(a4, 2) * pow(d0, 2) * d1 +
          4 * pow(a0, 2) * a1 * d3 * pow(d4, 2) -
          4 * pow(a0, 2) * a1 * pow(d2, 2) * d5 -
          8 * pow(a0, 2) * a2 * pow(d2, 2) * d4 -
          4 * pow(a0, 2) * a3 * d1 * pow(d4, 2) +
          4 * pow(a0, 2) * a5 * d1 * pow(d2, 2) +
          8 * pow(a2, 2) * a4 * pow(d0, 2) * d2 +
          4 * pow(a2, 2) * a5 * pow(d0, 2) * d1 -
          a0 * pow(a3, 2) * d1 * pow(d5, 2) -
          3 * a0 * pow(a5, 2) * d1 * pow(d3, 2) -
          16 * a1 * pow(a2, 2) * pow(d1, 2) * d5 +
          3 * a1 * pow(a3, 2) * d0 * pow(d5, 2) -
          3 * a1 * pow(a3, 2) * pow(d0, 2) * d5 +
          a1 * pow(a5, 2) * d0 * pow(d3, 2) -
          2 * a1 * pow(a5, 2) * pow(d0, 2) * d3 +
          2 * a2 * pow(a3, 2) * pow(d0, 2) * d4 -
          2 * a3 * pow(a5, 2) * pow(d0, 2) * d1 +
          2 * pow(a0, 2) * a1 * d3 * pow(d5, 2) -
          pow(a0, 2) * a1 * pow(d3, 2) * d5 -
          2 * pow(a0, 2) * a2 * pow(d3, 2) * d4 +
          2 * pow(a0, 2) * a3 * d1 * pow(d5, 2) -
          2 * pow(a0, 2) * a4 * d2 * pow(d3, 2) +
          3 * pow(a0, 2) * a5 * d1 * pow(d3, 2) -
          32 * pow(a1, 2) * a2 * pow(d2, 2) * d4 +
          16 * pow(a1, 2) * a5 * d1 * pow(d2, 2) +
          32 * pow(a2, 2) * a4 * pow(d1, 2) * d2 +
          2 * pow(a3, 2) * a4 * pow(d0, 2) * d2 +
          pow(a3, 2) * a5 * pow(d0, 2) * d1 +
          4 * a0 * pow(a4, 2) * d1 * pow(d5, 2) +
          4 * a0 * pow(a5, 2) * d1 * pow(d4, 2) +
          16 * a1 * pow(a2, 2) * d3 * pow(d4, 2) -
          4 * a1 * pow(a4, 2) * d0 * pow(d5, 2) +
          8 * a1 * pow(a4, 2) * pow(d0, 2) * d5 +
          16 * a1 * pow(a4, 2) * pow(d2, 2) * d3 -
          4 * a1 * pow(a5, 2) * d0 * pow(d4, 2) +
          4 * a1 * pow(a5, 2) * pow(d1, 2) * d3 -
          16 * a3 * pow(a4, 2) * d1 * pow(d2, 2) -
          8 * pow(a0, 2) * a5 * d1 * pow(d4, 2) -
          4 * pow(a1, 2) * a3 * d1 * pow(d5, 2) -
          16 * pow(a2, 2) * a3 * d1 * pow(d4, 2) -
          4 * a1 * pow(a2, 2) * d3 * pow(d5, 2) +
          4 * a1 * pow(a2, 2) * pow(d3, 2) * d5 -
          4 * a1 * pow(a3, 2) * pow(d2, 2) * d5 -
          a1 * pow(a5, 2) * pow(d0, 2) * d5 -
          4 * a1 * pow(a5, 2) * pow(d2, 2) * d3 -
          8 * a2 * pow(a3, 2) * pow(d2, 2) * d4 -
          2 * a2 * pow(a5, 2) * pow(d0, 2) * d4 +
          4 * a3 * pow(a5, 2) * d1 * pow(d2, 2) -
          2 * a4 * pow(a5, 2) * pow(d0, 2) * d2 +
          2 * pow(a0, 2) * a2 * d4 * pow(d5, 2) +
          2 * pow(a0, 2) * a4 * d2 * pow(d5, 2) +
          pow(a0, 2) * a5 * d1 * pow(d5, 2) +
          4 * pow(a2, 2) * a3 * d1 * pow(d5, 2) +
          8 * pow(a2, 2) * a4 * d2 * pow(d3, 2) +
          4 * pow(a2, 2) * a5 * d1 * pow(d3, 2) -
          4 * pow(a3, 2) * a5 * d1 * pow(d2, 2) -
          12 * a1 * pow(a5, 2) * pow(d1, 2) * d5 -
          8 * a2 * pow(a5, 2) * pow(d1, 2) * d4 -
          8 * a4 * pow(a5, 2) * pow(d1, 2) * d2 +
          8 * pow(a1, 2) * a2 * d4 * pow(d5, 2) +
          8 * pow(a1, 2) * a4 * d2 * pow(d5, 2) +
          12 * pow(a1, 2) * a5 * d1 * pow(d5, 2) +
          4 * a1 * pow(a4, 2) * d3 * pow(d5, 2) +
          4 * a1 * pow(a5, 2) * d3 * pow(d4, 2) -
          4 * a3 * pow(a4, 2) * d1 * pow(d5, 2) -
          4 * a3 * pow(a5, 2) * d1 * pow(d4, 2) -
          a1 * pow(a5, 2) * pow(d3, 2) * d5 +
          2 * a2 * pow(a3, 2) * d4 * pow(d5, 2) -
          2 * a2 * pow(a5, 2) * pow(d3, 2) * d4 -
          2 * a4 * pow(a5, 2) * d2 * pow(d3, 2) +
          2 * pow(a3, 2) * a4 * d2 * pow(d5, 2) +
          pow(a3, 2) * a5 * d1 * pow(d5, 2) +
          8 * a0 * a1 * a3 * d0 * pow(d1, 2) -
          2 * a0 * a1 * a3 * pow(d0, 2) * d3 -
          8 * a0 * a1 * a5 * d0 * pow(d1, 2) +
          8 * a0 * a1 * a2 * d2 * pow(d3, 2) -
          16 * a0 * a2 * a4 * d0 * pow(d2, 2) -
          4 * a0 * a1 * a3 * d0 * pow(d5, 2) +
          2 * a0 * a1 * a3 * pow(d0, 2) * d5 -
          8 * a0 * a1 * a3 * pow(d2, 2) * d3 +
          8 * a0 * a1 * a4 * pow(d0, 2) * d4 -
          2 * a0 * a1 * a5 * d0 * pow(d3, 2) +
          2 * a0 * a1 * a5 * pow(d0, 2) * d3 -
          4 * a0 * a2 * a3 * pow(d0, 2) * d4 +
          4 * a0 * a2 * a4 * d0 * pow(d3, 2) -
          4 * a0 * a2 * a4 * pow(d0, 2) * d3 -
          4 * a0 * a3 * a4 * pow(d0, 2) * d2 +
          2 * a0 * a3 * a5 * pow(d0, 2) * d1 +
          32 * a1 * a2 * a3 * pow(d1, 2) * d2 -
          8 * a0 * a1 * a3 * pow(d1, 2) * d5 +
          8 * a0 * a1 * a5 * d0 * pow(d4, 2) -
          8 * a0 * a1 * a5 * pow(d1, 2) * d3 -
          16 * a0 * a3 * a4 * pow(d1, 2) * d2 -
          64 * a1 * a2 * a4 * d1 * pow(d2, 2) -
          8 * a1 * a3 * a5 * d0 * pow(d1, 2) +
          16 * a2 * a3 * a4 * d0 * pow(d1, 2) +
          8 * a0 * a1 * a3 * pow(d2, 2) * d5 +
          2 * a0 * a1 * a5 * d0 * pow(d5, 2) -
          2 * a0 * a1 * a5 * pow(d0, 2) * d5 +
          16 * a0 * a2 * a3 * pow(d2, 2) * d4 -
          4 * a0 * a2 * a4 * d0 * pow(d5, 2) +
          4 * a0 * a2 * a4 * pow(d0, 2) * d5 +
          16 * a0 * a2 * a4 * pow(d2, 2) * d3 +
          4 * a0 * a2 * a5 * pow(d0, 2) * d4 +
          4 * a0 * a4 * a5 * pow(d0, 2) * d2 -
          32 * a1 * a2 * a5 * pow(d1, 2) * d2 -
          8 * a1 * a3 * a5 * d0 * pow(d2, 2) +
          16 * a2 * a3 * a4 * d0 * pow(d2, 2) -
          2 * a0 * a1 * a3 * d3 * pow(d5, 2) +
          16 * a0 * a1 * a5 * pow(d1, 2) * d5 +
          16 * a0 * a2 * a5 * pow(d1, 2) * d4 +
          2 * a1 * a3 * a5 * pow(d0, 2) * d3 -
          4 * a2 * a3 * a4 * pow(d0, 2) * d3 -
          16 * a2 * a4 * a5 * d0 * pow(d1, 2) -
          8 * a0 * a1 * a5 * d3 * pow(d4, 2) +
          8 * a0 * a3 * a5 * d1 * pow(d4, 2) -
          16 * a1 * a2 * a4 * d1 * pow(d5, 2) -
          8 * a1 * a2 * a5 * d2 * pow(d3, 2) -
          2 * a0 * a1 * a5 * d3 * pow(d5, 2) +
          2 * a0 * a1 * a5 * pow(d3, 2) * d5 -
          4 * a0 * a2 * a3 * d4 * pow(d5, 2) +
          4 * a0 * a2 * a4 * d3 * pow(d5, 2) -
          4 * a0 * a2 * a4 * pow(d3, 2) * d5 +
          4 * a0 * a2 * a5 * pow(d3, 2) * d4 -
          4 * a0 * a3 * a4 * d2 * pow(d5, 2) -
          2 * a0 * a3 * a5 * d1 * pow(d5, 2) +
          4 * a0 * a4 * a5 * d2 * pow(d3, 2) -
          2 * a1 * a3 * a5 * d0 * pow(d5, 2) +
          4 * a1 * a3 * a5 * pow(d0, 2) * d5 +
          8 * a1 * a3 * a5 * pow(d2, 2) * d3 -
          8 * a1 * a4 * a5 * pow(d0, 2) * d4 +
          4 * a2 * a3 * a4 * d0 * pow(d5, 2) -
          8 * a2 * a3 * a4 * pow(d0, 2) * d5 -
          16 * a2 * a3 * a4 * pow(d2, 2) * d3 -
          4 * a2 * a4 * a5 * d0 * pow(d3, 2) +
          8 * a2 * a4 * a5 * pow(d0, 2) * d3 +
          8 * a1 * a3 * a5 * pow(d1, 2) * d5 -
          16 * a2 * a3 * a4 * pow(d1, 2) * d5 +
          16 * a3 * a4 * a5 * pow(d1, 2) * d2 +
          4 * a2 * a4 * a5 * pow(d0, 2) * d5 +
          2 * a1 * a3 * a5 * d3 * pow(d5, 2) -
          4 * a2 * a3 * a4 * d3 * pow(d5, 2) +
          16 * a2 * a4 * a5 * pow(d1, 2) * d5 +
          4 * a2 * a4 * a5 * pow(d3, 2) * d5 -
          8 * a0 * pow(a1, 2) * d0 * d1 * d3 +
          8 * a0 * pow(a1, 2) * d0 * d1 * d5 +
          2 * pow(a0, 2) * a3 * d0 * d1 * d3 +
          16 * a0 * pow(a2, 2) * d0 * d2 * d4 -
          8 * a2 * pow(a3, 2) * d0 * d1 * d2 +
          2 * a0 * pow(a3, 2) * d0 * d1 * d5 -
          4 * a0 * pow(a3, 2) * d0 * d2 * d4 +
          4 * a0 * pow(a5, 2) * d0 * d1 * d3 -
          2 * pow(a0, 2) * a1 * d0 * d3 * d5 +
          4 * pow(a0, 2) * a2 * d0 * d3 * d4 -
          2 * pow(a0, 2) * a3 * d0 * d1 * d5 +
          4 * pow(a0, 2) * a3 * d0 * d2 * d4 -
          8 * pow(a0, 2) * a4 * d0 * d1 * d4 +
          4 * pow(a0, 2) * a4 * d0 * d2 * d3 -
          2 * pow(a0, 2) * a5 * d0 * d1 * d3 -
          32 * pow(a1, 2) * a2 * d1 * d2 * d3 +
          8 * pow(a2, 2) * a3 * d0 * d1 * d3 +
          8 * a0 * pow(a1, 2) * d1 * d3 * d5 -
          16 * a0 * pow(a1, 2) * d2 * d3 * d4 -
          8 * a0 * pow(a4, 2) * d0 * d1 * d5 +
          64 * a1 * pow(a2, 2) * d1 * d2 * d4 +
          16 * pow(a1, 2) * a2 * d0 * d3 * d4 +
          8 * pow(a1, 2) * a3 * d0 * d1 * d5 +
          8 * pow(a1, 2) * a5 * d0 * d1 * d3 +
          8 * a0 * pow(a2, 2) * d1 * d3 * d5 -
          16 * a0 * pow(a2, 2) * d2 * d3 * d4 -
          2 * a0 * pow(a5, 2) * d0 * d1 * d5 +
          4 * a0 * pow(a5, 2) * d0 * d2 * d4 -
          4 * pow(a0, 2) * a2 * d0 * d4 * d5 -
          4 * pow(a0, 2) * a4 * d0 * d2 * d5 +
          2 * pow(a0, 2) * a5 * d0 * d1 * d5 -
          4 * pow(a0, 2) * a5 * d0 * d2 * d4 +
          32 * pow(a1, 2) * a2 * d1 * d2 * d5 -
          16 * pow(a2, 2) * a3 * d0 * d2 * d4 -
          16 * pow(a2, 2) * a4 * d0 * d2 * d3 -
          8 * pow(a2, 2) * a5 * d0 * d1 * d3 +
          16 * a0 * pow(a1, 2) * d2 * d4 * d5 +
          2 * a3 * pow(a5, 2) * d0 * d1 * d3 -
          2 * pow(a0, 2) * a3 * d1 * d3 * d5 +
          4 * pow(a0, 2) * a3 * d2 * d3 * d4 -
          16 * pow(a1, 2) * a4 * d0 * d2 * d5 -
          16 * pow(a1, 2) * a5 * d0 * d1 * d5 -
          8 * a1 * pow(a4, 2) * d0 * d3 * d5 +
          16 * a1 * pow(a5, 2) * d1 * d2 * d4 +
          8 * a2 * pow(a3, 2) * d1 * d2 * d5 +
          8 * a3 * pow(a4, 2) * d0 * d1 * d5 +
          4 * a0 * pow(a3, 2) * d2 * d4 * d5 +
          2 * a0 * pow(a5, 2) * d1 * d3 * d5 -
          4 * a0 * pow(a5, 2) * d2 * d3 * d4 +
          2 * a1 * pow(a5, 2) * d0 * d3 * d5 -
          4 * a2 * pow(a3, 2) * d0 * d4 * d5 +
          4 * a2 * pow(a5, 2) * d0 * d3 * d4 +
          2 * a3 * pow(a5, 2) * d0 * d1 * d5 -
          4 * a3 * pow(a5, 2) * d0 * d2 * d4 +
          4 * a4 * pow(a5, 2) * d0 * d2 * d3 -
          8 * pow(a0, 2) * a3 * d2 * d4 * d5 +
          8 * pow(a0, 2) * a4 * d1 * d4 * d5 -
          4 * pow(a0, 2) * a5 * d1 * d3 * d5 +
          8 * pow(a0, 2) * a5 * d2 * d3 * d4 -
          8 * pow(a2, 2) * a3 * d1 * d3 * d5 +
          16 * pow(a2, 2) * a3 * d2 * d3 * d4 -
          4 * pow(a3, 2) * a4 * d0 * d2 * d5 -
          2 * pow(a3, 2) * a5 * d0 * d1 * d5 +
          4 * pow(a3, 2) * a5 * d0 * d2 * d4 -
          16 * pow(a1, 2) * a2 * d3 * d4 * d5 -
          8 * pow(a1, 2) * a5 * d1 * d3 * d5 +
          16 * pow(a1, 2) * a5 * d2 * d3 * d4 -
          4 * pow(a0, 2) * a5 * d2 * d4 * d5 -
          2 * a3 * pow(a5, 2) * d1 * d3 * d5 +
          4 * a3 * pow(a5, 2) * d2 * d3 * d4 -
          16 * pow(a1, 2) * a5 * d2 * d4 * d5 -
          4 * pow(a3, 2) * a5 * d2 * d4 * d5 +
          16 * a0 * a1 * a2 * d0 * d1 * d4 - 8 * a0 * a1 * a2 * d0 * d2 * d3 -
          16 * a0 * a1 * a4 * d0 * d1 * d2 + 8 * a0 * a2 * a3 * d0 * d1 * d2 +
          8 * a0 * a1 * a2 * d0 * d2 * d5 - 8 * a0 * a2 * a5 * d0 * d1 * d2 +
          16 * a0 * a1 * a3 * d1 * d2 * d4 + 16 * a0 * a1 * a4 * d1 * d2 * d3 -
          16 * a1 * a2 * a3 * d0 * d1 * d4 - 16 * a1 * a2 * a4 * d0 * d1 * d3 +
          4 * a0 * a1 * a3 * d0 * d3 * d5 - 8 * a0 * a1 * a4 * d0 * d3 * d4 +
          8 * a0 * a3 * a4 * d0 * d1 * d4 - 4 * a0 * a3 * a5 * d0 * d1 * d3 -
          16 * a0 * a1 * a2 * d1 * d4 * d5 - 8 * a0 * a1 * a2 * d2 * d3 * d5 -
          16 * a0 * a1 * a5 * d1 * d2 * d4 - 8 * a0 * a2 * a3 * d1 * d2 * d5 -
          32 * a0 * a2 * a4 * d1 * d2 * d4 + 16 * a1 * a2 * a4 * d0 * d1 * d5 +
          32 * a1 * a2 * a4 * d0 * d2 * d4 + 8 * a1 * a2 * a5 * d0 * d2 * d3 +
          16 * a1 * a4 * a5 * d0 * d1 * d2 + 8 * a2 * a3 * a5 * d0 * d1 * d2 -
          8 * a0 * a1 * a4 * d0 * d4 * d5 + 8 * a0 * a2 * a3 * d0 * d4 * d5 -
          8 * a0 * a2 * a5 * d0 * d3 * d4 + 8 * a0 * a3 * a4 * d0 * d2 * d5 +
          8 * a0 * a4 * a5 * d0 * d1 * d4 - 8 * a0 * a4 * a5 * d0 * d2 * d3 +
          8 * a0 * a2 * a5 * d1 * d2 * d5 - 8 * a1 * a2 * a5 * d0 * d2 * d5 +
          16 * a1 * a2 * a3 * d1 * d4 * d5 + 16 * a1 * a2 * a4 * d1 * d3 * d5 -
          32 * a1 * a2 * a4 * d2 * d3 * d4 - 16 * a1 * a3 * a5 * d1 * d2 * d4 -
          16 * a1 * a4 * a5 * d1 * d2 * d3 + 32 * a2 * a3 * a4 * d1 * d2 * d4 +
          8 * a0 * a1 * a4 * d3 * d4 * d5 - 8 * a0 * a3 * a4 * d1 * d4 * d5 +
          4 * a0 * a3 * a5 * d1 * d3 * d5 - 8 * a0 * a3 * a5 * d2 * d3 * d4 -
          4 * a1 * a3 * a5 * d0 * d3 * d5 + 8 * a1 * a4 * a5 * d0 * d3 * d4 +
          8 * a2 * a3 * a4 * d0 * d3 * d5 - 8 * a3 * a4 * a5 * d0 * d1 * d4 +
          8 * a1 * a2 * a5 * d2 * d3 * d5 - 8 * a2 * a3 * a5 * d1 * d2 * d5 +
          8 * a0 * a3 * a5 * d2 * d4 * d5 - 8 * a0 * a4 * a5 * d1 * d4 * d5 +
          8 * a1 * a4 * a5 * d0 * d4 * d5 - 8 * a2 * a4 * a5 * d0 * d3 * d5 -
          8 * a1 * a4 * a5 * d3 * d4 * d5 + 8 * a3 * a4 * a5 * d1 * d4 * d5,
      4 * pow(a1, 3) * c3 * pow(d0, 2) - pow(a0, 3) * c1 * pow(d3, 2) +
          4 * pow(a0, 3) * c1 * pow(d4, 2) - pow(a0, 3) * c1 * pow(d5, 2) +
          8 * pow(a1, 3) * c0 * pow(d5, 2) + 16 * pow(a1, 3) * c3 * pow(d2, 2) -
          4 * pow(a1, 3) * c5 * pow(d0, 2) - 8 * pow(a2, 3) * c4 * pow(d0, 2) +
          pow(a5, 3) * c1 * pow(d0, 2) - 32 * pow(a2, 3) * c4 * pow(d1, 2) +
          12 * pow(a5, 3) * c1 * pow(d1, 2) -
          16 * pow(a1, 3) * c5 * pow(d2, 2) + 4 * pow(a1, 3) * c3 * pow(d5, 2) -
          8 * pow(a2, 3) * c4 * pow(d3, 2) + pow(a5, 3) * c1 * pow(d3, 2) -
          12 * pow(a1, 3) * c5 * pow(d5, 2) + 8 * pow(a1, 3) * c0 * d0 * d3 -
          8 * pow(a1, 3) * c0 * d0 * d5 - 16 * pow(a2, 3) * c0 * d0 * d4 +
          2 * pow(a5, 3) * c0 * d0 * d1 - 2 * pow(a0, 3) * c3 * d1 * d3 +
          32 * pow(a1, 3) * c2 * d2 * d3 - 64 * pow(a2, 3) * c1 * d1 * d4 +
          2 * pow(a0, 3) * c1 * d3 * d5 - 4 * pow(a0, 3) * c2 * d3 * d4 +
          2 * pow(a0, 3) * c3 * d1 * d5 - 4 * pow(a0, 3) * c3 * d2 * d4 +
          8 * pow(a0, 3) * c4 * d1 * d4 - 4 * pow(a0, 3) * c4 * d2 * d3 +
          2 * pow(a0, 3) * c5 * d1 * d3 - 8 * pow(a1, 3) * c0 * d3 * d5 -
          8 * pow(a1, 3) * c3 * d0 * d5 - 8 * pow(a1, 3) * c5 * d0 * d3 +
          16 * pow(a2, 3) * c0 * d3 * d4 + 16 * pow(a2, 3) * c3 * d0 * d4 +
          16 * pow(a2, 3) * c4 * d0 * d3 - 2 * pow(a5, 3) * c0 * d1 * d3 -
          2 * pow(a5, 3) * c1 * d0 * d3 - 2 * pow(a5, 3) * c3 * d0 * d1 -
          32 * pow(a1, 3) * c2 * d2 * d5 + 4 * pow(a0, 3) * c2 * d4 * d5 +
          4 * pow(a0, 3) * c4 * d2 * d5 - 2 * pow(a0, 3) * c5 * d1 * d5 +
          4 * pow(a0, 3) * c5 * d2 * d4 + 16 * pow(a1, 3) * c5 * d0 * d5 -
          16 * pow(a2, 3) * c3 * d3 * d4 + 2 * pow(a5, 3) * c3 * d1 * d3 +
          8 * pow(a1, 3) * c5 * d3 * d5 - a0 * pow(a3, 2) * c1 * pow(d0, 2) +
          3 * a1 * pow(a3, 2) * c0 * pow(d0, 2) +
          pow(a0, 2) * a1 * c0 * pow(d3, 2) +
          4 * a0 * pow(a4, 2) * c1 * pow(d0, 2) -
          12 * a1 * pow(a4, 2) * c0 * pow(d0, 2) -
          4 * pow(a0, 2) * a1 * c0 * pow(d4, 2) +
          4 * pow(a0, 2) * a1 * c3 * pow(d1, 2) -
          12 * pow(a0, 2) * a3 * c1 * pow(d1, 2) -
          4 * pow(a1, 2) * a3 * c1 * pow(d0, 2) -
          4 * a0 * pow(a2, 2) * c1 * pow(d3, 2) +
          4 * a0 * pow(a3, 2) * c1 * pow(d2, 2) -
          a0 * pow(a5, 2) * c1 * pow(d0, 2) -
          4 * a1 * pow(a2, 2) * c0 * pow(d3, 2) +
          4 * a1 * pow(a2, 2) * c3 * pow(d0, 2) +
          4 * a1 * pow(a3, 2) * c0 * pow(d2, 2) +
          3 * a1 * pow(a5, 2) * c0 * pow(d0, 2) +
          pow(a0, 2) * a1 * c0 * pow(d5, 2) +
          4 * pow(a0, 2) * a1 * c3 * pow(d2, 2) -
          4 * pow(a0, 2) * a3 * c1 * pow(d2, 2) -
          4 * pow(a2, 2) * a3 * c1 * pow(d0, 2) -
          8 * a0 * pow(a1, 2) * c1 * pow(d5, 2) +
          16 * a0 * pow(a2, 2) * c1 * pow(d4, 2) +
          16 * a0 * pow(a4, 2) * c1 * pow(d2, 2) -
          24 * a0 * pow(a5, 2) * c1 * pow(d1, 2) -
          16 * a1 * pow(a2, 2) * c0 * pow(d4, 2) +
          16 * a1 * pow(a2, 2) * c3 * pow(d1, 2) -
          16 * a1 * pow(a4, 2) * c0 * pow(d2, 2) +
          8 * a1 * pow(a5, 2) * c0 * pow(d1, 2) -
          4 * pow(a0, 2) * a1 * c5 * pow(d1, 2) -
          8 * pow(a0, 2) * a2 * c4 * pow(d1, 2) +
          8 * pow(a0, 2) * a4 * c2 * pow(d1, 2) +
          12 * pow(a0, 2) * a5 * c1 * pow(d1, 2) -
          8 * pow(a1, 2) * a2 * c4 * pow(d0, 2) -
          16 * pow(a1, 2) * a3 * c1 * pow(d2, 2) +
          8 * pow(a1, 2) * a4 * c2 * pow(d0, 2) +
          4 * pow(a1, 2) * a5 * c1 * pow(d0, 2) -
          48 * pow(a2, 2) * a3 * c1 * pow(d1, 2) -
          4 * a0 * pow(a2, 2) * c1 * pow(d5, 2) -
          4 * a0 * pow(a5, 2) * c1 * pow(d2, 2) +
          4 * a1 * pow(a2, 2) * c0 * pow(d5, 2) -
          4 * a1 * pow(a2, 2) * c5 * pow(d0, 2) +
          4 * a1 * pow(a4, 2) * c3 * pow(d0, 2) +
          4 * a1 * pow(a5, 2) * c0 * pow(d2, 2) -
          4 * a3 * pow(a4, 2) * c1 * pow(d0, 2) +
          4 * pow(a0, 2) * a1 * c3 * pow(d4, 2) -
          4 * pow(a0, 2) * a1 * c5 * pow(d2, 2) -
          8 * pow(a0, 2) * a2 * c4 * pow(d2, 2) -
          4 * pow(a0, 2) * a3 * c1 * pow(d4, 2) +
          24 * pow(a0, 2) * a4 * c2 * pow(d2, 2) +
          4 * pow(a0, 2) * a5 * c1 * pow(d2, 2) +
          8 * pow(a2, 2) * a4 * c2 * pow(d0, 2) +
          4 * pow(a2, 2) * a5 * c1 * pow(d0, 2) -
          a0 * pow(a3, 2) * c1 * pow(d5, 2) -
          3 * a0 * pow(a5, 2) * c1 * pow(d3, 2) -
          16 * a1 * pow(a2, 2) * c5 * pow(d1, 2) +
          3 * a1 * pow(a3, 2) * c0 * pow(d5, 2) -
          3 * a1 * pow(a3, 2) * c5 * pow(d0, 2) +
          a1 * pow(a5, 2) * c0 * pow(d3, 2) -
          2 * a1 * pow(a5, 2) * c3 * pow(d0, 2) +
          2 * a2 * pow(a3, 2) * c4 * pow(d0, 2) -
          2 * a3 * pow(a5, 2) * c1 * pow(d0, 2) +
          2 * pow(a0, 2) * a1 * c3 * pow(d5, 2) -
          pow(a0, 2) * a1 * c5 * pow(d3, 2) -
          2 * pow(a0, 2) * a2 * c4 * pow(d3, 2) +
          2 * pow(a0, 2) * a3 * c1 * pow(d5, 2) -
          2 * pow(a0, 2) * a4 * c2 * pow(d3, 2) +
          3 * pow(a0, 2) * a5 * c1 * pow(d3, 2) -
          32 * pow(a1, 2) * a2 * c4 * pow(d2, 2) +
          96 * pow(a1, 2) * a4 * c2 * pow(d2, 2) +
          16 * pow(a1, 2) * a5 * c1 * pow(d2, 2) +
          32 * pow(a2, 2) * a4 * c2 * pow(d1, 2) +
          48 * pow(a2, 2) * a5 * c1 * pow(d1, 2) +
          2 * pow(a3, 2) * a4 * c2 * pow(d0, 2) +
          pow(a3, 2) * a5 * c1 * pow(d0, 2) +
          4 * a0 * pow(a4, 2) * c1 * pow(d5, 2) +
          4 * a0 * pow(a5, 2) * c1 * pow(d4, 2) +
          16 * a1 * pow(a2, 2) * c3 * pow(d4, 2) -
          4 * a1 * pow(a4, 2) * c0 * pow(d5, 2) +
          16 * a1 * pow(a4, 2) * c3 * pow(d2, 2) +
          8 * a1 * pow(a4, 2) * c5 * pow(d0, 2) -
          4 * a1 * pow(a5, 2) * c0 * pow(d4, 2) +
          4 * a1 * pow(a5, 2) * c3 * pow(d1, 2) -
          16 * a3 * pow(a4, 2) * c1 * pow(d2, 2) -
          12 * a3 * pow(a5, 2) * c1 * pow(d1, 2) -
          8 * pow(a0, 2) * a5 * c1 * pow(d4, 2) -
          4 * pow(a1, 2) * a3 * c1 * pow(d5, 2) -
          16 * pow(a2, 2) * a3 * c1 * pow(d4, 2) -
          4 * a1 * pow(a2, 2) * c3 * pow(d5, 2) +
          4 * a1 * pow(a2, 2) * c5 * pow(d3, 2) -
          4 * a1 * pow(a3, 2) * c5 * pow(d2, 2) -
          4 * a1 * pow(a5, 2) * c3 * pow(d2, 2) -
          a1 * pow(a5, 2) * c5 * pow(d0, 2) -
          8 * a2 * pow(a3, 2) * c4 * pow(d2, 2) -
          2 * a2 * pow(a5, 2) * c4 * pow(d0, 2) +
          4 * a3 * pow(a5, 2) * c1 * pow(d2, 2) -
          2 * a4 * pow(a5, 2) * c2 * pow(d0, 2) -
          3 * pow(a0, 2) * a1 * c5 * pow(d5, 2) +
          2 * pow(a0, 2) * a2 * c4 * pow(d5, 2) +
          2 * pow(a0, 2) * a4 * c2 * pow(d5, 2) +
          pow(a0, 2) * a5 * c1 * pow(d5, 2) +
          4 * pow(a2, 2) * a3 * c1 * pow(d5, 2) +
          8 * pow(a2, 2) * a4 * c2 * pow(d3, 2) +
          4 * pow(a2, 2) * a5 * c1 * pow(d3, 2) +
          24 * pow(a3, 2) * a4 * c2 * pow(d2, 2) -
          4 * pow(a3, 2) * a5 * c1 * pow(d2, 2) -
          12 * a1 * pow(a5, 2) * c5 * pow(d1, 2) -
          8 * a2 * pow(a5, 2) * c4 * pow(d1, 2) -
          8 * a4 * pow(a5, 2) * c2 * pow(d1, 2) +
          8 * pow(a1, 2) * a2 * c4 * pow(d5, 2) +
          8 * pow(a1, 2) * a4 * c2 * pow(d5, 2) +
          12 * pow(a1, 2) * a5 * c1 * pow(d5, 2) +
          4 * a1 * pow(a4, 2) * c3 * pow(d5, 2) +
          4 * a1 * pow(a5, 2) * c3 * pow(d4, 2) -
          4 * a3 * pow(a4, 2) * c1 * pow(d5, 2) -
          4 * a3 * pow(a5, 2) * c1 * pow(d4, 2) -
          3 * a1 * pow(a3, 2) * c5 * pow(d5, 2) -
          a1 * pow(a5, 2) * c5 * pow(d3, 2) +
          2 * a2 * pow(a3, 2) * c4 * pow(d5, 2) -
          2 * a2 * pow(a5, 2) * c4 * pow(d3, 2) -
          2 * a4 * pow(a5, 2) * c2 * pow(d3, 2) +
          2 * pow(a3, 2) * a4 * c2 * pow(d5, 2) +
          pow(a3, 2) * a5 * c1 * pow(d5, 2) +
          8 * a0 * a1 * a3 * c0 * pow(d1, 2) -
          2 * a0 * a1 * a3 * c3 * pow(d0, 2) -
          8 * a0 * a1 * a5 * c0 * pow(d1, 2) +
          8 * a0 * a1 * a2 * c2 * pow(d3, 2) -
          16 * a0 * a2 * a4 * c0 * pow(d2, 2) -
          4 * a0 * a1 * a3 * c0 * pow(d5, 2) -
          8 * a0 * a1 * a3 * c3 * pow(d2, 2) +
          2 * a0 * a1 * a3 * c5 * pow(d0, 2) +
          8 * a0 * a1 * a4 * c4 * pow(d0, 2) -
          2 * a0 * a1 * a5 * c0 * pow(d3, 2) +
          2 * a0 * a1 * a5 * c3 * pow(d0, 2) -
          4 * a0 * a2 * a3 * c4 * pow(d0, 2) +
          4 * a0 * a2 * a4 * c0 * pow(d3, 2) -
          4 * a0 * a2 * a4 * c3 * pow(d0, 2) -
          4 * a0 * a3 * a4 * c2 * pow(d0, 2) +
          2 * a0 * a3 * a5 * c1 * pow(d0, 2) +
          32 * a1 * a2 * a3 * c2 * pow(d1, 2) -
          6 * a1 * a3 * a5 * c0 * pow(d0, 2) +
          12 * a2 * a3 * a4 * c0 * pow(d0, 2) -
          8 * a0 * a1 * a3 * c5 * pow(d1, 2) +
          8 * a0 * a1 * a5 * c0 * pow(d4, 2) -
          8 * a0 * a1 * a5 * c3 * pow(d1, 2) -
          16 * a0 * a3 * a4 * c2 * pow(d1, 2) +
          24 * a0 * a3 * a5 * c1 * pow(d1, 2) -
          64 * a1 * a2 * a4 * c1 * pow(d2, 2) -
          8 * a1 * a3 * a5 * c0 * pow(d1, 2) +
          16 * a2 * a3 * a4 * c0 * pow(d1, 2) +
          8 * a0 * a1 * a3 * c5 * pow(d2, 2) +
          2 * a0 * a1 * a5 * c0 * pow(d5, 2) -
          2 * a0 * a1 * a5 * c5 * pow(d0, 2) +
          16 * a0 * a2 * a3 * c4 * pow(d2, 2) -
          4 * a0 * a2 * a4 * c0 * pow(d5, 2) +
          16 * a0 * a2 * a4 * c3 * pow(d2, 2) +
          4 * a0 * a2 * a4 * c5 * pow(d0, 2) +
          4 * a0 * a2 * a5 * c4 * pow(d0, 2) -
          48 * a0 * a3 * a4 * c2 * pow(d2, 2) +
          4 * a0 * a4 * a5 * c2 * pow(d0, 2) -
          32 * a1 * a2 * a5 * c2 * pow(d1, 2) -
          8 * a1 * a3 * a5 * c0 * pow(d2, 2) +
          16 * a2 * a3 * a4 * c0 * pow(d2, 2) -
          12 * a2 * a4 * a5 * c0 * pow(d0, 2) -
          2 * a0 * a1 * a3 * c3 * pow(d5, 2) +
          16 * a0 * a1 * a5 * c5 * pow(d1, 2) +
          16 * a0 * a2 * a5 * c4 * pow(d1, 2) +
          2 * a1 * a3 * a5 * c3 * pow(d0, 2) -
          4 * a2 * a3 * a4 * c3 * pow(d0, 2) -
          16 * a2 * a4 * a5 * c0 * pow(d1, 2) -
          8 * a0 * a1 * a5 * c3 * pow(d4, 2) +
          8 * a0 * a3 * a5 * c1 * pow(d4, 2) -
          16 * a1 * a2 * a4 * c1 * pow(d5, 2) -
          8 * a1 * a2 * a5 * c2 * pow(d3, 2) +
          6 * a0 * a1 * a3 * c5 * pow(d5, 2) -
          2 * a0 * a1 * a5 * c3 * pow(d5, 2) +
          2 * a0 * a1 * a5 * c5 * pow(d3, 2) -
          4 * a0 * a2 * a3 * c4 * pow(d5, 2) +
          4 * a0 * a2 * a4 * c3 * pow(d5, 2) -
          4 * a0 * a2 * a4 * c5 * pow(d3, 2) +
          4 * a0 * a2 * a5 * c4 * pow(d3, 2) -
          4 * a0 * a3 * a4 * c2 * pow(d5, 2) -
          2 * a0 * a3 * a5 * c1 * pow(d5, 2) +
          4 * a0 * a4 * a5 * c2 * pow(d3, 2) -
          2 * a1 * a3 * a5 * c0 * pow(d5, 2) +
          8 * a1 * a3 * a5 * c3 * pow(d2, 2) +
          4 * a1 * a3 * a5 * c5 * pow(d0, 2) -
          8 * a1 * a4 * a5 * c4 * pow(d0, 2) +
          4 * a2 * a3 * a4 * c0 * pow(d5, 2) -
          16 * a2 * a3 * a4 * c3 * pow(d2, 2) -
          8 * a2 * a3 * a4 * c5 * pow(d0, 2) -
          4 * a2 * a4 * a5 * c0 * pow(d3, 2) +
          8 * a2 * a4 * a5 * c3 * pow(d0, 2) +
          8 * a1 * a3 * a5 * c5 * pow(d1, 2) -
          16 * a2 * a3 * a4 * c5 * pow(d1, 2) +
          16 * a3 * a4 * a5 * c2 * pow(d1, 2) +
          4 * a2 * a4 * a5 * c5 * pow(d0, 2) +
          2 * a1 * a3 * a5 * c3 * pow(d5, 2) -
          4 * a2 * a3 * a4 * c3 * pow(d5, 2) +
          16 * a2 * a4 * a5 * c5 * pow(d1, 2) +
          4 * a2 * a4 * a5 * c5 * pow(d3, 2) -
          2 * a0 * pow(a3, 2) * c0 * d0 * d1 -
          8 * a0 * pow(a1, 2) * c0 * d1 * d3 -
          8 * a0 * pow(a1, 2) * c1 * d0 * d3 -
          8 * a0 * pow(a1, 2) * c3 * d0 * d1 +
          8 * a0 * pow(a4, 2) * c0 * d0 * d1 -
          8 * pow(a1, 2) * a3 * c0 * d0 * d1 -
          2 * a0 * pow(a5, 2) * c0 * d0 * d1 +
          8 * a1 * pow(a2, 2) * c0 * d0 * d3 +
          8 * pow(a0, 2) * a1 * c1 * d1 * d3 -
          8 * pow(a2, 2) * a3 * c0 * d0 * d1 +
          8 * a0 * pow(a1, 2) * c0 * d1 * d5 +
          8 * a0 * pow(a1, 2) * c1 * d0 * d5 +
          8 * a0 * pow(a1, 2) * c5 * d0 * d1 +
          2 * pow(a0, 2) * a1 * c3 * d0 * d3 +
          2 * pow(a0, 2) * a3 * c0 * d1 * d3 +
          2 * pow(a0, 2) * a3 * c1 * d0 * d3 +
          2 * pow(a0, 2) * a3 * c3 * d0 * d1 -
          16 * pow(a1, 2) * a2 * c0 * d0 * d4 +
          16 * pow(a1, 2) * a4 * c0 * d0 * d2 +
          8 * pow(a1, 2) * a5 * c0 * d0 * d1 +
          16 * a0 * pow(a2, 2) * c0 * d2 * d4 +
          16 * a0 * pow(a2, 2) * c2 * d0 * d4 +
          16 * a0 * pow(a2, 2) * c4 * d0 * d2 +
          8 * a0 * pow(a3, 2) * c2 * d1 * d2 -
          8 * a1 * pow(a2, 2) * c0 * d0 * d5 +
          32 * a1 * pow(a2, 2) * c1 * d1 * d3 +
          8 * a1 * pow(a3, 2) * c2 * d0 * d2 +
          8 * a1 * pow(a4, 2) * c0 * d0 * d3 +
          16 * a1 * pow(a5, 2) * c1 * d0 * d1 -
          8 * a2 * pow(a3, 2) * c0 * d1 * d2 -
          8 * a2 * pow(a3, 2) * c1 * d0 * d2 -
          8 * a2 * pow(a3, 2) * c2 * d0 * d1 -
          8 * a3 * pow(a4, 2) * c0 * d0 * d1 -
          8 * pow(a0, 2) * a1 * c1 * d1 * d5 +
          8 * pow(a0, 2) * a1 * c2 * d2 * d3 -
          16 * pow(a0, 2) * a2 * c1 * d1 * d4 -
          8 * pow(a0, 2) * a3 * c2 * d1 * d2 +
          16 * pow(a0, 2) * a4 * c1 * d1 * d2 +
          16 * pow(a2, 2) * a4 * c0 * d0 * d2 +
          8 * pow(a2, 2) * a5 * c0 * d0 * d1 -
          8 * a0 * pow(a2, 2) * c3 * d1 * d3 +
          2 * a0 * pow(a3, 2) * c0 * d1 * d5 -
          4 * a0 * pow(a3, 2) * c0 * d2 * d4 +
          2 * a0 * pow(a3, 2) * c1 * d0 * d5 -
          4 * a0 * pow(a3, 2) * c2 * d0 * d4 -
          4 * a0 * pow(a3, 2) * c4 * d0 * d2 +
          2 * a0 * pow(a3, 2) * c5 * d0 * d1 +
          32 * a0 * pow(a4, 2) * c2 * d1 * d2 +
          4 * a0 * pow(a5, 2) * c0 * d1 * d3 +
          4 * a0 * pow(a5, 2) * c1 * d0 * d3 +
          4 * a0 * pow(a5, 2) * c3 * d0 * d1 -
          8 * a1 * pow(a2, 2) * c3 * d0 * d3 -
          6 * a1 * pow(a3, 2) * c0 * d0 * d5 -
          32 * a1 * pow(a4, 2) * c2 * d0 * d2 -
          4 * a1 * pow(a5, 2) * c0 * d0 * d3 +
          4 * a2 * pow(a3, 2) * c0 * d0 * d4 -
          4 * a3 * pow(a5, 2) * c0 * d0 * d1 -
          2 * pow(a0, 2) * a1 * c0 * d3 * d5 -
          2 * pow(a0, 2) * a1 * c3 * d0 * d5 -
          8 * pow(a0, 2) * a1 * c4 * d0 * d4 -
          2 * pow(a0, 2) * a1 * c5 * d0 * d3 +
          4 * pow(a0, 2) * a2 * c0 * d3 * d4 +
          4 * pow(a0, 2) * a2 * c3 * d0 * d4 +
          4 * pow(a0, 2) * a2 * c4 * d0 * d3 -
          2 * pow(a0, 2) * a3 * c0 * d1 * d5 +
          4 * pow(a0, 2) * a3 * c0 * d2 * d4 -
          2 * pow(a0, 2) * a3 * c1 * d0 * d5 +
          4 * pow(a0, 2) * a3 * c2 * d0 * d4 +
          4 * pow(a0, 2) * a3 * c4 * d0 * d2 -
          2 * pow(a0, 2) * a3 * c5 * d0 * d1 -
          8 * pow(a0, 2) * a4 * c0 * d1 * d4 +
          4 * pow(a0, 2) * a4 * c0 * d2 * d3 -
          8 * pow(a0, 2) * a4 * c1 * d0 * d4 +
          4 * pow(a0, 2) * a4 * c2 * d0 * d3 +
          4 * pow(a0, 2) * a4 * c3 * d0 * d2 -
          8 * pow(a0, 2) * a4 * c4 * d0 * d1 -
          2 * pow(a0, 2) * a5 * c0 * d1 * d3 -
          2 * pow(a0, 2) * a5 * c1 * d0 * d3 -
          2 * pow(a0, 2) * a5 * c3 * d0 * d1 -
          32 * pow(a1, 2) * a2 * c1 * d2 * d3 -
          32 * pow(a1, 2) * a2 * c2 * d1 * d3 -
          32 * pow(a1, 2) * a2 * c3 * d1 * d2 -
          32 * pow(a1, 2) * a3 * c2 * d1 * d2 +
          8 * pow(a2, 2) * a3 * c0 * d1 * d3 +
          8 * pow(a2, 2) * a3 * c1 * d0 * d3 +
          8 * pow(a2, 2) * a3 * c3 * d0 * d1 +
          4 * pow(a3, 2) * a4 * c0 * d0 * d2 +
          2 * pow(a3, 2) * a5 * c0 * d0 * d1 +
          8 * a0 * pow(a1, 2) * c1 * d3 * d5 -
          16 * a0 * pow(a1, 2) * c2 * d3 * d4 +
          8 * a0 * pow(a1, 2) * c3 * d1 * d5 -
          16 * a0 * pow(a1, 2) * c3 * d2 * d4 -
          16 * a0 * pow(a1, 2) * c4 * d2 * d3 +
          8 * a0 * pow(a1, 2) * c5 * d1 * d3 -
          8 * a0 * pow(a4, 2) * c0 * d1 * d5 -
          8 * a0 * pow(a4, 2) * c1 * d0 * d5 -
          8 * a0 * pow(a4, 2) * c5 * d0 * d1 -
          8 * a0 * pow(a5, 2) * c2 * d1 * d2 -
          32 * a1 * pow(a2, 2) * c1 * d1 * d5 +
          64 * a1 * pow(a2, 2) * c1 * d2 * d4 +
          64 * a1 * pow(a2, 2) * c2 * d1 * d4 +
          64 * a1 * pow(a2, 2) * c4 * d1 * d2 +
          16 * a1 * pow(a4, 2) * c0 * d0 * d5 +
          8 * a1 * pow(a5, 2) * c2 * d0 * d2 -
          8 * pow(a0, 2) * a1 * c2 * d2 * d5 -
          16 * pow(a0, 2) * a2 * c2 * d2 * d4 +
          8 * pow(a0, 2) * a5 * c2 * d1 * d2 +
          16 * pow(a1, 2) * a2 * c0 * d3 * d4 +
          16 * pow(a1, 2) * a2 * c3 * d0 * d4 +
          16 * pow(a1, 2) * a2 * c4 * d0 * d3 +
          8 * pow(a1, 2) * a3 * c0 * d1 * d5 +
          8 * pow(a1, 2) * a3 * c1 * d0 * d5 +
          8 * pow(a1, 2) * a3 * c5 * d0 * d1 +
          8 * pow(a1, 2) * a5 * c0 * d1 * d3 +
          8 * pow(a1, 2) * a5 * c1 * d0 * d3 +
          8 * pow(a1, 2) * a5 * c3 * d0 * d1 +
          64 * pow(a2, 2) * a4 * c1 * d1 * d2 +
          8 * a0 * pow(a2, 2) * c1 * d3 * d5 -
          16 * a0 * pow(a2, 2) * c2 * d3 * d4 +
          8 * a0 * pow(a2, 2) * c3 * d1 * d5 -
          16 * a0 * pow(a2, 2) * c3 * d2 * d4 +
          32 * a0 * pow(a2, 2) * c4 * d1 * d4 -
          16 * a0 * pow(a2, 2) * c4 * d2 * d3 +
          8 * a0 * pow(a2, 2) * c5 * d1 * d3 -
          2 * a0 * pow(a5, 2) * c0 * d1 * d5 +
          4 * a0 * pow(a5, 2) * c0 * d2 * d4 -
          2 * a0 * pow(a5, 2) * c1 * d0 * d5 +
          4 * a0 * pow(a5, 2) * c2 * d0 * d4 +
          4 * a0 * pow(a5, 2) * c4 * d0 * d2 -
          2 * a0 * pow(a5, 2) * c5 * d0 * d1 -
          32 * a1 * pow(a2, 2) * c4 * d0 * d4 -
          2 * a1 * pow(a5, 2) * c0 * d0 * d5 +
          8 * a1 * pow(a5, 2) * c1 * d1 * d3 -
          4 * a2 * pow(a5, 2) * c0 * d0 * d4 -
          4 * a4 * pow(a5, 2) * c0 * d0 * d2 +
          2 * pow(a0, 2) * a1 * c5 * d0 * d5 -
          4 * pow(a0, 2) * a2 * c0 * d4 * d5 -
          4 * pow(a0, 2) * a2 * c4 * d0 * d5 -
          4 * pow(a0, 2) * a2 * c5 * d0 * d4 -
          4 * pow(a0, 2) * a4 * c0 * d2 * d5 -
          4 * pow(a0, 2) * a4 * c2 * d0 * d5 -
          4 * pow(a0, 2) * a4 * c5 * d0 * d2 +
          2 * pow(a0, 2) * a5 * c0 * d1 * d5 -
          4 * pow(a0, 2) * a5 * c0 * d2 * d4 +
          2 * pow(a0, 2) * a5 * c1 * d0 * d5 -
          4 * pow(a0, 2) * a5 * c2 * d0 * d4 -
          4 * pow(a0, 2) * a5 * c4 * d0 * d2 +
          2 * pow(a0, 2) * a5 * c5 * d0 * d1 +
          32 * pow(a1, 2) * a2 * c1 * d2 * d5 +
          32 * pow(a1, 2) * a2 * c2 * d1 * d5 -
          64 * pow(a1, 2) * a2 * c2 * d2 * d4 +
          32 * pow(a1, 2) * a2 * c5 * d1 * d2 +
          32 * pow(a1, 2) * a5 * c2 * d1 * d2 -
          16 * pow(a2, 2) * a3 * c0 * d2 * d4 -
          16 * pow(a2, 2) * a3 * c2 * d0 * d4 -
          16 * pow(a2, 2) * a3 * c4 * d0 * d2 -
          16 * pow(a2, 2) * a4 * c0 * d2 * d3 -
          16 * pow(a2, 2) * a4 * c2 * d0 * d3 -
          16 * pow(a2, 2) * a4 * c3 * d0 * d2 -
          8 * pow(a2, 2) * a5 * c0 * d1 * d3 -
          8 * pow(a2, 2) * a5 * c1 * d0 * d3 -
          8 * pow(a2, 2) * a5 * c3 * d0 * d1 +
          16 * a0 * pow(a1, 2) * c2 * d4 * d5 +
          16 * a0 * pow(a1, 2) * c4 * d2 * d5 -
          16 * a0 * pow(a1, 2) * c5 * d1 * d5 +
          16 * a0 * pow(a1, 2) * c5 * d2 * d4 -
          6 * a0 * pow(a5, 2) * c3 * d1 * d3 +
          32 * a1 * pow(a4, 2) * c2 * d2 * d3 +
          2 * a1 * pow(a5, 2) * c3 * d0 * d3 -
          32 * a3 * pow(a4, 2) * c2 * d1 * d2 +
          2 * a3 * pow(a5, 2) * c0 * d1 * d3 +
          2 * a3 * pow(a5, 2) * c1 * d0 * d3 +
          2 * a3 * pow(a5, 2) * c3 * d0 * d1 -
          2 * pow(a0, 2) * a1 * c3 * d3 * d5 +
          8 * pow(a0, 2) * a1 * c4 * d3 * d4 -
          4 * pow(a0, 2) * a2 * c3 * d3 * d4 -
          2 * pow(a0, 2) * a3 * c1 * d3 * d5 +
          4 * pow(a0, 2) * a3 * c2 * d3 * d4 -
          2 * pow(a0, 2) * a3 * c3 * d1 * d5 +
          4 * pow(a0, 2) * a3 * c3 * d2 * d4 -
          8 * pow(a0, 2) * a3 * c4 * d1 * d4 +
          4 * pow(a0, 2) * a3 * c4 * d2 * d3 -
          2 * pow(a0, 2) * a3 * c5 * d1 * d3 -
          4 * pow(a0, 2) * a4 * c3 * d2 * d3 +
          6 * pow(a0, 2) * a5 * c3 * d1 * d3 -
          16 * pow(a1, 2) * a4 * c0 * d2 * d5 -
          16 * pow(a1, 2) * a4 * c2 * d0 * d5 -
          16 * pow(a1, 2) * a4 * c5 * d0 * d2 -
          16 * pow(a1, 2) * a5 * c0 * d1 * d5 -
          16 * pow(a1, 2) * a5 * c1 * d0 * d5 -
          16 * pow(a1, 2) * a5 * c5 * d0 * d1 -
          8 * a0 * pow(a2, 2) * c5 * d1 * d5 +
          8 * a1 * pow(a2, 2) * c5 * d0 * d5 -
          8 * a1 * pow(a3, 2) * c2 * d2 * d5 -
          8 * a1 * pow(a4, 2) * c0 * d3 * d5 -
          8 * a1 * pow(a4, 2) * c3 * d0 * d5 -
          8 * a1 * pow(a4, 2) * c5 * d0 * d3 -
          24 * a1 * pow(a5, 2) * c1 * d1 * d5 +
          16 * a1 * pow(a5, 2) * c1 * d2 * d4 +
          16 * a1 * pow(a5, 2) * c2 * d1 * d4 -
          8 * a1 * pow(a5, 2) * c2 * d2 * d3 +
          16 * a1 * pow(a5, 2) * c4 * d1 * d2 +
          8 * a2 * pow(a3, 2) * c1 * d2 * d5 +
          8 * a2 * pow(a3, 2) * c2 * d1 * d5 -
          16 * a2 * pow(a3, 2) * c2 * d2 * d4 +
          8 * a2 * pow(a3, 2) * c5 * d1 * d2 -
          16 * a2 * pow(a5, 2) * c1 * d1 * d4 +
          8 * a3 * pow(a4, 2) * c0 * d1 * d5 +
          8 * a3 * pow(a4, 2) * c1 * d0 * d5 +
          8 * a3 * pow(a4, 2) * c5 * d0 * d1 +
          8 * a3 * pow(a5, 2) * c2 * d1 * d2 -
          16 * a4 * pow(a5, 2) * c1 * d1 * d2 -
          8 * pow(a3, 2) * a5 * c2 * d1 * d2 +
          4 * a0 * pow(a3, 2) * c2 * d4 * d5 +
          4 * a0 * pow(a3, 2) * c4 * d2 * d5 -
          2 * a0 * pow(a3, 2) * c5 * d1 * d5 +
          4 * a0 * pow(a3, 2) * c5 * d2 * d4 +
          2 * a0 * pow(a5, 2) * c1 * d3 * d5 -
          4 * a0 * pow(a5, 2) * c2 * d3 * d4 +
          2 * a0 * pow(a5, 2) * c3 * d1 * d5 -
          4 * a0 * pow(a5, 2) * c3 * d2 * d4 +
          8 * a0 * pow(a5, 2) * c4 * d1 * d4 -
          4 * a0 * pow(a5, 2) * c4 * d2 * d3 +
          2 * a0 * pow(a5, 2) * c5 * d1 * d3 +
          8 * a1 * pow(a2, 2) * c3 * d3 * d5 +
          32 * a1 * pow(a2, 2) * c4 * d3 * d4 +
          6 * a1 * pow(a3, 2) * c5 * d0 * d5 +
          2 * a1 * pow(a5, 2) * c0 * d3 * d5 +
          2 * a1 * pow(a5, 2) * c3 * d0 * d5 -
          8 * a1 * pow(a5, 2) * c4 * d0 * d4 +
          2 * a1 * pow(a5, 2) * c5 * d0 * d3 -
          4 * a2 * pow(a3, 2) * c0 * d4 * d5 -
          4 * a2 * pow(a3, 2) * c4 * d0 * d5 -
          4 * a2 * pow(a3, 2) * c5 * d0 * d4 +
          4 * a2 * pow(a5, 2) * c0 * d3 * d4 +
          4 * a2 * pow(a5, 2) * c3 * d0 * d4 +
          4 * a2 * pow(a5, 2) * c4 * d0 * d3 +
          2 * a3 * pow(a5, 2) * c0 * d1 * d5 -
          4 * a3 * pow(a5, 2) * c0 * d2 * d4 +
          2 * a3 * pow(a5, 2) * c1 * d0 * d5 -
          4 * a3 * pow(a5, 2) * c2 * d0 * d4 -
          4 * a3 * pow(a5, 2) * c4 * d0 * d2 +
          2 * a3 * pow(a5, 2) * c5 * d0 * d1 +
          4 * a4 * pow(a5, 2) * c0 * d2 * d3 +
          4 * a4 * pow(a5, 2) * c2 * d0 * d3 +
          4 * a4 * pow(a5, 2) * c3 * d0 * d2 +
          4 * pow(a0, 2) * a1 * c5 * d3 * d5 -
          8 * pow(a0, 2) * a3 * c2 * d4 * d5 -
          8 * pow(a0, 2) * a3 * c4 * d2 * d5 +
          4 * pow(a0, 2) * a3 * c5 * d1 * d5 -
          8 * pow(a0, 2) * a3 * c5 * d2 * d4 +
          8 * pow(a0, 2) * a4 * c1 * d4 * d5 +
          8 * pow(a0, 2) * a4 * c4 * d1 * d5 +
          8 * pow(a0, 2) * a4 * c5 * d1 * d4 -
          4 * pow(a0, 2) * a5 * c1 * d3 * d5 +
          8 * pow(a0, 2) * a5 * c2 * d3 * d4 -
          4 * pow(a0, 2) * a5 * c3 * d1 * d5 +
          8 * pow(a0, 2) * a5 * c3 * d2 * d4 -
          16 * pow(a0, 2) * a5 * c4 * d1 * d4 +
          8 * pow(a0, 2) * a5 * c4 * d2 * d3 -
          4 * pow(a0, 2) * a5 * c5 * d1 * d3 -
          8 * pow(a2, 2) * a3 * c1 * d3 * d5 +
          16 * pow(a2, 2) * a3 * c2 * d3 * d4 -
          8 * pow(a2, 2) * a3 * c3 * d1 * d5 +
          16 * pow(a2, 2) * a3 * c3 * d2 * d4 -
          32 * pow(a2, 2) * a3 * c4 * d1 * d4 +
          16 * pow(a2, 2) * a3 * c4 * d2 * d3 -
          8 * pow(a2, 2) * a3 * c5 * d1 * d3 +
          16 * pow(a2, 2) * a4 * c3 * d2 * d3 +
          8 * pow(a2, 2) * a5 * c3 * d1 * d3 -
          4 * pow(a3, 2) * a4 * c0 * d2 * d5 -
          4 * pow(a3, 2) * a4 * c2 * d0 * d5 -
          4 * pow(a3, 2) * a4 * c5 * d0 * d2 -
          2 * pow(a3, 2) * a5 * c0 * d1 * d5 +
          4 * pow(a3, 2) * a5 * c0 * d2 * d4 -
          2 * pow(a3, 2) * a5 * c1 * d0 * d5 +
          4 * pow(a3, 2) * a5 * c2 * d0 * d4 +
          4 * pow(a3, 2) * a5 * c4 * d0 * d2 -
          2 * pow(a3, 2) * a5 * c5 * d0 * d1 +
          8 * a0 * pow(a4, 2) * c5 * d1 * d5 -
          8 * a1 * pow(a4, 2) * c5 * d0 * d5 -
          16 * pow(a1, 2) * a2 * c3 * d4 * d5 -
          16 * pow(a1, 2) * a2 * c4 * d3 * d5 -
          16 * pow(a1, 2) * a2 * c5 * d3 * d4 -
          8 * pow(a1, 2) * a3 * c5 * d1 * d5 -
          8 * pow(a1, 2) * a5 * c1 * d3 * d5 +
          16 * pow(a1, 2) * a5 * c2 * d3 * d4 -
          8 * pow(a1, 2) * a5 * c3 * d1 * d5 +
          16 * pow(a1, 2) * a5 * c3 * d2 * d4 +
          16 * pow(a1, 2) * a5 * c4 * d2 * d3 -
          8 * pow(a1, 2) * a5 * c5 * d1 * d3 -
          8 * a1 * pow(a2, 2) * c5 * d3 * d5 +
          4 * pow(a0, 2) * a2 * c5 * d4 * d5 +
          4 * pow(a0, 2) * a4 * c5 * d2 * d5 -
          4 * pow(a0, 2) * a5 * c2 * d4 * d5 -
          4 * pow(a0, 2) * a5 * c4 * d2 * d5 +
          2 * pow(a0, 2) * a5 * c5 * d1 * d5 -
          4 * pow(a0, 2) * a5 * c5 * d2 * d4 +
          8 * pow(a2, 2) * a3 * c5 * d1 * d5 -
          2 * a1 * pow(a5, 2) * c3 * d3 * d5 +
          8 * a1 * pow(a5, 2) * c4 * d3 * d4 -
          4 * a2 * pow(a5, 2) * c3 * d3 * d4 -
          2 * a3 * pow(a5, 2) * c1 * d3 * d5 +
          4 * a3 * pow(a5, 2) * c2 * d3 * d4 -
          2 * a3 * pow(a5, 2) * c3 * d1 * d5 +
          4 * a3 * pow(a5, 2) * c3 * d2 * d4 -
          8 * a3 * pow(a5, 2) * c4 * d1 * d4 +
          4 * a3 * pow(a5, 2) * c4 * d2 * d3 -
          2 * a3 * pow(a5, 2) * c5 * d1 * d3 -
          4 * a4 * pow(a5, 2) * c3 * d2 * d3 +
          16 * pow(a1, 2) * a2 * c5 * d4 * d5 +
          16 * pow(a1, 2) * a4 * c5 * d2 * d5 -
          16 * pow(a1, 2) * a5 * c2 * d4 * d5 -
          16 * pow(a1, 2) * a5 * c4 * d2 * d5 +
          24 * pow(a1, 2) * a5 * c5 * d1 * d5 -
          16 * pow(a1, 2) * a5 * c5 * d2 * d4 +
          8 * a1 * pow(a4, 2) * c5 * d3 * d5 -
          8 * a3 * pow(a4, 2) * c5 * d1 * d5 +
          4 * a2 * pow(a3, 2) * c5 * d4 * d5 +
          4 * pow(a3, 2) * a4 * c5 * d2 * d5 -
          4 * pow(a3, 2) * a5 * c2 * d4 * d5 -
          4 * pow(a3, 2) * a5 * c4 * d2 * d5 +
          2 * pow(a3, 2) * a5 * c5 * d1 * d5 -
          4 * pow(a3, 2) * a5 * c5 * d2 * d4 +
          16 * a0 * a1 * a3 * c1 * d0 * d1 - 4 * a0 * a1 * a3 * c0 * d0 * d3 +
          16 * a0 * a1 * a2 * c0 * d1 * d4 - 8 * a0 * a1 * a2 * c0 * d2 * d3 +
          16 * a0 * a1 * a2 * c1 * d0 * d4 - 8 * a0 * a1 * a2 * c2 * d0 * d3 -
          8 * a0 * a1 * a2 * c3 * d0 * d2 + 16 * a0 * a1 * a2 * c4 * d0 * d1 -
          16 * a0 * a1 * a4 * c0 * d1 * d2 - 16 * a0 * a1 * a4 * c1 * d0 * d2 -
          16 * a0 * a1 * a4 * c2 * d0 * d1 - 16 * a0 * a1 * a5 * c1 * d0 * d1 +
          8 * a0 * a2 * a3 * c0 * d1 * d2 + 8 * a0 * a2 * a3 * c1 * d0 * d2 +
          8 * a0 * a2 * a3 * c2 * d0 * d1 + 4 * a0 * a1 * a3 * c0 * d0 * d5 +
          16 * a0 * a1 * a4 * c0 * d0 * d4 + 4 * a0 * a1 * a5 * c0 * d0 * d3 -
          8 * a0 * a2 * a3 * c0 * d0 * d4 - 8 * a0 * a2 * a4 * c0 * d0 * d3 -
          8 * a0 * a3 * a4 * c0 * d0 * d2 + 4 * a0 * a3 * a5 * c0 * d0 * d1 +
          8 * a0 * a1 * a2 * c0 * d2 * d5 + 8 * a0 * a1 * a2 * c2 * d0 * d5 +
          8 * a0 * a1 * a2 * c5 * d0 * d2 - 32 * a0 * a2 * a4 * c2 * d0 * d2 -
          8 * a0 * a2 * a5 * c0 * d1 * d2 - 8 * a0 * a2 * a5 * c1 * d0 * d2 -
          8 * a0 * a2 * a5 * c2 * d0 * d1 + 64 * a1 * a2 * a3 * c1 * d1 * d2 +
          16 * a0 * a1 * a2 * c3 * d2 * d3 - 16 * a0 * a1 * a3 * c1 * d1 * d5 +
          16 * a0 * a1 * a3 * c1 * d2 * d4 + 16 * a0 * a1 * a3 * c2 * d1 * d4 -
          16 * a0 * a1 * a3 * c2 * d2 * d3 + 16 * a0 * a1 * a3 * c4 * d1 * d2 +
          16 * a0 * a1 * a4 * c1 * d2 * d3 + 16 * a0 * a1 * a4 * c2 * d1 * d3 +
          16 * a0 * a1 * a4 * c3 * d1 * d2 - 4 * a0 * a1 * a5 * c0 * d0 * d5 -
          16 * a0 * a1 * a5 * c1 * d1 * d3 + 8 * a0 * a2 * a4 * c0 * d0 * d5 +
          8 * a0 * a2 * a5 * c0 * d0 * d4 - 32 * a0 * a3 * a4 * c1 * d1 * d2 +
          8 * a0 * a4 * a5 * c0 * d0 * d2 - 16 * a1 * a2 * a3 * c0 * d1 * d4 -
          16 * a1 * a2 * a3 * c1 * d0 * d4 - 16 * a1 * a2 * a3 * c4 * d0 * d1 -
          16 * a1 * a2 * a4 * c0 * d1 * d3 - 16 * a1 * a2 * a4 * c1 * d0 * d3 -
          16 * a1 * a2 * a4 * c3 * d0 * d1 - 16 * a1 * a3 * a5 * c1 * d0 * d1 +
          32 * a2 * a3 * a4 * c1 * d0 * d1 + 4 * a0 * a1 * a3 * c0 * d3 * d5 +
          4 * a0 * a1 * a3 * c3 * d0 * d5 + 4 * a0 * a1 * a3 * c5 * d0 * d3 -
          8 * a0 * a1 * a4 * c0 * d3 * d4 - 8 * a0 * a1 * a4 * c3 * d0 * d4 -
          8 * a0 * a1 * a4 * c4 * d0 * d3 - 4 * a0 * a1 * a5 * c3 * d0 * d3 +
          8 * a0 * a2 * a4 * c3 * d0 * d3 + 8 * a0 * a3 * a4 * c0 * d1 * d4 +
          8 * a0 * a3 * a4 * c1 * d0 * d4 + 8 * a0 * a3 * a4 * c4 * d0 * d1 -
          4 * a0 * a3 * a5 * c0 * d1 * d3 - 4 * a0 * a3 * a5 * c1 * d0 * d3 -
          4 * a0 * a3 * a5 * c3 * d0 * d1 - 128 * a1 * a2 * a4 * c2 * d1 * d2 -
          64 * a1 * a2 * a5 * c1 * d1 * d2 + 4 * a1 * a3 * a5 * c0 * d0 * d3 -
          8 * a2 * a3 * a4 * c0 * d0 * d3 - 16 * a0 * a1 * a2 * c1 * d4 * d5 -
          8 * a0 * a1 * a2 * c2 * d3 * d5 - 8 * a0 * a1 * a2 * c3 * d2 * d5 -
          16 * a0 * a1 * a2 * c4 * d1 * d5 - 16 * a0 * a1 * a2 * c5 * d1 * d4 -
          8 * a0 * a1 * a2 * c5 * d2 * d3 + 16 * a0 * a1 * a3 * c2 * d2 * d5 +
          32 * a0 * a1 * a5 * c1 * d1 * d5 - 16 * a0 * a1 * a5 * c1 * d2 * d4 -
          16 * a0 * a1 * a5 * c2 * d1 * d4 - 16 * a0 * a1 * a5 * c4 * d1 * d2 -
          8 * a0 * a2 * a3 * c1 * d2 * d5 - 8 * a0 * a2 * a3 * c2 * d1 * d5 +
          32 * a0 * a2 * a3 * c2 * d2 * d4 - 8 * a0 * a2 * a3 * c5 * d1 * d2 -
          32 * a0 * a2 * a4 * c1 * d2 * d4 - 32 * a0 * a2 * a4 * c2 * d1 * d4 +
          32 * a0 * a2 * a4 * c2 * d2 * d3 - 32 * a0 * a2 * a4 * c4 * d1 * d2 +
          32 * a0 * a2 * a5 * c1 * d1 * d4 + 16 * a1 * a2 * a4 * c0 * d1 * d5 +
          32 * a1 * a2 * a4 * c0 * d2 * d4 + 16 * a1 * a2 * a4 * c1 * d0 * d5 +
          32 * a1 * a2 * a4 * c2 * d0 * d4 + 32 * a1 * a2 * a4 * c4 * d0 * d2 +
          16 * a1 * a2 * a4 * c5 * d0 * d1 + 8 * a1 * a2 * a5 * c0 * d2 * d3 +
          8 * a1 * a2 * a5 * c2 * d0 * d3 + 8 * a1 * a2 * a5 * c3 * d0 * d2 -
          16 * a1 * a3 * a5 * c2 * d0 * d2 + 16 * a1 * a4 * a5 * c0 * d1 * d2 +
          16 * a1 * a4 * a5 * c1 * d0 * d2 + 16 * a1 * a4 * a5 * c2 * d0 * d1 +
          32 * a2 * a3 * a4 * c2 * d0 * d2 + 8 * a2 * a3 * a5 * c0 * d1 * d2 +
          8 * a2 * a3 * a5 * c1 * d0 * d2 + 8 * a2 * a3 * a5 * c2 * d0 * d1 -
          32 * a2 * a4 * a5 * c1 * d0 * d1 - 8 * a0 * a1 * a3 * c5 * d0 * d5 -
          8 * a0 * a1 * a4 * c0 * d4 * d5 - 8 * a0 * a1 * a4 * c4 * d0 * d5 -
          8 * a0 * a1 * a4 * c5 * d0 * d4 + 16 * a0 * a1 * a5 * c4 * d0 * d4 +
          8 * a0 * a2 * a3 * c0 * d4 * d5 + 8 * a0 * a2 * a3 * c4 * d0 * d5 +
          8 * a0 * a2 * a3 * c5 * d0 * d4 - 8 * a0 * a2 * a5 * c0 * d3 * d4 -
          8 * a0 * a2 * a5 * c3 * d0 * d4 - 8 * a0 * a2 * a5 * c4 * d0 * d3 +
          8 * a0 * a3 * a4 * c0 * d2 * d5 + 8 * a0 * a3 * a4 * c2 * d0 * d5 +
          8 * a0 * a3 * a4 * c5 * d0 * d2 + 8 * a0 * a4 * a5 * c0 * d1 * d4 -
          8 * a0 * a4 * a5 * c0 * d2 * d3 + 8 * a0 * a4 * a5 * c1 * d0 * d4 -
          8 * a0 * a4 * a5 * c2 * d0 * d3 - 8 * a0 * a4 * a5 * c3 * d0 * d2 +
          8 * a0 * a4 * a5 * c4 * d0 * d1 + 8 * a1 * a3 * a5 * c0 * d0 * d5 -
          16 * a1 * a4 * a5 * c0 * d0 * d4 - 16 * a2 * a3 * a4 * c0 * d0 * d5 +
          16 * a2 * a4 * a5 * c0 * d0 * d3 + 8 * a0 * a2 * a5 * c1 * d2 * d5 +
          8 * a0 * a2 * a5 * c2 * d1 * d5 + 8 * a0 * a2 * a5 * c5 * d1 * d2 -
          8 * a1 * a2 * a5 * c0 * d2 * d5 - 8 * a1 * a2 * a5 * c2 * d0 * d5 -
          8 * a1 * a2 * a5 * c5 * d0 * d2 + 4 * a0 * a1 * a5 * c5 * d0 * d5 -
          8 * a0 * a2 * a4 * c5 * d0 * d5 + 16 * a1 * a2 * a3 * c1 * d4 * d5 +
          16 * a1 * a2 * a3 * c4 * d1 * d5 + 16 * a1 * a2 * a3 * c5 * d1 * d4 +
          16 * a1 * a2 * a4 * c1 * d3 * d5 - 32 * a1 * a2 * a4 * c2 * d3 * d4 +
          16 * a1 * a2 * a4 * c3 * d1 * d5 - 32 * a1 * a2 * a4 * c3 * d2 * d4 -
          32 * a1 * a2 * a4 * c4 * d2 * d3 + 16 * a1 * a2 * a4 * c5 * d1 * d3 -
          16 * a1 * a2 * a5 * c3 * d2 * d3 + 16 * a1 * a3 * a5 * c1 * d1 * d5 -
          16 * a1 * a3 * a5 * c1 * d2 * d4 - 16 * a1 * a3 * a5 * c2 * d1 * d4 +
          16 * a1 * a3 * a5 * c2 * d2 * d3 - 16 * a1 * a3 * a5 * c4 * d1 * d2 -
          16 * a1 * a4 * a5 * c1 * d2 * d3 - 16 * a1 * a4 * a5 * c2 * d1 * d3 -
          16 * a1 * a4 * a5 * c3 * d1 * d2 - 32 * a2 * a3 * a4 * c1 * d1 * d5 +
          32 * a2 * a3 * a4 * c1 * d2 * d4 + 32 * a2 * a3 * a4 * c2 * d1 * d4 -
          32 * a2 * a3 * a4 * c2 * d2 * d3 + 32 * a2 * a3 * a4 * c4 * d1 * d2 +
          8 * a2 * a4 * a5 * c0 * d0 * d5 + 32 * a3 * a4 * a5 * c1 * d1 * d2 -
          4 * a0 * a1 * a3 * c5 * d3 * d5 + 8 * a0 * a1 * a4 * c3 * d4 * d5 +
          8 * a0 * a1 * a4 * c4 * d3 * d5 + 8 * a0 * a1 * a4 * c5 * d3 * d4 +
          4 * a0 * a1 * a5 * c3 * d3 * d5 - 16 * a0 * a1 * a5 * c4 * d3 * d4 -
          8 * a0 * a2 * a4 * c3 * d3 * d5 + 8 * a0 * a2 * a5 * c3 * d3 * d4 -
          8 * a0 * a3 * a4 * c1 * d4 * d5 - 8 * a0 * a3 * a4 * c4 * d1 * d5 -
          8 * a0 * a3 * a4 * c5 * d1 * d4 + 4 * a0 * a3 * a5 * c1 * d3 * d5 -
          8 * a0 * a3 * a5 * c2 * d3 * d4 + 4 * a0 * a3 * a5 * c3 * d1 * d5 -
          8 * a0 * a3 * a5 * c3 * d2 * d4 + 16 * a0 * a3 * a5 * c4 * d1 * d4 -
          8 * a0 * a3 * a5 * c4 * d2 * d3 + 4 * a0 * a3 * a5 * c5 * d1 * d3 +
          8 * a0 * a4 * a5 * c3 * d2 * d3 - 4 * a1 * a3 * a5 * c0 * d3 * d5 -
          4 * a1 * a3 * a5 * c3 * d0 * d5 - 4 * a1 * a3 * a5 * c5 * d0 * d3 +
          8 * a1 * a4 * a5 * c0 * d3 * d4 + 8 * a1 * a4 * a5 * c3 * d0 * d4 +
          8 * a1 * a4 * a5 * c4 * d0 * d3 + 8 * a2 * a3 * a4 * c0 * d3 * d5 +
          8 * a2 * a3 * a4 * c3 * d0 * d5 + 8 * a2 * a3 * a4 * c5 * d0 * d3 -
          8 * a2 * a4 * a5 * c3 * d0 * d3 - 8 * a3 * a4 * a5 * c0 * d1 * d4 -
          8 * a3 * a4 * a5 * c1 * d0 * d4 - 8 * a3 * a4 * a5 * c4 * d0 * d1 -
          32 * a1 * a2 * a4 * c5 * d1 * d5 + 8 * a1 * a2 * a5 * c2 * d3 * d5 +
          8 * a1 * a2 * a5 * c3 * d2 * d5 + 8 * a1 * a2 * a5 * c5 * d2 * d3 -
          8 * a2 * a3 * a5 * c1 * d2 * d5 - 8 * a2 * a3 * a5 * c2 * d1 * d5 -
          8 * a2 * a3 * a5 * c5 * d1 * d2 + 32 * a2 * a4 * a5 * c1 * d1 * d5 -
          4 * a0 * a1 * a5 * c5 * d3 * d5 - 8 * a0 * a2 * a3 * c5 * d4 * d5 +
          8 * a0 * a2 * a4 * c5 * d3 * d5 - 8 * a0 * a3 * a4 * c5 * d2 * d5 +
          8 * a0 * a3 * a5 * c2 * d4 * d5 + 8 * a0 * a3 * a5 * c4 * d2 * d5 -
          4 * a0 * a3 * a5 * c5 * d1 * d5 + 8 * a0 * a3 * a5 * c5 * d2 * d4 -
          8 * a0 * a4 * a5 * c1 * d4 * d5 - 8 * a0 * a4 * a5 * c4 * d1 * d5 -
          8 * a0 * a4 * a5 * c5 * d1 * d4 - 4 * a1 * a3 * a5 * c5 * d0 * d5 +
          8 * a1 * a4 * a5 * c0 * d4 * d5 + 8 * a1 * a4 * a5 * c4 * d0 * d5 +
          8 * a1 * a4 * a5 * c5 * d0 * d4 + 8 * a2 * a3 * a4 * c5 * d0 * d5 -
          8 * a2 * a4 * a5 * c0 * d3 * d5 - 8 * a2 * a4 * a5 * c3 * d0 * d5 -
          8 * a2 * a4 * a5 * c5 * d0 * d3 + 4 * a1 * a3 * a5 * c5 * d3 * d5 -
          8 * a1 * a4 * a5 * c3 * d4 * d5 - 8 * a1 * a4 * a5 * c4 * d3 * d5 -
          8 * a1 * a4 * a5 * c5 * d3 * d4 - 8 * a2 * a3 * a4 * c5 * d3 * d5 +
          8 * a2 * a4 * a5 * c3 * d3 * d5 + 8 * a3 * a4 * a5 * c1 * d4 * d5 +
          8 * a3 * a4 * a5 * c4 * d1 * d5 + 8 * a3 * a4 * a5 * c5 * d1 * d4,
      4 * pow(a1, 3) * pow(c0, 2) * d3 - pow(a0, 3) * pow(c3, 2) * d1 +
          4 * pow(a0, 3) * pow(c4, 2) * d1 - pow(a0, 3) * pow(c5, 2) * d1 -
          4 * pow(a1, 3) * pow(c0, 2) * d5 + 16 * pow(a1, 3) * pow(c2, 2) * d3 +
          8 * pow(a1, 3) * pow(c5, 2) * d0 - 8 * pow(a2, 3) * pow(c0, 2) * d4 +
          pow(a5, 3) * pow(c0, 2) * d1 - 32 * pow(a2, 3) * pow(c1, 2) * d4 +
          12 * pow(a5, 3) * pow(c1, 2) * d1 -
          16 * pow(a1, 3) * pow(c2, 2) * d5 + 4 * pow(a1, 3) * pow(c5, 2) * d3 -
          8 * pow(a2, 3) * pow(c3, 2) * d4 + pow(a5, 3) * pow(c3, 2) * d1 -
          12 * pow(a1, 3) * pow(c5, 2) * d5 + 8 * pow(a1, 3) * c0 * c3 * d0 -
          8 * pow(a1, 3) * c0 * c5 * d0 - 16 * pow(a2, 3) * c0 * c4 * d0 +
          2 * pow(a5, 3) * c0 * c1 * d0 - 2 * pow(a0, 3) * c1 * c3 * d3 +
          32 * pow(a1, 3) * c2 * c3 * d2 - 64 * pow(a2, 3) * c1 * c4 * d1 +
          2 * pow(a0, 3) * c1 * c3 * d5 + 8 * pow(a0, 3) * c1 * c4 * d4 +
          2 * pow(a0, 3) * c1 * c5 * d3 - 4 * pow(a0, 3) * c2 * c3 * d4 -
          4 * pow(a0, 3) * c2 * c4 * d3 - 4 * pow(a0, 3) * c3 * c4 * d2 +
          2 * pow(a0, 3) * c3 * c5 * d1 - 8 * pow(a1, 3) * c0 * c3 * d5 -
          8 * pow(a1, 3) * c0 * c5 * d3 - 8 * pow(a1, 3) * c3 * c5 * d0 +
          16 * pow(a2, 3) * c0 * c3 * d4 + 16 * pow(a2, 3) * c0 * c4 * d3 +
          16 * pow(a2, 3) * c3 * c4 * d0 - 2 * pow(a5, 3) * c0 * c1 * d3 -
          2 * pow(a5, 3) * c0 * c3 * d1 - 2 * pow(a5, 3) * c1 * c3 * d0 -
          32 * pow(a1, 3) * c2 * c5 * d2 - 2 * pow(a0, 3) * c1 * c5 * d5 +
          4 * pow(a0, 3) * c2 * c4 * d5 + 4 * pow(a0, 3) * c2 * c5 * d4 +
          4 * pow(a0, 3) * c4 * c5 * d2 + 16 * pow(a1, 3) * c0 * c5 * d5 -
          16 * pow(a2, 3) * c3 * c4 * d3 + 2 * pow(a5, 3) * c1 * c3 * d3 +
          8 * pow(a1, 3) * c3 * c5 * d5 - a0 * pow(a3, 2) * pow(c0, 2) * d1 +
          3 * a1 * pow(a3, 2) * pow(c0, 2) * d0 +
          pow(a0, 2) * a1 * pow(c3, 2) * d0 +
          4 * a0 * pow(a4, 2) * pow(c0, 2) * d1 -
          12 * a1 * pow(a4, 2) * pow(c0, 2) * d0 +
          4 * pow(a0, 2) * a1 * pow(c1, 2) * d3 -
          4 * pow(a0, 2) * a1 * pow(c4, 2) * d0 -
          12 * pow(a0, 2) * a3 * pow(c1, 2) * d1 -
          4 * pow(a1, 2) * a3 * pow(c0, 2) * d1 -
          4 * a0 * pow(a2, 2) * pow(c3, 2) * d1 +
          4 * a0 * pow(a3, 2) * pow(c2, 2) * d1 -
          a0 * pow(a5, 2) * pow(c0, 2) * d1 +
          4 * a1 * pow(a2, 2) * pow(c0, 2) * d3 -
          4 * a1 * pow(a2, 2) * pow(c3, 2) * d0 +
          4 * a1 * pow(a3, 2) * pow(c2, 2) * d0 +
          3 * a1 * pow(a5, 2) * pow(c0, 2) * d0 +
          4 * pow(a0, 2) * a1 * pow(c2, 2) * d3 +
          pow(a0, 2) * a1 * pow(c5, 2) * d0 -
          4 * pow(a0, 2) * a3 * pow(c2, 2) * d1 -
          4 * pow(a2, 2) * a3 * pow(c0, 2) * d1 -
          8 * a0 * pow(a1, 2) * pow(c5, 2) * d1 +
          16 * a0 * pow(a2, 2) * pow(c4, 2) * d1 +
          16 * a0 * pow(a4, 2) * pow(c2, 2) * d1 -
          24 * a0 * pow(a5, 2) * pow(c1, 2) * d1 +
          16 * a1 * pow(a2, 2) * pow(c1, 2) * d3 -
          16 * a1 * pow(a2, 2) * pow(c4, 2) * d0 -
          16 * a1 * pow(a4, 2) * pow(c2, 2) * d0 +
          8 * a1 * pow(a5, 2) * pow(c1, 2) * d0 -
          4 * pow(a0, 2) * a1 * pow(c1, 2) * d5 -
          8 * pow(a0, 2) * a2 * pow(c1, 2) * d4 +
          8 * pow(a0, 2) * a4 * pow(c1, 2) * d2 +
          12 * pow(a0, 2) * a5 * pow(c1, 2) * d1 -
          8 * pow(a1, 2) * a2 * pow(c0, 2) * d4 -
          16 * pow(a1, 2) * a3 * pow(c2, 2) * d1 +
          8 * pow(a1, 2) * a4 * pow(c0, 2) * d2 +
          4 * pow(a1, 2) * a5 * pow(c0, 2) * d1 -
          48 * pow(a2, 2) * a3 * pow(c1, 2) * d1 -
          4 * a0 * pow(a2, 2) * pow(c5, 2) * d1 -
          4 * a0 * pow(a5, 2) * pow(c2, 2) * d1 -
          4 * a1 * pow(a2, 2) * pow(c0, 2) * d5 +
          4 * a1 * pow(a2, 2) * pow(c5, 2) * d0 +
          4 * a1 * pow(a4, 2) * pow(c0, 2) * d3 +
          4 * a1 * pow(a5, 2) * pow(c2, 2) * d0 -
          4 * a3 * pow(a4, 2) * pow(c0, 2) * d1 -
          4 * pow(a0, 2) * a1 * pow(c2, 2) * d5 +
          4 * pow(a0, 2) * a1 * pow(c4, 2) * d3 -
          8 * pow(a0, 2) * a2 * pow(c2, 2) * d4 -
          4 * pow(a0, 2) * a3 * pow(c4, 2) * d1 +
          24 * pow(a0, 2) * a4 * pow(c2, 2) * d2 +
          4 * pow(a0, 2) * a5 * pow(c2, 2) * d1 +
          8 * pow(a2, 2) * a4 * pow(c0, 2) * d2 +
          4 * pow(a2, 2) * a5 * pow(c0, 2) * d1 -
          a0 * pow(a3, 2) * pow(c5, 2) * d1 -
          3 * a0 * pow(a5, 2) * pow(c3, 2) * d1 -
          16 * a1 * pow(a2, 2) * pow(c1, 2) * d5 -
          3 * a1 * pow(a3, 2) * pow(c0, 2) * d5 +
          3 * a1 * pow(a3, 2) * pow(c5, 2) * d0 -
          2 * a1 * pow(a5, 2) * pow(c0, 2) * d3 +
          a1 * pow(a5, 2) * pow(c3, 2) * d0 +
          2 * a2 * pow(a3, 2) * pow(c0, 2) * d4 -
          2 * a3 * pow(a5, 2) * pow(c0, 2) * d1 -
          pow(a0, 2) * a1 * pow(c3, 2) * d5 +
          2 * pow(a0, 2) * a1 * pow(c5, 2) * d3 -
          2 * pow(a0, 2) * a2 * pow(c3, 2) * d4 +
          2 * pow(a0, 2) * a3 * pow(c5, 2) * d1 -
          2 * pow(a0, 2) * a4 * pow(c3, 2) * d2 +
          3 * pow(a0, 2) * a5 * pow(c3, 2) * d1 -
          32 * pow(a1, 2) * a2 * pow(c2, 2) * d4 +
          96 * pow(a1, 2) * a4 * pow(c2, 2) * d2 +
          16 * pow(a1, 2) * a5 * pow(c2, 2) * d1 +
          32 * pow(a2, 2) * a4 * pow(c1, 2) * d2 +
          48 * pow(a2, 2) * a5 * pow(c1, 2) * d1 +
          2 * pow(a3, 2) * a4 * pow(c0, 2) * d2 +
          pow(a3, 2) * a5 * pow(c0, 2) * d1 +
          4 * a0 * pow(a4, 2) * pow(c5, 2) * d1 +
          4 * a0 * pow(a5, 2) * pow(c4, 2) * d1 +
          16 * a1 * pow(a2, 2) * pow(c4, 2) * d3 +
          8 * a1 * pow(a4, 2) * pow(c0, 2) * d5 +
          16 * a1 * pow(a4, 2) * pow(c2, 2) * d3 -
          4 * a1 * pow(a4, 2) * pow(c5, 2) * d0 +
          4 * a1 * pow(a5, 2) * pow(c1, 2) * d3 -
          4 * a1 * pow(a5, 2) * pow(c4, 2) * d0 -
          16 * a3 * pow(a4, 2) * pow(c2, 2) * d1 -
          12 * a3 * pow(a5, 2) * pow(c1, 2) * d1 -
          8 * pow(a0, 2) * a5 * pow(c4, 2) * d1 -
          4 * pow(a1, 2) * a3 * pow(c5, 2) * d1 -
          16 * pow(a2, 2) * a3 * pow(c4, 2) * d1 +
          4 * a1 * pow(a2, 2) * pow(c3, 2) * d5 -
          4 * a1 * pow(a2, 2) * pow(c5, 2) * d3 -
          4 * a1 * pow(a3, 2) * pow(c2, 2) * d5 -
          a1 * pow(a5, 2) * pow(c0, 2) * d5 -
          4 * a1 * pow(a5, 2) * pow(c2, 2) * d3 -
          8 * a2 * pow(a3, 2) * pow(c2, 2) * d4 -
          2 * a2 * pow(a5, 2) * pow(c0, 2) * d4 +
          4 * a3 * pow(a5, 2) * pow(c2, 2) * d1 -
          2 * a4 * pow(a5, 2) * pow(c0, 2) * d2 -
          3 * pow(a0, 2) * a1 * pow(c5, 2) * d5 +
          2 * pow(a0, 2) * a2 * pow(c5, 2) * d4 +
          2 * pow(a0, 2) * a4 * pow(c5, 2) * d2 +
          pow(a0, 2) * a5 * pow(c5, 2) * d1 +
          4 * pow(a2, 2) * a3 * pow(c5, 2) * d1 +
          8 * pow(a2, 2) * a4 * pow(c3, 2) * d2 +
          4 * pow(a2, 2) * a5 * pow(c3, 2) * d1 +
          24 * pow(a3, 2) * a4 * pow(c2, 2) * d2 -
          4 * pow(a3, 2) * a5 * pow(c2, 2) * d1 -
          12 * a1 * pow(a5, 2) * pow(c1, 2) * d5 -
          8 * a2 * pow(a5, 2) * pow(c1, 2) * d4 -
          8 * a4 * pow(a5, 2) * pow(c1, 2) * d2 +
          8 * pow(a1, 2) * a2 * pow(c5, 2) * d4 +
          8 * pow(a1, 2) * a4 * pow(c5, 2) * d2 +
          12 * pow(a1, 2) * a5 * pow(c5, 2) * d1 +
          4 * a1 * pow(a4, 2) * pow(c5, 2) * d3 +
          4 * a1 * pow(a5, 2) * pow(c4, 2) * d3 -
          4 * a3 * pow(a4, 2) * pow(c5, 2) * d1 -
          4 * a3 * pow(a5, 2) * pow(c4, 2) * d1 -
          3 * a1 * pow(a3, 2) * pow(c5, 2) * d5 -
          a1 * pow(a5, 2) * pow(c3, 2) * d5 +
          2 * a2 * pow(a3, 2) * pow(c5, 2) * d4 -
          2 * a2 * pow(a5, 2) * pow(c3, 2) * d4 -
          2 * a4 * pow(a5, 2) * pow(c3, 2) * d2 +
          2 * pow(a3, 2) * a4 * pow(c5, 2) * d2 +
          pow(a3, 2) * a5 * pow(c5, 2) * d1 +
          8 * a0 * a1 * a3 * pow(c1, 2) * d0 -
          2 * a0 * a1 * a3 * pow(c0, 2) * d3 -
          8 * a0 * a1 * a5 * pow(c1, 2) * d0 +
          8 * a0 * a1 * a2 * pow(c3, 2) * d2 -
          16 * a0 * a2 * a4 * pow(c2, 2) * d0 +
          2 * a0 * a1 * a3 * pow(c0, 2) * d5 -
          8 * a0 * a1 * a3 * pow(c2, 2) * d3 -
          4 * a0 * a1 * a3 * pow(c5, 2) * d0 +
          8 * a0 * a1 * a4 * pow(c0, 2) * d4 +
          2 * a0 * a1 * a5 * pow(c0, 2) * d3 -
          2 * a0 * a1 * a5 * pow(c3, 2) * d0 -
          4 * a0 * a2 * a3 * pow(c0, 2) * d4 -
          4 * a0 * a2 * a4 * pow(c0, 2) * d3 +
          4 * a0 * a2 * a4 * pow(c3, 2) * d0 -
          4 * a0 * a3 * a4 * pow(c0, 2) * d2 +
          2 * a0 * a3 * a5 * pow(c0, 2) * d1 +
          32 * a1 * a2 * a3 * pow(c1, 2) * d2 -
          6 * a1 * a3 * a5 * pow(c0, 2) * d0 +
          12 * a2 * a3 * a4 * pow(c0, 2) * d0 -
          8 * a0 * a1 * a3 * pow(c1, 2) * d5 -
          8 * a0 * a1 * a5 * pow(c1, 2) * d3 +
          8 * a0 * a1 * a5 * pow(c4, 2) * d0 -
          16 * a0 * a3 * a4 * pow(c1, 2) * d2 +
          24 * a0 * a3 * a5 * pow(c1, 2) * d1 -
          64 * a1 * a2 * a4 * pow(c2, 2) * d1 -
          8 * a1 * a3 * a5 * pow(c1, 2) * d0 +
          16 * a2 * a3 * a4 * pow(c1, 2) * d0 +
          8 * a0 * a1 * a3 * pow(c2, 2) * d5 -
          2 * a0 * a1 * a5 * pow(c0, 2) * d5 +
          2 * a0 * a1 * a5 * pow(c5, 2) * d0 +
          16 * a0 * a2 * a3 * pow(c2, 2) * d4 +
          4 * a0 * a2 * a4 * pow(c0, 2) * d5 +
          16 * a0 * a2 * a4 * pow(c2, 2) * d3 -
          4 * a0 * a2 * a4 * pow(c5, 2) * d0 +
          4 * a0 * a2 * a5 * pow(c0, 2) * d4 -
          48 * a0 * a3 * a4 * pow(c2, 2) * d2 +
          4 * a0 * a4 * a5 * pow(c0, 2) * d2 -
          32 * a1 * a2 * a5 * pow(c1, 2) * d2 -
          8 * a1 * a3 * a5 * pow(c2, 2) * d0 +
          16 * a2 * a3 * a4 * pow(c2, 2) * d0 -
          12 * a2 * a4 * a5 * pow(c0, 2) * d0 -
          2 * a0 * a1 * a3 * pow(c5, 2) * d3 +
          16 * a0 * a1 * a5 * pow(c1, 2) * d5 +
          16 * a0 * a2 * a5 * pow(c1, 2) * d4 +
          2 * a1 * a3 * a5 * pow(c0, 2) * d3 -
          4 * a2 * a3 * a4 * pow(c0, 2) * d3 -
          16 * a2 * a4 * a5 * pow(c1, 2) * d0 -
          8 * a0 * a1 * a5 * pow(c4, 2) * d3 +
          8 * a0 * a3 * a5 * pow(c4, 2) * d1 -
          16 * a1 * a2 * a4 * pow(c5, 2) * d1 -
          8 * a1 * a2 * a5 * pow(c3, 2) * d2 +
          6 * a0 * a1 * a3 * pow(c5, 2) * d5 +
          2 * a0 * a1 * a5 * pow(c3, 2) * d5 -
          2 * a0 * a1 * a5 * pow(c5, 2) * d3 -
          4 * a0 * a2 * a3 * pow(c5, 2) * d4 -
          4 * a0 * a2 * a4 * pow(c3, 2) * d5 +
          4 * a0 * a2 * a4 * pow(c5, 2) * d3 +
          4 * a0 * a2 * a5 * pow(c3, 2) * d4 -
          4 * a0 * a3 * a4 * pow(c5, 2) * d2 -
          2 * a0 * a3 * a5 * pow(c5, 2) * d1 +
          4 * a0 * a4 * a5 * pow(c3, 2) * d2 +
          4 * a1 * a3 * a5 * pow(c0, 2) * d5 +
          8 * a1 * a3 * a5 * pow(c2, 2) * d3 -
          2 * a1 * a3 * a5 * pow(c5, 2) * d0 -
          8 * a1 * a4 * a5 * pow(c0, 2) * d4 -
          8 * a2 * a3 * a4 * pow(c0, 2) * d5 -
          16 * a2 * a3 * a4 * pow(c2, 2) * d3 +
          4 * a2 * a3 * a4 * pow(c5, 2) * d0 +
          8 * a2 * a4 * a5 * pow(c0, 2) * d3 -
          4 * a2 * a4 * a5 * pow(c3, 2) * d0 +
          8 * a1 * a3 * a5 * pow(c1, 2) * d5 -
          16 * a2 * a3 * a4 * pow(c1, 2) * d5 +
          16 * a3 * a4 * a5 * pow(c1, 2) * d2 +
          4 * a2 * a4 * a5 * pow(c0, 2) * d5 +
          2 * a1 * a3 * a5 * pow(c5, 2) * d3 -
          4 * a2 * a3 * a4 * pow(c5, 2) * d3 +
          16 * a2 * a4 * a5 * pow(c1, 2) * d5 +
          4 * a2 * a4 * a5 * pow(c3, 2) * d5 -
          2 * a0 * pow(a3, 2) * c0 * c1 * d0 -
          8 * a0 * pow(a1, 2) * c0 * c1 * d3 -
          8 * a0 * pow(a1, 2) * c0 * c3 * d1 -
          8 * a0 * pow(a1, 2) * c1 * c3 * d0 +
          8 * a0 * pow(a4, 2) * c0 * c1 * d0 -
          8 * pow(a1, 2) * a3 * c0 * c1 * d0 -
          2 * a0 * pow(a5, 2) * c0 * c1 * d0 +
          8 * a1 * pow(a2, 2) * c0 * c3 * d0 +
          8 * pow(a0, 2) * a1 * c1 * c3 * d1 -
          8 * pow(a2, 2) * a3 * c0 * c1 * d0 +
          8 * a0 * pow(a1, 2) * c0 * c1 * d5 +
          8 * a0 * pow(a1, 2) * c0 * c5 * d1 +
          8 * a0 * pow(a1, 2) * c1 * c5 * d0 +
          2 * pow(a0, 2) * a1 * c0 * c3 * d3 +
          2 * pow(a0, 2) * a3 * c0 * c1 * d3 +
          2 * pow(a0, 2) * a3 * c0 * c3 * d1 +
          2 * pow(a0, 2) * a3 * c1 * c3 * d0 -
          16 * pow(a1, 2) * a2 * c0 * c4 * d0 +
          16 * pow(a1, 2) * a4 * c0 * c2 * d0 +
          8 * pow(a1, 2) * a5 * c0 * c1 * d0 +
          16 * a0 * pow(a2, 2) * c0 * c2 * d4 +
          16 * a0 * pow(a2, 2) * c0 * c4 * d2 +
          16 * a0 * pow(a2, 2) * c2 * c4 * d0 +
          8 * a0 * pow(a3, 2) * c1 * c2 * d2 -
          8 * a1 * pow(a2, 2) * c0 * c5 * d0 +
          32 * a1 * pow(a2, 2) * c1 * c3 * d1 +
          8 * a1 * pow(a3, 2) * c0 * c2 * d2 +
          8 * a1 * pow(a4, 2) * c0 * c3 * d0 +
          16 * a1 * pow(a5, 2) * c0 * c1 * d1 -
          8 * a2 * pow(a3, 2) * c0 * c1 * d2 -
          8 * a2 * pow(a3, 2) * c0 * c2 * d1 -
          8 * a2 * pow(a3, 2) * c1 * c2 * d0 -
          8 * a3 * pow(a4, 2) * c0 * c1 * d0 -
          8 * pow(a0, 2) * a1 * c1 * c5 * d1 +
          8 * pow(a0, 2) * a1 * c2 * c3 * d2 -
          16 * pow(a0, 2) * a2 * c1 * c4 * d1 -
          8 * pow(a0, 2) * a3 * c1 * c2 * d2 +
          16 * pow(a0, 2) * a4 * c1 * c2 * d1 +
          16 * pow(a2, 2) * a4 * c0 * c2 * d0 +
          8 * pow(a2, 2) * a5 * c0 * c1 * d0 -
          8 * a0 * pow(a2, 2) * c1 * c3 * d3 +
          2 * a0 * pow(a3, 2) * c0 * c1 * d5 -
          4 * a0 * pow(a3, 2) * c0 * c2 * d4 -
          4 * a0 * pow(a3, 2) * c0 * c4 * d2 +
          2 * a0 * pow(a3, 2) * c0 * c5 * d1 +
          2 * a0 * pow(a3, 2) * c1 * c5 * d0 -
          4 * a0 * pow(a3, 2) * c2 * c4 * d0 +
          32 * a0 * pow(a4, 2) * c1 * c2 * d2 +
          4 * a0 * pow(a5, 2) * c0 * c1 * d3 +
          4 * a0 * pow(a5, 2) * c0 * c3 * d1 +
          4 * a0 * pow(a5, 2) * c1 * c3 * d0 -
          8 * a1 * pow(a2, 2) * c0 * c3 * d3 -
          6 * a1 * pow(a3, 2) * c0 * c5 * d0 -
          32 * a1 * pow(a4, 2) * c0 * c2 * d2 -
          4 * a1 * pow(a5, 2) * c0 * c3 * d0 +
          4 * a2 * pow(a3, 2) * c0 * c4 * d0 -
          4 * a3 * pow(a5, 2) * c0 * c1 * d0 -
          2 * pow(a0, 2) * a1 * c0 * c3 * d5 -
          8 * pow(a0, 2) * a1 * c0 * c4 * d4 -
          2 * pow(a0, 2) * a1 * c0 * c5 * d3 -
          2 * pow(a0, 2) * a1 * c3 * c5 * d0 +
          4 * pow(a0, 2) * a2 * c0 * c3 * d4 +
          4 * pow(a0, 2) * a2 * c0 * c4 * d3 +
          4 * pow(a0, 2) * a2 * c3 * c4 * d0 -
          2 * pow(a0, 2) * a3 * c0 * c1 * d5 +
          4 * pow(a0, 2) * a3 * c0 * c2 * d4 +
          4 * pow(a0, 2) * a3 * c0 * c4 * d2 -
          2 * pow(a0, 2) * a3 * c0 * c5 * d1 -
          2 * pow(a0, 2) * a3 * c1 * c5 * d0 +
          4 * pow(a0, 2) * a3 * c2 * c4 * d0 -
          8 * pow(a0, 2) * a4 * c0 * c1 * d4 +
          4 * pow(a0, 2) * a4 * c0 * c2 * d3 +
          4 * pow(a0, 2) * a4 * c0 * c3 * d2 -
          8 * pow(a0, 2) * a4 * c0 * c4 * d1 -
          8 * pow(a0, 2) * a4 * c1 * c4 * d0 +
          4 * pow(a0, 2) * a4 * c2 * c3 * d0 -
          2 * pow(a0, 2) * a5 * c0 * c1 * d3 -
          2 * pow(a0, 2) * a5 * c0 * c3 * d1 -
          2 * pow(a0, 2) * a5 * c1 * c3 * d0 -
          32 * pow(a1, 2) * a2 * c1 * c2 * d3 -
          32 * pow(a1, 2) * a2 * c1 * c3 * d2 -
          32 * pow(a1, 2) * a2 * c2 * c3 * d1 -
          32 * pow(a1, 2) * a3 * c1 * c2 * d2 +
          8 * pow(a2, 2) * a3 * c0 * c1 * d3 +
          8 * pow(a2, 2) * a3 * c0 * c3 * d1 +
          8 * pow(a2, 2) * a3 * c1 * c3 * d0 +
          4 * pow(a3, 2) * a4 * c0 * c2 * d0 +
          2 * pow(a3, 2) * a5 * c0 * c1 * d0 +
          8 * a0 * pow(a1, 2) * c1 * c3 * d5 +
          8 * a0 * pow(a1, 2) * c1 * c5 * d3 -
          16 * a0 * pow(a1, 2) * c2 * c3 * d4 -
          16 * a0 * pow(a1, 2) * c2 * c4 * d3 -
          16 * a0 * pow(a1, 2) * c3 * c4 * d2 +
          8 * a0 * pow(a1, 2) * c3 * c5 * d1 -
          8 * a0 * pow(a4, 2) * c0 * c1 * d5 -
          8 * a0 * pow(a4, 2) * c0 * c5 * d1 -
          8 * a0 * pow(a4, 2) * c1 * c5 * d0 -
          8 * a0 * pow(a5, 2) * c1 * c2 * d2 +
          64 * a1 * pow(a2, 2) * c1 * c2 * d4 +
          64 * a1 * pow(a2, 2) * c1 * c4 * d2 -
          32 * a1 * pow(a2, 2) * c1 * c5 * d1 +
          64 * a1 * pow(a2, 2) * c2 * c4 * d1 +
          16 * a1 * pow(a4, 2) * c0 * c5 * d0 +
          8 * a1 * pow(a5, 2) * c0 * c2 * d2 -
          8 * pow(a0, 2) * a1 * c2 * c5 * d2 -
          16 * pow(a0, 2) * a2 * c2 * c4 * d2 +
          8 * pow(a0, 2) * a5 * c1 * c2 * d2 +
          16 * pow(a1, 2) * a2 * c0 * c3 * d4 +
          16 * pow(a1, 2) * a2 * c0 * c4 * d3 +
          16 * pow(a1, 2) * a2 * c3 * c4 * d0 +
          8 * pow(a1, 2) * a3 * c0 * c1 * d5 +
          8 * pow(a1, 2) * a3 * c0 * c5 * d1 +
          8 * pow(a1, 2) * a3 * c1 * c5 * d0 +
          8 * pow(a1, 2) * a5 * c0 * c1 * d3 +
          8 * pow(a1, 2) * a5 * c0 * c3 * d1 +
          8 * pow(a1, 2) * a5 * c1 * c3 * d0 +
          64 * pow(a2, 2) * a4 * c1 * c2 * d1 +
          8 * a0 * pow(a2, 2) * c1 * c3 * d5 +
          32 * a0 * pow(a2, 2) * c1 * c4 * d4 +
          8 * a0 * pow(a2, 2) * c1 * c5 * d3 -
          16 * a0 * pow(a2, 2) * c2 * c3 * d4 -
          16 * a0 * pow(a2, 2) * c2 * c4 * d3 -
          16 * a0 * pow(a2, 2) * c3 * c4 * d2 +
          8 * a0 * pow(a2, 2) * c3 * c5 * d1 -
          2 * a0 * pow(a5, 2) * c0 * c1 * d5 +
          4 * a0 * pow(a5, 2) * c0 * c2 * d4 +
          4 * a0 * pow(a5, 2) * c0 * c4 * d2 -
          2 * a0 * pow(a5, 2) * c0 * c5 * d1 -
          2 * a0 * pow(a5, 2) * c1 * c5 * d0 +
          4 * a0 * pow(a5, 2) * c2 * c4 * d0 -
          32 * a1 * pow(a2, 2) * c0 * c4 * d4 -
          2 * a1 * pow(a5, 2) * c0 * c5 * d0 +
          8 * a1 * pow(a5, 2) * c1 * c3 * d1 -
          4 * a2 * pow(a5, 2) * c0 * c4 * d0 -
          4 * a4 * pow(a5, 2) * c0 * c2 * d0 +
          2 * pow(a0, 2) * a1 * c0 * c5 * d5 -
          4 * pow(a0, 2) * a2 * c0 * c4 * d5 -
          4 * pow(a0, 2) * a2 * c0 * c5 * d4 -
          4 * pow(a0, 2) * a2 * c4 * c5 * d0 -
          4 * pow(a0, 2) * a4 * c0 * c2 * d5 -
          4 * pow(a0, 2) * a4 * c0 * c5 * d2 -
          4 * pow(a0, 2) * a4 * c2 * c5 * d0 +
          2 * pow(a0, 2) * a5 * c0 * c1 * d5 -
          4 * pow(a0, 2) * a5 * c0 * c2 * d4 -
          4 * pow(a0, 2) * a5 * c0 * c4 * d2 +
          2 * pow(a0, 2) * a5 * c0 * c5 * d1 +
          2 * pow(a0, 2) * a5 * c1 * c5 * d0 -
          4 * pow(a0, 2) * a5 * c2 * c4 * d0 +
          32 * pow(a1, 2) * a2 * c1 * c2 * d5 +
          32 * pow(a1, 2) * a2 * c1 * c5 * d2 -
          64 * pow(a1, 2) * a2 * c2 * c4 * d2 +
          32 * pow(a1, 2) * a2 * c2 * c5 * d1 +
          32 * pow(a1, 2) * a5 * c1 * c2 * d2 -
          16 * pow(a2, 2) * a3 * c0 * c2 * d4 -
          16 * pow(a2, 2) * a3 * c0 * c4 * d2 -
          16 * pow(a2, 2) * a3 * c2 * c4 * d0 -
          16 * pow(a2, 2) * a4 * c0 * c2 * d3 -
          16 * pow(a2, 2) * a4 * c0 * c3 * d2 -
          16 * pow(a2, 2) * a4 * c2 * c3 * d0 -
          8 * pow(a2, 2) * a5 * c0 * c1 * d3 -
          8 * pow(a2, 2) * a5 * c0 * c3 * d1 -
          8 * pow(a2, 2) * a5 * c1 * c3 * d0 -
          16 * a0 * pow(a1, 2) * c1 * c5 * d5 +
          16 * a0 * pow(a1, 2) * c2 * c4 * d5 +
          16 * a0 * pow(a1, 2) * c2 * c5 * d4 +
          16 * a0 * pow(a1, 2) * c4 * c5 * d2 -
          6 * a0 * pow(a5, 2) * c1 * c3 * d3 +
          32 * a1 * pow(a4, 2) * c2 * c3 * d2 +
          2 * a1 * pow(a5, 2) * c0 * c3 * d3 -
          32 * a3 * pow(a4, 2) * c1 * c2 * d2 +
          2 * a3 * pow(a5, 2) * c0 * c1 * d3 +
          2 * a3 * pow(a5, 2) * c0 * c3 * d1 +
          2 * a3 * pow(a5, 2) * c1 * c3 * d0 +
          8 * pow(a0, 2) * a1 * c3 * c4 * d4 -
          2 * pow(a0, 2) * a1 * c3 * c5 * d3 -
          4 * pow(a0, 2) * a2 * c3 * c4 * d3 -
          2 * pow(a0, 2) * a3 * c1 * c3 * d5 -
          8 * pow(a0, 2) * a3 * c1 * c4 * d4 -
          2 * pow(a0, 2) * a3 * c1 * c5 * d3 +
          4 * pow(a0, 2) * a3 * c2 * c3 * d4 +
          4 * pow(a0, 2) * a3 * c2 * c4 * d3 +
          4 * pow(a0, 2) * a3 * c3 * c4 * d2 -
          2 * pow(a0, 2) * a3 * c3 * c5 * d1 -
          4 * pow(a0, 2) * a4 * c2 * c3 * d3 +
          6 * pow(a0, 2) * a5 * c1 * c3 * d3 -
          16 * pow(a1, 2) * a4 * c0 * c2 * d5 -
          16 * pow(a1, 2) * a4 * c0 * c5 * d2 -
          16 * pow(a1, 2) * a4 * c2 * c5 * d0 -
          16 * pow(a1, 2) * a5 * c0 * c1 * d5 -
          16 * pow(a1, 2) * a5 * c0 * c5 * d1 -
          16 * pow(a1, 2) * a5 * c1 * c5 * d0 -
          8 * a0 * pow(a2, 2) * c1 * c5 * d5 +
          8 * a1 * pow(a2, 2) * c0 * c5 * d5 -
          8 * a1 * pow(a3, 2) * c2 * c5 * d2 -
          8 * a1 * pow(a4, 2) * c0 * c3 * d5 -
          8 * a1 * pow(a4, 2) * c0 * c5 * d3 -
          8 * a1 * pow(a4, 2) * c3 * c5 * d0 +
          16 * a1 * pow(a5, 2) * c1 * c2 * d4 +
          16 * a1 * pow(a5, 2) * c1 * c4 * d2 -
          24 * a1 * pow(a5, 2) * c1 * c5 * d1 -
          8 * a1 * pow(a5, 2) * c2 * c3 * d2 +
          16 * a1 * pow(a5, 2) * c2 * c4 * d1 +
          8 * a2 * pow(a3, 2) * c1 * c2 * d5 +
          8 * a2 * pow(a3, 2) * c1 * c5 * d2 -
          16 * a2 * pow(a3, 2) * c2 * c4 * d2 +
          8 * a2 * pow(a3, 2) * c2 * c5 * d1 -
          16 * a2 * pow(a5, 2) * c1 * c4 * d1 +
          8 * a3 * pow(a4, 2) * c0 * c1 * d5 +
          8 * a3 * pow(a4, 2) * c0 * c5 * d1 +
          8 * a3 * pow(a4, 2) * c1 * c5 * d0 +
          8 * a3 * pow(a5, 2) * c1 * c2 * d2 -
          16 * a4 * pow(a5, 2) * c1 * c2 * d1 -
          8 * pow(a3, 2) * a5 * c1 * c2 * d2 -
          2 * a0 * pow(a3, 2) * c1 * c5 * d5 +
          4 * a0 * pow(a3, 2) * c2 * c4 * d5 +
          4 * a0 * pow(a3, 2) * c2 * c5 * d4 +
          4 * a0 * pow(a3, 2) * c4 * c5 * d2 +
          2 * a0 * pow(a5, 2) * c1 * c3 * d5 +
          8 * a0 * pow(a5, 2) * c1 * c4 * d4 +
          2 * a0 * pow(a5, 2) * c1 * c5 * d3 -
          4 * a0 * pow(a5, 2) * c2 * c3 * d4 -
          4 * a0 * pow(a5, 2) * c2 * c4 * d3 -
          4 * a0 * pow(a5, 2) * c3 * c4 * d2 +
          2 * a0 * pow(a5, 2) * c3 * c5 * d1 +
          32 * a1 * pow(a2, 2) * c3 * c4 * d4 +
          8 * a1 * pow(a2, 2) * c3 * c5 * d3 +
          6 * a1 * pow(a3, 2) * c0 * c5 * d5 +
          2 * a1 * pow(a5, 2) * c0 * c3 * d5 -
          8 * a1 * pow(a5, 2) * c0 * c4 * d4 +
          2 * a1 * pow(a5, 2) * c0 * c5 * d3 +
          2 * a1 * pow(a5, 2) * c3 * c5 * d0 -
          4 * a2 * pow(a3, 2) * c0 * c4 * d5 -
          4 * a2 * pow(a3, 2) * c0 * c5 * d4 -
          4 * a2 * pow(a3, 2) * c4 * c5 * d0 +
          4 * a2 * pow(a5, 2) * c0 * c3 * d4 +
          4 * a2 * pow(a5, 2) * c0 * c4 * d3 +
          4 * a2 * pow(a5, 2) * c3 * c4 * d0 +
          2 * a3 * pow(a5, 2) * c0 * c1 * d5 -
          4 * a3 * pow(a5, 2) * c0 * c2 * d4 -
          4 * a3 * pow(a5, 2) * c0 * c4 * d2 +
          2 * a3 * pow(a5, 2) * c0 * c5 * d1 +
          2 * a3 * pow(a5, 2) * c1 * c5 * d0 -
          4 * a3 * pow(a5, 2) * c2 * c4 * d0 +
          4 * a4 * pow(a5, 2) * c0 * c2 * d3 +
          4 * a4 * pow(a5, 2) * c0 * c3 * d2 +
          4 * a4 * pow(a5, 2) * c2 * c3 * d0 +
          4 * pow(a0, 2) * a1 * c3 * c5 * d5 +
          4 * pow(a0, 2) * a3 * c1 * c5 * d5 -
          8 * pow(a0, 2) * a3 * c2 * c4 * d5 -
          8 * pow(a0, 2) * a3 * c2 * c5 * d4 -
          8 * pow(a0, 2) * a3 * c4 * c5 * d2 +
          8 * pow(a0, 2) * a4 * c1 * c4 * d5 +
          8 * pow(a0, 2) * a4 * c1 * c5 * d4 +
          8 * pow(a0, 2) * a4 * c4 * c5 * d1 -
          4 * pow(a0, 2) * a5 * c1 * c3 * d5 -
          16 * pow(a0, 2) * a5 * c1 * c4 * d4 -
          4 * pow(a0, 2) * a5 * c1 * c5 * d3 +
          8 * pow(a0, 2) * a5 * c2 * c3 * d4 +
          8 * pow(a0, 2) * a5 * c2 * c4 * d3 +
          8 * pow(a0, 2) * a5 * c3 * c4 * d2 -
          4 * pow(a0, 2) * a5 * c3 * c5 * d1 -
          8 * pow(a2, 2) * a3 * c1 * c3 * d5 -
          32 * pow(a2, 2) * a3 * c1 * c4 * d4 -
          8 * pow(a2, 2) * a3 * c1 * c5 * d3 +
          16 * pow(a2, 2) * a3 * c2 * c3 * d4 +
          16 * pow(a2, 2) * a3 * c2 * c4 * d3 +
          16 * pow(a2, 2) * a3 * c3 * c4 * d2 -
          8 * pow(a2, 2) * a3 * c3 * c5 * d1 +
          16 * pow(a2, 2) * a4 * c2 * c3 * d3 +
          8 * pow(a2, 2) * a5 * c1 * c3 * d3 -
          4 * pow(a3, 2) * a4 * c0 * c2 * d5 -
          4 * pow(a3, 2) * a4 * c0 * c5 * d2 -
          4 * pow(a3, 2) * a4 * c2 * c5 * d0 -
          2 * pow(a3, 2) * a5 * c0 * c1 * d5 +
          4 * pow(a3, 2) * a5 * c0 * c2 * d4 +
          4 * pow(a3, 2) * a5 * c0 * c4 * d2 -
          2 * pow(a3, 2) * a5 * c0 * c5 * d1 -
          2 * pow(a3, 2) * a5 * c1 * c5 * d0 +
          4 * pow(a3, 2) * a5 * c2 * c4 * d0 +
          8 * a0 * pow(a4, 2) * c1 * c5 * d5 -
          8 * a1 * pow(a4, 2) * c0 * c5 * d5 -
          16 * pow(a1, 2) * a2 * c3 * c4 * d5 -
          16 * pow(a1, 2) * a2 * c3 * c5 * d4 -
          16 * pow(a1, 2) * a2 * c4 * c5 * d3 -
          8 * pow(a1, 2) * a3 * c1 * c5 * d5 -
          8 * pow(a1, 2) * a5 * c1 * c3 * d5 -
          8 * pow(a1, 2) * a5 * c1 * c5 * d3 +
          16 * pow(a1, 2) * a5 * c2 * c3 * d4 +
          16 * pow(a1, 2) * a5 * c2 * c4 * d3 +
          16 * pow(a1, 2) * a5 * c3 * c4 * d2 -
          8 * pow(a1, 2) * a5 * c3 * c5 * d1 -
          8 * a1 * pow(a2, 2) * c3 * c5 * d5 +
          4 * pow(a0, 2) * a2 * c4 * c5 * d5 +
          4 * pow(a0, 2) * a4 * c2 * c5 * d5 +
          2 * pow(a0, 2) * a5 * c1 * c5 * d5 -
          4 * pow(a0, 2) * a5 * c2 * c4 * d5 -
          4 * pow(a0, 2) * a5 * c2 * c5 * d4 -
          4 * pow(a0, 2) * a5 * c4 * c5 * d2 +
          8 * pow(a2, 2) * a3 * c1 * c5 * d5 +
          8 * a1 * pow(a5, 2) * c3 * c4 * d4 -
          2 * a1 * pow(a5, 2) * c3 * c5 * d3 -
          4 * a2 * pow(a5, 2) * c3 * c4 * d3 -
          2 * a3 * pow(a5, 2) * c1 * c3 * d5 -
          8 * a3 * pow(a5, 2) * c1 * c4 * d4 -
          2 * a3 * pow(a5, 2) * c1 * c5 * d3 +
          4 * a3 * pow(a5, 2) * c2 * c3 * d4 +
          4 * a3 * pow(a5, 2) * c2 * c4 * d3 +
          4 * a3 * pow(a5, 2) * c3 * c4 * d2 -
          2 * a3 * pow(a5, 2) * c3 * c5 * d1 -
          4 * a4 * pow(a5, 2) * c2 * c3 * d3 +
          16 * pow(a1, 2) * a2 * c4 * c5 * d5 +
          16 * pow(a1, 2) * a4 * c2 * c5 * d5 +
          24 * pow(a1, 2) * a5 * c1 * c5 * d5 -
          16 * pow(a1, 2) * a5 * c2 * c4 * d5 -
          16 * pow(a1, 2) * a5 * c2 * c5 * d4 -
          16 * pow(a1, 2) * a5 * c4 * c5 * d2 +
          8 * a1 * pow(a4, 2) * c3 * c5 * d5 -
          8 * a3 * pow(a4, 2) * c1 * c5 * d5 +
          4 * a2 * pow(a3, 2) * c4 * c5 * d5 +
          4 * pow(a3, 2) * a4 * c2 * c5 * d5 +
          2 * pow(a3, 2) * a5 * c1 * c5 * d5 -
          4 * pow(a3, 2) * a5 * c2 * c4 * d5 -
          4 * pow(a3, 2) * a5 * c2 * c5 * d4 -
          4 * pow(a3, 2) * a5 * c4 * c5 * d2 +
          16 * a0 * a1 * a3 * c0 * c1 * d1 - 4 * a0 * a1 * a3 * c0 * c3 * d0 +
          16 * a0 * a1 * a2 * c0 * c1 * d4 - 8 * a0 * a1 * a2 * c0 * c2 * d3 -
          8 * a0 * a1 * a2 * c0 * c3 * d2 + 16 * a0 * a1 * a2 * c0 * c4 * d1 +
          16 * a0 * a1 * a2 * c1 * c4 * d0 - 8 * a0 * a1 * a2 * c2 * c3 * d0 -
          16 * a0 * a1 * a4 * c0 * c1 * d2 - 16 * a0 * a1 * a4 * c0 * c2 * d1 -
          16 * a0 * a1 * a4 * c1 * c2 * d0 - 16 * a0 * a1 * a5 * c0 * c1 * d1 +
          8 * a0 * a2 * a3 * c0 * c1 * d2 + 8 * a0 * a2 * a3 * c0 * c2 * d1 +
          8 * a0 * a2 * a3 * c1 * c2 * d0 + 4 * a0 * a1 * a3 * c0 * c5 * d0 +
          16 * a0 * a1 * a4 * c0 * c4 * d0 + 4 * a0 * a1 * a5 * c0 * c3 * d0 -
          8 * a0 * a2 * a3 * c0 * c4 * d0 - 8 * a0 * a2 * a4 * c0 * c3 * d0 -
          8 * a0 * a3 * a4 * c0 * c2 * d0 + 4 * a0 * a3 * a5 * c0 * c1 * d0 +
          8 * a0 * a1 * a2 * c0 * c2 * d5 + 8 * a0 * a1 * a2 * c0 * c5 * d2 +
          8 * a0 * a1 * a2 * c2 * c5 * d0 - 32 * a0 * a2 * a4 * c0 * c2 * d2 -
          8 * a0 * a2 * a5 * c0 * c1 * d2 - 8 * a0 * a2 * a5 * c0 * c2 * d1 -
          8 * a0 * a2 * a5 * c1 * c2 * d0 + 64 * a1 * a2 * a3 * c1 * c2 * d1 +
          16 * a0 * a1 * a2 * c2 * c3 * d3 + 16 * a0 * a1 * a3 * c1 * c2 * d4 +
          16 * a0 * a1 * a3 * c1 * c4 * d2 - 16 * a0 * a1 * a3 * c1 * c5 * d1 -
          16 * a0 * a1 * a3 * c2 * c3 * d2 + 16 * a0 * a1 * a3 * c2 * c4 * d1 +
          16 * a0 * a1 * a4 * c1 * c2 * d3 + 16 * a0 * a1 * a4 * c1 * c3 * d2 +
          16 * a0 * a1 * a4 * c2 * c3 * d1 - 4 * a0 * a1 * a5 * c0 * c5 * d0 -
          16 * a0 * a1 * a5 * c1 * c3 * d1 + 8 * a0 * a2 * a4 * c0 * c5 * d0 +
          8 * a0 * a2 * a5 * c0 * c4 * d0 - 32 * a0 * a3 * a4 * c1 * c2 * d1 +
          8 * a0 * a4 * a5 * c0 * c2 * d0 - 16 * a1 * a2 * a3 * c0 * c1 * d4 -
          16 * a1 * a2 * a3 * c0 * c4 * d1 - 16 * a1 * a2 * a3 * c1 * c4 * d0 -
          16 * a1 * a2 * a4 * c0 * c1 * d3 - 16 * a1 * a2 * a4 * c0 * c3 * d1 -
          16 * a1 * a2 * a4 * c1 * c3 * d0 - 16 * a1 * a3 * a5 * c0 * c1 * d1 +
          32 * a2 * a3 * a4 * c0 * c1 * d1 + 4 * a0 * a1 * a3 * c0 * c3 * d5 +
          4 * a0 * a1 * a3 * c0 * c5 * d3 + 4 * a0 * a1 * a3 * c3 * c5 * d0 -
          8 * a0 * a1 * a4 * c0 * c3 * d4 - 8 * a0 * a1 * a4 * c0 * c4 * d3 -
          8 * a0 * a1 * a4 * c3 * c4 * d0 - 4 * a0 * a1 * a5 * c0 * c3 * d3 +
          8 * a0 * a2 * a4 * c0 * c3 * d3 + 8 * a0 * a3 * a4 * c0 * c1 * d4 +
          8 * a0 * a3 * a4 * c0 * c4 * d1 + 8 * a0 * a3 * a4 * c1 * c4 * d0 -
          4 * a0 * a3 * a5 * c0 * c1 * d3 - 4 * a0 * a3 * a5 * c0 * c3 * d1 -
          4 * a0 * a3 * a5 * c1 * c3 * d0 - 128 * a1 * a2 * a4 * c1 * c2 * d2 -
          64 * a1 * a2 * a5 * c1 * c2 * d1 + 4 * a1 * a3 * a5 * c0 * c3 * d0 -
          8 * a2 * a3 * a4 * c0 * c3 * d0 - 16 * a0 * a1 * a2 * c1 * c4 * d5 -
          16 * a0 * a1 * a2 * c1 * c5 * d4 - 8 * a0 * a1 * a2 * c2 * c3 * d5 -
          8 * a0 * a1 * a2 * c2 * c5 * d3 - 8 * a0 * a1 * a2 * c3 * c5 * d2 -
          16 * a0 * a1 * a2 * c4 * c5 * d1 + 16 * a0 * a1 * a3 * c2 * c5 * d2 -
          16 * a0 * a1 * a5 * c1 * c2 * d4 - 16 * a0 * a1 * a5 * c1 * c4 * d2 +
          32 * a0 * a1 * a5 * c1 * c5 * d1 - 16 * a0 * a1 * a5 * c2 * c4 * d1 -
          8 * a0 * a2 * a3 * c1 * c2 * d5 - 8 * a0 * a2 * a3 * c1 * c5 * d2 +
          32 * a0 * a2 * a3 * c2 * c4 * d2 - 8 * a0 * a2 * a3 * c2 * c5 * d1 -
          32 * a0 * a2 * a4 * c1 * c2 * d4 - 32 * a0 * a2 * a4 * c1 * c4 * d2 +
          32 * a0 * a2 * a4 * c2 * c3 * d2 - 32 * a0 * a2 * a4 * c2 * c4 * d1 +
          32 * a0 * a2 * a5 * c1 * c4 * d1 + 16 * a1 * a2 * a4 * c0 * c1 * d5 +
          32 * a1 * a2 * a4 * c0 * c2 * d4 + 32 * a1 * a2 * a4 * c0 * c4 * d2 +
          16 * a1 * a2 * a4 * c0 * c5 * d1 + 16 * a1 * a2 * a4 * c1 * c5 * d0 +
          32 * a1 * a2 * a4 * c2 * c4 * d0 + 8 * a1 * a2 * a5 * c0 * c2 * d3 +
          8 * a1 * a2 * a5 * c0 * c3 * d2 + 8 * a1 * a2 * a5 * c2 * c3 * d0 -
          16 * a1 * a3 * a5 * c0 * c2 * d2 + 16 * a1 * a4 * a5 * c0 * c1 * d2 +
          16 * a1 * a4 * a5 * c0 * c2 * d1 + 16 * a1 * a4 * a5 * c1 * c2 * d0 +
          32 * a2 * a3 * a4 * c0 * c2 * d2 + 8 * a2 * a3 * a5 * c0 * c1 * d2 +
          8 * a2 * a3 * a5 * c0 * c2 * d1 + 8 * a2 * a3 * a5 * c1 * c2 * d0 -
          32 * a2 * a4 * a5 * c0 * c1 * d1 - 8 * a0 * a1 * a3 * c0 * c5 * d5 -
          8 * a0 * a1 * a4 * c0 * c4 * d5 - 8 * a0 * a1 * a4 * c0 * c5 * d4 -
          8 * a0 * a1 * a4 * c4 * c5 * d0 + 16 * a0 * a1 * a5 * c0 * c4 * d4 +
          8 * a0 * a2 * a3 * c0 * c4 * d5 + 8 * a0 * a2 * a3 * c0 * c5 * d4 +
          8 * a0 * a2 * a3 * c4 * c5 * d0 - 8 * a0 * a2 * a5 * c0 * c3 * d4 -
          8 * a0 * a2 * a5 * c0 * c4 * d3 - 8 * a0 * a2 * a5 * c3 * c4 * d0 +
          8 * a0 * a3 * a4 * c0 * c2 * d5 + 8 * a0 * a3 * a4 * c0 * c5 * d2 +
          8 * a0 * a3 * a4 * c2 * c5 * d0 + 8 * a0 * a4 * a5 * c0 * c1 * d4 -
          8 * a0 * a4 * a5 * c0 * c2 * d3 - 8 * a0 * a4 * a5 * c0 * c3 * d2 +
          8 * a0 * a4 * a5 * c0 * c4 * d1 + 8 * a0 * a4 * a5 * c1 * c4 * d0 -
          8 * a0 * a4 * a5 * c2 * c3 * d0 + 8 * a1 * a3 * a5 * c0 * c5 * d0 -
          16 * a1 * a4 * a5 * c0 * c4 * d0 - 16 * a2 * a3 * a4 * c0 * c5 * d0 +
          16 * a2 * a4 * a5 * c0 * c3 * d0 + 8 * a0 * a2 * a5 * c1 * c2 * d5 +
          8 * a0 * a2 * a5 * c1 * c5 * d2 + 8 * a0 * a2 * a5 * c2 * c5 * d1 -
          8 * a1 * a2 * a5 * c0 * c2 * d5 - 8 * a1 * a2 * a5 * c0 * c5 * d2 -
          8 * a1 * a2 * a5 * c2 * c5 * d0 + 4 * a0 * a1 * a5 * c0 * c5 * d5 -
          8 * a0 * a2 * a4 * c0 * c5 * d5 + 16 * a1 * a2 * a3 * c1 * c4 * d5 +
          16 * a1 * a2 * a3 * c1 * c5 * d4 + 16 * a1 * a2 * a3 * c4 * c5 * d1 +
          16 * a1 * a2 * a4 * c1 * c3 * d5 + 16 * a1 * a2 * a4 * c1 * c5 * d3 -
          32 * a1 * a2 * a4 * c2 * c3 * d4 - 32 * a1 * a2 * a4 * c2 * c4 * d3 -
          32 * a1 * a2 * a4 * c3 * c4 * d2 + 16 * a1 * a2 * a4 * c3 * c5 * d1 -
          16 * a1 * a2 * a5 * c2 * c3 * d3 - 16 * a1 * a3 * a5 * c1 * c2 * d4 -
          16 * a1 * a3 * a5 * c1 * c4 * d2 + 16 * a1 * a3 * a5 * c1 * c5 * d1 +
          16 * a1 * a3 * a5 * c2 * c3 * d2 - 16 * a1 * a3 * a5 * c2 * c4 * d1 -
          16 * a1 * a4 * a5 * c1 * c2 * d3 - 16 * a1 * a4 * a5 * c1 * c3 * d2 -
          16 * a1 * a4 * a5 * c2 * c3 * d1 + 32 * a2 * a3 * a4 * c1 * c2 * d4 +
          32 * a2 * a3 * a4 * c1 * c4 * d2 - 32 * a2 * a3 * a4 * c1 * c5 * d1 -
          32 * a2 * a3 * a4 * c2 * c3 * d2 + 32 * a2 * a3 * a4 * c2 * c4 * d1 +
          8 * a2 * a4 * a5 * c0 * c5 * d0 + 32 * a3 * a4 * a5 * c1 * c2 * d1 -
          4 * a0 * a1 * a3 * c3 * c5 * d5 + 8 * a0 * a1 * a4 * c3 * c4 * d5 +
          8 * a0 * a1 * a4 * c3 * c5 * d4 + 8 * a0 * a1 * a4 * c4 * c5 * d3 -
          16 * a0 * a1 * a5 * c3 * c4 * d4 + 4 * a0 * a1 * a5 * c3 * c5 * d3 -
          8 * a0 * a2 * a4 * c3 * c5 * d3 + 8 * a0 * a2 * a5 * c3 * c4 * d3 -
          8 * a0 * a3 * a4 * c1 * c4 * d5 - 8 * a0 * a3 * a4 * c1 * c5 * d4 -
          8 * a0 * a3 * a4 * c4 * c5 * d1 + 4 * a0 * a3 * a5 * c1 * c3 * d5 +
          16 * a0 * a3 * a5 * c1 * c4 * d4 + 4 * a0 * a3 * a5 * c1 * c5 * d3 -
          8 * a0 * a3 * a5 * c2 * c3 * d4 - 8 * a0 * a3 * a5 * c2 * c4 * d3 -
          8 * a0 * a3 * a5 * c3 * c4 * d2 + 4 * a0 * a3 * a5 * c3 * c5 * d1 +
          8 * a0 * a4 * a5 * c2 * c3 * d3 - 4 * a1 * a3 * a5 * c0 * c3 * d5 -
          4 * a1 * a3 * a5 * c0 * c5 * d3 - 4 * a1 * a3 * a5 * c3 * c5 * d0 +
          8 * a1 * a4 * a5 * c0 * c3 * d4 + 8 * a1 * a4 * a5 * c0 * c4 * d3 +
          8 * a1 * a4 * a5 * c3 * c4 * d0 + 8 * a2 * a3 * a4 * c0 * c3 * d5 +
          8 * a2 * a3 * a4 * c0 * c5 * d3 + 8 * a2 * a3 * a4 * c3 * c5 * d0 -
          8 * a2 * a4 * a5 * c0 * c3 * d3 - 8 * a3 * a4 * a5 * c0 * c1 * d4 -
          8 * a3 * a4 * a5 * c0 * c4 * d1 - 8 * a3 * a4 * a5 * c1 * c4 * d0 -
          32 * a1 * a2 * a4 * c1 * c5 * d5 + 8 * a1 * a2 * a5 * c2 * c3 * d5 +
          8 * a1 * a2 * a5 * c2 * c5 * d3 + 8 * a1 * a2 * a5 * c3 * c5 * d2 -
          8 * a2 * a3 * a5 * c1 * c2 * d5 - 8 * a2 * a3 * a5 * c1 * c5 * d2 -
          8 * a2 * a3 * a5 * c2 * c5 * d1 + 32 * a2 * a4 * a5 * c1 * c5 * d1 -
          4 * a0 * a1 * a5 * c3 * c5 * d5 - 8 * a0 * a2 * a3 * c4 * c5 * d5 +
          8 * a0 * a2 * a4 * c3 * c5 * d5 - 8 * a0 * a3 * a4 * c2 * c5 * d5 -
          4 * a0 * a3 * a5 * c1 * c5 * d5 + 8 * a0 * a3 * a5 * c2 * c4 * d5 +
          8 * a0 * a3 * a5 * c2 * c5 * d4 + 8 * a0 * a3 * a5 * c4 * c5 * d2 -
          8 * a0 * a4 * a5 * c1 * c4 * d5 - 8 * a0 * a4 * a5 * c1 * c5 * d4 -
          8 * a0 * a4 * a5 * c4 * c5 * d1 - 4 * a1 * a3 * a5 * c0 * c5 * d5 +
          8 * a1 * a4 * a5 * c0 * c4 * d5 + 8 * a1 * a4 * a5 * c0 * c5 * d4 +
          8 * a1 * a4 * a5 * c4 * c5 * d0 + 8 * a2 * a3 * a4 * c0 * c5 * d5 -
          8 * a2 * a4 * a5 * c0 * c3 * d5 - 8 * a2 * a4 * a5 * c0 * c5 * d3 -
          8 * a2 * a4 * a5 * c3 * c5 * d0 + 4 * a1 * a3 * a5 * c3 * c5 * d5 -
          8 * a1 * a4 * a5 * c3 * c4 * d5 - 8 * a1 * a4 * a5 * c3 * c5 * d4 -
          8 * a1 * a4 * a5 * c4 * c5 * d3 - 8 * a2 * a3 * a4 * c3 * c5 * d5 +
          8 * a2 * a4 * a5 * c3 * c5 * d3 + 8 * a3 * a4 * a5 * c1 * c4 * d5 +
          8 * a3 * a4 * a5 * c1 * c5 * d4 + 8 * a3 * a4 * a5 * c4 * c5 * d1,
      4 * pow(a5, 3) * pow(c1, 3) - 4 * pow(a1, 3) * pow(c5, 3) +
          a1 * pow(a3, 2) * pow(c0, 3) - 4 * pow(a0, 2) * a3 * pow(c1, 3) -
          4 * a1 * pow(a4, 2) * pow(c0, 3) - 8 * a0 * pow(a5, 2) * pow(c1, 3) +
          a1 * pow(a5, 2) * pow(c0, 3) - pow(a0, 2) * a1 * pow(c5, 3) +
          8 * pow(a0, 2) * a4 * pow(c2, 3) + 4 * pow(a0, 2) * a5 * pow(c1, 3) -
          16 * pow(a2, 2) * a3 * pow(c1, 3) +
          32 * pow(a1, 2) * a4 * pow(c2, 3) +
          16 * pow(a2, 2) * a5 * pow(c1, 3) - a1 * pow(a3, 2) * pow(c5, 3) -
          4 * a3 * pow(a5, 2) * pow(c1, 3) + 8 * pow(a3, 2) * a4 * pow(c2, 3) -
          pow(a0, 3) * c1 * pow(c3, 2) + 4 * pow(a1, 3) * pow(c0, 2) * c3 +
          4 * pow(a0, 3) * c1 * pow(c4, 2) - pow(a0, 3) * c1 * pow(c5, 2) +
          8 * pow(a1, 3) * c0 * pow(c5, 2) - 4 * pow(a1, 3) * pow(c0, 2) * c5 +
          16 * pow(a1, 3) * pow(c2, 2) * c3 - 8 * pow(a2, 3) * pow(c0, 2) * c4 +
          pow(a5, 3) * pow(c0, 2) * c1 - 32 * pow(a2, 3) * pow(c1, 2) * c4 -
          16 * pow(a1, 3) * pow(c2, 2) * c5 + 4 * pow(a1, 3) * c3 * pow(c5, 2) -
          8 * pow(a2, 3) * pow(c3, 2) * c4 + pow(a5, 3) * c1 * pow(c3, 2) +
          2 * a0 * a1 * a3 * pow(c5, 3) - 16 * a0 * a3 * a4 * pow(c2, 3) +
          8 * a0 * a3 * a5 * pow(c1, 3) - 2 * a1 * a3 * a5 * pow(c0, 3) +
          4 * a2 * a3 * a4 * pow(c0, 3) - 4 * a2 * a4 * a5 * pow(c0, 3) +
          2 * pow(a0, 3) * c1 * c3 * c5 - 4 * pow(a0, 3) * c2 * c3 * c4 -
          8 * pow(a1, 3) * c0 * c3 * c5 + 16 * pow(a2, 3) * c0 * c3 * c4 -
          2 * pow(a5, 3) * c0 * c1 * c3 + 4 * pow(a0, 3) * c2 * c4 * c5 -
          a0 * pow(a3, 2) * pow(c0, 2) * c1 +
          pow(a0, 2) * a1 * c0 * pow(c3, 2) +
          4 * a0 * pow(a4, 2) * pow(c0, 2) * c1 -
          4 * pow(a0, 2) * a1 * c0 * pow(c4, 2) +
          4 * pow(a0, 2) * a1 * pow(c1, 2) * c3 -
          4 * pow(a1, 2) * a3 * pow(c0, 2) * c1 -
          4 * a0 * pow(a2, 2) * c1 * pow(c3, 2) +
          4 * a0 * pow(a3, 2) * c1 * pow(c2, 2) -
          a0 * pow(a5, 2) * pow(c0, 2) * c1 -
          4 * a1 * pow(a2, 2) * c0 * pow(c3, 2) +
          4 * a1 * pow(a2, 2) * pow(c0, 2) * c3 +
          4 * a1 * pow(a3, 2) * c0 * pow(c2, 2) +
          pow(a0, 2) * a1 * c0 * pow(c5, 2) +
          4 * pow(a0, 2) * a1 * pow(c2, 2) * c3 -
          4 * pow(a0, 2) * a3 * c1 * pow(c2, 2) -
          4 * pow(a2, 2) * a3 * pow(c0, 2) * c1 -
          8 * a0 * pow(a1, 2) * c1 * pow(c5, 2) +
          16 * a0 * pow(a2, 2) * c1 * pow(c4, 2) +
          16 * a0 * pow(a4, 2) * c1 * pow(c2, 2) -
          16 * a1 * pow(a2, 2) * c0 * pow(c4, 2) +
          16 * a1 * pow(a2, 2) * pow(c1, 2) * c3 -
          16 * a1 * pow(a4, 2) * c0 * pow(c2, 2) +
          8 * a1 * pow(a5, 2) * c0 * pow(c1, 2) -
          4 * pow(a0, 2) * a1 * pow(c1, 2) * c5 -
          8 * pow(a0, 2) * a2 * pow(c1, 2) * c4 +
          8 * pow(a0, 2) * a4 * pow(c1, 2) * c2 -
          8 * pow(a1, 2) * a2 * pow(c0, 2) * c4 -
          16 * pow(a1, 2) * a3 * c1 * pow(c2, 2) +
          8 * pow(a1, 2) * a4 * pow(c0, 2) * c2 +
          4 * pow(a1, 2) * a5 * pow(c0, 2) * c1 -
          4 * a0 * pow(a2, 2) * c1 * pow(c5, 2) -
          4 * a0 * pow(a5, 2) * c1 * pow(c2, 2) +
          4 * a1 * pow(a2, 2) * c0 * pow(c5, 2) -
          4 * a1 * pow(a2, 2) * pow(c0, 2) * c5 +
          4 * a1 * pow(a4, 2) * pow(c0, 2) * c3 +
          4 * a1 * pow(a5, 2) * c0 * pow(c2, 2) -
          4 * a3 * pow(a4, 2) * pow(c0, 2) * c1 +
          4 * pow(a0, 2) * a1 * c3 * pow(c4, 2) -
          4 * pow(a0, 2) * a1 * pow(c2, 2) * c5 -
          8 * pow(a0, 2) * a2 * pow(c2, 2) * c4 -
          4 * pow(a0, 2) * a3 * c1 * pow(c4, 2) +
          4 * pow(a0, 2) * a5 * c1 * pow(c2, 2) +
          8 * pow(a2, 2) * a4 * pow(c0, 2) * c2 +
          4 * pow(a2, 2) * a5 * pow(c0, 2) * c1 -
          a0 * pow(a3, 2) * c1 * pow(c5, 2) -
          3 * a0 * pow(a5, 2) * c1 * pow(c3, 2) -
          16 * a1 * pow(a2, 2) * pow(c1, 2) * c5 +
          3 * a1 * pow(a3, 2) * c0 * pow(c5, 2) -
          3 * a1 * pow(a3, 2) * pow(c0, 2) * c5 +
          a1 * pow(a5, 2) * c0 * pow(c3, 2) -
          2 * a1 * pow(a5, 2) * pow(c0, 2) * c3 +
          2 * a2 * pow(a3, 2) * pow(c0, 2) * c4 -
          2 * a3 * pow(a5, 2) * pow(c0, 2) * c1 +
          2 * pow(a0, 2) * a1 * c3 * pow(c5, 2) -
          pow(a0, 2) * a1 * pow(c3, 2) * c5 -
          2 * pow(a0, 2) * a2 * pow(c3, 2) * c4 +
          2 * pow(a0, 2) * a3 * c1 * pow(c5, 2) -
          2 * pow(a0, 2) * a4 * c2 * pow(c3, 2) +
          3 * pow(a0, 2) * a5 * c1 * pow(c3, 2) -
          32 * pow(a1, 2) * a2 * pow(c2, 2) * c4 +
          16 * pow(a1, 2) * a5 * c1 * pow(c2, 2) +
          32 * pow(a2, 2) * a4 * pow(c1, 2) * c2 +
          2 * pow(a3, 2) * a4 * pow(c0, 2) * c2 +
          pow(a3, 2) * a5 * pow(c0, 2) * c1 +
          4 * a0 * pow(a4, 2) * c1 * pow(c5, 2) +
          4 * a0 * pow(a5, 2) * c1 * pow(c4, 2) +
          16 * a1 * pow(a2, 2) * c3 * pow(c4, 2) -
          4 * a1 * pow(a4, 2) * c0 * pow(c5, 2) +
          8 * a1 * pow(a4, 2) * pow(c0, 2) * c5 +
          16 * a1 * pow(a4, 2) * pow(c2, 2) * c3 -
          4 * a1 * pow(a5, 2) * c0 * pow(c4, 2) +
          4 * a1 * pow(a5, 2) * pow(c1, 2) * c3 -
          16 * a3 * pow(a4, 2) * c1 * pow(c2, 2) -
          8 * pow(a0, 2) * a5 * c1 * pow(c4, 2) -
          4 * pow(a1, 2) * a3 * c1 * pow(c5, 2) -
          16 * pow(a2, 2) * a3 * c1 * pow(c4, 2) -
          4 * a1 * pow(a2, 2) * c3 * pow(c5, 2) +
          4 * a1 * pow(a2, 2) * pow(c3, 2) * c5 -
          4 * a1 * pow(a3, 2) * pow(c2, 2) * c5 -
          a1 * pow(a5, 2) * pow(c0, 2) * c5 -
          4 * a1 * pow(a5, 2) * pow(c2, 2) * c3 -
          8 * a2 * pow(a3, 2) * pow(c2, 2) * c4 -
          2 * a2 * pow(a5, 2) * pow(c0, 2) * c4 +
          4 * a3 * pow(a5, 2) * c1 * pow(c2, 2) -
          2 * a4 * pow(a5, 2) * pow(c0, 2) * c2 +
          2 * pow(a0, 2) * a2 * c4 * pow(c5, 2) +
          2 * pow(a0, 2) * a4 * c2 * pow(c5, 2) +
          pow(a0, 2) * a5 * c1 * pow(c5, 2) +
          4 * pow(a2, 2) * a3 * c1 * pow(c5, 2) +
          8 * pow(a2, 2) * a4 * c2 * pow(c3, 2) +
          4 * pow(a2, 2) * a5 * c1 * pow(c3, 2) -
          4 * pow(a3, 2) * a5 * c1 * pow(c2, 2) -
          12 * a1 * pow(a5, 2) * pow(c1, 2) * c5 -
          8 * a2 * pow(a5, 2) * pow(c1, 2) * c4 -
          8 * a4 * pow(a5, 2) * pow(c1, 2) * c2 +
          8 * pow(a1, 2) * a2 * c4 * pow(c5, 2) +
          8 * pow(a1, 2) * a4 * c2 * pow(c5, 2) +
          12 * pow(a1, 2) * a5 * c1 * pow(c5, 2) +
          4 * a1 * pow(a4, 2) * c3 * pow(c5, 2) +
          4 * a1 * pow(a5, 2) * c3 * pow(c4, 2) -
          4 * a3 * pow(a4, 2) * c1 * pow(c5, 2) -
          4 * a3 * pow(a5, 2) * c1 * pow(c4, 2) -
          a1 * pow(a5, 2) * pow(c3, 2) * c5 +
          2 * a2 * pow(a3, 2) * c4 * pow(c5, 2) -
          2 * a2 * pow(a5, 2) * pow(c3, 2) * c4 -
          2 * a4 * pow(a5, 2) * c2 * pow(c3, 2) +
          2 * pow(a3, 2) * a4 * c2 * pow(c5, 2) +
          pow(a3, 2) * a5 * c1 * pow(c5, 2) +
          8 * a0 * a1 * a3 * c0 * pow(c1, 2) -
          2 * a0 * a1 * a3 * pow(c0, 2) * c3 -
          8 * a0 * a1 * a5 * c0 * pow(c1, 2) +
          8 * a0 * a1 * a2 * c2 * pow(c3, 2) -
          16 * a0 * a2 * a4 * c0 * pow(c2, 2) -
          4 * a0 * a1 * a3 * c0 * pow(c5, 2) +
          2 * a0 * a1 * a3 * pow(c0, 2) * c5 -
          8 * a0 * a1 * a3 * pow(c2, 2) * c3 +
          8 * a0 * a1 * a4 * pow(c0, 2) * c4 -
          2 * a0 * a1 * a5 * c0 * pow(c3, 2) +
          2 * a0 * a1 * a5 * pow(c0, 2) * c3 -
          4 * a0 * a2 * a3 * pow(c0, 2) * c4 +
          4 * a0 * a2 * a4 * c0 * pow(c3, 2) -
          4 * a0 * a2 * a4 * pow(c0, 2) * c3 -
          4 * a0 * a3 * a4 * pow(c0, 2) * c2 +
          2 * a0 * a3 * a5 * pow(c0, 2) * c1 +
          32 * a1 * a2 * a3 * pow(c1, 2) * c2 -
          8 * a0 * a1 * a3 * pow(c1, 2) * c5 +
          8 * a0 * a1 * a5 * c0 * pow(c4, 2) -
          8 * a0 * a1 * a5 * pow(c1, 2) * c3 -
          16 * a0 * a3 * a4 * pow(c1, 2) * c2 -
          64 * a1 * a2 * a4 * c1 * pow(c2, 2) -
          8 * a1 * a3 * a5 * c0 * pow(c1, 2) +
          16 * a2 * a3 * a4 * c0 * pow(c1, 2) +
          8 * a0 * a1 * a3 * pow(c2, 2) * c5 +
          2 * a0 * a1 * a5 * c0 * pow(c5, 2) -
          2 * a0 * a1 * a5 * pow(c0, 2) * c5 +
          16 * a0 * a2 * a3 * pow(c2, 2) * c4 -
          4 * a0 * a2 * a4 * c0 * pow(c5, 2) +
          4 * a0 * a2 * a4 * pow(c0, 2) * c5 +
          16 * a0 * a2 * a4 * pow(c2, 2) * c3 +
          4 * a0 * a2 * a5 * pow(c0, 2) * c4 +
          4 * a0 * a4 * a5 * pow(c0, 2) * c2 -
          32 * a1 * a2 * a5 * pow(c1, 2) * c2 -
          8 * a1 * a3 * a5 * c0 * pow(c2, 2) +
          16 * a2 * a3 * a4 * c0 * pow(c2, 2) -
          2 * a0 * a1 * a3 * c3 * pow(c5, 2) +
          16 * a0 * a1 * a5 * pow(c1, 2) * c5 +
          16 * a0 * a2 * a5 * pow(c1, 2) * c4 +
          2 * a1 * a3 * a5 * pow(c0, 2) * c3 -
          4 * a2 * a3 * a4 * pow(c0, 2) * c3 -
          16 * a2 * a4 * a5 * c0 * pow(c1, 2) -
          8 * a0 * a1 * a5 * c3 * pow(c4, 2) +
          8 * a0 * a3 * a5 * c1 * pow(c4, 2) -
          16 * a1 * a2 * a4 * c1 * pow(c5, 2) -
          8 * a1 * a2 * a5 * c2 * pow(c3, 2) -
          2 * a0 * a1 * a5 * c3 * pow(c5, 2) +
          2 * a0 * a1 * a5 * pow(c3, 2) * c5 -
          4 * a0 * a2 * a3 * c4 * pow(c5, 2) +
          4 * a0 * a2 * a4 * c3 * pow(c5, 2) -
          4 * a0 * a2 * a4 * pow(c3, 2) * c5 +
          4 * a0 * a2 * a5 * pow(c3, 2) * c4 -
          4 * a0 * a3 * a4 * c2 * pow(c5, 2) -
          2 * a0 * a3 * a5 * c1 * pow(c5, 2) +
          4 * a0 * a4 * a5 * c2 * pow(c3, 2) -
          2 * a1 * a3 * a5 * c0 * pow(c5, 2) +
          4 * a1 * a3 * a5 * pow(c0, 2) * c5 +
          8 * a1 * a3 * a5 * pow(c2, 2) * c3 -
          8 * a1 * a4 * a5 * pow(c0, 2) * c4 +
          4 * a2 * a3 * a4 * c0 * pow(c5, 2) -
          8 * a2 * a3 * a4 * pow(c0, 2) * c5 -
          16 * a2 * a3 * a4 * pow(c2, 2) * c3 -
          4 * a2 * a4 * a5 * c0 * pow(c3, 2) +
          8 * a2 * a4 * a5 * pow(c0, 2) * c3 +
          8 * a1 * a3 * a5 * pow(c1, 2) * c5 -
          16 * a2 * a3 * a4 * pow(c1, 2) * c5 +
          16 * a3 * a4 * a5 * pow(c1, 2) * c2 +
          4 * a2 * a4 * a5 * pow(c0, 2) * c5 +
          2 * a1 * a3 * a5 * c3 * pow(c5, 2) -
          4 * a2 * a3 * a4 * c3 * pow(c5, 2) +
          16 * a2 * a4 * a5 * pow(c1, 2) * c5 +
          4 * a2 * a4 * a5 * pow(c3, 2) * c5 -
          8 * a0 * pow(a1, 2) * c0 * c1 * c3 +
          8 * a0 * pow(a1, 2) * c0 * c1 * c5 +
          2 * pow(a0, 2) * a3 * c0 * c1 * c3 +
          16 * a0 * pow(a2, 2) * c0 * c2 * c4 -
          8 * a2 * pow(a3, 2) * c0 * c1 * c2 +
          2 * a0 * pow(a3, 2) * c0 * c1 * c5 -
          4 * a0 * pow(a3, 2) * c0 * c2 * c4 +
          4 * a0 * pow(a5, 2) * c0 * c1 * c3 -
          2 * pow(a0, 2) * a1 * c0 * c3 * c5 +
          4 * pow(a0, 2) * a2 * c0 * c3 * c4 -
          2 * pow(a0, 2) * a3 * c0 * c1 * c5 +
          4 * pow(a0, 2) * a3 * c0 * c2 * c4 -
          8 * pow(a0, 2) * a4 * c0 * c1 * c4 +
          4 * pow(a0, 2) * a4 * c0 * c2 * c3 -
          2 * pow(a0, 2) * a5 * c0 * c1 * c3 -
          32 * pow(a1, 2) * a2 * c1 * c2 * c3 +
          8 * pow(a2, 2) * a3 * c0 * c1 * c3 +
          8 * a0 * pow(a1, 2) * c1 * c3 * c5 -
          16 * a0 * pow(a1, 2) * c2 * c3 * c4 -
          8 * a0 * pow(a4, 2) * c0 * c1 * c5 +
          64 * a1 * pow(a2, 2) * c1 * c2 * c4 +
          16 * pow(a1, 2) * a2 * c0 * c3 * c4 +
          8 * pow(a1, 2) * a3 * c0 * c1 * c5 +
          8 * pow(a1, 2) * a5 * c0 * c1 * c3 +
          8 * a0 * pow(a2, 2) * c1 * c3 * c5 -
          16 * a0 * pow(a2, 2) * c2 * c3 * c4 -
          2 * a0 * pow(a5, 2) * c0 * c1 * c5 +
          4 * a0 * pow(a5, 2) * c0 * c2 * c4 -
          4 * pow(a0, 2) * a2 * c0 * c4 * c5 -
          4 * pow(a0, 2) * a4 * c0 * c2 * c5 +
          2 * pow(a0, 2) * a5 * c0 * c1 * c5 -
          4 * pow(a0, 2) * a5 * c0 * c2 * c4 +
          32 * pow(a1, 2) * a2 * c1 * c2 * c5 -
          16 * pow(a2, 2) * a3 * c0 * c2 * c4 -
          16 * pow(a2, 2) * a4 * c0 * c2 * c3 -
          8 * pow(a2, 2) * a5 * c0 * c1 * c3 +
          16 * a0 * pow(a1, 2) * c2 * c4 * c5 +
          2 * a3 * pow(a5, 2) * c0 * c1 * c3 -
          2 * pow(a0, 2) * a3 * c1 * c3 * c5 +
          4 * pow(a0, 2) * a3 * c2 * c3 * c4 -
          16 * pow(a1, 2) * a4 * c0 * c2 * c5 -
          16 * pow(a1, 2) * a5 * c0 * c1 * c5 -
          8 * a1 * pow(a4, 2) * c0 * c3 * c5 +
          16 * a1 * pow(a5, 2) * c1 * c2 * c4 +
          8 * a2 * pow(a3, 2) * c1 * c2 * c5 +
          8 * a3 * pow(a4, 2) * c0 * c1 * c5 +
          4 * a0 * pow(a3, 2) * c2 * c4 * c5 +
          2 * a0 * pow(a5, 2) * c1 * c3 * c5 -
          4 * a0 * pow(a5, 2) * c2 * c3 * c4 +
          2 * a1 * pow(a5, 2) * c0 * c3 * c5 -
          4 * a2 * pow(a3, 2) * c0 * c4 * c5 +
          4 * a2 * pow(a5, 2) * c0 * c3 * c4 +
          2 * a3 * pow(a5, 2) * c0 * c1 * c5 -
          4 * a3 * pow(a5, 2) * c0 * c2 * c4 +
          4 * a4 * pow(a5, 2) * c0 * c2 * c3 -
          8 * pow(a0, 2) * a3 * c2 * c4 * c5 +
          8 * pow(a0, 2) * a4 * c1 * c4 * c5 -
          4 * pow(a0, 2) * a5 * c1 * c3 * c5 +
          8 * pow(a0, 2) * a5 * c2 * c3 * c4 -
          8 * pow(a2, 2) * a3 * c1 * c3 * c5 +
          16 * pow(a2, 2) * a3 * c2 * c3 * c4 -
          4 * pow(a3, 2) * a4 * c0 * c2 * c5 -
          2 * pow(a3, 2) * a5 * c0 * c1 * c5 +
          4 * pow(a3, 2) * a5 * c0 * c2 * c4 -
          16 * pow(a1, 2) * a2 * c3 * c4 * c5 -
          8 * pow(a1, 2) * a5 * c1 * c3 * c5 +
          16 * pow(a1, 2) * a5 * c2 * c3 * c4 -
          4 * pow(a0, 2) * a5 * c2 * c4 * c5 -
          2 * a3 * pow(a5, 2) * c1 * c3 * c5 +
          4 * a3 * pow(a5, 2) * c2 * c3 * c4 -
          16 * pow(a1, 2) * a5 * c2 * c4 * c5 -
          4 * pow(a3, 2) * a5 * c2 * c4 * c5 +
          16 * a0 * a1 * a2 * c0 * c1 * c4 - 8 * a0 * a1 * a2 * c0 * c2 * c3 -
          16 * a0 * a1 * a4 * c0 * c1 * c2 + 8 * a0 * a2 * a3 * c0 * c1 * c2 +
          8 * a0 * a1 * a2 * c0 * c2 * c5 - 8 * a0 * a2 * a5 * c0 * c1 * c2 +
          16 * a0 * a1 * a3 * c1 * c2 * c4 + 16 * a0 * a1 * a4 * c1 * c2 * c3 -
          16 * a1 * a2 * a3 * c0 * c1 * c4 - 16 * a1 * a2 * a4 * c0 * c1 * c3 +
          4 * a0 * a1 * a3 * c0 * c3 * c5 - 8 * a0 * a1 * a4 * c0 * c3 * c4 +
          8 * a0 * a3 * a4 * c0 * c1 * c4 - 4 * a0 * a3 * a5 * c0 * c1 * c3 -
          16 * a0 * a1 * a2 * c1 * c4 * c5 - 8 * a0 * a1 * a2 * c2 * c3 * c5 -
          16 * a0 * a1 * a5 * c1 * c2 * c4 - 8 * a0 * a2 * a3 * c1 * c2 * c5 -
          32 * a0 * a2 * a4 * c1 * c2 * c4 + 16 * a1 * a2 * a4 * c0 * c1 * c5 +
          32 * a1 * a2 * a4 * c0 * c2 * c4 + 8 * a1 * a2 * a5 * c0 * c2 * c3 +
          16 * a1 * a4 * a5 * c0 * c1 * c2 + 8 * a2 * a3 * a5 * c0 * c1 * c2 -
          8 * a0 * a1 * a4 * c0 * c4 * c5 + 8 * a0 * a2 * a3 * c0 * c4 * c5 -
          8 * a0 * a2 * a5 * c0 * c3 * c4 + 8 * a0 * a3 * a4 * c0 * c2 * c5 +
          8 * a0 * a4 * a5 * c0 * c1 * c4 - 8 * a0 * a4 * a5 * c0 * c2 * c3 +
          8 * a0 * a2 * a5 * c1 * c2 * c5 - 8 * a1 * a2 * a5 * c0 * c2 * c5 +
          16 * a1 * a2 * a3 * c1 * c4 * c5 + 16 * a1 * a2 * a4 * c1 * c3 * c5 -
          32 * a1 * a2 * a4 * c2 * c3 * c4 - 16 * a1 * a3 * a5 * c1 * c2 * c4 -
          16 * a1 * a4 * a5 * c1 * c2 * c3 + 32 * a2 * a3 * a4 * c1 * c2 * c4 +
          8 * a0 * a1 * a4 * c3 * c4 * c5 - 8 * a0 * a3 * a4 * c1 * c4 * c5 +
          4 * a0 * a3 * a5 * c1 * c3 * c5 - 8 * a0 * a3 * a5 * c2 * c3 * c4 -
          4 * a1 * a3 * a5 * c0 * c3 * c5 + 8 * a1 * a4 * a5 * c0 * c3 * c4 +
          8 * a2 * a3 * a4 * c0 * c3 * c5 - 8 * a3 * a4 * a5 * c0 * c1 * c4 +
          8 * a1 * a2 * a5 * c2 * c3 * c5 - 8 * a2 * a3 * a5 * c1 * c2 * c5 +
          8 * a0 * a3 * a5 * c2 * c4 * c5 - 8 * a0 * a4 * a5 * c1 * c4 * c5 +
          8 * a1 * a4 * a5 * c0 * c4 * c5 - 8 * a2 * a4 * a5 * c0 * c3 * c5 -
          8 * a1 * a4 * a5 * c3 * c4 * c5 + 8 * a3 * a4 * a5 * c1 * c4 * c5;

  return sols;
}