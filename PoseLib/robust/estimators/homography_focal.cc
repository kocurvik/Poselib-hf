//
// Created by kocur on 12-Sep-24.
//
#include "homography_focal.h"

#include "PoseLib/robust.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/solvers/homo3f.h"
#include "PoseLib/solvers/homography_4pt.h"
#include "PoseLib/types.h"

#include <Eigen/Core>
#include <iostream>
namespace poselib {

void get_homographies(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, int iterations,
                      double inlier_threshold, double distance_threshold, std::vector<Eigen::Matrix3d> *Hs) {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = distance_threshold;
    bundle_opt.max_iterations = 25;

    Hs->reserve(100);
    double sq_distance_threshold = distance_threshold * distance_threshold;

    std::vector<Point3D> x1h(x1.size()), x2h(x2.size());

    size_t min_inlier_count = std::floor(x1.size() * inlier_threshold);

    for (size_t i = 0; i < x1.size(); ++i){
        x1h[i] = x1[i].homogeneous().normalized();
        x2h[i] = x2[i].homogeneous().normalized();
    }

    RandomSampler sampler(x1.size(), 4);
    std::vector<size_t> sample(4);
    std::vector<Point3D> x1s(4), x2s(4);
    Eigen::Matrix3d H;

    for (int k = 0; k < iterations; ++k){
        sampler.generate_sample(&sample);
        for (size_t i = 0; i < 4; ++i){
            x1s[i] = x1h[sample[i]];
            x2s[i] = x2h[sample[i]];
        }

        int sol = homography_4pt(x1s, x2s, &H, true);
        if (sol == 0)
            continue;
        size_t inlier_count;
        compute_homography_msac_score(H, x1, x2, sq_distance_threshold, &inlier_count);
        if (inlier_count > min_inlier_count){
            refine_homography(x1, x2, &H, bundle_opt);
            Hs->emplace_back(H);
        }
    }
}

std::vector<double> estimate_focal_homography(std::vector<Point2D> x1_1, std::vector<Point2D> x2_1,
                                              std::vector<Point2D> x1_2, std::vector<Point2D> x3_2,
                                              const Point2D& pp, int iterations, double inlier_threshold,
                                              double distance_threshold) {
    // preprocess files

    double scale = 0.0;

    for (size_t k = 0; k < x1_1.size(); k++) {
        x1_1[k] -= pp;
        x2_1[k] -= pp;
        scale += x1_1[k].norm();
        scale += x2_1[k].norm();
    }

    for (size_t k = 0; k < x1_2.size(); k++) {
        x1_2[k] -= pp;
        x3_2[k] -= pp;
        scale += x1_2[k].norm();
        scale += x3_2[k].norm();
    }

    scale /= (2.0 * x1_1.size() + 2.0 * x1_2.size()) / std::sqrt(2);

    distance_threshold /= scale;

    for (size_t i = 0; i < x1_1.size(); i++) {
        x1_1[i] /= scale;
        x2_1[i] /= scale;
    }

    for (size_t i = 0; i < x1_2.size(); i++) {
        x1_2[i] /= scale;
        x3_2[i] /= scale;
    }

    std::vector<Eigen::Matrix3d> H12, H13;
    get_homographies(x1_1, x2_1, iterations, inlier_threshold, distance_threshold, &H12);
    get_homographies(x1_2, x3_2, iterations, inlier_threshold, distance_threshold, &H13);

    std::vector<double> focals;
    focals.reserve(iterations * iterations * 10);

    for (Eigen::Matrix3d HH12: H12){
        for (Eigen::Matrix3d HH13: H13){
            std::vector<double> fs = solver_homo3f(HH12, HH13);
            focals.insert(focals.end(), fs.begin(), fs.end());
        }
    }

    for(size_t k = 0; k < focals.size(); ++k){
        focals[k] *= scale;
    }

    return focals;
}
} // namespace poselib