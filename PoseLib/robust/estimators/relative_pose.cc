// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "relative_pose.h"

#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gen_relpose_5p1pt.h"
#include "PoseLib/solvers/homo3f.h"
#include "PoseLib/solvers/homography_4pt.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "PoseLib/solvers/relpose_6pt_focal.h"
#include "PoseLib/solvers/relpose_7pt.h"

#include <iostream>

namespace poselib {

void RelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_5pt(x1s, x2s, models);
}

double RelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_sampson_msac_score(pose, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void RelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);

    if (num_inl <= 5) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
        }
    }
    refine_relpose(x1_inlier, x2_inlier, pose, bundle_opt);
}

void ThreeViewRelativePoseEstimator::generate_models(std::vector<ThreeViewCameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1n[k] = x1[sample[k]].homogeneous().normalized();
        x2n[k] = x2[sample[k]].homogeneous().normalized();
    }

    for (size_t k = 0; k < sample_sz_13; ++k) {
        x1s[k] = x1[sample[k]].homogeneous();
        x2s[k] = x2[sample[k]].homogeneous();
        x3s[k] = x3[sample[k]].homogeneous().normalized();
    }

    estimate_models(models);
}

void ThreeViewRelativePoseEstimator::estimate_models(std::vector<ThreeViewCameraPose> *models) {
    std::vector<CameraPose> models12;
    relpose_5pt(x1n, x2n, &models12);

    std::vector<Point3D> triangulated_12(3);

    for (CameraPose pose12 : models12){
        for (size_t i = 0; i < sample_sz_13; i++){
            triangulated_12[i] = triangulate(pose12, x1s[i], x2s[i]);
        }

        std::vector<CameraPose> models13;
        p3p(x3s, triangulated_12, &models13);

        for (CameraPose pose13 : models13){
            ThreeViewCameraPose three_view_pose = ThreeViewCameraPose(pose12, pose13);
            if (opt.threeview_check){
                size_t inlier_4p_13 = 0;
                size_t inlier_4p_23 = 0;
                std::vector<Point2D> x1c = {x1[sample[3]]};
                std::vector<Point2D> x2c = {x2[sample[3]]};
                std::vector<Point2D> x3c = {x3[sample[3]]};
                compute_sampson_msac_score(three_view_pose.pose13, x1c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_13);
                compute_sampson_msac_score(three_view_pose.pose23(), x2c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_23);
                if (inlier_4p_13 + inlier_4p_23 < 2) {
                    continue;
                }
            }

            if (opt.inner_refine > 0) {
                inner_refine(&three_view_pose);
            }
            models->emplace_back(three_view_pose);
        }
    }
}

void ThreeViewRelativePoseEstimator::inner_refine(ThreeViewCameraPose *three_view_pose) const {
    std::vector<Point2D> x1r, x2r, x3r;
    x1r.resize(4);
    x2r.resize(4);
    x3r.resize(4);
    for (size_t k = 0; k < 4; ++k) {
        x1r[k] = x1[sample[k]];
        x2r[k] = x2[sample[k]];
        x3r[k] = x3[sample[k]];
    }

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::CAUCHY;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.inner_refine;
    refine_3v_relpose(x1r, x2r, x3r, three_view_pose, bundle_opt);
}

double ThreeViewRelativePoseEstimator::score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const {
    size_t inlier_count12, inlier_count13, inlier_count23;
    // TODO: calc inliers better w/o redundant computation

    double score12 = compute_sampson_msac_score(three_view_pose.pose12, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count12);
    double score13 = compute_sampson_msac_score(three_view_pose.pose13, x1, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count13);
    double score23 = compute_sampson_msac_score(three_view_pose.pose23(), x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count23);

    std::vector<char> inliers;
    *inlier_count = get_inliers(three_view_pose, x1, x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inliers);
    return score12 + score13 + score23;
}

void ThreeViewRelativePoseEstimator::refine_model(ThreeViewCameraPose *pose) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;
//    bundle_opt.verbose = true;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, x3, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier, x3_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);
    x3_inlier.reserve(num_inl);

    if (num_inl <= 4) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
            x3_inlier.push_back(x3[pt_k]);
        }
    }
    refine_3v_relpose(x1_inlier, x2_inlier, x3_inlier, pose, bundle_opt);
}

void ThreeViewSharedFocalRelativePoseEstimator::generate_models(std::vector<ImageTriplet> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1n[k] = x1[sample[k]].homogeneous().normalized();
        x2n[k] = x2[sample[k]].homogeneous().normalized();
    }


    if (opt.use_homography and opt.use_p3p) {
        estimate_homography_p3p(models);
        return;
    }
    if (opt.use_homography) {
        estimate_homography(models);
        return;
    }
    estimate_relpose(models);
}

void ThreeViewSharedFocalRelativePoseEstimator::estimate_homography(std::vector<ImageTriplet> *models){
    Eigen::Matrix3d H12;
    int sols_1 = homography_4pt(x1n, x2n, &H12, true);
    if (sols_1 == 0)
        return;

    for (size_t k = 0; k < sample_sz; ++k) {
        x3n[k] = x3[sample[k]].homogeneous().normalized();
    }

    Eigen::Matrix3d H13;
    int sols_2 = homography_4pt(x1n, x3n, &H13, true);
    if (sols_2 ==  0)
        return;

    std::vector<double> focals = solver_homo3f(H12, H13);
    focals = solver_homo3f(H12, H13);

    //    std::cout << "Focals size: " << focals.size() << std::endl;

    if (focals.empty())
        return;

    models->reserve(focals.size() * 4);

    for (double focal : focals) {
        std::vector<CameraPose> poses12(2), poses13_unscaled(2);
        std::vector<Eigen::Matrix<double, 3, 1>> normals12(2), normals13(2);

        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, focal);
        Eigen::DiagonalMatrix<double, 3> K(focal, focal, 1.0);
        Eigen::Matrix3d HH12 = (K_inv * H12 * K), HH13 = (K_inv * H13 * K);

        motion_from_homography_svd(HH12, poses12, normals12);
        motion_from_homography_svd(HH13, poses13_unscaled, normals13);

        Camera camera("SIMPLE_PINHOLE", {focal, 0, 0}, -1, -1);

        for (const CameraPose& pose12: poses12){
            Point3D X = triangulate(ImagePair(pose12, camera, camera), x1[sample[0]].homogeneous(), x2[sample[0]].homogeneous());

            if (X(2) < 0.0)
                continue;

            for (CameraPose pose13 : poses13_unscaled){
                // We recover scale of the third view
                const Eigen::Matrix3d R = pose13.R();
                const Point3D t = pose13.t;
                Point2D m = x3[sample[0]];
                double scale = (-X(0)*focal*R(0, 0) + X(0)*m(0)*R(2, 0) - X(1)*focal*R(0, 1)
                                +X(1)*m(0)*R(2, 1) - X(2)*focal*R(0, 2) + X(2)*m(0)*R(2, 2))/(focal*t(0) - m(0)*t(2));
//                double scale2 = (-X(0)*focal*R(1, 0) + X(0)*m(1)*R(2, 0) - X(1)*focal*R(1, 1) + X(1)*m(1)*R(2, 1) - X(2)*focal*R(1, 2) + X(2)*m(1)*R(2, 2))/(focal*t(1) - m(1)*t(2));
//                std::cout << "scale1: " << scale1 << " scale2: " << scale2 << std::endl;
                pose13.t *= scale;

                ImageTriplet image_triplet = ImageTriplet(ThreeViewCameraPose(pose12, pose13), camera);
                models->emplace_back(image_triplet);
            }
        }
    }
}

void ThreeViewSharedFocalRelativePoseEstimator::estimate_homography_p3p(std::vector<ImageTriplet> *models){
    Eigen::Matrix3d H12;
    int sols_1 = homography_4pt(x1n, x2n, &H12, true);
    if (sols_1 == 0)
        return;

    for (size_t k = 0; k < sample_sz; ++k) {
        x3n[k] = x3[sample[k]].homogeneous().normalized();
    }

    Eigen::Matrix3d H13;
    int sols_2 = homography_4pt(x1n, x3n, &H13, true);
    if (sols_2 ==  0)
        return;

    std::vector<double> focals = solver_homo3f(H12, H13);
    focals = solver_homo3f(H12, H13);

    //    std::cout << "Focals size: " << focals.size() << std::endl;

    if (focals.empty())
        return;

    for (size_t k = 0; k < sample_sz_13; ++k) {
        x1s[k] = x1[sample[k]].homogeneous();
        x2s[k] = x2[sample[k]].homogeneous();
    }

    models->reserve(focals.size() * 4);

    for (double focal : focals) {
        std::vector<CameraPose> poses12(2);
        std::vector<Eigen::Matrix<double, 3, 1>> normals12(2);

        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, focal);
        Eigen::DiagonalMatrix<double, 3> K(focal, focal, 1.0);
        Eigen::Matrix3d HH12 = (K_inv * H12 * K);

        motion_from_homography_svd(HH12, poses12, normals12);

        Camera camera("SIMPLE_PINHOLE", {focal, 0, 0}, -1, -1);

        std::vector<Point3D> triangulated_12(3);

        for (CameraPose pose12 : poses12) {
            ImagePair pair12(pose12, camera, camera);

            for (size_t i = 0; i < sample_sz_13; i++) {
                triangulated_12[i] = triangulate(pair12, x1s[i], x2s[i]);
            }

            for (size_t k = 0; k < sample_sz_13; ++k) {
                Eigen::Vector2d x;
                camera.unproject(x3[sample[k]], &x);
                x3s[k] = x.homogeneous().normalized();
            }

            std::vector<CameraPose> models13;

            p3p(x3s, triangulated_12, &models13);

            for (CameraPose pose13 : models13) {
                ImageTriplet image_triplet = ImageTriplet(ThreeViewCameraPose(pose12, pose13), camera);
                models->emplace_back(image_triplet);
            }
        }
    }
}

void ThreeViewSharedFocalRelativePoseEstimator::estimate_relpose(std::vector<ImageTriplet> *models){
    std::vector<ImagePair> models12;
    relpose_6pt_focal(x1n, x2n, &models12);

    std::vector<Point3D> triangulated_12(3);

    for (size_t k = 0; k < sample_sz_13; ++k) {
        x1s[k] = x1[sample[k]].homogeneous();
        x2s[k] = x2[sample[k]].homogeneous();
    }

    for (ImagePair pair12 : models12){
        for (size_t i = 0; i < sample_sz_13; i++){
            triangulated_12[i] = triangulate(pair12, x1s[i], x2s[i]);
        }

        for (size_t k = 0; k < sample_sz_13; ++k) {
            Eigen::Vector2d x;
            pair12.camera1.unproject(x3[sample[k]], &x);
            x3s[k] = x.homogeneous().normalized();
        }

        std::vector<CameraPose> models13;

        p3p(x3s, triangulated_12, &models13);

        for (CameraPose pose13 : models13){
            ImageTriplet image_triplet = ImageTriplet(ThreeViewCameraPose(pair12.pose, pose13), pair12.camera1);
            if (opt.threeview_check){
                size_t inlier_4p_13 = 0;
                size_t inlier_4p_23 = 0;
                std::vector<Point2D> x1c = {x1[sample[3]]};
                std::vector<Point2D> x2c = {x2[sample[3]]};
                std::vector<Point2D> x3c = {x3[sample[3]]};

                Eigen::DiagonalMatrix<double, 3> K_inv(1, 1, image_triplet.camera.focal());

                Eigen::Matrix3d F13, F23;
                essential_from_motion(image_triplet.poses.pose13, &F13);
                essential_from_motion(image_triplet.poses.pose23(), &F23);
                F13 = K_inv * F13 * K_inv;
                F23 = K_inv * F23 * K_inv;

                compute_sampson_msac_score(F13, x1c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_13);
                compute_sampson_msac_score(F23, x2c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_23);
                if (inlier_4p_13 + inlier_4p_23 < 2) {
                    continue;
                }
            }

            if (opt.inner_refine > 0) {
                inner_refine(&image_triplet);
            }
            models->emplace_back(image_triplet);
        }
    }
}

void ThreeViewSharedFocalRelativePoseEstimator::inner_refine(ImageTriplet *image_triplet) const {
    std::vector<Point2D> x1r, x2r, x3r;
    x1r.resize(4);
    x2r.resize(4);
    x3r.resize(4);
    for (size_t k = 0; k < 4; ++k) {
        x1r[k] = x1[sample[k]];
        x2r[k] = x2[sample[k]];
        x3r[k] = x3[sample[k]];
    }

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::CAUCHY;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.inner_refine;
    refine_3v_shared_focal_relpose(x1r, x2r, x3r, image_triplet, bundle_opt);
}

double ThreeViewSharedFocalRelativePoseEstimator::score_model(const ImageTriplet &image_triplet, size_t *inlier_count) const {
    Eigen::DiagonalMatrix<double, 3> K_inv(1, 1, image_triplet.camera.focal());
    Eigen::Matrix3d F12, F13, F23;
    essential_from_motion(image_triplet.poses.pose12, &F12);
    essential_from_motion(image_triplet.poses.pose13, &F13);
    essential_from_motion(image_triplet.poses.pose23(), &F23);
    F12 = K_inv * F12 * K_inv;
    F13 = K_inv * F13 * K_inv;
    F23 = K_inv * F23 * K_inv;

    std::vector<char> inliers12, inliers13, inliers23;

    double sq_t = opt.max_epipolar_error * opt.max_epipolar_error;

    double score12 = compute_sampson_msac_score(F12, x1, x2, sq_t, &inliers12);
    double score13 = compute_sampson_msac_score(F13, x1, x3, sq_t, &inliers13);
    double score23 = compute_sampson_msac_score(F23, x2, x3, sq_t, &inliers23);

    *inlier_count = 0;

    bool val;
    for (size_t i = 0; i < x1.size(); i++){
        val = (inliers12[i] and inliers13[i]) and inliers23[i];
        if (val)
            (*inlier_count)++;
    }

    return score12 + score13 + score23;
}

void ThreeViewSharedFocalRelativePoseEstimator::refine_model(ImageTriplet *image_triplet) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;
//    bundle_opt.verbose = true;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl =
        get_inliers(*image_triplet, x1, x2, x3, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers, true);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier, x3_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);
    x3_inlier.reserve(num_inl);

    if (num_inl <= 4) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
            x3_inlier.push_back(x3[pt_k]);
        }
    }

    refine_3v_shared_focal_relpose(x1_inlier, x2_inlier, x3_inlier, image_triplet, bundle_opt);
}

void ThreeViewSharedFocalUnscaledRelativePoseEstimator::generate_models(std::vector<ImageTriplet> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1n[k] = x1[sample[k]].homogeneous().normalized();
        x2n[k] = x2[sample[k]].homogeneous().normalized();
    }

    if (opt.use_homography){
        estimate_models_homography(models);
        return;
    }
    estimate_models_relpose(models);
}

void ThreeViewSharedFocalUnscaledRelativePoseEstimator::estimate_models_relpose(std::vector<ImageTriplet> *models) {
    std::vector<ImagePair> models12;
    relpose_6pt_focal(x1n, x2n, &models12);

    for (ImagePair pair12 : models12){
        for (size_t i = 0; i < sample_sz_13; ++i){
            Point2D xu1, xu3;
            pair12.camera1.unproject(x1[sample[i]], &xu1);
            pair12.camera1.unproject(x3[sample[i]], &xu3);
            x1s[i] = xu1.homogeneous().normalized();
            x3s[i] = xu3.homogeneous().normalized();
        }

        std::vector<CameraPose> models13;
        relpose_5pt(x1s, x3s, &models13);
        for (CameraPose pose13 : models13){
            ImageTriplet image_triplet = ImageTriplet(ThreeViewCameraPose(pair12.pose, pose13), pair12.camera1);
            models->emplace_back(image_triplet);
        }
    }
}

void ThreeViewSharedFocalUnscaledRelativePoseEstimator::estimate_models_homography(std::vector<ImageTriplet> *models) {
    Eigen::Matrix3d H12;
    int sols_1 = homography_4pt(x1n, x2n, &H12, true);
    if (sols_1 == 0)
        return;

    for (size_t k = 0; k < sample_sz_13; ++k) {
        x3n[k] = x3[sample[k]].homogeneous().normalized();
    }

    Eigen::Matrix3d H13;
    int sols_2 = homography_4pt(x1n, x3n, &H13, true);
    if (sols_2 ==  0)
        return;

//    size_t inlier_count12;
//    double score12 = compute_homography_msac_score(H12, x1, x2, opt.max_reproj_error * opt.max_reproj_error, &inlier_count12);
//    size_t inlier_count13;
//    double score13 = compute_homography_msac_score(H13, x1, x3, opt.max_reproj_error * opt.max_reproj_error, &inlier_count13);
//    std::cout << "H12 inliers: " << inlier_count12 << " score: " << score12 << std::endl;
//    std::cout << "H13 inliers: " << inlier_count13 << " score: " << score13 << std::endl;

    std::vector<double> focals = solver_homo3f(H12, H13);
    focals = solver_homo3f(H12, H13);

//    std::cout << "Focals size: " << focals.size() << std::endl;

    models->reserve(focals.size() * 4);

    for (size_t i = 0; i < focals.size(); ++i) {
        double focal = focals[i];
//        std::cout << "Focal: " << focal << std::endl;
        std::vector<CameraPose> poses12, poses13;
        std::vector<Eigen::Matrix<double, 3, 1>> normals12, normals13;
        poses12.reserve(2);
        poses13.reserve(2);
        normals12.reserve(2);
        normals13.reserve(2);

        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, focal);
        Eigen::DiagonalMatrix<double, 3> K(focal, focal, 1.0);
        Eigen::Matrix3d HH12 = (K_inv * H12 * K), HH13 = (K_inv * H13 * K);

        motion_from_homography_svd(HH12, poses12, normals12);
        motion_from_homography_svd(HH13, poses13, normals13);

        Camera camera("SIMPLE_PINHOLE", {focal, 0, 0}, -1, -1);

        for (const CameraPose& pose12: poses12){
            for (const CameraPose& pose13: poses13){
                models->emplace_back(ThreeViewCameraPose(pose12, pose13), camera);
            }
        }
    }
}

double ThreeViewSharedFocalUnscaledRelativePoseEstimator::score_model(const ImageTriplet &image_triplet,
                                                                      size_t *inlier_count) const {
    size_t inlier_count12, inlier_count13;

    Eigen::DiagonalMatrix<double, 3> K_inv(1, 1, image_triplet.camera.focal());
    Eigen::Matrix3d F12, F13;
    essential_from_motion(image_triplet.poses.pose12, &F12);
    essential_from_motion(image_triplet.poses.pose13, &F13);
    F12 = K_inv * F12 * K_inv;
    F13 = K_inv * F13 * K_inv;
    double score12 = compute_sampson_msac_score(F12, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count12);
    double score13 = compute_sampson_msac_score(F13, x1, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count13);

    std::vector<char> inliers;
    *inlier_count =
        get_inliers(image_triplet, x1, x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inliers, false);

//    std::cout << "Model f: " << image_triplet.camera.focal() << " score: " << score12 + score13 << " inliers: " << *inlier_count << std::endl;

    return score12 + score13;
}

void ThreeViewSharedFocalUnscaledRelativePoseEstimator::refine_model(ImageTriplet *image_triplet) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;
//    bundle_opt.verbose = true;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl =
        get_inliers(*image_triplet, x1, x2, x3, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers, false);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier, x3_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);
    x3_inlier.reserve(num_inl);

    if (num_inl <= 4) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
            x3_inlier.push_back(x3[pt_k]);
        }
    }

    refine_3v_shared_focal_unscaled_relpose(x1_inlier, x2_inlier, x3_inlier, image_triplet, bundle_opt);
}

void SharedFocalRelativePoseEstimator::generate_models(ImagePairVector *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_6pt_focal(x1s, x2s, models);
}

double SharedFocalRelativePoseEstimator::score_model(const ImagePair &image_pair, size_t *inlier_count) const {
    Eigen::Matrix3d K_inv;
    K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera1.focal();
    // K_inv << 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d E;
    essential_from_motion(image_pair.pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void SharedFocalRelativePoseEstimator::refine_model(ImagePair *image_pair) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    Eigen::Matrix3d K_inv;
    // K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, calib_pose->camera.focal();
    K_inv << 1.0 / image_pair->camera1.focal(), 0.0, 0.0, 0.0, 1.0 / image_pair->camera1.focal(), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d E;
    essential_from_motion(image_pair->pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(F, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);

    if (num_inl <= 6) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
        }
    }

    refine_shared_focal_relpose(x1_inlier, x2_inlier, image_pair, bundle_opt);
}

void GeneralizedRelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    // TODO replace by general 6pt solver?

    bool done = false;
    int pair0 = 0, pair1 = 1;
    while (!done) {
        pair0 = random_int(rng) % matches.size();
        if (matches[pair0].x1.size() < 5)
            continue;

        pair1 = random_int(rng) % matches.size();
        if (pair0 == pair1 || matches[pair1].x1.size() == 0)
            continue;

        done = true;
    }

    // Sample 5 points from the first camera pair
    CameraPose pose1 = rig1_poses[matches[pair0].cam_id1];
    CameraPose pose2 = rig2_poses[matches[pair0].cam_id2];
    Eigen::Vector3d p1 = pose1.center();
    Eigen::Vector3d p2 = pose2.center();
    draw_sample(5, matches[pair0].x1.size(), &sample, rng);
    for (size_t k = 0; k < 5; ++k) {
        x1s[k] = pose1.derotate(matches[pair0].x1[sample[k]].homogeneous().normalized());
        p1s[k] = p1;
        x2s[k] = pose2.derotate(matches[pair0].x2[sample[k]].homogeneous().normalized());
        p2s[k] = p2;
    }

    // Sample one point from the second camera pair
    pose1 = rig1_poses[matches[pair1].cam_id1];
    pose2 = rig2_poses[matches[pair1].cam_id2];
    p1 = pose1.center();
    p2 = pose2.center();
    size_t ind = random_int(rng) % matches[pair1].x1.size();
    x1s[5] = pose1.derotate(matches[pair1].x1[ind].homogeneous().normalized());
    p1s[5] = p1;
    x2s[5] = pose2.derotate(matches[pair1].x2[ind].homogeneous().normalized());
    p2s[5] = p2;

    gen_relpose_5p1pt(p1s, x1s, p2s, x2s, models);
}

double GeneralizedRelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    *inlier_count = 0;
    double cost = 0;
    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose.t);
        pose2.q = quat_multiply(pose2.q, pose.q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        size_t local_inlier_count = 0;
        cost += compute_sampson_msac_score(relpose, m.x1, m.x2, opt.max_epipolar_error * opt.max_epipolar_error,
                                           &local_inlier_count);
        *inlier_count += local_inlier_count;
    }

    return cost;
}

void GeneralizedRelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    std::vector<PairwiseMatches> inlier_matches;
    inlier_matches.resize(matches.size());

    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose->t);
        pose2.q = quat_multiply(pose2.q, pose->q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        // Compute inliers with a relaxed threshold
        std::vector<char> inliers;
        int num_inl = get_inliers(relpose, m.x1, m.x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);

        inlier_matches[match_k].cam_id1 = m.cam_id1;
        inlier_matches[match_k].cam_id2 = m.cam_id2;
        inlier_matches[match_k].x1.reserve(num_inl);
        inlier_matches[match_k].x2.reserve(num_inl);

        for (size_t k = 0; k < m.x1.size(); ++k) {
            if (inliers[k]) {
                inlier_matches[match_k].x1.push_back(m.x1[k]);
                inlier_matches[match_k].x2.push_back(m.x2[k]);
            }
        }
    }

    refine_generalized_relpose(inlier_matches, rig1_poses, rig2_poses, pose, bundle_opt);
}

void FundamentalEstimator::generate_models(std::vector<Eigen::Matrix3d> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_7pt(x1s, x2s, models);

    if (opt.real_focal_check) {
        for (int i = models->size() - 1; i >= 0; i--) {
            if (!calculate_RFC((*models)[i]))
                models->erase(models->begin() + i);
        }
    }
}

double FundamentalEstimator::score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const {
    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void FundamentalEstimator::refine_model(Eigen::Matrix3d *F) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    refine_fundamental(x1, x2, F, bundle_opt);
}

} // namespace poselib
