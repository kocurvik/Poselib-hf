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

#ifndef POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/types.h"

namespace poselib {

class RelativePoseEstimator {
  public:
    RelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                          const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class ThreeViewRelativePoseEstimator {
  public:
    ThreeViewRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                   const std::vector<Point2D> &points2D_2, const std::vector<Point2D> &points2D_3)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), x3(points2D_3),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1n.resize(sample_sz);
        x2n.resize(sample_sz);
        x1s.resize(sample_sz_13);
        x2s.resize(sample_sz_13);
        x3s.resize(sample_sz_13);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<ThreeViewCameraPose> *models);
    void estimate_models(std::vector<ThreeViewCameraPose> *models);
    double score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const;
    void refine_model(ThreeViewCameraPose *three_view_pose) const;
    void inner_refine(ThreeViewCameraPose *three_view_pose) const;

    const size_t sample_sz = 5;
    const size_t sample_sz_13 = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1n, x2n, x1s, x2s, x3s;
    std::vector<size_t> sample;
};

class ThreeViewSharedFocalRelativePoseEstimator {
  public:
    ThreeViewSharedFocalRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                              const std::vector<Point2D> &points2D_2,
                                              const std::vector<Point2D> &points2D_3)
        : sample_sz(ransac_opt.use_homography ? 4 : 6), num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1),
          x2(points2D_2), x3(points2D_3),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations),
          tuples(opt.use_degensac ? std::vector<std::vector<int>>{{0, 1, 2, 3, 4},
                                                                  {0, 1, 2, 3, 5},
                                                                  {0, 1, 2, 4, 5},
                                                                  {0, 1, 3, 4, 5},
                                                                  {1, 2, 3, 4, 5}}
                                  : std::vector<std::vector<int>>()) {
        x1n.resize(sample_sz);
        x2n.resize(sample_sz);
        x3n.resize(sample_sz);
        x1s.resize(sample_sz_13);
        x2s.resize(sample_sz_13);
        x3s.resize(sample_sz_13);
        sample.resize(sample_sz);
        best_h_inliers = 0;
        best_degenerate_model_found = false;
    }

    void generate_models(std::vector<ImageTriplet> *models);
    void estimate_homography(std::vector<ImageTriplet> *models);
    void estimate_homography_p3p(std::vector<ImageTriplet> *models);
    void estimate_relpose(std::vector<ImageTriplet> *models);
    double score_model(const ImageTriplet &image_triplet, size_t *inlier_count) const;
    void refine_model(ImageTriplet *image_triplet) const;
    void inner_refine(ImageTriplet *image_triplet) const;

    const size_t sample_sz;
    const size_t sample_sz_13 = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1n, x2n, x3n, x1s, x2s, x3s;
    std::vector<size_t> sample;

    const std::vector<std::vector<int>> tuples;
    size_t best_h_inliers;
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;
    bool relpose_degeneracy(std::vector<ImagePair> *models);
    ImagePair best_degenerate_model;
    bool best_degenerate_model_found;
};

class ThreeViewCase2RelativePoseEstimator {
  public:
    ThreeViewCase2RelativePoseEstimator(const RansacOptions &ransac_opt, const Camera &camera3,
                                        const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                        const std::vector<Point2D> &points2D_3)
        : sample_sz(ransac_opt.use_homography ? 4 : 6), num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1),
          x2(points2D_2), x3(points2D_3), camera3(camera3),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations),
          tuples(opt.use_degensac ? std::vector<std::vector<int>>{{0, 1, 2, 3, 4},
                                                                  {0, 1, 2, 3, 5},
                                                                  {0, 1, 2, 4, 5},
                                                                  {0, 1, 3, 4, 5},
                                                                  {1, 2, 3, 4, 5}}
                                  : std::vector<std::vector<int>>()) {
        x1n.resize(sample_sz);
        x2n.resize(sample_sz);
        x3n.resize(sample_sz);
        x1s.resize(sample_sz_13);
        x2s.resize(sample_sz_13);
        x3p.resize(sample_sz_13);
        x3u.resize(num_data);
        for (size_t k = 0; k < num_data; ++k) {
            Eigen::Vector2d x;
            camera3.unproject(x3[k], &x);
            x3u[k] = x.homogeneous().normalized();
        }

        K3 = Eigen::DiagonalMatrix<double, 3>(camera3.focal(), camera3.focal(), 1.0);

        sample.resize(sample_sz);
        best_h_inliers = 0;
        best_degenerate_model_found = false;
    }

    void generate_models(std::vector<ImageTriplet> *models);
    void estimate_homography_p3p(std::vector<ImageTriplet> *models);
    void estimate_relpose(std::vector<ImageTriplet> *models);
    void estimate_relpose_onefocal(std::vector<ImageTriplet> *models);
    double score_model(const ImageTriplet &image_triplet, size_t *inlier_count) const;
    void refine_model(ImageTriplet *image_triplet) const;
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;
    bool relpose_degeneracy(std::vector<ImagePair> *models);

    const size_t sample_sz;
    const size_t sample_sz_13 = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;
    const Camera camera3;
    Eigen::DiagonalMatrix<double, 3> K3;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1n, x2n, x3n, x1s, x2s, x3p, x3u;
    std::vector<size_t> sample;

    const std::vector<std::vector<int>> tuples;
    size_t best_h_inliers;
    ImagePair best_degenerate_model;
    bool best_degenerate_model_found;
};

class ThreeViewCase3RelativePoseEstimator {
  public:
    ThreeViewCase3RelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                        const std::vector<Point2D> &points2D_2, const std::vector<Point2D> &points2D_3)
        : sample_sz(ransac_opt.use_homography ? 4 : 6), sample_sz_13(ransac_opt.use_homography ? 3 : 4),
          num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), x3(points2D_3),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations),
          tuples(opt.use_degensac ? std::vector<std::vector<int>>{{0, 1, 2, 3, 4},
                                                                  {0, 1, 2, 3, 5},
                                                                  {0, 1, 2, 4, 5},
                                                                  {0, 1, 3, 4, 5},
                                                                  {1, 2, 3, 4, 5}}
                                  : std::vector<std::vector<int>>()) {
        x1n.resize(sample_sz);
        x2n.resize(sample_sz);
        x3n.resize(sample_sz);
        x1s.resize(sample_sz_13);
        x2s.resize(sample_sz_13);
        x3p.resize(sample_sz_13);
        x3p_2d.resize(sample_sz_13);
        sample.resize(sample_sz);
        best_h_inliers = 0;
        best_degenerate_model_found = false;
    }

    void generate_models(std::vector<ImageTriplet> *models);
    void estimate_homography_p3p(std::vector<ImageTriplet> *models);
    void estimate_relpose(std::vector<ImageTriplet> *models);
    double score_model(const ImageTriplet &image_triplet, size_t *inlier_count) const;
    void refine_model(ImageTriplet *image_triplet) const;
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;
    bool relpose_degeneracy(std::vector<ImagePair> *models);

    const size_t sample_sz;
    const size_t sample_sz_13;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1n, x2n, x3n, x1s, x2s, x3p;
    std::vector<Eigen::Vector2d> x3p_2d;
    std::vector<size_t> sample;

    const std::vector<std::vector<int>> tuples;
    size_t best_h_inliers;
    ImagePair best_degenerate_model;
    bool best_degenerate_model_found;
};

class ThreeViewCase4RelativePoseEstimator {
  public:
    ThreeViewCase4RelativePoseEstimator(const RansacOptions &ransac_opt, const Camera &camera3,
                                        const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                        const std::vector<Point2D> &points2D_3)
        : sample_sz(ransac_opt.use_homography ? 4 : 6), sample_sz_13(ransac_opt.use_homography ? 3 : 4),
          num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), x3(points2D_3),
          camera3(camera3), sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations),
          tuples(opt.use_degensac ? std::vector<std::vector<int>>{{0, 1, 2, 3, 4},
                                                                  {0, 1, 2, 3, 5},
                                                                  {0, 1, 2, 4, 5},
                                                                  {0, 1, 3, 4, 5},
                                                                  {1, 2, 3, 4, 5}}
                                  : std::vector<std::vector<int>>()) {
        x1n.resize(sample_sz);
        x2n.resize(sample_sz);
        x3n.resize(sample_sz);
        x1s.resize(sample_sz_13);
        x2s.resize(sample_sz_13);
        x3s.resize(sample_sz_13);
        x3p.resize(sample_sz_13);
        x2p_2d.resize(sample_sz_13);
        sample.resize(sample_sz);
        K3 = Eigen::DiagonalMatrix<double, 3>(camera3.focal(), camera3.focal(), 1.0);
        x3u.resize(num_data);
        for (size_t k = 0; k < num_data; ++k) {
            Eigen::Vector2d x;
            camera3.unproject(x3[k], &x);
            x3u[k] = x.homogeneous().normalized();
        }
    }

    void generate_models(std::vector<ImageTriplet> *models);
    void estimate_homography_p3p(std::vector<ImageTriplet> *models);
    void estimate_relpose(std::vector<ImageTriplet> *models);
    double score_model(const ImageTriplet &image_triplet, size_t *inlier_count) const;
    void refine_model(ImageTriplet *image_triplet) const;
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;

    const size_t sample_sz;
    const size_t sample_sz_13;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;
    const Camera &camera3;
    Eigen::DiagonalMatrix<double, 3> K3;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1n, x2n, x3n, x1s, x2s, x3s, x3p, x3u;
    std::vector<Eigen::Vector2d> x2p_2d;
    std::vector<size_t> sample;

    const std::vector<std::vector<int>> tuples;
};

class ThreeViewSharedFocalUnscaledRelativePoseEstimator {
  public:
    ThreeViewSharedFocalUnscaledRelativePoseEstimator(const RansacOptions &ransac_opt,
                                                      const std::vector<Point2D> &points2D_1,
                                                      const std::vector<Point2D> &points2D_2,
                                                      const std::vector<Point2D> &points2D_3)
        : sample_sz(ransac_opt.use_homography ? 4 : 6), sample_sz_13(ransac_opt.use_homography ? 4 : 5),
          num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), x3(points2D_3),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1n.resize(sample_sz);
        x2n.resize(sample_sz);
        x3n.resize(sample_sz_13);
        x1s.resize(sample_sz_13);
        x3s.resize(sample_sz_13);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<ImageTriplet> *models);
    void estimate_models_relpose(std::vector<ImageTriplet> *models);
    void estimate_models_homography(std::vector<ImageTriplet> *models);
    double score_model(const ImageTriplet &image_triplet, size_t *inlier_count) const;
    void refine_model(ImageTriplet *image_triplet) const;

    const size_t sample_sz;
    const size_t sample_sz_13;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1n, x2n, x3n, x1s, x3s;
    std::vector<size_t> sample;
};

class SharedFocalRelativePoseEstimator {
  public:
    SharedFocalRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                     const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations),
          tuples(opt.use_degensac ? std::vector<std::vector<int>>{{0, 1, 2, 3, 4},
                                                                  {0, 1, 2, 3, 5},
                                                                  {0, 1, 2, 4, 5},
                                                                  {0, 1, 3, 4, 5},
                                                                  {1, 2, 3, 4, 5}}
                                  : std::vector<std::vector<int>>()) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
        best_h_inliers = 0;
    }

    void generate_models(ImagePairVector *models);
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;
    void refine_model(ImagePair *image_pair) const;
    int degeneracy(ImagePair *model);

    const size_t sample_sz = 6;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;

    const std::vector<std::vector<int>> tuples;
    size_t best_h_inliers;
};

class RelativeOneFocalPoseEstimator {
  public:
    RelativeOneFocalPoseEstimator(const RansacOptions &ransac_opt, const Camera &camera2,
                                  const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), camera2(camera2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        x2_unproj.resize(x2.size());

        double f2 = camera2.focal();

        for (size_t k = 0; k < x1.size(); ++k) {
            x2_unproj[k](0) = x2[k](0) / f2;
            x2_unproj[k](1) = x2[k](1) / f2;
        }

        K2_inv = Eigen::DiagonalMatrix<double, 3>(1, 1, f2);

        sample.resize(sample_sz);
    }

    void generate_models(std::vector<ImagePair> *models);
    double score_model(const ImagePair &focal_pose, size_t *inlier_count) const;
    void refine_model(ImagePair *focal_pose) const;

    const size_t sample_sz = 6;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const Camera camera2;
    Eigen::DiagonalMatrix<double, 3> K2_inv;
    std::vector<Point2D> x2_unproj;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class GeneralizedRelativePoseEstimator {
  public:
    GeneralizedRelativePoseEstimator(const RansacOptions &ransac_opt,
                                     const std::vector<PairwiseMatches> &pairwise_matches,
                                     const std::vector<CameraPose> &camera1_ext,
                                     const std::vector<CameraPose> &camera2_ext)
        : opt(ransac_opt), matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext) {
        rng = opt.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        p1s.resize(sample_sz);
        p2s.resize(sample_sz);
        sample.resize(sample_sz);

        num_data = 0;
        for (const PairwiseMatches &m : matches) {
            num_data += m.x1.size();
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 6;
    size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s, p1s, p2s;
    std::vector<size_t> sample;
};

class FundamentalEstimator {
  public:
    FundamentalEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                         const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<Eigen::Matrix3d> *models);
    double score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const;
    void refine_model(Eigen::Matrix3d *F) const;

    const size_t sample_sz = 7;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

} // namespace poselib

#endif