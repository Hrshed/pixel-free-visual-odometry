#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace basalt {

struct DirectMvRigidRegistrationResult {
  bool valid = false;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  std::vector<size_t> input_inliers;
};

bool directMvTeaserAvailable();

DirectMvRigidRegistrationResult solveDirectMvTeaser(
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& source,
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& destination,
    double noise_bound_m);

}  // namespace basalt
