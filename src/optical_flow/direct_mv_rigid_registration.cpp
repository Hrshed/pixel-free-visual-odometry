#include <basalt/optical_flow/direct_mv_rigid_registration.h>

#include <cmath>
#include <exception>

#ifdef BASALT_HAVE_TEASERPP
#include <teaser/registration.h>
#endif

namespace basalt {

bool directMvTeaserAvailable() {
#ifdef BASALT_HAVE_TEASERPP
  return true;
#else
  return false;
#endif
}

DirectMvRigidRegistrationResult solveDirectMvTeaser(
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& source,
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& destination,
    double noise_bound_m) {
  DirectMvRigidRegistrationResult result;
  if (source.cols() != destination.cols() || source.cols() < 3 ||
      !source.array().isFinite().all() || !destination.array().isFinite().all() ||
      !std::isfinite(noise_bound_m) || noise_bound_m <= 0.0) {
    return result;
  }

#ifdef BASALT_HAVE_TEASERPP
  try {
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = noise_bound_m;
    params.cbar2 = 1.0;
    params.estimate_scaling = false;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.inlier_selection_mode =
        teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;
    params.max_clique_exact_solution = false;
    params.max_clique_num_threads = 1;

    teaser::RobustRegistrationSolver solver(params);
    const teaser::RegistrationSolution solution = solver.solve(source, destination);
    if (!solution.valid || !solution.rotation.array().isFinite().all() ||
        !solution.translation.array().isFinite().all() ||
        std::abs(solution.rotation.determinant() - 1.0) > 1e-3) {
      return result;
    }

    result.rotation = solution.rotation;
    result.translation = solution.translation;
    for (int index : solver.getInputOrderedTranslationInliers()) {
      if (index >= 0 && index < source.cols())
        result.input_inliers.emplace_back(static_cast<size_t>(index));
    }
    result.valid = result.input_inliers.size() >= 3;
  } catch (const std::exception&) {
    result = DirectMvRigidRegistrationResult{};
  }
#else
  static_cast<void>(source);
  static_cast<void>(destination);
  static_cast<void>(noise_bound_m);
#endif

  return result;
}

}  // namespace basalt
