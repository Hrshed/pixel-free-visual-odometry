/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <unordered_set>

#include <Eigen/Eigenvalues>
#include <sophus/se2.hpp>
#include "basalt/imu/imu_types.h"
#include "basalt/imu/preintegration.h"
#include "basalt/utils/common_types.h"
#include "basalt/utils/imu_types.h"
#include "basalt/utils/time_utils.hpp"
#include "basalt/utils/vio_config.h"
#include "sophus/se3.hpp"

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <opencv2/imgproc.hpp>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/optical_flow/patch.h>
#include <basalt/optical_flow/direct_mv_rigid_registration.h>

#include <basalt/utils/keypoints.h>
#include <basalt/vi_estimator/landmark_database.h>

namespace basalt {

/// Unlike PatchOpticalFlow, FrameToFrameOpticalFlow always tracks patches
/// against the previous frame, not the initial frame where a track was created.
/// While it might cause more drift of the patch location, it leads to longer
/// tracks in practice.
template <typename Scalar, template <typename> typename Pattern>
class FrameToFrameOpticalFlow final : public OpticalFlowTyped<Scalar, Pattern> {
 public:
  using Vector3d = Eigen::Matrix<double, 3, 1>;

  using typename OpticalFlowTyped<Scalar, Pattern>::PatchT;
  using typename OpticalFlowTyped<Scalar, Pattern>::Vector2;
  using typename OpticalFlowTyped<Scalar, Pattern>::Matrix2;
  using typename OpticalFlowTyped<Scalar, Pattern>::Vector3;
  using typename OpticalFlowTyped<Scalar, Pattern>::Matrix3;
  using typename OpticalFlowTyped<Scalar, Pattern>::Vector4;
  using typename OpticalFlowTyped<Scalar, Pattern>::Matrix4;
  using typename OpticalFlowTyped<Scalar, Pattern>::SE2;
  using typename OpticalFlowTyped<Scalar, Pattern>::SE3;
  using OpticalFlowTyped<Scalar, Pattern>::getNumCams;
  using OpticalFlowTyped<Scalar, Pattern>::calib;
  using OpticalFlowTyped<Scalar, Pattern>::E;

  using OpticalFlowBase::config;
  using OpticalFlowBase::depth_guess;
  using OpticalFlowBase::first_state_arrived;
  using OpticalFlowBase::frame_counter;
  using OpticalFlowBase::input_depth_queue;
  using OpticalFlowBase::input_img_queue;
  using OpticalFlowBase::input_imu_queue;
  using OpticalFlowBase::input_lm_bundle_queue;
  using OpticalFlowBase::input_state_queue;
  using OpticalFlowBase::last_keypoint_id;
  using OpticalFlowBase::latest_lm_bundle;
  using OpticalFlowBase::latest_state;
  using OpticalFlowBase::old_pyramid;
  using OpticalFlowBase::output_queue;
  using OpticalFlowBase::patch_coord;
  using OpticalFlowBase::predicted_state;
  using OpticalFlowBase::processing_thread;
  using OpticalFlowBase::pyramid;
  using OpticalFlowBase::show_gui;
  using OpticalFlowBase::t_ns;
  using OpticalFlowBase::transforms;
  ExecutionStats frontend_stats{};
  std::map<int, int> guess_error_histogram{};
  std::map<FrameId, std::vector<int>> fallback_recovered_points{};
  mutable int64_t mv_total_tracking_time = 0;
  mutable int64_t mv_total_tracked_points = 0;
  mutable int64_t of_total_tracking_time = 0;
  mutable int64_t of_total_tracked_points = 0;

  FrameToFrameOpticalFlow(const VioConfig& conf, const Calibration<double>& cal)
      : OpticalFlowTyped<Scalar, Pattern>(conf, cal),
        accel_cov(cal.dicrete_time_accel_noise_std().array().square()),
        gyro_cov(cal.dicrete_time_gyro_noise_std().array().square()),
        c(config.optical_flow_detection_grid_size),
        w(cal.resolution.at(0).x()),
        h(cal.resolution.at(0).y()),
        x_start((w % c) / 2),
        x_stop(x_start + c * (w / c - 1)),
        y_start((h % c) / 2),
        y_stop(y_start + c * (h / c - 1)) {
    latest_state = std::make_shared<PoseVelBiasState<double>>();
    predicted_state = std::make_shared<PoseVelBiasState<double>>();
    if (config.optical_flow_recall_enable) patches.reserve(3000);
    cells.resize(getNumCams());
    recalls.resize(getNumCams());
    for (size_t i = 0; i < getNumCams(); i++) cells.at(i).setZero(h / c + 1, w / c + 1);
    initializeDirectStereoBaselineRectification();
  }

  void processingLoop() override {
    using std::make_shared;
    OpticalFlowInput::Ptr img;

    while (true) {
      input_img_queue.pop(img);

      if (img == nullptr) {
        if (isDirectMotionVectorMode()) {
          for (const auto& [id, active] : direct_active_tracks) {
            static_cast<void>(id);
            if (!active.confirmed) direct_mv_provisional_tracks_rejected++;
          }
          pushPendingDirectMotionVectorFrames(true);
        }
        if (output_queue) output_queue->push(nullptr);
        break;
      }
      img->addTime("frontend_frames_received");

      while (input_depth_queue.try_pop(depth_guess)) continue;
      if (show_gui) img->depth_guess = depth_guess;

      while (input_lm_bundle_queue.try_pop(latest_lm_bundle)) continue;

      if (!input_state_queue.empty()) {
        while (input_state_queue.try_pop(latest_state)) continue;  // Flush
        first_state_arrived = true;
      } else if (first_state_arrived) {
        latest_state = make_shared<PoseVelBiasState<double>>(*predicted_state);
      }

      if (first_state_arrived) {
        auto pim = processImu(img->t_ns);
        pim.predictState(*latest_state, constants::g, *predicted_state);
      }
      img->addTime("frontend_preintegration_computed");

      processFrame(img->t_ns, img);
    }

    saveFrontendStats();
  }

  bool isDirectMotionVectorMode() const {
    return config.optical_flow_subtype == OpticalFlowSubtype::F2F_DIRECT_MV;
  }

  OpticalFlowResult::Ptr makeDirectMotionVectorResult(int64_t curr_t_ns, OpticalFlowInput::Ptr& input) {
    const size_t num_cams = getNumCams();
    auto result = std::make_shared<OpticalFlowResult>();
    result->keypoints.resize(num_cams);
    result->keypoint_responses.resize(num_cams);
    result->keypoint_stereo_depths.resize(num_cams);
    result->tracking_guesses.resize(num_cams);
    result->fallback_guesses.resize(num_cams);
    result->matching_guesses.resize(num_cams);
    result->recall_guesses.resize(num_cams);
    result->t_ns = curr_t_ns;
    result->input_images = input;
    return result;
  }

  void finishDirectMotionVectorTiming(const OpticalFlowInput::Ptr& input) {
    // Populate the shared timing schema for direct MV frames.
    input->addTime("frontend_pyramid_created");
    input->addTime("frontend_tracking_ended");
    input->addTime("frontend_recall_ended");
    input->addTime("frontend_detection_cam0_ended");
    input->addTime("frontend_matching_ended");
    input->addTime("frontend_detection_cami_ended");
    input->addTime("frontend_filter_ended");
  }

  struct PendingDirectMotionVectorFrame {
    size_t frame_index = 0;
    OpticalFlowResult::Ptr result;
  };

  OpticalFlowResult::Ptr pendingDirectMotionVectorFrame(int64_t timestamp) {
    for (auto& pending : pending_direct_frames) {
      if (pending.result && pending.result->t_ns == timestamp)
        return pending.result;
    }
    return nullptr;
  }

  void pushPendingDirectMotionVectorFrames(bool flush_all) {
    const size_t delay =
        config.optical_flow_direct_mv_strict_confirmation ? 2 : 1;
    while (!pending_direct_frames.empty() &&
           (flush_all || pending_direct_frames.size() > delay)) {
      PendingDirectMotionVectorFrame pending = std::move(pending_direct_frames.front());
      pending_direct_frames.pop_front();
      finishDirectMotionVectorTiming(pending.result->input_images);
      if (output_queue && pending.frame_index % config.optical_flow_skip_frames == 0) {
        pending.result->input_images->addTime("frontend_keypoints_pushed");
        output_queue->push(pending.result);
      }
    }

    if (flush_all) transforms.reset();
  }

  enum DirectTemporalOrigin {
    DIRECT_TEMPORAL_ORIGIN_UNKNOWN = 0,
    DIRECT_TEMPORAL_ORIGIN_FORWARD = 1,
    DIRECT_TEMPORAL_ORIGIN_REVERSE = 2,
    DIRECT_TEMPORAL_ORIGIN_RECIPROCAL = 3,
  };

  struct DirectMotionVectorTrack {
    const MotionVector* mv = nullptr;
    Vector4 source_bearing = Vector4::Zero();
    Vector4 destination_bearing = Vector4::Zero();
    KeypointId id = std::numeric_limits<KeypointId>::max();
    int temporal_origin = DIRECT_TEMPORAL_ORIGIN_UNKNOWN;
    bool bearings_valid = false;
  };

  struct DirectTemporalField {
    const std::vector<DirectMotionVectorTrack>* tracks = nullptr;
    std::vector<std::vector<size_t>> bins;
    int bin_cols = 0;
    int bin_rows = 0;
  };

  struct DirectStereoCandidate {
    size_t anchor_index = 0;
    size_t other_index = 0;
    Scalar cost = 0;
  };

  struct DirectStereoField {
    const std::vector<MotionVector>* motion_vectors = nullptr;
    std::vector<std::vector<size_t>> bins;
    int bin_cols = 0;
    int bin_rows = 0;
  };

  struct DirectStereoProjection {
    Vector2 point = Vector2::Zero();
    Scalar depth = 0;
    Scalar cost = std::numeric_limits<Scalar>::max();
  };

  struct DirectTeaserConsensus {
    bool valid = false;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  };

  struct DirectTrackStereoProjections {
    DirectStereoProjection previous;
    DirectStereoProjection current;
    bool previous_evaluated = false;
    bool current_evaluated = false;
    bool previous_valid = false;
    bool current_valid = false;
  };

  struct DirectActiveTrack {
    KeypointId id = std::numeric_limits<KeypointId>::max();
    Vector2 left_point = Vector2::Zero();
    size_t age_frames = 2;
    Scalar last_stereo_depth = 0;
    int64_t last_stereo_t_ns = 0;
    size_t last_stereo_age_frames = 0;
    uint64_t diagnostic_selection_count = 0;
    uint64_t diagnostic_forward_selection_count = 0;
    uint64_t diagnostic_reverse_selection_count = 0;
    uint64_t diagnostic_reciprocal_selection_count = 0;
    uint64_t diagnostic_shared_selection_count = 0;
    long double diagnostic_teaser_residual_sum_m = 0;
    long double diagnostic_center_cost_sum = 0;
    long double diagnostic_depth_delta_sum_m = 0;
    long double diagnostic_depth_log_ratio_sum = 0;
    long double diagnostic_stereo_cost_sum = 0;
    long double diagnostic_block_area_sum_px2 = 0;
    long double diagnostic_group_multiplicity_sum = 0;
    long double diagnostic_shared_depth_range_sum_m = 0;
    double diagnostic_max_shared_depth_range_m = 0;
    Vector2 provisional_source_left = Vector2::Zero();
    Vector2 provisional_source_right = Vector2::Zero();
    Scalar provisional_source_depth = 0;
    int64_t provisional_source_t_ns = 0;
    bool confirmed = true;
    bool last_stereo_depth_valid = false;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct DirectAcceptedMeasurement {
    Vector2 previous_left = Vector2::Zero();
    Vector2 current_left = Vector2::Zero();
    Scalar previous_depth = 0;
    Scalar current_depth = 0;
  };

  std::pair<Scalar, Scalar> directDepthQuantizationInterval(
      Scalar depth, const SE3& T_0_1) const {
    const Scalar endpoint_quantization_px =
        Scalar(config.optical_flow_direct_mv_merge_radius_px) / Scalar(2);
    const Scalar focal = direct_stereo_rectified_focal.x();
    const Scalar baseline = T_0_1.translation().norm();
    if (!std::isfinite(depth) || depth <= Scalar(0) ||
        !std::isfinite(focal) || focal <= Scalar(0) ||
        !std::isfinite(baseline) || baseline <= Scalar(0)) {
      return {depth, depth};
    }
    const Scalar focal_baseline = focal * baseline;
    const Scalar disparity = focal_baseline / depth;
    const Scalar disparity_error = Scalar(2) * endpoint_quantization_px;
    const Scalar minimum =
        focal_baseline / (disparity + disparity_error);
    const Scalar maximum =
        disparity > disparity_error
            ? focal_baseline / (disparity - disparity_error)
            : std::numeric_limits<Scalar>::infinity();
    return {minimum, maximum};
  }

  bool directMeasurementsIndistinguishable(
      const DirectAcceptedMeasurement& lhs,
      const DirectAcceptedMeasurement& rhs, const SE3& T_0_1) const {
    const Scalar overlap_distance_px =
        Scalar(config.optical_flow_direct_mv_merge_radius_px);
    if ((lhs.previous_left - rhs.previous_left)
                .template lpNorm<Eigen::Infinity>() > overlap_distance_px ||
        (lhs.current_left - rhs.current_left)
                .template lpNorm<Eigen::Infinity>() > overlap_distance_px) {
      return false;
    }
    const auto lhs_previous =
        directDepthQuantizationInterval(lhs.previous_depth, T_0_1);
    const auto rhs_previous =
        directDepthQuantizationInterval(rhs.previous_depth, T_0_1);
    const auto lhs_current =
        directDepthQuantizationInterval(lhs.current_depth, T_0_1);
    const auto rhs_current =
        directDepthQuantizationInterval(rhs.current_depth, T_0_1);
    const auto overlaps = [](const auto& first, const auto& second) {
      return std::max(first.first, second.first) <=
             std::min(first.second, second.second);
    };
    return overlaps(lhs_previous, rhs_previous) &&
           overlaps(lhs_current, rhs_current);
  }

  static constexpr double kDirectDepthResidualBinWidthMeters = 0.05;
  static constexpr size_t kDirectDepthResidualHistogramBins = 601;
  static constexpr double kDirectDepthLogRatioBinWidth = 0.01;
  static constexpr size_t kDirectDepthLogRatioHistogramBins = 301;
  static constexpr double kDirectSharedSpreadBinWidthMeters = 0.01;
  static constexpr size_t kDirectSharedSpreadHistogramBins = 301;

  enum DirectTrackTerminationReason {
    DIRECT_TRACK_TERMINATION_OPTIMIZER_OBSERVATION = 0,
    DIRECT_TRACK_TERMINATION_PREVIOUS_STEREO = 1,
    DIRECT_TRACK_TERMINATION_TEMPORAL_MATCH = 2,
    DIRECT_TRACK_TERMINATION_POST_GATE = 3,
    DIRECT_TRACK_TERMINATION_I_FRAME = 4,
    DIRECT_TRACK_TERMINATION_SHUTDOWN = 5,
    DIRECT_TRACK_TERMINATION_REASON_COUNT = 6,
  };

  struct DirectTrackQualityAggregate {
    uint64_t tracks = 0;
    uint64_t frames = 0;
    uint64_t selections = 0;
    uint64_t forward_selections = 0;
    uint64_t reverse_selections = 0;
    uint64_t reciprocal_selections = 0;
    uint64_t shared_selections = 0;
    long double teaser_residual_sum_m = 0;
    long double center_cost_sum = 0;
    long double depth_delta_sum_m = 0;
    long double depth_log_ratio_sum = 0;
    long double stereo_cost_sum = 0;
    long double block_area_sum_px2 = 0;
    long double group_multiplicity_sum = 0;
    long double shared_depth_range_sum_m = 0;
    double max_shared_depth_range_m = 0;
    std::array<uint64_t, DIRECT_TRACK_TERMINATION_REASON_COUNT>
        termination_reasons{};
  };

  struct DirectSelectedMvUse {
    uint64_t count = 0;
    double previous_depth_min_m = std::numeric_limits<double>::infinity();
    double previous_depth_max_m = -std::numeric_limits<double>::infinity();
    double current_depth_min_m = std::numeric_limits<double>::infinity();
    double current_depth_max_m = -std::numeric_limits<double>::infinity();
    double teaser_residual_min_m = std::numeric_limits<double>::infinity();
    double teaser_residual_max_m = -std::numeric_limits<double>::infinity();
  };

  struct DirectSelectedMvAssignment {
    KeypointId id = std::numeric_limits<KeypointId>::max();
    size_t temporal_index = 0;
  };

  struct DirectOriginSelectionAggregate {
    uint64_t selections = 0;
    long double teaser_residual_sum_m = 0;
    long double center_cost_sum = 0;
    long double depth_delta_sum_m = 0;
    long double depth_log_ratio_sum = 0;
    long double stereo_cost_sum = 0;
    long double block_area_sum_px2 = 0;
  };

  void recordDirectCam0TrackLength(size_t age_frames) {
    direct_mv_cam0_track_length_histogram_frames[age_frames]++;
  }

  size_t directTrackQualityBucket(size_t age_frames) const {
    if (age_frames <= 2) return 0;
    if (age_frames <= 4) return 1;
    if (age_frames <= 9) return 2;
    return 3;
  }

  void accumulateDirectTrackQuality(
      DirectTrackQualityAggregate& aggregate, const DirectActiveTrack& track,
      DirectTrackTerminationReason reason) const {
    aggregate.tracks++;
    aggregate.frames += track.age_frames;
    aggregate.selections += track.diagnostic_selection_count;
    aggregate.forward_selections += track.diagnostic_forward_selection_count;
    aggregate.reverse_selections += track.diagnostic_reverse_selection_count;
    aggregate.reciprocal_selections +=
        track.diagnostic_reciprocal_selection_count;
    aggregate.shared_selections += track.diagnostic_shared_selection_count;
    aggregate.teaser_residual_sum_m +=
        track.diagnostic_teaser_residual_sum_m;
    aggregate.center_cost_sum += track.diagnostic_center_cost_sum;
    aggregate.depth_delta_sum_m += track.diagnostic_depth_delta_sum_m;
    aggregate.depth_log_ratio_sum += track.diagnostic_depth_log_ratio_sum;
    aggregate.stereo_cost_sum += track.diagnostic_stereo_cost_sum;
    aggregate.block_area_sum_px2 += track.diagnostic_block_area_sum_px2;
    aggregate.group_multiplicity_sum +=
        track.diagnostic_group_multiplicity_sum;
    aggregate.shared_depth_range_sum_m +=
        track.diagnostic_shared_depth_range_sum_m;
    aggregate.max_shared_depth_range_m =
        std::max(aggregate.max_shared_depth_range_m,
                 track.diagnostic_max_shared_depth_range_m);
    aggregate.termination_reasons.at(static_cast<size_t>(reason))++;
  }

  void recordDirectTrackQuality(const DirectActiveTrack& track,
                                DirectTrackTerminationReason reason) {
    accumulateDirectTrackQuality(
        direct_mv_track_quality_by_length.at(
            directTrackQualityBucket(track.age_frames)),
        track, reason);
  }

  void recordDirectSelectedMvSharing(
      const std::vector<DirectSelectedMvUse>& uses,
      const std::vector<DirectSelectedMvAssignment>& assignments,
      std::map<KeypointId, DirectActiveTrack>& next_active_tracks) {
    for (const DirectSelectedMvUse& use : uses) {
      if (use.count == 0) continue;
      direct_mv_selected_unique_mv_groups++;
      direct_mv_selected_mv_multiplicity_histogram[use.count]++;
      direct_mv_selected_mv_max_multiplicity =
          std::max(direct_mv_selected_mv_max_multiplicity, use.count);
      if (use.count < 2) continue;

      const double previous_depth_range =
          use.previous_depth_max_m - use.previous_depth_min_m;
      const double current_depth_range =
          use.current_depth_max_m - use.current_depth_min_m;
      const double teaser_residual_range =
          use.teaser_residual_max_m - use.teaser_residual_min_m;
      if (!std::isfinite(previous_depth_range) ||
          !std::isfinite(current_depth_range) ||
          !std::isfinite(teaser_residual_range)) {
        continue;
      }

      direct_mv_selected_shared_mv_groups++;
      direct_mv_selected_shared_assignments += use.count;
      direct_mv_selected_duplicate_assignments += use.count - 1;
      direct_mv_selected_shared_previous_depth_range_sum_m +=
          previous_depth_range;
      direct_mv_selected_shared_current_depth_range_sum_m +=
          current_depth_range;
      direct_mv_selected_shared_teaser_residual_range_sum_m +=
          teaser_residual_range;
      direct_mv_selected_shared_previous_depth_range_max_m =
          std::max(direct_mv_selected_shared_previous_depth_range_max_m,
                   previous_depth_range);
      direct_mv_selected_shared_current_depth_range_max_m =
          std::max(direct_mv_selected_shared_current_depth_range_max_m,
                   current_depth_range);
      direct_mv_selected_shared_teaser_residual_range_max_m =
          std::max(direct_mv_selected_shared_teaser_residual_range_max_m,
                   teaser_residual_range);
      const size_t previous_bin = std::min(
          static_cast<size_t>(previous_depth_range /
                              kDirectSharedSpreadBinWidthMeters),
          kDirectSharedSpreadHistogramBins - 1);
      const size_t current_bin = std::min(
          static_cast<size_t>(current_depth_range /
                              kDirectSharedSpreadBinWidthMeters),
          kDirectSharedSpreadHistogramBins - 1);
      const size_t teaser_bin = std::min(
          static_cast<size_t>(teaser_residual_range /
                              kDirectSharedSpreadBinWidthMeters),
          kDirectSharedSpreadHistogramBins - 1);
      direct_mv_selected_shared_previous_depth_range_histogram.at(
          previous_bin)++;
      direct_mv_selected_shared_current_depth_range_histogram.at(current_bin)++;
      direct_mv_selected_shared_teaser_residual_range_histogram.at(teaser_bin)++;
    }

    for (const DirectSelectedMvAssignment& assignment : assignments) {
      const DirectSelectedMvUse& use = uses.at(assignment.temporal_index);
      auto active_it = next_active_tracks.find(assignment.id);
      if (active_it == next_active_tracks.end()) continue;
      DirectActiveTrack& active = active_it->second;
      active.diagnostic_group_multiplicity_sum += use.count;
      if (use.count < 2) continue;
      const double depth_range =
          use.current_depth_max_m - use.current_depth_min_m;
      active.diagnostic_shared_selection_count++;
      active.diagnostic_shared_depth_range_sum_m += depth_range;
      active.diagnostic_max_shared_depth_range_m =
          std::max(active.diagnostic_max_shared_depth_range_m, depth_range);
    }
  }

  void recordDirectTemporalDepthResidual(Scalar previous_depth, Scalar current_depth, size_t gap_frames,
                                         int64_t previous_t_ns, int64_t current_t_ns) {
    const double previous = static_cast<double>(previous_depth);
    const double current = static_cast<double>(current_depth);
    if (!std::isfinite(previous) || !std::isfinite(current) || previous <= 0 || current <= 0) return;

    const double residual = current - previous;
    const double absolute_residual = std::abs(residual);
    const double absolute_log_ratio = std::abs(std::log(current / previous));
    const size_t bin = std::min(
        static_cast<size_t>(absolute_residual / kDirectDepthResidualBinWidthMeters),
        kDirectDepthResidualHistogramBins - 1);
    const size_t log_ratio_bin =
        std::min(static_cast<size_t>(absolute_log_ratio / kDirectDepthLogRatioBinWidth),
                 kDirectDepthLogRatioHistogramBins - 1);

    direct_mv_temporal_depth_residual_samples++;
    direct_mv_temporal_depth_residual_sum_m += residual;
    direct_mv_temporal_depth_abs_residual_sum_m += absolute_residual;
    direct_mv_temporal_depth_squared_residual_sum_m2 += residual * residual;
    direct_mv_temporal_depth_max_abs_residual_m =
        std::max(direct_mv_temporal_depth_max_abs_residual_m, absolute_residual);
    direct_mv_temporal_depth_abs_log_ratio_sum += absolute_log_ratio;
    direct_mv_temporal_depth_max_abs_log_ratio =
        std::max(direct_mv_temporal_depth_max_abs_log_ratio, absolute_log_ratio);
    direct_mv_temporal_depth_abs_residual_histogram.at(bin)++;
    direct_mv_temporal_depth_abs_log_ratio_histogram.at(log_ratio_bin)++;
    direct_mv_temporal_depth_gap_histogram_frames[gap_frames]++;

    const double dt_seconds = static_cast<double>(current_t_ns - previous_t_ns) * 1e-9;
    if (std::isfinite(dt_seconds) && dt_seconds > 0) {
      direct_mv_temporal_depth_rate_samples++;
      direct_mv_temporal_depth_abs_rate_sum_mps += absolute_residual / dt_seconds;
    }

    if (gap_frames == 1) {
      direct_mv_temporal_depth_immediate_samples++;
      direct_mv_temporal_depth_immediate_residual_sum_m += residual;
      direct_mv_temporal_depth_immediate_abs_residual_sum_m += absolute_residual;
      direct_mv_temporal_depth_immediate_squared_residual_sum_m2 += residual * residual;
      direct_mv_temporal_depth_immediate_max_abs_residual_m =
          std::max(direct_mv_temporal_depth_immediate_max_abs_residual_m, absolute_residual);
      direct_mv_temporal_depth_immediate_abs_log_ratio_sum += absolute_log_ratio;
      direct_mv_temporal_depth_immediate_max_abs_log_ratio =
          std::max(direct_mv_temporal_depth_immediate_max_abs_log_ratio, absolute_log_ratio);
      direct_mv_temporal_depth_immediate_abs_residual_histogram.at(bin)++;
      direct_mv_temporal_depth_immediate_abs_log_ratio_histogram.at(log_ratio_bin)++;
    }
  }

  template <size_t NumBins>
  double directHistogramQuantile(const std::array<uint64_t, NumBins>& histogram, uint64_t sample_count,
                                 double quantile, double bin_width) const {
    if (sample_count == 0) return 0;
    const uint64_t target = static_cast<uint64_t>(std::ceil(quantile * static_cast<double>(sample_count)));
    uint64_t cumulative = 0;
    for (size_t bin = 0; bin < histogram.size(); ++bin) {
      cumulative += histogram.at(bin);
      if (cumulative >= target) return static_cast<double>(bin + 1) * bin_width;
    }
    return static_cast<double>(histogram.size()) * bin_width;
  }

  size_t directTrackLengthQuantile(const std::map<size_t, uint64_t>& histogram, uint64_t sample_count,
                                   double quantile) const {
    if (sample_count == 0) return 0;
    const uint64_t target = static_cast<uint64_t>(std::ceil(quantile * static_cast<double>(sample_count)));
    uint64_t cumulative = 0;
    for (const auto& [age_frames, count] : histogram) {
      cumulative += count;
      if (cumulative >= target) return age_frames;
    }
    return histogram.empty() ? 0 : histogram.rbegin()->first;
  }

  struct GenericStereoDepthState {
    Scalar depth = 0;
    size_t frame_index = 0;
    int64_t t_ns = 0;
  };

  struct GenericDepthResidualStats {
    uint64_t samples = 0;
    uint64_t rate_samples = 0;
    long double residual_sum_m = 0;
    long double abs_residual_sum_m = 0;
    long double squared_residual_sum_m2 = 0;
    long double abs_rate_sum_mps = 0;
    long double abs_log_ratio_sum = 0;
    double max_abs_residual_m = 0;
    double max_abs_log_ratio = 0;
    std::array<uint64_t, kDirectDepthResidualHistogramBins> abs_residual_histogram{};
    std::array<uint64_t, kDirectDepthLogRatioHistogramBins> abs_log_ratio_histogram{};
    std::map<size_t, uint64_t> gap_histogram_frames;
    uint64_t immediate_samples = 0;
    long double immediate_residual_sum_m = 0;
    long double immediate_abs_residual_sum_m = 0;
    long double immediate_squared_residual_sum_m2 = 0;
    long double immediate_abs_log_ratio_sum = 0;
    double immediate_max_abs_residual_m = 0;
    double immediate_max_abs_log_ratio = 0;
    std::array<uint64_t, kDirectDepthResidualHistogramBins> immediate_abs_residual_histogram{};
    std::array<uint64_t, kDirectDepthLogRatioHistogramBins> immediate_abs_log_ratio_histogram{};
  };

  void recordGenericTemporalDepthResidual(Scalar previous_depth, Scalar current_depth, size_t gap_frames,
                                          int64_t previous_t_ns, int64_t current_t_ns) {
    const double previous = static_cast<double>(previous_depth);
    const double current = static_cast<double>(current_depth);
    if (!std::isfinite(previous) || !std::isfinite(current) || previous <= 0 || current <= 0) return;

    const double residual = current - previous;
    const double absolute_residual = std::abs(residual);
    const double absolute_log_ratio = std::abs(std::log(current / previous));
    const size_t bin = std::min(
        static_cast<size_t>(absolute_residual / kDirectDepthResidualBinWidthMeters),
        kDirectDepthResidualHistogramBins - 1);
    const size_t log_ratio_bin =
        std::min(static_cast<size_t>(absolute_log_ratio / kDirectDepthLogRatioBinWidth),
                 kDirectDepthLogRatioHistogramBins - 1);

    generic_depth_residual_stats.samples++;
    generic_depth_residual_stats.residual_sum_m += residual;
    generic_depth_residual_stats.abs_residual_sum_m += absolute_residual;
    generic_depth_residual_stats.squared_residual_sum_m2 += residual * residual;
    generic_depth_residual_stats.abs_log_ratio_sum += absolute_log_ratio;
    generic_depth_residual_stats.max_abs_residual_m =
        std::max(generic_depth_residual_stats.max_abs_residual_m, absolute_residual);
    generic_depth_residual_stats.max_abs_log_ratio =
        std::max(generic_depth_residual_stats.max_abs_log_ratio, absolute_log_ratio);
    generic_depth_residual_stats.abs_residual_histogram.at(bin)++;
    generic_depth_residual_stats.abs_log_ratio_histogram.at(log_ratio_bin)++;
    generic_depth_residual_stats.gap_histogram_frames[gap_frames]++;

    const double dt_seconds = static_cast<double>(current_t_ns - previous_t_ns) * 1e-9;
    if (std::isfinite(dt_seconds) && dt_seconds > 0) {
      generic_depth_residual_stats.rate_samples++;
      generic_depth_residual_stats.abs_rate_sum_mps += absolute_residual / dt_seconds;
    }

    if (gap_frames == 1) {
      generic_depth_residual_stats.immediate_samples++;
      generic_depth_residual_stats.immediate_residual_sum_m += residual;
      generic_depth_residual_stats.immediate_abs_residual_sum_m += absolute_residual;
      generic_depth_residual_stats.immediate_squared_residual_sum_m2 += residual * residual;
      generic_depth_residual_stats.immediate_abs_log_ratio_sum += absolute_log_ratio;
      generic_depth_residual_stats.immediate_max_abs_residual_m =
          std::max(generic_depth_residual_stats.immediate_max_abs_residual_m, absolute_residual);
      generic_depth_residual_stats.immediate_max_abs_log_ratio =
          std::max(generic_depth_residual_stats.immediate_max_abs_log_ratio, absolute_log_ratio);
      generic_depth_residual_stats.immediate_abs_residual_histogram.at(bin)++;
      generic_depth_residual_stats.immediate_abs_log_ratio_histogram.at(log_ratio_bin)++;
    }
  }

  void recordGenericFrameStats(const OpticalFlowResult::Ptr& result) {
    if (!result || result->keypoints.empty()) return;
    const auto& cam0_keypoints = result->keypoints.at(0);

    for (auto it = generic_cam0_active_track_lengths.begin(); it != generic_cam0_active_track_lengths.end();) {
      if (cam0_keypoints.count(it->first) == 0) {
        generic_cam0_track_length_histogram_frames[it->second]++;
        generic_last_stereo_depth.erase(it->first);
        it = generic_cam0_active_track_lengths.erase(it);
      } else {
        ++it;
      }
    }
    for (const auto& [id, keypoint] : cam0_keypoints) {
      static_cast<void>(keypoint);
      auto [it, inserted] = generic_cam0_active_track_lengths.emplace(id, 1);
      if (!inserted) it->second++;
    }

    if (result->keypoints.size() >= 2) {
      const auto& cam1_keypoints = result->keypoints.at(1);
      const SE3 T_0_1 = calib.T_i_c[0].inverse() * calib.T_i_c[1];
      for (const auto& [id, cam0_keypoint] : cam0_keypoints) {
        const auto cam1_it = cam1_keypoints.find(id);
        if (cam1_it == cam1_keypoints.end()) continue;

        Vector4 cam0_bearing;
        Vector4 cam1_bearing;
        const Vector2 cam0_point = cam0_keypoint.translation().template cast<Scalar>();
        const Vector2 cam1_point = cam1_it->second.translation().template cast<Scalar>();
        if (!calib.intrinsics.at(0).unproject(cam0_point, cam0_bearing) ||
            !calib.intrinsics.at(1).unproject(cam1_point, cam1_bearing)) {
          continue;
        }
        Scalar cam0_depth = 0;
        Scalar cam1_depth = 0;
        if (!directStereoDepths(cam0_bearing, cam1_bearing, T_0_1, cam0_depth, cam1_depth) ||
            cam0_depth <= Scalar(1.0 / 3.0) || cam1_depth <= Scalar(1.0 / 3.0) || cam0_depth > Scalar(30)) {
          continue;
        }

        const auto previous = generic_last_stereo_depth.find(id);
        if (previous != generic_last_stereo_depth.end()) {
          recordGenericTemporalDepthResidual(previous->second.depth, cam0_depth,
                                             generic_stats_frame_index - previous->second.frame_index,
                                             previous->second.t_ns, result->t_ns);
        }
        generic_last_stereo_depth[id] = {cam0_depth, generic_stats_frame_index, result->t_ns};
      }
    }
    generic_stats_frame_index++;
  }

  bool directStereoDepths(const Vector4& bearing_0, const Vector4& bearing_other, const SE3& T_0_other,
                          Scalar& depth_0, Scalar& depth_other) const {
    const Vector3 ray_0 = bearing_0.template head<3>().normalized();
    const Vector3 ray_other = T_0_other.rotationMatrix() * bearing_other.template head<3>().normalized();
    const Vector3 translation = T_0_other.translation();
    const Scalar ray_dot = ray_0.dot(ray_other);
    const Scalar denominator = Scalar(1) - ray_dot * ray_dot;
    if (!std::isfinite(denominator) || denominator <= Scalar(1e-6)) return false;

    const Scalar ray_0_translation = ray_0.dot(translation);
    const Scalar ray_other_translation = ray_other.dot(translation);
    depth_0 = (ray_0_translation - ray_dot * ray_other_translation) / denominator;
    depth_other = (ray_dot * ray_0_translation - ray_other_translation) / denominator;
    return std::isfinite(depth_0) && std::isfinite(depth_other);
  }

  bool directStereoPositiveDepths(const Vector4& bearing_0, const Vector4& bearing_other, const SE3& T_0_other,
                                  Scalar minimum_depth = Scalar(1.0 / 3.0)) const {
    Scalar depth_0 = 0;
    Scalar depth_other = 0;
    return directStereoDepths(bearing_0, bearing_other, T_0_other, depth_0, depth_other) &&
           depth_0 > minimum_depth && depth_other > minimum_depth;
  }

  Vector2 directMotionVectorDisplacement(const MotionVector& mv) const {
    if (mv.motion_scale > 0) {
      return Vector2(Scalar(mv.quantized_flow_x), Scalar(mv.quantized_flow_y));
    }
    return Vector2(Scalar(mv.dst_x - mv.src_x), Scalar(mv.dst_y - mv.src_y));
  }

  Vector2 directMotionVectorDestination(const Vector2& source, const MotionVector& mv) const {
    return source + directMotionVectorDisplacement(mv);
  }

  DirectStereoField makeDirectStereoField(const std::vector<MotionVector>& motion_vectors) const {
    constexpr int kBinSize = 8;
    DirectStereoField field;
    field.motion_vectors = &motion_vectors;
    field.bin_cols = (static_cast<int>(w) + kBinSize - 1) / kBinSize;
    field.bin_rows = (static_cast<int>(h) + kBinSize - 1) / kBinSize;
    field.bins.resize(static_cast<size_t>(field.bin_cols * field.bin_rows));

    for (size_t index = 0; index < motion_vectors.size(); ++index) {
      const MotionVector& mv = motion_vectors.at(index);
      if (mv.source != -1 || mv.width <= 0 || mv.height <= 0) continue;

      const Scalar half_width = Scalar(0.5) * Scalar(mv.width);
      const Scalar half_height = Scalar(0.5) * Scalar(mv.height);
      const int min_x = std::clamp(static_cast<int>(std::floor(Scalar(mv.src_x) - half_width)) / kBinSize, 0,
                                   field.bin_cols - 1);
      const int max_x = std::clamp(static_cast<int>(std::floor(Scalar(mv.src_x) + half_width)) / kBinSize, 0,
                                   field.bin_cols - 1);
      const int min_y = std::clamp(static_cast<int>(std::floor(Scalar(mv.src_y) - half_height)) / kBinSize, 0,
                                   field.bin_rows - 1);
      const int max_y = std::clamp(static_cast<int>(std::floor(Scalar(mv.src_y) + half_height)) / kBinSize, 0,
                                   field.bin_rows - 1);
      for (int bin_y = min_y; bin_y <= max_y; ++bin_y) {
        for (int bin_x = min_x; bin_x <= max_x; ++bin_x) {
          field.bins.at(static_cast<size_t>(bin_y * field.bin_cols + bin_x)).push_back(index);
        }
      }
    }
    return field;
  }

  DirectTemporalField makeDirectTemporalField(
      const std::vector<DirectMotionVectorTrack>& tracks) const {
    constexpr int kBinSize = 16;
    DirectTemporalField field;
    field.tracks = &tracks;
    field.bin_cols = (static_cast<int>(w) + kBinSize - 1) / kBinSize;
    field.bin_rows = (static_cast<int>(h) + kBinSize - 1) / kBinSize;
    field.bins.resize(static_cast<size_t>(field.bin_cols * field.bin_rows));

    for (size_t index = 0; index < tracks.size(); ++index) {
      const MotionVector& mv = *tracks.at(index).mv;
      if (mv.source != -1 || mv.width <= 0 || mv.height <= 0) continue;

      const Scalar half_width = Scalar(0.5) * Scalar(mv.width);
      const Scalar half_height = Scalar(0.5) * Scalar(mv.height);
      const int min_x = std::clamp(
          static_cast<int>(std::floor(Scalar(mv.src_x) - half_width)) /
              kBinSize,
          0, field.bin_cols - 1);
      const int max_x = std::clamp(
          static_cast<int>(std::floor(Scalar(mv.src_x) + half_width)) /
              kBinSize,
          0, field.bin_cols - 1);
      const int min_y = std::clamp(
          static_cast<int>(std::floor(Scalar(mv.src_y) - half_height)) /
              kBinSize,
          0, field.bin_rows - 1);
      const int max_y = std::clamp(
          static_cast<int>(std::floor(Scalar(mv.src_y) + half_height)) /
              kBinSize,
          0, field.bin_rows - 1);
      for (int bin_y = min_y; bin_y <= max_y; ++bin_y) {
        for (int bin_x = min_x; bin_x <= max_x; ++bin_x) {
          field.bins.at(
              static_cast<size_t>(bin_y * field.bin_cols + bin_x))
              .push_back(index);
        }
      }
    }
    return field;
  }

  const std::vector<size_t>* directTemporalCandidates(
      const DirectTemporalField& field, const Vector2& source_point) const {
    constexpr int kBinSize = 16;
    if (!field.tracks || field.bins.empty() ||
        !source_point.array().isFinite().all() || source_point.x() < Scalar(0) ||
        source_point.y() < Scalar(0) || source_point.x() >= Scalar(w) ||
        source_point.y() >= Scalar(h)) {
      return nullptr;
    }
    const int bin_x = std::clamp(static_cast<int>(source_point.x()) / kBinSize,
                                 0, field.bin_cols - 1);
    const int bin_y = std::clamp(static_cast<int>(source_point.y()) / kBinSize,
                                 0, field.bin_rows - 1);
    return &field.bins.at(
        static_cast<size_t>(bin_y * field.bin_cols + bin_x));
  }

  void initializeDirectStereoBaselineRectification() {
    direct_stereo_baseline_rectification_valid = false;
    if (getNumCams() < 2 || calib.resolution.at(0) != calib.resolution.at(1)) return;

    const SE3 T_0_1 = calib.T_i_c[0].inverse() * calib.T_i_c[1];
    Vector3 axis_x = T_0_1.translation();
    const Scalar baseline_norm = axis_x.norm();
    if (!std::isfinite(baseline_norm) || baseline_norm <= Scalar(1e-8)) return;
    axis_x /= baseline_norm;

    const Vector3 optical_axis = Vector3::UnitZ();
    const Vector3 forward = optical_axis + T_0_1.rotationMatrix() * optical_axis;
    Vector3 axis_z = forward - axis_x * axis_x.dot(forward);
    if (axis_z.norm() <= Scalar(1e-8)) {
      Vector3 fallback = Vector3::UnitX();
      if (std::abs(axis_x.y()) < std::abs(axis_x.x()) &&
          std::abs(axis_x.y()) <= std::abs(axis_x.z())) {
        fallback = Vector3::UnitY();
      } else if (std::abs(axis_x.z()) < std::abs(axis_x.x()) &&
                 std::abs(axis_x.z()) < std::abs(axis_x.y())) {
        fallback = Vector3::UnitZ();
      }
      axis_z = fallback - axis_x * axis_x.dot(fallback);
    }
    const Scalar axis_z_norm = axis_z.norm();
    if (!std::isfinite(axis_z_norm) || axis_z_norm <= Scalar(1e-8)) return;
    axis_z /= axis_z_norm;
    Vector3 axis_y = axis_z.cross(axis_x).normalized();
    axis_z = axis_x.cross(axis_y).normalized();
    if (axis_z.dot(forward) < Scalar(0)) {
      axis_y = -axis_y;
      axis_z = -axis_z;
    }

    direct_stereo_R_0_rectified.col(0) = axis_x;
    direct_stereo_R_0_rectified.col(1) = axis_y;
    direct_stereo_R_0_rectified.col(2) = axis_z;
    direct_stereo_R_1_rectified =
        T_0_1.rotationMatrix().transpose() * direct_stereo_R_0_rectified;

    const auto intrinsics_0 = calib.intrinsics.at(0).getParam();
    const auto intrinsics_1 = calib.intrinsics.at(1).getParam();
    if (intrinsics_0.size() < 2 || intrinsics_1.size() < 2) return;
    const Scalar focal = std::min(
        {Scalar(intrinsics_0[0]), Scalar(intrinsics_0[1]),
         Scalar(intrinsics_1[0]), Scalar(intrinsics_1[1])});
    if (!std::isfinite(focal) || focal <= Scalar(0)) {
      return;
    }
    direct_stereo_rectified_focal = Vector2::Constant(focal);
    direct_stereo_rectified_center =
        Vector2(Scalar(calib.resolution.at(0).x() - 1) / Scalar(2),
                Scalar(calib.resolution.at(0).y() - 1) / Scalar(2));
    direct_stereo_baseline_rectification_valid = true;
  }

  void configureDirectStereoBaselineRectification(const ImageData& image) {
    if (image.stereo_motion_vector_rectification_mode !=
        STEREO_MV_BASELINE_PINHOLE) {
      return;
    }
    if (!direct_stereo_baseline_rectification_valid || getNumCams() < 2 ||
        image.stereo_rectified_resolution[0] !=
            calib.resolution.at(0).x() ||
        image.stereo_rectified_resolution[1] !=
            calib.resolution.at(0).y()) {
      std::cerr << "Stereo baseline rectification metadata is incompatible with the runtime calibration."
                << std::endl;
      std::abort();
    }

    Matrix3 manifest_R_0_rectified;
    Matrix3 manifest_R_1_rectified;
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        const size_t index = static_cast<size_t>(3 * row + column);
        manifest_R_0_rectified(row, column) =
            Scalar(image.stereo_rotation_cam0_rectified[index]);
        manifest_R_1_rectified(row, column) =
            Scalar(image.stereo_rotation_cam1_rectified[index]);
      }
    }
    const Vector2 manifest_focal(
        Scalar(image.stereo_rectified_pinhole[0]),
        Scalar(image.stereo_rectified_pinhole[1]));
    const Vector2 manifest_center(
        Scalar(image.stereo_rectified_pinhole[2]),
        Scalar(image.stereo_rectified_pinhole[3]));
    const SE3 T_0_1 = calib.T_i_c[0].inverse() * calib.T_i_c[1];
    const Scalar rotation_tolerance = Scalar(1e-5);
    const bool metadata_valid =
        manifest_R_0_rectified.array().isFinite().all() &&
        manifest_R_1_rectified.array().isFinite().all() &&
        manifest_focal.array().isFinite().all() &&
        manifest_center.array().isFinite().all() &&
        (manifest_focal.array() > Scalar(0)).all() &&
        (manifest_R_0_rectified - direct_stereo_R_0_rectified).norm() <=
            rotation_tolerance &&
        (manifest_R_1_rectified -
         T_0_1.rotationMatrix().transpose() * manifest_R_0_rectified).norm() <=
            rotation_tolerance &&
        std::abs(manifest_R_0_rectified.col(0).normalized().dot(
                     T_0_1.translation().normalized()) -
                 Scalar(1)) <= rotation_tolerance;
    if (!metadata_valid) {
      std::cerr << "Stereo baseline rectification metadata does not match the calibrated baseline."
                << std::endl;
      std::abort();
    }
    direct_stereo_R_0_rectified = manifest_R_0_rectified;
    direct_stereo_R_1_rectified = manifest_R_1_rectified;
    direct_stereo_rectified_focal = manifest_focal;
    direct_stereo_rectified_center = manifest_center;
  }

  bool nativeCam0ToBaselineRectified(const Vector4& native_bearing,
                                     Vector2& rectified_point) const {
    if (!direct_stereo_baseline_rectification_valid) return false;
    const Vector3 rectified_ray = direct_stereo_R_0_rectified.transpose() *
                                  native_bearing.template head<3>();
    if (!rectified_ray.array().isFinite().all() ||
        rectified_ray.z() <= Scalar(1e-8)) {
      return false;
    }
    rectified_point = direct_stereo_rectified_center +
                      direct_stereo_rectified_focal.cwiseProduct(
                          rectified_ray.template head<2>() / rectified_ray.z());
    return rectified_point.array().isFinite().all();
  }

  bool baselineRectifiedToNative(size_t camera, const Vector2& rectified_point,
                                 Vector4& native_bearing,
                                 Vector2* native_point = nullptr) const {
    if (!direct_stereo_baseline_rectification_valid || camera > 1 ||
        !rectified_point.array().isFinite().all()) {
      return false;
    }
    Vector3 rectified_ray(
        (rectified_point.x() - direct_stereo_rectified_center.x()) /
            direct_stereo_rectified_focal.x(),
        (rectified_point.y() - direct_stereo_rectified_center.y()) /
            direct_stereo_rectified_focal.y(),
        Scalar(1));
    rectified_ray.normalize();
    native_bearing.setZero();
    native_bearing.template head<3>() =
        (camera == 0 ? direct_stereo_R_0_rectified
                     : direct_stereo_R_1_rectified) *
        rectified_ray;
    if (!native_bearing.array().isFinite().all()) return false;
    return !native_point ||
           calib.intrinsics.at(camera).project(native_bearing, *native_point);
  }

  bool findDirectStereoProjection(const DirectStereoField& field, const Vector2& left_point,
                                  const Vector4& left_bearing, const Matrix4& essential, const SE3& T_0_1,
                                  int rectification_mode, DirectStereoProjection& best,
                                  bool record_stats = true) {
    constexpr int kBinSize = 8;
    if (record_stats) direct_mv_codec_stereo_queries++;
    if (!field.motion_vectors || field.bins.empty() || left_point.x() < Scalar(0) || left_point.y() < Scalar(0) ||
        left_point.x() >= Scalar(w) || left_point.y() >= Scalar(h)) {
      return false;
    }

    Vector2 encoded_left_point = left_point;
    if (rectification_mode == STEREO_MV_BASELINE_PINHOLE &&
        !nativeCam0ToBaselineRectified(left_bearing, encoded_left_point)) {
      return false;
    }
    if (encoded_left_point.x() < Scalar(0) ||
        encoded_left_point.y() < Scalar(0) ||
        encoded_left_point.x() >= Scalar(w) ||
        encoded_left_point.y() >= Scalar(h)) {
      return false;
    }

    const int bin_x = std::clamp(static_cast<int>(encoded_left_point.x()) / kBinSize, 0, field.bin_cols - 1);
    const int bin_y = std::clamp(static_cast<int>(encoded_left_point.y()) / kBinSize, 0, field.bin_rows - 1);
    bool found = false;
    const Scalar right_width = Scalar(calib.resolution.at(1).x());
    const Scalar right_height = Scalar(calib.resolution.at(1).y());
    for (size_t index : field.bins.at(static_cast<size_t>(bin_y * field.bin_cols + bin_x))) {
      const MotionVector& mv = field.motion_vectors->at(index);
      const Scalar half_width = Scalar(0.5) * Scalar(mv.width);
      const Scalar half_height = Scalar(0.5) * Scalar(mv.height);
      if (std::abs(encoded_left_point.x() - Scalar(mv.src_x)) > half_width ||
          std::abs(encoded_left_point.y() - Scalar(mv.src_y)) > half_height) {
        continue;
      }
      if (record_stats) direct_mv_codec_stereo_candidates++;

      const Vector2 encoded_right_point =
          directMotionVectorDestination(encoded_left_point, mv);
      if (!encoded_right_point.array().isFinite().all() || encoded_right_point.x() < Scalar(0) ||
          encoded_right_point.y() < Scalar(0) || encoded_right_point.x() >= Scalar(w) ||
          encoded_right_point.y() >= Scalar(h)) {
        continue;
      }

      Vector4 right_bearing;
      Vector2 right_point = encoded_right_point;
      if (rectification_mode == STEREO_MV_BASELINE_PINHOLE) {
        if (!baselineRectifiedToNative(1, encoded_right_point, right_bearing,
                                       &right_point)) {
          continue;
        }
      } else if (rectification_mode == STEREO_MV_RIGHT_TO_LEFT) {
        Vector4 rectified_right_bearing;
        if (!calib.intrinsics.at(0).unproject(encoded_right_point, rectified_right_bearing)) continue;
        right_bearing.setZero();
        right_bearing.template head<3>() =
            T_0_1.rotationMatrix().transpose() * rectified_right_bearing.template head<3>();
        if (!calib.intrinsics.at(1).project(right_bearing, right_point)) continue;
      } else if (!calib.intrinsics.at(1).unproject(right_point, right_bearing)) {
        continue;
      }
      if (!right_point.array().isFinite().all() || right_point.x() < Scalar(0) || right_point.y() < Scalar(0) ||
          right_point.x() >= right_width || right_point.y() >= right_height) {
        continue;
      }
      const Scalar epipolar_error = std::abs((left_bearing.transpose() * essential * right_bearing).value());
      if (!std::isfinite(epipolar_error) || epipolar_error > Scalar(config.optical_flow_epipolar_error)) continue;
      Scalar left_depth = 0;
      Scalar right_depth = 0;
      if (!directStereoDepths(left_bearing, right_bearing, T_0_1, left_depth, right_depth) ||
          left_depth <= Scalar(1.0 / 3.0) || right_depth <= Scalar(1.0 / 3.0) || left_depth > Scalar(30)) {
        continue;
      }
      Vector2 reprojected_right;
      Scalar reprojected_depth = 0;
      if (!calib.projectBetweenCams(left_point, left_depth, reprojected_right, reprojected_depth, 0, 1) ||
          !reprojected_right.array().isFinite().all()) {
        continue;
      }
      const Scalar reprojection_error = (right_point - reprojected_right).norm();
      if (!std::isfinite(reprojection_error) ||
          reprojection_error > Scalar(config.optical_flow_direct_mv_stereo_reprojection_gate)) {
        continue;
      }

      const Scalar normalized_center_distance =
          (encoded_left_point.x() - Scalar(mv.src_x)) *
                  (encoded_left_point.x() - Scalar(mv.src_x)) /
              std::max(half_width * half_width, Scalar(1)) +
          (encoded_left_point.y() - Scalar(mv.src_y)) *
                  (encoded_left_point.y() - Scalar(mv.src_y)) /
              std::max(half_height * half_height, Scalar(1));
      const Scalar cost = reprojection_error + epipolar_error + Scalar(1e-4) * Scalar(mv.width * mv.height) +
                          Scalar(1e-3) * normalized_center_distance;
      if (!found || cost < best.cost) {
        best.point = right_point;
        best.depth = left_depth;
        best.cost = cost;
        found = true;
      }
    }
    return found;
  }

  bool findDirectStereoAtPoint(const DirectStereoField& field, const Vector2& left_point, const Matrix4& essential,
                               const SE3& T_0_1, int rectification_mode,
                               DirectStereoProjection& projection, bool record_stats = true) {
    Vector4 left_bearing;
    return calib.intrinsics.at(0).unproject(left_point, left_bearing) &&
           findDirectStereoProjection(field, left_point, left_bearing, essential, T_0_1,
                                      rectification_mode, projection, record_stats);
  }

  bool directStereoPoint3d(const Vector2& left_point, Scalar depth,
                           Eigen::Vector3d& point) const {
    Vector4 bearing;
    if (!std::isfinite(depth) || depth <= Scalar(0) ||
        !calib.intrinsics.at(0).unproject(left_point, bearing)) {
      return false;
    }
    const Vector3 ray = bearing.template head<3>();
    const Scalar ray_norm = ray.norm();
    if (!std::isfinite(ray_norm) || ray_norm <= Scalar(1e-9)) return false;
    point = (ray * (depth / ray_norm)).template cast<double>();
    return point.array().isFinite().all();
  }

  DirectTeaserConsensus estimateDirectTeaserConsensus(
      const std::vector<DirectMotionVectorTrack>& tracks,
      const DirectStereoField& previous_stereo_field,
      const DirectStereoField& current_stereo_field, const Matrix4& essential,
      const SE3& T_0_1, int rectification_mode,
      std::vector<DirectTrackStereoProjections>* projection_cache) {
    DirectTeaserConsensus consensus;

    direct_mv_teaser_frames++;
    direct_mv_teaser_pair_input_tracks += tracks.size();
    Eigen::aligned_vector<Eigen::Vector3d> candidate_sources;
    Eigen::aligned_vector<Eigen::Vector3d> candidate_destinations;
    std::vector<double> candidate_costs;
    std::map<size_t, size_t> selected_by_cell;
    const size_t cell_columns = (w + c - 1) / c;

    for (size_t track_index = 0; track_index < tracks.size(); ++track_index) {
      const DirectMotionVectorTrack& track = tracks.at(track_index);
      const MotionVector& mv = *track.mv;
      const Vector2 source(Scalar(mv.src_x), Scalar(mv.src_y));
      const Vector2 destination = directMotionVectorDestination(source, mv);
      DirectTrackStereoProjections local_projections;
      DirectTrackStereoProjections& projections =
          projection_cache ? projection_cache->at(track_index)
                           : local_projections;
      if (!projections.previous_evaluated) {
        projections.previous_valid = findDirectStereoAtPoint(
            previous_stereo_field, source, essential, T_0_1,
            rectification_mode, projections.previous, false);
        projections.previous_evaluated = true;
      }
      if (!projections.current_evaluated) {
        projections.current_valid = findDirectStereoAtPoint(
            current_stereo_field, destination, essential, T_0_1,
            rectification_mode, projections.current, false);
        projections.current_evaluated = true;
      }
      if (!projections.previous_valid || !projections.current_valid) {
        if (!projections.previous_valid && !projections.current_valid) {
          direct_mv_teaser_pair_missing_both_stereo++;
        } else if (!projections.previous_valid) {
          direct_mv_teaser_pair_missing_previous_stereo++;
        } else {
          direct_mv_teaser_pair_missing_current_stereo++;
        }
        continue;
      }
      const DirectStereoProjection& previous_stereo = projections.previous;
      const DirectStereoProjection& current_stereo = projections.current;

      Eigen::Vector3d source_3d;
      Eigen::Vector3d destination_3d;
      if (!directStereoPoint3d(source, previous_stereo.depth, source_3d) ||
          !directStereoPoint3d(destination, current_stereo.depth, destination_3d)) {
        direct_mv_teaser_pair_invalid_3d++;
        continue;
      }

      const size_t candidate_index = candidate_sources.size();
      candidate_sources.emplace_back(source_3d);
      candidate_destinations.emplace_back(destination_3d);
      candidate_costs.emplace_back(
          static_cast<double>(previous_stereo.cost + current_stereo.cost));

      const size_t cell_x = std::min(static_cast<size_t>(source.x()) / c,
                                     cell_columns - 1);
      const size_t cell_y = std::min(static_cast<size_t>(source.y()) / c,
                                     (h + c - 1) / c - 1);
      const size_t cell = cell_y * cell_columns + cell_x;
      const auto selected_it = selected_by_cell.find(cell);
      if (selected_it == selected_by_cell.end() ||
          candidate_costs.at(candidate_index) < candidate_costs.at(selected_it->second)) {
        selected_by_cell[cell] = candidate_index;
      }
    }

    direct_mv_teaser_candidate_pairs += candidate_sources.size();
    direct_mv_teaser_solver_pairs += selected_by_cell.size();
    if (selected_by_cell.size() < 3) {
      direct_mv_teaser_insufficient_frames++;
      return consensus;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> source_matrix(
        3, static_cast<Eigen::Index>(selected_by_cell.size()));
    Eigen::Matrix<double, 3, Eigen::Dynamic> destination_matrix(
        3, static_cast<Eigen::Index>(selected_by_cell.size()));
    Eigen::Index column = 0;
    for (const auto& [cell, candidate_index] : selected_by_cell) {
      static_cast<void>(cell);
      source_matrix.col(column) = candidate_sources.at(candidate_index);
      destination_matrix.col(column) = candidate_destinations.at(candidate_index);
      column++;
    }

    const DirectMvRigidRegistrationResult result = solveDirectMvTeaser(
        source_matrix, destination_matrix,
        static_cast<double>(config.optical_flow_direct_mv_teaser_noise_bound_m));
    direct_mv_teaser_solver_inliers += result.input_inliers.size();
    if (!result.valid) {
      direct_mv_teaser_failed_frames++;
      return consensus;
    }

    consensus.valid = true;
    consensus.rotation = result.rotation;
    consensus.translation = result.translation;
    direct_mv_teaser_successful_frames++;
    return consensus;
  }

  bool directTeaserAcceptsTemporalEdge(const DirectTeaserConsensus& consensus,
                                       const Vector2& source, Scalar source_depth,
                                       const Vector2& destination,
                                       Scalar destination_depth,
                                       double* accepted_residual = nullptr) {
    if (accepted_residual) {
      *accepted_residual = std::numeric_limits<double>::infinity();
    }
    direct_mv_teaser_validated_edges++;
    if (!consensus.valid) {
      direct_mv_teaser_rejected_no_model++;
      return false;
    }

    Eigen::Vector3d source_3d;
    Eigen::Vector3d destination_3d;
    if (!directStereoPoint3d(source, source_depth, source_3d) ||
        !directStereoPoint3d(destination, destination_depth, destination_3d)) {
      direct_mv_teaser_rejected_invalid_depth++;
      return false;
    }

    const Eigen::Vector3d predicted = consensus.rotation * source_3d +
                                      consensus.translation;
    const double residual = (predicted - destination_3d).norm();
    if (!std::isfinite(residual) ||
        residual > static_cast<double>(
                       config.optical_flow_direct_mv_teaser_noise_bound_m)) {
      direct_mv_teaser_rejected_metric++;
      return false;
    }

    if (std::max(source_depth, destination_depth) >
        Scalar(config.optical_flow_direct_mv_teaser_far_depth_m)) {
      const Vector3 predicted_scalar = predicted.template cast<Scalar>();
      Vector2 predicted_pixel;
      if (!calib.intrinsics.at(0).project(predicted_scalar, predicted_pixel) ||
          !predicted_pixel.array().isFinite().all() ||
          (predicted_pixel - destination).norm() >
              Scalar(config.optical_flow_direct_mv_teaser_far_reprojection_gate_px)) {
        direct_mv_teaser_rejected_far_reprojection++;
        return false;
      }
    }

    if (accepted_residual) *accepted_residual = residual;
    direct_mv_teaser_accepted_edges++;
    return true;
  }

  bool sampleDirectTeaserTemporalTrack(
      const DirectTemporalField& field,
      const Vector2& source_point,
      Scalar source_depth, const DirectStereoField& current_stereo_field,
      const Matrix4& essential, const SE3& T_0_1,
      int rectification_mode, const DirectTeaserConsensus& consensus,
      size_t& best_index, Vector2& destination_point,
      DirectStereoProjection& destination_stereo,
      double& selected_residual, Scalar& selected_center_cost) {
    direct_mv_teaser_authority_queries++;
    selected_residual = std::numeric_limits<double>::infinity();
    selected_center_cost = std::numeric_limits<Scalar>::max();
    const std::vector<size_t>* candidates =
        directTemporalCandidates(field, source_point);
    if (!candidates) {
      direct_mv_teaser_authority_no_candidate++;
      direct_mv_teaser_authority_failed_no_spatial_bin++;
      return false;
    }
    const auto& tracks = *field.tracks;
    bool found = false;
    bool saw_available = false;
    bool saw_covering = false;
    bool saw_valid_destination = false;
    bool saw_current_stereo = false;
    bool saw_teaser_acceptance = false;
    double best_residual = std::numeric_limits<double>::infinity();
    Scalar best_center_cost = std::numeric_limits<Scalar>::max();
    std::array<bool, 4> origin_found{};
    std::array<double, 4> origin_best_residual;
    std::array<Scalar, 4> origin_best_center_cost;
    std::array<Vector2, 4> origin_best_destination;
    std::array<Scalar, 4> origin_best_depth;
    origin_best_residual.fill(std::numeric_limits<double>::infinity());
    origin_best_center_cost.fill(std::numeric_limits<Scalar>::max());

    for (size_t index : *candidates) {
      direct_mv_teaser_authority_indexed_candidates++;
      saw_available = true;
      const MotionVector& mv = *tracks.at(index).mv;
      const Scalar half_width = Scalar(mv.width) / Scalar(2);
      const Scalar half_height = Scalar(mv.height) / Scalar(2);
      if (std::abs(source_point.x() - Scalar(mv.src_x)) > half_width ||
          std::abs(source_point.y() - Scalar(mv.src_y)) > half_height) {
        direct_mv_teaser_authority_rejected_not_covering_source++;
        continue;
      }
      saw_covering = true;

      const Vector2 candidate = directMotionVectorDestination(source_point, mv);
      if (!candidate.array().isFinite().all() || candidate.x() < Scalar(0) ||
          candidate.y() < Scalar(0) || candidate.x() >= Scalar(w) ||
          candidate.y() >= Scalar(h)) {
        direct_mv_teaser_authority_rejected_invalid_destination++;
        continue;
      }
      saw_valid_destination = true;

      DirectStereoProjection candidate_stereo;
      if (!findDirectStereoAtPoint(current_stereo_field, candidate, essential,
                                   T_0_1, rectification_mode,
                                   candidate_stereo)) {
        direct_mv_teaser_authority_rejected_current_stereo++;
        continue;
      }
      saw_current_stereo = true;

      direct_mv_teaser_authority_candidate_edges++;
      double residual = std::numeric_limits<double>::infinity();
      if (!directTeaserAcceptsTemporalEdge(
              consensus, source_point, source_depth, candidate,
              candidate_stereo.depth, &residual)) {
        direct_mv_teaser_authority_rejected_teaser++;
        continue;
      }
      saw_teaser_acceptance = true;
      direct_mv_teaser_authority_accepted_candidates++;

      const Scalar center_cost =
          (source_point.x() - Scalar(mv.src_x)) *
                  (source_point.x() - Scalar(mv.src_x)) /
              std::max(half_width * half_width, Scalar(1)) +
          (source_point.y() - Scalar(mv.src_y)) *
                  (source_point.y() - Scalar(mv.src_y)) /
              std::max(half_height * half_height, Scalar(1));
      const int raw_origin = tracks.at(index).temporal_origin;
      const size_t origin =
          raw_origin >= DIRECT_TEMPORAL_ORIGIN_UNKNOWN &&
                  raw_origin <= DIRECT_TEMPORAL_ORIGIN_RECIPROCAL
              ? static_cast<size_t>(raw_origin)
              : static_cast<size_t>(DIRECT_TEMPORAL_ORIGIN_UNKNOWN);
      if (!origin_found.at(origin) ||
          residual < origin_best_residual.at(origin) ||
          (residual == origin_best_residual.at(origin) &&
           center_cost < origin_best_center_cost.at(origin))) {
        origin_found.at(origin) = true;
        origin_best_residual.at(origin) = residual;
        origin_best_center_cost.at(origin) = center_cost;
        origin_best_destination.at(origin) = candidate;
        origin_best_depth.at(origin) = candidate_stereo.depth;
      }
      if (!found || residual < best_residual ||
          (residual == best_residual && center_cost < best_center_cost)) {
        found = true;
        best_residual = residual;
        best_center_cost = center_cost;
        best_index = index;
        destination_point = candidate;
        destination_stereo = candidate_stereo;
      }
    }

    if (!found) {
      direct_mv_teaser_authority_no_candidate++;
      if (!saw_available) {
        direct_mv_teaser_authority_failed_no_available_candidate++;
      } else if (!saw_covering) {
        direct_mv_teaser_authority_failed_no_covering_candidate++;
      } else if (!saw_valid_destination) {
        direct_mv_teaser_authority_failed_invalid_destination++;
      } else if (!saw_current_stereo) {
        direct_mv_teaser_authority_failed_current_stereo++;
      } else if (!saw_teaser_acceptance) {
        direct_mv_teaser_authority_failed_teaser_gate++;
      }
      return false;
    }
    direct_mv_teaser_authority_selected_edges++;
    const bool has_forward =
        origin_found.at(DIRECT_TEMPORAL_ORIGIN_FORWARD);
    const bool has_reverse =
        origin_found.at(DIRECT_TEMPORAL_ORIGIN_REVERSE);
    if (has_forward && has_reverse) {
      direct_mv_teaser_authority_bidirectional_accepted_queries++;
      const double residual_difference = std::abs(
          origin_best_residual.at(DIRECT_TEMPORAL_ORIGIN_FORWARD) -
          origin_best_residual.at(DIRECT_TEMPORAL_ORIGIN_REVERSE));
      const double destination_difference =
          static_cast<double>(
              (origin_best_destination.at(DIRECT_TEMPORAL_ORIGIN_FORWARD) -
               origin_best_destination.at(DIRECT_TEMPORAL_ORIGIN_REVERSE))
                  .norm());
      const double depth_difference = std::abs(static_cast<double>(
          origin_best_depth.at(DIRECT_TEMPORAL_ORIGIN_FORWARD) -
          origin_best_depth.at(DIRECT_TEMPORAL_ORIGIN_REVERSE)));
      direct_mv_teaser_authority_bidirectional_residual_difference_sum_m +=
          residual_difference;
      direct_mv_teaser_authority_bidirectional_destination_difference_sum_px +=
          destination_difference;
      direct_mv_teaser_authority_bidirectional_depth_difference_sum_m +=
          depth_difference;
      direct_mv_teaser_authority_bidirectional_residual_difference_max_m =
          std::max(
              direct_mv_teaser_authority_bidirectional_residual_difference_max_m,
              residual_difference);
      direct_mv_teaser_authority_bidirectional_destination_difference_max_px =
          std::max(
              direct_mv_teaser_authority_bidirectional_destination_difference_max_px,
              destination_difference);
      direct_mv_teaser_authority_bidirectional_depth_difference_max_m =
          std::max(
              direct_mv_teaser_authority_bidirectional_depth_difference_max_m,
              depth_difference);
    } else if (has_forward) {
      direct_mv_teaser_authority_forward_only_accepted_queries++;
    } else if (has_reverse) {
      direct_mv_teaser_authority_reverse_only_accepted_queries++;
    } else {
      direct_mv_teaser_authority_other_origin_accepted_queries++;
    }
    selected_residual = best_residual;
    selected_center_cost = best_center_cost;
    return true;
  }

  void insertDirectPoint(const OpticalFlowResult::Ptr& result, size_t camera, KeypointId id,
                         const Vector2& point) {
    Keypoint keypoint = Keypoint::Identity();
    keypoint.translation() = point.template cast<float>();
    const auto [iterator, inserted] = result->keypoints.at(camera).emplace(id, keypoint);
    if (!inserted && (iterator->second.translation() - keypoint.translation()).norm() > 1e-3f) {
      std::cerr << "Conflicting direct-MV observation for ID " << id << " on camera " << camera << "."
                << std::endl;
      std::abort();
    }
    result->keypoint_responses.at(camera).emplace(id, -1.0f);
  }

  void insertDirectStereoDepth(const OpticalFlowResult::Ptr& result, size_t camera,
                               KeypointId id, Scalar depth) const {
    if (!result || camera >= result->keypoints.size() || !std::isfinite(depth) ||
        depth <= Scalar(0)) {
      return;
    }
    if (result->keypoint_stereo_depths.size() < result->keypoints.size())
      result->keypoint_stereo_depths.resize(result->keypoints.size());

    result->keypoint_stereo_depths.at(camera).emplace(id,
                                                       static_cast<float>(depth));
  }

  void recordDirectRigidFlows(const OpticalFlowResult::Ptr& previous,
                                  const OpticalFlowResult::Ptr& current) const {
    if (!previous || !current || previous->t_ns >= current->t_ns) return;
    current->keypoint_rigid_flows.resize(current->keypoints.size());
    const size_t num_cams = std::min(previous->keypoints.size(), current->keypoints.size());
    for (size_t camera = 0; camera < num_cams; camera++) {
      for (const auto& [id, current_keypoint] : current->keypoints.at(camera)) {
        const auto previous_it = previous->keypoints.at(camera).find(id);
        if (previous_it == previous->keypoints.at(camera).end()) continue;
        KeypointRigidFlow flow;
        flow.predecessor_t_ns = previous->t_ns;
        flow.displacement = current_keypoint.translation() - previous_it->second.translation();
        if (flow.displacement.array().isFinite().all())
          current->keypoint_rigid_flows.at(camera).emplace(id, flow);
      }
    }
  }

  void processDirectMotionVectorFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    constexpr float kMedianDistanceThreshold = 5.0f;
    const size_t num_cams = getNumCams();
    auto current = makeDirectMotionVectorResult(curr_t_ns, new_img_vec);
    if (!new_img_vec->img_data.empty() &&
        new_img_vec->img_data.at(0).stereo_motion_vectors_available) {
      configureDirectStereoBaselineRectification(new_img_vec->img_data.at(0));
    }
    std::vector<std::vector<DirectMotionVectorTrack>> direct_tracks(num_cams);
    std::vector<std::vector<DirectMotionVectorTrack>> strict_direct_tracks(
        num_cams);

    const auto append_direct_tracks =
        [&](size_t cam, const ImageData& image_data,
            const std::vector<MotionVector>& mvs, bool reciprocal_pool,
            std::vector<DirectMotionVectorTrack>& output) {
          const auto median_dist = scoreMVsMedianDistance(mvs);
          const Scalar safe_radius =
              Scalar(config.optical_flow_image_safe_radius);
          const Vector2 image_center(Scalar(w) / Scalar(2),
                                     Scalar(h) / Scalar(2));
          if (!reciprocal_pool && cam == 0)
            direct_mv_cam0_input_vectors += mvs.size();

          for (size_t i = 0; i < mvs.size(); ++i) {
            const MotionVector& mv = mvs.at(i);
            if (!reciprocal_pool && mv.motion_scale > 0) {
              direct_mv_temporal_motion_scale_histogram[mv.motion_scale]++;
              if (std::abs(mv.quantized_flow_x -
                           std::round(mv.quantized_flow_x)) > 1e-6f ||
                  std::abs(mv.quantized_flow_y -
                           std::round(mv.quantized_flow_y)) > 1e-6f) {
                direct_mv_temporal_subpixel_vectors++;
              }
            }
            const float score = median_dist.at(i);
            if (mv.source != -1) {
              if (!reciprocal_pool && cam == 0)
                direct_mv_cam0_rejected_nonpast_source++;
              continue;
            }
            if (!std::isfinite(score) || score > kMedianDistanceThreshold) {
              if (!reciprocal_pool && cam == 0)
                direct_mv_cam0_rejected_median_filter++;
              continue;
            }
            const Vector2 source(Scalar(mv.src_x), Scalar(mv.src_y));
            const Vector2 destination =
                directMotionVectorDestination(source, mv);
            if (!source.array().isFinite().all() ||
                !destination.array().isFinite().all() ||
                source.x() < Scalar(0) || source.y() < Scalar(0) ||
                source.x() >= Scalar(w) || source.y() >= Scalar(h) ||
                destination.x() < Scalar(0) ||
                destination.y() < Scalar(0) ||
                destination.x() >= Scalar(w) ||
                destination.y() >= Scalar(h)) {
              if (!reciprocal_pool && cam == 0)
                direct_mv_cam0_rejected_image_bounds++;
              continue;
            }
            if (safe_radius > Scalar(0) &&
                ((source - image_center).norm() > safe_radius ||
                 (destination - image_center).norm() > safe_radius)) {
              if (!reciprocal_pool && cam == 0)
                direct_mv_cam0_rejected_safe_radius++;
              continue;
            }
            if (transforms->input_images->masks.at(cam).inBounds(
                    source.x(), source.y()) ||
                new_img_vec->masks.at(cam).inBounds(destination.x(),
                                                     destination.y())) {
              if (!reciprocal_pool && cam == 0)
                direct_mv_cam0_rejected_mask++;
              continue;
            }
            DirectMotionVectorTrack track;
            track.mv = &mv;
            if (reciprocal_pool) {
              if (image_data.temporal_single_direction_fallback) {
                track.temporal_origin =
                    image_data.temporal_forward_is_i_frame
                        ? DIRECT_TEMPORAL_ORIGIN_REVERSE
                        : DIRECT_TEMPORAL_ORIGIN_FORWARD;
              } else {
                track.temporal_origin = DIRECT_TEMPORAL_ORIGIN_RECIPROCAL;
              }
            } else if (image_data.temporal_reverse_policy == 2 ||
                       image_data.temporal_reverse_policy == 3) {
              track.temporal_origin =
                  i < image_data.temporal_forward_motion_vector_count
                      ? DIRECT_TEMPORAL_ORIGIN_FORWARD
                      : DIRECT_TEMPORAL_ORIGIN_REVERSE;
            } else if (image_data.temporal_reverse_policy == 1) {
              if (image_data.temporal_single_direction_fallback) {
                track.temporal_origin =
                    image_data.temporal_forward_is_i_frame
                        ? DIRECT_TEMPORAL_ORIGIN_REVERSE
                        : DIRECT_TEMPORAL_ORIGIN_FORWARD;
              } else {
                track.temporal_origin = DIRECT_TEMPORAL_ORIGIN_RECIPROCAL;
              }
            } else {
              track.temporal_origin = DIRECT_TEMPORAL_ORIGIN_FORWARD;
            }
            track.bearings_valid =
                calib.intrinsics[cam].unproject(source,
                                                track.source_bearing) &&
                calib.intrinsics[cam].unproject(destination,
                                                track.destination_bearing);
            output.emplace_back(track);
            if (reciprocal_pool) {
              if (cam == 0) direct_mv_hybrid_reciprocal_tracks++;
            } else {
              direct_mv_tracks++;
              if (cam == 0) direct_mv_cam0_accepted_tracks++;
            }
          }
        };

    for (size_t cam = 0; cam < num_cams; ++cam) {
      if (!transforms) continue;

      const ImageData& image_data = new_img_vec->img_data.at(cam);
      const auto& mvs = image_data.motion_vectors;
      if (cam == 0 &&
          new_img_vec->img_data.at(cam).temporal_reverse_policy != 0) {
        const ImageData& image = new_img_vec->img_data.at(cam);
        direct_mv_reverse_temporal_frames++;
        direct_mv_reverse_temporal_policy = image.temporal_reverse_policy;
        direct_mv_reverse_temporal_forward_vectors +=
            image.temporal_forward_motion_vector_count;
        direct_mv_reverse_temporal_reverse_vectors +=
            image.temporal_reverse_motion_vector_count;
        direct_mv_reverse_temporal_reciprocal_vectors +=
            image.temporal_reciprocal_motion_vector_count;
        if (image.temporal_single_direction_fallback) {
          direct_mv_reverse_temporal_single_direction_fallback_frames++;
          direct_mv_reverse_temporal_single_direction_fallback_vectors +=
              image.temporal_single_direction_fallback_motion_vector_count;
        }
        direct_mv_reverse_temporal_output_vectors += mvs.size();
      }
      append_direct_tracks(cam, image_data, mvs, false,
                           direct_tracks.at(cam));
      if (image_data.temporal_reverse_policy == 3) {
        append_direct_tracks(
            cam, image_data,
            image_data.temporal_reciprocal_motion_vectors, true,
            strict_direct_tracks.at(cam));
      }
    }

    if (transforms) {
      std::vector<DirectTemporalField> direct_temporal_fields;
      direct_temporal_fields.reserve(direct_tracks.size());
      for (const auto& camera_tracks : direct_tracks) {
        direct_temporal_fields.emplace_back(
            makeDirectTemporalField(camera_tracks));
      }
      const bool strict_confirmation =
          config.optical_flow_direct_mv_strict_confirmation;
      const auto& continuation_tracks =
          strict_confirmation ? strict_direct_tracks.front()
                              : direct_tracks.front();
      const DirectTemporalField continuation_field =
          strict_confirmation
              ? makeDirectTemporalField(strict_direct_tracks.front())
              : direct_temporal_fields.front();

      const bool codec_stereo_available =
          num_cams >= 2 && transforms->input_images && transforms->input_images->img_data.size() >= 2 &&
          new_img_vec->img_data.size() >= 2 &&
          transforms->input_images->img_data.at(0).stereo_motion_vectors_available &&
          new_img_vec->img_data.at(0).stereo_motion_vectors_available;

      if (!codec_stereo_available) {
        std::cerr << "Passthrough stereo artifacts are required." << std::endl;
        std::abort();
      }
      {
        direct_mv_codec_stereo_frames++;
        if (new_img_vec->img_data.at(0).stereo_rectification_validity_mask_available) {
          direct_mv_stereo_rectification_mask_frames++;
        }
        direct_mv_stereo_rectification_mask_rejected_vectors +=
            new_img_vec->img_data.at(0)
                .stereo_motion_vectors_rejected_by_rectification_mask;
        const int rectification_mode =
            new_img_vec->img_data.at(0).stereo_motion_vector_rectification_mode;
        direct_mv_stereo_rectification_mode = rectification_mode;
        if (transforms->input_images->img_data.at(0)
                .stereo_motion_vector_rectification_mode != rectification_mode) {
          std::cerr << "Stereo MV rectification changed within a sequence." << std::endl;
          std::abort();
        }
        const SE3 T_0_1 = calib.T_i_c[0].inverse() * calib.T_i_c[1];
        const int merge_bin_size = std::max(
            1, static_cast<int>(std::ceil(
                   config.optical_flow_direct_mv_merge_radius_px)));
        std::map<std::pair<int, int>,
                 std::vector<DirectAcceptedMeasurement>>
            accepted_measurements;
        const auto retain_measurement =
            [&](const DirectAcceptedMeasurement& measurement) {
              if (!config.optical_flow_direct_mv_merge_indistinguishable)
                return true;
              const int bin_x = static_cast<int>(std::floor(
                  measurement.previous_left.x() / Scalar(merge_bin_size)));
              const int bin_y = static_cast<int>(std::floor(
                  measurement.previous_left.y() / Scalar(merge_bin_size)));
              for (int offset_y = -1; offset_y <= 1; ++offset_y) {
                for (int offset_x = -1; offset_x <= 1; ++offset_x) {
                  const auto found = accepted_measurements.find(
                      {bin_x + offset_x, bin_y + offset_y});
                  if (found == accepted_measurements.end()) continue;
                  for (const auto& retained : found->second) {
                    if (directMeasurementsIndistinguishable(
                            retained, measurement, T_0_1)) {
                      return false;
                    }
                  }
                }
              }
              accepted_measurements[{bin_x, bin_y}].push_back(measurement);
              return true;
            };
        Eigen::Matrix4d essential_double;
        computeEssential(T_0_1.template cast<double>(), essential_double);
        const Matrix4 essential = essential_double.template cast<Scalar>();
        for (const MotionVector& stereo_mv :
             new_img_vec->img_data.at(0).stereo_motion_vectors) {
          if (stereo_mv.motion_scale > 0)
            direct_mv_stereo_motion_scale_histogram[stereo_mv.motion_scale]++;
        }
        const DirectStereoField previous_stereo_field = makeDirectStereoField(
            transforms->input_images->img_data.at(0).stereo_motion_vectors);
        const DirectStereoField current_stereo_field =
            makeDirectStereoField(new_img_vec->img_data.at(0).stereo_motion_vectors);
        std::vector<DirectTrackStereoProjections> direct_track_stereo_cache(
            direct_tracks.front().size());
        std::vector<DirectTrackStereoProjections> consensus_stereo_cache(
            continuation_tracks.size());
        std::vector<bool> anchor_used(direct_tracks.front().size(), false);
        std::map<KeypointId, DirectActiveTrack> next_active_tracks;
        std::vector<DirectSelectedMvUse> selected_mv_uses(
            continuation_tracks.size());
        std::vector<DirectSelectedMvAssignment> selected_mv_assignments;
        selected_mv_assignments.reserve(direct_active_tracks.size());
        const bool temporal_field_reused = new_img_vec->img_data.at(0).motion_vectors_reused_from_i_frame;
        const DirectTeaserConsensus teaser_consensus =
            temporal_field_reused
                ? DirectTeaserConsensus{}
                : estimateDirectTeaserConsensus(continuation_tracks, previous_stereo_field,
                                                current_stereo_field, essential, T_0_1,
                                                rectification_mode,
                                                &consensus_stereo_cache);
        if (!temporal_field_reused) {
          for (const auto& [id, active] : direct_active_tracks) {
            direct_mv_track_continuation_attempts++;
            const auto pending_it = transforms->keypoints.at(0).find(id);
            if (active.confirmed &&
                (pending_it == transforms->keypoints.at(0).end() ||
                 (pending_it->second.translation().template cast<Scalar>() -
                  active.left_point)
                         .norm() > Scalar(1e-3))) {
              direct_mv_persistent_tracks_killed++;
              recordDirectCam0TrackLength(active.age_frames);
              recordDirectTrackQuality(
                  active, DIRECT_TRACK_TERMINATION_OPTIMIZER_OBSERVATION);
              direct_mv_track_rejected_optimizer_observation++;
              continue;
            }

            size_t temporal_index = 0;
            Vector2 current_left;
            DirectStereoProjection previous_stereo;
            DirectStereoProjection current_stereo;
            bool previous_stereo_valid = false;
            bool temporal_match_valid = false;
            double selected_teaser_residual =
                std::numeric_limits<double>::infinity();
            Scalar selected_center_cost =
                std::numeric_limits<Scalar>::max();
            previous_stereo_valid = findDirectStereoAtPoint(
                previous_stereo_field, active.left_point, essential, T_0_1,
                rectification_mode, previous_stereo);
            temporal_match_valid =
                previous_stereo_valid &&
                sampleDirectTeaserTemporalTrack(
                    continuation_field, active.left_point,
                    previous_stereo.depth, current_stereo_field, essential,
                    T_0_1, rectification_mode, teaser_consensus,
                    temporal_index, current_left, current_stereo,
                    selected_teaser_residual, selected_center_cost);
            if (!temporal_match_valid) {
              if (active.confirmed) {
                direct_mv_persistent_tracks_killed++;
                recordDirectCam0TrackLength(active.age_frames);
                recordDirectTrackQuality(
                    active, previous_stereo_valid
                                ? DIRECT_TRACK_TERMINATION_TEMPORAL_MATCH
                                : DIRECT_TRACK_TERMINATION_PREVIOUS_STEREO);
              } else {
                direct_mv_provisional_tracks_rejected++;
              }
              if (previous_stereo_valid)
                direct_mv_track_rejected_temporal_match++;
              else
                direct_mv_track_rejected_previous_stereo++;
              continue;
            }
            const DirectAcceptedMeasurement accepted_measurement{
                active.left_point, current_left, previous_stereo.depth,
                current_stereo.depth};
            if (!retain_measurement(accepted_measurement)) {
              if (active.confirmed) {
                direct_mv_persistent_tracks_killed++;
                recordDirectCam0TrackLength(active.age_frames);
                recordDirectTrackQuality(
                    active, DIRECT_TRACK_TERMINATION_POST_GATE);
              } else {
                direct_mv_provisional_tracks_rejected++;
              }
              direct_mv_merged_continuations++;
              continue;
            }
            const size_t current_age_frames =
                active.confirmed ? active.age_frames + 1 : 3;
            if (active.last_stereo_depth_valid) {
              recordDirectTemporalDepthResidual(
                  active.last_stereo_depth, current_stereo.depth,
                  current_age_frames - active.last_stereo_age_frames,
                  active.last_stereo_t_ns, curr_t_ns);
            }
            const MotionVector& temporal_mv =
                *continuation_tracks.at(temporal_index).mv;
            if (!active.confirmed) {
              const OpticalFlowResult::Ptr source_result =
                  pendingDirectMotionVectorFrame(
                      active.provisional_source_t_ns);
              if (!source_result) {
                std::cerr << "Missing provisional direct-MV source frame."
                          << std::endl;
                std::abort();
              }
              insertDirectPoint(source_result, 0, id,
                                active.provisional_source_left);
              insertDirectPoint(source_result, 1, id,
                                active.provisional_source_right);
              insertDirectStereoDepth(source_result, 0, id,
                                      active.provisional_source_depth);
              insertDirectStereoDepth(source_result, 1, id,
                                      active.provisional_source_depth);
              insertDirectPoint(transforms, 1, id, previous_stereo.point);
              insertDirectStereoDepth(transforms, 0, id,
                                      previous_stereo.depth);
              insertDirectStereoDepth(transforms, 1, id,
                                      previous_stereo.depth);
              recordDirectRigidFlows(source_result, transforms);
              recordDirectTemporalDepthResidual(
                  active.provisional_source_depth, previous_stereo.depth, 1,
                  active.provisional_source_t_ns, transforms->t_ns);
              direct_mv_codec_stereo_matches += 2;
              direct_mv_stereo_matches += 2;
              direct_mv_provisional_tracks_confirmed++;
              direct_mv_persistent_tracks_started++;
              direct_mv_teaser_authority_started++;
            }
            insertDirectPoint(transforms, 0, id, active.left_point);
            insertDirectPoint(current, 0, id, current_left);
            insertDirectPoint(current, 1, id, current_stereo.point);
            insertDirectStereoDepth(current, 0, id, current_stereo.depth);
            insertDirectStereoDepth(current, 1, id, current_stereo.depth);
            direct_mv_codec_stereo_matches++;
            direct_mv_stereo_matches++;
            if (!strict_confirmation) anchor_used.at(temporal_index) = true;
            DirectActiveTrack continued = active;
            continued.left_point = current_left;
            continued.age_frames = current_age_frames;
            continued.confirmed = true;
              continued.diagnostic_selection_count++;
              continued.diagnostic_teaser_residual_sum_m +=
                  selected_teaser_residual;
              continued.diagnostic_center_cost_sum += selected_center_cost;
              continued.diagnostic_depth_delta_sum_m += std::abs(
                  static_cast<double>(current_stereo.depth -
                                      previous_stereo.depth));
              if (current_stereo.depth > Scalar(0) &&
                  previous_stereo.depth > Scalar(0)) {
                continued.diagnostic_depth_log_ratio_sum += std::abs(std::log(
                    static_cast<double>(current_stereo.depth /
                                        previous_stereo.depth)));
              }
              continued.diagnostic_stereo_cost_sum +=
                  static_cast<double>(previous_stereo.cost +
                                      current_stereo.cost);
              continued.diagnostic_block_area_sum_px2 +=
                  static_cast<long double>(temporal_mv.width) *
                  temporal_mv.height;
              const int temporal_origin =
                  continuation_tracks.at(temporal_index).temporal_origin;
              const size_t origin_index =
                  temporal_origin >= DIRECT_TEMPORAL_ORIGIN_UNKNOWN &&
                          temporal_origin <= DIRECT_TEMPORAL_ORIGIN_RECIPROCAL
                      ? static_cast<size_t>(temporal_origin)
                      : static_cast<size_t>(DIRECT_TEMPORAL_ORIGIN_UNKNOWN);
              DirectOriginSelectionAggregate& origin_quality =
                  direct_mv_selection_quality_by_origin.at(origin_index);
              origin_quality.selections++;
              origin_quality.teaser_residual_sum_m +=
                  selected_teaser_residual;
              origin_quality.center_cost_sum += selected_center_cost;
              origin_quality.depth_delta_sum_m += std::abs(
                  static_cast<double>(current_stereo.depth -
                                      previous_stereo.depth));
              if (current_stereo.depth > Scalar(0) &&
                  previous_stereo.depth > Scalar(0)) {
                origin_quality.depth_log_ratio_sum += std::abs(std::log(
                    static_cast<double>(current_stereo.depth /
                                        previous_stereo.depth)));
              }
              origin_quality.stereo_cost_sum +=
                  static_cast<double>(previous_stereo.cost +
                                      current_stereo.cost);
              origin_quality.block_area_sum_px2 +=
                  static_cast<long double>(temporal_mv.width) *
                  temporal_mv.height;
              if (temporal_origin == DIRECT_TEMPORAL_ORIGIN_FORWARD) {
                continued.diagnostic_forward_selection_count++;
                direct_mv_selected_forward_assignments++;
              } else if (temporal_origin == DIRECT_TEMPORAL_ORIGIN_REVERSE) {
                continued.diagnostic_reverse_selection_count++;
                direct_mv_selected_reverse_assignments++;
              } else if (temporal_origin ==
                         DIRECT_TEMPORAL_ORIGIN_RECIPROCAL) {
                continued.diagnostic_reciprocal_selection_count++;
                direct_mv_selected_reciprocal_assignments++;
              } else {
                direct_mv_selected_unknown_assignments++;
              }

              DirectSelectedMvUse& use =
                  selected_mv_uses.at(temporal_index);
              use.count++;
              use.previous_depth_min_m = std::min(
                  use.previous_depth_min_m,
                  static_cast<double>(previous_stereo.depth));
              use.previous_depth_max_m = std::max(
                  use.previous_depth_max_m,
                  static_cast<double>(previous_stereo.depth));
              use.current_depth_min_m = std::min(
                  use.current_depth_min_m,
                  static_cast<double>(current_stereo.depth));
              use.current_depth_max_m = std::max(
                  use.current_depth_max_m,
                  static_cast<double>(current_stereo.depth));
              use.teaser_residual_min_m = std::min(
                  use.teaser_residual_min_m, selected_teaser_residual);
              use.teaser_residual_max_m = std::max(
                  use.teaser_residual_max_m, selected_teaser_residual);
              selected_mv_assignments.push_back({id, temporal_index});
            continued.last_stereo_depth = current_stereo.depth;
            continued.last_stereo_t_ns = curr_t_ns;
            continued.last_stereo_age_frames = current_age_frames;
            continued.last_stereo_depth_valid = true;
            next_active_tracks.emplace(id, continued);
            direct_mv_persistent_track_continuations++;
            direct_mv_teaser_authority_continuations++;
            direct_mv_max_track_frames = std::max(direct_mv_max_track_frames, continued.age_frames);
          }
          recordDirectSelectedMvSharing(selected_mv_uses,
                                        selected_mv_assignments,
                                        next_active_tracks);
        } else if (temporal_field_reused) {
          for (const auto& [id, active] : direct_active_tracks) {
            static_cast<void>(id);
            if (active.confirmed) {
              recordDirectCam0TrackLength(active.age_frames);
              recordDirectTrackQuality(active,
                                       DIRECT_TRACK_TERMINATION_I_FRAME);
            } else {
              direct_mv_provisional_tracks_rejected++;
            }
          }
          direct_mv_track_rejected_i_frame += direct_active_tracks.size();
          direct_mv_persistent_tracks_killed += direct_active_tracks.size();
        }

        if (!temporal_field_reused) {
          for (size_t index = 0; index < direct_tracks.front().size(); ++index) {
            if (anchor_used.at(index)) continue;
            DirectMotionVectorTrack& track = direct_tracks.front().at(index);
            const MotionVector& mv = *track.mv;
            const Vector2 source(Scalar(mv.src_x), Scalar(mv.src_y));
            const Vector2 destination = directMotionVectorDestination(source, mv);
            DirectTrackStereoProjections& cached_stereo =
                direct_track_stereo_cache.at(index);
            if (!cached_stereo.previous_evaluated) {
              cached_stereo.previous_valid = findDirectStereoAtPoint(
                  previous_stereo_field, source, essential, T_0_1,
                  rectification_mode, cached_stereo.previous);
              cached_stereo.previous_evaluated = true;
            }
            if (!cached_stereo.current_evaluated) {
              cached_stereo.current_valid = findDirectStereoAtPoint(
                  current_stereo_field, destination, essential, T_0_1,
                  rectification_mode, cached_stereo.current);
              cached_stereo.current_evaluated = true;
            }
            const bool previous_stereo_valid = cached_stereo.previous_valid;
            const bool current_stereo_valid = cached_stereo.current_valid;
            const DirectStereoProjection& previous_stereo =
                cached_stereo.previous;
            const DirectStereoProjection& current_stereo = cached_stereo.current;
            if (!previous_stereo_valid || !current_stereo_valid ||
                !directTeaserAcceptsTemporalEdge(
                    teaser_consensus, source, previous_stereo.depth, destination,
                    current_stereo.depth)) {
              continue;
            }

            const DirectAcceptedMeasurement accepted_measurement{
                source, destination, previous_stereo.depth,
                current_stereo.depth};
            if (!retain_measurement(accepted_measurement)) {
              direct_mv_merged_births++;
              continue;
            }

            track.id = last_keypoint_id++;
            DirectActiveTrack active;
            active.id = track.id;
            active.left_point = destination;
            active.last_stereo_depth = current_stereo.depth;
            active.last_stereo_t_ns = curr_t_ns;
            active.last_stereo_age_frames = 2;
            active.last_stereo_depth_valid = true;
            if (strict_confirmation) {
              active.provisional_source_left = source;
              active.provisional_source_right = previous_stereo.point;
              active.provisional_source_depth = previous_stereo.depth;
              active.provisional_source_t_ns = transforms->t_ns;
              active.confirmed = false;
              direct_mv_provisional_tracks_started++;
            } else {
              insertDirectPoint(transforms, 0, track.id, source);
              insertDirectPoint(current, 0, track.id, destination);
              insertDirectPoint(transforms, 1, track.id,
                                previous_stereo.point);
              insertDirectPoint(current, 1, track.id,
                                current_stereo.point);
              insertDirectStereoDepth(transforms, 0, track.id,
                                      previous_stereo.depth);
              insertDirectStereoDepth(transforms, 1, track.id,
                                      previous_stereo.depth);
              insertDirectStereoDepth(current, 0, track.id,
                                      current_stereo.depth);
              insertDirectStereoDepth(current, 1, track.id,
                                      current_stereo.depth);
              direct_mv_codec_stereo_matches += 2;
              direct_mv_stereo_matches += 2;
              recordDirectTemporalDepthResidual(
                  previous_stereo.depth, current_stereo.depth, 1,
                  transforms->t_ns, curr_t_ns);
              direct_mv_persistent_tracks_started++;
              direct_mv_teaser_authority_started++;
            }
            next_active_tracks.emplace(track.id, active);
            direct_mv_max_track_frames = std::max(direct_mv_max_track_frames, active.age_frames);
          }
        }

        direct_active_tracks = std::move(next_active_tracks);
      }
    }

    recordDirectRigidFlows(transforms, current);
    transforms = std::move(current);
    pending_direct_frames.push_back({frame_counter, transforms});
    pushPendingDirectMotionVectorFrames(false);
    t_ns = curr_t_ns;
    frame_counter++;
  }

  void saveFrontendStats() {
    std::ofstream stats{"stats_frontend.json"};
    std::map<size_t, uint64_t> observed_track_lengths = direct_mv_cam0_track_length_histogram_frames;
    uint64_t active_confirmed_track_count = 0;
    for (const auto& [id, active] : direct_active_tracks) {
      static_cast<void>(id);
      if (!active.confirmed) continue;
      active_confirmed_track_count++;
      observed_track_lengths[active.age_frames]++;
    }
    uint64_t finalized_track_count = 0;
    uint64_t observed_track_count = 0;
    long double observed_track_length_sum = 0;
    for (const auto& [age_frames, count] : direct_mv_cam0_track_length_histogram_frames) {
      static_cast<void>(age_frames);
      finalized_track_count += count;
    }
    for (const auto& [age_frames, count] : observed_track_lengths) {
      observed_track_count += count;
      observed_track_length_sum += static_cast<long double>(age_frames) * count;
    }
    std::map<size_t, uint64_t> generic_observed_track_lengths = generic_cam0_track_length_histogram_frames;
    for (const auto& [id, age_frames] : generic_cam0_active_track_lengths) {
      static_cast<void>(id);
      generic_observed_track_lengths[age_frames]++;
    }
    uint64_t generic_finalized_track_count = 0;
    uint64_t generic_observed_track_count = 0;
    long double generic_observed_track_length_sum = 0;
    for (const auto& [age_frames, count] : generic_cam0_track_length_histogram_frames) {
      static_cast<void>(age_frames);
      generic_finalized_track_count += count;
    }
    for (const auto& [age_frames, count] : generic_observed_track_lengths) {
      generic_observed_track_count += count;
      generic_observed_track_length_sum += static_cast<long double>(age_frames) * count;
    }
    auto observed_track_quality = direct_mv_track_quality_by_length;
    for (const auto& [id, active] : direct_active_tracks) {
      static_cast<void>(id);
      if (!active.confirmed) continue;
      accumulateDirectTrackQuality(
          observed_track_quality.at(directTrackQualityBucket(active.age_frames)),
          active, DIRECT_TRACK_TERMINATION_SHUTDOWN);
    }
    const auto safe_mean = [](long double sum, uint64_t count) {
      return count == 0 ? 0.0 : static_cast<double>(sum / count);
    };
    stats << "{\n";

    stats << "  \"guess_error_histogram\": {\n";
    for (auto it = guess_error_histogram.begin(); it != guess_error_histogram.end(); it++) {
      auto& [error, count] = *it;
      bool is_last = it == std::prev(guess_error_histogram.end());
      stats << "    \"" << error << "\": " << count << (is_last ? "" : ",") << "\n";
    }
    stats << "  },\n";

    stats << "  \"fallback_recovered_points\": {\n";
    int64_t fallback_recovered_points_total = 0;
    for (auto it = fallback_recovered_points.begin(); it != fallback_recovered_points.end(); it++) {
      auto& [frame_id, count_per_cam] = *it;
      bool is_last = it == std::prev(fallback_recovered_points.end());

      std::string count_string = "[";
      for (size_t i = 0; i < count_per_cam.size(); i++) {
        count_string.append(std::to_string(count_per_cam[i]));
        fallback_recovered_points_total += count_per_cam[i];
        if (i < count_per_cam.size() - 1) count_string.append(", ");
      }
      count_string.append("]");
      stats << "    \"" << frame_id << "\": " << count_string << (is_last ? "" : ",") << "\n";
    }
    stats << "  },\n";

    stats << "  \"fallback_recovered_points_total\": " << fallback_recovered_points_total << ",\n";
    stats << "  \"direct_mv_tracks\": " << direct_mv_tracks << ",\n";
    stats << "  \"direct_mv_hybrid_reciprocal_tracks\": "
          << direct_mv_hybrid_reciprocal_tracks << ",\n";
    stats << "  \"direct_mv_strict_confirmation\": "
          << (config.optical_flow_direct_mv_strict_confirmation ? 1 : 0)
          << ",\n";
    stats << "  \"direct_mv_merge_indistinguishable\": "
          << (config.optical_flow_direct_mv_merge_indistinguishable ? 1 : 0)
          << ",\n";
    stats << "  \"direct_mv_merge_radius_px\": "
          << config.optical_flow_direct_mv_merge_radius_px << ",\n";
    stats << "  \"direct_mv_cam0_input_vectors\": "
          << direct_mv_cam0_input_vectors << ",\n";
    stats << "  \"direct_mv_cam0_rejected_nonpast_source\": "
          << direct_mv_cam0_rejected_nonpast_source << ",\n";
    stats << "  \"direct_mv_cam0_rejected_median_filter\": "
          << direct_mv_cam0_rejected_median_filter << ",\n";
    stats << "  \"direct_mv_cam0_rejected_image_bounds\": "
          << direct_mv_cam0_rejected_image_bounds << ",\n";
    stats << "  \"direct_mv_cam0_rejected_safe_radius\": "
          << direct_mv_cam0_rejected_safe_radius << ",\n";
    stats << "  \"direct_mv_cam0_rejected_mask\": "
          << direct_mv_cam0_rejected_mask << ",\n";
    stats << "  \"direct_mv_cam0_accepted_tracks\": "
          << direct_mv_cam0_accepted_tracks << ",\n";
    stats << "  \"direct_mv_reverse_temporal_policy\": "
          << direct_mv_reverse_temporal_policy << ",\n";
    stats << "  \"direct_mv_reverse_temporal_frames\": "
          << direct_mv_reverse_temporal_frames << ",\n";
    stats << "  \"direct_mv_reverse_temporal_forward_vectors\": "
          << direct_mv_reverse_temporal_forward_vectors << ",\n";
    stats << "  \"direct_mv_reverse_temporal_reverse_vectors\": "
          << direct_mv_reverse_temporal_reverse_vectors << ",\n";
    stats << "  \"direct_mv_reverse_temporal_reciprocal_vectors\": "
          << direct_mv_reverse_temporal_reciprocal_vectors << ",\n";
    stats << "  \"direct_mv_reverse_temporal_single_direction_fallback_frames\": "
          << direct_mv_reverse_temporal_single_direction_fallback_frames << ",\n";
    stats << "  \"direct_mv_reverse_temporal_single_direction_fallback_vectors\": "
          << direct_mv_reverse_temporal_single_direction_fallback_vectors << ",\n";
    stats << "  \"direct_mv_reverse_temporal_output_vectors\": "
          << direct_mv_reverse_temporal_output_vectors << ",\n";
    stats << "  \"direct_mv_stereo_rectification_mask_frames\": "
          << direct_mv_stereo_rectification_mask_frames << ",\n";
    stats << "  \"direct_mv_stereo_rectification_mode\": "
          << direct_mv_stereo_rectification_mode << ",\n";
    stats << "  \"direct_mv_stereo_rectification_mask_rejected_vectors\": "
          << direct_mv_stereo_rectification_mask_rejected_vectors << ",\n";
    stats << "  \"direct_mv_temporal_subpixel_vectors\": "
          << direct_mv_temporal_subpixel_vectors << ",\n";
    stats << "  \"direct_mv_teaser_noise_bound_m\": "
          << config.optical_flow_direct_mv_teaser_noise_bound_m << ",\n";
    stats << "  \"direct_mv_teaser_far_depth_m\": "
          << config.optical_flow_direct_mv_teaser_far_depth_m << ",\n";
    stats << "  \"direct_mv_teaser_far_reprojection_gate_px\": "
          << config.optical_flow_direct_mv_teaser_far_reprojection_gate_px << ",\n";
    stats << "  \"direct_mv_teaser_frames\": " << direct_mv_teaser_frames << ",\n";
    stats << "  \"direct_mv_teaser_successful_frames\": "
          << direct_mv_teaser_successful_frames << ",\n";
    stats << "  \"direct_mv_teaser_failed_frames\": " << direct_mv_teaser_failed_frames << ",\n";
    stats << "  \"direct_mv_teaser_insufficient_frames\": "
          << direct_mv_teaser_insufficient_frames << ",\n";
    stats << "  \"direct_mv_teaser_candidate_pairs\": "
          << direct_mv_teaser_candidate_pairs << ",\n";
    stats << "  \"direct_mv_teaser_pair_input_tracks\": "
          << direct_mv_teaser_pair_input_tracks << ",\n";
    stats << "  \"direct_mv_teaser_pair_missing_both_stereo\": "
          << direct_mv_teaser_pair_missing_both_stereo << ",\n";
    stats << "  \"direct_mv_teaser_pair_missing_previous_stereo\": "
          << direct_mv_teaser_pair_missing_previous_stereo << ",\n";
    stats << "  \"direct_mv_teaser_pair_missing_current_stereo\": "
          << direct_mv_teaser_pair_missing_current_stereo << ",\n";
    stats << "  \"direct_mv_teaser_pair_invalid_3d\": "
          << direct_mv_teaser_pair_invalid_3d << ",\n";
    stats << "  \"direct_mv_teaser_solver_pairs\": " << direct_mv_teaser_solver_pairs << ",\n";
    stats << "  \"direct_mv_teaser_solver_inliers\": " << direct_mv_teaser_solver_inliers << ",\n";
    stats << "  \"direct_mv_teaser_validated_edges\": "
          << direct_mv_teaser_validated_edges << ",\n";
    stats << "  \"direct_mv_teaser_accepted_edges\": "
          << direct_mv_teaser_accepted_edges << ",\n";
    stats << "  \"direct_mv_teaser_rejected_no_model\": "
          << direct_mv_teaser_rejected_no_model << ",\n";
    stats << "  \"direct_mv_teaser_rejected_invalid_depth\": "
          << direct_mv_teaser_rejected_invalid_depth << ",\n";
    stats << "  \"direct_mv_teaser_rejected_metric\": "
          << direct_mv_teaser_rejected_metric << ",\n";
    stats << "  \"direct_mv_teaser_rejected_far_reprojection\": "
          << direct_mv_teaser_rejected_far_reprojection << ",\n";
    stats << "  \"direct_mv_teaser_authority_candidate_edges\": "
          << direct_mv_teaser_authority_candidate_edges << ",\n";
    stats << "  \"direct_mv_teaser_authority_accepted_candidates\": "
          << direct_mv_teaser_authority_accepted_candidates << ",\n";
    stats << "  \"direct_mv_teaser_authority_selected_edges\": "
          << direct_mv_teaser_authority_selected_edges << ",\n";
    stats << "  \"direct_mv_teaser_authority_no_candidate\": "
          << direct_mv_teaser_authority_no_candidate << ",\n";
    stats << "  \"direct_mv_teaser_authority_queries\": "
          << direct_mv_teaser_authority_queries << ",\n";
    stats << "  \"direct_mv_teaser_authority_indexed_candidates\": "
          << direct_mv_teaser_authority_indexed_candidates << ",\n";
    stats << "  \"direct_mv_teaser_authority_rejected_not_covering_source\": "
          << direct_mv_teaser_authority_rejected_not_covering_source << ",\n";
    stats << "  \"direct_mv_teaser_authority_rejected_invalid_destination\": "
          << direct_mv_teaser_authority_rejected_invalid_destination << ",\n";
    stats << "  \"direct_mv_teaser_authority_rejected_current_stereo\": "
          << direct_mv_teaser_authority_rejected_current_stereo << ",\n";
    stats << "  \"direct_mv_teaser_authority_rejected_teaser\": "
          << direct_mv_teaser_authority_rejected_teaser << ",\n";
    stats << "  \"direct_mv_teaser_authority_failed_no_spatial_bin\": "
          << direct_mv_teaser_authority_failed_no_spatial_bin << ",\n";
    stats << "  \"direct_mv_teaser_authority_failed_no_available_candidate\": "
          << direct_mv_teaser_authority_failed_no_available_candidate << ",\n";
    stats << "  \"direct_mv_teaser_authority_failed_no_covering_candidate\": "
          << direct_mv_teaser_authority_failed_no_covering_candidate << ",\n";
    stats << "  \"direct_mv_teaser_authority_failed_invalid_destination\": "
          << direct_mv_teaser_authority_failed_invalid_destination << ",\n";
    stats << "  \"direct_mv_teaser_authority_failed_current_stereo\": "
          << direct_mv_teaser_authority_failed_current_stereo << ",\n";
    stats << "  \"direct_mv_teaser_authority_failed_teaser_gate\": "
          << direct_mv_teaser_authority_failed_teaser_gate << ",\n";
    stats << "  \"direct_mv_teaser_authority_bidirectional_accepted_queries\": "
          << direct_mv_teaser_authority_bidirectional_accepted_queries
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_forward_only_accepted_queries\": "
          << direct_mv_teaser_authority_forward_only_accepted_queries
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_reverse_only_accepted_queries\": "
          << direct_mv_teaser_authority_reverse_only_accepted_queries
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_other_origin_accepted_queries\": "
          << direct_mv_teaser_authority_other_origin_accepted_queries
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_bidirectional_residual_difference_mean_m\": "
          << safe_mean(
                 direct_mv_teaser_authority_bidirectional_residual_difference_sum_m,
                 direct_mv_teaser_authority_bidirectional_accepted_queries)
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_bidirectional_residual_difference_max_m\": "
          << direct_mv_teaser_authority_bidirectional_residual_difference_max_m
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_bidirectional_destination_difference_mean_px\": "
          << safe_mean(
                 direct_mv_teaser_authority_bidirectional_destination_difference_sum_px,
                 direct_mv_teaser_authority_bidirectional_accepted_queries)
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_bidirectional_destination_difference_max_px\": "
          << direct_mv_teaser_authority_bidirectional_destination_difference_max_px
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_bidirectional_depth_difference_mean_m\": "
          << safe_mean(
                 direct_mv_teaser_authority_bidirectional_depth_difference_sum_m,
                 direct_mv_teaser_authority_bidirectional_accepted_queries)
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_bidirectional_depth_difference_max_m\": "
          << direct_mv_teaser_authority_bidirectional_depth_difference_max_m
          << ",\n";
    stats << "  \"direct_mv_teaser_authority_started\": "
          << direct_mv_teaser_authority_started << ",\n";
    stats << "  \"direct_mv_teaser_authority_continuations\": "
          << direct_mv_teaser_authority_continuations << ",\n";
    stats << "  \"direct_mv_track_continuation_attempts\": "
          << direct_mv_track_continuation_attempts << ",\n";
    stats << "  \"direct_mv_track_rejected_optimizer_observation\": "
          << direct_mv_track_rejected_optimizer_observation << ",\n";
    stats << "  \"direct_mv_track_rejected_previous_stereo\": "
          << direct_mv_track_rejected_previous_stereo << ",\n";
    stats << "  \"direct_mv_track_rejected_temporal_match\": "
          << direct_mv_track_rejected_temporal_match << ",\n";
    stats << "  \"direct_mv_track_rejected_i_frame\": "
          << direct_mv_track_rejected_i_frame << ",\n";
    stats << "  \"direct_mv_stereo_matches\": " << direct_mv_stereo_matches << ",\n";
    stats << "  \"direct_mv_codec_stereo_frames\": " << direct_mv_codec_stereo_frames << ",\n";
    stats << "  \"direct_mv_codec_stereo_queries\": " << direct_mv_codec_stereo_queries << ",\n";
    stats << "  \"direct_mv_codec_stereo_candidates\": " << direct_mv_codec_stereo_candidates << ",\n";
    stats << "  \"direct_mv_codec_stereo_matches\": " << direct_mv_codec_stereo_matches << ",\n";
    stats << "  \"direct_mv_temporal_motion_scale_histogram\": {";
    size_t emitted_temporal_scales = 0;
    for (const auto& [scale, count] : direct_mv_temporal_motion_scale_histogram) {
      stats << (emitted_temporal_scales++ == 0 ? "\n" : ",\n");
      stats << "    \"" << scale << "\": " << count;
    }
    if (emitted_temporal_scales > 0) stats << "\n  ";
    stats << "},\n";
    stats << "  \"direct_mv_stereo_motion_scale_histogram\": {";
    size_t emitted_stereo_scales = 0;
    for (const auto& [scale, count] : direct_mv_stereo_motion_scale_histogram) {
      stats << (emitted_stereo_scales++ == 0 ? "\n" : ",\n");
      stats << "    \"" << scale << "\": " << count;
    }
    if (emitted_stereo_scales > 0) stats << "\n  ";
    stats << "},\n";
    stats << "  \"direct_mv_persistent_tracks_started\": " << direct_mv_persistent_tracks_started << ",\n";
    stats << "  \"direct_mv_provisional_tracks_started\": "
          << direct_mv_provisional_tracks_started << ",\n";
    stats << "  \"direct_mv_provisional_tracks_confirmed\": "
          << direct_mv_provisional_tracks_confirmed << ",\n";
    stats << "  \"direct_mv_provisional_tracks_rejected\": "
          << direct_mv_provisional_tracks_rejected << ",\n";
    stats << "  \"direct_mv_merged_continuations\": "
          << direct_mv_merged_continuations << ",\n";
    stats << "  \"direct_mv_merged_births\": "
          << direct_mv_merged_births << ",\n";
    stats << "  \"direct_mv_persistent_track_continuations\": " << direct_mv_persistent_track_continuations
          << ",\n";
    stats << "  \"direct_mv_persistent_tracks_killed\": " << direct_mv_persistent_tracks_killed << ",\n";
    stats << "  \"direct_mv_max_track_frames\": " << direct_mv_max_track_frames << ",\n";
    stats << "  \"direct_mv_cam0_track_length\": {\n";
    stats << "    \"finalized_tracks\": " << finalized_track_count << ",\n";
    stats << "    \"active_tracks_censored_at_shutdown\": "
          << active_confirmed_track_count << ",\n";
    stats << "    \"observed_tracks\": " << observed_track_count << ",\n";
    stats << "    \"mean_observed_frames\": "
          << safe_mean(observed_track_length_sum, observed_track_count) << ",\n";
    stats << "    \"p50_observed_frames\": "
          << directTrackLengthQuantile(observed_track_lengths, observed_track_count, 0.50) << ",\n";
    stats << "    \"p90_observed_frames\": "
          << directTrackLengthQuantile(observed_track_lengths, observed_track_count, 0.90) << ",\n";
    stats << "    \"p95_observed_frames\": "
          << directTrackLengthQuantile(observed_track_lengths, observed_track_count, 0.95) << ",\n";
    stats << "    \"max_observed_frames\": "
          << (observed_track_lengths.empty() ? 0 : observed_track_lengths.rbegin()->first) << ",\n";
    stats << "    \"histogram_frames\": {";
    size_t emitted_track_bins = 0;
    for (const auto& [age_frames, count] : observed_track_lengths) {
      stats << (emitted_track_bins++ == 0 ? "\n" : ",\n");
      stats << "      \"" << age_frames << "\": " << count;
    }
    if (emitted_track_bins > 0) stats << "\n    ";
    stats << "}\n";
    stats << "  },\n";

    stats << "  \"direct_mv_selection_quality_by_origin\": {\n";
    const std::array<const char*, 4> origin_names = {
        "unknown", "forward", "reverse", "reciprocal"};
    for (size_t origin = 0;
         origin < direct_mv_selection_quality_by_origin.size(); ++origin) {
      const DirectOriginSelectionAggregate& quality =
          direct_mv_selection_quality_by_origin.at(origin);
      stats << "    \"" << origin_names.at(origin) << "\": {\n";
      stats << "      \"selections\": " << quality.selections << ",\n";
      stats << "      \"mean_teaser_residual_m\": "
            << safe_mean(quality.teaser_residual_sum_m, quality.selections)
            << ",\n";
      stats << "      \"mean_normalized_center_cost\": "
            << safe_mean(quality.center_cost_sum, quality.selections)
            << ",\n";
      stats << "      \"mean_abs_depth_delta_m\": "
            << safe_mean(quality.depth_delta_sum_m, quality.selections)
            << ",\n";
      stats << "      \"mean_abs_depth_log_ratio\": "
            << safe_mean(quality.depth_log_ratio_sum, quality.selections)
            << ",\n";
      stats << "      \"mean_stereo_cost\": "
            << safe_mean(quality.stereo_cost_sum, quality.selections)
            << ",\n";
      stats << "      \"mean_block_area_px2\": "
            << safe_mean(quality.block_area_sum_px2, quality.selections)
            << "\n";
      stats << "    }"
            << (origin + 1 == direct_mv_selection_quality_by_origin.size()
                    ? "\n"
                    : ",\n");
    }
    stats << "  },\n";

    const uint64_t selected_assignment_count =
        direct_mv_selected_forward_assignments +
        direct_mv_selected_reverse_assignments +
        direct_mv_selected_reciprocal_assignments +
        direct_mv_selected_unknown_assignments;
    stats << "  \"direct_mv_selected_mv_sharing\": {\n";
    stats << "    \"selected_assignments\": " << selected_assignment_count
          << ",\n";
    stats << "    \"forward_assignments\": "
          << direct_mv_selected_forward_assignments << ",\n";
    stats << "    \"reverse_assignments\": "
          << direct_mv_selected_reverse_assignments << ",\n";
    stats << "    \"reciprocal_assignments\": "
          << direct_mv_selected_reciprocal_assignments << ",\n";
    stats << "    \"unknown_assignments\": "
          << direct_mv_selected_unknown_assignments << ",\n";
    stats << "    \"unique_mv_groups\": "
          << direct_mv_selected_unique_mv_groups << ",\n";
    stats << "    \"shared_mv_groups\": "
          << direct_mv_selected_shared_mv_groups << ",\n";
    stats << "    \"assignments_in_shared_groups\": "
          << direct_mv_selected_shared_assignments << ",\n";
    stats << "    \"duplicate_assignments_beyond_one_per_mv\": "
          << direct_mv_selected_duplicate_assignments << ",\n";
    stats << "    \"max_multiplicity\": "
          << direct_mv_selected_mv_max_multiplicity << ",\n";
    stats << "    \"mean_assignments_per_selected_mv\": "
          << safe_mean(selected_assignment_count,
                       direct_mv_selected_unique_mv_groups)
          << ",\n";
    stats << "    \"shared_assignment_fraction\": "
          << safe_mean(direct_mv_selected_shared_assignments,
                       selected_assignment_count)
          << ",\n";
    stats << "    \"previous_depth_range_mean_m\": "
          << safe_mean(
                 direct_mv_selected_shared_previous_depth_range_sum_m,
                 direct_mv_selected_shared_mv_groups)
          << ",\n";
    stats << "    \"previous_depth_range_p50_m_bin_upper_bound\": "
          << directHistogramQuantile(
                 direct_mv_selected_shared_previous_depth_range_histogram,
                 direct_mv_selected_shared_mv_groups, 0.50,
                 kDirectSharedSpreadBinWidthMeters)
          << ",\n";
    stats << "    \"previous_depth_range_p95_m_bin_upper_bound\": "
          << directHistogramQuantile(
                 direct_mv_selected_shared_previous_depth_range_histogram,
                 direct_mv_selected_shared_mv_groups, 0.95,
                 kDirectSharedSpreadBinWidthMeters)
          << ",\n";
    stats << "    \"previous_depth_range_max_m\": "
          << direct_mv_selected_shared_previous_depth_range_max_m << ",\n";
    stats << "    \"current_depth_range_mean_m\": "
          << safe_mean(
                 direct_mv_selected_shared_current_depth_range_sum_m,
                 direct_mv_selected_shared_mv_groups)
          << ",\n";
    stats << "    \"current_depth_range_p50_m_bin_upper_bound\": "
          << directHistogramQuantile(
                 direct_mv_selected_shared_current_depth_range_histogram,
                 direct_mv_selected_shared_mv_groups, 0.50,
                 kDirectSharedSpreadBinWidthMeters)
          << ",\n";
    stats << "    \"current_depth_range_p95_m_bin_upper_bound\": "
          << directHistogramQuantile(
                 direct_mv_selected_shared_current_depth_range_histogram,
                 direct_mv_selected_shared_mv_groups, 0.95,
                 kDirectSharedSpreadBinWidthMeters)
          << ",\n";
    stats << "    \"current_depth_range_max_m\": "
          << direct_mv_selected_shared_current_depth_range_max_m << ",\n";
    stats << "    \"teaser_residual_range_mean_m\": "
          << safe_mean(
                 direct_mv_selected_shared_teaser_residual_range_sum_m,
                 direct_mv_selected_shared_mv_groups)
          << ",\n";
    stats << "    \"teaser_residual_range_p95_m_bin_upper_bound\": "
          << directHistogramQuantile(
                 direct_mv_selected_shared_teaser_residual_range_histogram,
                 direct_mv_selected_shared_mv_groups, 0.95,
                 kDirectSharedSpreadBinWidthMeters)
          << ",\n";
    stats << "    \"teaser_residual_range_max_m\": "
          << direct_mv_selected_shared_teaser_residual_range_max_m << ",\n";
    stats << "    \"multiplicity_histogram\": {";
    size_t emitted_multiplicity_bins = 0;
    for (const auto& [multiplicity, count] :
         direct_mv_selected_mv_multiplicity_histogram) {
      stats << (emitted_multiplicity_bins++ == 0 ? "\n" : ",\n");
      stats << "      \"" << multiplicity << "\": " << count;
    }
    if (emitted_multiplicity_bins > 0) stats << "\n    ";
    stats << "}\n";
    stats << "  },\n";

    stats << "  \"direct_mv_track_quality_by_final_length\": {\n";
    const std::array<const char*, 4> quality_bucket_names = {
        "2", "3-4", "5-9", "10-plus"};
    for (size_t bucket = 0; bucket < observed_track_quality.size(); ++bucket) {
      const DirectTrackQualityAggregate& quality =
          observed_track_quality.at(bucket);
      stats << "    \"" << quality_bucket_names.at(bucket) << "\": {\n";
      stats << "      \"promoted_tracks\": " << quality.tracks << ",\n";
      stats << "      \"mean_frames\": "
            << safe_mean(quality.frames, quality.tracks) << ",\n";
      stats << "      \"selections\": " << quality.selections << ",\n";
      stats << "      \"forward_selections\": "
            << quality.forward_selections << ",\n";
      stats << "      \"reverse_selections\": "
            << quality.reverse_selections << ",\n";
      stats << "      \"reciprocal_selections\": "
            << quality.reciprocal_selections << ",\n";
      stats << "      \"shared_selections\": "
            << quality.shared_selections << ",\n";
      stats << "      \"shared_selection_fraction\": "
            << safe_mean(quality.shared_selections, quality.selections)
            << ",\n";
      stats << "      \"mean_teaser_residual_m\": "
            << safe_mean(quality.teaser_residual_sum_m, quality.selections)
            << ",\n";
      stats << "      \"mean_normalized_center_cost\": "
            << safe_mean(quality.center_cost_sum, quality.selections)
            << ",\n";
      stats << "      \"mean_abs_depth_delta_m\": "
            << safe_mean(quality.depth_delta_sum_m, quality.selections)
            << ",\n";
      stats << "      \"mean_abs_depth_log_ratio\": "
            << safe_mean(quality.depth_log_ratio_sum, quality.selections)
            << ",\n";
      stats << "      \"mean_stereo_cost\": "
            << safe_mean(quality.stereo_cost_sum, quality.selections)
            << ",\n";
      stats << "      \"mean_block_area_px2\": "
            << safe_mean(quality.block_area_sum_px2, quality.selections)
            << ",\n";
      stats << "      \"mean_group_multiplicity\": "
            << safe_mean(quality.group_multiplicity_sum,
                         quality.selections)
            << ",\n";
      stats << "      \"mean_shared_depth_range_m\": "
            << safe_mean(quality.shared_depth_range_sum_m,
                         quality.shared_selections)
            << ",\n";
      stats << "      \"max_shared_depth_range_m\": "
            << quality.max_shared_depth_range_m << ",\n";
      stats << "      \"termination_optimizer_observation\": "
            << quality.termination_reasons.at(
                   DIRECT_TRACK_TERMINATION_OPTIMIZER_OBSERVATION)
            << ",\n";
      stats << "      \"termination_previous_stereo\": "
            << quality.termination_reasons.at(
                   DIRECT_TRACK_TERMINATION_PREVIOUS_STEREO)
            << ",\n";
      stats << "      \"termination_temporal_match\": "
            << quality.termination_reasons.at(
                   DIRECT_TRACK_TERMINATION_TEMPORAL_MATCH)
            << ",\n";
      stats << "      \"termination_post_gate\": "
            << quality.termination_reasons.at(
                   DIRECT_TRACK_TERMINATION_POST_GATE)
            << ",\n";
      stats << "      \"termination_i_frame\": "
            << quality.termination_reasons.at(
                   DIRECT_TRACK_TERMINATION_I_FRAME)
            << ",\n";
      stats << "      \"censored_at_shutdown\": "
            << quality.termination_reasons.at(
                   DIRECT_TRACK_TERMINATION_SHUTDOWN)
            << "\n";
      stats << "    }" << (bucket + 1 == observed_track_quality.size()
                                 ? "\n"
                                 : ",\n");
    }
    stats << "  },\n";

    stats << "  \"direct_mv_temporal_depth_residual\": {\n";
    stats << "    \"sample_count\": " << direct_mv_temporal_depth_residual_samples << ",\n";
    stats << "    \"signed_mean_m\": "
          << safe_mean(direct_mv_temporal_depth_residual_sum_m, direct_mv_temporal_depth_residual_samples)
          << ",\n";
    stats << "    \"mean_abs_m\": "
          << safe_mean(direct_mv_temporal_depth_abs_residual_sum_m, direct_mv_temporal_depth_residual_samples)
          << ",\n";
    stats << "    \"rmse_m\": "
          << std::sqrt(safe_mean(direct_mv_temporal_depth_squared_residual_sum_m2,
                                 direct_mv_temporal_depth_residual_samples))
          << ",\n";
    stats << "    \"p50_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(direct_mv_temporal_depth_abs_residual_histogram,
                                     direct_mv_temporal_depth_residual_samples, 0.50,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "    \"p95_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(direct_mv_temporal_depth_abs_residual_histogram,
                                     direct_mv_temporal_depth_residual_samples, 0.95,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "    \"p99_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(direct_mv_temporal_depth_abs_residual_histogram,
                                     direct_mv_temporal_depth_residual_samples, 0.99,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "    \"max_abs_m\": " << direct_mv_temporal_depth_max_abs_residual_m << ",\n";
    stats << "    \"mean_abs_rate_mps\": "
          << safe_mean(direct_mv_temporal_depth_abs_rate_sum_mps, direct_mv_temporal_depth_rate_samples)
          << ",\n";
    stats << "    \"mean_abs_log_ratio\": "
          << safe_mean(direct_mv_temporal_depth_abs_log_ratio_sum,
                       direct_mv_temporal_depth_residual_samples)
          << ",\n";
    stats << "    \"p95_abs_log_ratio_bin_upper_bound\": "
          << directHistogramQuantile(direct_mv_temporal_depth_abs_log_ratio_histogram,
                                     direct_mv_temporal_depth_residual_samples, 0.95,
                                     kDirectDepthLogRatioBinWidth)
          << ",\n";
    stats << "    \"max_abs_log_ratio\": " << direct_mv_temporal_depth_max_abs_log_ratio << ",\n";
    stats << "    \"immediate_successor\": {\n";
    stats << "      \"sample_count\": " << direct_mv_temporal_depth_immediate_samples << ",\n";
    stats << "      \"signed_mean_m\": "
          << safe_mean(direct_mv_temporal_depth_immediate_residual_sum_m,
                       direct_mv_temporal_depth_immediate_samples)
          << ",\n";
    stats << "      \"mean_abs_m\": "
          << safe_mean(direct_mv_temporal_depth_immediate_abs_residual_sum_m,
                       direct_mv_temporal_depth_immediate_samples)
          << ",\n";
    stats << "      \"rmse_m\": "
          << std::sqrt(safe_mean(direct_mv_temporal_depth_immediate_squared_residual_sum_m2,
                                 direct_mv_temporal_depth_immediate_samples))
          << ",\n";
    stats << "      \"p50_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(direct_mv_temporal_depth_immediate_abs_residual_histogram,
                                     direct_mv_temporal_depth_immediate_samples, 0.50,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "      \"p95_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(direct_mv_temporal_depth_immediate_abs_residual_histogram,
                                     direct_mv_temporal_depth_immediate_samples, 0.95,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "      \"p99_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(direct_mv_temporal_depth_immediate_abs_residual_histogram,
                                     direct_mv_temporal_depth_immediate_samples, 0.99,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "      \"max_abs_m\": " << direct_mv_temporal_depth_immediate_max_abs_residual_m << ",\n";
    stats << "      \"mean_abs_log_ratio\": "
          << safe_mean(direct_mv_temporal_depth_immediate_abs_log_ratio_sum,
                       direct_mv_temporal_depth_immediate_samples)
          << ",\n";
    stats << "      \"p95_abs_log_ratio_bin_upper_bound\": "
          << directHistogramQuantile(direct_mv_temporal_depth_immediate_abs_log_ratio_histogram,
                                     direct_mv_temporal_depth_immediate_samples, 0.95,
                                     kDirectDepthLogRatioBinWidth)
          << ",\n";
    stats << "      \"max_abs_log_ratio\": "
          << direct_mv_temporal_depth_immediate_max_abs_log_ratio << "\n";
    stats << "    },\n";
    stats << "    \"gap_histogram_frames\": {";
    size_t emitted_gap_bins = 0;
    for (const auto& [gap_frames, count] : direct_mv_temporal_depth_gap_histogram_frames) {
      stats << (emitted_gap_bins++ == 0 ? "\n" : ",\n");
      stats << "      \"" << gap_frames << "\": " << count;
    }
    if (emitted_gap_bins > 0) stats << "\n    ";
    stats << "},\n";
    stats << "    \"abs_residual_histogram_bin_width_m\": "
          << kDirectDepthResidualBinWidthMeters << ",\n";
    stats << "    \"abs_residual_histogram\": {";
    size_t emitted_depth_bins = 0;
    for (size_t bin = 0; bin < direct_mv_temporal_depth_abs_residual_histogram.size(); ++bin) {
      const uint64_t count = direct_mv_temporal_depth_abs_residual_histogram.at(bin);
      if (count == 0) continue;
      stats << (emitted_depth_bins++ == 0 ? "\n" : ",\n");
      stats << "      \"" << bin << "\": " << count;
    }
    if (emitted_depth_bins > 0) stats << "\n    ";
    stats << "}\n";
    stats << "  },\n";

    stats << "  \"generic_cam0_track_length\": {\n";
    stats << "    \"finalized_tracks\": " << generic_finalized_track_count << ",\n";
    stats << "    \"active_tracks_censored_at_shutdown\": " << generic_cam0_active_track_lengths.size()
          << ",\n";
    stats << "    \"observed_tracks\": " << generic_observed_track_count << ",\n";
    stats << "    \"mean_observed_frames\": "
          << safe_mean(generic_observed_track_length_sum, generic_observed_track_count) << ",\n";
    stats << "    \"p50_observed_frames\": "
          << directTrackLengthQuantile(generic_observed_track_lengths, generic_observed_track_count, 0.50)
          << ",\n";
    stats << "    \"p90_observed_frames\": "
          << directTrackLengthQuantile(generic_observed_track_lengths, generic_observed_track_count, 0.90)
          << ",\n";
    stats << "    \"p95_observed_frames\": "
          << directTrackLengthQuantile(generic_observed_track_lengths, generic_observed_track_count, 0.95)
          << ",\n";
    stats << "    \"max_observed_frames\": "
          << (generic_observed_track_lengths.empty() ? 0 : generic_observed_track_lengths.rbegin()->first)
          << ",\n";
    stats << "    \"histogram_frames\": {";
    size_t emitted_generic_track_bins = 0;
    for (const auto& [age_frames, count] : generic_observed_track_lengths) {
      stats << (emitted_generic_track_bins++ == 0 ? "\n" : ",\n");
      stats << "      \"" << age_frames << "\": " << count;
    }
    if (emitted_generic_track_bins > 0) stats << "\n    ";
    stats << "}\n";
    stats << "  },\n";

    const auto& generic_depth = generic_depth_residual_stats;
    stats << "  \"generic_temporal_stereo_depth_residual\": {\n";
    stats << "    \"sample_count\": " << generic_depth.samples << ",\n";
    stats << "    \"signed_mean_m\": " << safe_mean(generic_depth.residual_sum_m, generic_depth.samples)
          << ",\n";
    stats << "    \"mean_abs_m\": " << safe_mean(generic_depth.abs_residual_sum_m, generic_depth.samples)
          << ",\n";
    stats << "    \"rmse_m\": "
          << std::sqrt(safe_mean(generic_depth.squared_residual_sum_m2, generic_depth.samples)) << ",\n";
    stats << "    \"p50_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(generic_depth.abs_residual_histogram, generic_depth.samples, 0.50,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "    \"p95_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(generic_depth.abs_residual_histogram, generic_depth.samples, 0.95,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "    \"p99_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(generic_depth.abs_residual_histogram, generic_depth.samples, 0.99,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "    \"max_abs_m\": " << generic_depth.max_abs_residual_m << ",\n";
    stats << "    \"mean_abs_rate_mps\": "
          << safe_mean(generic_depth.abs_rate_sum_mps, generic_depth.rate_samples) << ",\n";
    stats << "    \"mean_abs_log_ratio\": "
          << safe_mean(generic_depth.abs_log_ratio_sum, generic_depth.samples) << ",\n";
    stats << "    \"p95_abs_log_ratio_bin_upper_bound\": "
          << directHistogramQuantile(generic_depth.abs_log_ratio_histogram, generic_depth.samples, 0.95,
                                     kDirectDepthLogRatioBinWidth)
          << ",\n";
    stats << "    \"max_abs_log_ratio\": " << generic_depth.max_abs_log_ratio << ",\n";
    stats << "    \"immediate_successor\": {\n";
    stats << "      \"sample_count\": " << generic_depth.immediate_samples << ",\n";
    stats << "      \"signed_mean_m\": "
          << safe_mean(generic_depth.immediate_residual_sum_m, generic_depth.immediate_samples) << ",\n";
    stats << "      \"mean_abs_m\": "
          << safe_mean(generic_depth.immediate_abs_residual_sum_m, generic_depth.immediate_samples) << ",\n";
    stats << "      \"rmse_m\": "
          << std::sqrt(safe_mean(generic_depth.immediate_squared_residual_sum_m2,
                                 generic_depth.immediate_samples))
          << ",\n";
    stats << "      \"p50_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(generic_depth.immediate_abs_residual_histogram,
                                     generic_depth.immediate_samples, 0.50,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "      \"p95_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(generic_depth.immediate_abs_residual_histogram,
                                     generic_depth.immediate_samples, 0.95,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "      \"p99_abs_m_bin_upper_bound\": "
          << directHistogramQuantile(generic_depth.immediate_abs_residual_histogram,
                                     generic_depth.immediate_samples, 0.99,
                                     kDirectDepthResidualBinWidthMeters)
          << ",\n";
    stats << "      \"max_abs_m\": " << generic_depth.immediate_max_abs_residual_m << ",\n";
    stats << "      \"mean_abs_log_ratio\": "
          << safe_mean(generic_depth.immediate_abs_log_ratio_sum, generic_depth.immediate_samples) << ",\n";
    stats << "      \"p95_abs_log_ratio_bin_upper_bound\": "
          << directHistogramQuantile(generic_depth.immediate_abs_log_ratio_histogram,
                                     generic_depth.immediate_samples, 0.95,
                                     kDirectDepthLogRatioBinWidth)
          << ",\n";
    stats << "      \"max_abs_log_ratio\": " << generic_depth.immediate_max_abs_log_ratio << "\n";
    stats << "    },\n";
    stats << "    \"gap_histogram_frames\": {";
    size_t emitted_generic_gap_bins = 0;
    for (const auto& [gap_frames, count] : generic_depth.gap_histogram_frames) {
      stats << (emitted_generic_gap_bins++ == 0 ? "\n" : ",\n");
      stats << "      \"" << gap_frames << "\": " << count;
    }
    if (emitted_generic_gap_bins > 0) stats << "\n    ";
    stats << "}\n";
    stats << "  },\n";
    stats << "  \"mv_total_tracking_time\": " << mv_total_tracking_time << ",\n";
    stats << "  \"mv_total_tracked_points\": " << mv_total_tracked_points << ",\n";
    stats << "  \"of_total_tracking_time\": " << of_total_tracking_time << ",\n";
    stats << "  \"of_total_tracked_points\": " << of_total_tracked_points << "\n";
    stats << "}\n";
  }

  IntegratedImuMeasurement<double> processImu(int64_t curr_t_ns) {
    int64_t prev_t_ns = t_ns;
    Vector3d bg = latest_state->bias_gyro;
    Vector3d ba = latest_state->bias_accel;
    IntegratedImuMeasurement<double> pim{prev_t_ns, bg, ba};

    if (input_imu_queue.empty()) return pim;

    auto pop_imu = [&](ImuData<double>::Ptr& data) -> bool {
      ImuData<double>::Ptr sample;
      input_imu_queue.pop(sample);  // Blocking pop
      if (sample == nullptr) return false;

      // Calibrate sample
      int64_t t = sample->t_ns;
      Vector3 a = calib.calib_accel_bias.getCalibrated(sample->accel.cast<Scalar>());
      Vector3 g = calib.calib_gyro_bias.getCalibrated(sample->gyro.cast<Scalar>());

      data = std::make_shared<ImuData<double>>();
      data->t_ns = t;
      data->accel = a.template cast<double>();
      data->gyro = g.template cast<double>();
      return true;
    };

    typename ImuData<double>::Ptr data = nullptr;
    if (!pop_imu(data)) return pim;

    while (data->t_ns <= prev_t_ns) {
      if (!pop_imu(data)) return pim;
    }

    while (data->t_ns <= curr_t_ns) {
      pim.integrate(*data, accel_cov, gyro_cov);
      if (!pop_imu(data)) return pim;
    }

    // Pretend last IMU sample before "now" happened now
    if (pim.get_start_t_ns() + pim.get_dt_ns() < curr_t_ns) {
      data->t_ns = curr_t_ns;
      pim.integrate(*data, accel_cov, gyro_cov);
    }

    return pim;
  }

  int trackOpticalFlowFallbackMotionVectors(const OpticalFlowResult::Ptr& new_transforms, size_t i,
                                            const OpticalFlowInput::Ptr& new_img_vec, const SE3& T_c1_c2) {
    trackPoints(old_pyramid->at(i), pyramid->at(i),  //
                transforms->keypoints[i], new_transforms->keypoints[i],
                new_transforms->tracking_guesses[i],  //
                new_img_vec->masks.at(i), new_img_vec->masks.at(i), T_c1_c2, i, i, false);

    int before_count = new_transforms->keypoints[i].size();

    Keypoints lost_kps{};
    for (const auto& [kpid, kp] : transforms->keypoints[i])
      if (new_transforms->keypoints[i].count(kpid) == 0) lost_kps[kpid] = kp;

    set_guesses_from_motion_vector(lost_kps, new_img_vec->img_data[i].motion_vectors,
                                   new_transforms->tracking_guesses[i]);
    trackPoints(old_pyramid->at(i), pyramid->at(i),  //
                lost_kps, new_transforms->keypoints[i],
                new_transforms->tracking_guesses[i],  //
                new_img_vec->masks.at(i), new_img_vec->masks.at(i), T_c1_c2, i, i, true);

    for (const auto& [kpid, _] : lost_kps)
      if (new_transforms->keypoints[i].count(kpid) > 0)
        new_transforms->fallback_guesses[i][kpid] = new_transforms->tracking_guesses[i].at(kpid);

    int after_count = new_transforms->keypoints[i].size();
    return after_count - before_count;
  }

  int trackMotionVectorsFallbackOpticalFlow(const OpticalFlowResult::Ptr& new_transforms, size_t i,
                                            const OpticalFlowInput::Ptr& new_img_vec, const SE3& T_c1_c2) {
    set_guesses_from_motion_vector(transforms->keypoints[i], new_img_vec->img_data[i].motion_vectors,
                                   new_transforms->tracking_guesses[i]);
    trackPoints(old_pyramid->at(i), pyramid->at(i),  //
                transforms->keypoints[i], new_transforms->keypoints[i],
                new_transforms->tracking_guesses[i],  //
                new_img_vec->masks.at(i), new_img_vec->masks.at(i), T_c1_c2, i, i, true);

    int before_count = new_transforms->keypoints[i].size();

    Keypoints lost_kps{};
    for (const auto& [kpid, kp] : transforms->keypoints[i])
      if (new_transforms->keypoints[i].count(kpid) == 0) lost_kps[kpid] = kp;

    trackPoints(old_pyramid->at(i), pyramid->at(i),  //
                lost_kps, new_transforms->keypoints[i],
                new_transforms->tracking_guesses[i],  //
                new_img_vec->masks.at(i), new_img_vec->masks.at(i), T_c1_c2, i, i, false);

    for (const auto& [kpid, _] : lost_kps)
      if (new_transforms->keypoints[i].count(kpid) > 0)
        new_transforms->fallback_guesses[i][kpid] = new_transforms->tracking_guesses[i].at(kpid);

    int after_count = new_transforms->keypoints[i].size();
    return after_count - before_count;
  }

  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    if (isDirectMotionVectorMode()) {
      processDirectMotionVectorFrame(curr_t_ns, new_img_vec);
      return;
    }

    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }

    const size_t num_cams = getNumCams();

    if (t_ns < 0) {
      t_ns = curr_t_ns;

      transforms.reset(new OpticalFlowResult);
      transforms->keypoints.resize(num_cams);
      transforms->keypoint_responses.resize(num_cams);
      transforms->tracking_guesses.resize(num_cams);
      transforms->fallback_guesses.resize(num_cams);
      transforms->matching_guesses.resize(num_cams);
      transforms->recall_guesses.resize(num_cams);
      transforms->t_ns = t_ns;

      pyramid.reset(new std::vector<ManagedImagePyr<uint16_t>>);
      pyramid->resize(num_cams);

      tbb::parallel_for(tbb::blocked_range<size_t>(0, num_cams), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          pyramid->at(i).setFromImage(*new_img_vec->img_data[i].img, config.optical_flow_levels);
        }
      });
      new_img_vec->addTime("frontend_pyramid_created");
      new_img_vec->addTime("frontend_tracking_ended");

      transforms->input_images = new_img_vec;
      transforms->input_images->addTime("frontend_recall_ended");

      addPoints();
      filterPoints();
      transforms->input_images->addTime("frontend_filter_ended");
    } else {
      t_ns = curr_t_ns;

      old_pyramid = pyramid;

      pyramid.reset(new std::vector<ManagedImagePyr<uint16_t>>);
      pyramid->resize(num_cams);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, num_cams), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          pyramid->at(i).setFromImage(*new_img_vec->img_data[i].img, config.optical_flow_levels);
        }
      });
      new_img_vec->addTime("frontend_pyramid_created");

      OpticalFlowResult::Ptr new_transforms;
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->keypoints.resize(num_cams);
      new_transforms->keypoint_responses.resize(num_cams);
      new_transforms->tracking_guesses.resize(num_cams);
      new_transforms->fallback_guesses.resize(num_cams);
      new_transforms->matching_guesses.resize(num_cams);
      new_transforms->recall_guesses.resize(num_cams);
      fallback_recovered_points[frame_counter].resize(num_cams);
      new_transforms->t_ns = t_ns;

      SE3 T_i1 = latest_state->T_w_i.template cast<Scalar>();
      SE3 T_i2 = predicted_state->T_w_i.template cast<Scalar>();
      for (size_t i = 0; i < num_cams; i++) {
        SE3 T_c1 = T_i1 * calib.T_i_c[i];
        SE3 T_c2 = T_i2 * calib.T_i_c[i];
        SE3 T_c1_c2 = T_c1.inverse() * T_c2;

        bool has_mvs = !new_img_vec->img_data[i].motion_vectors.empty();
        if (has_mvs) {
          int fallback_count = -1;
          if (config.optical_flow_subtype == OpticalFlowSubtype::F2F_MV_FALLBACK_OF) {
            fallback_count = trackMotionVectorsFallbackOpticalFlow(new_transforms, i, new_img_vec, T_c1_c2);
          } else if (config.optical_flow_subtype == OpticalFlowSubtype::F2F_OF_FALLBACK_MV) {
            fallback_count = trackOpticalFlowFallbackMotionVectors(new_transforms, i, new_img_vec, T_c1_c2);
          } else {
            BASALT_ASSERT_MSG(false, "Unknown optical flow subtype");
          }
          fallback_recovered_points[frame_counter][i] += fallback_count;
        } else {
          trackPoints(old_pyramid->at(i), pyramid->at(i),  //
                      transforms->keypoints[i], new_transforms->keypoints[i],
                      new_transforms->tracking_guesses[i],  //
                      new_img_vec->masks.at(i), new_img_vec->masks.at(i), T_c1_c2, i, i, false);
        }

        for (const auto& [kpid, kp] : new_transforms->keypoints[i]) {
          auto guess_it = new_transforms->tracking_guesses[i].find(kpid);
          bool found = guess_it != new_transforms->tracking_guesses[i].end();
          Eigen::Vector2f guess =
              found ? guess_it->second.translation() : transforms->keypoints[i].at(kpid).translation();
          Eigen::Vector2f actual = kp.translation();
          float pixel_error = (actual - guess).norm();
          int rounded_error = int(std::round(pixel_error));
          guess_error_histogram[rounded_error]++;
        }
      }
      new_img_vec->addTime("frontend_tracking_ended");

      transforms = new_transforms;
      transforms->input_images = new_img_vec;

      for (size_t i = 0; i < num_cams; i++) updateCellCounts(i);

      if (config.optical_flow_recall_enable) recallPoints();
      transforms->input_images->addTime("frontend_recall_ended");
      addPoints();
      filterPoints();
      transforms->input_images->addTime("frontend_filter_ended");
    }

    recordGenericFrameStats(transforms);
    if (output_queue && frame_counter % config.optical_flow_skip_frames == 0) {
      transforms->input_images->addTime("frontend_keypoints_pushed");
      output_queue->push(transforms);
    }

    frame_counter++;
  }

  // Median-distance filter from cvvo/trajplot.
  std::vector<float> scoreMVsMedianDistance(const std::vector<MotionVector>& mvs) {
    const int cell = 16;
    int grid_w = static_cast<int>(w) / cell;
    int grid_h = static_cast<int>(h) / cell;

    std::vector<float> median_dist(mvs.size(), std::numeric_limits<float>::quiet_NaN());
    if (grid_w < 3 || grid_h < 3) return median_dist;

    cv::Mat u = cv::Mat::zeros(grid_h, grid_w, CV_32F);
    cv::Mat v = cv::Mat::zeros(grid_h, grid_w, CV_32F);
    cv::Mat cnt = cv::Mat::zeros(grid_h, grid_w, CV_32F);

    for (const auto& mv : mvs) {
      int gx = static_cast<int>(mv.src_x) / cell;
      int gy = static_cast<int>(mv.src_y) / cell;
      if (gx < 0 || gy < 0 || gx >= grid_w || gy >= grid_h) continue;
      const Vector2 displacement = directMotionVectorDisplacement(mv);
      u.at<float>(gy, gx) += static_cast<float>(displacement.x());
      v.at<float>(gy, gx) += static_cast<float>(displacement.y());
      cnt.at<float>(gy, gx) += 1.0f;
    }

    for (int gy = 0; gy < grid_h; gy++)
      for (int gx = 0; gx < grid_w; gx++)
        if (cnt.at<float>(gy, gx) > 0) {
          u.at<float>(gy, gx) /= cnt.at<float>(gy, gx);
          v.at<float>(gy, gx) /= cnt.at<float>(gy, gx);
        }

    auto has_mv = [&](int gy, int gx) -> bool { return cnt.at<float>(gy, gx) > 0; };

    cv::Mat med_u = cv::Mat::zeros(grid_h, grid_w, CV_32F);
    cv::Mat med_v = cv::Mat::zeros(grid_h, grid_w, CV_32F);
    cv::Mat med_valid = cv::Mat::zeros(grid_h, grid_w, CV_8U);

    for (int gy = 1; gy < grid_h - 1; gy++) {
      for (int gx = 1; gx < grid_w - 1; gx++) {
        std::vector<float> nbr_u, nbr_v;
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            int ny = gy + dy, nx = gx + dx;
            if (cnt.at<float>(ny, nx) > 0) {
              nbr_u.push_back(u.at<float>(ny, nx));
              nbr_v.push_back(v.at<float>(ny, nx));
            }
          }
        }
        if (nbr_u.size() < 3) continue;
        std::nth_element(nbr_u.begin(), nbr_u.begin() + nbr_u.size() / 2, nbr_u.end());
        std::nth_element(nbr_v.begin(), nbr_v.begin() + nbr_v.size() / 2, nbr_v.end());
        med_u.at<float>(gy, gx) = nbr_u[nbr_u.size() / 2];
        med_v.at<float>(gy, gx) = nbr_v[nbr_v.size() / 2];
        med_valid.at<uint8_t>(gy, gx) = 1;
      }
    }

    for (size_t i = 0; i < mvs.size(); i++) {
      int gx = static_cast<int>(mvs[i].src_x) / cell;
      int gy = static_cast<int>(mvs[i].src_y) / cell;
      if (gx <= 0 || gy <= 0 || gx >= grid_w - 1 || gy >= grid_h - 1) continue;
      if (!has_mv(gy, gx)) continue;
      if (med_valid.at<uint8_t>(gy, gx) == 0) continue;

      float cu = u.at<float>(gy, gx);
      float cv_val = v.at<float>(gy, gx);
      float mu = med_u.at<float>(gy, gx);
      float mv_val = med_v.at<float>(gy, gx);

      median_dist[i] = std::sqrt((cu - mu) * (cu - mu) + (cv_val - mv_val) * (cv_val - mv_val));
    }

    return median_dist;
  }

  // TODO@mateosss: Do this in trackPoints parallel_for
  // TODO@mateosss: Use a quadtree or something more efficient for saving the MotionVectors
  void set_guesses_from_motion_vector(const Keypoints& keypoint_map, const std::vector<MotionVector>& mvs,
                                      Keypoints& guesses) {
    size_t num_mvs = mvs.size();
    for (const auto& [kpid, affine] : keypoint_map) {
      // There is always maximum of one mv per block.
      // TODO@mateosss: previous comment is false. it depends on the x264 "ref" encoding parameter
      // The Motion Vector always starts in the middle of the block.
      for (size_t i = 0; i < num_mvs; ++i) {
        float sx = mvs[i].src_x;
        float sy = mvs[i].src_y;
        float dx = mvs[i].dst_x;
        float dy = mvs[i].dst_y;
        float bh = float(mvs[i].height) / 2;
        float bw = float(mvs[i].width) / 2;
        if (affine.translation().x() >= sx - bw && affine.translation().x() < sx + bw &&
            affine.translation().y() >= sy - bh && affine.translation().y() < sy + bh) {
          Eigen::AffineCompact2f guess = affine;
          guess.translation() = affine.translation() + Eigen::Vector2f{dx - sx, dy - sy};

          auto p = guess.translation();
          if (p.x() < 0 || p.y() < 0 || p.x() >= w || p.y() >= h) continue;

          guesses[kpid] = guess;
          continue;
        }
      }
    }
  }

  void trackPoints(const ManagedImagePyr<uint16_t>& pyr_1, const ManagedImagePyr<uint16_t>& pyr_2,  //
                   const Keypoints& keypoint_map_1, Keypoints& keypoint_map_2, Keypoints& guesses,  //
                   const Masks& masks1, const Masks& masks2, const SE3& T_c1_c2, size_t cam1, size_t cam2,
                   bool has_mvs) const {
    size_t num_points = keypoint_map_1.size();
    size_t existing_points = keypoint_map_2.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Keypoint> init_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);

    for (const auto& [kpid, affine] : keypoint_map_1) {
      ids.push_back(kpid);
      init_vec.push_back(affine);
    }

    tbb::concurrent_unordered_map<KeypointId, Keypoint, std::hash<KeypointId>> result, guesses_tbb;
    guesses_tbb.insert(guesses.begin(), guesses.end());

    bool tracking = cam1 == cam2;
    bool matching = cam1 != cam2;
    MatchingGuessType guess_type = config.optical_flow_matching_guess_type;
    bool match_guess_uses_depth = guess_type != MatchingGuessType::SAME_PIXEL;
    bool has_imu = false;
    const bool use_depth = (tracking && has_imu) || (matching && match_guess_uses_depth);
    const bool use_mvs = !use_depth && has_mvs;
    const double depth = depth_guess;

    auto compute_func = [&](const tbb::blocked_range<size_t>& range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];

        const Eigen::AffineCompact2f& transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = transform_1;

        auto t1 = transform_1.translation();
        auto t2 = transform_2.translation();

        if (masks1.inBounds(t1.x(), t1.y())) continue;

        bool valid = true;

        Eigen::Vector2f off{0, 0};

        if (use_depth) {
          Vector2 t2_guess;
          Scalar _;
          calib.projectBetweenCams(t1, depth, t2_guess, _, T_c1_c2, cam1, cam2);
          off = t2 - t2_guess;
        } else if (use_mvs) {
          auto git = guesses_tbb.find(id);
          if (git != guesses_tbb.end()) off = t2 - git->second.translation();
        }

        t2 -= off;  // This modifies transform_2

        if (true) {  // TODO@mateosss: Put this back to if (show_gui)
          guesses_tbb[id] = transform_2;
        }

        valid = t2(0) >= 0 && t2(1) >= 0 && t2(0) < w && t2(1) < h;
        if (!valid) continue;

        valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2);
        if (!valid) continue;

        if (masks2.inBounds(t2.x(), t2.y())) continue;

        Eigen::AffineCompact2f transform_1_recovered = transform_2;
        auto t1_recovered = transform_1_recovered.translation();

        t1_recovered += off;

        valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);
        if (!valid) continue;

        Scalar dist2 = (t1 - t1_recovered).squaredNorm();

        if (dist2 < config.optical_flow_max_recovered_dist2) {
          result[id] = transform_2;
        }
      }
    };

    int64_t in_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    tbb::blocked_range<size_t> range(0, num_points);
    tbb::parallel_for(range, compute_func);
    int64_t out_ns = std::chrono::steady_clock::now().time_since_epoch().count();

    keypoint_map_2.insert(result.begin(), result.end());
    guesses.insert(guesses_tbb.begin(), guesses_tbb.end());

    if (tracking) {
      if (has_mvs) {
        mv_total_tracking_time += out_ns - in_ns;
        mv_total_tracked_points += keypoint_map_2.size() - existing_points;
      } else {
        of_total_tracking_time += out_ns - in_ns;
        of_total_tracked_points += keypoint_map_2.size() - existing_points;
      }
    }
  }

  inline bool trackPoint(const ManagedImagePyr<uint16_t>& old_pyr, const ManagedImagePyr<uint16_t>& pyr,
                         const Eigen::AffineCompact2f& old_transform, Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    transform.linear().setIdentity();

    for (int level = config.optical_flow_levels; level >= 0 && patch_valid; level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      patch_valid &= p.valid;
      if (patch_valid) {
        // Perform tracking on current level
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);
      }

      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const uint16_t>& img_2, const PatchT& dp, Eigen::AffineCompact2f& transform,
                                Scalar* out_error = nullptr) const {
    bool patch_valid = true;

    for (int iteration = 0; patch_valid && iteration < config.optical_flow_max_iterations; iteration++) {
      typename PatchT::VectorP res;

      typename PatchT::Matrix2P transformed_pat = transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      patch_valid &= dp.residual(img_2, transformed_pat, res);

      if (out_error) *out_error = res.norm();

      if (patch_valid) {
        const Vector3 inc = -dp.H_se2_inv_J_se2_T * res;

        // avoid NaN in increment (leads to SE2::exp crashing)
        patch_valid &= inc.array().isFinite().all();

        // avoid very large increment
        patch_valid &= inc.template lpNorm<Eigen::Infinity>() < 1e6;

        if (patch_valid) {
          transform *= SE2::exp(inc).matrix();

          const int filter_margin = 2;

          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
        }
      }
    }

    return patch_valid;
  }

  inline bool trackPointFromPyrPatch(const ManagedImagePyr<uint16_t>& pyr,
                                     const Eigen::aligned_vector<PatchT>& patch_vec,
                                     Eigen::AffineCompact2f& transform) const {
    bool patch_valid = true;

    //! @note For some reason resetting transform linear part would be wrong only in patch_optical_flow?
    // transform.linear().setIdentity();

    for (int level = config.optical_flow_levels; level >= 0 && patch_valid; level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      const auto& p = patch_vec[level];
      patch_valid &= p.valid;
      if (patch_valid) {  // Perform tracking on current level
        Scalar error = -1;
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform, &error);
        patch_valid &= error <= config.optical_flow_recall_max_patch_norms.at(level);
      }

      transform.translation() *= scale;
    }

    return patch_valid;
  }

  /**
   * @brief Project the landmarks into the current frame.
   * Returns the landmarks that are in the frame along with their corresponding 2D projections.
   *
   * @param[in] cam_id: The camera id of the current frame.
   * @param[out] landmarks: A reference where landmarks will be returned.
   * @param[out] projections: A reference where the landmark's projections will be returned.
   */
  Eigen::aligned_map<LandmarkId, Vector2> getProjectedLandmarks(size_t cam_id) const {
    using Eigen::Matrix4Xf, Eigen::Map, Eigen::aligned_vector, Eigen::Vector4f, Eigen::Aligned;

    SE3 T_i1 = predicted_state->T_w_i.template cast<Scalar>();
    SE3 T_i_cj = calib.T_i_c[cam_id];
    SE3 T_w_cj = T_i1 * T_i_cj;
    SE3 T_cj_w = T_w_cj.inverse();

    const aligned_vector<LandmarkId>& lmids = latest_lm_bundle->lmids;
    const aligned_vector<Vector4f>& lms = latest_lm_bundle->lms;
    const aligned_vector<Vector4f> cj_xyz_vec(lms.size());

    Map<Matrix4Xf, Aligned> w_xyz((float*)lms.data(), 4, lms.size());
    Map<Matrix4Xf, Aligned> cj_xyz((float*)cj_xyz_vec.data(), 4, cj_xyz_vec.size());
    cj_xyz.noalias() = T_cj_w.matrix() * w_xyz;

    aligned_vector<Vector2> cj_uvs;
    std::vector<bool> valid_uvs;
    calib.intrinsics[cam_id].project(cj_xyz_vec, cj_uvs, valid_uvs);

    // TODO: optionally make project() use prefilled valid_uvs and prefill with already tracked
    if (!config.optical_flow_recall_over_tracking) {  // Optionally skip recall for already tracked lms
      // NOTE: With the already_tracked check, this works like patch-OF with fallback
      // to f2f-OF without the check it works as f2f-OF with fallback to patch-OF.
      for (size_t i = 0; i < lms.size(); i++)
        valid_uvs[i] = valid_uvs[i] && transforms->keypoints.at(cam_id).count(lmids[i]) == 0;
    }

    const Masks& masks = transforms->input_images->masks.at(cam_id);
    for (size_t i = 0; i < lms.size(); i++)
      valid_uvs[i] = valid_uvs[i] && !masks.inBounds(cj_uvs[i].x(), cj_uvs[i].y());

    // Safe radius check
    Scalar sr = config.optical_flow_image_safe_radius;  // TODO: This should come from calib not config
    Vector2 center{w / 2, h / 2};
    if (sr != 0)
      for (size_t i = 0; i < lms.size(); i++) valid_uvs[i] = valid_uvs[i] && (cj_uvs[i] - center).norm() <= sr;

    // In-bounds check
    for (size_t i = 0; i < lms.size(); i++)
      valid_uvs[i] = valid_uvs[i] && cj_uvs[i].x() >= 0 && cj_uvs[i].x() < w && cj_uvs[i].y() >= 0 && cj_uvs[i].y() < h;

    Eigen::aligned_map<LandmarkId, Vector2> projections;
    for (size_t i = 0; i < lms.size(); i++)
      if (valid_uvs[i]) projections[lmids[i]] = cj_uvs[i];

    return projections;
  }

  //! Recover tracks from older landmarks into the current frame
  //! TODO: Parallelize recall
  void recallPointsForCamera(size_t cam_id) {
    if (latest_lm_bundle == nullptr) return;

    recalls[cam_id].clear();

    // Project the landmarks from the map into the new frame to obtain their projections.
    Eigen::aligned_map<LandmarkId, Vector2> projections = getProjectedLandmarks(cam_id);

    for (const auto& [lm_id, proj_pos] : projections) {
      Eigen::AffineCompact2f proj_pose = Eigen::AffineCompact2f::Identity();
      proj_pose.translation() = proj_pos;
      if (show_gui) transforms->recall_guesses.at(cam_id)[lm_id] = proj_pose;

      // Optionally cap recalls to max number of points in gridcell
      if (config.optical_flow_recall_num_points_cell)
        if (getCellCount(proj_pos, cam_id) >= config.optical_flow_detection_num_points_cell) continue;

      Eigen::aligned_vector<PatchT>& lm_patch = patches.at(lm_id);
      Eigen::AffineCompact2f curr_pose = proj_pose;
      bool valid = trackPointFromPyrPatch(pyramid->at(cam_id), lm_patch, curr_pose);
      if (!valid) continue;

      // Optionally limit recalled patch reprojected distance
      if (config.optical_flow_recall_max_patch_dist > 0) {
        float max_patch_dist = config.optical_flow_recall_max_patch_dist / 100 * w;
        valid &= (curr_pose.translation() - proj_pos).norm() <= max_patch_dist;
        if (!valid) continue;
      }

      if (config.optical_flow_recall_update_patch_viewpoint) {
        // Update patch viewpoint, see point 2 of the following discussion
        // https://gitlab.com/VladyslavUsenko/basalt/-/issues/69#note_906947578
        // TODO: Efficiently storing more than one patch for the same feature from
        // multiple viewpoints could yield even more accurate feature detection.
        for (int l = 0; l <= config.optical_flow_levels; l++) {
          Scalar scale = 1 << l;
          Vector2 pos_scaled = curr_pose.translation() / scale;
          lm_patch[l] = PatchT(pyramid->at(0).lvl(l), pos_scaled);
        }
      }

      addKeypoint(cam_id, lm_id, curr_pose);
      recalls[cam_id][lm_id] = curr_pose;
    }
  }

  void recallPoints() {
    size_t max_cams = config.optical_flow_recall_all_cams ? getNumCams() : 1;
    for (size_t i = 0; i < max_cams; i++) recallPointsForCamera(i);
  }

  Keypoints addPointsForCamera(size_t cam_id) {
    KeypointsData kd;  // Detected new points
    detectKeypointsWithCells(pyramid->at(cam_id).lvl(0), kd, cells.at(cam_id), config.optical_flow_detection_grid_size,
                             config.optical_flow_detection_num_points_cell, config.optical_flow_detection_min_threshold,
                             config.optical_flow_detection_max_threshold, config.optical_flow_image_safe_radius,
                             transforms->input_images->masks.at(cam_id));

    Keypoints new_kpts;
    for (size_t i = 0; i < kd.corners.size(); i++) {  // Set new points as keypoints
      const Eigen::Vector2d& corner = kd.corners[i];
      const float response = kd.corner_responses[i];

      if (config.optical_flow_recall_enable) {  // Save patch
        // TODO: Patches are never getting deleted
        Eigen::aligned_vector<PatchT>& p = patches[last_keypoint_id];
        Vector2 pos = corner.cast<Scalar>();
        for (int l = 0; l <= config.optical_flow_levels; l++) {
          Scalar scale = 1 << l;
          Vector2 pos_scaled = pos / scale;
          p.emplace_back(pyramid->at(0).lvl(l), pos_scaled);
        }
      }

      auto transform = Eigen::AffineCompact2f::Identity();
      transform.translation() = corner.cast<Scalar>();

      addKeypoint(cam_id, last_keypoint_id, transform, response);
      new_kpts[last_keypoint_id] = transform;

      last_keypoint_id++;
    }

    return new_kpts;
  }

  Masks cam0OverlapCellsMasksForCam(size_t cam_id) const {
    int x_first = x_start + c / 2;
    int y_first = y_start + c / 2;

    int x_last = x_stop + c / 2;
    int y_last = y_stop + c / 2;

    Masks masks;
    for (int y = y_first; y <= y_last; y += c) {
      for (int x = x_first; x <= x_last; x += c) {
        Vector2 ci_uv{x, y};
        Vector2 c0_uv;
        Scalar _;
        bool projected = calib.projectBetweenCams(ci_uv, depth_guess, c0_uv, _, cam_id, 0);
        bool in_bounds = c0_uv.x() >= 0 && c0_uv.x() < w && c0_uv.y() >= 0 && c0_uv.y() < h;
        bool valid = projected && in_bounds;
        if (valid) {
          Rect cell_mask(x - c / 2, y - c / 2, c, c);
          masks.masks.push_back(cell_mask);
        }
      }
    }
    return masks;
  }

  void addPoints() {
    Masks& ms0 = transforms->input_images->masks.at(0);
    Keypoints kpts0 = addPointsForCamera(0);
    transforms->input_images->addTime("frontend_detection_cam0_ended");
    if (config.optical_flow_recall_enable) kpts0.insert(recalls[0].begin(), recalls[0].end());

    for (size_t i = 1; i < getNumCams(); i++) {
      Masks& ms = transforms->input_images->masks.at(i);
      Keypoints& mgs = transforms->matching_guesses.at(i);

      // Match features on areas that overlap with cam0 using optical flow
      auto& pyr0 = pyramid->at(0);
      auto& pyri = pyramid->at(i);
      Keypoints kpts;
      SE3 T_c0_ci = calib.T_i_c[0].inverse() * calib.T_i_c[i];
      trackPoints(pyr0, pyri, kpts0, kpts, mgs, ms0, ms, T_c0_ci, 0, i, false);
      addKeypoints(i, kpts);
    }
    transforms->input_images->addTime("frontend_matching_ended");

    for (size_t i = 1; i < getNumCams(); i++) {
      Masks& ms = transforms->input_images->masks.at(i);

      // Update masks and detect features on area not overlapping with cam0
      if (!config.optical_flow_detection_nonoverlap) continue;
      ms += cam0OverlapCellsMasksForCam(i);
      Keypoints kpts_no = addPointsForCamera(i);
    }
    transforms->input_images->addTime("frontend_detection_cami_ended");
  }

  void filterPointsForCam(int cam_id) {
    std::set<KeypointId> kp_to_remove;

    std::vector<KeypointId> kpids;
    Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

    for (const auto& [kpid, affine] : transforms->keypoints.at(cam_id)) {
      auto it = transforms->keypoints.at(0).find(kpid);

      if (it != transforms->keypoints.at(0).end()) {
        proj0.emplace_back(it->second.translation());
        proj1.emplace_back(affine.translation());
        kpids.emplace_back(kpid);
      }
    }

    Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
    std::vector<bool> p3d0_success, p3d1_success;

    calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success);
    calib.intrinsics[cam_id].unproject(proj1, p3d1, p3d1_success);

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) {
        const double epipolar_error = std::abs(p3d0[i].transpose() * E[cam_id] * p3d1[i]);

        if (epipolar_error > config.optical_flow_epipolar_error) {
          kp_to_remove.emplace(kpids[i]);
        }
      } else {
        kp_to_remove.emplace(kpids[i]);
      }
    }

    for (int id : kp_to_remove) removeKeypoint(cam_id, id);
  }

  void filterPoints() {
    for (size_t i = 1; i < getNumCams(); i++) {
      filterPointsForCam(i);
    }
  }

  void updateCellCounts(size_t cam_id) {
    cells.at(cam_id).setZero(h / c + 1, w / c + 1);
    for (const auto& [lmid, lm] : transforms->keypoints.at(cam_id)) {
      Vector2 p = lm.translation();
      if (p[0] < x_start || p[1] < y_start || p[0] >= x_stop + c || p[1] >= y_stop + c) continue;
      int x = (p[0] - x_start) / c;
      int y = (p[1] - y_start) / c;
      cells.at(cam_id)(y, x)++;
    }
  }

  //! Number of points that are in the grid cell of point xy
  int getCellCount(const Vector2& xy, size_t cam_id) const {
    int x = (xy[0] - x_start) / c;
    int y = (xy[1] - y_start) / c;
    int neighbors = cells.at(cam_id)(y, x);
    return neighbors;
  }

  void addKeypoint(size_t cam_id, KeypointId kpid, Eigen::Affine2f kp, float response = -1.0) {
    int x = (kp.translation().x() - x_start) / c;
    int y = (kp.translation().y() - y_start) / c;
    cells.at(cam_id)(y, x)++;
    transforms->keypoint_responses.at(cam_id)[kpid] = response;
    transforms->keypoints.at(cam_id)[kpid] = kp;
  }

  void addKeypoints(size_t cam_id, Keypoints kpts) {
    for (const auto& [kpid, kp] : kpts) {
      int x = (kp.translation().x() - x_start) / c;
      int y = (kp.translation().y() - y_start) / c;
      cells.at(cam_id)(y, x)++;
    }
    transforms->keypoints.at(cam_id).insert(kpts.begin(), kpts.end());
  }

  void removeKeypoint(size_t cam_id, KeypointId kpid) {
    Vector2 pos = transforms->keypoints.at(cam_id).at(kpid).translation();
    int x = (pos.x() - x_start) / c;
    int y = (pos.y() - y_start) / c;
    cells.at(cam_id)(y, x)--;
    transforms->keypoints.at(cam_id).erase(kpid);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  Eigen::aligned_unordered_map<KeypointId, Eigen::aligned_vector<PatchT>> patches;
  Eigen::aligned_vector<Eigen::MatrixXi> cells;  // Number of features in each gridcell
  std::vector<Keypoints> recalls;
  bool direct_stereo_baseline_rectification_valid = false;
  Matrix3 direct_stereo_R_0_rectified = Matrix3::Identity();
  Matrix3 direct_stereo_R_1_rectified = Matrix3::Identity();
  Vector2 direct_stereo_rectified_focal = Vector2::Zero();
  Vector2 direct_stereo_rectified_center = Vector2::Zero();
  uint64_t direct_mv_tracks = 0;
  uint64_t direct_mv_hybrid_reciprocal_tracks = 0;
  uint64_t direct_mv_cam0_input_vectors = 0;
  uint64_t direct_mv_cam0_rejected_nonpast_source = 0;
  uint64_t direct_mv_cam0_rejected_median_filter = 0;
  uint64_t direct_mv_cam0_rejected_image_bounds = 0;
  uint64_t direct_mv_cam0_rejected_safe_radius = 0;
  uint64_t direct_mv_cam0_rejected_mask = 0;
  uint64_t direct_mv_cam0_accepted_tracks = 0;
  uint64_t direct_mv_teaser_frames = 0;
  uint64_t direct_mv_teaser_successful_frames = 0;
  uint64_t direct_mv_teaser_failed_frames = 0;
  uint64_t direct_mv_teaser_insufficient_frames = 0;
  uint64_t direct_mv_teaser_candidate_pairs = 0;
  uint64_t direct_mv_teaser_pair_input_tracks = 0;
  uint64_t direct_mv_teaser_pair_missing_both_stereo = 0;
  uint64_t direct_mv_teaser_pair_missing_previous_stereo = 0;
  uint64_t direct_mv_teaser_pair_missing_current_stereo = 0;
  uint64_t direct_mv_teaser_pair_invalid_3d = 0;
  uint64_t direct_mv_teaser_solver_pairs = 0;
  uint64_t direct_mv_teaser_solver_inliers = 0;
  uint64_t direct_mv_teaser_validated_edges = 0;
  uint64_t direct_mv_teaser_accepted_edges = 0;
  uint64_t direct_mv_teaser_rejected_no_model = 0;
  uint64_t direct_mv_teaser_rejected_invalid_depth = 0;
  uint64_t direct_mv_teaser_rejected_metric = 0;
  uint64_t direct_mv_teaser_rejected_far_reprojection = 0;
  uint64_t direct_mv_teaser_authority_candidate_edges = 0;
  uint64_t direct_mv_teaser_authority_accepted_candidates = 0;
  uint64_t direct_mv_teaser_authority_selected_edges = 0;
  uint64_t direct_mv_teaser_authority_no_candidate = 0;
  uint64_t direct_mv_teaser_authority_queries = 0;
  uint64_t direct_mv_teaser_authority_indexed_candidates = 0;
  uint64_t direct_mv_teaser_authority_rejected_not_covering_source = 0;
  uint64_t direct_mv_teaser_authority_rejected_invalid_destination = 0;
  uint64_t direct_mv_teaser_authority_rejected_current_stereo = 0;
  uint64_t direct_mv_teaser_authority_rejected_teaser = 0;
  uint64_t direct_mv_teaser_authority_failed_no_spatial_bin = 0;
  uint64_t direct_mv_teaser_authority_failed_no_available_candidate = 0;
  uint64_t direct_mv_teaser_authority_failed_no_covering_candidate = 0;
  uint64_t direct_mv_teaser_authority_failed_invalid_destination = 0;
  uint64_t direct_mv_teaser_authority_failed_current_stereo = 0;
  uint64_t direct_mv_teaser_authority_failed_teaser_gate = 0;
  uint64_t direct_mv_teaser_authority_bidirectional_accepted_queries = 0;
  uint64_t direct_mv_teaser_authority_forward_only_accepted_queries = 0;
  uint64_t direct_mv_teaser_authority_reverse_only_accepted_queries = 0;
  uint64_t direct_mv_teaser_authority_other_origin_accepted_queries = 0;
  long double
      direct_mv_teaser_authority_bidirectional_residual_difference_sum_m = 0;
  long double
      direct_mv_teaser_authority_bidirectional_destination_difference_sum_px = 0;
  long double
      direct_mv_teaser_authority_bidirectional_depth_difference_sum_m = 0;
  double direct_mv_teaser_authority_bidirectional_residual_difference_max_m = 0;
  double direct_mv_teaser_authority_bidirectional_destination_difference_max_px = 0;
  double direct_mv_teaser_authority_bidirectional_depth_difference_max_m = 0;
  uint64_t direct_mv_teaser_authority_started = 0;
  uint64_t direct_mv_teaser_authority_continuations = 0;
  uint64_t direct_mv_track_continuation_attempts = 0;
  uint64_t direct_mv_track_rejected_optimizer_observation = 0;
  uint64_t direct_mv_track_rejected_previous_stereo = 0;
  uint64_t direct_mv_track_rejected_temporal_match = 0;
  uint64_t direct_mv_track_rejected_i_frame = 0;
  uint64_t direct_mv_selected_forward_assignments = 0;
  uint64_t direct_mv_selected_reverse_assignments = 0;
  uint64_t direct_mv_selected_reciprocal_assignments = 0;
  uint64_t direct_mv_selected_unknown_assignments = 0;
  uint64_t direct_mv_selected_unique_mv_groups = 0;
  uint64_t direct_mv_selected_shared_mv_groups = 0;
  uint64_t direct_mv_selected_shared_assignments = 0;
  uint64_t direct_mv_selected_duplicate_assignments = 0;
  uint64_t direct_mv_selected_mv_max_multiplicity = 0;
  long double direct_mv_selected_shared_previous_depth_range_sum_m = 0;
  long double direct_mv_selected_shared_current_depth_range_sum_m = 0;
  long double direct_mv_selected_shared_teaser_residual_range_sum_m = 0;
  double direct_mv_selected_shared_previous_depth_range_max_m = 0;
  double direct_mv_selected_shared_current_depth_range_max_m = 0;
  double direct_mv_selected_shared_teaser_residual_range_max_m = 0;
  std::map<uint64_t, uint64_t> direct_mv_selected_mv_multiplicity_histogram;
  std::array<uint64_t, kDirectSharedSpreadHistogramBins>
      direct_mv_selected_shared_previous_depth_range_histogram{};
  std::array<uint64_t, kDirectSharedSpreadHistogramBins>
      direct_mv_selected_shared_current_depth_range_histogram{};
  std::array<uint64_t, kDirectSharedSpreadHistogramBins>
      direct_mv_selected_shared_teaser_residual_range_histogram{};
  std::array<DirectTrackQualityAggregate, 4>
      direct_mv_track_quality_by_length{};
  std::array<DirectOriginSelectionAggregate, 4>
      direct_mv_selection_quality_by_origin{};
  int direct_mv_reverse_temporal_policy = 0;
  uint64_t direct_mv_reverse_temporal_frames = 0;
  uint64_t direct_mv_reverse_temporal_forward_vectors = 0;
  uint64_t direct_mv_reverse_temporal_reverse_vectors = 0;
  uint64_t direct_mv_reverse_temporal_reciprocal_vectors = 0;
  uint64_t direct_mv_reverse_temporal_single_direction_fallback_frames = 0;
  uint64_t direct_mv_reverse_temporal_single_direction_fallback_vectors = 0;
  uint64_t direct_mv_reverse_temporal_output_vectors = 0;
  uint64_t direct_mv_stereo_rectification_mask_frames = 0;
  int direct_mv_stereo_rectification_mode = STEREO_MV_NATIVE;
  uint64_t direct_mv_stereo_rectification_mask_rejected_vectors = 0;
  uint64_t direct_mv_temporal_subpixel_vectors = 0;
  uint64_t direct_mv_stereo_matches = 0;
  uint64_t direct_mv_codec_stereo_frames = 0;
  uint64_t direct_mv_codec_stereo_queries = 0;
  uint64_t direct_mv_codec_stereo_candidates = 0;
  uint64_t direct_mv_codec_stereo_matches = 0;
  std::map<int, uint64_t> direct_mv_temporal_motion_scale_histogram;
  std::map<int, uint64_t> direct_mv_stereo_motion_scale_histogram;
  uint64_t direct_mv_persistent_tracks_started = 0;
  uint64_t direct_mv_provisional_tracks_started = 0;
  uint64_t direct_mv_provisional_tracks_confirmed = 0;
  uint64_t direct_mv_provisional_tracks_rejected = 0;
  uint64_t direct_mv_merged_continuations = 0;
  uint64_t direct_mv_merged_births = 0;
  uint64_t direct_mv_persistent_track_continuations = 0;
  uint64_t direct_mv_persistent_tracks_killed = 0;
  size_t direct_mv_max_track_frames = 0;
  std::map<size_t, uint64_t> direct_mv_cam0_track_length_histogram_frames;
  uint64_t direct_mv_temporal_depth_residual_samples = 0;
  uint64_t direct_mv_temporal_depth_rate_samples = 0;
  long double direct_mv_temporal_depth_residual_sum_m = 0;
  long double direct_mv_temporal_depth_abs_residual_sum_m = 0;
  long double direct_mv_temporal_depth_squared_residual_sum_m2 = 0;
  long double direct_mv_temporal_depth_abs_rate_sum_mps = 0;
  long double direct_mv_temporal_depth_abs_log_ratio_sum = 0;
  double direct_mv_temporal_depth_max_abs_residual_m = 0;
  double direct_mv_temporal_depth_max_abs_log_ratio = 0;
  std::array<uint64_t, kDirectDepthResidualHistogramBins> direct_mv_temporal_depth_abs_residual_histogram{};
  std::array<uint64_t, kDirectDepthLogRatioHistogramBins> direct_mv_temporal_depth_abs_log_ratio_histogram{};
  std::map<size_t, uint64_t> direct_mv_temporal_depth_gap_histogram_frames;
  uint64_t direct_mv_temporal_depth_immediate_samples = 0;
  long double direct_mv_temporal_depth_immediate_residual_sum_m = 0;
  long double direct_mv_temporal_depth_immediate_abs_residual_sum_m = 0;
  long double direct_mv_temporal_depth_immediate_squared_residual_sum_m2 = 0;
  long double direct_mv_temporal_depth_immediate_abs_log_ratio_sum = 0;
  double direct_mv_temporal_depth_immediate_max_abs_residual_m = 0;
  double direct_mv_temporal_depth_immediate_max_abs_log_ratio = 0;
  std::array<uint64_t, kDirectDepthResidualHistogramBins>
      direct_mv_temporal_depth_immediate_abs_residual_histogram{};
  std::array<uint64_t, kDirectDepthLogRatioHistogramBins>
      direct_mv_temporal_depth_immediate_abs_log_ratio_histogram{};
  std::map<KeypointId, DirectActiveTrack> direct_active_tracks;
  std::map<KeypointId, size_t> generic_cam0_active_track_lengths;
  std::map<size_t, uint64_t> generic_cam0_track_length_histogram_frames;
  std::map<KeypointId, GenericStereoDepthState> generic_last_stereo_depth;
  GenericDepthResidualStats generic_depth_residual_stats;
  size_t generic_stats_frame_index = 0;
  std::deque<PendingDirectMotionVectorFrame> pending_direct_frames;
  const Vector3d accel_cov;
  const Vector3d gyro_cov;
  const size_t c, w, h, x_start, x_stop, y_start, y_stop;  // Grid properties
};

}  // namespace basalt
