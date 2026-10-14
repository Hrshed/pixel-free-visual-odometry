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
#ifndef DATASET_IO_EUROC_H
#define DATASET_IO_EUROC_H

#include <memory>

#include <basalt/io/dataset_io.h>
#include <basalt/io/motion_vector_cache.h>
#include <basalt/utils/filesystem.h>
#include <libavutil/error.h>
#include <mvextractor/video_cap.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <tuple>

namespace basalt {

class EurocVioDataset : public VioDataset {
  struct StereoMvCapture {
    VideoCap cap;
    int disparity_shift = 0;
    int grid_shift_x = 0;
    int grid_shift_y = 0;
  };

  size_t num_cams;

  std::string path;

  std::vector<int64_t> image_timestamps;
  std::unordered_map<int64_t, std::string> image_path;
  std::vector<cv::VideoCapture> caps;
  std::vector<VideoCap> mv_caps;
  std::vector<StereoMvCapture> stereo_mv_caps;
  bool temporal_mv_caps_initialized = false;
  std::vector<bool> temporal_mv_source_available;
  std::vector<size_t> temporal_mv_capture_positions;
  std::vector<std::unique_ptr<MotionVectorCacheReader>>
      temporal_mv_cache_readers;
  std::vector<std::unique_ptr<MotionVectorCacheWriter>>
      temporal_mv_cache_writers;
  std::vector<std::unique_ptr<MotionVectorCacheReader>>
      reverse_temporal_mv_cache_readers;
  std::unique_ptr<MotionVectorCacheReader> stereo_mv_cache_reader;
  std::unique_ptr<MotionVectorCacheWriter> stereo_mv_cache_writer;
  size_t stereo_mv_capture_position = 0;
  std::vector<std::vector<MotionVector>> previous_motion_vectors;
  std::vector<std::vector<std::vector<MotionVector>>>
      reverse_temporal_motion_vectors;
  std::vector<std::vector<bool>> reverse_temporal_i_frames;
  bool reverse_temporal_motion_vectors_initialized = false;
  bool stereo_mv_caps_initialized = false;
  bool stereo_mvs_right_rectified_to_left = false;
  int stereo_mv_rectification_mode = STEREO_MV_NATIVE;
  std::array<double, 4> stereo_rectified_pinhole{{0, 0, 0, 0}};
  std::array<int, 2> stereo_rectified_resolution{{0, 0}};
  std::array<double, 9> stereo_rotation_cam0_rectified{{0, 0, 0, 0, 0, 0, 0, 0, 0}};
  std::array<double, 9> stereo_rotation_cam1_rectified{{0, 0, 0, 0, 0, 0, 0, 0, 0}};
  cv::Mat stereo_rectified_left_invalid_integral;
  cv::Mat stereo_rectified_right_invalid_integral;
  size_t stereo_rectification_mask_rejections = 0;
  size_t frameNumber = 0;
  size_t mv_frame_counter = 0;
  size_t stereo_mv_pair_counter = 0;

  std::set<size_t> mv_warning_shown_cams;

  // vector of images for every timestamp
  // assumes vectors size is num_cams for every timestamp with null pointers for
  // missing frames
  // std::unordered_map<int64_t, std::vector<ImageData>> image_data;

  Eigen::aligned_vector<AccelData> accel_data;
  Eigen::aligned_vector<GyroData> gyro_data;

  std::vector<int64_t> gt_timestamps;                // ordered gt timestamps
  Eigen::aligned_vector<Sophus::SE3d> gt_pose_data;  // TODO: change to eigen aligned

  int64_t mocap_to_imu_offset_ns = 0;

  std::vector<std::unordered_map<int64_t, double>> exposure_times;

  bool append_temporal_motion_vectors(VideoCap& cap, int64_t t_ns,
                                      size_t cam_idx,
                                      std::vector<MotionVector>& output,
                                      bool& is_i_frame) {
    int ret = cap.grab();
    if (ret < 0) {
      if (ret == AVERROR(EOF) || ret == AVERROR_EOF || ret == -2) return false;
      std::cerr << "Failed to grab temporal MV frame at timestamp " << t_ns << " on cam" << cam_idx << "."
                << std::endl;
      std::abort();
    }

    uint8_t* frame = nullptr;
    int step = 0;
    int width = 0;
    int height = 0;
    int cn = 0;
    char frame_type[2]{};
    MVS_DTYPE* motion_vectors = nullptr;
    MVS_DTYPE num_mvs = 0;
    double frame_timestamp = 0;
    ret = cap.retrieve(&frame, &step, &width, &height, &cn, frame_type, &motion_vectors, &num_mvs, &frame_timestamp);
    if (ret < 0) {
      std::free(motion_vectors);
      if (ret == AVERROR(EOF) || ret == AVERROR_EOF) return false;
      std::cerr << "Failed to retrieve temporal MV frame at timestamp " << t_ns << " on cam" << cam_idx << "."
                << std::endl;
      std::abort();
    }

    is_i_frame = frame_type[0] == 'I';
    if (!is_i_frame) {
      output.reserve(output.size() + num_mvs);
      for (MVS_DTYPE j = 0; j < num_mvs * 10; j += 10) {
        const int source = motion_vectors[j];
        const int block_width = motion_vectors[j + 1];
        const int block_height = motion_vectors[j + 2];
        const int motion_x = motion_vectors[j + 7];
        const int motion_y = motion_vectors[j + 8];
        const int motion_scale = motion_vectors[j + 9];
        if (source != -1 || block_width <= 0 || block_height <= 0 || motion_scale <= 0) {
          continue;
        }
        const float encoded_dst_x = static_cast<float>(motion_vectors[j + 5]);
        const float encoded_dst_y = static_cast<float>(motion_vectors[j + 6]);
        const float encoded_src_x =
            encoded_dst_x + static_cast<float>(motion_x) / static_cast<float>(motion_scale);
        const float encoded_src_y =
            encoded_dst_y + static_cast<float>(motion_y) / static_cast<float>(motion_scale);
        const float src_x = encoded_src_x;
        const float src_y = encoded_src_y;
        const float dst_x = encoded_dst_x;
        const float dst_y = encoded_dst_y;
        if (src_x < 0 || src_y < 0 || dst_x < 0 || dst_y < 0 || src_x >= width || dst_x >= width ||
            src_y >= height || dst_y >= height) {
          continue;
        }
        const float integer_src_x = static_cast<float>(motion_vectors[j + 3]);
        const float integer_src_y = static_cast<float>(motion_vectors[j + 4]);
        if (std::abs(encoded_src_x - integer_src_x) > 1.0f ||
            std::abs(encoded_src_y - integer_src_y) > 1.0f) {
          std::free(motion_vectors);
          std::cerr << "Inconsistent subpixel temporal MV coordinates at timestamp "
                    << t_ns << " on cam" << cam_idx << "." << std::endl;
          std::abort();
        }
        const float quantized_flow_x = -static_cast<float>(motion_x) / static_cast<float>(motion_scale);
        const float quantized_flow_y = -static_cast<float>(motion_y) / static_cast<float>(motion_scale);
        output.push_back({source, block_width, block_height, src_x, src_y, dst_x, dst_y,
                          quantized_flow_x, quantized_flow_y, motion_scale});
      }
    }
    std::free(motion_vectors);
    return true;
  }

  bool decode_temporal_motion_vector_frame(size_t image_index, size_t cam_idx,
                                           std::vector<MotionVector>& output,
                                           bool& is_i_frame) {
    if (cam_idx >= temporal_mv_source_available.size() ||
        !temporal_mv_source_available.at(cam_idx)) {
      return false;
    }
    const int64_t t_ns = image_timestamps.at(image_index);
    output.clear();
    bool base_is_i_frame = false;
    const bool read_ok = append_temporal_motion_vectors(
        mv_caps.at(cam_idx), t_ns, cam_idx, output, base_is_i_frame);
    is_i_frame = read_ok && base_is_i_frame;
    temporal_mv_capture_positions.at(cam_idx)++;
    return read_ok;
  }

  bool read_temporal_motion_vector_frame(size_t image_index, size_t cam_idx,
                                         std::vector<MotionVector>& output,
                                         bool& is_i_frame) {
    uint32_t flags = 0;
    MotionVectorCacheReader* reader =
        cam_idx < temporal_mv_cache_readers.size()
            ? temporal_mv_cache_readers.at(cam_idx).get()
            : nullptr;
    MotionVectorCacheWriter* writer =
        cam_idx < temporal_mv_cache_writers.size()
            ? temporal_mv_cache_writers.at(cam_idx).get()
            : nullptr;
    if (reader && reader->readFrame(static_cast<uint32_t>(image_index),
                                    output, flags)) {
      is_i_frame = (flags & 1U) != 0;
      if (writer) writer->writeFrame(static_cast<uint32_t>(image_index), output, flags);
      return true;
    }

    while (cam_idx < temporal_mv_capture_positions.size() &&
           temporal_mv_capture_positions.at(cam_idx) < image_index) {
      std::vector<MotionVector> discarded;
      bool discarded_i_frame = false;
      if (!decode_temporal_motion_vector_frame(
              temporal_mv_capture_positions.at(cam_idx), cam_idx, discarded,
              discarded_i_frame)) {
        return false;
      }
    }
    const bool read_ok = decode_temporal_motion_vector_frame(
        image_index, cam_idx, output, is_i_frame);
    if (read_ok && writer) {
      writer->writeFrame(static_cast<uint32_t>(image_index), output,
                         is_i_frame ? 1U : 0U);
    }
    return read_ok;
  }

  void appendReciprocalTemporalMotionVectors(
      const std::vector<MotionVector>& forward_mvs,
      const std::vector<MotionVector>& reverse_mvs,
      std::vector<MotionVector>& output) const {
    constexpr int kBinSize = 8;
    constexpr float kZeroMotionSquared = 1e-12f;
    if (forward_mvs.empty() || reverse_mvs.empty()) return;

    struct Rectangle {
      float min_x;
      float max_x;
      float min_y;
      float max_y;
    };
    struct Intersection {
      float min_x;
      float max_x;
      float min_y;
      float max_y;
      float area;
    };
    const auto rectangle = [](float x, float y, int width, int height) {
      const float half_width = 0.5f * static_cast<float>(width);
      const float half_height = 0.5f * static_cast<float>(height);
      return Rectangle{x - half_width, x + half_width, y - half_height, y + half_height};
    };
    const auto intersection = [](const Rectangle& lhs, const Rectangle& rhs) {
      Intersection result{std::max(lhs.min_x, rhs.min_x), std::min(lhs.max_x, rhs.max_x),
                          std::max(lhs.min_y, rhs.min_y), std::min(lhs.max_y, rhs.max_y), 0.0f};
      result.area = std::max(0.0f, result.max_x - result.min_x) *
                    std::max(0.0f, result.max_y - result.min_y);
      return result;
    };
    const auto rectangle_area = [](const Rectangle& value) {
      return std::max(0.0f, value.max_x - value.min_x) * std::max(0.0f, value.max_y - value.min_y);
    };
    const auto iou = [&](const Rectangle& lhs, const Rectangle& rhs, const Intersection& overlap) {
      const float union_area = rectangle_area(lhs) + rectangle_area(rhs) - overlap.area;
      return union_area > 0.0f ? overlap.area / union_area : 0.0f;
    };

    int max_x = 0;
    int max_y = 0;
    for (const MotionVector& reverse : reverse_mvs) {
      max_x = std::max(max_x, static_cast<int>(std::ceil(reverse.src_x + 0.5f * reverse.width)));
      max_y = std::max(max_y, static_cast<int>(std::ceil(reverse.src_y + 0.5f * reverse.height)));
    }
    const int bin_cols = std::max(1, (max_x + kBinSize - 1) / kBinSize);
    const int bin_rows = std::max(1, (max_y + kBinSize - 1) / kBinSize);
    std::vector<std::vector<size_t>> bins(static_cast<size_t>(bin_cols * bin_rows));
    for (size_t reverse_index = 0; reverse_index < reverse_mvs.size(); ++reverse_index) {
      const MotionVector& reverse = reverse_mvs.at(reverse_index);
      const Rectangle source = rectangle(reverse.src_x, reverse.src_y, reverse.width, reverse.height);
      const int min_bin_x = std::clamp(static_cast<int>(std::floor(source.min_x)) / kBinSize, 0, bin_cols - 1);
      const int max_bin_x = std::clamp(static_cast<int>(std::floor(source.max_x)) / kBinSize, 0, bin_cols - 1);
      const int min_bin_y = std::clamp(static_cast<int>(std::floor(source.min_y)) / kBinSize, 0, bin_rows - 1);
      const int max_bin_y = std::clamp(static_cast<int>(std::floor(source.max_y)) / kBinSize, 0, bin_rows - 1);
      for (int bin_y = min_bin_y; bin_y <= max_bin_y; ++bin_y)
        for (int bin_x = min_bin_x; bin_x <= max_bin_x; ++bin_x)
          bins.at(static_cast<size_t>(bin_y * bin_cols + bin_x)).push_back(reverse_index);
    }

    struct Candidate {
      size_t forward_index;
      size_t reverse_index;
      float overlap;
      float cosine;
      float cycle_error_squared;
      float source_x;
      float source_y;
      float destination_x;
      float destination_y;
      int width;
      int height;
    };
    std::vector<Candidate> candidates;
    for (size_t forward_index = 0; forward_index < forward_mvs.size(); ++forward_index) {
      const MotionVector& forward = forward_mvs.at(forward_index);
      const Rectangle forward_source = rectangle(forward.src_x, forward.src_y, forward.width, forward.height);
      const Rectangle forward_destination = rectangle(forward.dst_x, forward.dst_y, forward.width, forward.height);
      const int min_bin_x =
          std::clamp(static_cast<int>(std::floor(forward_destination.min_x)) / kBinSize, 0, bin_cols - 1);
      const int max_bin_x =
          std::clamp(static_cast<int>(std::floor(forward_destination.max_x)) / kBinSize, 0, bin_cols - 1);
      const int min_bin_y =
          std::clamp(static_cast<int>(std::floor(forward_destination.min_y)) / kBinSize, 0, bin_rows - 1);
      const int max_bin_y =
          std::clamp(static_cast<int>(std::floor(forward_destination.max_y)) / kBinSize, 0, bin_rows - 1);
      std::vector<size_t> nearby_reverse;
      for (int bin_y = min_bin_y; bin_y <= max_bin_y; ++bin_y)
        for (int bin_x = min_bin_x; bin_x <= max_bin_x; ++bin_x) {
          const auto& bin = bins.at(static_cast<size_t>(bin_y * bin_cols + bin_x));
          nearby_reverse.insert(nearby_reverse.end(), bin.begin(), bin.end());
        }
      std::sort(nearby_reverse.begin(), nearby_reverse.end());
      nearby_reverse.erase(std::unique(nearby_reverse.begin(), nearby_reverse.end()), nearby_reverse.end());

      const float forward_x = forward.dst_x - forward.src_x;
      const float forward_y = forward.dst_y - forward.src_y;
      const float forward_norm_squared = forward_x * forward_x + forward_y * forward_y;
      for (size_t reverse_index : nearby_reverse) {
        const MotionVector& reverse = reverse_mvs.at(reverse_index);
        const Rectangle reverse_source = rectangle(reverse.src_x, reverse.src_y, reverse.width, reverse.height);
        const Rectangle reverse_destination = rectangle(reverse.dst_x, reverse.dst_y, reverse.width, reverse.height);
        const Intersection source_overlap = intersection(forward_source, reverse_destination);
        const Intersection destination_overlap = intersection(forward_destination, reverse_source);
        if (source_overlap.area <= 0.0f || destination_overlap.area <= 0.0f) continue;

        const float reverse_x = reverse.src_x - reverse.dst_x;
        const float reverse_y = reverse.src_y - reverse.dst_y;
        const float reverse_norm_squared = reverse_x * reverse_x + reverse_y * reverse_y;
        float cosine = -1.0f;
        if (forward_norm_squared <= kZeroMotionSquared && reverse_norm_squared <= kZeroMotionSquared) {
          cosine = 1.0f;
        } else if (forward_norm_squared <= kZeroMotionSquared || reverse_norm_squared <= kZeroMotionSquared) {
          continue;
        } else {
          cosine = (forward_x * reverse_x + forward_y * reverse_y) /
                   std::sqrt(forward_norm_squared * reverse_norm_squared);
        }
        if (!std::isfinite(cosine) || cosine < temporal_reverse_cosine_threshold) continue;

        const float source_iou = iou(forward_source, reverse_destination, source_overlap);
        const float destination_iou = iou(forward_destination, reverse_source, destination_overlap);
        const float cycle_x = forward_x - reverse_x;
        const float cycle_y = forward_y - reverse_y;
        const float cycle_error_squared = cycle_x * cycle_x + cycle_y * cycle_y;
        if (std::min(source_iou, destination_iou) < temporal_reverse_min_iou ||
            (temporal_reverse_max_cycle_error >= 0.0f &&
             cycle_error_squared > temporal_reverse_max_cycle_error * temporal_reverse_max_cycle_error)) {
          continue;
        }

        const Rectangle destination_support_in_source{
            destination_overlap.min_x - forward_x, destination_overlap.max_x - forward_x,
            destination_overlap.min_y - forward_y, destination_overlap.max_y - forward_y};
        const Rectangle source_support{source_overlap.min_x, source_overlap.max_x, source_overlap.min_y,
                                       source_overlap.max_y};
        const Intersection translated_support = intersection(source_support, destination_support_in_source);
        const int validated_width =
            static_cast<int>(std::floor(translated_support.max_x - translated_support.min_x));
        const int validated_height =
            static_cast<int>(std::floor(translated_support.max_y - translated_support.min_y));
        if (translated_support.area <= 0.0f || validated_width < 1 || validated_height < 1) continue;

        const float source_x = 0.5f * (translated_support.min_x + translated_support.max_x);
        const float source_y = 0.5f * (translated_support.min_y + translated_support.max_y);
        candidates.push_back({forward_index,
                              reverse_index,
                              std::min(source_iou, destination_iou),
                              cosine,
                              cycle_error_squared,
                              source_x,
                              source_y,
                              source_x + forward_x,
                              source_y + forward_y,
                              validated_width,
                              validated_height});
      }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
      if (lhs.overlap != rhs.overlap) return lhs.overlap > rhs.overlap;
      if (lhs.cosine != rhs.cosine) return lhs.cosine > rhs.cosine;
      if (lhs.cycle_error_squared != rhs.cycle_error_squared)
        return lhs.cycle_error_squared < rhs.cycle_error_squared;
      if (lhs.forward_index != rhs.forward_index) return lhs.forward_index < rhs.forward_index;
      return lhs.reverse_index < rhs.reverse_index;
    });
    std::vector<bool> forward_used(forward_mvs.size(), false);
    std::vector<bool> reverse_used(reverse_mvs.size(), false);
    for (const Candidate& candidate : candidates) {
      if (forward_used.at(candidate.forward_index) || reverse_used.at(candidate.reverse_index)) continue;
      const float flow_x = candidate.destination_x - candidate.source_x;
      const float flow_y = candidate.destination_y - candidate.source_y;
      output.push_back({-1, candidate.width, candidate.height, candidate.source_x, candidate.source_y,
                        candidate.destination_x, candidate.destination_y, flow_x, flow_y,
                        forward_mvs.at(candidate.forward_index).motion_scale});
      forward_used.at(candidate.forward_index) = true;
      reverse_used.at(candidate.reverse_index) = true;
    }
  }

  static MotionVector invert_reverse_temporal_motion_vector(
      const MotionVector& reverse) {
    MotionVector forward = reverse;
    forward.source = -1;
    std::swap(forward.src_x, forward.dst_x);
    std::swap(forward.src_y, forward.dst_y);
    forward.quantized_flow_x = -reverse.quantized_flow_x;
    forward.quantized_flow_y = -reverse.quantized_flow_y;
    return forward;
  }

  static void append_bidirectional_temporal_candidates(
      const std::vector<MotionVector>& forward_mvs,
      const std::vector<MotionVector>& reverse_mvs,
      std::vector<MotionVector>& output) {
    output = forward_mvs;
    output.reserve(output.size() + reverse_mvs.size());
    for (const MotionVector& reverse : reverse_mvs) {
      output.emplace_back(invert_reverse_temporal_motion_vector(reverse));
    }
  }

  void initialize_reverse_temporal_motion_vectors(
      const std::string& video_root) {
    if (reverse_temporal_motion_vectors_initialized ||
        temporal_reverse_policy == 0) {
      return;
    }

    reverse_temporal_motion_vectors_initialized = true;
    reverse_temporal_motion_vectors.resize(num_cams);
    reverse_temporal_i_frames.resize(num_cams);
    reverse_temporal_mv_cache_readers.resize(num_cams);
    for (size_t cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
      const std::string reverse_video_path =
          video_root + "/mav0/cam" + std::to_string(cam_idx) +
          "/data_reverse.mp4";
      if (!fs::exists(reverse_video_path)) {
        std::cerr << "Reverse temporal MVs requested but artifact is missing: "
                  << reverse_video_path << "." << std::endl;
        std::abort();
      }

      const std::string cache_path =
          video_root + "/mav0/cam" + std::to_string(cam_idx) +
          "/.basalt-mv-cache/reverse.bin";
      const uint64_t fingerprint = motionVectorCacheFingerprint(
          {reverse_video_path}, UINT64_C(0x7265766572736531));
      reverse_temporal_mv_cache_readers.at(cam_idx) =
          std::make_unique<MotionVectorCacheReader>();
      MotionVectorCacheReader& cache =
          *reverse_temporal_mv_cache_readers.at(cam_idx);
      if (cache.open(cache_path, fingerprint,
                     static_cast<uint32_t>(image_timestamps.size())) &&
          cache.complete()) {
        std::cout << "Loaded cached reverse VOCA temporal MVs for cam"
                  << cam_idx << "." << std::endl;
        continue;
      }
      cache.close();

      VideoCap reverse_cap{};
      if (reverse_cap.open(reverse_video_path.c_str()) < 0) {
        std::cerr << "Unable to open reverse temporal MV video at "
                  << reverse_video_path << "." << std::endl;
        std::abort();
      }
      std::vector<std::vector<MotionVector>> decoded_reverse_frames;
      std::vector<bool> decoded_reverse_i_frames;
      while (true) {
        std::vector<MotionVector> frame;
        bool is_i_frame = false;
        if (!append_temporal_motion_vectors(reverse_cap, 0, cam_idx, frame,
                                            is_i_frame)) {
          break;
        }
        decoded_reverse_frames.emplace_back(std::move(frame));
        decoded_reverse_i_frames.emplace_back(is_i_frame);
      }
      reverse_cap.release();
      if (decoded_reverse_frames.size() < 2 ||
          decoded_reverse_frames.size() > image_timestamps.size()) {
        std::cerr << "Reverse temporal video on cam" << cam_idx << " has "
                  << decoded_reverse_frames.size() << " frames for a dataset with "
                  << image_timestamps.size() << " timestamps." << std::endl;
        std::abort();
      }

      auto& by_current_image = reverse_temporal_motion_vectors.at(cam_idx);
      by_current_image.resize(image_timestamps.size());
      auto& i_frame_by_current_image = reverse_temporal_i_frames.at(cam_idx);
      i_frame_by_current_image.resize(image_timestamps.size(), true);
      const size_t encoded_frames = decoded_reverse_frames.size();
      for (size_t reverse_index = 1; reverse_index < encoded_frames;
           ++reverse_index) {
        const size_t current_image_index = encoded_frames - reverse_index;
        by_current_image.at(current_image_index) =
            std::move(decoded_reverse_frames.at(reverse_index));
        i_frame_by_current_image.at(current_image_index) =
            decoded_reverse_i_frames.at(reverse_index);
      }
      MotionVectorCacheWriter writer;
      if (writer.open(cache_path, fingerprint,
                      static_cast<uint32_t>(image_timestamps.size()))) {
        for (size_t frame_index = 0; frame_index < image_timestamps.size();
             ++frame_index) {
          writer.writeFrame(
              static_cast<uint32_t>(frame_index),
              by_current_image.at(frame_index),
              i_frame_by_current_image.at(frame_index) ? 1U : 0U);
        }
        writer.finish();
        if (cache.open(cache_path, fingerprint,
                       static_cast<uint32_t>(image_timestamps.size())) &&
            cache.complete()) {
          by_current_image.clear();
          i_frame_by_current_image.clear();
        }
      }
      std::cout << "Loaded " << encoded_frames
                << " reverse VOCA temporal frames for cam" << cam_idx << "."
                << std::endl;
    }
  }

  void loadBaselineRectificationManifest(const std::string& manifest_path) {
    cv::FileStorage manifest(manifest_path,
                             cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!manifest.isOpened()) {
      std::cerr << "Unable to open stereo baseline rectification manifest at "
                << manifest_path << "." << std::endl;
      std::abort();
    }
    int version = 0;
    std::string model;
    manifest["version"] >> version;
    manifest["model"] >> model;
    manifest["width"] >> stereo_rectified_resolution[0];
    manifest["height"] >> stereo_rectified_resolution[1];
    manifest["fx"] >> stereo_rectified_pinhole[0];
    manifest["fy"] >> stereo_rectified_pinhole[1];
    manifest["cx"] >> stereo_rectified_pinhole[2];
    manifest["cy"] >> stereo_rectified_pinhole[3];

    const auto read_rotation = [&](const char* name,
                                   std::array<double, 9>& values) {
      const cv::FileNode rotation = manifest[name];
      if (rotation.type() != cv::FileNode::SEQ || rotation.size() != 3)
        return false;
      size_t index = 0;
      for (const cv::FileNode& row : rotation) {
        if (row.type() != cv::FileNode::SEQ || row.size() != 3) return false;
        for (const cv::FileNode& value : row) {
          if (index >= values.size()) return false;
          values[index++] = static_cast<double>(value);
        }
      }
      return index == values.size();
    };
    if (!read_rotation("rotation_cam0_rectified",
                       stereo_rotation_cam0_rectified) ||
        !read_rotation("rotation_cam1_rectified",
                       stereo_rotation_cam1_rectified)) {
      std::cerr << "Invalid rotation arrays in stereo baseline rectification manifest at "
                << manifest_path << "." << std::endl;
      std::abort();
    }

    const auto rotation_is_valid = [](const std::array<double, 9>& values) {
      Eigen::Matrix3d rotation;
      for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
          rotation(row, column) =
              values[static_cast<size_t>(3 * row + column)];
      return rotation.array().isFinite().all() &&
             (rotation.transpose() * rotation -
              Eigen::Matrix3d::Identity()).norm() <= 1e-6 &&
             std::abs(rotation.determinant() - 1.0) <= 1e-6;
    };
    const bool pinhole_valid =
        std::all_of(stereo_rectified_pinhole.begin(),
                    stereo_rectified_pinhole.end(),
                    [](double value) { return std::isfinite(value); }) &&
        stereo_rectified_pinhole[0] > 0 &&
        stereo_rectified_pinhole[1] > 0 &&
        stereo_rectified_resolution[0] > 0 &&
        stereo_rectified_resolution[1] > 0 &&
        stereo_rectified_pinhole[2] >= 0 &&
        stereo_rectified_pinhole[2] < stereo_rectified_resolution[0] &&
        stereo_rectified_pinhole[3] >= 0 &&
        stereo_rectified_pinhole[3] < stereo_rectified_resolution[1];
    if (version != 1 || model != "pinhole" || !pinhole_valid ||
        !rotation_is_valid(stereo_rotation_cam0_rectified) ||
        !rotation_is_valid(stereo_rotation_cam1_rectified)) {
      std::cerr << "Invalid stereo baseline rectification manifest at "
                << manifest_path << "." << std::endl;
      std::abort();
    }
  }

  bool rectifiedSupportIsValid(const cv::Mat& invalid_integral,
                               float center_x, float center_y,
                               int block_width, int block_height,
                               int image_width, int image_height) const {
    if (invalid_integral.empty()) return true;
    if (invalid_integral.cols != image_width + 1 ||
        invalid_integral.rows != image_height + 1) {
      std::cerr << "Stereo rectification validity mask dimensions do not match the encoded video."
                << std::endl;
      std::abort();
    }

    const float half_width = 0.5f * static_cast<float>(block_width);
    const float half_height = 0.5f * static_cast<float>(block_height);
    const int min_x = static_cast<int>(std::floor(center_x - half_width));
    const int max_x = static_cast<int>(std::ceil(center_x + half_width));
    const int min_y = static_cast<int>(std::floor(center_y - half_height));
    const int max_y = static_cast<int>(std::ceil(center_y + half_height));
    if (min_x < 0 || min_y < 0 || max_x > image_width || max_y > image_height ||
        min_x >= max_x || min_y >= max_y) {
      return false;
    }

    const int* top = invalid_integral.ptr<int>(min_y);
    const int* bottom = invalid_integral.ptr<int>(max_y);
    const int invalid_pixels =
        bottom[max_x] - bottom[min_x] - top[max_x] + top[min_x];
    return invalid_pixels == 0;
  }

  void read_stereo_mv_frame(StereoMvCapture& capture, int64_t t_ns, char expected_type, bool source_is_left,
                            std::vector<MotionVector>& output) {
    int ret = capture.cap.grab();
    if (ret < 0) {
      std::cerr << "Failed to grab stereo MV frame for pair " << stereo_mv_pair_counter << " at timestamp " << t_ns
                << "." << std::endl;
      std::abort();
    }

    uint8_t* frame = nullptr;
    int step = 0;
    int width = 0;
    int height = 0;
    int cn = 0;
    char frame_type[2]{};
    MVS_DTYPE* motion_vectors = nullptr;
    MVS_DTYPE num_mvs = 0;
    double frame_timestamp = 0;
    ret = capture.cap.retrieve(&frame, &step, &width, &height, &cn, frame_type, &motion_vectors, &num_mvs,
                               &frame_timestamp);
    if (ret < 0) {
      std::free(motion_vectors);
      std::cerr << "Failed to retrieve stereo MV frame for pair " << stereo_mv_pair_counter << " at timestamp "
                << t_ns << "." << std::endl;
      std::abort();
    }
    if (stereo_mv_rectification_mode == STEREO_MV_BASELINE_PINHOLE &&
        (width != stereo_rectified_resolution[0] ||
         height != stereo_rectified_resolution[1])) {
      std::free(motion_vectors);
      std::cerr << "Decoded stereo video dimensions do not match the baseline rectification manifest."
                << std::endl;
      std::abort();
    }
    if (frame_type[0] != expected_type) {
      std::free(motion_vectors);
      std::cerr << "Invalid stereo GOP at pair " << stereo_mv_pair_counter << ": expected " << expected_type
                << " but decoded " << frame_type[0] << "." << std::endl;
      std::abort();
    }

    if (expected_type == 'P') {
      output.reserve(output.size() + num_mvs);
      for (MVS_DTYPE j = 0; j < num_mvs * 10; j += 10) {
        const int source = motion_vectors[j];
        const int block_width = motion_vectors[j + 1];
        const int block_height = motion_vectors[j + 2];
        const int motion_x = motion_vectors[j + 7];
        const int motion_y = motion_vectors[j + 8];
        const int motion_scale = motion_vectors[j + 9];
        if (source != -1 || block_width <= 0 || block_height <= 0 || motion_scale <= 0) {
          std::free(motion_vectors);
          std::cerr << "Invalid stereo MV metadata in pair " << stereo_mv_pair_counter << "." << std::endl;
          std::abort();
        }

        const float encoded_dst_x = static_cast<float>(motion_vectors[j + 5]);
        const float encoded_dst_y = static_cast<float>(motion_vectors[j + 6]);
        const float encoded_src_x = encoded_dst_x + static_cast<float>(motion_x) / static_cast<float>(motion_scale);
        const float encoded_src_y = encoded_dst_y + static_cast<float>(motion_y) / static_cast<float>(motion_scale);
        const float left_adjustment = static_cast<float>(capture.disparity_shift);
        const float grid_x = static_cast<float>(capture.grid_shift_x);
        const float grid_y = static_cast<float>(capture.grid_shift_y);
        const float src_x = encoded_src_x - grid_x + (source_is_left ? left_adjustment : 0.0f);
        const float src_y = encoded_src_y - grid_y;
        const float dst_x = encoded_dst_x - grid_x + (source_is_left ? 0.0f : left_adjustment);
        const float dst_y = encoded_dst_y - grid_y;
        if (stereo_mv_rectification_mode == STEREO_MV_BASELINE_PINHOLE &&
            std::abs(dst_y - src_y) > 1.0f) {
          continue;
        }
        const float integer_src_x = static_cast<float>(motion_vectors[j + 3]);
        const float integer_src_y = static_cast<float>(motion_vectors[j + 4]);
        if (std::abs(encoded_src_x - integer_src_x) > 1.0f || std::abs(encoded_src_y - integer_src_y) > 1.0f) {
          std::free(motion_vectors);
          std::cerr << "Inconsistent subpixel stereo MV coordinates in pair " << stereo_mv_pair_counter << "."
                    << std::endl;
          std::abort();
        }

        const float half_width = 0.5f * static_cast<float>(block_width);
        const float half_height = 0.5f * static_cast<float>(block_height);
        const auto block_inside_image = [&](float x, float y) {
          return x - half_width >= 0 && y - half_height >= 0 && x + half_width <= width &&
                 y + half_height <= height;
        };
        if (!block_inside_image(src_x, src_y) || !block_inside_image(dst_x, dst_y)) {
          continue;
        }
        const float rectified_left_x = source_is_left ? src_x : dst_x;
        const float rectified_left_y = source_is_left ? src_y : dst_y;
        const float rectified_right_x = source_is_left ? dst_x : src_x;
        const float rectified_right_y = source_is_left ? dst_y : src_y;
        const bool left_support_valid =
            stereo_mv_rectification_mode != STEREO_MV_BASELINE_PINHOLE ||
            rectifiedSupportIsValid(
                stereo_rectified_left_invalid_integral, rectified_left_x,
                rectified_left_y, block_width, block_height, width, height);
        const bool right_support_valid = rectifiedSupportIsValid(
            stereo_rectified_right_invalid_integral, rectified_right_x,
            rectified_right_y, block_width, block_height, width, height);
        if (!left_support_valid || !right_support_valid) {
          stereo_rectification_mask_rejections++;
          continue;
        }
        output.push_back({source, block_width, block_height, src_x, src_y, dst_x, dst_y,
                          dst_x - src_x, dst_y - src_y, motion_scale});
      }
    }
    std::free(motion_vectors);
  }

  void append_left_right_consistent(const std::vector<MotionVector>& left_to_right,
                                    const std::vector<MotionVector>& right_to_left,
                                    std::vector<MotionVector>& output) {
    constexpr float kCycleThresholdPixels = 1.0f;
    constexpr int kBinSize = 8;
    if (left_to_right.empty() || right_to_left.empty()) return;

    int max_x = 0;
    int max_y = 0;
    for (const MotionVector& mv : right_to_left) {
      max_x = std::max(max_x, static_cast<int>(std::ceil(mv.src_x + 0.5f * mv.width)));
      max_y = std::max(max_y, static_cast<int>(std::ceil(mv.src_y + 0.5f * mv.height)));
    }
    const int bin_cols = std::max(1, (max_x + kBinSize - 1) / kBinSize);
    const int bin_rows = std::max(1, (max_y + kBinSize - 1) / kBinSize);
    std::vector<std::vector<size_t>> bins(static_cast<size_t>(bin_cols * bin_rows));
    for (size_t index = 0; index < right_to_left.size(); ++index) {
      const MotionVector& mv = right_to_left.at(index);
      const int min_bin_x = std::clamp(static_cast<int>(std::floor(mv.src_x - 0.5f * mv.width)) / kBinSize, 0,
                                       bin_cols - 1);
      const int max_bin_x = std::clamp(static_cast<int>(std::floor(mv.src_x + 0.5f * mv.width)) / kBinSize, 0,
                                       bin_cols - 1);
      const int min_bin_y = std::clamp(static_cast<int>(std::floor(mv.src_y - 0.5f * mv.height)) / kBinSize, 0,
                                       bin_rows - 1);
      const int max_bin_y = std::clamp(static_cast<int>(std::floor(mv.src_y + 0.5f * mv.height)) / kBinSize, 0,
                                       bin_rows - 1);
      for (int bin_y = min_bin_y; bin_y <= max_bin_y; ++bin_y)
        for (int bin_x = min_bin_x; bin_x <= max_bin_x; ++bin_x)
          bins.at(static_cast<size_t>(bin_y * bin_cols + bin_x)).push_back(index);
    }

    struct Candidate {
      size_t left_index;
      size_t right_index;
      float cost;
    };
    std::vector<Candidate> candidates;
    for (size_t left_index = 0; left_index < left_to_right.size(); ++left_index) {
      const MotionVector& forward = left_to_right.at(left_index);
      const int center_bin_x = std::clamp(static_cast<int>(forward.dst_x) / kBinSize, 0, bin_cols - 1);
      const int center_bin_y = std::clamp(static_cast<int>(forward.dst_y) / kBinSize, 0, bin_rows - 1);
      {
        const int bin_y = center_bin_y;
        const int bin_x = center_bin_x;
        for (size_t right_index : bins.at(static_cast<size_t>(bin_y * bin_cols + bin_x))) {
            const MotionVector& reverse = right_to_left.at(right_index);
            if (std::abs(forward.dst_x - reverse.src_x) > 0.5f * reverse.width ||
                std::abs(forward.dst_y - reverse.src_y) > 0.5f * reverse.height) {
              continue;
            }
            const float cycle_x = (forward.dst_x - forward.src_x) + (reverse.dst_x - reverse.src_x);
            const float cycle_y = (forward.dst_y - forward.src_y) + (reverse.dst_y - reverse.src_y);
            if (std::abs(cycle_x) > kCycleThresholdPixels || std::abs(cycle_y) > kCycleThresholdPixels) continue;
            const float center_dx = forward.dst_x - reverse.src_x;
            const float center_dy = forward.dst_y - reverse.src_y;
            candidates.push_back({left_index, right_index,
                                  cycle_x * cycle_x + cycle_y * cycle_y +
                                      1e-3f * (center_dx * center_dx + center_dy * center_dy)});
        }
      }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
      if (lhs.cost != rhs.cost) return lhs.cost < rhs.cost;
      if (lhs.left_index != rhs.left_index) return lhs.left_index < rhs.left_index;
      return lhs.right_index < rhs.right_index;
    });
    std::vector<bool> left_used(left_to_right.size(), false);
    std::vector<bool> right_used(right_to_left.size(), false);
    for (const Candidate& candidate : candidates) {
      if (left_used.at(candidate.left_index) || right_used.at(candidate.right_index)) continue;

      const MotionVector& forward = left_to_right.at(candidate.left_index);
      const MotionVector& reverse = right_to_left.at(candidate.right_index);
      const float flow_x = forward.dst_x - forward.src_x;
      const float flow_y = forward.dst_y - forward.src_y;
      const float min_x = std::max(forward.src_x - 0.5f * forward.width,
                                   reverse.src_x - 0.5f * reverse.width - flow_x);
      const float max_x = std::min(forward.src_x + 0.5f * forward.width,
                                   reverse.src_x + 0.5f * reverse.width - flow_x);
      const float min_y = std::max(forward.src_y - 0.5f * forward.height,
                                   reverse.src_y - 0.5f * reverse.height - flow_y);
      const float max_y = std::min(forward.src_y + 0.5f * forward.height,
                                   reverse.src_y + 0.5f * reverse.height - flow_y);
      const int validated_width = static_cast<int>(std::floor(max_x - min_x));
      const int validated_height = static_cast<int>(std::floor(max_y - min_y));
      if (validated_width < 1 || validated_height < 1) continue;

      MotionVector validated = forward;
      validated.width = validated_width;
      validated.height = validated_height;
      validated.src_x = 0.5f * (min_x + max_x);
      validated.src_y = 0.5f * (min_y + max_y);
      validated.dst_x = validated.src_x + flow_x;
      validated.dst_y = validated.src_y + flow_y;
      left_used.at(candidate.left_index) = true;
      right_used.at(candidate.right_index) = true;
      output.push_back(validated);
    }
  }

  bool decode_stereo_motion_vector_frame(
      size_t image_index, std::vector<MotionVector>& output,
      uint32_t& mask_rejections) {
    if (stereo_mv_caps.empty()) return false;
    stereo_mv_pair_counter = image_index;
    const int64_t t_ns = image_timestamps.at(image_index);
    const size_t rejections_before = stereo_rectification_mask_rejections;
    output.clear();
    for (StereoMvCapture& capture : stereo_mv_caps) {
      std::vector<MotionVector> left_to_right;
      std::vector<MotionVector> right_to_left;
      read_stereo_mv_frame(capture, t_ns, 'I', true, left_to_right);
      read_stereo_mv_frame(capture, t_ns, 'P', true, left_to_right);
      read_stereo_mv_frame(capture, t_ns, 'P', false, right_to_left);
      append_left_right_consistent(left_to_right, right_to_left, output);
    }
    mask_rejections = static_cast<uint32_t>(
        stereo_rectification_mask_rejections - rejections_before);
    stereo_mv_capture_position++;
    stereo_mv_pair_counter = image_index + 1;
    return true;
  }

  bool read_stereo_motion_vector_frame(
      size_t image_index, std::vector<MotionVector>& output,
      uint32_t& mask_rejections) {
    if (stereo_mv_cache_reader && stereo_mv_cache_reader->readFrame(
                                      static_cast<uint32_t>(image_index), output,
                                      mask_rejections)) {
      if (stereo_mv_cache_writer) {
        stereo_mv_cache_writer->writeFrame(
            static_cast<uint32_t>(image_index), output, mask_rejections);
      }
      return true;
    }
    while (stereo_mv_capture_position < image_index) {
      std::vector<MotionVector> discarded;
      uint32_t discarded_rejections = 0;
      if (!decode_stereo_motion_vector_frame(stereo_mv_capture_position,
                                             discarded,
                                             discarded_rejections)) {
        return false;
      }
    }
    if (!decode_stereo_motion_vector_frame(image_index, output,
                                           mask_rejections)) {
      return false;
    }
    if (stereo_mv_cache_writer) {
      stereo_mv_cache_writer->writeFrame(static_cast<uint32_t>(image_index),
                                         output, mask_rejections);
    }
    return true;
  }

 public:
  ~EurocVioDataset() override {
    for (cv::VideoCapture& cap : caps) {
      cap.release();
    }
    for (VideoCap& mv_cap : mv_caps) {
      mv_cap.release();
    }
    for (StereoMvCapture& stereo_mv_cap : stereo_mv_caps) stereo_mv_cap.cap.release();
    reverse_temporal_motion_vectors.clear();
    reverse_temporal_i_frames.clear();
  }

  size_t get_num_cams() const override { return num_cams; }

  std::vector<int64_t> &get_image_timestamps() override { return image_timestamps; }

  const Eigen::aligned_vector<AccelData> &get_accel_data() const override { return accel_data; }
  const Eigen::aligned_vector<GyroData> &get_gyro_data() const override { return gyro_data; }
  const std::vector<int64_t> &get_gt_timestamps() const override { return gt_timestamps; }
  const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data() const override { return gt_pose_data; }

  int64_t get_mocap_to_imu_offset_ns() const override { return mocap_to_imu_offset_ns; }

  std::vector<ImageData> get_image_data(int64_t t_ns) override {
    std::vector<ImageData> res(num_cams);
    const auto image_timestamp_it = std::lower_bound(image_timestamps.begin(), image_timestamps.end(), t_ns);
    if (image_timestamp_it == image_timestamps.end() || *image_timestamp_it != t_ns) {
      std::cerr << "Requested unknown image timestamp " << t_ns << "." << std::endl;
      std::abort();
    }
    const size_t image_index = static_cast<size_t>(std::distance(image_timestamps.begin(), image_timestamp_it));
    if (previous_motion_vectors.size() != num_cams) previous_motion_vectors.resize(num_cams);
    // The dataset is first read (and caps are opened for every video file found)
    // and after that the use_mvs flag is set on the dataset object making this necessary
    if (!use_mvs && (!mv_caps.empty() || !stereo_mv_caps.empty())) {
      //			std::cout << "NOT using motion vectors for tracking guesses!" << std::endl;
      for (VideoCap& mv_cap : mv_caps) {
        mv_cap.release();
      }
      mv_caps.clear();
      temporal_mv_caps_initialized = false;
      temporal_mv_source_available.clear();
      temporal_mv_capture_positions.clear();
      temporal_mv_cache_readers.clear();
      temporal_mv_cache_writers.clear();
      reverse_temporal_motion_vectors.clear();
      reverse_temporal_i_frames.clear();
      reverse_temporal_mv_cache_readers.clear();
      reverse_temporal_motion_vectors_initialized = false;
      for (StereoMvCapture& stereo_mv_cap : stereo_mv_caps) stereo_mv_cap.cap.release();
      stereo_mv_caps.clear();
      stereo_mv_cache_reader.reset();
      stereo_mv_cache_writer.reset();
      stereo_mv_capture_position = 0;
      stereo_mv_caps_initialized = false;
      stereo_mv_rectification_mode = STEREO_MV_NATIVE;
      stereo_rectified_left_invalid_integral.release();
      stereo_rectified_right_invalid_integral.release();
      stereo_rectification_mask_rejections = 0;
      stereo_mv_pair_counter = 0;
    }

    const std::string video_root = video_dataset_path.empty() ? path : video_dataset_path;
    if (use_mvs && temporal_reverse_policy != 0) {
      initialize_reverse_temporal_motion_vectors(video_root);
    }

    // Open RGB video captures lazily so an optional video_dataset_path can be
    // set after read() and still be respected.
    if (load_image_data && caps.empty()) {
      for (size_t cam_idx = 0; cam_idx < num_cams; cam_idx++) {
        std::string video_path = video_root + "/mav0/cam" + std::to_string(cam_idx) + "/data.mp4";
        if (fs::exists(video_path)) {
          cv::VideoCapture cap{video_path};
          caps.push_back(cap);
          if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera at " << video_path << std::endl;
            std::abort();
          }
        }
      }
    }


    if (use_mvs &&
        !temporal_mv_caps_initialized) {
      temporal_mv_caps_initialized = true;
      mv_caps.resize(num_cams);
      temporal_mv_source_available.assign(num_cams, false);
      temporal_mv_capture_positions.assign(num_cams, 0);
      temporal_mv_cache_readers.resize(num_cams);
      temporal_mv_cache_writers.resize(num_cams);
      for (size_t cam_idx = 0; cam_idx < num_cams; cam_idx++) {
        const std::string camera_video_dir = video_root + "/mav0/cam" + std::to_string(cam_idx);
        const std::string video_path = camera_video_dir + "/data.mp4";
        if (!fs::exists(video_path)) continue;

        std::vector<std::string> source_paths{video_path};
        const uint64_t fingerprint = motionVectorCacheFingerprint(
            source_paths, UINT64_C(0x74656d706f72616c));
        const std::string cache_path =
            camera_video_dir + "/.basalt-mv-cache/temporal.bin";
        temporal_mv_cache_readers.at(cam_idx) =
            std::make_unique<MotionVectorCacheReader>();
        MotionVectorCacheReader& reader =
            *temporal_mv_cache_readers.at(cam_idx);
        const bool cache_valid = reader.open(
            cache_path, fingerprint,
            static_cast<uint32_t>(image_timestamps.size()));
        if (cache_valid && reader.complete()) {
          std::cout << "Loaded cached forward VOCA temporal MVs for cam"
                    << cam_idx << "." << std::endl;
          continue;
        }

        if (mv_caps.at(cam_idx).open(video_path.c_str()) < 0) {
          std::cerr << "ERROR! Unable to open camera for Motionvector extraction at "
                    << video_path << std::endl;
          std::abort();
        }
        temporal_mv_source_available.at(cam_idx) = true;
        temporal_mv_cache_writers.at(cam_idx) =
            std::make_unique<MotionVectorCacheWriter>();
        if (!temporal_mv_cache_writers.at(cam_idx)->open(
                cache_path, fingerprint,
                static_cast<uint32_t>(image_timestamps.size()),
                reader.cachedFrames())) {
          temporal_mv_cache_writers.at(cam_idx).reset();
        }

      }
    }

    if (use_mvs && !stereo_mv_caps_initialized) {
      stereo_mv_caps_initialized = true;
      const std::string stereo_root = video_root + "/mav0/stereo";
      const std::string baseline_rectification_dir =
          stereo_root + "/baseline-pinhole-v1";
      const std::string baseline_rectification_manifest =
          baseline_rectification_dir + "/stereo-baseline-rectification.json";
      const bool has_baseline_rectification_manifest =
          fs::exists(baseline_rectification_manifest);
      const std::string stereo_dir = has_baseline_rectification_manifest
                                         ? baseline_rectification_dir
                                         : stereo_root;
      stereo_mvs_right_rectified_to_left =
          fs::exists(stereo_root + "/right-rectified-to-left");
      if (stereo_mvs_right_rectified_to_left &&
          has_baseline_rectification_manifest) {
        std::cerr << "Stereo artifacts contain both legacy and baseline-aligned "
                     "rectification metadata in "
                  << stereo_dir << "." << std::endl;
        std::abort();
      }
      stereo_mv_rectification_mode =
          has_baseline_rectification_manifest
              ? STEREO_MV_BASELINE_PINHOLE
              : (stereo_mvs_right_rectified_to_left
                     ? STEREO_MV_RIGHT_TO_LEFT
                     : STEREO_MV_NATIVE);
      stereo_rectified_left_invalid_integral.release();
      stereo_rectified_right_invalid_integral.release();
      stereo_rectification_mask_rejections = 0;
      if (has_baseline_rectification_manifest)
        loadBaselineRectificationManifest(baseline_rectification_manifest);
      const auto load_rectification_mask = [&](const std::string& validity_mask_path,
                                               cv::Mat& invalid_integral) {
        if (fs::exists(validity_mask_path)) {
          const cv::Mat validity_mask =
              cv::imread(validity_mask_path, cv::IMREAD_GRAYSCALE);
          if (validity_mask.empty() || validity_mask.type() != CV_8UC1) {
            std::cerr << "Unable to read stereo rectification validity mask at "
                      << validity_mask_path << "." << std::endl;
            std::abort();
          }
          cv::Mat invalid_mask;
          cv::compare(validity_mask, 0, invalid_mask, cv::CMP_EQ);
          invalid_mask /= 255;
          cv::integral(invalid_mask, invalid_integral, CV_32S);
          std::cout << "Loaded stereo rectification validity mask from "
                    << validity_mask_path << "." << std::endl;
        } else {
          std::cerr << "WARNING: rectified stereo videos have no validity mask at "
                    << validity_mask_path << "; invalid warp regions cannot be rejected."
                    << std::endl;
        }
      };
      if (stereo_mv_rectification_mode == STEREO_MV_BASELINE_PINHOLE) {
        load_rectification_mask(
            stereo_dir + "/left-rectified-valid-mask.png",
            stereo_rectified_left_invalid_integral);
        load_rectification_mask(
            stereo_dir + "/right-rectified-valid-mask.png",
            stereo_rectified_right_invalid_integral);
        if (stereo_rectified_left_invalid_integral.empty() ||
            stereo_rectified_right_invalid_integral.empty()) {
          std::cerr << "Baseline-aligned stereo artifacts require both rectification validity masks."
                    << std::endl;
          std::abort();
        }
        const int expected_integral_columns =
            stereo_rectified_resolution[0] + 1;
        const int expected_integral_rows =
            stereo_rectified_resolution[1] + 1;
        if (stereo_rectified_left_invalid_integral.cols !=
                expected_integral_columns ||
            stereo_rectified_left_invalid_integral.rows !=
                expected_integral_rows ||
            stereo_rectified_right_invalid_integral.cols !=
                expected_integral_columns ||
            stereo_rectified_right_invalid_integral.rows !=
                expected_integral_rows) {
          std::cerr << "Stereo rectification mask dimensions do not match the manifest."
                    << std::endl;
          std::abort();
        }
      } else if (stereo_mv_rectification_mode == STEREO_MV_RIGHT_TO_LEFT) {
        load_rectification_mask(
            stereo_dir + "/right-rectified-valid-mask.png",
            stereo_rectified_right_invalid_integral);
      }
      if (fs::exists(stereo_dir)) {
        struct StereoPath {
          std::string path;
          int disparity_shift;
          int grid_shift_x;
          int grid_shift_y;
        };
        std::vector<StereoPath> paths;
        for (const auto& entry : fs::directory_iterator(stereo_dir)) {
          const std::string filename = entry.path().filename().string();
          int disparity_shift = 0;
          int shift_x = 0;
          int shift_y = 0;
          char trailing = '\0';
          if (std::sscanf(filename.c_str(), "data_ipp_d%d_g%d_%d.mp4%c", &disparity_shift, &shift_x, &shift_y,
                          &trailing) == 3 &&
              disparity_shift >= 0 && shift_x >= 0 && shift_y >= 0) {
            paths.push_back({entry.path().string(), disparity_shift, shift_x, shift_y});
          }
        }
        std::sort(paths.begin(), paths.end(), [](const StereoPath& lhs, const StereoPath& rhs) {
          return std::tie(lhs.disparity_shift, lhs.grid_shift_x, lhs.grid_shift_y) <
                 std::tie(rhs.disparity_shift, rhs.grid_shift_x, rhs.grid_shift_y);
        });
        if (!paths.empty()) {
          std::set<int> disparity_shifts;
          int grid_step = 0;
          for (const StereoPath& stereo_path : paths) {
            disparity_shifts.insert(stereo_path.disparity_shift);
            grid_step = std::max({grid_step, stereo_path.grid_shift_x, stereo_path.grid_shift_y});
          }
          const std::set<std::pair<int, int>> expected_phases{
              {0, 0}, {0, grid_step}, {grid_step, 0}, {grid_step, grid_step}};
          for (int disparity_shift : disparity_shifts) {
            std::set<std::pair<int, int>> actual_phases;
            for (const StereoPath& stereo_path : paths)
              if (stereo_path.disparity_shift == disparity_shift)
                actual_phases.emplace(stereo_path.grid_shift_x, stereo_path.grid_shift_y);
            if (grid_step <= 0 || actual_phases != expected_phases) {
              std::cerr << "Incomplete stereo grid phases for disparity shift " << disparity_shift << " in "
                        << stereo_dir << "." << std::endl;
              std::abort();
            }
          }

          std::vector<std::string> stereo_source_paths;
          stereo_source_paths.reserve(paths.size() + 3);
          for (const StereoPath& stereo_path : paths)
            stereo_source_paths.emplace_back(stereo_path.path);
          if (has_baseline_rectification_manifest) {
            stereo_source_paths.emplace_back(baseline_rectification_manifest);
            stereo_source_paths.emplace_back(stereo_dir +
                                             "/left-rectified-valid-mask.png");
            stereo_source_paths.emplace_back(stereo_dir +
                                             "/right-rectified-valid-mask.png");
          } else if (stereo_mvs_right_rectified_to_left) {
            stereo_source_paths.emplace_back(stereo_dir +
                                             "/right-rectified-valid-mask.png");
          }
          const uint64_t stereo_fingerprint = motionVectorCacheFingerprint(
              stereo_source_paths,
              UINT64_C(0x73746572656f7631) ^
                  static_cast<uint64_t>(stereo_mv_rectification_mode));
          const std::string stereo_cache_path =
              stereo_dir + "/.basalt-mv-cache/stereo.bin";
          stereo_mv_cache_reader =
              std::make_unique<MotionVectorCacheReader>();
          const bool stereo_cache_complete =
              stereo_mv_cache_reader->open(
                  stereo_cache_path, stereo_fingerprint,
                  static_cast<uint32_t>(image_timestamps.size())) &&
              stereo_mv_cache_reader->complete();
          if (!stereo_cache_complete) {
            stereo_mv_cache_writer =
                std::make_unique<MotionVectorCacheWriter>();
            if (!stereo_mv_cache_writer->open(
                    stereo_cache_path, stereo_fingerprint,
                    static_cast<uint32_t>(image_timestamps.size()),
                    stereo_mv_cache_reader->cachedFrames())) {
              stereo_mv_cache_writer.reset();
            }
          }

          if (!stereo_cache_complete) {
            stereo_mv_caps.reserve(paths.size());
            for (const StereoPath& stereo_path : paths) {
              stereo_mv_caps.emplace_back();
              StereoMvCapture& capture = stereo_mv_caps.back();
              capture.disparity_shift = stereo_path.disparity_shift;
              capture.grid_shift_x = stereo_path.grid_shift_x;
              capture.grid_shift_y = stereo_path.grid_shift_y;
              if (capture.cap.open(stereo_path.path.c_str()) < 0) {
                std::cerr << "Unable to open stereo MV video at "
                          << stereo_path.path << "." << std::endl;
                std::abort();
              }
            }
          }
          std::cout << (stereo_cache_complete ? "Loaded cached" : "Opened")
                    << (stereo_cache_complete ? "" : " " + std::to_string(stereo_mv_caps.size()))
                    << " strict I-P-P stereo MVs from " << stereo_dir
                    << (stereo_mv_rectification_mode == STEREO_MV_BASELINE_PINHOLE
                            ? " with cam0/cam1 baseline-aligned pinhole rectification."
                            : (stereo_mvs_right_rectified_to_left
                                   ? " with cam1 rectified into cam0 coordinates."
                                   : "."))
                    << std::endl;
        }
      }
    }

    if ((stereo_mv_cache_reader && stereo_mv_cache_reader->valid()) ||
        !stereo_mv_caps.empty()) {
      if (num_cams < 2) {
        std::cerr << "Stereo MV videos require at least two cameras." << std::endl;
        std::abort();
      }
      res.at(0).stereo_motion_vectors_available = true;
      res.at(0).stereo_motion_vectors_right_rectified_to_left =
          stereo_mvs_right_rectified_to_left;
      res.at(0).stereo_motion_vector_rectification_mode =
          stereo_mv_rectification_mode;
      res.at(0).stereo_rectified_pinhole = stereo_rectified_pinhole;
      res.at(0).stereo_rectified_resolution = stereo_rectified_resolution;
      res.at(0).stereo_rotation_cam0_rectified =
          stereo_rotation_cam0_rectified;
      res.at(0).stereo_rotation_cam1_rectified =
          stereo_rotation_cam1_rectified;
      res.at(0).stereo_rectification_left_validity_mask_available =
          !stereo_rectified_left_invalid_integral.empty();
      res.at(0).stereo_rectification_right_validity_mask_available =
          !stereo_rectified_right_invalid_integral.empty();
      res.at(0).stereo_rectification_validity_mask_available =
          !stereo_rectified_left_invalid_integral.empty() ||
          !stereo_rectified_right_invalid_integral.empty();
      uint32_t mask_rejections = 0;
      if (!read_stereo_motion_vector_frame(
              image_index, res.at(0).stereo_motion_vectors,
              mask_rejections)) {
        std::cerr << "Unable to read stereo MV frame " << image_index << "."
                  << std::endl;
        std::abort();
      }
      res.at(0).stereo_motion_vectors_rejected_by_rectification_mask =
          mask_rejections;
    }

    for (size_t i = 0; i < num_cams; i++) {
      //			std::cout << "getting image data of cam " << i << std::endl;
      std::string full_path = path + "/mav0/cam" + std::to_string(i);
      std::string full_video_path = video_root + "/mav0/cam" + std::to_string(i) + "/data.mp4";
      std::string full_image_path = full_path + "/data/" + image_path[t_ns];
      cv::Mat img;
      bool temporal_forward_is_i_frame = false;

      if (use_mvs && !fs::exists(full_video_path) && mv_warning_shown_cams.count(i) == 0) {
        std::cout << "WARNING: --use-mvs is enabled but no video file found at "
                  << full_video_path << ". Motion vectors will not be available for cam" << i
                  << ". To use motion vectors, encode your image sequences into data.mp4 "
                  << "with the provided ffmpeg command." << std::endl;
        mv_warning_shown_cams.insert(i);
      }

      //&& image_timestamps[frameNumber] == t_ns
      if (use_mvs &&
          temporal_mv_caps_initialized && fs::exists(full_video_path)) {
        bool base_is_i_frame = false;
        const bool read_ok = read_temporal_motion_vector_frame(
            image_index, i, res.at(i).motion_vectors, base_is_i_frame);
        temporal_forward_is_i_frame = read_ok && base_is_i_frame;
        if (read_ok && base_is_i_frame) {
          if (temporal_reverse_policy == 0) {
            res.at(i).motion_vectors_reused_from_i_frame = true;
            res.at(i).motion_vectors = previous_motion_vectors.at(i);
          } else {
            res.at(i).motion_vectors.clear();
          }
        } else {
          previous_motion_vectors.at(i) = res.at(i).motion_vectors;
          if (read_ok && !res.at(i).motion_vectors.empty()) mv_frame_counter++;
        }
      }

      if (use_mvs && temporal_reverse_policy != 0) {
        std::vector<MotionVector> cached_reverse;
        const std::vector<MotionVector>* reverse = nullptr;
        bool reverse_is_i_frame = false;
        uint32_t reverse_flags = 0;
        if (i < reverse_temporal_mv_cache_readers.size() &&
            reverse_temporal_mv_cache_readers.at(i) &&
            reverse_temporal_mv_cache_readers.at(i)->readFrame(
                static_cast<uint32_t>(image_index), cached_reverse,
                reverse_flags)) {
          reverse = &cached_reverse;
          reverse_is_i_frame = (reverse_flags & 1U) != 0;
        } else if (i < reverse_temporal_motion_vectors.size() &&
                   i < reverse_temporal_i_frames.size() &&
                   image_index < reverse_temporal_motion_vectors.at(i).size() &&
                   image_index < reverse_temporal_i_frames.at(i).size()) {
          reverse = &reverse_temporal_motion_vectors.at(i).at(image_index);
          reverse_is_i_frame = reverse_temporal_i_frames.at(i).at(image_index);
        }
        if (temporal_reverse_policy < 1 || temporal_reverse_policy > 3 ||
            !reverse) {
          std::cerr << "Invalid reverse temporal MV state for cam" << i
                    << " at image " << image_index << "." << std::endl;
          std::abort();
        }
        ImageData& image = res.at(i);
        image.temporal_reverse_policy = temporal_reverse_policy;
        image.temporal_forward_is_i_frame = temporal_forward_is_i_frame;
        image.temporal_reverse_is_i_frame = reverse_is_i_frame;
        image.temporal_forward_motion_vector_count =
            image.motion_vectors.size();
        image.temporal_reverse_motion_vector_count = reverse->size();
        if (temporal_reverse_policy == 1 ||
            temporal_reverse_policy == 3) {
          std::vector<MotionVector> reciprocal;
          if (temporal_forward_is_i_frame != reverse_is_i_frame) {
            if (temporal_forward_is_i_frame) {
              reciprocal.reserve(reverse->size());
              for (const MotionVector& reverse_mv : *reverse) {
                reciprocal.emplace_back(
                    invert_reverse_temporal_motion_vector(reverse_mv));
              }
            } else {
              reciprocal = image.motion_vectors;
            }
            image.temporal_single_direction_fallback = true;
            image.temporal_single_direction_fallback_motion_vector_count =
                reciprocal.size();
          } else if (!temporal_forward_is_i_frame) {
            appendReciprocalTemporalMotionVectors(image.motion_vectors,
                                                  *reverse, reciprocal);
            image.temporal_reciprocal_motion_vector_count = reciprocal.size();
          }
          if (temporal_reverse_policy == 1) {
            image.motion_vectors = std::move(reciprocal);
          } else {
            image.temporal_reciprocal_motion_vectors =
                std::move(reciprocal);
            append_bidirectional_temporal_candidates(image.motion_vectors,
                                                     *reverse,
                                                     image.motion_vectors);
          }
        } else {
          append_bidirectional_temporal_candidates(image.motion_vectors,
                                                   *reverse,
                                                   image.motion_vectors);
        }
        previous_motion_vectors.at(i) = image.motion_vectors;
      }

      if (!load_image_data) continue;

      if (!use_video_frames) {
        if (!fs::exists(full_image_path)) {
          std::cerr << "Image file not found while --use-video-frames is disabled: " << full_image_path << std::endl;
          std::abort();
        }
        img = cv::imread(full_image_path, cv::IMREAD_UNCHANGED);
      } else {
        if (!(caps.size() > 0 && fs::exists(full_video_path))) {
          std::cerr << "Video frame requested but no video file found at " << full_video_path
                    << ". Provide --video-dataset-path and ensure data.mp4 exists." << std::endl;
          std::abort();
        }
        if (frameNumber >= image_timestamps.size() || image_timestamps[frameNumber] != t_ns) {
          auto lower = std::lower_bound(image_timestamps.begin(), image_timestamps.end(), t_ns);
          frameNumber = std::distance(image_timestamps.begin(), lower);
        }

        if (caps[i].get(cv::CAP_PROP_POS_FRAMES) != frameNumber) {
          caps[i].set(cv::CAP_PROP_POS_FRAMES, frameNumber);
        }

        if (!caps[i].read(img)) {
          std::cerr << "Failed to read frame (num: " << frameNumber << ") at timestamp " << t_ns << "." << std::endl;
          std::abort();
        }
        frameNumber++;
      }

      if (img.empty()) {
        std::cout << "Failed to load image!" << std::endl;
        return {};
      } else if (img.type() == CV_8UC1) {
        res[i].img.reset(new ManagedImage<uint16_t>(img.cols, img.rows));

        const uint8_t *data_in = img.ptr();
        uint16_t *data_out = res[i].img->ptr;

        size_t full_size = img.cols * img.rows;
        for (size_t i = 0; i < full_size; i++) {
          int val = data_in[i];
          val = val << 8;
          data_out[i] = val;
        }
      } else if (img.type() == CV_8UC3) {
        res[i].img.reset(new ManagedImage<uint16_t>(img.cols, img.rows));

        const uint8_t *data_in = img.ptr();
        uint16_t *data_out = res[i].img->ptr;

        size_t full_size = img.cols * img.rows;
        for (size_t i = 0; i < full_size; i++) {
          int val = data_in[i * 3];
          val = val << 8;
          data_out[i] = val;
        }
      } else if (img.type() == CV_16UC1) {
        res[i].img.reset(new ManagedImage<uint16_t>(img.cols, img.rows));
        std::memcpy(res[i].img->ptr, img.ptr(), img.cols * img.rows * sizeof(uint16_t));

      } else {
        std::cerr << "img.fmt.bpp " << img.type() << std::endl;
        std::abort();
      }

      auto exp_it = exposure_times[i].find(t_ns);
      if (exp_it != exposure_times[i].end()) {
        res[i].exposure = exp_it->second;
      }
    }

    if (use_mvs && (!mv_caps.empty() || !stereo_mv_caps.empty()) &&
        image_index + 1 >= image_timestamps.size()) {
      std::cout << "clearing motionvector captures after reading mvs of " << mv_frame_counter / num_cams
                << " frames from a total " << image_index + 1 << " of frames!" << std::endl;
      for (VideoCap& mv_cap : mv_caps) mv_cap.release();
      mv_caps.clear();
      temporal_mv_caps_initialized = false;
      temporal_mv_source_available.clear();
      temporal_mv_capture_positions.clear();
      temporal_mv_cache_readers.clear();
      temporal_mv_cache_writers.clear();
      reverse_temporal_motion_vectors.clear();
      reverse_temporal_i_frames.clear();
      reverse_temporal_mv_cache_readers.clear();
      reverse_temporal_motion_vectors_initialized = false;
      for (StereoMvCapture& stereo_mv_cap : stereo_mv_caps) stereo_mv_cap.cap.release();
      stereo_mv_caps.clear();
      stereo_mv_cache_reader.reset();
      stereo_mv_cache_writer.reset();
      stereo_mv_capture_position = 0;
      stereo_mv_caps_initialized = false;
      stereo_mv_rectification_mode = STEREO_MV_NATIVE;
      stereo_rectified_left_invalid_integral.release();
      stereo_rectified_right_invalid_integral.release();
      stereo_rectification_mask_rejections = 0;
      stereo_mv_pair_counter = 0;
    }

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend class EurocIO;
};

class EurocIO : public DatasetIoInterface {
 public:
  EurocIO(bool load_mocap_as_gt) : load_mocap_as_gt(load_mocap_as_gt) {}

  void read(const std::string &path) override {
    if (!fs::exists(path)) std::cerr << "No dataset found in " << path << std::endl;

    data = std::make_shared<EurocVioDataset>();

    // Detect camera count
    int i = 0;
    while (fs::exists(path + "/mav0/cam" + std::to_string(i))) {
      i++;
    }

    data->num_cams = i;
    data->path = path;

    read_image_timestamps(path + "/mav0/cam0/");

    read_imu_data(path + "/mav0/imu0/");

    if (!load_mocap_as_gt && fs::exists(path + "/mav0/state_groundtruth_estimate0/data.csv")) {
      read_gt_data_state(path + "/mav0/state_groundtruth_estimate0/");
    } else if (!load_mocap_as_gt && fs::exists(path + "/mav0/gt/data.csv")) {
      read_gt_data_pose(path + "/mav0/gt/");
    } else if (fs::exists(path + "/mav0/mocap0/data.csv")) {
      read_gt_data_pose(path + "/mav0/mocap0/");
    }

    data->exposure_times.resize(data->num_cams);
    for (size_t i = 0; i < data->num_cams; i++) {
      std::string istr = std::to_string(i);
      std::string cam_dir = path;
      cam_dir.append("/mav0/cam").append(istr).append("/");
      if (fs::exists(cam_dir + "exposure.csv")) {
        std::cout << "Loading exposure times for cam" << istr << std::endl;
        read_exposure(cam_dir, data->exposure_times[i]);
      }
    }
  }

  void reset() override { data.reset(); }

  VioDatasetPtr get_data() override { return data; }

 private:
  void read_exposure(const std::string &path, std::unordered_map<int64_t, double> &exposure_data) {
    exposure_data.clear();

    std::ifstream f(path + "exposure.csv");
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      char tmp;
      int64_t timestamp, exposure_int;
      Eigen::Vector3d gyro, accel;

      ss >> timestamp >> tmp >> exposure_int;

      exposure_data[timestamp] = exposure_int * 1e-9;
    }
  }

  void read_image_timestamps(const std::string &path) {
    std::ifstream f(path + "data.csv");
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;
      std::stringstream ss(line);
      char tmp;
      int64_t t_ns;
      std::string path;
      ss >> t_ns >> tmp >> path;

      data->image_timestamps.emplace_back(t_ns);
      data->image_path[t_ns] = path;
    }
  }

  void read_imu_data(const std::string &path) {
    data->accel_data.clear();
    data->gyro_data.clear();

    std::ifstream f(path + "data.csv");
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      char tmp;
      uint64_t timestamp;
      Eigen::Vector3d gyro, accel;

      ss >> timestamp >> tmp >> gyro[0] >> tmp >> gyro[1] >> tmp >> gyro[2] >> tmp >> accel[0] >> tmp >> accel[1] >>
          tmp >> accel[2];

      data->accel_data.emplace_back();
      data->accel_data.back().timestamp_ns = timestamp;
      data->accel_data.back().data = accel;

      data->gyro_data.emplace_back();
      data->gyro_data.back().timestamp_ns = timestamp;
      data->gyro_data.back().data = gyro;
    }
  }

  void read_gt_data_state(const std::string &path) {
    data->gt_timestamps.clear();
    data->gt_pose_data.clear();

    std::ifstream f(path + "data.csv");
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      char tmp;
      uint64_t timestamp;
      Eigen::Quaterniond q;
      Eigen::Vector3d pos, vel, accel_bias, gyro_bias;

      ss >> timestamp >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2] >> tmp >> q.w() >> tmp >> q.x() >> tmp >>
          q.y() >> tmp >> q.z() >> tmp >> vel[0] >> tmp >> vel[1] >> tmp >> vel[2] >> tmp >> accel_bias[0] >> tmp >>
          accel_bias[1] >> tmp >> accel_bias[2] >> tmp >> gyro_bias[0] >> tmp >> gyro_bias[1] >> tmp >> gyro_bias[2];

      data->gt_timestamps.emplace_back(timestamp);
      data->gt_pose_data.emplace_back(q, pos);
    }
  }

  void read_gt_data_pose(const std::string &path) {
    data->gt_timestamps.clear();
    data->gt_pose_data.clear();

    std::ifstream f(path + "data.csv");
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      char tmp;
      uint64_t timestamp;
      Eigen::Quaterniond q;
      Eigen::Vector3d pos;

      ss >> timestamp >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2] >> tmp >> q.w() >> tmp >> q.x() >> tmp >>
          q.y() >> tmp >> q.z();

      data->gt_timestamps.emplace_back(timestamp);
      data->gt_pose_data.emplace_back(q, pos);
    }
  }

  std::shared_ptr<EurocVioDataset> data;
  bool load_mocap_as_gt;
};  // namespace basalt

}  // namespace basalt

#endif  // DATASET_IO_H
