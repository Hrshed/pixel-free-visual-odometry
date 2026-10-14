#pragma once

#include <basalt/io/dataset_io.h>
#include <basalt/utils/filesystem.h>

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace basalt {

class MotionVectorCacheReader {
 public:
  bool open(const std::string& path, uint64_t fingerprint,
            uint32_t total_frames) {
    close();
    stream_.open(path, std::ios::binary);
    if (!stream_) return false;

    stream_.read(reinterpret_cast<char*>(&header_), sizeof(header_));
    if (!stream_ || header_.magic != kMagic || header_.version != kVersion ||
        header_.fingerprint != fingerprint ||
        header_.total_frames != total_frames ||
        header_.cached_frames > header_.total_frames) {
      close();
      return false;
    }
    frames_.resize(header_.total_frames);
    stream_.read(reinterpret_cast<char*>(frames_.data()),
                 static_cast<std::streamsize>(frames_.size() * sizeof(Frame)));
    if (!stream_) {
      close();
      return false;
    }
    return true;
  }

  void close() {
    if (stream_.is_open()) stream_.close();
    stream_.clear();
    frames_.clear();
    header_ = Header{};
  }

  bool valid() const { return stream_.is_open(); }
  uint32_t cachedFrames() const {
    return valid() ? header_.cached_frames : 0;
  }
  bool complete() const {
    return valid() && header_.cached_frames == header_.total_frames;
  }

  bool readFrame(uint32_t frame_index, std::vector<MotionVector>& output,
                 uint32_t& flags) {
    if (!valid() || frame_index >= header_.cached_frames) return false;
    const Frame& frame = frames_.at(frame_index);
    if (frame.offset > header_.record_count ||
        frame.count > header_.record_count - frame.offset) {
      close();
      return false;
    }
    const uint64_t byte_offset =
        sizeof(Header) + uint64_t(header_.total_frames) * sizeof(Frame) +
        frame.offset * sizeof(PackedMotionVector);
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
    std::vector<PackedMotionVector> packed(frame.count);
    if (!packed.empty()) {
      stream_.read(reinterpret_cast<char*>(packed.data()),
                   static_cast<std::streamsize>(packed.size() *
                                                sizeof(PackedMotionVector)));
    }
    if (!stream_) return false;

    output.clear();
    output.reserve(packed.size());
    for (const PackedMotionVector& mv : packed) {
      output.push_back({mv.source, mv.width, mv.height, mv.src_x, mv.src_y,
                        mv.dst_x, mv.dst_y, mv.quantized_flow_x,
                        mv.quantized_flow_y, mv.motion_scale});
    }
    flags = frame.flags;
    return true;
  }

 private:
  struct Header {
    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t total_frames = 0;
    uint32_t cached_frames = 0;
    uint32_t reserved = 0;
    uint64_t fingerprint = 0;
    uint64_t record_count = 0;
  };

  struct Frame {
    uint64_t offset = 0;
    uint32_t count = 0;
    uint32_t flags = 0;
  };

  struct PackedMotionVector {
    int16_t source = 0;
    int16_t width = 0;
    int16_t height = 0;
    int16_t motion_scale = 0;
    float src_x = 0;
    float src_y = 0;
    float dst_x = 0;
    float dst_y = 0;
    float quantized_flow_x = 0;
    float quantized_flow_y = 0;
  };

  static constexpr uint64_t kMagic = UINT64_C(0x314b50434d564d42);
  static constexpr uint32_t kVersion = 1;

  friend class MotionVectorCacheWriter;
  std::ifstream stream_;
  Header header_;
  std::vector<Frame> frames_;
};

class MotionVectorCacheWriter {
 public:
  ~MotionVectorCacheWriter() { finish(); }

  bool open(const std::string& path, uint64_t fingerprint,
            uint32_t total_frames,
            uint32_t minimum_cached_frames_to_replace = 0) {
    finish();
    path_ = path;
    temp_path_ = path + ".tmp";
    std::error_code error;
    fs::create_directories(fs::path(path).parent_path(), error);
    stream_.open(temp_path_, std::ios::binary | std::ios::trunc);
    if (!stream_) return false;

    header_.magic = MotionVectorCacheReader::kMagic;
    header_.version = MotionVectorCacheReader::kVersion;
    header_.total_frames = total_frames;
    header_.fingerprint = fingerprint;
    minimum_cached_frames_to_replace_ = minimum_cached_frames_to_replace;
    frames_.assign(total_frames, MotionVectorCacheReader::Frame{});
    const uint64_t data_offset =
        sizeof(header_) + uint64_t(total_frames) * sizeof(frames_.front());
    if (data_offset > 0) {
      stream_.seekp(static_cast<std::streamoff>(data_offset - 1),
                    std::ios::beg);
      stream_.put('\0');
    }
    return bool(stream_);
  }

  bool writeFrame(uint32_t frame_index,
                  const std::vector<MotionVector>& motion_vectors,
                  uint32_t flags) {
    if (!stream_ || frame_index != header_.cached_frames ||
        frame_index >= header_.total_frames) {
      return false;
    }
    MotionVectorCacheReader::Frame& frame = frames_.at(frame_index);
    frame.offset = header_.record_count;
    frame.count = static_cast<uint32_t>(motion_vectors.size());
    frame.flags = flags;
    for (const MotionVector& mv : motion_vectors) {
      if (mv.source < std::numeric_limits<int16_t>::min() ||
          mv.source > std::numeric_limits<int16_t>::max() || mv.width < 0 ||
          mv.width > std::numeric_limits<int16_t>::max() || mv.height < 0 ||
          mv.height > std::numeric_limits<int16_t>::max() ||
          mv.motion_scale < std::numeric_limits<int16_t>::min() ||
          mv.motion_scale > std::numeric_limits<int16_t>::max()) {
        return false;
      }
      const MotionVectorCacheReader::PackedMotionVector packed{
          static_cast<int16_t>(mv.source),
          static_cast<int16_t>(mv.width),
          static_cast<int16_t>(mv.height),
          static_cast<int16_t>(mv.motion_scale),
          mv.src_x,
          mv.src_y,
          mv.dst_x,
          mv.dst_y,
          mv.quantized_flow_x,
          mv.quantized_flow_y};
      stream_.write(reinterpret_cast<const char*>(&packed), sizeof(packed));
    }
    if (!stream_) return false;
    header_.record_count += motion_vectors.size();
    header_.cached_frames++;
    return true;
  }

  uint32_t cachedFrames() const { return header_.cached_frames; }
  bool valid() const { return stream_.is_open(); }

  void finish() {
    if (!stream_.is_open()) return;
    stream_.seekp(0, std::ios::beg);
    stream_.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
    stream_.write(reinterpret_cast<const char*>(frames_.data()),
                  static_cast<std::streamsize>(frames_.size() *
                                               sizeof(frames_.front())));
    stream_.flush();
    const bool write_ok = bool(stream_);
    stream_.close();
    if (write_ok && header_.cached_frames > 0 &&
        header_.cached_frames >= minimum_cached_frames_to_replace_) {
      std::error_code error;
      fs::rename(temp_path_, path_, error);
      if (error) {
        fs::remove(path_, error);
        error.clear();
        fs::rename(temp_path_, path_, error);
      }
    } else {
      std::error_code error;
      fs::remove(temp_path_, error);
    }
    frames_.clear();
    header_ = MotionVectorCacheReader::Header{};
    path_.clear();
    temp_path_.clear();
    minimum_cached_frames_to_replace_ = 0;
  }

 private:
  std::ofstream stream_;
  MotionVectorCacheReader::Header header_;
  std::vector<MotionVectorCacheReader::Frame> frames_;
  std::string path_;
  std::string temp_path_;
  uint32_t minimum_cached_frames_to_replace_ = 0;
};

inline uint64_t motionVectorCacheFingerprint(
    const std::vector<std::string>& paths, uint64_t seed) {
  uint64_t hash = UINT64_C(1469598103934665603) ^ seed;
  const auto append = [&hash](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= UINT64_C(1099511628211);
    }
  };
  for (const std::string& path : paths) {
    append(path.data(), path.size());
    std::error_code error;
    const uint64_t size = fs::is_regular_file(path, error)
                              ? static_cast<uint64_t>(fs::file_size(path, error))
                              : 0;
    append(&size, sizeof(size));
    error.clear();
    const auto write_time = fs::last_write_time(path, error);
    const auto ticks = error ? int64_t(0)
                             : static_cast<int64_t>(
                                   write_time.time_since_epoch().count());
    append(&ticks, sizeof(ticks));
  }
  return hash;
}

}  // namespace basalt
