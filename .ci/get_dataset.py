#!/usr/bin/env python3

"""
Gets a dataset.

This script assumes that the specified working directory lives on a fast
storage. If the specified dataset lives on a different storage, it will wait
until there is enough space on the working storage to uncompress the dataset. In
the end it returns the path of the final dataset to use.

It uses a lockfile to ensure that only one instance of the script can be
transferring files between disks to ensure best transfer speeds.

# Two examples: (notice .zip extension when not present)

# If /slow_disk/euroc/MH_01_easy.zip exists, and assuming CWD is /fast_disk/work
# Get /slow_disk/euroc/MH_01_easy.zip, uncompress it to
# /fast_disk/work/MH_01_easy, and output "/fast_disk/work/MH_01_easy"
get_dataset.py /slow_disk/euroc/MH_01_easy .

# If /fast_disk/euroc/MH_01_easy/ exists, and assuming CWD is /fast_disk/work
# Do not copy anything and output "/fast_disk/euroc/MH_01_easy"
get_dataset.py /fast_disk/euroc/MH_01_easy .

"""


import os
import sys
import json
import time
import math
import argparse
import shutil
import shlex

from subprocess import run, PIPE, Popen
from pathlib import Path

PADDING_KB = 50 * 2**20  # Padding in KB (50 GB)
WHILE_LOCK_SLEEP_SECONDS = 1  # Sleep time betweek lock checks
WHILE_SPACE_WAIT_SECONDS = 1  # Sleep time between space checks
MISMATCH_ALLOWLIST_DATASETS = ["V2_03_difficult", "V1_02_medium"]  # Dataset names allowed to use only shared cam timestamps


def shout(command, silence=False):
    "Executes command in shell and returns its stdout"
    if not silence:
        print(command)
    result = run(command, shell=True, check=False, stdout=PIPE)
    stdout = result.stdout.decode("utf-8")
    return stdout, result.returncode


def get_mount_point(path):
    """Get the source mount point of a path using findmnt."""
    path = os.path.abspath(path)
    stdout, _ = shout(f"findmnt -T {path} --json")
    data = json.loads(stdout)
    # On docker executor CI you can get e.g., "/dev/sda1[/hdd/monado-slam-datasets/M_monado_datasets/MG_reverb_g2]"
    # While on shell executor CI you get just "/dev/sda1", the following split() works for both
    return data["filesystems"][0]["source"].split("[")[0]


def get_file_size_kb(path):
    """Return the size of a file in kilobytes"""
    size_bytes = os.stat(path).st_size
    size_kb = math.ceil(size_bytes / 1024)
    return size_kb


def get_available_kb(path):
    """Return available space (in KB) on the given path."""
    usage = shutil.disk_usage(path)
    return usage.free // 1024


def wait_space_for_file(file_path, working_dir, padding_kb=PADDING_KB):
    "Wait until enough space on the working directory for the file to be uncompressed"
    required_kb = get_file_size_kb(file_path) + padding_kb
    prev_avail_gb = 0
    while True:
        available_kb = get_available_kb(working_dir)
        avail = available_kb / 2**20  # Convert to GB
        if available_kb < required_kb:
            perc = 100 * available_kb / required_kb
            req = required_kb / 2**20  # Convert to GB
            if avail - prev_avail_gb > 1:  # Print each GB of progress
                print(f"[{perc:.2f}%] Waiting for {req:.2f} GB, but {avail:.2f} GB avail")
                prev_avail_gb = avail
            time.sleep(WHILE_SPACE_WAIT_SECONDS)
        else:
            print(f"Sufficient space available ({avail:.2f} GB), continuing...")
            break

def compute_framerate(cam_dir):
    """Compute framerate from a EuRoC cam data.csv file."""
    data_csv = Path(cam_dir) / "data.csv"
    assert data_csv.exists(), \
        f"data.csv not found in {cam_dir} — invalid EuRoC camera directory"
    timestamps = get_timestamps_from_csv(data_csv)
    assert len(timestamps) >= 2, \
        f"data.csv in {cam_dir} has fewer than 2 entries — cannot compute framerate"
    # Timestamps are in nanoseconds
    avg_dt_ns = (int(timestamps[-1]) - int(timestamps[0])) / (len(timestamps) - 1)
    return round(1e9 / avg_dt_ns)

def get_euroc_cam_dirs(dataset_path):
    """Fail hard if the expected EuRoC mav0/cam* layout is missing."""
    mav0_path = Path(dataset_path) / "mav0"
    assert mav0_path.is_dir(), \
        f"Expected EuRoC 'mav0' directory not found under {dataset_path}"
    cam_dirs = sorted(mav0_path.glob("cam*"))
    assert cam_dirs, \
        f"No cam* directories found under {mav0_path} — invalid EuRoC dataset"
    return cam_dirs

def get_timestamps_from_csv(data_csv):
    """Return timestamp strings from a EuRoC data.csv in file order."""
    timestamps = []
    with open(data_csv, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            timestamps.append(stripped.split(",")[0])
    return timestamps

def get_filenames_from_csv(data_csv):
    """Return filename strings from a EuRoC data.csv second column, in file order."""
    filenames = []
    with open(data_csv, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            filenames.append(stripped.split(",")[1])
    return filenames


def get_video_processing_config(config_path):
    """Return (generate_videos, ffmpeg_options) from config with defaults."""
    options = {}
    generate_videos = False

    with open(Path(config_path), "r", encoding="utf-8") as f:
        config = json.load(f)
    value0 = config.get("value0", {})
    generate_videos = bool(value0.get("config.use_video_frames", False))

    options["use_dataset_framerate"] = bool(value0.get("config.ffmpeg_use_dataset_framerate", False))
    options["framerate"] = int(value0.get("config.ffmpeg_framerate", 30))
    options["encoder"] = str(value0.get("config.ffmpeg_encoder", "libx264"))

    x264opts = value0.get(
        "config.ffmpeg_x264opts",
        "partitions=p8x8,p4x4,i8x8:keyint=1000:me=umh:merange=64:subme=6:bframes=0:ref=1",
    )
    options["x264opts"] = None if x264opts in (None, "") else str(x264opts)

    bitrate = value0.get("config.ffmpeg_x264_bitrate", "500k")
    options["bitrate"] = None if bitrate in (None, "") else str(bitrate)

    options["pix_fmt"] = str(value0.get("config.ffmpeg_pix_fmt", "yuv420p"))
    options["image_safe_radius"] = float(
        value0.get("config.optical_flow_image_safe_radius", 0.0)
    )
    assert options["image_safe_radius"] >= 0.0, \
        "config.optical_flow_image_safe_radius must be non-negative"
    options["stereo_mv_grid_step"] = int(value0.get("config.ffmpeg_stereo_mv_grid_step", 0))
    assert options["stereo_mv_grid_step"] >= 0, \
        "config.ffmpeg_stereo_mv_grid_step must be non-negative"
    disparity_shifts = value0.get("config.ffmpeg_stereo_mv_disparity_shifts", [0, 8, 32])
    options["stereo_mv_disparity_shifts"] = [int(value) for value in disparity_shifts]
    assert options["stereo_mv_disparity_shifts"] and all(
        value >= 0 for value in options["stereo_mv_disparity_shifts"]
    ), "config.ffmpeg_stereo_mv_disparity_shifts must contain non-negative integers"
    stereo_crf = value0.get("config.ffmpeg_stereo_mv_crf", 23)
    options["stereo_mv_crf"] = None if stereo_crf in (None, "") else int(stereo_crf)
    rectify_calib = value0.get("config.ffmpeg_stereo_mv_rectify_calib", None)
    if rectify_calib in (None, ""):
        options["stereo_mv_rectify_calib"] = None
    else:
        rectify_calib = Path(rectify_calib)
        if not rectify_calib.is_absolute():
            rectify_calib = Path(config_path).resolve().parent / rectify_calib
        options["stereo_mv_rectify_calib"] = rectify_calib.resolve()

    return generate_videos, options


def quaternion_rotation(transform):
    """Return the rotation matrix for Basalt's x-y-z-w quaternion JSON."""
    import numpy as np

    x = float(transform["qx"])
    y = float(transform["qy"])
    z = float(transform["qz"])
    w = float(transform["qw"])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    assert norm > 0, "Invalid zero calibration quaternion"
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def double_sphere_unproject_grid(intrinsics, width, height):
    """Unproject every pixel using Basalt's Double Sphere camera model."""
    import numpy as np

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    xi = float(intrinsics["xi"])
    alpha = float(intrinsics["alpha"])
    pixel_y, pixel_x = np.mgrid[0:height, 0:width]
    mx = (pixel_x.astype(np.float64) - cx) / fx
    my = (pixel_y.astype(np.float64) - cy) / fy
    radius_squared = mx * mx + my * my
    valid = np.ones_like(radius_squared, dtype=bool)
    if alpha > 0.5:
        valid &= radius_squared < 1.0 / (2.0 * alpha - 1.0)
    square_root_2 = np.sqrt(np.maximum(0.0, 1.0 - (2.0 * alpha - 1.0) * radius_squared))
    denominator_2 = alpha * square_root_2 + 1.0 - alpha
    mz = (1.0 - alpha * alpha * radius_squared) / denominator_2
    denominator_1 = mz * mz + radius_squared
    square_root_1 = np.sqrt(np.maximum(0.0, mz * mz + (1.0 - xi * xi) * radius_squared))
    scale = (mz * xi + square_root_1) / denominator_1
    rays = np.stack((scale * mx, scale * my, scale * mz - xi), axis=0)
    return rays, valid


def double_sphere_project(rays, intrinsics):
    """Project a vectorized ray grid using Basalt's Double Sphere model."""
    import numpy as np

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    xi = float(intrinsics["xi"])
    alpha = float(intrinsics["alpha"])
    x, y, z = rays
    distance_1 = np.sqrt(x * x + y * y + z * z)
    k = xi * distance_1 + z
    distance_2 = np.sqrt(x * x + y * y + k * k)
    denominator = alpha * distance_2 + (1.0 - alpha) * k
    w1 = (1.0 - alpha) / alpha if alpha > 0.5 else alpha / (1.0 - alpha)
    w2 = (w1 + xi) / math.sqrt(2.0 * w1 * xi + xi * xi + 1.0)
    valid = (z > -w2 * distance_1) & (denominator > 0)
    pixel_x = fx * x / denominator + cx
    pixel_y = fy * y / denominator + cy
    return pixel_x, pixel_y, valid


def pixel_grid(width, height):
    """Return vectorized pixel coordinates for an image."""
    import numpy as np

    pixel_y, pixel_x = np.indices((height, width), dtype=np.float64)
    return pixel_x, pixel_y


def kb4_unproject_grid(intrinsics, width, height):
    """Unproject every pixel using Basalt's Kannala-Brandt 4 model."""
    import numpy as np

    pixel_x, pixel_y = pixel_grid(width, height)
    mx = (pixel_x - float(intrinsics["cx"])) / float(intrinsics["fx"])
    my = (pixel_y - float(intrinsics["cy"])) / float(intrinsics["fy"])
    theta_distorted = np.sqrt(mx * mx + my * my)
    theta = theta_distorted.copy()
    k1, k2, k3, k4 = (float(intrinsics[f"k{i}"]) for i in range(1, 5))
    derivative = np.ones_like(theta)
    for _ in range(3):
        theta_squared = theta * theta
        function = theta * (
            1.0 + theta_squared * (
                k1 + theta_squared * (k2 + theta_squared * (k3 + theta_squared * k4))
            )
        )
        derivative = 1.0 + theta_squared * (
            3.0 * k1 + theta_squared * (
                5.0 * k2 + theta_squared * (7.0 * k3 + theta_squared * 9.0 * k4)
            )
        )
        theta += (theta_distorted - function) / derivative
    noncentral = theta_distorted > math.sqrt(sys.float_info.epsilon)
    scale = np.ones_like(theta)
    scale[noncentral] = np.sin(theta[noncentral]) / theta_distorted[noncentral]
    rays = np.stack((mx * scale, my * scale, np.cos(theta)), axis=0)
    valid = np.isfinite(rays).all(axis=0) & np.isfinite(derivative)
    return rays, valid


def kb4_project(rays, intrinsics):
    """Project vectorized rays using Basalt's Kannala-Brandt 4 model."""
    import numpy as np

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    k1, k2, k3, k4 = (float(intrinsics[f"k{i}"]) for i in range(1, 5))
    x, y, z = rays
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(radius, z)
    theta_squared = theta * theta
    distorted_radius = theta * (
        1.0 + theta_squared * (
            k1 + theta_squared * (k2 + theta_squared * (k3 + theta_squared * k4))
        )
    )
    scale = np.divide(
        distorted_radius, radius, out=np.ones_like(radius),
        where=radius > math.sqrt(sys.float_info.epsilon)
    )
    pixel_x = fx * x * scale + cx
    pixel_y = fy * y * scale + cy
    valid = (
        np.isfinite(pixel_x) & np.isfinite(pixel_y) &
        (np.linalg.norm(rays, axis=0) > math.sqrt(sys.float_info.epsilon))
    )
    return pixel_x, pixel_y, valid


def radtan8_distort(x, y, intrinsics, with_jacobian=False):
    """Apply Basalt's rational radial-tangential distortion model."""
    k1 = float(intrinsics["k1"])
    k2 = float(intrinsics["k2"])
    p1 = float(intrinsics["p1"])
    p2 = float(intrinsics["p2"])
    k3 = float(intrinsics["k3"])
    k4 = float(intrinsics["k4"])
    k5 = float(intrinsics["k5"])
    k6 = float(intrinsics["k6"])
    radius_squared = x * x + y * y
    numerator = 1.0 + radius_squared * (
        k1 + radius_squared * (k2 + radius_squared * k3)
    )
    denominator = 1.0 + radius_squared * (
        k4 + radius_squared * (k5 + radius_squared * k6)
    )
    radial = numerator / denominator
    distorted_x = (
        x * radial + 2.0 * p1 * x * y +
        p2 * (radius_squared + 2.0 * x * x)
    )
    distorted_y = (
        y * radial + 2.0 * p2 * x * y +
        p1 * (radius_squared + 2.0 * y * y)
    )
    if not with_jacobian:
        return distorted_x, distorted_y
    radius_fourth = radius_squared * radius_squared
    numerator_derivative = (
        k1 + 2.0 * k2 * radius_squared + 3.0 * k3 * radius_fourth
    )
    denominator_derivative = (
        k4 + 2.0 * k5 * radius_squared + 3.0 * k6 * radius_fourth
    )
    radial_derivative = (
        (numerator_derivative * denominator - numerator * denominator_derivative) /
        (denominator * denominator)
    )
    radial_x = 2.0 * x * radial_derivative
    radial_y = 2.0 * y * radial_derivative
    j00 = radial + x * radial_x + 2.0 * p1 * y + 6.0 * p2 * x
    j01 = x * radial_y + 2.0 * p1 * x + 2.0 * p2 * y
    j10 = y * radial_x + 2.0 * p2 * y + 2.0 * p1 * x
    j11 = radial + y * radial_y + 2.0 * p2 * x + 6.0 * p1 * y
    return distorted_x, distorted_y, j00, j01, j10, j11


def radtan8_unproject_grid(intrinsics, width, height):
    """Unproject every pixel using Basalt's pinhole-radtan8 model."""
    import numpy as np

    pixel_x, pixel_y = pixel_grid(width, height)
    target_x = (pixel_x - float(intrinsics["cx"])) / float(intrinsics["fx"])
    target_y = (pixel_y - float(intrinsics["cy"])) / float(intrinsics["fy"])
    x = target_x.copy()
    y = target_y.copy()
    for _ in range(5):
        current_x, current_y, j00, j01, j10, j11 = radtan8_distort(
            x, y, intrinsics, True
        )
        residual_x = current_x - target_x
        residual_y = current_y - target_y
        determinant = j00 * j11 - j01 * j10
        safe = abs(determinant) > sys.float_info.epsilon
        step_x = np.divide(
            j11 * residual_x - j01 * residual_y, determinant,
            out=np.zeros_like(x), where=safe
        )
        step_y = np.divide(
            -j10 * residual_x + j00 * residual_y, determinant,
            out=np.zeros_like(y), where=safe
        )
        x -= step_x
        y -= step_y
    solved_x, solved_y = radtan8_distort(x, y, intrinsics)
    converged = np.hypot(solved_x - target_x, solved_y - target_y) <= 1e-7
    norm = np.sqrt(x * x + y * y + 1.0)
    rays = np.stack((x / norm, y / norm, 1.0 / norm), axis=0)
    radius_squared = x * x + y * y
    radius_max = float(intrinsics.get("rpmax", 0.0))
    valid = np.isfinite(rays).all(axis=0) & converged
    if radius_max != 0.0:
        valid &= radius_squared <= radius_max * radius_max
    return rays, valid


def radtan8_project(rays, intrinsics):
    """Project vectorized rays using Basalt's pinhole-radtan8 model."""
    import numpy as np

    x, y, z = rays
    normalized_x = x / z
    normalized_y = y / z
    distorted_x, distorted_y = radtan8_distort(
        normalized_x, normalized_y, intrinsics
    )
    pixel_x = float(intrinsics["fx"]) * distorted_x + float(intrinsics["cx"])
    pixel_y = float(intrinsics["fy"]) * distorted_y + float(intrinsics["cy"])
    radius_squared = normalized_x * normalized_x + normalized_y * normalized_y
    radius_max = float(intrinsics.get("rpmax", 0.0))
    valid = (
        (z >= math.sqrt(sys.float_info.epsilon)) &
        np.isfinite(pixel_x) & np.isfinite(pixel_y)
    )
    if radius_max != 0.0:
        valid &= radius_squared <= radius_max * radius_max
    return pixel_x, pixel_y, valid


def camera_project(rays, camera):
    """Project vectorized rays with a supported Basalt camera model."""
    camera_type = camera["camera_type"]
    intrinsics = camera["intrinsics"]
    if camera_type == "ds":
        return double_sphere_project(rays, intrinsics)
    if camera_type == "kb4":
        return kb4_project(rays, intrinsics)
    if camera_type == "pinhole-radtan8":
        return radtan8_project(rays, intrinsics)
    raise ValueError(f"Unsupported camera type for stereo rectification: {camera_type}")


def baseline_aligned_virtual_camera(calibration):
    """Construct the shared pinhole frame used by the rectified stereo pair."""
    import numpy as np

    cameras = calibration["intrinsics"]
    transforms = calibration["T_imu_cam"]
    resolutions = calibration["resolution"]
    assert len(cameras) >= 2 and len(transforms) >= 2 and len(resolutions) >= 2, \
        "Stereo rectification requires at least two calibrated cameras"
    width, height = map(int, resolutions[0])
    assert (width, height) == tuple(map(int, resolutions[1])), \
        "Stereo rectification requires equal cam0/cam1 resolutions"

    rotation_i_0 = quaternion_rotation(transforms[0])
    rotation_i_1 = quaternion_rotation(transforms[1])
    rotation_0_1 = rotation_i_0.T @ rotation_i_1
    position_i_0 = np.asarray(
        [transforms[0][key] for key in ("px", "py", "pz")], dtype=np.float64
    )
    position_i_1 = np.asarray(
        [transforms[1][key] for key in ("px", "py", "pz")], dtype=np.float64
    )
    baseline_0 = rotation_i_0.T @ (position_i_1 - position_i_0)
    baseline_norm = float(np.linalg.norm(baseline_0))
    assert baseline_norm > 1e-8, "Stereo cameras have a degenerate calibrated baseline"
    axis_x = baseline_0 / baseline_norm

    optical_axis_0 = np.asarray([0.0, 0.0, 1.0])
    optical_axis_1_in_0 = rotation_0_1 @ optical_axis_0
    forward = optical_axis_0 + optical_axis_1_in_0
    axis_z = forward - axis_x * float(np.dot(axis_x, forward))
    if float(np.linalg.norm(axis_z)) <= 1e-8:
        fallback = min(
            np.eye(3), key=lambda candidate: abs(float(np.dot(candidate, axis_x)))
        )
        axis_z = fallback - axis_x * float(np.dot(axis_x, fallback))
    axis_z /= np.linalg.norm(axis_z)
    axis_y = np.cross(axis_z, axis_x)
    axis_y /= np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    axis_z /= np.linalg.norm(axis_z)
    if float(np.dot(axis_z, forward)) < 0.0:
        axis_y = -axis_y
        axis_z = -axis_z

    rotation_0_rectified = np.column_stack((axis_x, axis_y, axis_z))
    rotation_1_rectified = rotation_0_1.T @ rotation_0_rectified
    focal = min(
        float(cameras[index]["intrinsics"][parameter])
        for index in (0, 1) for parameter in ("fx", "fy")
    )
    assert focal > 0.0, "Stereo rectification requires positive focal lengths"
    return {
        "width": width,
        "height": height,
        "fx": focal,
        "fy": focal,
        "cx": 0.5 * (width - 1),
        "cy": 0.5 * (height - 1),
        "rotation_cam0_rectified": rotation_0_rectified,
        "rotation_cam1_rectified": rotation_1_rectified,
        "baseline_cam0": baseline_0,
        "baseline_m": baseline_norm,
    }


def make_rectification_remap(camera, resolution, rotation_camera_rectified,
                             virtual_camera, image_safe_radius=0.0):
    """Map a native camera image into the shared undistorted pinhole grid."""
    import numpy as np

    source_width, source_height = map(int, resolution)
    width = virtual_camera["width"]
    height = virtual_camera["height"]
    pixel_x, pixel_y = pixel_grid(width, height)
    rectified_rays = np.stack((
        (pixel_x - virtual_camera["cx"]) / virtual_camera["fx"],
        (pixel_y - virtual_camera["cy"]) / virtual_camera["fy"],
        np.ones_like(pixel_x),
    ), axis=0)
    rectified_rays /= np.linalg.norm(rectified_rays, axis=0)
    native_rays = (
        rotation_camera_rectified @ rectified_rays.reshape(3, -1)
    ).reshape(rectified_rays.shape)
    source_x, source_y, valid = camera_project(native_rays, camera)
    valid &= (source_x >= 0.0) & (source_y >= 0.0)
    valid &= (source_x <= source_width - 1) & (source_y <= source_height - 1)
    if image_safe_radius > 0.0:
        radius_squared = float(image_safe_radius) ** 2
        valid &= (
            (source_x - 0.5 * source_width) ** 2 +
            (source_y - 0.5 * source_height) ** 2
        ) <= radius_squared

    source_x = np.clip(source_x, 0.0, source_width - 1)
    source_y = np.clip(source_y, 0.0, source_height - 1)
    x0 = np.floor(source_x).astype(np.int32)
    y0 = np.floor(source_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    weight_x = source_x - x0
    weight_y = source_y - y0
    return {
        "width": width,
        "height": height,
        "source_width": source_width,
        "source_height": source_height,
        "valid": valid,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "w00": (1.0 - weight_x) * (1.0 - weight_y),
        "w10": weight_x * (1.0 - weight_y),
        "w01": (1.0 - weight_x) * weight_y,
        "w11": weight_x * weight_y,
    }


def make_baseline_aligned_stereo_remaps(calib_path, image_safe_radius=0.0):
    """Build two remaps into one baseline-aligned undistorted pinhole frame."""
    import numpy as np

    with Path(calib_path).open(encoding="utf-8") as stream:
        calibration = json.load(stream)["value0"]
    cameras = calibration["intrinsics"]
    transforms = calibration["T_imu_cam"]
    resolutions = calibration["resolution"]
    assert len(cameras) >= 2 and len(transforms) >= 2 and len(resolutions) >= 2, \
        "Stereo rectification requires two calibrated cameras"
    virtual_camera = baseline_aligned_virtual_camera(calibration)
    left_remap = make_rectification_remap(
        cameras[0], resolutions[0], virtual_camera["rotation_cam0_rectified"],
        virtual_camera, image_safe_radius
    )
    right_remap = make_rectification_remap(
        cameras[1], resolutions[1], virtual_camera["rotation_cam1_rectified"],
        virtual_camera, image_safe_radius
    )
    return virtual_camera, left_remap, right_remap


def image_to_uint8_grayscale(image):
    """Convert grayscale input to the codec's 8-bit range without clipping 16-bit PNGs."""
    import numpy as np

    source = np.asarray(image)
    if source.ndim != 2:
        return np.asarray(image.convert("L"), dtype=np.uint8)
    if source.dtype == np.uint8:
        return source
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"Unsupported grayscale dtype: {source.dtype}")
    if source.size and (int(source.min()) < 0 or int(source.max()) > 65535):
        raise ValueError(
            f"16-bit grayscale values outside [0, 65535]: "
            f"[{int(source.min())}, {int(source.max())}]"
        )
    return np.right_shift(source.astype(np.uint32), 8).astype(np.uint8)


def write_rectification_validity_mask(remap, validity_mask_path):
    """Store one reusable validity mask for the rectified encoder grid."""
    import numpy as np
    from PIL import Image

    Image.fromarray(
        np.where(remap["valid"], 255, 0).astype(np.uint8), mode="L"
    ).save(validity_mask_path)


def write_baseline_rectification_metadata(metadata_path, virtual_camera,
                                          camera_count):
    """Write the virtual pinhole model shared by preprocessing and runtime."""
    serializable = {
        "version": 1,
        "model": "pinhole",
        "camera_indices": [0, 1],
        "ignored_camera_count": max(0, int(camera_count) - 2),
        "width": int(virtual_camera["width"]),
        "height": int(virtual_camera["height"]),
        "fx": float(virtual_camera["fx"]),
        "fy": float(virtual_camera["fy"]),
        "cx": float(virtual_camera["cx"]),
        "cy": float(virtual_camera["cy"]),
        "baseline_m": float(virtual_camera["baseline_m"]),
        "baseline_cam0": virtual_camera["baseline_cam0"].tolist(),
        "rotation_cam0_rectified": (
            virtual_camera["rotation_cam0_rectified"].tolist()
        ),
        "rotation_cam1_rectified": (
            virtual_camera["rotation_cam1_rectified"].tolist()
        ),
        "left_validity_mask": "left-rectified-valid-mask.png",
        "right_validity_mask": "right-rectified-valid-mask.png",
    }
    Path(metadata_path).write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def encode_rectified_video(image_files, remap, output_video, framerate):
    """Create one lossless native-to-rectified intermediate video."""
    import numpy as np
    from PIL import Image

    width = remap["width"]
    height = remap["height"]
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "gray", "-video_size", f"{width}x{height}",
        "-framerate", str(framerate), "-i", "-", "-c:v", "ffv1",
        "-level", "3", "-g", "1", str(output_video),
    ]
    process = Popen(command, stdin=PIPE)
    try:
        assert process.stdin is not None
        for image_path in image_files:
            with Image.open(image_path) as image:
                source = image_to_uint8_grayscale(image)
            expected_shape = (remap["source_height"], remap["source_width"])
            assert source.shape == expected_shape, \
                f"Unexpected native image size {source.shape} in {image_path}; expected {expected_shape}"
            rectified = (
                source[remap["y0"], remap["x0"]] * remap["w00"] +
                source[remap["y0"], remap["x1"]] * remap["w10"] +
                source[remap["y1"], remap["x0"]] * remap["w01"] +
                source[remap["y1"], remap["x1"]] * remap["w11"]
            )
            rectified = np.where(remap["valid"], np.rint(rectified), 0).astype(np.uint8)
            process.stdin.write(rectified.tobytes())
        process.stdin.close()
        returncode = process.wait()
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    assert returncode == 0, f"Failed to encode rectified video: {output_video}"

def recreate_euroc_with_shared_timestamps(dataset_path, video_base, shared_timestamps, cam_dirs):
    """Recreate EuRoC under video_base with only shared cam timestamps in data.csv files."""
    dataset_path = Path(dataset_path)
    video_base = Path(video_base)

    print(f"Recreating filtered EuRoC dataset at {video_base}")
    if video_base.exists():
        shutil.rmtree(video_base)
    # Skip copying cam*/data image folders.
    def _ignore_cam_image_dirs(src, names):
        if Path(src).name.startswith("cam") and Path(src).parent.name == "mav0":
            return {"data"} & set(names)
        return set()
        
    shutil.copytree(dataset_path, video_base, ignore=_ignore_cam_image_dirs)

    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        orig_csv = cam_dir / "data.csv"
        new_csv = video_base / "mav0" / cam_name / "data.csv"

        header_lines = []
        filtered_rows = []
        with open(orig_csv, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    header_lines.append(line)
                else:
                    ts = stripped.split(",")[0]
                    if ts in shared_timestamps:
                        filtered_rows.append(line)

        print(f"  {cam_name}: {len(filtered_rows)} rows kept (from {orig_csv.name})")
        with open(new_csv, "w", encoding="utf-8") as f:
            f.writelines(header_lines)
            f.writelines(filtered_rows)

    return video_base

def build_ffmpeg_concat_file(filenames, original_data_dir, output_path, framerate):
    """Write an ffmpeg concat file that points to exact PNG paths.

    The 'ffconcat version 1.0' header and per-entry 'duration' lines are required
    by the concat demuxer for still images, ensuring correct sequential timestamps
    and preventing frames from being dropped.
    """
    frame_duration = 1.0 / framerate
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n\n")
        for filename in filenames:
            png_path = (Path(original_data_dir) / filename).resolve()
            assert png_path.exists(), \
                f"PNG listed in data.csv not found on disk: {png_path}"
            f.write(f"file '{png_path}'\n")
            f.write(f"duration {frame_duration}\n")
    return output_path


def encode_video(concat_txt, output_video, framerate, encoding_options,
                 x264opts=None):
    """Encode a two-pass video with the configured x1 compression settings."""
    base_cmd_parts = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-r", str(framerate),
        "-i", str(concat_txt),
        "-c:v", encoding_options["encoder"],
    ]
    if encoding_options["bitrate"] is not None:
        base_cmd_parts.extend(["-b:v", encoding_options["bitrate"]])
    base_cmd_parts.extend([
        "-pix_fmt", encoding_options["pix_fmt"],
        "-r", str(framerate),
        "-vf", "setpts=PTS-STARTPTS",
    ])

    if x264opts is None:
        x264opts = encoding_options["x264opts"]
    if x264opts is not None and encoding_options["encoder"] == "libx264":
        base_cmd_parts.extend(["-x264opts", x264opts])
    passlogfile = str(output_video.parent / f"ffmpeg2pass-{output_video.stem}")
    pass1_cmd_parts = base_cmd_parts + [
        "-passlogfile", passlogfile,
        "-pass", "1",
        "-an",
        "-f", "null",
        "/dev/null",
    ]
    pass1_cmd = " ".join(shlex.quote(p) for p in pass1_cmd_parts)
    stdout, returncode = shout(pass1_cmd)
    assert returncode == 0, f"ffmpeg pass 1 failed for {output_video}:\n{stdout}"

    pass2_cmd_parts = base_cmd_parts + [
        "-passlogfile", passlogfile,
        "-pass", "2",
        "-an",
        "-f", "mp4",
        str(output_video),
    ]
    pass2_cmd = " ".join(shlex.quote(p) for p in pass2_cmd_parts)
    stdout, returncode = shout(pass2_cmd)
    assert returncode == 0, f"ffmpeg pass 2 failed for {output_video}:\n{stdout}"

    Path(f"{passlogfile}-0.log").unlink(missing_ok=True)
    Path(f"{passlogfile}-0.log.mbtree").unlink(missing_ok=True)


def ipp_x264opts(x264opts):
    """Keep x1's motion search settings but force independent I-P-P triples."""
    assert x264opts is not None, "I-P-P generation requires libx264 options"
    forced_keys = {"keyint", "min-keyint", "scenecut", "bframes", "ref"}
    kept = []
    for option in x264opts.split(":"):
        key = option.split("=", 1)[0]
        if key not in forced_keys:
            kept.append(option)
    kept.extend(["keyint=3", "min-keyint=3", "scenecut=0", "bframes=0", "ref=1"])
    return ":".join(kept)


def encode_stereo_ipp_video(left_concat, right_input, output_video, framerate,
                            encoding_options, disparity_shift, grid_shift_x,
                            grid_shift_y, left_input_is_video=False,
                            right_input_is_video=False):
    """Encode the paper's reset I(left)-P(right)-P(left) H.264 sequence."""
    left_filters = []
    if disparity_shift:
        left_filters.extend([
            f"crop=iw-{disparity_shift}:ih:{disparity_shift}:0",
            f"pad=iw+{disparity_shift}:ih:0:0:black",
        ])
    if grid_shift_x or grid_shift_y:
        grid_filter = (
            f"pad=iw+{grid_shift_x}:ih+{grid_shift_y}:"
            f"{grid_shift_x}:{grid_shift_y}:black,"
            f"crop=iw-{grid_shift_x}:ih-{grid_shift_y}:0:0"
        )
        left_filters.append(grid_filter)
        right_filters = [grid_filter]
    else:
        right_filters = []

    left_chain = ",".join(left_filters) if left_filters else "null"
    right_chain = ",".join(right_filters) if right_filters else "null"
    output_rate = 3 * framerate
    filter_complex = (
        f"[0:v]{left_chain},split=2[left0][left2];"
        f"[1:v]{right_chain}[right1];"
        f"[left0]settb=1/{output_rate},setpts=3*N[left0t];"
        f"[right1]settb=1/{output_rate},setpts=3*N+1[right1t];"
        f"[left2]settb=1/{output_rate},setpts=3*N+2[left2t];"
        "[left0t][right1t][left2t]interleave=nb_inputs=3:duration=longest[out]"
    )
    command = ["ffmpeg", "-y"]
    if left_input_is_video:
        command.extend(["-i", str(left_concat)])
    else:
        command.extend([
            "-f", "concat", "-safe", "0", "-r", str(framerate),
            "-i", str(left_concat),
        ])
    if right_input_is_video:
        command.extend(["-i", str(right_input)])
    else:
        command.extend([
            "-f", "concat", "-safe", "0", "-r", str(framerate), "-i", str(right_input)
        ])
    command.extend([
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", encoding_options["encoder"],
    ])
    if encoding_options["stereo_mv_crf"] is not None:
        command.extend(["-crf", str(encoding_options["stereo_mv_crf"])])
    elif encoding_options["bitrate"] is not None:
        command.extend(["-b:v", encoding_options["bitrate"]])
    command.extend([
        "-pix_fmt", encoding_options["pix_fmt"],
        "-r", str(output_rate),
        "-x264opts", ipp_x264opts(encoding_options["x264opts"]),
        "-an", "-f", "mp4", str(output_video),
    ])
    stdout, returncode = shout(" ".join(shlex.quote(part) for part in command))
    assert returncode == 0, f"ffmpeg I-P-P encode failed for {output_video}:\n{stdout}"


def verify_ipp_video(video_path, pair_count, description):
    """Require one independent I-P-P group for every requested pair."""
    result = run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=pict_type", "-of", "csv=p=0",
            str(video_path),
        ],
        check=False,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    assert result.returncode == 0, \
        f"ffprobe failed for {video_path}:\n{result.stderr}"
    frame_types = [line.strip().split(",", 1)[0]
                   for line in result.stdout.splitlines() if line.strip()]
    assert len(frame_types) == 3 * pair_count, \
        f"{description} frame count mismatch for {video_path}: {len(frame_types)} vs {3 * pair_count}"
    bad = [(i, value) for i, value in enumerate(frame_types)
           if value != ("I" if i % 3 == 0 else "P")]
    assert not bad, f"{description} GOP is not strict I-P-P in {video_path}: {bad[:10]}"


def video_frame_types(video_path):
    """Return decoded frame types for an existing temporal H.264 video."""
    result = run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=pict_type", "-of", "csv=p=0",
            str(video_path),
        ],
        check=False,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    assert result.returncode == 0, \
        f"ffprobe failed for {video_path}:\n{result.stderr}"
    return [line.strip().split(",", 1)[0]
            for line in result.stdout.splitlines() if line.strip()]


def output_videos(working_dir, dataset_path, device, encoding_options, stereo_only=False,
                  reverse_temporal_only=False, rectification_mask_only=False,
                  max_video_frames=0):
    """Encode PNGs from a EuRoC-format dataset into per-camera videos.

    We always feed ffmpeg from the original cam*/data PNG paths using a concat
    text file. If camera timestamps do not match and dataset is allowlisted, we
    recreate EuRoC metadata under video_base with filtered data.csv files.

    Returns:
        (video_base, new_dataset_path)
          video_base: output directory for per-camera videos, or None on failure.
          new_dataset_path: recreated filtered EuRoC path, else None.
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name
    video_base = Path(working_dir) / "videos" / device / dataset_name

    # Capture original image dirs once; these stay the source of PNG frames.
    original_cam_dirs = get_euroc_cam_dirs(dataset_path)
    original_data_dirs = {}
    for cam_dir in original_cam_dirs:
        data_dir = cam_dir / "data"
        assert data_dir.is_dir(), \
            f"EuRoC image directory not found: {data_dir}"
        original_data_dirs[cam_dir.name] = data_dir

    # CSV paths per cam; these may be redirected to filtered copies later.
    csv_per_cam = {cam_dir.name: cam_dir / "data.csv" for cam_dir in original_cam_dirs}
    for cam_name, csv_path in csv_per_cam.items():
        assert csv_path.exists(), \
            f"data.csv missing from EuRoC camera directory: {csv_path.parent}"

    # Read timestamps from each camera CSV.
    timestamps_per_cam = {
        cam_name: get_timestamps_from_csv(csv_path)
        for cam_name, csv_path in csv_per_cam.items()
    }

    filenames_per_cam = {
        cam_name: get_filenames_from_csv(csv_path)
        for cam_name, csv_path in csv_per_cam.items()
    }

    # Check whether camera timestamp sets match.
    timestamp_sets = {cam: set(ts_list) for cam, ts_list in timestamps_per_cam.items()}
    all_equal = len(set(frozenset(s) for s in timestamp_sets.values())) == 1

    new_dataset_path = None  # Set only for mismatch + allowlist datasets.

    if not all_equal:
        counts = {cam: len(ts) for cam, ts in timestamps_per_cam.items()}
        assert dataset_name in MISMATCH_ALLOWLIST_DATASETS, (
            f"Timestamp mismatch across cameras for dataset '{dataset_name}': {counts}. "
            f"Add it to MISMATCH_ALLOWLIST_DATASETS if this is intentional."
        )

        print(f"WARNING: timestamp mismatch across cameras: {counts}")
        print(f"WARNING: '{dataset_name}' is allowlisted — recreating filtered EuRoC dataset")

        shared_timestamps = set.intersection(*timestamp_sets.values())
        assert shared_timestamps, \
            f"No shared timestamps across all cameras for dataset '{dataset_name}'"

        new_dataset_path = recreate_euroc_with_shared_timestamps(
            dataset_path, video_base, shared_timestamps, original_cam_dirs
        )

        # Use filtered CSVs from recreated dataset; PNGs stay at original paths.
        filtered_cam_dirs = get_euroc_cam_dirs(new_dataset_path)
        csv_per_cam = {cam_dir.name: cam_dir / "data.csv" for cam_dir in filtered_cam_dirs}
        timestamps_per_cam = {
            cam_name: get_timestamps_from_csv(csv_path)
            for cam_name, csv_path in csv_per_cam.items()
        }
        filenames_per_cam = {
            cam_name: get_filenames_from_csv(csv_path)
            for cam_name, csv_path in csv_per_cam.items()
        }

    # Encode one video per camera.
    encoded_any = False

    for cam_name, timestamps in timestamps_per_cam.items():
        output_dir = video_base / "mav0" / cam_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if encoding_options["use_dataset_framerate"]:
            framerate = compute_framerate(csv_per_cam[cam_name].parent)
        else:
            framerate = encoding_options["framerate"]

        concat_txt = output_dir / "input.txt"
        build_ffmpeg_concat_file(filenames_per_cam[cam_name], original_data_dirs[cam_name], concat_txt, framerate)
        print(f"Written concat file: {concat_txt} ({len(timestamps)} entries, {framerate} fps)")

        if reverse_temporal_only:
            reverse_names = filenames_per_cam[cam_name]
            if max_video_frames > 0:
                reverse_names = reverse_names[:max_video_frames]
            assert len(reverse_names) >= 2, \
                "Reverse temporal MV generation requires at least two frames"
            reverse_concat = output_dir / "reverse-input.txt"
            build_ffmpeg_concat_file(
                list(reversed(reverse_names)),
                original_data_dirs[cam_name],
                reverse_concat,
                framerate,
            )
            reverse_video = output_dir / "data_reverse.mp4"
            print(
                f"Encoding {len(reverse_names)} reversed frames into "
                f"{reverse_video} with VOCA H.264 settings"
            )
            encode_video(
                reverse_concat,
                reverse_video,
                framerate,
                encoding_options,
            )
            frame_types = video_frame_types(reverse_video)
            assert len(frame_types) == len(reverse_names), (
                f"Reverse temporal video contains {len(frame_types)} frames, "
                f"expected {len(reverse_names)}"
            )
            assert frame_types[0] == "I", \
                f"Reverse temporal video does not begin with an I frame: {reverse_video}"
            print(f"Verified reverse VOCA temporal video: {reverse_video}")
            encoded_any = True
            continue

        if stereo_only or rectification_mask_only:
            continue

        output_video = output_dir / "data.mp4"
        print(
            f"Encoding {len(timestamps)} frames from {original_data_dirs[cam_name]} "
            f"into {output_video} at {framerate} fps"
        )
        encode_video(concat_txt, output_video, framerate, encoding_options)

        print(f"Successfully created {output_video}")
        encoded_any = True

    grid_step = encoding_options["stereo_mv_grid_step"]
    if (grid_step > 0
            and not reverse_temporal_only):
        assert encoding_options["encoder"] == "libx264", \
            "Stereo MV generation requires the libx264 encoder"
        assert "cam0" in timestamps_per_cam and "cam1" in timestamps_per_cam, \
            "Stereo MV generation requires cam0 and cam1"
        assert timestamps_per_cam["cam0"] == timestamps_per_cam["cam1"], \
            "Stereo MV generation requires synchronized, identically ordered cam0/cam1 timestamps"

        stereo_root = video_base / "mav0" / "stereo"
        rectify_calib = encoding_options["stereo_mv_rectify_calib"]
        stereo_dir = (
            stereo_root / "baseline-pinhole-v1"
            if rectify_calib is not None else stereo_root
        )
        stereo_dir.mkdir(parents=True, exist_ok=True)
        stereo_framerate = (
            compute_framerate(csv_per_cam["cam0"].parent)
            if encoding_options["use_dataset_framerate"]
            else encoding_options["framerate"]
        )
        left_names = filenames_per_cam["cam0"]
        right_names = filenames_per_cam["cam1"]
        if max_video_frames > 0:
            left_names = left_names[:max_video_frames]
            right_names = right_names[:max_video_frames]
        left_files = [original_data_dirs["cam0"] / name for name in left_names]
        right_files = [original_data_dirs["cam1"] / name for name in right_names]
        left_concat = stereo_dir / "left-input.txt"
        right_concat = stereo_dir / "right-input.txt"
        build_ffmpeg_concat_file(
            left_names, original_data_dirs["cam0"], left_concat, stereo_framerate
        )
        build_ffmpeg_concat_file(
            right_names, original_data_dirs["cam1"], right_concat, stereo_framerate
        )
        rectified_left_video = stereo_dir / "left-baseline-rectified.mkv"
        rectified_right_video = stereo_dir / "right-baseline-rectified.mkv"
        left_validity_mask = stereo_dir / "left-rectified-valid-mask.png"
        right_validity_mask = stereo_dir / "right-rectified-valid-mask.png"
        baseline_metadata = stereo_dir / "stereo-baseline-rectification.json"
        right_input = right_concat
        left_input = left_concat
        left_input_is_video = False
        right_input_is_video = False
        (stereo_dir / "right-rectified-to-left").unlink(missing_ok=True)
        baseline_metadata.unlink(missing_ok=True)
        if rectify_calib is not None:
            assert Path(rectify_calib).is_file(), \
                f"Stereo rectification calibration not found: {rectify_calib}"
            with Path(rectify_calib).open(encoding="utf-8") as stream:
                camera_count = len(json.load(stream)["value0"]["intrinsics"])
            if camera_count > 2:
                print(
                    f"Calibration has {camera_count} cameras; Passthrough stereo "
                    "uses the primary cam0/cam1 pair and leaves additional cameras unchanged."
                )
            virtual_camera, left_remap, right_remap = \
                make_baseline_aligned_stereo_remaps(
                    rectify_calib, encoding_options["image_safe_radius"]
                )
            write_rectification_validity_mask(left_remap, left_validity_mask)
            write_rectification_validity_mask(right_remap, right_validity_mask)
            if rectification_mask_only:
                write_baseline_rectification_metadata(
                    baseline_metadata, virtual_camera, camera_count
                )
                print(
                    f"Written baseline rectification metadata and masks to {stereo_dir}"
                )
                return video_base, new_dataset_path
            print(
                f"Rectifying {len(left_files)} synchronized cam0/cam1 pairs into "
                f"a common baseline-aligned pinhole grid using {rectify_calib}"
            )
            encode_rectified_video(
                left_files, left_remap, rectified_left_video, stereo_framerate
            )
            encode_rectified_video(
                right_files, right_remap, rectified_right_video, stereo_framerate
            )
            left_input = rectified_left_video
            right_input = rectified_right_video
            left_input_is_video = True
            right_input_is_video = True
        grid_phases = [(0, 0), (grid_step, 0), (0, grid_step), (grid_step, grid_step)]
        stereo_encodings = [
            (disparity_shift, shift_x, shift_y,
             stereo_dir / f"data_ipp_d{disparity_shift}_g{shift_x}_{shift_y}.mp4")
            for disparity_shift in encoding_options["stereo_mv_disparity_shifts"]
            for shift_x, shift_y in grid_phases
        ]
        try:
            for disparity_shift, shift_x, shift_y, output_video in stereo_encodings:
                print(
                    f"Encoding {len(left_files)} synchronized stereo pairs into {output_video} "
                    f"as I-P-P at {3 * stereo_framerate} fps with disparity shift "
                    f"{disparity_shift} and grid shift ({shift_x}, {shift_y})"
                )
                encode_stereo_ipp_video(
                    left_input,
                    right_input,
                    output_video,
                    stereo_framerate,
                    encoding_options,
                    disparity_shift,
                    shift_x,
                    shift_y,
                    left_input_is_video=left_input_is_video,
                    right_input_is_video=right_input_is_video,
                )
                verify_ipp_video(output_video, len(left_files), "Stereo")
                print(f"Verified strict I-P-P stereo GOP: {output_video}")
                encoded_any = True
            if rectify_calib is not None:
                write_baseline_rectification_metadata(
                    baseline_metadata, virtual_camera, camera_count
                )
        finally:
            if rectify_calib is not None:
                rectified_left_video.unlink(missing_ok=True)
                rectified_right_video.unlink(missing_ok=True)

    if not encoded_any:
        print(f"No videos were encoded from dataset {dataset_path}")
        return None, None

    return video_base, new_dataset_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset file to uncompress (without .zip extension).",
    )
    parser.add_argument(
        "working_dir",
        type=Path,
        help="Fast working directory where the dataset will be uncompressed.",
    )
    parser.add_argument(
        "--lock_file",
        type=str,
        default="/tmp/uncompression.lock",
        help="Path to the lock file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the basalt config JSON file. Used to check if MV video "
        "generation is needed (looks for config.use_video_frames).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device name (e.g., euroc, tumvi, msdmo). Used for video output path under <working_dir>/videos/.",
    )
    parser.add_argument(
        "--stereo-only",
        action="store_true",
        help="Generate only the stereo I-P-P streams, leaving temporal videos untouched.",
    )
    parser.add_argument(
        "--reverse-temporal-only",
        action="store_true",
        help="Generate only a reverse-time temporal stream with the ordinary VOCA H.264 settings.",
    )
    parser.add_argument(
        "--stereo-rectification-mask-only",
        action="store_true",
        help="Generate only the stereo rectification validity mask, leaving videos untouched.",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=0,
        help="Limit experimental stereo encoding to the first N pairs; zero means all.",
    )
    return parser.parse_args()



def main():
    """Main function to handle dataset uncompression and optional video generation."""
    args = parse_args()
    selected_video_modes = sum((args.stereo_only, args.reverse_temporal_only,
                                args.stereo_rectification_mask_only))
    if selected_video_modes > 1:
        raise SystemExit("--stereo-only, --reverse-temporal-only, and "
                         "--stereo-rectification-mask-only "
                         "are mutually exclusive")
    dataset_path = args.dataset_path
    working_dir = args.working_dir
    lock_file = Path(args.lock_file)

    zip_path = dataset_path.with_suffix(".zip")

    working_mount_point = get_mount_point(working_dir)
    print(f"{working_mount_point=}")

    final_ds_path = None

    if not zip_path.exists():
        dataset_mount_point = get_mount_point(dataset_path)
        print(f"{dataset_mount_point=}")
        if dataset_mount_point == working_mount_point:
            final_ds_path = dataset_path
        else:
            print("No zip found and mount points differ — cannot proceed")
            sys.exit(1)
    else:
        dataset_mount_point = get_mount_point(zip_path)
        print(f"{dataset_mount_point=}")
        print(f"Uncompressing {zip_path} to {working_dir}")

        # Naive locking (race conditions possible but rare)
        prev_lock_contents = ""
        while lock_file.exists():
            with lock_file.open(encoding="utf-8") as f:
                lock_contents = f.read().strip()
            if lock_contents != prev_lock_contents:
                print(f"Waiting for lockfile to be removed... {lock_file} = {lock_contents}")
                prev_lock_contents = lock_contents
            time.sleep(WHILE_LOCK_SLEEP_SECONDS)

        # Acquire lock
        with lock_file.open("w", encoding="utf-8") as f:
            f.write(f"{zip_path} ")

        try:
            wait_space_for_file(zip_path, working_dir)
            shout(f"7z x -y {zip_path} -o{working_dir}")
            print("New dataset path name is:")
            final_ds_path = working_dir / dataset_path.name
        except Exception as e:
            print(f"Failed to obtain dataset: {e}")

        # Release lock
        lock_file.unlink(missing_ok=True)

    # optional video generation 
    if args.config_path is not None:
        generate_videos, encoding_options = get_video_processing_config(args.config_path)
        if final_ds_path is not None and generate_videos:
            device = args.device or "unknown"
            print(f"Config requests MV video generation for device '{device}'")
            video_path, new_dataset_path = output_videos(
                working_dir, final_ds_path, device, encoding_options, args.stereo_only,
                args.reverse_temporal_only, args.stereo_rectification_mask_only,
                args.max_video_frames
            )
            if video_path:
                print(f"VIDEO_PATH={video_path}")
                # When a filtered EuRoC was recreated due to a timestamp mismatch,
                # the recreated dataset IS the final dataset path to use downstream.
                if new_dataset_path is not None:
                    final_ds_path = new_dataset_path
            else:
                print("VIDEO_PATH=")

    # Final output line: dataset path (must remain last for backward compatibility)
    print(final_ds_path)


if __name__ == "__main__":
    main()
