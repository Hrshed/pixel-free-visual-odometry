# Build and run the thesis configuration

The release contains the locked codec-only implementation plus the configurable suppression-radius changes used for the selected PVO settings. It preserves the original dependency revisions. The thesis's archived numerical results are reported as published; a fresh full-benchmark evaluation has not been run as part of this repository packaging.

## Build

The evaluated environment used Ubuntu 22.04, GCC 11.4, OpenCV 4.5.4, FFmpeg 4.4.2, and TEASER++ revision `52a9c52`. Dependency availability and compiler changes can affect a fresh build.

```bash
git submodule update --init --recursive
./scripts/install_deps.sh
sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev python3-numpy python3-opencv p7zip-full
```

Install [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) separately, using revision `52a9c52` for the recorded environment. Set `TEASER_PREFIX` to the installation containing `include/teaser/registration.h` and `lib/libteaser_registration.so`.

```bash
export TEASER_PREFIX=/absolute/path/to/teaser-install
cmake -S . -B build-locked -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
  -DBASALT_ENABLE_TEASERPP=ON \
  -DBASALT_TEASERPP_ROOT="$TEASER_PREFIX"
cmake --build build-locked --target basalt_vio -j2
```

The [upstream README](../README.upstream.md) and [development guide](Development.md) provide additional platform context.

## Prepare EuRoC inputs

Download and extract a sequence from the [official EuRoC dataset provider](https://projects.asl.ethz.ch/datasets/euroc-mav/). Start with `MH_01_easy`. Keep the original `mav0` directory and timestamp CSVs intact. The preprocessing script expects its working directory to exist and, for an already extracted dataset without an adjacent archive, to be on the same filesystem as the dataset.

```bash
export DATASET=/absolute/path/to/MH_01_easy
export WORK=/absolute/path/to/encoding-work
mkdir -p "$WORK"
python3 .ci/get_dataset.py "$DATASET" "$WORK" \
  --config_path data/euroc/euroc_config_vo.json --device euroc
```

The final line prints the dataset path to use downstream. If timestamps have to be synchronized, this can be a recreated dataset rather than the original path. `VIDEO_PATH=` identifies the generated video directory. Use those printed paths for the reverse pass and estimator.

```bash
export PREPARED_DATASET=/absolute/path/printed/on/the/final/line
export VIDEOS=/absolute/path/from/VIDEO_PATH
python3 .ci/get_dataset.py "$PREPARED_DATASET" "$WORK" \
  --config_path data/euroc/euroc_config_vo.json --device euroc \
  --reverse-temporal-only
```

Preparation creates forward and reverse temporal videos plus twelve rectified stereo streams (three disparity shifts × four grid phases) and validity masks. It uses software x264, two-pass temporal encoding, and full-sequence reversal, so allow storage and preparation time. The selected estimator settings in `configs/pvo.json` are deliberately separate from the preprocessing configuration, whose relative calibration path must remain valid.

## Run PVO

```bash
export PVO_BINARY="$PWD/build-locked/basalt_vio"
export LD_LIBRARY_PATH="$TEASER_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./scripts/run_pvo.sh "$PREPARED_DATASET" "$VIDEOS" "$PWD/results/MH_01_easy"
```

The wrapper supplies the selected reciprocal, suppression, and backend settings documented in the thesis. It writes `tracking.csv` and `result.json` under the output directory. It refuses to overwrite either existing result. The JSON config alone does not contain every command-line frontend setting; use the wrapper or copy its complete invocation.

The estimator consumes codec metadata and calibrated geometry; the current FFmpeg extraction implementation may reconstruct images internally. Inertial input is disabled. This is the evaluated research pipeline, not an end-to-end live hardware demonstration.

## Metrics and documents

ATE is position RMSE after one full-trajectory rigid alignment, without scale fitting. RTE-6 is relative translation RMSE over non-overlapping six-frame intervals. The thesis's reproducibility appendix records the metric conventions, reference configuration hashes, and original evaluation commands. Archived result directories referenced there are historical provenance and are not bundled into this compact release.

Read the final [thesis PDF](thesis/thesis.pdf) or [defense slides](presentation/defense-slides.pdf). To rebuild the thesis source, install the dependencies listed in `doc/thesis/install-dependencies-ubuntu.sh`, together with `latexmk`, Biber and the glossary tools required by your TeX distribution, then run:

```bash
make -C doc/thesis pdf
```

The distributed thesis PDF is the unsigned final build. Its scientific content and source are retained; the handwritten signature is not distributed.
