# Snapshot provenance

This repository starts with a single `main` snapshot of the finished thesis work. It does not import the original development branches or Git history.

- Locked source commit: `29d09aa0b14ecefe7278d201fd31cd2a96dbf5d3`.
- Document worktree HEAD at export: `8fe88629677ec132e05ce1ce5352cd6119d30912`.
- Included uncommitted software changes: parameterized suppression radius in `frame_to_frame_optical_flow.h`, `vio_config.h`, `vio_config.cpp`, and `vio.cpp`. These preserve the selected 1.125-pixel setting documented in the thesis.
- The release wrapper adds path-neutral invocation of the published selected settings; it does not alter the estimator.
- The final thesis build is included without the handwritten signature. The defense PDF is dated 28 September 2026.

## File identities

| File | SHA-256 |
|---|---|
| `configs/pvo.json` | `8457ae6d36f8c137bdbf9f4dcaa29a23d480c9f2cad1bee41176c0578c27ee1b` |
| `data/euroc/euroc_ds_calib.json` | `ad8c5a18c48c55dacf61d18ebbc18cd7d4f3acccd5840bd5a5cd8646adb6271c` |
| `doc/thesis/thesis.pdf` | `dac18afc8a2bf639f7dbbce233d8f6073e9a77bcb1441b56af5e01efc6a81605` |
| `doc/presentation/defense-slides.pdf` | `0d308a7baf320d0187c6223e71a8e75aed35f12b49f306b52173f87e9f34c318` |

The selected PVO configuration and EuRoC calibration hashes match the thesis appendix. The archived evaluation executable is identified there by SHA-256 `bd9b6641fb7ba57df508cf188125838474fb29a689d96d7b271a99e2a1464659`; binaries are not distributed, and a new build is not claimed to be byte-identical to that executable.

## Dependency revisions

| Submodule | Commit |
|---|---|
| `thirdparty/CLI11` | `4af78beef777e313814b4daff70e2da9171a385a` |
| `thirdparty/Pangolin` | `73967b39ad2cf84245cf877ce0879dc0909d3be4` |
| `thirdparty/basalt-headers` | `1a519e990b2cbd62223a7cdab5e19a288412c5d9` |
| `thirdparty/magic_enum` | `3d1f6a5a2a3fbcba077e00ad0ccc2dd9fefc2ca7` |
| `thirdparty/opengv` | `91f4b19c73450833a40e463ad3648aae80b3a7f3` |
| `thirdparty/ros/console_bridge` | `f0b423c0c2d02651db1776c96887c0a314908063` |
| `thirdparty/ros/ros_comm` | `194c776737134a429619f6851f1850734843e210` |
| `thirdparty/ros/roscpp_core` | `1a74781ce707b1a2198d1bd6799b892b3c1fa195` |

Each submodule preserves its nested dependency pins. See `.gitmodules` for its public upstream URL.

## Publication scope

Included: the locked source implementation, selected PVO settings, final thesis sources and used figures, final unsigned thesis PDF, defense PDF, and upstream license notices.

Excluded: other research branches, Git history, CV files, downloaded papers, raw datasets, encoded streams, compiled binaries, private preparation notes, and obsolete presentation revisions.
