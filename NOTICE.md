# Licenses and attribution

## Software

This is a thesis-derived source snapshot of Basalt for Monado, based on Basalt by Vladyslav Usenko, Nikolaus Demmel, and contributors, with the Monado/VOCA work of Mateo de Mayo and other contributors. Existing copyright headers and the BSD-3-Clause [LICENSE](LICENSE) are retained. Thesis-specific software modifications by Emir Özer follow that license unless a file states otherwise.

The root software license does not replace more specific file or dependency licenses:

- `thirdparty/apriltag/ethz_apriltag2`: its retained [license](thirdparty/apriltag/ethz_apriltag2/LICENSE) includes an advertising acknowledgement requirement. **This product includes software developed by the Autonomous Systems Lab and Skybotix AG.**
- `thirdparty/vit`: [Boost Software License 1.0](thirdparty/vit/LICENSE), with Collabora copyright notices retained in the files.
- `thirdparty/json/nlohmann/json.hpp`: MIT license and Niels Lohmann copyright notice retained in the header.
- Files marked CC0-1.0 retain that designation.
- Git submodules remain pinned to their original revisions and retain the licenses supplied by their respective projects. Their source is fetched separately through `.gitmodules`.
- TEASER++, FFmpeg, x264, and other external build/runtime dependencies retain their own terms. Their compiled libraries and executables are not redistributed here.

Upstream projects:

- https://gitlab.com/VladyslavUsenko/basalt
- https://gitlab.freedesktop.org/mateosss/basalt
- https://gitlab.freedesktop.org/mateosss/basalt-headers
- https://github.com/MIT-SPARK/TEASER-plusplus

## Thesis and presentation

The thesis and presentation are authored by Emir Özer. Publishing them here does not place their text, figures, or third-party material under the software's BSD license. Copyright and existing third-party rights are retained; no additional blanket reuse license is granted for these documents.

The LaTeX template is derived from **Florian Walch and contributors' TUM thesis template**, licensed under **Creative Commons Attribution-ShareAlike 4.0 International**. The template's attribution and license information are retained in [doc/thesis/README.md](doc/thesis/README.md). Template adaptations remain subject to that license; it does not automatically license the thesis's authored content or resulting PDF.

- Template: https://github.com/fwalch/tum-thesis-latex
- License: https://creativecommons.org/licenses/by-sa/4.0/

TUM logos and corporate-design elements identify the academic institution and retain their respective rights. Their presence does not imply institutional endorsement of the software or grant permission for unrelated reuse of TUM branding.

## Dataset material

Experiments use the **EuRoC MAV dataset**, by Michael Burri, Janosch Nikolic, Pascal Gohl, Thomas Schneider, Joern Rehder, Sammy Omari, Markus W. Achtelik, and Roland Siegwart, ETH Zurich.

- Dataset: https://projects.asl.ethz.ch/datasets/euroc-mav/
- Dataset record: https://doi.org/10.3929/ethz-b-000690084
- Publication: *The EuRoC micro aerial vehicle datasets*, IJRR, 2016, https://doi.org/10.1177/0278364915620033

The ETH Research Collection identifies the dataset as **In Copyright — Non-Commercial Use Permitted**. Dataset-derived image excerpts, reference-trajectory material, and calibration retain the applicable dataset rights and are not relicensed as BSD software. Image excerpts and derived figures are included for the thesis's research explanation and evaluation, with citations in the thesis. Raw dataset archives and encoded video streams are not included. Obtain datasets from their original providers and consult their terms for other uses.

Other cited works are referenced bibliographically; downloaded publisher PDFs and other authors' thesis/presentation files are not redistributed.
