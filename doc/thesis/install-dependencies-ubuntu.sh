#!/usr/bin/env bash

set -euo pipefail

sudo apt-get update
sudo apt-get install --no-install-recommends \
  biber \
  latexmk \
  texlive-bibtex-extra \
  texlive-fonts-recommended \
  texlive-lang-german \
  texlive-latex-extra \
  texlive-plain-generic \
  texlive-science \
  xindy
