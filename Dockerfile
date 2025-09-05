# fMRIPrep Docker Container Image distribution
#
# MIT License
#
# Copyright (c) The NiPreps Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Ubuntu 22.04 LTS - Jammy
ARG BASE_IMAGE=ubuntu:jammy-20250730

#
# Build pixi environment
# The Pixi environment includes:
#   - Python
#     - Scientific Python stack (via conda-forge)
#     - General Python dependencies (via PyPI)
#   - NodeJS
#     - bids-validator
#     - svgo
#   - FSL (via fslconda)
#   - ants (via conda-forge)
#   - connectome-workbench (via conda-forge)
#   - ...
#
FROM ghcr.io/prefix-dev/pixi:0.53.0 AS build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    ca-certificates \
                    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Run post-link scripts during install, but use global to keep out of source tree
RUN pixi config set --global run-post-link-scripts insecure

# Install dependencies before the package itself to leverage caching
RUN mkdir /app
COPY pixi.lock pyproject.toml /app
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e fmriprep -e test --frozen --skip fmriprep
RUN --mount=type=cache,target=/root/.npm pixi run --as-is -e fmriprep npm install -g svgo@^3.2.0 bids-validator@1.14.10
# Note that PATH gets hard-coded. Remove it and re-apply in final image
RUN pixi shell-hook -e fmriprep --as-is | grep -v PATH > /shell-hook.sh
RUN pixi shell-hook -e test --as-is | grep -v PATH > /test-shell-hook.sh

# Finally, install the package
COPY . /app
RUN --mount=type=cache,target=/root/.cache/rattler pixi install -e fmriprep -e test --frozen

#
# Pre-fetch templates
#
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS templates
ENV TEMPLATEFLOW_HOME="/templateflow"
RUN uv pip install --system templateflow
COPY scripts/fetch_templates.py fetch_templates.py
RUN python fetch_templates.py

#
# Download stages
#

# Utilities for downloading packages
FROM ${BASE_IMAGE} AS downloader
ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"

# Bump the date to current to refresh curl/certificates/etc
RUN echo "2025.08.20"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    binutils \
                    bzip2 \
                    ca-certificates \
                    curl \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# FreeSurfer 7.3.2
FROM downloader AS freesurfer
COPY docker/files/freesurfer7.3.2-exclude.txt /usr/local/etc/freesurfer7.3.2-exclude.txt
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-ubuntu22_amd64-7.3.2.tar.gz \
     | tar zxv --no-same-owner -C /opt --exclude-from=/usr/local/etc/freesurfer7.3.2-exclude.txt

# MSM HOCR (Nov 19, 2019 release)
FROM downloader AS msm
RUN curl -L -H "Accept: application/octet-stream" https://api.github.com/repos/ecr05/MSM_HOCR/releases/assets/16253707 -o /usr/local/bin/msm \
    && chmod +x /usr/local/bin/msm

#
# Main stage
#
FROM ${BASE_IMAGE} AS base

# Configure apt
ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"

# Some baseline tools; bc is needed for FreeSurfer, so don't drop it
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    bc \
                    ca-certificates \
                    curl \
                    libgomp1 \
                    libopenblas0-openmp \
                    lsb-release \
                    netbase \
                    tcsh \
                    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install downloaded files from stages
COPY --link --from=freesurfer /opt/freesurfer /opt/freesurfer
COPY --link --from=msm /usr/local/bin/msm /usr/local/bin/msm

# Install AFNI from Docker container
# Find libraries with `ldd $BINARIES | grep afni`
COPY --link --from=afni/afni_make_build:AFNI_25.2.09 \
    /opt/afni/install/libf2c.so  \
    /opt/afni/install/libmri.so  \
    /usr/local/lib/
COPY --link --from=afni/afni_make_build:AFNI_25.2.09 \
    /opt/afni/install/3dAutomask \
    /opt/afni/install/3dTshift \
    /opt/afni/install/3dUnifize \
    /opt/afni/install/3dvolreg \
    /usr/local/bin/

# Changing library paths requires a re-ldconfig
RUN ldconfig

# Simulate SetUpFreeSurfer.sh
ENV OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/opt/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"

# AFNI config
ENV AFNI_IMSAVE_WARNINGS="NO"

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users fmriprep
WORKDIR /home/fmriprep
ENV HOME="/home/fmriprep"

COPY --link --from=templates /templateflow /home/fmriprep/.cache/templateflow

# FSL environment
ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1 \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /tmp

FROM base AS test

COPY --link --from=build /app/.pixi/envs/test /app/.pixi/envs/test
COPY --link --from=build /test-shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/test/bin:$PATH"

ENV FSLDIR="/app/.pixi/envs/test"

FROM base AS fmriprep

# Keep synced with wrapper's PKG_PATH
COPY --link --from=build /app/.pixi/envs/fmriprep /app/.pixi/envs/fmriprep
COPY --link --from=build /shell-hook.sh /shell-hook.sh
RUN cat /shell-hook.sh >> $HOME/.bashrc
ENV PATH="/app/.pixi/envs/fmriprep/bin:$PATH"

ENV FSLDIR="/app/.pixi/envs/fmriprep"

# For detecting the container
ENV IS_DOCKER_8395080871=1

ENTRYPOINT ["/app/.pixi/envs/fmriprep/bin/fmriprep"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="fMRIPrep" \
      org.label-schema.description="fMRIPrep - robust fMRI preprocessing tool" \
      org.label-schema.url="https://fmriprep.org" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/fmriprep" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
