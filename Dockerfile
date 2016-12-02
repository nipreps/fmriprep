# Copyright (c) 2016, The developers of the Stanford CRN
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of crn_base nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This Dockerfile is to be built for testing purposes inside CircleCI,
# not for distribution within Docker hub.
# For that purpose, the Dockerfile is found in build/Dockerfile.

FROM oesteban/crn_nipype:freesurfer

RUN mkdir -p /opt/c3d && \
    curl -sSL "https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/57f341d6594d9001f591bac2" \
    | tar -xzC /opt/c3d --strip-components 1
ENV C3DPATH /opt/c3d
ENV PATH $C3DPATH:$PATH

RUN rm -rf /usr/local/miniconda/lib/python*/site-packages/nipype* && \
    pip install -e git+https://github.com/nipy/nipype.git@8ddca5a03fcad26887c862dc23c82ef23f2ee506#egg=nipype-8ddc && \
    pip install -e git+https://github.com/incf/pybids.git@158dac2062dc6b5a4ab2f92090108eedc3387575#egg=pybids-158d && \
    conda install -y mock && \
    conda install -y pandas && \
    python -c "from matplotlib import font_manager"

RUN pip install --process-dependency-links -e git+https://github.com/poldracklab/niworkflows.git@0d8f5584d456ebe2505d644e04a8a5f859e78b1b#egg=niworkflows

WORKDIR /root/src

COPY . fmriprep/
RUN cd fmriprep && \
    pip install -e .[all]
    #  python setup.py develop && \

WORKDIR /root/
COPY build/files/run_* /usr/bin/
RUN chmod +x /usr/bin/run_*

ENTRYPOINT ["/usr/bin/run_fmriprep"]
CMD ["--help"]
